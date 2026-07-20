#!/usr/bin/env python3
"""Strictly finalize one complete LongBench16 C0--C6 run.

Finalization is a verified state transition.  This module never treats the
presence of output files, a pilot, or a self-declared ``"complete"`` field as
evidence of completion.  It independently reloads the pinned prepared token
artifact, rebuilds the C0--C5 join and both oracle analyses, and reruns the
locked held-out C6 CPU protocol before atomically replacing the original
``running`` manifest.

The resulting manifest is idempotent and immutable by content: a second call
revalidates every required artifact and returns the byte-identical completed
manifest.  Any stale or mutated input fails closed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Linux/WSL is the production environment.
    import fcntl
except ImportError:  # pragma: no cover - unsupported production platform.
    fcntl = None

try:
    from .longbench16_c6 import (
        C6_ANALYSIS_SCHEMA,
        C6_ROW_SCHEMA,
        DEFAULT_ROWS_FILENAME as C6_ROWS_FILENAME,
        DEFAULT_SUMMARY_FILENAME as C6_SUMMARY_FILENAME,
        evaluate_c6_file,
    )
    from .longbench16_io import (
        FIXED_CONFIGS,
        RunLayout,
        assert_safe_write_path,
        atomic_write_json,
        artifact_metadata,
        file_sha256,
    )
    from .longbench16_join import (
        ANALYSIS_SCHEMA,
        JOIN_SCHEMA,
        PRIMARY_ORACLE_OBJECTIVE,
        QUALITY_ORACLE_DIAGNOSTIC,
        build_analysis,
        join_journals,
        load_prepared_expectations,
    )
    from .longbench16_protocol import (
        EXPECTED_GENERATIONS_PER_MODEL,
        EXPECTED_TOTAL_EXAMPLES,
        protocol_config_hash,
    )
    from .longbench16_run_config import (
        DEFAULT_RUN_CONFIG,
        PILOT_PROFILE_WORST_CASE_24K_512,
        RUNNER_SCHEMA,
        WORST_CASE_MIN_HEADROOM_BYTES,
        LockedRunConfig,
        PreparedDataset,
        load_locked_run_config,
        load_prepared_dataset,
        validate_runtime_source_code,
        validate_worst_case_pilot_gate,
    )
except ImportError:  # Direct execution: python scripts/longbench16_finalize.py
    from longbench16_c6 import (
        C6_ANALYSIS_SCHEMA,
        C6_ROW_SCHEMA,
        DEFAULT_ROWS_FILENAME as C6_ROWS_FILENAME,
        DEFAULT_SUMMARY_FILENAME as C6_SUMMARY_FILENAME,
        evaluate_c6_file,
    )
    from longbench16_io import (
        FIXED_CONFIGS,
        RunLayout,
        assert_safe_write_path,
        atomic_write_json,
        artifact_metadata,
        file_sha256,
    )
    from longbench16_join import (
        ANALYSIS_SCHEMA,
        JOIN_SCHEMA,
        PRIMARY_ORACLE_OBJECTIVE,
        QUALITY_ORACLE_DIAGNOSTIC,
        build_analysis,
        join_journals,
        load_prepared_expectations,
    )
    from longbench16_protocol import (
        EXPECTED_GENERATIONS_PER_MODEL,
        EXPECTED_TOTAL_EXAMPLES,
        protocol_config_hash,
    )
    from longbench16_run_config import (
        DEFAULT_RUN_CONFIG,
        PILOT_PROFILE_WORST_CASE_24K_512,
        RUNNER_SCHEMA,
        WORST_CASE_MIN_HEADROOM_BYTES,
        LockedRunConfig,
        PreparedDataset,
        load_locked_run_config,
        load_prepared_dataset,
        validate_runtime_source_code,
        validate_worst_case_pilot_gate,
    )


FINALIZATION_SCHEMA = "adaptiveserve-longbench16-finalization/v1"
JOINED_FILENAME = "joined.jsonl"
ANALYSIS_FILENAME = "summary.json"
MANIFEST_FILENAME = "manifest.json"
ENVIRONMENT_FILENAME = "environment.json"
LOCK_FILENAME = ".finalize.lock"

_RUNNING_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "started_at",
        "run_id",
        "model_alias",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "run_config",
        "prepared_inputs",
        "source_code",
    }
)
_COMPLETION_FIELDS = frozenset({"completed_at", "finalization"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
    "numpy",
    "scikit-learn",
    "rouge",
)


class FinalizationError(RuntimeError):
    """Raised when a run cannot be proven complete and immutable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} is not finite JSON data") from exc


def _canonical_json_sha256(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, label=label)).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise FinalizationError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise FinalizationError(f"{label} is unreadable: {path}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object: {path}")
    _canonical_json_bytes(value, label=label)
    return value


def _load_jsonl(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise FinalizationError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise FinalizationError(f"{label} is unreadable: {path}") from exc
    if not payload:
        raise FinalizationError(f"{label} is empty: {path}")
    if not payload.endswith(b"\n"):
        raise FinalizationError(f"{label} has a partial trailing line: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line:
            raise FinalizationError(f"{label} line {line_number} is blank")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalizationError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise FinalizationError(
                f"{label} line {line_number} must be an object"
            )
        _canonical_json_bytes(row, label=f"{label} line {line_number}")
        rows.append(row)
    return tuple(rows)


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FinalizationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalizationError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalizationError(f"{field} must be an integer >= {minimum}")
    return value


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    text = _require_nonempty_string(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalizationError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FinalizationError(f"{field} must include a UTC offset")
    return parsed


def _required_file(root: Path, relative: Path | str, *, label: str) -> Path:
    candidate = root / Path(relative)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise FinalizationError(
            f"{label} is missing or resolves outside the run: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise FinalizationError(f"{label} is not a regular file: {resolved}")
    if candidate.is_symlink():
        raise FinalizationError(f"{label} may not be a symlink: {candidate}")
    return resolved


def _prepared_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FinalizationError(f"{label} path must be a non-empty string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise FinalizationError(f"{label} path must stay inside prepared_dir")
    try:
        resolved = (root / relative_path).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise FinalizationError(f"{label} is missing or escapes prepared_dir") from exc
    if not resolved.is_file():
        raise FinalizationError(f"{label} is not a regular file")
    return resolved


def _artifact(
    path: Path,
    *,
    run_dir: Path,
    rows: int | None = None,
) -> dict[str, Any]:
    metadata = artifact_metadata(path, relative_to=run_dir)
    if rows is not None:
        metadata["rows"] = rows
    return metadata


def _compare_rows(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if len(actual) != len(expected):
        raise FinalizationError(
            f"{label} has {len(actual)} rows; expected {len(expected)}"
        )
    for index, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
        if observed != wanted:
            observed_key = (observed.get("task"), observed.get("benchmark_id"))
            wanted_key = (wanted.get("task"), wanted.get("benchmark_id"))
            raise FinalizationError(
                f"{label} row {index} differs from independent recomputation; "
                f"observed_key={observed_key!r}, expected_key={wanted_key!r}"
            )


def _validate_artifact_declaration(
    declaration: Any,
    path: Path,
    *,
    label: str,
    rows: int | None = None,
) -> None:
    if not isinstance(declaration, Mapping):
        raise FinalizationError(f"{label} artifact declaration must be an object")
    expected: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        expected["rows"] = rows
    if dict(declaration) != expected:
        raise FinalizationError(
            f"{label} artifact declaration is stale or does not match {path}"
        )


def _validate_summary_artifacts(
    block: Any,
    *,
    per_prompt_key: str,
    per_prompt_path: Path,
    summary_key: str,
    summary_path: Path,
    rows: int,
    label: str,
) -> None:
    if not isinstance(block, Mapping) or set(block) != {
        per_prompt_key,
        summary_key,
    }:
        raise FinalizationError(f"{label}.artifacts has an unexpected shape")
    _validate_artifact_declaration(
        block[per_prompt_key],
        per_prompt_path,
        label=f"{label}.{per_prompt_key}",
        rows=rows,
    )
    if block[summary_key] != {"path": str(summary_path)}:
        raise FinalizationError(
            f"{label}.{summary_key} must identify the canonical summary path"
        )


def _validate_run_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    prepared: PreparedDataset,
    locked: LockedRunConfig,
) -> None:
    status = manifest.get("status")
    allowed_fields = (
        _RUNNING_MANIFEST_FIELDS
        if status == "running"
        else _RUNNING_MANIFEST_FIELDS | _COMPLETION_FIELDS
    )
    if status not in {"running", "complete"}:
        raise FinalizationError(
            f"run manifest status must be 'running' or 'complete', got {status!r}"
        )
    if set(manifest) != allowed_fields:
        missing = sorted(allowed_fields - set(manifest))
        extra = sorted(set(manifest) - allowed_fields)
        raise FinalizationError(
            f"run manifest fields differ from the v1 identity contract; "
            f"missing={missing}, extra={extra}"
        )
    expected_scalars = {
        "schema_version": RUNNER_SCHEMA,
        "run_id": run_dir.name,
        "model_alias": run_dir.parent.name,
        "model_id": locked.model["model_id"],
        "model_revision": locked.model["model_revision"],
        "tokenizer_id": locked.model["tokenizer_id"],
        "tokenizer_revision": locked.model["tokenizer_revision"],
    }
    for field, expected in expected_scalars.items():
        if manifest.get(field) != expected:
            raise FinalizationError(
                f"run manifest {field} does not match immutable run identity"
            )
    if manifest["model_alias"] != locked.model_alias:
        raise FinalizationError("run path/model alias differs from locked config")
    _parse_utc_timestamp(manifest.get("started_at"), field="manifest.started_at")

    run_config = manifest.get("run_config")
    if not isinstance(run_config, Mapping) or set(run_config) != {"path", "sha256"}:
        raise FinalizationError("manifest.run_config has an unexpected shape")
    if run_config.get("sha256") != locked.sha256:
        raise FinalizationError("manifest.run_config SHA-256 differs from config")
    try:
        declared_config = Path(str(run_config["path"])).expanduser().resolve(
            strict=True
        )
    except (OSError, RuntimeError) as exc:
        raise FinalizationError("manifest.run_config path is not resolvable") from exc
    if declared_config != locked.path:
        raise FinalizationError("manifest.run_config path differs from CLI config")

    prepared_inputs = manifest.get("prepared_inputs")
    if not isinstance(prepared_inputs, Mapping) or set(prepared_inputs) != {
        "path",
        "manifest_sha256",
        "index_sha256",
        "records",
    }:
        raise FinalizationError("manifest.prepared_inputs has an unexpected shape")
    try:
        declared_prepared = Path(
            str(prepared_inputs["path"])
        ).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FinalizationError("manifest.prepared_inputs path is not resolvable") from exc
    if declared_prepared != prepared.root:
        raise FinalizationError(
            "manifest.prepared_inputs path differs from the pinned artifact"
        )
    expected_prepared = {
        "manifest_sha256": prepared.manifest_sha256,
        "index_sha256": prepared.index_sha256,
        "records": EXPECTED_TOTAL_EXAMPLES,
    }
    for field, expected in expected_prepared.items():
        if prepared_inputs.get(field) != expected:
            raise FinalizationError(
                f"manifest.prepared_inputs.{field} is stale or inconsistent"
            )
    source_code = manifest.get("source_code")
    prepared_source = prepared.manifest.get("source_code")
    if (
        not isinstance(source_code, Mapping)
        or not isinstance(prepared_source, Mapping)
        or dict(source_code) != dict(prepared_source)
        or not isinstance(source_code.get("commit"), str)
        or not _GIT_COMMIT_RE.fullmatch(source_code["commit"])
        or source_code.get("dirty") is not False
    ):
        raise FinalizationError(
            "manifest.source_code must equal the clean prepared source identity"
        )


def _validate_environment(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    locked: LockedRunConfig,
    prepared: PreparedDataset,
) -> dict[str, Any]:
    environment = _load_json_object(path, label="run environment metadata")
    required_top = {
        "captured_at",
        "python",
        "platform",
        "packages",
        "environment",
        "git",
        "gpu",
        "extra",
    }
    missing = required_top - set(environment)
    if missing:
        raise FinalizationError(
            f"run environment metadata lacks fields: {sorted(missing)}"
        )
    _parse_utc_timestamp(
        environment.get("captured_at"), field="environment.captured_at"
    )
    python = environment.get("python")
    if not isinstance(python, Mapping):
        raise FinalizationError("environment.python must be an object")
    for field in ("version", "implementation", "executable"):
        _require_nonempty_string(
            python.get(field), field=f"environment.python.{field}"
        )
    platform = environment.get("platform")
    if not isinstance(platform, Mapping):
        raise FinalizationError("environment.platform must be an object")
    for field in ("system", "release", "version", "machine"):
        _require_nonempty_string(
            platform.get(field), field=f"environment.platform.{field}"
        )
    packages = environment.get("packages")
    if not isinstance(packages, Mapping):
        raise FinalizationError("environment.packages must be an object")
    for package in _REQUIRED_PACKAGES:
        _require_nonempty_string(
            packages.get(package),
            field=f"environment.packages[{package!r}]",
        )
    if not isinstance(environment.get("environment"), Mapping):
        raise FinalizationError("environment.environment must be an object")

    source_code = prepared.manifest.get("source_code")
    if not isinstance(source_code, Mapping):
        raise FinalizationError("prepared manifest lacks source_code provenance")
    prepared_commit = source_code.get("commit")
    if (
        not isinstance(prepared_commit, str)
        or not _GIT_COMMIT_RE.fullmatch(prepared_commit)
        or source_code.get("dirty") is not False
    ):
        raise FinalizationError(
            "prepared artifact was not created from a clean pinned git commit"
        )
    git = environment.get("git")
    if (
        not isinstance(git, Mapping)
        or git.get("commit") != prepared_commit
        or git.get("dirty") is not False
    ):
        raise FinalizationError(
            "run environment git state must be clean and match preparation"
        )

    gpu = environment.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("available") is not True:
        raise FinalizationError("run environment must contain available GPU metadata")
    devices = gpu.get("devices")
    if not isinstance(devices, list) or not devices:
        raise FinalizationError("run environment GPU device list is empty")
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping):
            raise FinalizationError(f"environment.gpu.devices[{index}] is invalid")
        _require_int(device.get("index"), field=f"gpu.devices[{index}].index")
        for field in ("name", "uuid", "driver_version", "compute_capability"):
            _require_nonempty_string(
                device.get(field), field=f"gpu.devices[{index}].{field}"
            )
        total = _require_int(
            device.get("memory_total_mib"),
            field=f"gpu.devices[{index}].memory_total_mib",
            minimum=1,
        )
        used = _require_int(
            device.get("memory_used_mib"),
            field=f"gpu.devices[{index}].memory_used_mib",
        )
        free = _require_int(
            device.get("memory_free_mib"),
            field=f"gpu.devices[{index}].memory_free_mib",
        )
        if used > total or free > total:
            raise FinalizationError(
                f"environment.gpu.devices[{index}] memory values are impossible"
            )

    extra = environment.get("extra")
    expected_extra = {
        "run_id": manifest["run_id"],
        "model_alias": locked.model_alias,
        "run_config_sha256": locked.sha256,
        "prepared_manifest_sha256": prepared.manifest_sha256,
    }
    if not isinstance(extra, Mapping) or dict(extra) != expected_extra:
        raise FinalizationError(
            "run environment extra identity does not match the run manifest"
        )
    return environment


def _reject_pilot_markers(
    journal_paths: Mapping[str, Path],
) -> None:
    for config, path in journal_paths.items():
        events = _load_jsonl(path, label=f"{config} canonical journal")
        for line_number, event in enumerate(events, 1):
            metadata = event.get("method_metadata")
            if "pilot_profile" in event or (
                isinstance(metadata, Mapping)
                and (
                    "pilot_profile" in metadata
                    or "cuda_feasibility" in metadata
                )
            ):
                raise FinalizationError(
                    f"{config} canonical journal line {line_number} contains "
                    "pilot-only metadata"
                )


def _validate_worst_case_pilots(
    *,
    run_dir: Path,
    prepared: PreparedDataset,
) -> tuple[dict[str, Path], dict[str, int]]:
    """Independently enforce and locate every final-mode feasibility pilot."""

    layout = RunLayout.open(
        run_dir.parent.parent,
        run_dir.parent.name,
        run_dir.name,
    )
    pilot_paths: dict[str, Path] = {}
    event_counts: dict[str, int] = {}
    for config in FIXED_CONFIGS:
        relative = (
            Path("pilots")
            / PILOT_PROFILE_WORST_CASE_24K_512
            / config
            / "per_prompt.jsonl"
        )
        path = _required_file(
            run_dir,
            relative,
            label=f"{config} worst-case 24K+512 pilot journal",
        )
        pilot_paths[config] = path
        event_counts[config] = len(
            _load_jsonl(
                path,
                label=f"{config} worst-case 24K+512 pilot journal",
            )
        )

    observed_hashes = validate_worst_case_pilot_gate(layout, prepared)
    if set(observed_hashes) != set(FIXED_CONFIGS):
        raise FinalizationError(
            "worst-case pilot gate did not validate exactly C0-C5"
        )
    for config, path in pilot_paths.items():
        observed = _require_sha256(
            observed_hashes.get(config),
            field=f"worst-case pilot hash for {config}",
        )
        if observed != file_sha256(path):
            raise FinalizationError(
                f"{config} worst-case pilot changed after gate validation"
            )
    return pilot_paths, event_counts


def _validate_join_outputs(
    *,
    run_dir: Path,
    prepared_dir: Path,
    journal_paths: Mapping[str, Path],
) -> tuple[
    Any,
    tuple[dict[str, Any], ...],
    dict[str, Any],
    Path,
    Path,
]:
    result = join_journals(
        journal_paths,
        prepared_dir=prepared_dir,
        require_complete=True,
    )
    rebuilt_rows, rebuilt_analysis = build_analysis(
        result,
        oracle_objective=PRIMARY_ORACLE_OBJECTIVE,
        require_complete=True,
    )
    joined_path = _required_file(
        run_dir,
        Path("analysis") / JOINED_FILENAME,
        label="joined per-prompt artifact",
    )
    summary_path = _required_file(
        run_dir,
        Path("analysis") / ANALYSIS_FILENAME,
        label="joined analysis artifact",
    )
    actual_rows = _load_jsonl(joined_path, label="joined per-prompt artifact")
    if len(actual_rows) != EXPECTED_TOTAL_EXAMPLES:
        raise FinalizationError(
            "joined per-prompt artifact is partial; "
            f"found {len(actual_rows)}, expected {EXPECTED_TOTAL_EXAMPLES}"
        )
    if any(row.get("schema_version") != JOIN_SCHEMA for row in actual_rows):
        raise FinalizationError("joined rows contain an unsupported schema")
    _compare_rows(actual_rows, rebuilt_rows, label="joined per-prompt artifact")

    actual_analysis = _load_json_object(
        summary_path, label="joined analysis artifact"
    )
    if actual_analysis.get("schema_version") != ANALYSIS_SCHEMA:
        raise FinalizationError("joined analysis schema is unsupported")
    artifacts = actual_analysis.get("artifacts")
    without_artifacts = copy.deepcopy(actual_analysis)
    without_artifacts.pop("artifacts", None)
    if without_artifacts != rebuilt_analysis:
        raise FinalizationError(
            "joined analysis/oracle summary differs from independent recomputation"
        )
    _validate_summary_artifacts(
        artifacts,
        per_prompt_key="joined_jsonl",
        per_prompt_path=joined_path,
        summary_key="analysis_json",
        summary_path=summary_path,
        rows=EXPECTED_TOTAL_EXAMPLES,
        label="joined analysis",
    )
    oracle = actual_analysis.get("oracle_analysis")
    if not isinstance(oracle, Mapping):
        raise FinalizationError("joined analysis lacks oracle evidence")
    primary = oracle.get("primary")
    diagnostic = oracle.get("required_quality_first_diagnostic")
    if (
        not isinstance(primary, Mapping)
        or primary.get("name") != PRIMARY_ORACLE_OBJECTIVE
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("name") != QUALITY_ORACLE_DIAGNOSTIC
        or oracle.get("joined_oracle_columns_objective")
        != PRIMARY_ORACLE_OBJECTIVE
    ):
        raise FinalizationError(
            "joined analysis lacks the locked primary and diagnostic oracles"
        )
    return (
        result,
        tuple(copy.deepcopy(actual_rows)),
        actual_analysis,
        joined_path,
        summary_path,
    )


def _validate_c6_outputs(
    *,
    run_dir: Path,
    joined_path: Path,
    run_config: Path,
    manifest: Mapping[str, Any],
    prepared: PreparedDataset,
    locked: LockedRunConfig,
    running_manifest_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], Path, Path]:
    c6_rows_path = _required_file(
        run_dir,
        Path("analysis") / C6_ROWS_FILENAME,
        label="C6 per-prompt artifact",
    )
    c6_summary_path = _required_file(
        run_dir,
        Path("analysis") / C6_SUMMARY_FILENAME,
        label="C6 summary artifact",
    )
    actual_rows = _load_jsonl(c6_rows_path, label="C6 per-prompt artifact")
    if len(actual_rows) != EXPECTED_TOTAL_EXAMPLES:
        raise FinalizationError(
            "C6 per-prompt artifact is partial; "
            f"found {len(actual_rows)}, expected {EXPECTED_TOTAL_EXAMPLES}"
        )
    if any(row.get("schema_version") != C6_ROW_SCHEMA for row in actual_rows):
        raise FinalizationError("C6 rows contain an unsupported schema")

    # No test-count, injected-fold, scaler, or regressor boundary is supplied:
    # this is deliberately the exact primary 3,750-row/10-fold recomputation.
    rebuilt = evaluate_c6_file(joined_path, run_config=run_config)
    if rebuilt.summary.get("evaluation_mode") != "primary":
        raise FinalizationError("independent C6 recomputation was not primary")
    _compare_rows(actual_rows, rebuilt.rows, label="C6 per-prompt artifact")

    actual_summary = _load_json_object(
        c6_summary_path, label="C6 summary artifact"
    )
    if actual_summary.get("schema_version") != C6_ANALYSIS_SCHEMA:
        raise FinalizationError("C6 summary schema is unsupported")
    artifacts = actual_summary.get("artifacts")
    run_identity = actual_summary.get("run_identity")
    without_artifacts = copy.deepcopy(actual_summary)
    without_artifacts.pop("artifacts", None)
    without_artifacts.pop("run_identity", None)
    if without_artifacts != rebuilt.summary:
        raise FinalizationError(
            "C6 summary differs from an independent held-out recomputation"
        )
    expected_identity = {
        "run_manifest": {
            "path": str((run_dir / MANIFEST_FILENAME).resolve()),
            "sha256_at_c6": running_manifest_sha256,
            "schema_version": RUNNER_SCHEMA,
            "status_at_c6": "running",
        },
        "run_id": manifest["run_id"],
        "started_at": manifest["started_at"],
        "model_alias": manifest["model_alias"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_id": manifest["tokenizer_id"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "run_config": {
            "path": str(locked.path),
            "sha256": locked.sha256,
        },
        "prepared_inputs": {
            "path": str(prepared.root),
            "manifest_sha256": prepared.manifest_sha256,
            "index_sha256": prepared.index_sha256,
            "records": EXPECTED_TOTAL_EXAMPLES,
        },
        "source_code": {
            "commit": manifest["source_code"]["commit"],
            "dirty": False,
            "dirty_check_ignored_only": "runs/longbench16_24k/**",
        },
    }
    if not isinstance(run_identity, Mapping) or dict(run_identity) != (
        expected_identity
    ):
        raise FinalizationError(
            "C6 run_identity is missing, stale, or differs from the "
            "pre-finalization run identity"
        )
    if (
        actual_summary.get("status") != "complete"
        or actual_summary.get("evaluation_mode") != "primary"
        or actual_summary.get("row_count") != EXPECTED_TOTAL_EXAMPLES
    ):
        raise FinalizationError(
            "C6 summary is test/smoke, partial, or not complete"
        )
    cross_validation = actual_summary.get("cross_validation")
    if (
        not isinstance(cross_validation, Mapping)
        or cross_validation.get("fold_count") != 10
        or cross_validation.get("folds_injected") is not False
        or cross_validation.get("each_example_held_out_exactly_once") is not True
    ):
        raise FinalizationError("C6 output is not the locked held-out 10-fold run")
    _validate_summary_artifacts(
        artifacts,
        per_prompt_key="per_prompt_jsonl",
        per_prompt_path=c6_rows_path,
        summary_key="summary_json",
        summary_path=c6_summary_path,
        rows=EXPECTED_TOTAL_EXAMPLES,
        label="C6 summary",
    )
    return (
        tuple(copy.deepcopy(actual_rows)),
        actual_summary,
        c6_rows_path,
        c6_summary_path,
    )


def _prepared_artifacts(
    prepared: PreparedDataset,
    *,
    run_dir: Path,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    root = prepared.root
    manifest_path = _prepared_file(root, "manifest.json", label="prepared manifest")
    status_path = _prepared_file(root, "status.json", label="prepared status")
    index_path = prepared.index_path.resolve(strict=True)
    token_entries = prepared.manifest.get("token_files")
    if not isinstance(token_entries, list) or len(token_entries) != 16:
        raise FinalizationError(
            "prepared manifest must declare exactly 16 token NPZ artifacts"
        )
    token_artifacts: dict[str, Any] = {}
    paths: list[Path] = [manifest_path, status_path, index_path]
    for index, entry in enumerate(token_entries):
        if not isinstance(entry, Mapping):
            raise FinalizationError(f"prepared token_files[{index}] is invalid")
        task = _require_nonempty_string(
            entry.get("task"), field=f"prepared token_files[{index}].task"
        )
        if task in token_artifacts:
            raise FinalizationError(f"prepared token file repeats task {task}")
        path = _prepared_file(
            root,
            entry.get("path"),
            label=f"prepared {task} token NPZ",
        )
        rows = _require_int(
            entry.get("rows"),
            field=f"prepared token_files[{index}].rows",
            minimum=1,
        )
        token_artifacts[task] = _artifact(path, run_dir=run_dir, rows=rows)
        paths.append(path)
    return (
        {
            "directory": str(root),
            "manifest": _artifact(manifest_path, run_dir=run_dir),
            "status": _artifact(status_path, run_dir=run_dir),
            "index": _artifact(
                index_path,
                run_dir=run_dir,
                rows=EXPECTED_TOTAL_EXAMPLES,
            ),
            "token_npz": token_artifacts,
        },
        tuple(paths),
    )


def _assert_unchanged(
    paths: Iterable[Path],
    expected: Mapping[Path, tuple[int, str]],
) -> None:
    for path in paths:
        try:
            current = (path.stat().st_size, file_sha256(path))
        except OSError as exc:
            raise FinalizationError(
                f"required artifact disappeared during finalization: {path}"
            ) from exc
        if current != expected[path]:
            raise FinalizationError(
                f"required artifact changed during finalization: {path}"
            )


def _running_payload_from_completed(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    running = {
        field: copy.deepcopy(manifest[field])
        for field in _RUNNING_MANIFEST_FIELDS
    }
    running["status"] = "running"
    return running


@contextlib.contextmanager
def _finalization_lock(run_dir: Path) -> Iterable[None]:
    if fcntl is None:  # pragma: no cover
        raise FinalizationError("strict finalization requires POSIX fcntl locking")
    lock_path = assert_safe_write_path(run_dir / LOCK_FILENAME)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def finalize_run(
    run_dir: Path | str,
    prepared_dir: Path | str,
    *,
    run_config: Path | str = DEFAULT_RUN_CONFIG,
) -> dict[str, Any]:
    """Revalidate and atomically mark one exact production run complete."""

    try:
        resolved_run = Path(run_dir).expanduser().resolve(strict=True)
        resolved_prepared = Path(prepared_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise FinalizationError("run_dir and prepared_dir must already exist") from exc
    if not resolved_run.is_dir() or not resolved_prepared.is_dir():
        raise FinalizationError("run_dir and prepared_dir must be directories")
    manifest_path = _required_file(
        resolved_run, MANIFEST_FILENAME, label="run manifest"
    )
    assert_safe_write_path(manifest_path)

    with _finalization_lock(resolved_run):
        initial_manifest_hash = file_sha256(manifest_path)
        manifest = _load_json_object(manifest_path, label="run manifest")
        if initial_manifest_hash != _canonical_json_sha256(
            manifest, label="run manifest"
        ):
            raise FinalizationError(
                "run manifest is not in the canonical atomic JSON encoding"
            )

        locked = load_locked_run_config(run_config)

        # Canonical production journals must exist before expensive prepared
        # token and C6 validation.  Pilot files are never accepted as a
        # substitute, though retaining successful pilot evidence is allowed.
        journal_paths: dict[str, Path] = {}
        missing_journals: list[str] = []
        for config in FIXED_CONFIGS:
            try:
                journal_paths[config] = _required_file(
                    resolved_run,
                    Path(config) / "per_prompt.jsonl",
                    label=f"{config} canonical per-prompt journal",
                )
            except FinalizationError:
                missing_journals.append(config)
        if missing_journals:
            raise FinalizationError(
                "run is pilot-only or partial; canonical per-prompt journals "
                f"are missing for {missing_journals}"
            )

        prepared = load_prepared_dataset(resolved_prepared, locked)
        validate_runtime_source_code(prepared)
        prepared_expectations = load_prepared_expectations(
            resolved_prepared, require_complete=True
        )
        if (
            prepared_expectations.manifest_sha256 != prepared.manifest_sha256
            or prepared_expectations.index_sha256 != prepared.index_sha256
            or prepared_expectations.record_count != EXPECTED_TOTAL_EXAMPLES
        ):
            raise FinalizationError(
                "independent prepared-artifact validators disagree"
            )
        _validate_run_manifest_identity(
            manifest,
            run_dir=resolved_run,
            prepared=prepared,
            locked=locked,
        )
        running_manifest_sha256 = (
            initial_manifest_hash
            if manifest["status"] == "running"
            else _require_sha256(
                manifest.get("finalization", {}).get(
                    "running_manifest_sha256"
                )
                if isinstance(manifest.get("finalization"), Mapping)
                else None,
                field="manifest.finalization.running_manifest_sha256",
            )
        )

        environment_path = _required_file(
            resolved_run,
            ENVIRONMENT_FILENAME,
            label="run environment metadata",
        )
        _validate_environment(
            environment_path,
            manifest=manifest,
            locked=locked,
            prepared=prepared,
        )
        _reject_pilot_markers(journal_paths)
        pilot_paths, pilot_event_counts = _validate_worst_case_pilots(
            run_dir=resolved_run,
            prepared=prepared,
        )

        (
            join_result,
            joined_rows,
            joined_analysis,
            joined_path,
            analysis_path,
        ) = _validate_join_outputs(
            run_dir=resolved_run,
            prepared_dir=resolved_prepared,
            journal_paths=journal_paths,
        )
        (
            c6_rows,
            c6_summary,
            c6_rows_path,
            c6_summary_path,
        ) = _validate_c6_outputs(
            run_dir=resolved_run,
            joined_path=joined_path,
            run_config=locked.path,
            manifest=manifest,
            prepared=prepared,
            locked=locked,
            running_manifest_sha256=running_manifest_sha256,
        )

        prepared_artifacts, prepared_paths = _prepared_artifacts(
            prepared, run_dir=resolved_run
        )
        fixed_artifacts = {
            config: _artifact(
                journal_paths[config],
                run_dir=resolved_run,
                rows=join_result.validations[config].journal_event_count,
            )
            for config in FIXED_CONFIGS
        }
        pilot_artifacts = {
            config: _artifact(
                pilot_paths[config],
                run_dir=resolved_run,
                rows=pilot_event_counts[config],
            )
            for config in FIXED_CONFIGS
        }
        artifacts: dict[str, Any] = {
            "run_config": _artifact(locked.path, run_dir=resolved_run),
            "environment": _artifact(environment_path, run_dir=resolved_run),
            "prepared_inputs": prepared_artifacts,
            "worst_case_24k_512_pilot_journals": pilot_artifacts,
            "fixed_per_prompt_journals": fixed_artifacts,
            "joined_per_prompt": _artifact(
                joined_path,
                run_dir=resolved_run,
                rows=len(joined_rows),
            ),
            "joined_analysis": _artifact(
                analysis_path, run_dir=resolved_run
            ),
            "c6_per_prompt": _artifact(
                c6_rows_path,
                run_dir=resolved_run,
                rows=len(c6_rows),
            ),
            "c6_analysis": _artifact(
                c6_summary_path, run_dir=resolved_run
            ),
        }
        required_paths = (
            locked.path,
            environment_path,
            *prepared_paths,
            *(pilot_paths[config] for config in FIXED_CONFIGS),
            *(journal_paths[config] for config in FIXED_CONFIGS),
            joined_path,
            analysis_path,
            c6_rows_path,
            c6_summary_path,
        )
        if len(required_paths) != len(set(required_paths)):
            raise FinalizationError("required artifact paths are not unique")
        snapshot = {
            path: (path.stat().st_size, file_sha256(path))
            for path in required_paths
        }

        fixed_validation = {
            config: {
                "completed_records": join_result.validations[config].record_count,
                "journal_events": join_result.validations[
                    config
                ].journal_event_count,
                "failed_attempts": join_result.validations[
                    config
                ].failed_attempt_count,
                "retried_keys": len(
                    join_result.validations[config].retried_keys
                ),
            }
            for config in FIXED_CONFIGS
        }
        finalization = {
            "schema_version": FINALIZATION_SCHEMA,
            "validation_status": "passed",
            "running_manifest_sha256": running_manifest_sha256,
            "artifacts": artifacts,
            "validations": {
                "prepared_inputs": {
                    "status": "passed",
                    "records": EXPECTED_TOTAL_EXAMPLES,
                    "protocol_config_hash": protocol_config_hash(),
                    "source_hashes_verified_against_pinned_release": True,
                    "exact_token_ids_rehashed": True,
                },
                "fixed_configurations": {
                    "status": "passed",
                    "configurations": fixed_validation,
                    "completed_generation_records": sum(
                        item["completed_records"]
                        for item in fixed_validation.values()
                    ),
                    "expected_generation_records": (
                        EXPECTED_GENERATIONS_PER_MODEL
                    ),
                    "cross_config_final_input_hash_sha256": (
                        join_result.cross_validation.combined_input_hash_sha256
                    ),
                    "pilot_outputs_used_in_3750_example_metrics": False,
                },
                "worst_case_24k_512_pilot_gate": {
                    "status": "passed",
                    "configurations": {
                        config: {
                            "journal_events": pilot_event_counts[config],
                            "full_512_token_decode": True,
                            "minimum_headroom_bytes": (
                                WORST_CASE_MIN_HEADROOM_BYTES
                            ),
                            "headroom_sufficient": True,
                        }
                        for config in FIXED_CONFIGS
                    },
                },
                "joined_analysis": {
                    "status": "passed",
                    "rows": len(joined_rows),
                    "schema_version": joined_analysis["schema_version"],
                },
                "oracle_analysis": {
                    "status": "passed",
                    "primary": PRIMARY_ORACLE_OBJECTIVE,
                    "required_diagnostic": QUALITY_ORACLE_DIAGNOSTIC,
                },
                "c6_held_out_router": {
                    "status": "passed",
                    "rows": len(c6_rows),
                    "schema_version": c6_summary["schema_version"],
                    "evaluation_mode": c6_summary["evaluation_mode"],
                    "folds": c6_summary["cross_validation"]["fold_count"],
                    "each_example_held_out_exactly_once": True,
                },
                "environment": {
                    "status": "passed",
                    "git_commit_matches_clean_preparation": True,
                    "clean_source_revalidated_at_finalization": True,
                    "gpu_metadata_present": True,
                },
            },
        }
        if (
            finalization["validations"]["fixed_configurations"][
                "completed_generation_records"
            ]
            != EXPECTED_GENERATIONS_PER_MODEL
        ):
            raise FinalizationError(
                "C0-C5 completed record count is not exactly 22,500"
            )

        if manifest["status"] == "complete":
            completed_at = manifest.get("completed_at")
            started = _parse_utc_timestamp(
                manifest.get("started_at"), field="manifest.started_at"
            )
            completed = _parse_utc_timestamp(
                completed_at, field="manifest.completed_at"
            )
            if completed < started:
                raise FinalizationError(
                    "manifest.completed_at precedes manifest.started_at"
                )
            running_payload = _running_payload_from_completed(manifest)
            reconstructed_hash = _canonical_json_sha256(
                running_payload, label="reconstructed running manifest"
            )
            if (
                reconstructed_hash
                != finalization["running_manifest_sha256"]
            ):
                raise FinalizationError(
                    "completed manifest no longer reconstructs its original "
                    "running identity"
                )
        else:
            if "completed_at" in manifest or "finalization" in manifest:
                raise FinalizationError(
                    "running manifest contains premature completion fields"
                )
            completed_at = _utc_now()
            started = _parse_utc_timestamp(
                manifest.get("started_at"), field="manifest.started_at"
            )
            completed = _parse_utc_timestamp(
                completed_at, field="new completion timestamp"
            )
            if completed < started:
                raise FinalizationError(
                    "system completion time precedes manifest.started_at"
                )
            running_payload = copy.deepcopy(dict(manifest))

        # Duplicate the transition time inside the validated finalization
        # envelope so a completed manifest cannot silently change only its
        # top-level completion timestamp on an idempotent recheck.
        finalization["completed_at"] = completed_at
        completed_payload = copy.deepcopy(running_payload)
        completed_payload["status"] = "complete"
        completed_payload["completed_at"] = completed_at
        completed_payload["finalization"] = finalization

        _assert_unchanged(required_paths, snapshot)
        if file_sha256(manifest_path) != initial_manifest_hash:
            raise FinalizationError("run manifest changed during finalization")

        if manifest["status"] == "complete":
            if dict(manifest) != completed_payload:
                raise FinalizationError(
                    "completed manifest is stale, mutated, or not reproducible"
                )
            return copy.deepcopy(dict(manifest))

        atomic_write_json(manifest_path, completed_payload, overwrite=True)
        published = _load_json_object(
            manifest_path, label="published completed manifest"
        )
        if published != completed_payload:
            raise FinalizationError(
                "atomically published manifest differs from validated payload"
            )
        return completed_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Exact production run directory containing manifest.json.",
    )
    parser.add_argument(
        "--prepared-dir",
        required=True,
        type=Path,
        help="Pinned complete prepared-token artifact used for generation.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
        help="Locked LongBench16 run configuration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = finalize_run(
        args.run_dir,
        args.prepared_dir,
        run_config=args.run_config,
    )
    manifest_path = Path(args.run_dir).expanduser().resolve() / MANIFEST_FILENAME
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_id": manifest["run_id"],
                "completed_at": manifest["completed_at"],
                "manifest_sha256": file_sha256(manifest_path),
                "validation_status": manifest["finalization"][
                    "validation_status"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FINALIZATION_SCHEMA",
    "FinalizationError",
    "finalize_run",
    "main",
]
