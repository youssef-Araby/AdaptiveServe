#!/usr/bin/env python3
"""Run exactly one C0-C5 configuration on immutable LongBench16 token IDs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .longbench16_io import (
        FIXED_CONFIGS,
        KV_ACCOUNTING_BY_CONFIG,
        AtomicJsonlJournal,
        RunLayout,
        assert_safe_write_path,
        atomic_write_json,
        collect_environment_metadata,
        file_sha256,
    )
    from .longbench16_prepare import (
        PREPARED_SCHEMA,
        TOKEN_HASH_ALGORITHM,
        token_ids_sha256,
    )
    from .longbench16_protocol import (
        EXPECTED_GENERATIONS_PER_MODEL,
        EXPECTED_TOTAL_EXAMPLES,
        FINAL_INPUT_TOKEN_CAP,
        LONG_BENCH_DATASET_ID,
        LONG_BENCH_DATASET_REVISION,
        LONG_BENCH_RESOLVED_DATASET_ID,
        LONG_BENCH_SPLIT,
        TASK_SPECS,
        config_file_hashes,
        postprocess_prediction,
        protocol_config_hash,
        score_prediction,
    )
except ImportError:  # Direct execution: python scripts/longbench16_run_config.py
    from longbench16_io import (
        FIXED_CONFIGS,
        KV_ACCOUNTING_BY_CONFIG,
        AtomicJsonlJournal,
        RunLayout,
        assert_safe_write_path,
        atomic_write_json,
        collect_environment_metadata,
        file_sha256,
    )
    from longbench16_prepare import (
        PREPARED_SCHEMA,
        TOKEN_HASH_ALGORITHM,
        token_ids_sha256,
    )
    from longbench16_protocol import (
        EXPECTED_GENERATIONS_PER_MODEL,
        EXPECTED_TOTAL_EXAMPLES,
        FINAL_INPUT_TOKEN_CAP,
        LONG_BENCH_DATASET_ID,
        LONG_BENCH_DATASET_REVISION,
        LONG_BENCH_RESOLVED_DATASET_ID,
        LONG_BENCH_SPLIT,
        TASK_SPECS,
        config_file_hashes,
        postprocess_prediction,
        protocol_config_hash,
        score_prediction,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_CONFIG = REPO_ROOT / "configs" / "longbench16_24k.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "longbench16_24k"

RUNNER_SCHEMA = "adaptiveserve-longbench16-runner/v1"
RUN_CONFIG_SCHEMA = "adaptiveserve-longbench16-run-config/v1"
LOCKED_PROTOCOL_ID = "longbench-v1-non-zh-24k"
LOCKED_MODEL = {
    "alias": "llama31_8b",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    "tokenizer_id": "meta-llama/Llama-3.1-8B-Instruct",
    "tokenizer_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    "dtype": "float16",
    "single_gpu_required": True,
    "cpu_or_disk_offload_allowed": False,
}
LOCKED_NATIVE_STOP_TOKEN_IDS = (128001, 128008, 128009)
PILOT_PROFILE_WORST_CASE_24K_512 = "worst_case_24k_512"
PILOT_PROFILES = (PILOT_PROFILE_WORST_CASE_24K_512,)
WORST_CASE_MIN_HEADROOM_BYTES = 1536 * 1024 * 1024
FEATURE_KEYS = (
    "seq_len_tokens",
    "seq_len_chars",
    "token_entropy",
    "gzip_ratio",
    "unique_token_ratio",
    "question_position",
    "newline_density",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class RunnerValidationError(RuntimeError):
    """Raised when a locked runner invariant does not hold."""


class PreparedInputValidationError(RunnerValidationError):
    """Raised when prepared metadata or token artifacts are inconsistent."""


class ModelPlacementError(RunnerValidationError):
    """Raised when the model is not wholly resident on one selected GPU."""


@dataclass(frozen=True)
class LockedRunConfig:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @property
    def model(self) -> Mapping[str, Any]:
        return self.payload["model"]

    @property
    def model_alias(self) -> str:
        return self.model["alias"]

    def method(self, configuration: str) -> Mapping[str, Any]:
        return self.payload["configurations"][configuration]


@dataclass(frozen=True)
class PreparedExample:
    model: str
    model_id: str
    task: str
    category: str
    benchmark_id: str
    source_index: int
    metric: str
    references: tuple[Any, ...]
    all_classes: tuple[str, ...] | None
    features: Mapping[str, Any]
    pre_truncation_token_count: int
    post_truncation_token_count: int
    truncated: bool
    final_input_token_sha256: str
    token_hash_algorithm: str
    max_new_tokens: int
    minimum_new_tokens: int
    stop_on_newline_token: bool
    input_ids: np.ndarray

    @property
    def key(self) -> tuple[str, str]:
        return self.task, self.benchmark_id


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    index_path: Path
    index_sha256: str
    examples: tuple[PreparedExample, ...]

    @property
    def expected_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(example.key for example in self.examples)


@dataclass(frozen=True)
class GenerationDispatch:
    generate: Callable[..., tuple[str, int, dict]]
    kwargs: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RunSummary:
    configuration: str
    journal_path: Path
    selected: int
    completed_before: int
    completed_now: int
    skipped_completed: int
    pilot: bool


class CudaFeasibilityProbe:
    """Measure conservative CUDA headroom around one feasibility generation."""

    def __init__(self, torch_module: Any, device: Any) -> None:
        self._torch = torch_module
        self._device = device
        self._baseline: dict[str, int] | None = None

    def begin(self) -> None:
        if self._baseline is not None:
            raise RunnerValidationError("CUDA feasibility probe was reused")
        cuda = self._torch.cuda
        cuda.synchronize(self._device)
        free_bytes, total_bytes = cuda.mem_get_info(self._device)
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
        reserved_bytes = int(cuda.memory_reserved(self._device))
        if (
            total_bytes <= 0
            or free_bytes < 0
            or free_bytes > total_bytes
            or reserved_bytes < 0
        ):
            raise RunnerValidationError(
                "CUDA memory APIs returned invalid pre-run values"
            )
        non_torch_bytes = max(
            0, total_bytes - free_bytes - reserved_bytes
        )
        self._baseline = {
            "total_vram_bytes": total_bytes,
            "pre_run_free_bytes": free_bytes,
            "pre_run_torch_reserved_bytes": reserved_bytes,
            "pre_run_non_torch_bytes": non_torch_bytes,
        }
        cuda.reset_peak_memory_stats(self._device)
        cuda.synchronize(self._device)

    def finish(self) -> dict[str, Any]:
        if self._baseline is None:
            raise RunnerValidationError(
                "CUDA feasibility probe was not started"
            )
        cuda = self._torch.cuda
        cuda.synchronize(self._device)
        peak_allocated = int(cuda.max_memory_allocated(self._device))
        peak_reserved = int(cuda.max_memory_reserved(self._device))
        if peak_allocated < 0 or peak_reserved < 0:
            raise RunnerValidationError(
                "CUDA memory APIs returned invalid peak values"
            )
        total = self._baseline["total_vram_bytes"]
        non_torch = self._baseline["pre_run_non_torch_bytes"]
        headroom = total - non_torch - peak_reserved
        return {
            **self._baseline,
            "peak_torch_allocated_bytes": peak_allocated,
            "peak_torch_reserved_bytes": peak_reserved,
            "conservative_headroom_bytes": headroom,
            "required_headroom_bytes": WORST_CASE_MIN_HEADROOM_BYTES,
            "headroom_sufficient": headroom >= WORST_CASE_MIN_HEADROOM_BYTES,
            "feasibility_only": True,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerValidationError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerValidationError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerValidationError(f"{label} must contain a JSON object")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerValidationError(f"{field} must be an object")
    return value


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise RunnerValidationError(
            f"{field} mismatch: observed {actual!r}, expected {expected!r}"
        )


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunnerValidationError(f"{field} must be a positive integer")
    return value


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def reject_legacy_sample_limit(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Reject the legacy environment sample limiter in every execution path."""
    environment = os.environ if environ is None else environ
    if "ADAPTIVESERVE_LB_N" in environment:
        raise RunnerValidationError(
            "ADAPTIVESERVE_LB_N is forbidden by the final 3,750-example protocol"
        )


def load_locked_run_config(path: Path | str) -> LockedRunConfig:
    """Load and semantically validate the one locked LongBench16 run config."""
    config_path = Path(path).expanduser().resolve()
    payload = _load_json(config_path, label="run config")

    _require_equal(
        payload.get("schema_version"),
        RUN_CONFIG_SCHEMA,
        field="run config schema_version",
    )
    _require_equal(
        payload.get("protocol_id"),
        LOCKED_PROTOCOL_ID,
        field="run config protocol_id",
    )

    scope = _mapping(payload.get("scope"), field="run config scope")
    _require_equal(
        scope.get("benchmark_examples"),
        EXPECTED_TOTAL_EXAMPLES,
        field="scope.benchmark_examples",
    )
    _require_equal(
        scope.get("configurations"),
        list(FIXED_CONFIGS),
        field="scope.configurations",
    )
    _require_equal(
        scope.get("expected_generation_records"),
        EXPECTED_GENERATIONS_PER_MODEL,
        field="scope.expected_generation_records",
    )
    _require_equal(scope.get("speed_in_scope"), False, field="scope.speed_in_scope")
    _require_equal(
        scope.get("perplexity_in_scope"),
        False,
        field="scope.perplexity_in_scope",
    )

    model = _mapping(payload.get("model"), field="run config model")
    for key, expected in LOCKED_MODEL.items():
        _require_equal(model.get(key), expected, field=f"model.{key}")
    if not _MODEL_REVISION_RE.fullmatch(model["model_revision"]):
        raise RunnerValidationError("model.model_revision is not a commit SHA")
    if not _MODEL_REVISION_RE.fullmatch(model["tokenizer_revision"]):
        raise RunnerValidationError("model.tokenizer_revision is not a commit SHA")

    input_policy = _mapping(
        payload.get("input_policy"), field="run config input_policy"
    )
    _require_equal(
        input_policy.get("final_input_token_cap"),
        FINAL_INPUT_TOKEN_CAP,
        field="input_policy.final_input_token_cap",
    )
    _require_equal(
        input_policy.get("decode_and_retokenize_after_truncation"),
        False,
        field="input_policy.decode_and_retokenize_after_truncation",
    )
    _require_equal(
        input_policy.get("persist_prepared_token_ids"),
        True,
        field="input_policy.persist_prepared_token_ids",
    )
    _require_equal(
        input_policy.get("token_id_hash"),
        TOKEN_HASH_ALGORITHM,
        field="input_policy.token_id_hash",
    )
    _require_equal(
        input_policy.get("no_chat_add_special_tokens"),
        True,
        field="input_policy.no_chat_add_special_tokens",
    )
    _require_equal(
        input_policy.get("chat_template_date_string"),
        "20 Jul 2026",
        field="input_policy.chat_template_date_string",
    )

    generation = _mapping(
        payload.get("generation_policy"), field="run config generation_policy"
    )
    expected_generation = {
        "decoding": "greedy",
        "do_sample": False,
        "num_beams": 1,
        "task_specific_max_new_tokens": True,
        "default_stop": "pinned_model_native_eos_token_ids",
        "native_stop_token_ids": list(LOCKED_NATIVE_STOP_TOKEN_IDS),
        "samsum_minimum_new_tokens": 1,
        "samsum_additional_stop": "last_tokenizer_id_for_newline",
    }
    for key, expected in expected_generation.items():
        _require_equal(
            generation.get(key), expected, field=f"generation_policy.{key}"
        )

    configurations = _mapping(
        payload.get("configurations"), field="run config configurations"
    )
    _require_equal(
        tuple(configurations),
        FIXED_CONFIGS,
        field="configurations order",
    )
    for configuration in FIXED_CONFIGS:
        method_config = _mapping(
            configurations.get(configuration),
            field=f"configurations.{configuration}",
        )
        _require_equal(
            method_config.get("kv_accounting"),
            KV_ACCOUNTING_BY_CONFIG[configuration],
            field=f"configurations.{configuration}.kv_accounting",
        )

    router = _mapping(payload.get("router_dataset"), field="router_dataset")
    _require_equal(
        router.get("primary_key"),
        ["task", "benchmark_id"],
        field="router_dataset.primary_key",
    )
    _require_equal(
        router.get("feature_keys"),
        list(FEATURE_KEYS),
        field="router_dataset.feature_keys",
    )
    _require_equal(
        router.get("label_configurations"),
        list(FIXED_CONFIGS),
        field="router_dataset.label_configurations",
    )

    return LockedRunConfig(
        path=config_path,
        sha256=file_sha256(config_path),
        payload=copy.deepcopy(payload),
    )


def _artifact_path(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreparedInputValidationError(f"{label} path must be non-empty")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise PreparedInputValidationError(f"{label} path must be relative")
    try:
        resolved = (root / relative_path).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise PreparedInputValidationError(
            f"{label} escapes or is missing from prepared root: {relative!r}"
        ) from exc
    if not resolved.is_file():
        raise PreparedInputValidationError(f"{label} is not a file: {resolved}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise PreparedInputValidationError(
            f"prepared index has a partial trailing line: {path}"
        )
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            raise PreparedInputValidationError(
                f"{path}:{line_number} is blank"
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreparedInputValidationError(
                f"{path}:{line_number} is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(value, dict):
            raise PreparedInputValidationError(
                f"{path}:{line_number} must contain an object"
            )
        records.append(value)
    return records


def _prepared_input_hash(record: Mapping[str, Any], *, row_number: int) -> str:
    canonical = record.get("final_input_token_sha256")
    legacy = record.get("final_input_ids_sha256")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise PreparedInputValidationError(
            f"index row {row_number} disagrees across canonical and legacy "
            "final-input hashes"
        )
    digest = canonical if canonical is not None else legacy
    if not _valid_sha256(digest):
        raise PreparedInputValidationError(
            f"index row {row_number} lacks a valid final-input token hash"
        )
    return str(digest).lower()


def _validate_index_record(
    record: Mapping[str, Any],
    *,
    row_number: int,
    expected_task: str,
    expected_source_index: int,
    spec: Any,
    locked: LockedRunConfig,
) -> str:
    prefix = f"index row {row_number}"
    required_equal = {
        "schema_version": PREPARED_SCHEMA,
        "model": locked.model_alias,
        "model_id": locked.model["model_id"],
        "task": expected_task,
        "category": spec.category,
        "source_index": expected_source_index,
        "metric": spec.scorer,
        "token_hash_algorithm": TOKEN_HASH_ALGORITHM,
        "max_new_tokens": spec.max_new_tokens,
        "minimum_new_tokens": spec.minimum_new_tokens,
        "stop_on_newline_token": spec.stop_on_newline_token,
        "token_file": f"tokens/{expected_task}.npz",
        "token_offset_index": expected_source_index,
    }
    for field, expected in required_equal.items():
        if record.get(field) != expected:
            raise PreparedInputValidationError(
                f"{prefix} field {field!r} is {record.get(field)!r}; "
                f"expected {expected!r}"
            )

    benchmark_id = record.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise PreparedInputValidationError(
            f"{prefix} has an invalid benchmark_id"
        )
    references = record.get("references")
    if not isinstance(references, list) or not references:
        raise PreparedInputValidationError(f"{prefix} has no references")
    all_classes = record.get("all_classes")
    if all_classes is not None and (
        not isinstance(all_classes, list)
        or not all(isinstance(value, str) for value in all_classes)
    ):
        raise PreparedInputValidationError(f"{prefix} has invalid all_classes")
    features = record.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURE_KEYS):
        raise PreparedInputValidationError(
            f"{prefix} feature keys do not match the locked router contract"
        )

    pre_count = record.get("pre_truncation_token_count")
    post_count = record.get("post_truncation_token_count")
    truncated = record.get("truncated")
    if (
        isinstance(pre_count, bool)
        or not isinstance(pre_count, int)
        or pre_count <= 0
        or isinstance(post_count, bool)
        or not isinstance(post_count, int)
        or not 0 < post_count <= FINAL_INPUT_TOKEN_CAP
        or not isinstance(truncated, bool)
    ):
        raise PreparedInputValidationError(
            f"{prefix} has invalid token counts or truncation flag"
        )
    if truncated:
        if pre_count <= FINAL_INPUT_TOKEN_CAP or post_count != FINAL_INPUT_TOKEN_CAP:
            raise PreparedInputValidationError(
                f"{prefix} truncated counts violate the 24K policy"
            )
    elif pre_count != post_count:
        raise PreparedInputValidationError(
            f"{prefix} changed token count without truncation"
        )
    return _prepared_input_hash(record, row_number=row_number)


def load_prepared_dataset(
    prepared_dir: Path | str,
    locked: LockedRunConfig,
    *,
    task_specs: Mapping[str, Any] = TASK_SPECS,
    expected_total: int = EXPECTED_TOTAL_EXAMPLES,
    np_module: Any = np,
) -> PreparedDataset:
    """Validate every prepared artifact and return immutable row/token views."""
    root = Path(prepared_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PreparedInputValidationError(
            f"prepared input root is not a directory: {root}"
        )
    manifest_path = root / "manifest.json"
    status_path = root / "status.json"
    manifest = _load_json(manifest_path, label="prepared manifest")
    status = _load_json(status_path, label="prepared status")
    manifest_digest = file_sha256(manifest_path)

    _require_equal(
        manifest.get("schema_version"),
        PREPARED_SCHEMA,
        field="prepared manifest schema_version",
    )
    _require_equal(
        manifest.get("status"), "complete", field="prepared manifest status"
    )
    _require_equal(
        status.get("schema_version"),
        PREPARED_SCHEMA,
        field="prepared status schema_version",
    )
    _require_equal(status.get("status"), "complete", field="prepared status")
    _require_equal(
        status.get("manifest_sha256"),
        manifest_digest,
        field="prepared status manifest_sha256",
    )
    _require_equal(
        manifest.get("records"), expected_total, field="prepared manifest records"
    )
    _require_equal(
        manifest.get("protocol_config_hash"),
        protocol_config_hash(),
        field="prepared protocol_config_hash",
    )
    _require_equal(
        manifest.get("protocol_config_files"),
        config_file_hashes(),
        field="prepared protocol_config_files",
    )

    manifest_run_config = _mapping(
        manifest.get("run_config"), field="prepared manifest run_config"
    )
    _require_equal(
        manifest_run_config.get("sha256"),
        locked.sha256,
        field="prepared run_config sha256",
    )
    tokenizer_metadata = _mapping(
        manifest.get("tokenizer"), field="prepared manifest tokenizer"
    )
    _require_equal(
        tokenizer_metadata.get("model_id"),
        locked.model["tokenizer_id"],
        field="prepared tokenizer model_id",
    )
    _require_equal(
        tokenizer_metadata.get("requested_revision"),
        locked.model["tokenizer_revision"],
        field="prepared tokenizer revision",
    )
    prepared_policy = _mapping(
        manifest.get("input_policy"), field="prepared manifest input_policy"
    )
    _require_equal(
        prepared_policy.get("cap"),
        FINAL_INPUT_TOKEN_CAP,
        field="prepared input cap",
    )
    _require_equal(
        prepared_policy.get("token_hash_algorithm"),
        TOKEN_HASH_ALGORITHM,
        field="prepared token hash algorithm",
    )
    _require_equal(
        prepared_policy.get("decode_and_retokenize_after_truncation"),
        False,
        field="prepared decode/re-tokenize policy",
    )
    _require_equal(
        prepared_policy.get("no_chat_add_special_tokens"),
        True,
        field="prepared no-chat add_special_tokens policy",
    )
    _require_equal(
        prepared_policy.get("chat_template_date_string"),
        "20 Jul 2026",
        field="prepared chat-template date string",
    )
    benchmark = _mapping(
        manifest.get("benchmark"), field="prepared manifest benchmark"
    )
    _require_equal(
        benchmark.get("requested_dataset_id"),
        LONG_BENCH_DATASET_ID,
        field="prepared requested dataset ID",
    )
    _require_equal(
        benchmark.get("resolved_dataset_id"),
        LONG_BENCH_RESOLVED_DATASET_ID,
        field="prepared resolved dataset ID",
    )
    _require_equal(
        benchmark.get("dataset_revision"),
        LONG_BENCH_DATASET_REVISION,
        field="prepared dataset revision",
    )
    _require_equal(
        benchmark.get("split"),
        LONG_BENCH_SPLIT,
        field="prepared dataset split",
    )
    if benchmark.get("source_hashes_verified_against_pinned_release") is not True:
        raise PreparedInputValidationError(
            "prepared benchmark source hashes were not verified against the "
            "pinned release"
        )
    source_code = _mapping(
        manifest.get("source_code"), field="prepared manifest source_code"
    )
    source_commit = source_code.get("commit")
    if (
        not isinstance(source_commit, str)
        or not _MODEL_REVISION_RE.fullmatch(source_commit)
    ):
        raise PreparedInputValidationError(
            "prepared inputs do not identify an exact source-code commit"
        )
    if source_code.get("dirty") is not False:
        raise PreparedInputValidationError(
            "prepared inputs were not built from a clean source tree"
        )

    index_metadata = _mapping(
        manifest.get("prepared_index"), field="prepared manifest prepared_index"
    )
    index_path = _artifact_path(
        root, index_metadata.get("path"), label="prepared index"
    )
    index_digest = file_sha256(index_path)
    _require_equal(
        index_metadata.get("sha256"),
        index_digest,
        field="prepared index sha256",
    )
    _require_equal(
        index_metadata.get("rows"), expected_total, field="prepared index rows"
    )
    records = _read_jsonl(index_path)
    if len(records) != expected_total:
        raise PreparedInputValidationError(
            f"prepared index has {len(records)} rows; expected {expected_total}"
        )

    expected_from_specs = sum(
        int(spec.expected_test_examples) for spec in task_specs.values()
    )
    if expected_from_specs != expected_total:
        raise PreparedInputValidationError(
            f"task specs sum to {expected_from_specs}; expected {expected_total}"
        )

    rows_by_task: dict[str, list[tuple[dict[str, Any], str]]] = {
        task: [] for task in task_specs
    }
    cursor = 0
    composite_keys: set[tuple[str, str]] = set()
    observed_truncations: Counter[str] = Counter()
    for task, spec in task_specs.items():
        for source_index in range(spec.expected_test_examples):
            record = records[cursor]
            digest = _validate_index_record(
                record,
                row_number=cursor + 1,
                expected_task=task,
                expected_source_index=source_index,
                spec=spec,
                locked=locked,
            )
            key = (task, record["benchmark_id"])
            if key in composite_keys:
                raise PreparedInputValidationError(
                    f"prepared index contains duplicate key {key!r}"
                )
            composite_keys.add(key)
            rows_by_task[task].append((record, digest))
            observed_truncations[task] += int(record["truncated"])
            cursor += 1

    expected_truncations = (
        locked.payload.get("input_policy", {})
        .get("expected_primary_truncation_counts")
    )
    if expected_truncations is not None and dict(observed_truncations) != dict(
        expected_truncations
    ):
        raise PreparedInputValidationError(
            "prepared truncation counts differ from the locked run config"
        )
    manifest_truncations = manifest.get("truncation_counts")
    if manifest_truncations is not None and dict(observed_truncations) != dict(
        manifest_truncations
    ):
        raise PreparedInputValidationError(
            "prepared index truncation counts differ from its manifest"
        )
    _require_equal(
        manifest.get("total_truncated"),
        sum(observed_truncations.values()),
        field="prepared total_truncated",
    )

    token_entries = manifest.get("token_files")
    if not isinstance(token_entries, list):
        raise PreparedInputValidationError(
            "prepared manifest token_files must be a list"
        )
    if [entry.get("task") for entry in token_entries if isinstance(entry, dict)] != list(
        task_specs
    ):
        raise PreparedInputValidationError(
            "prepared token file task order does not match the protocol"
        )

    examples: list[PreparedExample] = []
    for entry, (task, spec) in zip(
        token_entries, task_specs.items(), strict=True
    ):
        if not isinstance(entry, dict):
            raise PreparedInputValidationError(
                f"prepared token entry for {task} is not an object"
            )
        token_path = _artifact_path(
            root, entry.get("path"), label=f"{task} token NPZ"
        )
        token_digest = file_sha256(token_path)
        _require_equal(
            entry.get("sha256"),
            token_digest,
            field=f"{task} token NPZ sha256",
        )
        _require_equal(
            entry.get("rows"),
            spec.expected_test_examples,
            field=f"{task} token rows",
        )
        try:
            with np_module.load(token_path, allow_pickle=False) as archive:
                if set(archive.files) != {"input_ids", "offsets"}:
                    raise PreparedInputValidationError(
                        f"{task} token NPZ has unexpected arrays {archive.files}"
                    )
                token_ids = np_module.asarray(archive["input_ids"])
                offsets = np_module.asarray(archive["offsets"])
        except PreparedInputValidationError:
            raise
        except Exception as exc:
            raise PreparedInputValidationError(
                f"cannot load {task} token NPZ: {token_path}"
            ) from exc

        if token_ids.ndim != 1 or token_ids.dtype != np_module.dtype("<u4"):
            raise PreparedInputValidationError(
                f"{task} input_ids must be one-dimensional little-endian uint32"
            )
        if offsets.ndim != 1 or offsets.dtype != np_module.dtype("<u8"):
            raise PreparedInputValidationError(
                f"{task} offsets must be one-dimensional little-endian uint64"
            )
        if len(offsets) != spec.expected_test_examples + 1:
            raise PreparedInputValidationError(
                f"{task} offsets length does not equal rows + 1"
            )
        offsets_int = [int(value) for value in offsets]
        if (
            offsets_int[0] != 0
            or offsets_int[-1] != len(token_ids)
            or any(
                right <= left
                for left, right in zip(offsets_int, offsets_int[1:])
            )
        ):
            raise PreparedInputValidationError(
                f"{task} offsets are not strict, bounded row offsets"
            )
        _require_equal(
            entry.get("stored_token_ids"),
            len(token_ids),
            field=f"{task} stored_token_ids",
        )

        for source_index, (record, expected_digest) in enumerate(
            rows_by_task[task]
        ):
            start, end = offsets_int[source_index : source_index + 2]
            row_ids = token_ids[start:end]
            if len(row_ids) != record["post_truncation_token_count"]:
                raise PreparedInputValidationError(
                    f"{task} row {source_index} token count differs from index"
                )
            observed_digest = token_ids_sha256(row_ids)
            if observed_digest != expected_digest:
                raise PreparedInputValidationError(
                    f"{task} row {source_index} token hash mismatch"
                )
            examples.append(
                PreparedExample(
                    model=record["model"],
                    model_id=record["model_id"],
                    task=task,
                    category=record["category"],
                    benchmark_id=record["benchmark_id"],
                    source_index=source_index,
                    metric=record["metric"],
                    references=tuple(record["references"]),
                    all_classes=(
                        tuple(record["all_classes"])
                        if record["all_classes"] is not None
                        else None
                    ),
                    features=copy.deepcopy(record["features"]),
                    pre_truncation_token_count=record[
                        "pre_truncation_token_count"
                    ],
                    post_truncation_token_count=record[
                        "post_truncation_token_count"
                    ],
                    truncated=record["truncated"],
                    final_input_token_sha256=expected_digest,
                    token_hash_algorithm=record["token_hash_algorithm"],
                    max_new_tokens=record["max_new_tokens"],
                    minimum_new_tokens=record["minimum_new_tokens"],
                    stop_on_newline_token=record["stop_on_newline_token"],
                    input_ids=row_ids,
                )
            )

    if len(examples) != expected_total:
        raise PreparedInputValidationError(
            f"validated {len(examples)} examples; expected {expected_total}"
        )
    return PreparedDataset(
        root=root,
        manifest=copy.deepcopy(manifest),
        manifest_sha256=manifest_digest,
        index_path=index_path,
        index_sha256=index_digest,
        examples=tuple(examples),
    )


def _run_manifest_payload(
    layout: RunLayout,
    locked: LockedRunConfig,
    prepared: PreparedDataset,
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA,
        "status": "running",
        "started_at": _utc_now(),
        "run_id": layout.run_id,
        "model_alias": locked.model_alias,
        "model_id": locked.model["model_id"],
        "model_revision": locked.model["model_revision"],
        "tokenizer_id": locked.model["tokenizer_id"],
        "tokenizer_revision": locked.model["tokenizer_revision"],
        "run_config": {
            "path": str(locked.path),
            "sha256": locked.sha256,
        },
        "prepared_inputs": {
            "path": str(prepared.root),
            "manifest_sha256": prepared.manifest_sha256,
            "index_sha256": prepared.index_sha256,
            "records": len(prepared.examples),
        },
        "source_code": copy.deepcopy(dict(prepared.manifest["source_code"])),
    }


def _validate_existing_run_manifest(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for field in (
        "schema_version",
        "run_id",
        "model_alias",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "run_config",
        "prepared_inputs",
        "source_code",
    ):
        if existing.get(field) != expected.get(field):
            raise RunnerValidationError(
                f"existing run manifest field {field!r} does not match this run"
            )
    if existing.get("status") == "complete":
        raise RunnerValidationError(
            "completed run identities are immutable and cannot be resumed"
        )
    if existing.get("status") != "running":
        raise RunnerValidationError(
            f"existing run manifest has invalid status {existing.get('status')!r}"
        )


def current_source_code_state() -> dict[str, Any]:
    """Return the current code revision, ignoring only generated run outputs."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                ":(exclude)runs/longbench16_24k/**",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerValidationError(
            "cannot resolve the current source-code revision"
        ) from exc
    if not _MODEL_REVISION_RE.fullmatch(commit):
        raise RunnerValidationError(
            "current source-code revision is not an exact git commit"
        )
    return {"commit": commit, "dirty": bool(status.strip())}


def validate_runtime_source_code(
    prepared: PreparedDataset,
    *,
    current_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require every configuration to run from the preparation commit."""

    expected = _mapping(
        prepared.manifest.get("source_code"),
        field="prepared manifest source_code",
    )
    observed = (
        current_source_code_state()
        if current_state is None
        else dict(current_state)
    )
    if observed.get("dirty") is not False:
        raise RunnerValidationError(
            "canonical generation requires a clean source tree"
        )
    if observed.get("commit") != expected.get("commit"):
        raise RunnerValidationError(
            "current source-code commit differs from the prepared-input commit"
        )
    return copy.deepcopy(dict(observed))


def open_or_create_run_layout(
    *,
    run_root: Path | str,
    run_id: str,
    locked: LockedRunConfig,
    prepared: PreparedDataset,
    environment_factory: Callable[..., Mapping[str, Any]] = (
        collect_environment_metadata
    ),
) -> RunLayout:
    """Create one immutable run namespace or validate it for exact resume."""
    safe_root = assert_safe_write_path(run_root)
    environment: Mapping[str, Any] | None = None
    try:
        layout = RunLayout.open(safe_root, locked.model_alias, run_id)
        created = False
    except FileNotFoundError:
        environment = environment_factory(
            extra={
                "run_id": run_id,
                "model_alias": locked.model_alias,
                "run_config_sha256": locked.sha256,
                "prepared_manifest_sha256": prepared.manifest_sha256,
            }
        )
        layout = RunLayout.create(safe_root, locked.model_alias, run_id)
        created = True

    expected = _run_manifest_payload(layout, locked, prepared)
    if created:
        atomic_write_json(layout.manifest_path, expected)
        if environment is None:  # defensive: created runs always capture first
            raise AssertionError("run environment was not captured")
        atomic_write_json(layout.environment_path, environment)
    else:
        existing = _load_json(layout.manifest_path, label="run manifest")
        _validate_existing_run_manifest(existing, expected)
        if not layout.environment_path.is_file():
            raise RunnerValidationError(
                f"existing run lacks environment metadata: {layout.environment_path}"
            )
    return layout


def journal_for_configuration(
    layout: RunLayout,
    configuration: str,
    *,
    pilot: bool,
    pilot_profile: str | None = None,
) -> AtomicJsonlJournal:
    if configuration not in FIXED_CONFIGS:
        raise ValueError(f"unknown fixed configuration {configuration!r}")
    if pilot_profile is not None:
        if not pilot:
            raise RunnerValidationError(
                "a pilot profile requires pilot mode"
            )
        if pilot_profile not in PILOT_PROFILES:
            raise RunnerValidationError(
                f"unknown pilot profile {pilot_profile!r}"
            )
        path = (
            layout.pilots_dir
            / pilot_profile
            / configuration
            / "per_prompt.jsonl"
        )
        return AtomicJsonlJournal(path)
    if pilot:
        path = layout.pilots_dir / configuration / "per_prompt.jsonl"
        return AtomicJsonlJournal(path)
    return layout.journal(configuration)


def select_prepared_examples(
    prepared: PreparedDataset,
    *,
    pilot: bool,
    limit: int | None,
    pilot_profile: str | None,
) -> tuple[PreparedExample, ...]:
    """Select final, generic-pilot, or deterministic profile rows."""
    if pilot_profile is not None and not pilot:
        raise RunnerValidationError("a pilot profile requires --pilot")
    if limit is not None:
        if not pilot:
            raise RunnerValidationError("--limit is allowed only with --pilot")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RunnerValidationError("--limit must be a positive integer")
        if limit > len(prepared.examples):
            raise RunnerValidationError(
                "--limit exceeds the number of prepared examples"
            )
    if pilot_profile is not None and limit is not None:
        raise RunnerValidationError(
            "--limit cannot be combined with --pilot-profile"
        )
    if pilot_profile is None:
        return (
            prepared.examples[:limit]
            if limit is not None
            else prepared.examples
        )
    if pilot_profile != PILOT_PROFILE_WORST_CASE_24K_512:
        raise RunnerValidationError(
            f"unknown pilot profile {pilot_profile!r}"
        )
    for example in prepared.examples:
        if (
            example.post_truncation_token_count == FINAL_INPUT_TOKEN_CAP
            and example.max_new_tokens == 512
        ):
            return (example,)
    raise RunnerValidationError(
        "worst_case_24k_512 requires a prepared example with exactly "
        "24,000 input tokens and max_new_tokens=512"
    )


def _validate_profile_journal_scope(
    snapshot: Any,
    selected: Sequence[PreparedExample],
    *,
    pilot_profile: str | None,
) -> None:
    if pilot_profile is None:
        return
    selected_keys = {example.key for example in selected}
    extra = set(snapshot.histories) - selected_keys
    if extra:
        raise RunnerValidationError(
            f"pilot profile {pilot_profile!r} journal contains unrelated "
            f"prepared keys: {sorted(map(str, extra))[:10]}"
        )
    for key, history in snapshot.histories.items():
        for event in history:
            if event.get("pilot_profile") != pilot_profile:
                raise RunnerValidationError(
                    f"pilot profile journal key {key!r} has the wrong profile"
                )
            metadata = event.get("method_metadata")
            if not isinstance(metadata, Mapping) or metadata.get(
                "pilot_profile"
            ) != pilot_profile:
                raise RunnerValidationError(
                    f"pilot profile journal key {key!r} lacks profile metadata"
                )
            feasibility = metadata.get("cuda_feasibility")
            if feasibility is not None:
                validated = _validated_cuda_feasibility(feasibility)
                if (
                    event.get("status") == "completed"
                    and validated["headroom_sufficient"] is not True
                ):
                    raise RunnerValidationError(
                        f"pilot profile journal key {key!r} completed without "
                        "the required CUDA headroom"
                    )
            elif event.get("status") == "completed":
                raise RunnerValidationError(
                    f"pilot profile journal key {key!r} completed without "
                    "CUDA feasibility metadata"
                )
            if event.get("status") == "completed":
                if event.get("generated_token_count") != event.get(
                    "max_new_tokens"
                ):
                    raise RunnerValidationError(
                        f"pilot profile journal key {key!r} did not execute "
                        "the full configured decode"
                    )
                if metadata.get("forced_full_decode") is not True:
                    raise RunnerValidationError(
                        f"pilot profile journal key {key!r} lacks the forced "
                        "full-decode marker"
                    )


def validate_worst_case_pilot_gate(
    layout: RunLayout,
    prepared: PreparedDataset,
) -> dict[str, str]:
    """Require passed full-decode 24K+512 evidence for every fixed method."""

    selected = select_prepared_examples(
        prepared,
        pilot=True,
        limit=None,
        pilot_profile=PILOT_PROFILE_WORST_CASE_24K_512,
    )
    expected_keys = frozenset(example.key for example in selected)
    hashes: dict[str, str] = {}
    for configuration in FIXED_CONFIGS:
        journal = journal_for_configuration(
            layout,
            configuration,
            pilot=True,
            pilot_profile=PILOT_PROFILE_WORST_CASE_24K_512,
        )
        snapshot = validate_resume_journal(
            journal,
            prepared,
            configuration=configuration,
        )
        _validate_profile_journal_scope(
            snapshot,
            selected,
            pilot_profile=PILOT_PROFILE_WORST_CASE_24K_512,
        )
        if snapshot.completed_keys != expected_keys:
            raise RunnerValidationError(
                "final generation is blocked until the worst-case 24K+512 "
                f"pilot passes for {configuration}"
            )
        hashes[configuration] = file_sha256(journal.path)
    return hashes


def _runtime_tokenizer_check(
    tokenizer: Any,
    *,
    locked: LockedRunConfig,
    prepared_manifest: Mapping[str, Any],
) -> None:
    metadata = _mapping(
        prepared_manifest.get("tokenizer"), field="prepared tokenizer metadata"
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise RunnerValidationError("runtime tokenizer has no valid EOS token ID")
    _require_equal(
        eos_token_id,
        metadata.get("eos_token_id"),
        field="runtime tokenizer eos_token_id",
    )
    _require_equal(
        int(vocab_size),
        metadata.get("vocab_size"),
        field="runtime tokenizer vocab_size",
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_digest = (
        hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if isinstance(chat_template, str)
        else None
    )
    _require_equal(
        chat_digest,
        metadata.get("chat_template_sha256"),
        field="runtime tokenizer chat_template hash",
    )
    commit_hash = getattr(tokenizer, "_commit_hash", None)
    if commit_hash is not None and commit_hash != locked.model["tokenizer_revision"]:
        raise RunnerValidationError(
            f"runtime tokenizer resolved unexpected revision {commit_hash}"
        )


def assert_model_on_single_cuda(model: Any, device: Any) -> None:
    """Fail unless every parameter and buffer is on the selected CUDA device."""
    if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) is None:
        raise ModelPlacementError("selected device must be an explicit CUDA index")
    expected_index = device.index
    tensors = [
        *(("parameter", name, tensor) for name, tensor in model.named_parameters()),
        *(("buffer", name, tensor) for name, tensor in model.named_buffers()),
    ]
    if not tensors:
        raise ModelPlacementError("model exposes no parameters or buffers")
    misplaced = [
        f"{kind}:{name}={tensor.device}"
        for kind, name, tensor in tensors
        if getattr(tensor.device, "type", None) != "cuda"
        or getattr(tensor.device, "index", None) != expected_index
    ]
    if misplaced:
        raise ModelPlacementError(
            "model is not wholly resident on the selected CUDA device; "
            + ", ".join(misplaced[:10])
        )
    if getattr(model, "training", True):
        raise ModelPlacementError("model must be in eval mode")
    modules = getattr(model, "modules", lambda: ())()
    if any(getattr(module, "training", False) for module in modules):
        raise ModelPlacementError("at least one model submodule remains in train mode")
    if getattr(model, "hf_device_map", None):
        raise ModelPlacementError("device_map/offload placement is forbidden")


def load_model_and_tokenizer(
    *,
    locked: LockedRunConfig,
    prepared_manifest: Mapping[str, Any],
    configuration: str,
    device_name: str,
    allow_download: bool,
    torch_module: Any | None = None,
    auto_model_cls: Any | None = None,
    auto_tokenizer_cls: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Load the pinned FP16 model without device maps or offload."""
    if torch_module is None:
        import torch as torch_module
    if auto_model_cls is None or auto_tokenizer_cls is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        auto_model_cls = auto_model_cls or AutoModelForCausalLM
        auto_tokenizer_cls = auto_tokenizer_cls or AutoTokenizer

    device = torch_module.device(device_name)
    if device.type != "cuda" or device.index is None:
        raise ModelPlacementError(
            "--device must be an explicit CUDA device such as cuda:0"
        )
    if not torch_module.cuda.is_available():
        raise ModelPlacementError("CUDA is required for the locked run")
    if not 0 <= device.index < torch_module.cuda.device_count():
        raise ModelPlacementError(f"CUDA device index is unavailable: {device_name}")
    torch_module.cuda.set_device(device)

    tokenizer = auto_tokenizer_cls.from_pretrained(
        locked.model["tokenizer_id"],
        revision=locked.model["tokenizer_revision"],
        local_files_only=not allow_download,
    )
    _runtime_tokenizer_check(
        tokenizer, locked=locked, prepared_manifest=prepared_manifest
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict[str, Any] = {
        "revision": locked.model["model_revision"],
        "local_files_only": not allow_download,
        "dtype": torch_module.float16,
    }
    attention_implementation = {
        "C0": None,
        "C1": "eager",
        "C2": "eager",
        "C3": "sdpa",
        "C4": "sdpa",
        "C5": "sdpa",
    }[configuration]
    if attention_implementation is not None:
        load_kwargs["attn_implementation"] = attention_implementation
    model = auto_model_cls.from_pretrained(
        locked.model["model_id"],
        **load_kwargs,
    )
    model = model.to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if (
        resolved_revision is not None
        and resolved_revision != locked.model["model_revision"]
    ):
        raise RunnerValidationError(
            f"runtime model resolved unexpected revision {resolved_revision}"
        )
    generation_config = getattr(model, "generation_config", None)
    runtime_eos = getattr(generation_config, "eos_token_id", None)
    if isinstance(runtime_eos, int) and not isinstance(runtime_eos, bool):
        runtime_eos_ids = (runtime_eos,)
    elif isinstance(runtime_eos, (list, tuple)) and all(
        isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
        for token_id in runtime_eos
    ):
        runtime_eos_ids = tuple(runtime_eos)
    else:
        raise RunnerValidationError(
            "runtime model generation config has invalid EOS token IDs"
        )
    _require_equal(
        runtime_eos_ids,
        LOCKED_NATIVE_STOP_TOKEN_IDS,
        field="runtime model native EOS token IDs",
    )
    assert_model_on_single_cuda(model, device)
    return model, tokenizer, device


def _load_benchmark_module(configuration: str) -> Any:
    names = {
        "C0": "benchmark_c0_baseline",
        "C1": "benchmark_c1_tailorkv",
        "C2": "benchmark_c2_qaq",
        "C3": "benchmark_c3_kvquant",
        "C4": "benchmark_c4_dynamickv",
        "C5": "benchmark_c5_adakv",
    }
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return importlib.import_module(names[configuration])


def _assert_module_constant(
    module: Any,
    attribute: str,
    expected: Any,
    *,
    configuration: str,
) -> None:
    observed = getattr(module, attribute, object())
    if observed != expected:
        raise RunnerValidationError(
            f"{configuration} implementation constant {attribute}={observed!r} "
            f"does not match locked run config value {expected!r}"
        )


def build_generation_dispatch(
    *,
    configuration: str,
    locked: LockedRunConfig,
    model: Any,
    tokenizer: Any,
    device: Any,
    torch_module: Any | None = None,
    modules: Mapping[str, Any] | None = None,
) -> GenerationDispatch:
    """Bind one prepared generator to exact locked method parameters."""
    if configuration not in FIXED_CONFIGS:
        raise ValueError(f"unknown fixed configuration {configuration!r}")
    if torch_module is None:
        import torch as torch_module
    module = (
        modules[configuration]
        if modules is not None
        else _load_benchmark_module(configuration)
    )
    config = locked.method(configuration)
    kwargs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    if configuration == "C0":
        generate = module.generate_prepared_c0
    elif configuration == "C1":
        constants = {
            "TKV_BITS": config.get("quantization_bits"),
            "TKV_GROUP_SIZE": config.get("group_size"),
            "TKV_N_LOCAL": config.get("recent_tokens"),
            "TKV_N_TOPK": config.get("topk_prefix_tokens"),
            "TKV_WINDOW_SIZE": config.get("attention_window"),
            "TKV_KERNEL_SIZE": config.get("smoothing_kernel"),
        }
        for attribute, expected in constants.items():
            _assert_module_constant(
                module, attribute, expected, configuration=configuration
            )
        raw_layers = config.get("quantized_layers")
        if (
            not isinstance(raw_layers, list)
            or not raw_layers
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                for layer in raw_layers
            )
            or len(raw_layers) != len(set(raw_layers))
        ):
            raise RunnerValidationError("C1 quantized_layers is invalid")
        kwargs["q_layers"] = set(raw_layers)
        metadata["q_layers"] = sorted(raw_layers)
        generate = module.generate_prepared_c1
    elif configuration == "C2":
        constants = {
            "QAQ_OUTLIER_RATIO": config.get("outlier_fraction"),
            "QAQ_N_BITS_MIN": config.get("minimum_bits"),
            "QAQ_N_BITS_MAX": config.get("maximum_bits"),
            "QAQ_LAST_N_ATTENTIONS": config.get("last_attention_queries"),
            "QAQ_TARGET_ERROR": config.get("target_error"),
            "QAQ_Q_NORM_PERCENTILE": config.get("q_norm_percentile"),
        }
        for attribute, expected in constants.items():
            _assert_module_constant(
                module, attribute, expected, configuration=configuration
            )
        _require_equal(
            config.get("q_norm_calibration"),
            "first_128_tokens_of_versioned_calibration_text",
            field="C2 q_norm calibration policy",
        )
        _require_equal(
            config.get("q_norm_calibration_text_source"),
            "scripts/_common.py:SPEED_TEXT",
            field="C2 q_norm calibration text source",
        )
        _require_equal(
            config.get("q_norm_calibration_add_special_tokens"),
            False,
            field="C2 q_norm calibration add_special_tokens",
        )
        calibration_token_count = _require_positive_int(
            config.get("q_norm_calibration_token_count"),
            field="C2 q_norm calibration token count",
        )
        _require_equal(
            calibration_token_count,
            128,
            field="C2 q_norm calibration token count",
        )
        calibration_text_digest = hashlib.sha256(
            module.SPEED_TEXT.encode("utf-8")
        ).hexdigest()
        _require_equal(
            calibration_text_digest,
            config.get("q_norm_calibration_text_sha256"),
            field="C2 q_norm calibration text sha256",
        )
        calibration_ids = tokenizer.encode(
            module.SPEED_TEXT, add_special_tokens=False
        )
        if (
            not isinstance(calibration_ids, list)
            or len(calibration_ids) < calibration_token_count
            or not all(
                isinstance(token_id, int) and token_id >= 0
                for token_id in calibration_ids
            )
        ):
            raise RunnerValidationError(
                "C2 calibration tokenizer output is not at least 128 token IDs"
            )
        calibration_ids = calibration_ids[:calibration_token_count]
        calibration_token_digest = token_ids_sha256(calibration_ids)
        _require_equal(
            calibration_token_digest,
            config.get("q_norm_calibration_token_sha256"),
            field="C2 q_norm calibration token sha256",
        )
        calibration_tensor = torch_module.tensor(
            [calibration_ids], dtype=torch_module.long, device=device
        )
        calibration_result = module._precompute_q_norm(
            model,
            calibration_tensor,
            str(device),
            return_metadata=True,
        )
        if (
            not isinstance(calibration_result, tuple)
            or len(calibration_result) != 2
            or not isinstance(calibration_result[1], Mapping)
        ):
            raise RunnerValidationError(
                "C2 q_norm calibration did not return coverage metadata"
            )
        q_norm = float(calibration_result[0])
        calibration_coverage = dict(calibration_result[1])
        captured_layers = calibration_coverage.get("captured_layers")
        expected_layers = calibration_coverage.get("expected_layers")
        captured_values = calibration_coverage.get("captured_q_norm_values")
        if (
            isinstance(captured_layers, bool)
            or not isinstance(captured_layers, int)
            or isinstance(expected_layers, bool)
            or not isinstance(expected_layers, int)
            or captured_layers <= 0
            or captured_layers != expected_layers
            or isinstance(captured_values, bool)
            or not isinstance(captured_values, int)
            or captured_values <= 0
        ):
            raise RunnerValidationError(
                "C2 q_norm calibration has incomplete layer/value coverage"
            )
        if not math.isfinite(q_norm) or q_norm <= 0:
            raise RunnerValidationError(f"C2 produced invalid q_norm {q_norm!r}")
        attention_aware_decode = config.get("attention_aware_decode")
        if not isinstance(attention_aware_decode, bool):
            raise RunnerValidationError(
                "C2 attention_aware_decode must be boolean"
            )
        kwargs.update(
            q_norm=q_norm,
            attn_aware_decode=attention_aware_decode,
        )
        metadata.update(
            q_norm=q_norm,
            q_norm_calibration_text_sha256=calibration_text_digest,
            q_norm_calibration_token_count=calibration_token_count,
            q_norm_calibration_token_sha256=calibration_token_digest,
            q_norm_calibration_coverage=calibration_coverage,
        )
        generate = module.generate_prepared_c2
    elif configuration == "C3":
        constants = {
            "KVQ_BITS": config.get("quantization_bits"),
            "KVQ_GROUP_SIZE": config.get("group_size"),
            "KVQ_OUTLIER_FRAC": config.get("outlier_fraction"),
        }
        for attribute, expected in constants.items():
            _assert_module_constant(
                module, attribute, expected, configuration=configuration
            )
        generate = module.generate_prepared_c3
    elif configuration == "C4":
        constants = {
            "DKV_WINDOW_SIZE": config.get("recent_window"),
            "DKV_KERNEL_SIZE": config.get("smoothing_kernel"),
            "DKV_RATIO_MAX": config.get("maximum_layer_budget_ratio"),
        }
        for attribute, expected in constants.items():
            _assert_module_constant(
                module, attribute, expected, configuration=configuration
            )
        budget = _require_positive_int(
            config.get("mean_per_layer_budget"),
            field="C4 mean_per_layer_budget",
        )
        kwargs["budget"] = budget
        metadata["budget"] = budget
        generate = module.generate_prepared_c4
    else:
        constants = {
            "ADA_WINDOW_SIZE": config.get("recent_window"),
            "ADA_KERNEL_SIZE": config.get("smoothing_kernel"),
        }
        for attribute, expected in constants.items():
            _assert_module_constant(
                module, attribute, expected, configuration=configuration
            )
        budget = _require_positive_int(
            config.get("per_head_budget"), field="C5 per_head_budget"
        )
        n_kv_heads = _require_positive_int(
            module._get_n_kv_heads(model), field="C5 model n_kv_heads"
        )
        kwargs.update(
            n_kv_heads=n_kv_heads,
            budget_per_head=budget,
        )
        metadata.update(
            n_kv_heads=n_kv_heads,
            budget_per_head=budget,
        )
        generate = module.generate_prepared_c5

    metadata["kv_accounting"] = copy.deepcopy(
        KV_ACCOUNTING_BY_CONFIG[configuration]
    )
    return GenerationDispatch(
        generate=generate,
        kwargs=kwargs,
        metadata=metadata,
    )


def stop_token_ids_for_example(
    example: PreparedExample,
    tokenizer: Any,
    *,
    model: Any | None = None,
) -> frozenset[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
        raise RunnerValidationError("tokenizer EOS token ID is invalid")
    generation_config = (
        getattr(model, "generation_config", None)
        if model is not None
        else None
    )
    native_eos = getattr(generation_config, "eos_token_id", None)
    if native_eos is None:
        stop_ids = {eos_id}
    elif isinstance(native_eos, int) and not isinstance(native_eos, bool):
        stop_ids = {native_eos}
    elif isinstance(native_eos, (list, tuple)) and all(
        isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
        for token_id in native_eos
    ):
        stop_ids = set(native_eos)
    else:
        raise RunnerValidationError(
            "model generation config has invalid EOS token IDs"
        )
    if eos_id not in stop_ids:
        raise RunnerValidationError(
            "tokenizer EOS token ID is absent from the model native EOS set"
        )
    if example.stop_on_newline_token:
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if (
            not isinstance(newline_ids, list)
            or not newline_ids
            or not all(
                isinstance(token_id, int) and token_id >= 0
                for token_id in newline_ids
            )
        ):
            raise RunnerValidationError(
                "SAMSum newline did not encode to a non-empty token-ID list"
            )
        stop_ids.add(newline_ids[-1])
    return frozenset(stop_ids)


def _base_record(
    configuration: str,
    example: PreparedExample,
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "model": example.model,
        "model_id": example.model_id,
        "task": example.task,
        "category": example.category,
        "benchmark_id": example.benchmark_id,
        "source_index": example.source_index,
        "metric": example.metric,
        "references": list(example.references),
        "all_classes": (
            list(example.all_classes)
            if example.all_classes is not None
            else None
        ),
        "prediction": None,
        "score": None,
        "features": copy.deepcopy(dict(example.features)),
        "pre_truncation_token_count": example.pre_truncation_token_count,
        "post_truncation_token_count": example.post_truncation_token_count,
        "truncated": example.truncated,
        "final_input_token_sha256": example.final_input_token_sha256,
        "token_hash_algorithm": example.token_hash_algorithm,
        "max_new_tokens": example.max_new_tokens,
        "minimum_new_tokens": example.minimum_new_tokens,
        "stop_on_newline_token": example.stop_on_newline_token,
        "generated_token_count": None,
        "kv_bytes": None,
        "kv_bytes_fp16": None,
        "compression": None,
    }


def _validated_generation_result(
    result: Any,
    *,
    minimum_new_tokens: int,
    max_new_tokens: int,
) -> tuple[str, int, float, float, float]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise RunnerValidationError(
            "prepared generator must return (prediction, n_generated, kv_stats)"
        )
    prediction, generated_count, kv_stats = result
    if not isinstance(prediction, str):
        raise RunnerValidationError("prepared generator prediction must be text")
    if (
        isinstance(generated_count, bool)
        or not isinstance(generated_count, int)
        or not minimum_new_tokens <= generated_count <= max_new_tokens
    ):
        raise RunnerValidationError(
            "prepared generator returned a generated-token count outside the "
            "prepared minimum/maximum bounds"
        )
    if not isinstance(kv_stats, Mapping):
        raise RunnerValidationError("prepared generator KV stats must be an object")
    values: dict[str, float] = {}
    for field in ("kv_bytes", "kv_bytes_fp16", "compression"):
        value = kv_stats.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RunnerValidationError(f"KV stat {field} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise RunnerValidationError(f"KV stat {field} must be finite and positive")
        values[field] = number
    ratio = values["kv_bytes_fp16"] / values["kv_bytes"]
    if not math.isclose(
        values["compression"], ratio, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise RunnerValidationError(
            "prepared generator compression disagrees with FP16/effective bytes"
        )
    return (
        prediction,
        generated_count,
        values["kv_bytes"],
        values["kv_bytes_fp16"],
        ratio,
    )


def validate_resume_journal(
    journal: AtomicJsonlJournal,
    prepared: PreparedDataset,
    *,
    configuration: str,
) -> Any:
    """Validate existing history against prepared keys and immutable hashes."""
    snapshot = journal.snapshot()
    snapshot.ensure_consistent()
    by_key = {example.key: example for example in prepared.examples}
    unexpected = set(snapshot.histories) - set(by_key)
    if unexpected:
        raise RunnerValidationError(
            f"journal contains unexpected prepared keys: "
            f"{sorted(map(str, unexpected))[:10]}"
        )
    for key, history in snapshot.histories.items():
        example = by_key[key]
        for event in history:
            if event.get("configuration") != configuration:
                raise RunnerValidationError(
                    f"journal key {key!r} declares wrong configuration"
                )
            if event.get("source_index") != example.source_index:
                raise RunnerValidationError(
                    f"journal key {key!r} source_index changed"
                )
            if (
                event.get("final_input_token_sha256")
                != example.final_input_token_sha256
            ):
                raise RunnerValidationError(
                    f"journal key {key!r} final-input hash changed"
                )
    return snapshot


def _validated_cuda_feasibility(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerValidationError(
            "CUDA feasibility probe must return an object"
        )
    integer_fields = (
        "total_vram_bytes",
        "pre_run_free_bytes",
        "pre_run_torch_reserved_bytes",
        "pre_run_non_torch_bytes",
        "peak_torch_allocated_bytes",
        "peak_torch_reserved_bytes",
        "conservative_headroom_bytes",
        "required_headroom_bytes",
    )
    result: dict[str, Any] = {}
    for field in integer_fields:
        field_value = value.get(field)
        if isinstance(field_value, bool) or not isinstance(field_value, int):
            raise RunnerValidationError(
                f"CUDA feasibility field {field} must be an integer"
            )
        if field != "conservative_headroom_bytes" and field_value < 0:
            raise RunnerValidationError(
                f"CUDA feasibility field {field} cannot be negative"
            )
        result[field] = field_value
    if result["total_vram_bytes"] <= 0:
        raise RunnerValidationError(
            "CUDA feasibility total_vram_bytes must be positive"
        )
    if result["peak_torch_allocated_bytes"] > result[
        "peak_torch_reserved_bytes"
    ]:
        raise RunnerValidationError(
            "CUDA peak allocated bytes exceed peak reserved bytes"
        )
    expected_headroom = (
        result["total_vram_bytes"]
        - result["pre_run_non_torch_bytes"]
        - result["peak_torch_reserved_bytes"]
    )
    if result["conservative_headroom_bytes"] != expected_headroom:
        raise RunnerValidationError(
            "CUDA feasibility conservative headroom is inconsistent"
        )
    if result["required_headroom_bytes"] != WORST_CASE_MIN_HEADROOM_BYTES:
        raise RunnerValidationError(
            "CUDA feasibility uses the wrong required headroom"
        )
    sufficient = expected_headroom >= WORST_CASE_MIN_HEADROOM_BYTES
    if value.get("headroom_sufficient") is not sufficient:
        raise RunnerValidationError(
            "CUDA feasibility headroom_sufficient is inconsistent"
        )
    if value.get("feasibility_only") is not True:
        raise RunnerValidationError(
            "CUDA feasibility measurement must be marked feasibility-only"
        )
    result["headroom_sufficient"] = sufficient
    result["feasibility_only"] = True
    return result


def run_prepared_journal(
    *,
    configuration: str,
    prepared: PreparedDataset,
    journal: AtomicJsonlJournal,
    dispatch: GenerationDispatch,
    model: Any,
    tokenizer: Any,
    device: Any,
    pilot: bool,
    limit: int | None = None,
    pilot_profile: str | None = None,
    cuda_feasibility_factory: Callable[[Any], Any] | None = None,
    score_fn: Callable[..., float] = score_prediction,
    postprocess_fn: Callable[[str, str], str] = postprocess_prediction,
) -> RunSummary:
    """Run or exactly resume one append-only configuration journal."""
    if configuration not in FIXED_CONFIGS:
        raise ValueError(f"unknown fixed configuration {configuration!r}")
    selected = select_prepared_examples(
        prepared,
        pilot=pilot,
        limit=limit,
        pilot_profile=pilot_profile,
    )
    snapshot = validate_resume_journal(
        journal, prepared, configuration=configuration
    )
    _validate_profile_journal_scope(
        snapshot, selected, pilot_profile=pilot_profile
    )
    completed_before = len(snapshot.completed_keys)
    completed_now = 0
    skipped = 0

    for example in selected:
        if example.key in snapshot.completed_keys:
            skipped += 1
            continue
        allow_retry = example.key in snapshot.histories
        base = _base_record(configuration, example)
        method_metadata = copy.deepcopy(dict(dispatch.metadata))
        if "kv_accounting" in method_metadata:
            base["kv_accounting"] = copy.deepcopy(
                method_metadata["kv_accounting"]
            )
        if pilot_profile is not None:
            base["pilot_profile"] = pilot_profile
            method_metadata["pilot_profile"] = pilot_profile
            method_metadata["forced_full_decode"] = True
        try:
            stop_ids = stop_token_ids_for_example(
                example, tokenizer, model=model
            )
            if pilot_profile == PILOT_PROFILE_WORST_CASE_24K_512:
                # This is a memory-feasibility probe, not a quality result.
                # Suppressing terminal IDs is required to materialize all
                # 512 decode positions rather than merely declare max_new=512.
                stop_ids = frozenset()
            base["effective_stop_token_ids"] = sorted(stop_ids)
            feasibility_probe = None
            if pilot_profile == PILOT_PROFILE_WORST_CASE_24K_512:
                if cuda_feasibility_factory is None:
                    raise RunnerValidationError(
                        "worst_case_24k_512 requires a CUDA feasibility probe"
                    )
                feasibility_probe = cuda_feasibility_factory(device)
                if not callable(getattr(feasibility_probe, "begin", None)):
                    raise RunnerValidationError(
                        "CUDA feasibility probe lacks begin()"
                    )
                if not callable(getattr(feasibility_probe, "finish", None)):
                    raise RunnerValidationError(
                        "CUDA feasibility probe lacks finish()"
                    )
                feasibility_probe.begin()
            try:
                result = dispatch.generate(
                    model,
                    tokenizer,
                    example.input_ids.tolist(),
                    example.max_new_tokens,
                    str(device),
                    stop_token_ids=stop_ids,
                    min_new_tokens=example.minimum_new_tokens,
                    **dispatch.kwargs,
                )
            finally:
                if feasibility_probe is not None:
                    method_metadata["cuda_feasibility"] = (
                        _validated_cuda_feasibility(
                            feasibility_probe.finish()
                        )
                    )
            feasibility = method_metadata.get("cuda_feasibility")
            if (
                isinstance(feasibility, Mapping)
                and feasibility.get("headroom_sufficient") is not True
            ):
                raise RunnerValidationError(
                    "worst_case_24k_512 has less than 1.5 GiB of "
                    "conservative CUDA headroom"
                )
            (
                raw_prediction,
                generated_count,
                kv_bytes,
                kv_bytes_fp16,
                compression,
            ) = _validated_generation_result(
                result,
                minimum_new_tokens=example.minimum_new_tokens,
                max_new_tokens=example.max_new_tokens,
            )
            prediction = postprocess_fn(example.task, raw_prediction)
            score = float(
                score_fn(
                    example.task,
                    raw_prediction,
                    example.references,
                    all_classes=example.all_classes,
                )
            )
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RunnerValidationError(
                    f"protocol scorer returned invalid score {score!r}"
                )
            completed_record = {
                **base,
                "prediction": prediction,
                "score": score,
                "generated_token_count": generated_count,
                "kv_bytes": kv_bytes,
                "kv_bytes_fp16": kv_bytes_fp16,
                "compression": compression,
                "status": "completed",
            }
            if method_metadata:
                completed_record["method_metadata"] = method_metadata
        except Exception as exc:
            failed_record = {
                **base,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if method_metadata:
                failed_record["method_metadata"] = method_metadata
            journal.append(failed_record, allow_retry=allow_retry)
            raise
        journal.append(completed_record, allow_retry=allow_retry)
        completed_now += 1

    final_snapshot = validate_resume_journal(
        journal, prepared, configuration=configuration
    )
    _validate_profile_journal_scope(
        final_snapshot, selected, pilot_profile=pilot_profile
    )
    if not pilot and final_snapshot.completed_keys != frozenset(
        prepared.expected_keys
    ):
        missing = set(prepared.expected_keys) - set(final_snapshot.completed_keys)
        raise RunnerValidationError(
            f"final {configuration} journal remains incomplete: "
            f"{len(missing)} missing keys"
        )
    return RunSummary(
        configuration=configuration,
        journal_path=journal.path,
        selected=len(selected),
        completed_before=completed_before,
        completed_now=completed_now,
        skipped_completed=skipped,
        pilot=pilot,
    )


def execute(args: argparse.Namespace) -> RunSummary:
    """Validate, load, and execute one CLI-selected configuration."""
    reject_legacy_sample_limit()
    locked = load_locked_run_config(args.run_config)
    prepared = load_prepared_dataset(args.prepared_dir, locked)
    validate_runtime_source_code(prepared)
    selected = select_prepared_examples(
        prepared,
        pilot=args.pilot,
        limit=args.limit,
        pilot_profile=args.pilot_profile,
    )
    layout = open_or_create_run_layout(
        run_root=args.run_root,
        run_id=args.run_id,
        locked=locked,
        prepared=prepared,
    )
    if not args.pilot:
        validate_worst_case_pilot_gate(layout, prepared)
    journal = journal_for_configuration(
        layout,
        args.configuration,
        pilot=args.pilot,
        pilot_profile=args.pilot_profile,
    )

    snapshot = validate_resume_journal(
        journal, prepared, configuration=args.configuration
    )
    _validate_profile_journal_scope(
        snapshot, selected, pilot_profile=args.pilot_profile
    )
    if all(example.key in snapshot.completed_keys for example in selected):
        if not args.pilot and snapshot.completed_keys != frozenset(
            prepared.expected_keys
        ):
            raise RunnerValidationError(
                "selected rows are complete but the final journal is incomplete"
            )
        return RunSummary(
            configuration=args.configuration,
            journal_path=journal.path,
            selected=len(selected),
            completed_before=len(snapshot.completed_keys),
            completed_now=0,
            skipped_completed=len(selected),
            pilot=args.pilot,
        )

    import torch

    model, tokenizer, device = load_model_and_tokenizer(
        locked=locked,
        prepared_manifest=prepared.manifest,
        configuration=args.configuration,
        device_name=args.device,
        allow_download=args.allow_download,
        torch_module=torch,
    )
    dispatch = build_generation_dispatch(
        configuration=args.configuration,
        locked=locked,
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_module=torch,
    )
    try:
        return run_prepared_journal(
            configuration=args.configuration,
            prepared=prepared,
            journal=journal,
            dispatch=dispatch,
            model=model,
            tokenizer=tokenizer,
            device=device,
            pilot=args.pilot,
            limit=args.limit,
            pilot_profile=args.pilot_profile,
            cuda_feasibility_factory=(
                lambda selected_device: CudaFeasibilityProbe(
                    torch, selected_device
                )
            ),
        )
    finally:
        del model
        torch.cuda.empty_cache()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        "--config",
        dest="configuration",
        choices=FIXED_CONFIGS,
        required=True,
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow pinned Hugging Face downloads; default is cache-only",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="write under the pilots namespace, never the final C0-C5 journal",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="pilot-only total row limit",
    )
    parser.add_argument(
        "--pilot-profile",
        choices=PILOT_PROFILES,
        default=None,
        help=(
            "deterministic pilot selector with a profile-specific journal "
            "namespace"
        ),
    )
    args = parser.parse_args(argv)
    if args.limit is not None and not args.pilot:
        parser.error("--limit is allowed only with --pilot")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.pilot_profile is not None and not args.pilot:
        parser.error("--pilot-profile requires --pilot")
    if args.pilot_profile is not None and args.limit is not None:
        parser.error("--limit cannot be combined with --pilot-profile")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = execute(args)
    mode = (
        f"pilot:{args.pilot_profile}"
        if args.pilot_profile is not None
        else ("pilot" if summary.pilot else "final")
    )
    print(
        f"{summary.configuration} {mode}: selected={summary.selected}, "
        f"completed_now={summary.completed_now}, "
        f"skipped={summary.skipped_completed}, journal={summary.journal_path}"
    )


if __name__ == "__main__":
    main()
