#!/usr/bin/env python3
"""Leakage-safe C6 routing for the locked LongBench16 24K experiment.

C6 is a post-hoc router: it trains only on the joined C0--C5 measurements and
does not run the language model again.  Production defaults are deliberately
fail-closed against ``configs/longbench16_24k.json``.  The only injection
boundaries are explicit expected counts, folds, scalers, and regressors for
small CPU tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from .longbench16_io import (
        DEFAULT_KEY_FIELDS,
        KV_ACCOUNTING_BY_CONFIG,
        RunLayout,
        assert_safe_write_path,
        atomic_write_json,
        file_sha256,
    )
    from .longbench16_join import (
        COMPRESSION_ABS_TOLERANCE,
        COMPRESSION_REL_TOLERANCE,
        CONFIG_OUTPUT_FIELDS,
        FEATURE_KEYS,
        JOIN_SCHEMA,
        REQUIRED_SHARED_FIELDS,
        atomic_write_jsonl,
        load_prepared_expectations,
    )
    from .longbench16_protocol import (
        EXPECTED_TOTAL_EXAMPLES,
        TASK_ORDER,
        TASK_SPECS,
        aggregate_compression,
        aggregate_quality,
    )
    from .longbench16_run_config import (
        RUNNER_SCHEMA,
        current_source_code_state,
    )
except ImportError:  # Direct execution: python scripts/longbench16_c6.py
    from longbench16_io import (
        DEFAULT_KEY_FIELDS,
        KV_ACCOUNTING_BY_CONFIG,
        RunLayout,
        assert_safe_write_path,
        atomic_write_json,
        file_sha256,
    )
    from longbench16_join import (
        COMPRESSION_ABS_TOLERANCE,
        COMPRESSION_REL_TOLERANCE,
        CONFIG_OUTPUT_FIELDS,
        FEATURE_KEYS,
        JOIN_SCHEMA,
        REQUIRED_SHARED_FIELDS,
        atomic_write_jsonl,
        load_prepared_expectations,
    )
    from longbench16_protocol import (
        EXPECTED_TOTAL_EXAMPLES,
        TASK_ORDER,
        TASK_SPECS,
        aggregate_compression,
        aggregate_quality,
    )
    from longbench16_run_config import (
        RUNNER_SCHEMA,
        current_source_code_state,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = REPO_ROOT / "configs" / "longbench16_24k.json"

C6_ROW_SCHEMA = "adaptiveserve-longbench16-c6-row/v1"
C6_ANALYSIS_SCHEMA = "adaptiveserve-longbench16-c6-analysis/v1"
REFERENCE_CONFIGURATION = "C0"
CANDIDATE_CONFIGURATIONS = ("C1", "C2", "C3", "C4", "C5")
LABEL_CONFIGURATIONS = (REFERENCE_CONFIGURATION, *CANDIDATE_CONFIGURATIONS)
QUESTION_POSITION_SENTINEL = -1.0
DEFAULT_ROWS_FILENAME = "c6_per_prompt.jsonl"
DEFAULT_SUMMARY_FILENAME = "c6_summary.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_EXPECTED_ROUTING_POLICY: dict[str, Any] = {
    "reference_configuration": REFERENCE_CONFIGURATION,
    "candidate_configurations": list(CANDIDATE_CONFIGURATIONS),
    "iso_quality_tau": 0.99,
    "quality_threshold": (
        "predicted_candidate_score_gte_tau_times_max_of_predicted_c0_and_"
        "train_mean_actual_c0"
    ),
    "candidate_ranking": "descending_train_fold_harmonic_mean_compression",
    "fallback": "highest_predicted_quality_candidate",
    "filter_candidates_by_train_mean_quality": False,
    "deterministic_tie_order": list(CANDIDATE_CONFIGURATIONS),
    "regressor": {
        "class": "sklearn.ensemble.HistGradientBoostingRegressor",
        "max_iter": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": 0,
    },
    "feature_scaler": (
        "sklearn.preprocessing.StandardScaler_fit_on_training_fold"
    ),
}
_EXPECTED_CROSS_VALIDATION: dict[str, Any] = {
    "kind": "task_stratified_k_fold",
    "folds": 10,
    "shuffle": True,
    "seed": 0,
    "regressor_random_state": 0,
}


class C6PolicyError(ValueError):
    """Raised when the run config differs from the locked C6 policy."""


class C6ValidationError(ValueError):
    """Raised when joined rows or injected folds violate the C6 contract."""


class C6ProvenanceError(C6ValidationError):
    """Raised when a production run is not bound to clean source provenance."""


@dataclass(frozen=True, slots=True)
class C6Policy:
    """Validated, immutable view of the primary C6 operating point."""

    reference_configuration: str
    candidate_configurations: tuple[str, ...]
    tau: float
    folds: int
    shuffle: bool
    seed: int
    regressor_parameters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_configuration": self.reference_configuration,
            "candidate_configurations": list(self.candidate_configurations),
            "iso_quality_tau": self.tau,
            "quality_threshold": _EXPECTED_ROUTING_POLICY[
                "quality_threshold"
            ],
            "candidate_ranking": _EXPECTED_ROUTING_POLICY["candidate_ranking"],
            "fallback": _EXPECTED_ROUTING_POLICY["fallback"],
            "filter_candidates_by_train_mean_quality": False,
            "deterministic_tie_order": list(self.candidate_configurations),
            "regressor": dict(self.regressor_parameters),
            "feature_scaler": _EXPECTED_ROUTING_POLICY["feature_scaler"],
        }


@dataclass(frozen=True, slots=True)
class C6Result:
    """Held-out routing rows and the associated protocol summary."""

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


ScalerFactory = Callable[[str, int], Any]
RegressorFactory = Callable[[str, int], Any]
Fold = tuple[Sequence[int], Sequence[int]]


def load_locked_policy(path: Path | str = DEFAULT_RUN_CONFIG) -> C6Policy:
    """Load the run config and reject any drift from the locked C6 protocol."""

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise C6PolicyError(f"run config is missing: {config_path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C6PolicyError(f"run config is not readable JSON: {config_path}") from exc
    if not isinstance(document, dict):
        raise C6PolicyError("run config must be a JSON object")
    router_dataset = document.get("router_dataset")
    if not isinstance(router_dataset, dict):
        raise C6PolicyError("run config lacks router_dataset")
    locked_fields = {
        "primary_key": list(DEFAULT_KEY_FIELDS),
        "feature_keys": list(FEATURE_KEYS),
        "label_configurations": list(LABEL_CONFIGURATIONS),
        "routing_policy": _EXPECTED_ROUTING_POLICY,
        "primary_cross_validation": _EXPECTED_CROSS_VALIDATION,
    }
    for field, expected in locked_fields.items():
        if router_dataset.get(field) != expected:
            raise C6PolicyError(
                f"router_dataset.{field} differs from the locked C6 protocol"
            )
    return C6Policy(
        reference_configuration=REFERENCE_CONFIGURATION,
        candidate_configurations=CANDIDATE_CONFIGURATIONS,
        tau=0.99,
        folds=10,
        shuffle=True,
        seed=0,
        regressor_parameters=copy.deepcopy(
            _EXPECTED_ROUTING_POLICY["regressor"]
        ),
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C6ValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise C6ValidationError(f"{field} must be finite")
    return number


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise C6ValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise C6ValidationError(f"{field} must be at least {minimum}")
    return value


def _expected_counts(
    expected_task_counts: Mapping[str, int] | None,
) -> OrderedDict[str, int]:
    source: Mapping[str, Any]
    if expected_task_counts is None:
        source = {
            task: TASK_SPECS[task].expected_test_examples for task in TASK_ORDER
        }
    else:
        source = expected_task_counts
    if set(source) != set(TASK_ORDER):
        missing = sorted(set(TASK_ORDER) - set(source))
        extra = sorted(set(source) - set(TASK_ORDER))
        raise C6ValidationError(
            "expected_task_counts must cover the exact 16-task panel; "
            f"missing={missing}, extra={extra}"
        )
    normalized: OrderedDict[str, int] = OrderedDict()
    for task in TASK_ORDER:
        count = _integer(source[task], f"expected_task_counts[{task!r}]", minimum=1)
        normalized[task] = count
    if expected_task_counts is None and sum(normalized.values()) != (
        EXPECTED_TOTAL_EXAMPLES
    ):
        raise C6ValidationError(
            "official task counts do not sum to the locked 3,750 examples"
        )
    return normalized


def _feature_vector(features: Any, key: tuple[str, str]) -> tuple[float, ...]:
    if not isinstance(features, Mapping):
        raise C6ValidationError(f"{key}: features must be an object")
    if set(features) != set(FEATURE_KEYS):
        missing = sorted(set(FEATURE_KEYS) - set(features))
        extra = sorted(set(features) - set(FEATURE_KEYS))
        raise C6ValidationError(
            f"{key}: expected exactly seven feature keys; "
            f"missing={missing}, extra={extra}"
        )

    seq_tokens = _integer(
        features["seq_len_tokens"],
        f"{key} features.seq_len_tokens",
        minimum=0,
    )
    seq_chars = _integer(
        features["seq_len_chars"],
        f"{key} features.seq_len_chars",
        minimum=0,
    )
    entropy = _finite_number(
        features["token_entropy"], f"{key} features.token_entropy"
    )
    gzip_ratio = _finite_number(
        features["gzip_ratio"], f"{key} features.gzip_ratio"
    )
    unique_ratio = _finite_number(
        features["unique_token_ratio"], f"{key} features.unique_token_ratio"
    )
    newline_density = _finite_number(
        features["newline_density"], f"{key} features.newline_density"
    )
    if entropy < 0 or gzip_ratio <= 0:
        raise C6ValidationError(
            f"{key}: token_entropy must be nonnegative and gzip_ratio positive"
        )
    if not 0 <= unique_ratio <= 1 or not 0 <= newline_density <= 1:
        raise C6ValidationError(
            f"{key}: ratio features must fall in [0, 1]"
        )
    question_position = features["question_position"]
    if question_position is None:
        question_value = QUESTION_POSITION_SENTINEL
    else:
        question_value = _finite_number(
            question_position, f"{key} features.question_position"
        )
        if not 0 <= question_value <= 1:
            raise C6ValidationError(
                f"{key}: features.question_position must be null or in [0, 1]"
            )
    return (
        float(seq_tokens),
        float(seq_chars),
        entropy,
        gzip_ratio,
        unique_ratio,
        question_value,
        newline_density,
    )


def _validate_labels(row: Mapping[str, Any], key: tuple[str, str]) -> None:
    for config in LABEL_CONFIGURATIONS:
        missing = [
            f"{config}_{field}"
            for field in CONFIG_OUTPUT_FIELDS
            if f"{config}_{field}" not in row
        ]
        if missing:
            raise C6ValidationError(
                f"{key}: joined row lacks {config} output fields {missing}"
            )
        if not isinstance(row[f"{config}_prediction"], str):
            raise C6ValidationError(
                f"{key}: {config}_prediction must be a string"
            )
        if row[f"{config}_kv_accounting"] != KV_ACCOUNTING_BY_CONFIG[config]:
            raise C6ValidationError(
                f"{key}: {config}_kv_accounting differs from the locked "
                "physical/modeled byte-accounting contract"
            )
        _integer(
            row[f"{config}_generated_token_count"],
            f"{key} {config}_generated_token_count",
            minimum=0,
        )
        score = _finite_number(row.get(f"{config}_score"), f"{key} {config}_score")
        if not 0 <= score <= 1:
            raise C6ValidationError(f"{key}: {config}_score must be in [0, 1]")
        kv_bytes = _finite_number(
            row.get(f"{config}_kv_bytes"), f"{key} {config}_kv_bytes"
        )
        fp16_bytes = _finite_number(
            row.get(f"{config}_kv_bytes_fp16"),
            f"{key} {config}_kv_bytes_fp16",
        )
        compression = _finite_number(
            row.get(f"{config}_compression"),
            f"{key} {config}_compression",
        )
        if kv_bytes <= 0 or fp16_bytes <= 0 or compression <= 0:
            raise C6ValidationError(
                f"{key}: {config} byte labels and compression must be positive"
            )
        derived = fp16_bytes / kv_bytes
        if not math.isclose(
            compression,
            derived,
            rel_tol=COMPRESSION_REL_TOLERANCE,
            abs_tol=COMPRESSION_ABS_TOLERANCE,
        ):
            raise C6ValidationError(
                f"{key}: {config}_compression does not equal "
                f"{config}_kv_bytes_fp16/{config}_kv_bytes"
            )


def validate_joined_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_task_counts: Mapping[str, int] | None = None,
) -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    """Validate and canonically order an exact joined task/source-index panel.

    The production default requires all 3,750 official rows.  Explicit counts
    are an intentionally narrow test/smoke boundary and must still cover all 16
    tasks with exact ``0..count-1`` source indexes.
    """

    counts = _expected_counts(expected_task_counts)
    copied: list[dict[str, Any]] = []
    features_by_key: dict[tuple[str, str], tuple[float, ...]] = {}
    keys: set[tuple[str, str]] = set()
    source_pairs: set[tuple[str, int]] = set()
    observed_counts: Counter[str] = Counter()
    models: set[str] = set()
    model_ids: set[str] = set()

    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise C6ValidationError(f"joined row {index} must be an object")
        row = copy.deepcopy(dict(source))
        if row.get("schema_version") != JOIN_SCHEMA:
            raise C6ValidationError(
                f"joined row {index} has unsupported schema_version"
            )
        required = {
            "schema_version",
            "task",
            "category",
            "benchmark_id",
            *REQUIRED_SHARED_FIELDS,
            *(
                f"{config}_{field}"
                for config in LABEL_CONFIGURATIONS
                for field in CONFIG_OUTPUT_FIELDS
            ),
        }
        missing = sorted(required - set(row))
        if missing:
            raise C6ValidationError(
                f"joined row {index} lacks schema fields {missing}"
            )
        task = row.get("task")
        if task not in TASK_SPECS:
            raise C6ValidationError(f"joined row {index} has unknown task {task!r}")
        benchmark_id = row.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise C6ValidationError(
                f"joined row {index} benchmark_id must be a non-empty string"
            )
        key = (task, benchmark_id)
        if key in keys:
            raise C6ValidationError(f"duplicate joined key {key}")
        keys.add(key)
        source_index = _integer(
            row.get("source_index"), f"{key} source_index", minimum=0
        )
        source_pair = (task, source_index)
        if source_pair in source_pairs:
            raise C6ValidationError(f"duplicate task/source_index {source_pair}")
        source_pairs.add(source_pair)
        if row.get("category") != TASK_SPECS[task].category:
            raise C6ValidationError(f"{key}: category differs from task registry")
        model = row.get("model")
        model_id = row.get("model_id")
        if not isinstance(model, str) or not model:
            raise C6ValidationError(f"{key}: model must be a non-empty string")
        if not isinstance(model_id, str) or not model_id:
            raise C6ValidationError(f"{key}: model_id must be a non-empty string")
        models.add(model)
        model_ids.add(model_id)
        features_by_key[key] = _feature_vector(row.get("features"), key)
        _validate_labels(row, key)
        observed_counts[task] += 1
        copied.append(row)

    expected_total = sum(counts.values())
    if len(copied) != expected_total:
        raise C6ValidationError(
            f"joined panel has {len(copied)} rows; expected {expected_total}"
        )
    if len(models) != 1 or len(model_ids) != 1:
        raise C6ValidationError("joined panel mixes model aliases or model IDs")
    for task, expected_count in counts.items():
        observed = observed_counts.get(task, 0)
        if observed != expected_count:
            raise C6ValidationError(
                f"{task} has {observed} rows; expected {expected_count}"
            )
        actual_indices = {
            source_index
            for source_task, source_index in source_pairs
            if source_task == task
        }
        expected_indices = set(range(expected_count))
        if actual_indices != expected_indices:
            raise C6ValidationError(
                f"{task} source_index values must be exactly "
                f"0..{expected_count - 1}"
            )

    task_position = {task: index for index, task in enumerate(TASK_ORDER)}
    copied.sort(key=lambda row: (task_position[row["task"]], row["source_index"]))
    matrix = np.asarray(
        [
            features_by_key[(row["task"], row["benchmark_id"])]
            for row in copied
        ],
        dtype=np.float64,
    )
    if matrix.shape != (expected_total, len(FEATURE_KEYS)):
        raise C6ValidationError("seven-feature matrix has an unexpected shape")
    if not np.isfinite(matrix).all():
        raise C6ValidationError("seven-feature matrix contains non-finite values")
    return tuple(copied), matrix


def load_joined_jsonl(path: Path | str) -> tuple[dict[str, Any], ...]:
    """Read a canonical JSONL artifact without silently skipping blank rows."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise C6ValidationError(f"joined JSONL is unreadable: {source}") from exc
    if not payload:
        raise C6ValidationError("joined JSONL is empty")
    if not payload.endswith(b"\n"):
        raise C6ValidationError("joined JSONL lacks a final newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise C6ValidationError(f"joined JSONL line {line_number} is blank")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise C6ValidationError(
                f"joined JSONL line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise C6ValidationError(
                f"joined JSONL line {line_number} must be an object"
            )
        rows.append(row)
    return tuple(rows)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise C6ProvenanceError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C6ProvenanceError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise C6ProvenanceError(f"{label} must be a JSON object")
    return value


def _run_directory(run: RunLayout | Path | str) -> Path:
    path = run.run_dir if isinstance(run, RunLayout) else Path(run)
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {resolved}")
    return resolved


def _identity_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C6ProvenanceError(f"run manifest {field} must be an object")
    return value


def _identity_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C6ProvenanceError(
            f"run manifest {field} must be a lowercase SHA-256 digest"
        )
    return value


def validate_production_run_identity(
    run: RunLayout | Path | str,
    *,
    run_config: Path | str = DEFAULT_RUN_CONFIG,
    current_source_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind C6 to one still-running run, prepared artifact, and clean commit.

    ``current_source_state`` is an explicit test seam.  Production callers must
    omit it so :func:`longbench16_run_config.current_source_code_state` checks
    the exact Git commit and dirtiness while ignoring only generated
    ``runs/longbench16_24k`` outputs.
    """

    run_dir = _run_directory(run)
    manifest_path = run_dir / "manifest.json"
    try:
        manifest_digest_before = file_sha256(manifest_path)
    except OSError as exc:
        raise C6ProvenanceError(
            f"run manifest is missing or unreadable: {manifest_path}"
        ) from exc
    manifest = _read_json_object(manifest_path, "run manifest")
    if file_sha256(manifest_path) != manifest_digest_before:
        raise C6ProvenanceError("run manifest changed while C6 validated it")
    if manifest.get("schema_version") != RUNNER_SCHEMA:
        raise C6ProvenanceError("run manifest schema_version is unsupported")
    if manifest.get("status") == "complete":
        raise C6ProvenanceError(
            "C6 cannot execute or write after the run manifest is complete"
        )
    if manifest.get("status") != "running":
        raise C6ProvenanceError("run manifest status must be 'running'")

    started_at = manifest.get("started_at")
    run_id = manifest.get("run_id")
    model_alias = manifest.get("model_alias")
    if not isinstance(started_at, str) or not started_at:
        raise C6ProvenanceError("run manifest started_at identity is missing")
    if not isinstance(run_id, str) or not run_id or run_id != run_dir.name:
        raise C6ProvenanceError(
            "run manifest run_id does not match the run directory"
        )
    if (
        not isinstance(model_alias, str)
        or not model_alias
        or model_alias != run_dir.parent.name
    ):
        raise C6ProvenanceError(
            "run manifest model_alias does not match the run directory"
        )

    config_path = Path(run_config).expanduser().resolve(strict=False)
    try:
        load_locked_policy(config_path)
    except Exception as exc:
        raise C6ProvenanceError(
            "the requested run config does not contain the locked C6 policy"
        ) from exc
    config_document = _read_json_object(config_path, "run config")
    if config_document.get("schema_version") != (
        "adaptiveserve-longbench16-run-config/v1"
    ):
        raise C6ProvenanceError("run config schema_version is unsupported")
    scope = config_document.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("benchmark_examples") != EXPECTED_TOTAL_EXAMPLES
    ):
        raise C6ProvenanceError("run config does not identify 3,750 examples")
    locked_model = _identity_mapping(config_document.get("model"), "config model")
    locked_model_alias = locked_model.get("alias")
    if not isinstance(locked_model_alias, str) or not locked_model_alias:
        raise C6ProvenanceError("run config model.alias is missing")
    config_digest = file_sha256(config_path)
    for field in (
        "model_alias",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
    ):
        expected = (
            locked_model_alias if field == "model_alias" else locked_model.get(field)
        )
        if not isinstance(expected, str) or not expected:
            raise C6ProvenanceError(f"run config {field} identity is missing")
        if manifest.get(field) != expected:
            raise C6ProvenanceError(
                f"run manifest {field} differs from the locked run config"
            )

    manifest_config = _identity_mapping(
        manifest.get("run_config"), "run_config"
    )
    config_path_value = manifest_config.get("path")
    if not isinstance(config_path_value, str) or not config_path_value:
        raise C6ProvenanceError("run manifest run_config.path is missing")
    manifest_config_path = Path(config_path_value).expanduser().resolve(
        strict=False
    )
    if manifest_config_path != config_path:
        raise C6ProvenanceError(
            "run manifest run_config.path differs from the requested config"
        )
    manifest_config_digest = _identity_digest(
        manifest_config.get("sha256"), "run_config.sha256"
    )
    if manifest_config_digest != config_digest:
        raise C6ProvenanceError(
            "run manifest run_config.sha256 differs from the requested config"
        )

    prepared_identity = _identity_mapping(
        manifest.get("prepared_inputs"), "prepared_inputs"
    )
    prepared_path_value = prepared_identity.get("path")
    if not isinstance(prepared_path_value, str) or not prepared_path_value:
        raise C6ProvenanceError("run manifest prepared_inputs.path is missing")
    prepared_path = Path(prepared_path_value).expanduser().resolve(strict=False)
    declared_prepared_manifest_hash = _identity_digest(
        prepared_identity.get("manifest_sha256"),
        "prepared_inputs.manifest_sha256",
    )
    declared_prepared_index_hash = _identity_digest(
        prepared_identity.get("index_sha256"),
        "prepared_inputs.index_sha256",
    )
    if prepared_identity.get("records") != EXPECTED_TOTAL_EXAMPLES:
        raise C6ProvenanceError(
            "run manifest prepared_inputs.records must be exactly 3,750"
        )
    try:
        prepared = load_prepared_expectations(prepared_path)
    except Exception as exc:
        raise C6ProvenanceError(
            "run manifest prepared_inputs do not resolve to a valid "
            "complete prepared artifact"
        ) from exc
    if prepared.manifest_sha256 != declared_prepared_manifest_hash:
        raise C6ProvenanceError(
            "prepared manifest hash differs from the run identity"
        )
    if prepared.index_sha256 != declared_prepared_index_hash:
        raise C6ProvenanceError(
            "prepared index hash differs from the run identity"
        )
    if prepared.record_count != EXPECTED_TOTAL_EXAMPLES:
        raise C6ProvenanceError("prepared identity is not the exact 3,750-row panel")

    prepared_manifest = _read_json_object(
        prepared.manifest_path, "prepared manifest"
    )
    prepared_run_config = _identity_mapping(
        prepared_manifest.get("run_config"), "prepared run_config"
    )
    if prepared_run_config.get("sha256") != config_digest:
        raise C6ProvenanceError(
            "prepared artifact was built from a different run config"
        )
    manifest_source = _identity_mapping(
        manifest.get("source_code"), "source_code"
    )
    prepared_source = _identity_mapping(
        prepared_manifest.get("source_code"), "prepared source_code"
    )
    source_commit = manifest_source.get("commit")
    if (
        not isinstance(source_commit, str)
        or _COMMIT_RE.fullmatch(source_commit) is None
    ):
        raise C6ProvenanceError(
            "run manifest source_code.commit is not an exact Git commit"
        )
    if manifest_source.get("dirty") is not False:
        raise C6ProvenanceError(
            "run manifest source_code must identify a clean source tree"
        )
    if dict(prepared_source) != dict(manifest_source):
        raise C6ProvenanceError(
            "run source_code identity differs from the prepared artifact"
        )
    try:
        observed_source = (
            current_source_code_state()
            if current_source_state is None
            else copy.deepcopy(dict(current_source_state))
        )
    except Exception as exc:
        raise C6ProvenanceError(
            "cannot resolve the current source-code state"
        ) from exc
    observed_commit = observed_source.get("commit")
    if (
        not isinstance(observed_commit, str)
        or _COMMIT_RE.fullmatch(observed_commit) is None
    ):
        raise C6ProvenanceError(
            "current source_code.commit is not an exact Git commit"
        )
    if observed_source.get("dirty") is not False:
        raise C6ProvenanceError(
            "production C6 requires a clean source tree"
        )
    if observed_commit != source_commit:
        raise C6ProvenanceError(
            "current source-code commit differs from the run manifest"
        )

    return {
        "run_manifest": {
            "path": str(manifest_path),
            "sha256_at_c6": manifest_digest_before,
            "schema_version": RUNNER_SCHEMA,
            "status_at_c6": "running",
        },
        "run_id": run_id,
        "started_at": started_at,
        "model_alias": model_alias,
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_id": manifest["tokenizer_id"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "run_config": {
            "path": str(config_path),
            "sha256": config_digest,
        },
        "prepared_inputs": {
            "path": str(prepared.prepared_dir),
            "manifest_sha256": prepared.manifest_sha256,
            "index_sha256": prepared.index_sha256,
            "records": prepared.record_count,
        },
        "source_code": {
            "commit": observed_commit,
            "dirty": False,
            "dirty_check_ignored_only": "runs/longbench16_24k/**",
        },
    }


def evaluate_production_c6(
    run: RunLayout | Path | str,
    *,
    run_config: Path | str = DEFAULT_RUN_CONFIG,
    current_source_state: Mapping[str, Any] | None = None,
) -> C6Result:
    """Evaluate only the canonical joined artifact of a validated running run."""

    run_dir = _run_directory(run)
    identity = validate_production_run_identity(
        run_dir,
        run_config=run_config,
        current_source_state=current_source_state,
    )
    joined_path = (run_dir / "analysis" / "joined.jsonl").resolve(strict=False)
    result = evaluate_c6_file(joined_path, run_config=run_config)
    if result.summary["evaluation_mode"] != "primary":
        raise C6ProvenanceError("production C6 did not execute the primary protocol")
    if result.summary["model"] != identity["model_alias"]:
        raise C6ProvenanceError(
            "joined artifact model differs from the run manifest"
        )
    if result.summary["model_id"] != identity["model_id"]:
        raise C6ProvenanceError(
            "joined artifact model_id differs from the run manifest"
        )
    try:
        prepared = load_prepared_expectations(
            identity["prepared_inputs"]["path"]
        )
    except Exception as exc:
        raise C6ProvenanceError(
            "prepared key authority became invalid during C6 evaluation"
        ) from exc
    prepared_panel = tuple(
        (
            record["task"],
            record["benchmark_id"],
            record["source_index"],
        )
        for record in prepared.records
    )
    joined_panel = tuple(
        (row["task"], row["benchmark_id"], row["source_index"])
        for row in result.rows
    )
    if joined_panel != prepared_panel:
        raise C6ProvenanceError(
            "joined task/benchmark_id/source_index panel differs from the "
            "prepared key authority"
        )
    source_artifact = result.summary.get("source_artifact")
    if (
        not isinstance(source_artifact, Mapping)
        or Path(source_artifact.get("path", "")).resolve(strict=False)
        != joined_path
    ):
        raise C6ProvenanceError(
            "production C6 source is not the canonical run analysis/joined.jsonl"
        )

    # Revalidate after model fitting so a concurrently completed or altered run
    # cannot be published under the identity checked before evaluation.
    identity_after = validate_production_run_identity(
        run_dir,
        run_config=run_config,
        current_source_state=current_source_state,
    )
    if identity_after != identity:
        raise C6ProvenanceError("run identity changed while C6 was evaluating")
    summary = copy.deepcopy(result.summary)
    summary["run_identity"] = identity
    return C6Result(rows=result.rows, summary=summary)


def _materialize_folds(
    matrix: np.ndarray,
    tasks: Sequence[str],
    policy: C6Policy,
    folds: Iterable[Fold] | None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    n_rows = len(tasks)
    if folds is None:
        splitter = StratifiedKFold(
            n_splits=policy.folds,
            shuffle=policy.shuffle,
            random_state=policy.seed,
        )
        raw_folds: Iterable[Fold] = splitter.split(matrix, np.asarray(tasks))
    else:
        raw_folds = folds

    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    test_coverage = np.zeros(n_rows, dtype=np.int64)
    all_indices = set(range(n_rows))
    per_task_test_counts: dict[str, list[int]] = {
        task: [] for task in TASK_ORDER
    }
    for fold_index, pair in enumerate(raw_folds):
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise C6ValidationError(
                f"fold {fold_index} must be a (train_indices, test_indices) pair"
            )
        train = np.asarray(pair[0])
        test = np.asarray(pair[1])
        if train.ndim != 1 or test.ndim != 1:
            raise C6ValidationError(
                f"fold {fold_index} indices must be one-dimensional"
            )
        if train.dtype.kind not in "iu" or test.dtype.kind not in "iu":
            raise C6ValidationError(f"fold {fold_index} indices must be integers")
        train = train.astype(np.int64, copy=False)
        test = test.astype(np.int64, copy=False)
        if train.size == 0 or test.size == 0:
            raise C6ValidationError(
                f"fold {fold_index} train/test sets must be nonempty"
            )
        if (
            np.any(train < 0)
            or np.any(test < 0)
            or np.any(train >= n_rows)
            or np.any(test >= n_rows)
        ):
            raise C6ValidationError(f"fold {fold_index} contains an out-of-range index")
        train_set = set(train.tolist())
        test_set = set(test.tolist())
        if len(train_set) != train.size or len(test_set) != test.size:
            raise C6ValidationError(f"fold {fold_index} contains duplicate indices")
        if train_set & test_set:
            raise C6ValidationError(f"fold {fold_index} leaks test rows into training")
        if train_set != all_indices - test_set:
            raise C6ValidationError(
                f"fold {fold_index} training indices are not the test complement"
            )
        test_coverage[test] += 1
        for task in TASK_ORDER:
            per_task_test_counts[task].append(
                sum(tasks[index] == task for index in test)
            )
        normalized.append((train, test))

    if not normalized:
        raise C6ValidationError("at least one fold is required")
    if not np.all(test_coverage == 1):
        bad = np.flatnonzero(test_coverage != 1).tolist()
        raise C6ValidationError(
            "every example must be held out exactly once; "
            f"violating_indices={bad[:20]}"
        )
    for task, fold_counts in per_task_test_counts.items():
        if max(fold_counts) - min(fold_counts) > 1:
            raise C6ValidationError(
                f"injected folds are not task-stratified for {task}"
            )
    return tuple(normalized)


def _default_scaler_factory(_config: str, _fold_index: int) -> StandardScaler:
    return StandardScaler()


def _default_regressor_factory(
    _config: str, _fold_index: int
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=0,
    )


def _harmonic_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise C6ValidationError("compression values must be nonempty and positive")
    return len(values) / math.fsum(1.0 / value for value in values)


def _fit_metric(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float | None]:
    mae = float(mean_absolute_error(actual, predicted))
    score = float(r2_score(actual, predicted))
    return mae, score if math.isfinite(score) else None


def evaluate_c6(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_config: Path | str = DEFAULT_RUN_CONFIG,
    expected_task_counts: Mapping[str, int] | None = None,
    folds: Iterable[Fold] | None = None,
    scaler_factory: ScalerFactory | None = None,
    regressor_factory: RegressorFactory | None = None,
    source_artifact: Mapping[str, Any] | None = None,
) -> C6Result:
    """Run the held-out C6 protocol and return per-row and aggregate evidence.

    ``expected_task_counts`` and ``folds`` exist for CPU tests and smoke runs.
    Omitting both executes the locked 3,750-row, 10-fold primary protocol.
    Factories receive ``(configuration, fold_index)`` and are likewise intended
    only for instrumented tests; production uses one fresh StandardScaler and
    one exactly configured HistGradientBoostingRegressor per config and fold.
    """

    policy = load_locked_policy(run_config)
    expected_counts = _expected_counts(expected_task_counts)
    panel, matrix = validate_joined_panel(
        rows, expected_task_counts=expected_counts
    )
    tasks = tuple(row["task"] for row in panel)
    materialized_folds = _materialize_folds(matrix, tasks, policy, folds)
    make_scaler = scaler_factory or _default_scaler_factory
    make_regressor = regressor_factory or _default_regressor_factory

    output_by_index: list[dict[str, Any] | None] = [None] * len(panel)
    fit_metrics: dict[str, list[dict[str, Any]]] = {
        config: [] for config in LABEL_CONFIGURATIONS
    }
    fold_diagnostics: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(materialized_folds):
        train_matrix = matrix[train_indices]
        test_matrix = matrix[test_indices]
        predictions: dict[str, np.ndarray] = {}
        for config in LABEL_CONFIGURATIONS:
            train_labels = np.asarray(
                [panel[index][f"{config}_score"] for index in train_indices],
                dtype=np.float64,
            )
            test_labels = np.asarray(
                [panel[index][f"{config}_score"] for index in test_indices],
                dtype=np.float64,
            )
            scaler = make_scaler(config, fold_index)
            regressor = make_regressor(config, fold_index)
            scaled_train = np.asarray(scaler.fit_transform(train_matrix))
            scaled_test = np.asarray(scaler.transform(test_matrix))
            regressor.fit(scaled_train, train_labels)
            predicted = np.asarray(regressor.predict(scaled_test), dtype=np.float64)
            if predicted.shape != (len(test_indices),):
                raise C6ValidationError(
                    f"{config} fold {fold_index} returned predictions with "
                    f"shape {predicted.shape}; expected {(len(test_indices),)}"
                )
            if not np.isfinite(predicted).all():
                raise C6ValidationError(
                    f"{config} fold {fold_index} returned non-finite predictions"
                )
            predictions[config] = predicted
            mae, r2 = _fit_metric(test_labels, predicted)
            fit_metrics[config].append(
                {
                    "fold": fold_index,
                    "train_rows": int(len(train_indices)),
                    "test_rows": int(len(test_indices)),
                    "mae": mae,
                    "r2": r2,
                }
            )

        train_c0_mean = math.fsum(
            float(panel[index]["C0_score"]) for index in train_indices
        ) / len(train_indices)
        train_compression = {
            config: _harmonic_mean(
                [
                    float(panel[index][f"{config}_compression"])
                    for index in train_indices
                ]
            )
            for config in CANDIDATE_CONFIGURATIONS
        }
        compression_order = sorted(
            CANDIDATE_CONFIGURATIONS,
            key=lambda config: (
                -train_compression[config],
                CANDIDATE_CONFIGURATIONS.index(config),
            ),
        )
        fold_diagnostics.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "train_mean_actual_c0": train_c0_mean,
                "train_harmonic_mean_compression": train_compression,
                "candidate_compression_order": compression_order,
                "test_task_counts": dict(
                    Counter(tasks[index] for index in test_indices)
                ),
            }
        )

        for local_index, panel_index in enumerate(test_indices):
            predicted_scores = {
                config: float(predictions[config][local_index])
                for config in LABEL_CONFIGURATIONS
            }
            quality_anchor = max(
                predicted_scores[REFERENCE_CONFIGURATION], train_c0_mean
            )
            quality_threshold = policy.tau * quality_anchor
            eligible = [
                config
                for config in CANDIDATE_CONFIGURATIONS
                if predicted_scores[config] >= quality_threshold
            ]
            if eligible:
                eligible_set = set(eligible)
                ranked_eligible = [
                    config
                    for config in compression_order
                    if config in eligible_set
                ]
                chosen = ranked_eligible[0]
                reason = "eligible_train_compression"
            else:
                ranked_eligible = []
                chosen = max(
                    CANDIDATE_CONFIGURATIONS,
                    key=lambda config: (
                        predicted_scores[config],
                        -CANDIDATE_CONFIGURATIONS.index(config),
                    ),
                )
                reason = "highest_predicted_quality_fallback"

            source = panel[panel_index]
            actual_score = float(source[f"{chosen}_score"])
            actual_c0 = float(source["C0_score"])
            kv_bytes = float(source[f"{chosen}_kv_bytes"])
            fp16_bytes = float(source[f"{chosen}_kv_bytes_fp16"])
            compression = fp16_bytes / kv_bytes
            output_by_index[panel_index] = {
                "schema_version": C6_ROW_SCHEMA,
                "configuration": "C6",
                "model": source["model"],
                "model_id": source["model_id"],
                "task": source["task"],
                "category": source["category"],
                "benchmark_id": source["benchmark_id"],
                "source_index": source["source_index"],
                "fold": fold_index,
                "chosen_configuration": chosen,
                "selection_reason": reason,
                "predicted_scores": predicted_scores,
                "predicted_quality_anchor": quality_anchor,
                "predicted_quality_threshold": quality_threshold,
                "eligible_candidates": eligible,
                "ranked_eligible_candidates": ranked_eligible,
                "train_mean_actual_c0": train_c0_mean,
                "train_harmonic_mean_compression": train_compression,
                "score": actual_score,
                "kv_bytes": kv_bytes,
                "kv_bytes_fp16": fp16_bytes,
                "compression": compression,
                "kv_accounting": copy.deepcopy(
                    source[f"{chosen}_kv_accounting"]
                ),
                "actual_c0_score": actual_c0,
                "actual_iso_quality_threshold": policy.tau * actual_c0,
                "actual_iso_quality_violation": (
                    actual_score < policy.tau * actual_c0
                ),
                "actual_below_c0": actual_score < actual_c0,
            }

    if any(row is None for row in output_by_index):
        raise C6ValidationError("internal error: at least one row was not held out")
    output_rows = tuple(
        row for row in output_by_index if row is not None
    )
    quality = aggregate_quality(
        (
            {"task": row["task"], "score": row["score"]}
            for row in output_rows
        ),
        require_complete=expected_task_counts is None,
    )
    compression = aggregate_compression(
        (
            {
                "task": row["task"],
                "kv_bytes": row["kv_bytes"],
                "kv_bytes_fp16": row["kv_bytes_fp16"],
            }
            for row in output_rows
        ),
        require_complete=expected_task_counts is None,
    )
    selection_counter = Counter(
        row["chosen_configuration"] for row in output_rows
    )
    selection_by_task = {
        task: {
            config: sum(
                row["task"] == task
                and row["chosen_configuration"] == config
                for row in output_rows
            )
            for config in CANDIDATE_CONFIGURATIONS
        }
        for task in TASK_ORDER
    }
    violations = sum(
        bool(row["actual_iso_quality_violation"]) for row in output_rows
    )
    below_c0 = sum(bool(row["actual_below_c0"]) for row in output_rows)
    quality_deltas = [
        float(row["score"]) - float(row["actual_c0_score"])
        for row in output_rows
    ]
    fit_summary = {
        config: {
            "folds": metrics,
            "mean_mae": math.fsum(item["mae"] for item in metrics) / len(metrics),
            "mean_r2": (
                math.fsum(
                    item["r2"] for item in metrics if item["r2"] is not None
                )
                / sum(item["r2"] is not None for item in metrics)
                if any(item["r2"] is not None for item in metrics)
                else None
            ),
        }
        for config, metrics in fit_metrics.items()
    }
    primary_mode = (
        expected_task_counts is None
        and folds is None
        and scaler_factory is None
        and regressor_factory is None
    )
    summary: dict[str, Any] = {
        "schema_version": C6_ANALYSIS_SCHEMA,
        "status": "complete",
        "configuration": "C6",
        "evaluation_mode": "primary" if primary_mode else "test_or_smoke",
        "source_schema_version": JOIN_SCHEMA,
        "row_count": len(output_rows),
        "key_fields": list(DEFAULT_KEY_FIELDS),
        "model": output_rows[0]["model"],
        "model_id": output_rows[0]["model_id"],
        "task_counts": dict(expected_counts),
        "feature_contract": {
            "keys": list(FEATURE_KEYS),
            "question_position_null_encoding": QUESTION_POSITION_SENTINEL,
            "scaling": "fresh StandardScaler fitted on each config/fold training set",
        },
        "routing_policy": policy.as_dict(),
        "cross_validation": {
            "kind": _EXPECTED_CROSS_VALIDATION["kind"],
            "fold_count": len(materialized_folds),
            "shuffle": policy.shuffle if folds is None else None,
            "seed": policy.seed if folds is None else None,
            "folds_injected": folds is not None,
            "each_example_held_out_exactly_once": True,
            "fold_diagnostics": fold_diagnostics,
        },
        "quality": quality,
        "compression": compression,
        "kv_accounting_by_configuration": copy.deepcopy(
            KV_ACCOUNTING_BY_CONFIG
        ),
        "violation_rates_vs_actual_c0": {
            "threshold_tau": policy.tau,
            "below_tau_times_actual_c0_count": violations,
            "below_tau_times_actual_c0_rate": violations / len(output_rows),
            "below_actual_c0_count": below_c0,
            "below_actual_c0_rate": below_c0 / len(output_rows),
            "mean_actual_quality_delta_vs_c0": (
                math.fsum(quality_deltas) / len(quality_deltas)
            ),
        },
        "selection_counts": {
            config: selection_counter.get(config, 0)
            for config in CANDIDATE_CONFIGURATIONS
        },
        "selection_counts_by_task": selection_by_task,
        "fit_metrics": fit_summary,
    }
    if source_artifact is not None:
        summary["source_artifact"] = copy.deepcopy(dict(source_artifact))
    return C6Result(rows=output_rows, summary=summary)


def evaluate_c6_file(
    joined_path: Path | str,
    *,
    run_config: Path | str = DEFAULT_RUN_CONFIG,
    expected_task_counts: Mapping[str, int] | None = None,
    folds: Iterable[Fold] | None = None,
    scaler_factory: ScalerFactory | None = None,
    regressor_factory: RegressorFactory | None = None,
) -> C6Result:
    """Load one joined JSONL file and retain its digest in the C6 summary."""

    path = Path(joined_path).expanduser().resolve(strict=False)
    rows = load_joined_jsonl(path)
    return evaluate_c6(
        rows,
        run_config=run_config,
        expected_task_counts=expected_task_counts,
        folds=folds,
        scaler_factory=scaler_factory,
        regressor_factory=regressor_factory,
        source_artifact={
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows": len(rows),
        },
    )


def _analysis_directory(run: RunLayout | Path | str) -> Path:
    if isinstance(run, RunLayout):
        run_dir = run.run_dir
        analysis_dir = run.analysis_dir
    else:
        run_dir = Path(run).expanduser().resolve(strict=False)
        analysis_dir = run_dir / "analysis"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not analysis_dir.is_dir():
        raise FileNotFoundError(
            f"run analysis directory does not exist: {analysis_dir}"
        )
    safe_analysis = assert_safe_write_path(analysis_dir)
    try:
        safe_analysis.relative_to(run_dir.resolve(strict=False))
    except ValueError as exc:
        raise C6ValidationError("analysis directory resolves outside the run") from exc
    return safe_analysis


def _validate_write_time_identity(result: C6Result, run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json_object(manifest_path, "run manifest")
        if manifest.get("status") == "complete":
            raise C6ProvenanceError(
                "C6 cannot execute or write after the run manifest is complete"
            )
        if manifest.get("status") != "running":
            raise C6ProvenanceError(
                "run manifest status must be 'running' before C6 writes"
            )

    identity = result.summary.get("run_identity")
    if result.summary.get("evaluation_mode") == "primary" and not isinstance(
        identity, Mapping
    ):
        raise C6ProvenanceError(
            "primary C6 outputs require a validated run identity"
        )
    if not isinstance(identity, Mapping):
        return
    config_identity = _identity_mapping(
        identity.get("run_config"), "C6 run_identity.run_config"
    )
    config_path = config_identity.get("path")
    if not isinstance(config_path, str) or not config_path:
        raise C6ProvenanceError("C6 run identity lacks its run-config path")
    current_identity = validate_production_run_identity(
        run_dir, run_config=config_path
    )
    if current_identity != dict(identity):
        raise C6ProvenanceError("run identity changed before C6 publication")

    source = result.summary.get("source_artifact")
    if not isinstance(source, Mapping):
        raise C6ProvenanceError("primary C6 summary lacks joined-source identity")
    source_path_value = source.get("path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise C6ProvenanceError("primary C6 joined-source path is missing")
    source_path = Path(source_path_value).expanduser().resolve(strict=False)
    canonical_source = (run_dir / "analysis" / "joined.jsonl").resolve(
        strict=False
    )
    if source_path != canonical_source:
        raise C6ProvenanceError(
            "primary C6 was not evaluated from the canonical joined artifact"
        )
    if file_sha256(source_path) != source.get("sha256"):
        raise C6ProvenanceError(
            "joined artifact changed between C6 evaluation and publication"
        )


def write_c6_outputs(
    result: C6Result,
    run: RunLayout | Path | str,
) -> dict[str, Any]:
    """Publish immutable C6 JSONL/JSON artifacts under ``<run>/analysis``.

    Both destinations are preflighted and the JSONL is removed if publishing
    the summary fails, so callers never receive a half-published pair.
    """

    if not isinstance(result, C6Result):
        raise TypeError("result must be a C6Result")
    run_dir = _run_directory(run)
    _validate_write_time_identity(result, run_dir)
    analysis_dir = _analysis_directory(run)
    rows_path = analysis_dir / DEFAULT_ROWS_FILENAME
    summary_path = analysis_dir / DEFAULT_SUMMARY_FILENAME
    if rows_path.exists():
        raise FileExistsError(rows_path)
    if summary_path.exists():
        raise FileExistsError(summary_path)
    rows_metadata = atomic_write_jsonl(rows_path, result.rows)
    payload = copy.deepcopy(result.summary)
    payload["artifacts"] = {
        "per_prompt_jsonl": rows_metadata,
        "summary_json": {"path": str(summary_path)},
    }
    try:
        atomic_write_json(summary_path, payload, overwrite=False)
    except BaseException:
        rows_path.unlink(missing_ok=True)
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the locked leakage-safe C6 router on a complete "
            "LongBench16 joined artifact."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing immutable run directory; outputs go under analysis/.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
        help="Locked LongBench16 run config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_production_c6(
        args.run_dir,
        run_config=args.run_config,
    )
    published = write_c6_outputs(result, args.run_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": result.summary["row_count"],
                "quality": result.summary["quality"]["category_balanced_mean"],
                "compression": result.summary["compression"]["harmonic_mean"],
                "artifacts": published["artifacts"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C6_ANALYSIS_SCHEMA",
    "C6Policy",
    "C6PolicyError",
    "C6ProvenanceError",
    "C6Result",
    "C6ValidationError",
    "C6_ROW_SCHEMA",
    "CANDIDATE_CONFIGURATIONS",
    "DEFAULT_RUN_CONFIG",
    "LABEL_CONFIGURATIONS",
    "QUESTION_POSITION_SENTINEL",
    "REFERENCE_CONFIGURATION",
    "evaluate_c6",
    "evaluate_c6_file",
    "evaluate_production_c6",
    "load_joined_jsonl",
    "load_locked_policy",
    "main",
    "validate_joined_panel",
    "validate_production_run_identity",
    "write_c6_outputs",
]
