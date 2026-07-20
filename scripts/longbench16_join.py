#!/usr/bin/env python3
"""Strict C0--C5 LongBench16 join, aggregation, and oracle analysis.

The module is CPU-only.  Its primary post-hoc oracle is the explicitly locked
tau=0.99 iso-quality policy in ``configs/longbench16_24k.json``; the required
quality-first oracle is reported alongside it as a diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .longbench16_io import (
        DEFAULT_KEY_FIELDS,
        FINAL_INPUT_HASH_FIELD,
        FIXED_CONFIGS,
        KV_ACCOUNTING_BY_CONFIG,
        ConfigValidation,
        CrossConfigValidation,
        assert_safe_write_path,
        atomic_write_json,
        file_sha256,
        validate_config_records,
        validate_cross_config_hashes,
    )
    from .longbench16_protocol import (
        EXPECTED_TOTAL_EXAMPLES,
        FINAL_INPUT_TOKEN_CAP,
        LONG_BENCH_DATASET_ID,
        LONG_BENCH_DATASET_REVISION,
        LONG_BENCH_RESOLVED_DATASET_ID,
        LONG_BENCH_SPLIT,
        MIDDLE_TRUNCATION_TOKENS_PER_SIDE,
        TASK_ORDER,
        TASK_SPECS,
        aggregate_compression,
        aggregate_quality,
        config_file_hashes,
        protocol_config_hash,
        score_prediction,
        source_manifest,
    )
except ImportError:  # Direct execution: python scripts/longbench16_join.py
    from longbench16_io import (
        DEFAULT_KEY_FIELDS,
        FINAL_INPUT_HASH_FIELD,
        FIXED_CONFIGS,
        KV_ACCOUNTING_BY_CONFIG,
        ConfigValidation,
        CrossConfigValidation,
        assert_safe_write_path,
        atomic_write_json,
        file_sha256,
        validate_config_records,
        validate_cross_config_hashes,
    )
    from longbench16_protocol import (
        EXPECTED_TOTAL_EXAMPLES,
        FINAL_INPUT_TOKEN_CAP,
        LONG_BENCH_DATASET_ID,
        LONG_BENCH_DATASET_REVISION,
        LONG_BENCH_RESOLVED_DATASET_ID,
        LONG_BENCH_SPLIT,
        MIDDLE_TRUNCATION_TOKENS_PER_SIDE,
        TASK_ORDER,
        TASK_SPECS,
        aggregate_compression,
        aggregate_quality,
        config_file_hashes,
        protocol_config_hash,
        score_prediction,
        source_manifest,
    )


JOIN_SCHEMA = "adaptiveserve-longbench16-joined/v1"
ANALYSIS_SCHEMA = "adaptiveserve-longbench16-analysis/v1"
PREPARED_SCHEMA = "adaptiveserve-longbench16-prepared/v1"
PREPARED_INPUT_HASH_FIELD = "final_input_token_sha256"
TOKEN_HASH_ALGORITHM = "sha256-little-endian-uint32"
CHAT_TEMPLATE_DATE_STRING = "20 Jul 2026"
ORACLE_CANDIDATES = ("C1", "C2", "C3", "C4", "C5")
PRIMARY_ORACLE_OBJECTIVE = "iso_quality_tau_0.99_max_compression"
QUALITY_ORACLE_DIAGNOSTIC = "max_quality_then_compression_then_config"
ISO_QUALITY_TAU = 0.99
ORACLE_OBJECTIVES = (
    PRIMARY_ORACLE_OBJECTIVE,
    QUALITY_ORACLE_DIAGNOSTIC,
)
FEATURE_KEYS = (
    "seq_len_tokens",
    "seq_len_chars",
    "token_entropy",
    "gzip_ratio",
    "unique_token_ratio",
    "question_position",
    "newline_density",
)
CONFIG_OUTPUT_FIELDS = (
    "score",
    "prediction",
    "kv_bytes",
    "kv_bytes_fp16",
    "compression",
    "generated_token_count",
    "kv_accounting",
)
REQUIRED_SHARED_FIELDS = (
    "model",
    "model_id",
    "source_index",
    "metric",
    "references",
    "all_classes",
    "features",
    "pre_truncation_token_count",
    "post_truncation_token_count",
    "truncated",
    FINAL_INPUT_HASH_FIELD,
    "token_hash_algorithm",
    "max_new_tokens",
)
OPTIONAL_SHARED_FIELDS = (
    "category",
    "minimum_new_tokens",
    "stop_on_newline_token",
    "effective_stop_token_ids",
    "token_file",
    "token_offset_index",
)
COMPRESSION_REL_TOLERANCE = 1e-6
COMPRESSION_ABS_TOLERANCE = 5e-5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ORACLE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "longbench16_24k.json"
)


class JoinValidationError(ValueError):
    """Raised when journals cannot form one trustworthy routing table."""


class OraclePolicyError(ValueError):
    """Raised when an oracle objective is absent or unsupported."""


@dataclass(frozen=True, slots=True)
class PreparedExpectations:
    """Validated authority for prompt IDs, metadata, and token hashes."""

    prepared_dir: Path
    manifest_path: Path
    status_path: Path
    index_path: Path
    manifest_sha256: str
    status_sha256: str
    index_sha256: str
    token_file_hashes: tuple[tuple[Path, str], ...]
    records: tuple[dict[str, Any], ...]
    ordered_keys: tuple[tuple[str, str], ...]
    records_by_key: Mapping[tuple[str, str], dict[str, Any]]

    @property
    def record_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class JoinResult:
    """Validated join rows plus their input integrity evidence."""

    rows: tuple[dict[str, Any], ...]
    validations: Mapping[str, ConfigValidation]
    cross_validation: CrossConfigValidation
    prepared: PreparedExpectations | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JoinValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise JoinValidationError(f"{field} must be finite")
    return number


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JoinValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise JoinValidationError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise JoinValidationError(f"{field} must be <= {maximum}")
    return value


def _canonical_json(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise JoinValidationError(f"{field} is not finite JSON data") from exc


def _validate_references(value: Any, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise JoinValidationError(f"{field} must be a non-empty list")
    for index, reference in enumerate(value):
        if isinstance(reference, bool) or not isinstance(
            reference, (str, int, float)
        ):
            raise JoinValidationError(
                f"{field}[{index}] must be a string or finite number"
            )
        if isinstance(reference, float) and not math.isfinite(reference):
            raise JoinValidationError(f"{field}[{index}] must be finite")


def _validate_all_classes(value: Any, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise JoinValidationError(
            f"{field} must be null or a list of non-empty strings"
        )


def _require_equal(
    records: Mapping[str, Mapping[str, Any]],
    field: str,
    key: tuple[str, str],
    *,
    required: bool,
) -> Any:
    present = {config: field in records[config] for config in FIXED_CONFIGS}
    if required and not all(present.values()):
        missing = [config for config, is_present in present.items() if not is_present]
        raise JoinValidationError(
            f"{key}: required shared field {field!r} is missing from {missing}"
        )
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise JoinValidationError(
            f"{key}: shared field {field!r} has inconsistent presence: {present}"
        )
    baseline = _canonical_json(records["C0"][field], f"{key} C0 {field}")
    for config in FIXED_CONFIGS[1:]:
        observed = _canonical_json(
            records[config][field], f"{key} {config} {field}"
        )
        if observed != baseline:
            raise JoinValidationError(
                f"{key}: shared field {field!r} differs between C0 and {config}"
            )
    return copy.deepcopy(records["C0"][field])


def _normalize_expected_keys(
    expected_keys: Iterable[Sequence[Any] | Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for index, item in enumerate(expected_keys):
        if isinstance(item, Mapping):
            try:
                task = item["task"]
                benchmark_id = item["benchmark_id"]
            except KeyError as exc:
                raise JoinValidationError(
                    f"expected key {index} lacks task or benchmark_id"
                ) from exc
        else:
            if isinstance(item, (str, bytes)):
                raise JoinValidationError(
                    f"expected key {index} must be a two-value sequence"
                )
            try:
                values = tuple(item)
            except TypeError as exc:
                raise JoinValidationError(
                    f"expected key {index} must be a two-value sequence"
                ) from exc
            if len(values) != 2:
                raise JoinValidationError(
                    f"expected key {index} must contain two values"
                )
            task, benchmark_id = values
        if task not in TASK_SPECS:
            raise JoinValidationError(
                f"expected key {index} has unknown task {task!r}"
            )
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise JoinValidationError(
                f"expected key {index} has invalid benchmark_id"
            )
        normalized.append((task, benchmark_id))
    if len(normalized) != len(set(normalized)):
        raise JoinValidationError("expected_keys contains duplicate composite IDs")
    return tuple(normalized)


def _validate_expected_panel(
    expected_keys: Sequence[tuple[str, str]], *, require_complete: bool
) -> None:
    if not expected_keys:
        raise JoinValidationError("expected key set is empty")
    counts = Counter(task for task, _ in expected_keys)
    if not require_complete:
        return
    if len(expected_keys) != EXPECTED_TOTAL_EXAMPLES:
        raise JoinValidationError(
            f"full join requires exactly {EXPECTED_TOTAL_EXAMPLES} keys; "
            f"received {len(expected_keys)}"
        )
    for task, spec in TASK_SPECS.items():
        observed = counts.get(task, 0)
        if observed != spec.expected_test_examples:
            raise JoinValidationError(
                f"{task} has {observed} expected keys; "
                f"requires {spec.expected_test_examples}"
            )


def _validate_features(features: Any, key: tuple[str, str]) -> None:
    if not isinstance(features, Mapping):
        raise JoinValidationError(f"{key}: features must be an object")
    if set(features) != set(FEATURE_KEYS):
        missing = sorted(set(FEATURE_KEYS) - set(features))
        extra = sorted(set(features) - set(FEATURE_KEYS))
        raise JoinValidationError(
            f"{key}: feature keys differ from the locked seven-feature contract; "
            f"missing={missing}, extra={extra}"
        )
    _integer(features["seq_len_tokens"], f"{key} features.seq_len_tokens", minimum=0)
    _integer(features["seq_len_chars"], f"{key} features.seq_len_chars", minimum=0)
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
        raise JoinValidationError(
            f"{key}: entropy must be nonnegative and gzip_ratio positive"
        )
    if not 0 <= unique_ratio <= 1 or not 0 <= newline_density <= 1:
        raise JoinValidationError(
            f"{key}: unique_token_ratio/newline_density must be in [0, 1]"
        )
    question_position = features["question_position"]
    if question_position is not None:
        question_position = _finite_number(
            question_position, f"{key} features.question_position"
        )
        if not 0 <= question_position <= 1:
            raise JoinValidationError(
                f"{key}: features.question_position must be null or in [0, 1]"
            )


def _validate_shared_record(
    record: Mapping[str, Any],
    config: str,
    key: tuple[str, str],
) -> None:
    task, benchmark_id = key
    if record.get("task") != task or record.get("benchmark_id") != benchmark_id:
        raise JoinValidationError(f"{key}: {config} composite ID changed during join")
    model = record.get("model")
    if not isinstance(model, str) or not model:
        raise JoinValidationError(f"{key}: {config} model must be a non-empty string")
    model_id = record.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise JoinValidationError(
            f"{key}: {config} model_id must be a non-empty string"
        )
    _integer(record.get("source_index"), f"{key} {config} source_index", minimum=0)
    spec = TASK_SPECS[task]
    if record.get("metric") != spec.scorer:
        raise JoinValidationError(
            f"{key}: {config} metric {record.get('metric')!r} "
            f"does not match {spec.scorer!r}"
        )
    references = record.get("references")
    _validate_references(references, f"{key} {config} references")
    _canonical_json(references, f"{key} {config} references")
    all_classes = record.get("all_classes")
    _validate_all_classes(all_classes, f"{key} {config} all_classes")
    if spec.scorer == "classification_score":
        if not isinstance(all_classes, list) or not all_classes:
            raise JoinValidationError(
                f"{key}: {config} classification task requires all_classes"
            )
    elif all_classes is not None:
        raise JoinValidationError(
            f"{key}: {config} non-classification task has unexpected all_classes"
        )
    _validate_features(record.get("features"), key)

    pre_count = _integer(
        record.get("pre_truncation_token_count"),
        f"{key} {config} pre_truncation_token_count",
        minimum=1,
    )
    post_count = _integer(
        record.get("post_truncation_token_count"),
        f"{key} {config} post_truncation_token_count",
        minimum=1,
        maximum=FINAL_INPUT_TOKEN_CAP,
    )
    truncated = record.get("truncated")
    if not isinstance(truncated, bool):
        raise JoinValidationError(f"{key}: {config} truncated must be boolean")
    if pre_count > FINAL_INPUT_TOKEN_CAP:
        if not truncated or post_count != FINAL_INPUT_TOKEN_CAP:
            raise JoinValidationError(
                f"{key}: {config} overlength input violates exact 24K truncation"
            )
    elif truncated or post_count != pre_count:
        raise JoinValidationError(
            f"{key}: {config} non-overlength token audit fields are inconsistent"
        )
    max_new_tokens = _integer(
        record.get("max_new_tokens"),
        f"{key} {config} max_new_tokens",
        minimum=1,
    )
    if max_new_tokens != spec.max_new_tokens:
        raise JoinValidationError(
            f"{key}: {config} max_new_tokens={max_new_tokens}; "
            f"expected {spec.max_new_tokens}"
        )
    if "category" in record and record["category"] != spec.category:
        raise JoinValidationError(
            f"{key}: {config} category does not match the task registry"
        )
    if record.get("token_hash_algorithm") != TOKEN_HASH_ALGORITHM:
        raise JoinValidationError(
            f"{key}: {config} token_hash_algorithm must be "
            f"{TOKEN_HASH_ALGORITHM!r}"
        )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JoinValidationError(f"prepared {label} is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinValidationError(
            f"prepared {label} is not readable JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise JoinValidationError(f"prepared {label} must be a JSON object")
    _canonical_json(value, f"prepared {label}")
    return value


def _read_prepared_index(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise JoinValidationError(f"prepared index is missing: {path}") from exc
    except OSError as exc:
        raise JoinValidationError(f"prepared index is unreadable: {path}") from exc
    if not data:
        raise JoinValidationError("prepared index is empty")
    if not data.endswith(b"\n"):
        raise JoinValidationError("prepared index lacks a final newline")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), 1):
        if not raw_line:
            raise JoinValidationError(
                f"prepared index line {line_number} is blank"
            )
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JoinValidationError(
                f"prepared index line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise JoinValidationError(
                f"prepared index line {line_number} must be an object"
            )
        _canonical_json(record, f"prepared index line {line_number}")
        records.append(record)
    return records


def _validate_prepared_record(
    record: Mapping[str, Any], index: int
) -> tuple[str, str]:
    required = {
        "schema_version",
        "model",
        "model_id",
        "task",
        "category",
        "benchmark_id",
        "source_index",
        "metric",
        "references",
        "all_classes",
        "features",
        "pre_truncation_token_count",
        "post_truncation_token_count",
        "truncated",
        PREPARED_INPUT_HASH_FIELD,
        "token_hash_algorithm",
        "max_new_tokens",
        "minimum_new_tokens",
        "stop_on_newline_token",
        "token_file",
        "token_offset_index",
    }
    missing = required - set(record)
    if missing:
        raise JoinValidationError(
            f"prepared index row {index} lacks {sorted(missing)}"
        )
    if record["schema_version"] != PREPARED_SCHEMA:
        raise JoinValidationError(
            f"prepared index row {index} has unsupported schema"
        )
    task = record["task"]
    benchmark_id = record["benchmark_id"]
    if task not in TASK_SPECS:
        raise JoinValidationError(
            f"prepared index row {index} has unknown task {task!r}"
        )
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise JoinValidationError(
            f"prepared index row {index} has invalid benchmark_id"
        )
    key = (task, benchmark_id)
    _validate_shared_record(record, "prepared", key)
    spec = TASK_SPECS[task]
    source_index = record["source_index"]
    if source_index >= spec.expected_test_examples:
        raise JoinValidationError(
            f"{key}: prepared source_index exceeds the official task size"
        )
    if record["minimum_new_tokens"] != spec.minimum_new_tokens:
        raise JoinValidationError(
            f"{key}: prepared minimum_new_tokens differs from the registry"
        )
    if record["stop_on_newline_token"] is not spec.stop_on_newline_token:
        raise JoinValidationError(
            f"{key}: prepared stop_on_newline_token differs from the registry"
        )
    token_file = record["token_file"]
    if not isinstance(token_file, str) or not token_file:
        raise JoinValidationError(f"{key}: prepared token_file is invalid")
    if record["token_offset_index"] != source_index:
        raise JoinValidationError(
            f"{key}: prepared token_offset_index differs from source_index"
        )
    input_hash = record[PREPARED_INPUT_HASH_FIELD]
    if not isinstance(input_hash, str) or not _SHA256_RE.fullmatch(input_hash):
        raise JoinValidationError(
            f"{key}: prepared {PREPARED_INPUT_HASH_FIELD} is invalid"
        )
    return key


def load_prepared_expectations(
    prepared_dir: Path | str,
    *,
    require_complete: bool = True,
) -> PreparedExpectations:
    """Validate an immutable preparation artifact and return its key authority."""

    directory = Path(prepared_dir).expanduser().resolve(strict=False)
    if not directory.is_dir():
        raise JoinValidationError(
            f"prepared artifact directory does not exist: {directory}"
        )
    manifest_path = directory / "manifest.json"
    status_path = directory / "status.json"
    manifest = _read_json_object(manifest_path, "manifest")
    status = _read_json_object(status_path, "status")
    if manifest.get("schema_version") != PREPARED_SCHEMA:
        raise JoinValidationError("prepared manifest schema is unsupported")
    if status.get("schema_version") != PREPARED_SCHEMA:
        raise JoinValidationError("prepared status schema is unsupported")
    if manifest.get("status") != "complete" or status.get("status") != "complete":
        raise JoinValidationError(
            "prepared manifest and status must both declare complete"
        )

    manifest_digest = file_sha256(manifest_path)
    status_digest = file_sha256(status_path)
    if status.get("manifest_sha256") != manifest_digest:
        raise JoinValidationError(
            "prepared status manifest_sha256 does not match manifest.json"
        )
    if status.get("completed_at_utc") != manifest.get("completed_at_utc"):
        raise JoinValidationError(
            "prepared status and manifest completion timestamps differ"
        )
    if manifest.get("protocol_config_hash") != protocol_config_hash():
        raise JoinValidationError(
            "prepared manifest was built from a different protocol configuration"
        )
    input_policy = manifest.get("input_policy")
    if not isinstance(input_policy, dict):
        raise JoinValidationError("prepared manifest lacks input_policy")
    if input_policy.get("cap") != FINAL_INPUT_TOKEN_CAP:
        raise JoinValidationError("prepared manifest does not use the 24K cap")
    if (
        input_policy.get("middle_truncation_tokens_per_side")
        != MIDDLE_TRUNCATION_TOKENS_PER_SIDE
    ):
        raise JoinValidationError(
            "prepared manifest does not use exact 12K-per-side truncation"
        )
    if input_policy.get("token_hash_algorithm") != TOKEN_HASH_ALGORITHM:
        raise JoinValidationError(
            "prepared manifest uses an unexpected token hash algorithm"
        )
    if input_policy.get("decode_and_retokenize_after_truncation") is not False:
        raise JoinValidationError(
            "prepared manifest permits decode/re-tokenize after truncation"
        )
    if input_policy.get("no_chat_add_special_tokens") is not True:
        raise JoinValidationError(
            "prepared manifest does not lock no-chat add_special_tokens=True"
        )
    if (
        input_policy.get("chat_template_date_string")
        != CHAT_TEMPLATE_DATE_STRING
    ):
        raise JoinValidationError(
            "prepared manifest chat-template date string is not locked"
        )
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, dict):
        raise JoinValidationError("prepared manifest lacks benchmark metadata")
    expected_benchmark = {
        "requested_dataset_id": LONG_BENCH_DATASET_ID,
        "resolved_dataset_id": LONG_BENCH_RESOLVED_DATASET_ID,
        "dataset_revision": LONG_BENCH_DATASET_REVISION,
        "split": LONG_BENCH_SPLIT,
    }
    for field, expected_value in expected_benchmark.items():
        if benchmark.get(field) != expected_value:
            raise JoinValidationError(
                f"prepared benchmark {field} differs from the locked protocol"
            )
    if benchmark.get("source_hashes_verified_against_pinned_release") is not True:
        raise JoinValidationError(
            "prepared benchmark source hashes were not verified against the "
            "pinned release"
        )
    source_code = manifest.get("source_code")
    if not isinstance(source_code, dict):
        raise JoinValidationError("prepared manifest lacks source_code")
    source_commit = source_code.get("commit")
    if (
        not isinstance(source_commit, str)
        or not _GIT_COMMIT_RE.fullmatch(source_commit)
    ):
        raise JoinValidationError(
            "prepared manifest lacks an exact source-code commit"
        )
    if source_code.get("dirty") is not False:
        raise JoinValidationError(
            "prepared inputs were not built from a clean source tree"
        )
    if manifest.get("protocol_config_files") != config_file_hashes():
        raise JoinValidationError(
            "prepared manifest protocol file hashes differ from the current "
            "locked protocol"
        )
    prepared_run_config = manifest.get("run_config")
    if not isinstance(prepared_run_config, dict):
        raise JoinValidationError("prepared manifest lacks run_config metadata")
    current_run_config_hash = file_sha256(ORACLE_CONFIG_PATH)
    if prepared_run_config.get("sha256") != current_run_config_hash:
        raise JoinValidationError(
            "prepared manifest run-config hash differs from the current "
            "locked configuration"
        )

    if require_complete:
        source_entries = benchmark.get("source_files")
        if not isinstance(source_entries, list):
            raise JoinValidationError(
                "prepared benchmark lacks source-file evidence"
            )
        source_by_task = {
            entry.get("task"): entry
            for entry in source_entries
            if isinstance(entry, dict)
        }
        if set(source_by_task) != set(TASK_SPECS):
            raise JoinValidationError(
                "prepared source-file evidence does not cover all 16 tasks"
            )
        pinned_hashes = source_manifest()["longbench_dataset"][
            "extracted_task_sha256"
        ]
        for task, spec in TASK_SPECS.items():
            entry = source_by_task[task]
            if (
                entry.get("sha256") != pinned_hashes[task]
                or entry.get("rows") != spec.expected_test_examples
            ):
                raise JoinValidationError(
                    f"prepared source-file evidence is invalid for {task}"
                )

    prepared_index = manifest.get("prepared_index")
    if not isinstance(prepared_index, dict):
        raise JoinValidationError("prepared manifest lacks prepared_index")
    index_relative = prepared_index.get("path")
    if not isinstance(index_relative, str) or not index_relative:
        raise JoinValidationError("prepared index path is invalid")
    relative_path = Path(index_relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise JoinValidationError("prepared index path must stay inside prepared_dir")
    index_path = (directory / relative_path).resolve(strict=False)
    try:
        index_path.relative_to(directory)
    except ValueError as exc:
        raise JoinValidationError(
            "prepared index path resolves outside prepared_dir"
        ) from exc
    try:
        index_digest = file_sha256(index_path)
    except OSError as exc:
        raise JoinValidationError(
            f"prepared index is missing or unreadable: {index_path}"
        ) from exc
    declared_index_hash = prepared_index.get("sha256")
    if (
        not isinstance(declared_index_hash, str)
        or not _SHA256_RE.fullmatch(declared_index_hash)
        or declared_index_hash != index_digest
    ):
        raise JoinValidationError(
            "prepared index SHA-256 does not match the manifest"
        )

    records = _read_prepared_index(index_path)
    manifest_records = manifest.get("records")
    declared_index_rows = prepared_index.get("rows")
    if (
        isinstance(manifest_records, bool)
        or not isinstance(manifest_records, int)
        or isinstance(declared_index_rows, bool)
        or not isinstance(declared_index_rows, int)
        or manifest_records != declared_index_rows
        or manifest_records != len(records)
    ):
        raise JoinValidationError(
            "prepared manifest/index row counts are inconsistent"
        )

    token_entries = manifest.get("token_files")
    if not isinstance(token_entries, list):
        raise JoinValidationError("prepared manifest lacks token-file evidence")
    token_by_task = {
        entry.get("task"): entry
        for entry in token_entries
        if isinstance(entry, dict)
    }
    expected_token_tasks = (
        set(TASK_SPECS)
        if require_complete
        else {record.get("task") for record in records}
    )
    if set(token_by_task) != expected_token_tasks:
        raise JoinValidationError(
            "prepared token-file evidence does not cover the expected tasks"
        )
    token_paths: dict[str, Path] = {}
    token_file_hashes: list[tuple[Path, str]] = []
    for task in TASK_ORDER:
        if task not in token_by_task:
            continue
        entry = token_by_task[task]
        relative_token_path = entry.get("path")
        if (
            not isinstance(relative_token_path, str)
            or not relative_token_path
        ):
            raise JoinValidationError(
                f"prepared token-file path is invalid for {task}"
            )
        relative_token = Path(relative_token_path)
        if relative_token.is_absolute() or ".." in relative_token.parts:
            raise JoinValidationError(
                f"prepared token-file path escapes prepared_dir for {task}"
            )
        token_path = (directory / relative_token).resolve(strict=False)
        try:
            token_path.relative_to(directory)
            token_digest = file_sha256(token_path)
        except (ValueError, OSError) as exc:
            raise JoinValidationError(
                f"prepared token file is missing or unsafe for {task}"
            ) from exc
        if (
            entry.get("sha256") != token_digest
            or isinstance(entry.get("rows"), bool)
            or not isinstance(entry.get("rows"), int)
            or entry.get("rows") <= 0
            or isinstance(entry.get("stored_token_ids"), bool)
            or not isinstance(entry.get("stored_token_ids"), int)
            or entry.get("stored_token_ids") <= 0
        ):
            raise JoinValidationError(
                f"prepared token-file evidence is invalid for {task}"
            )
        if (
            require_complete
            and entry.get("rows") != TASK_SPECS[task].expected_test_examples
        ):
            raise JoinValidationError(
                f"prepared token-file row count is invalid for {task}"
            )
        token_paths[task] = relative_token
        token_file_hashes.append((token_path, token_digest))

    for record in records:
        task = record.get("task")
        if task in token_paths and record.get("token_file") != str(
            token_paths[task]
        ):
            raise JoinValidationError(
                f"prepared index token_file differs from manifest for {task}"
            )
    if require_complete and len(records) != EXPECTED_TOTAL_EXAMPLES:
        raise JoinValidationError(
            f"prepared artifact has {len(records)} rows; "
            f"expected {EXPECTED_TOTAL_EXAMPLES}"
        )

    ordered_keys = tuple(
        _validate_prepared_record(record, index)
        for index, record in enumerate(records)
    )
    if len(ordered_keys) != len(set(ordered_keys)):
        raise JoinValidationError(
            "prepared index contains duplicate (task, benchmark_id) keys"
        )
    task_source_pairs = tuple(
        (record["task"], record["source_index"]) for record in records
    )
    if len(task_source_pairs) != len(set(task_source_pairs)):
        raise JoinValidationError(
            "prepared index contains duplicate (task, source_index) pairs"
        )
    task_position = {task: index for index, task in enumerate(TASK_ORDER)}
    if task_source_pairs != tuple(
        sorted(
            task_source_pairs,
            key=lambda item: (task_position[item[0]], item[1]),
        )
    ):
        raise JoinValidationError(
            "prepared index is not in locked task/source_index order"
        )
    if require_complete:
        expected_task_source = tuple(
            (task, source_index)
            for task, spec in TASK_SPECS.items()
            for source_index in range(spec.expected_test_examples)
        )
        if task_source_pairs != expected_task_source:
            raise JoinValidationError(
                "prepared index does not contain the exact 3,750 "
                "task/source_index panel"
            )
        computed_truncations = {
            task: sum(
                bool(record["truncated"])
                for record in records
                if record["task"] == task
            )
            for task in TASK_ORDER
        }
        if manifest.get("truncation_counts") != computed_truncations:
            raise JoinValidationError(
                "prepared manifest truncation_counts differ from the index"
            )
        if manifest.get("total_truncated") != sum(
            computed_truncations.values()
        ):
            raise JoinValidationError(
                "prepared manifest total_truncated differs from the index"
            )

    models = {record["model"] for record in records}
    model_ids = {record["model_id"] for record in records}
    if len(models) != 1 or len(model_ids) != 1:
        raise JoinValidationError(
            "prepared index mixes model aliases or model IDs"
        )
    tokenizer_metadata = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer_metadata, dict)
        or tokenizer_metadata.get("model_id") not in model_ids
    ):
        raise JoinValidationError(
            "prepared manifest tokenizer model_id differs from the index"
        )

    try:
        final_hashes = (
            file_sha256(manifest_path),
            file_sha256(status_path),
            file_sha256(index_path),
        )
    except OSError as exc:
        raise JoinValidationError(
            "prepared artifact changed or disappeared during validation"
        ) from exc
    if final_hashes != (manifest_digest, status_digest, index_digest):
        raise JoinValidationError(
            "prepared artifact changed during validation"
        )

    return PreparedExpectations(
        prepared_dir=directory,
        manifest_path=manifest_path,
        status_path=status_path,
        index_path=index_path,
        manifest_sha256=manifest_digest,
        status_sha256=status_digest,
        index_sha256=index_digest,
        token_file_hashes=tuple(token_file_hashes),
        records=tuple(copy.deepcopy(records)),
        ordered_keys=ordered_keys,
        records_by_key={
            key: copy.deepcopy(record)
            for key, record in zip(ordered_keys, records, strict=True)
        },
    )


def _validate_journals_against_prepared(
    validations: Mapping[str, ConfigValidation],
    prepared: PreparedExpectations,
) -> None:
    common_fields = tuple(
        field for field in REQUIRED_SHARED_FIELDS if field != FINAL_INPUT_HASH_FIELD
    )
    for config in FIXED_CONFIGS:
        validation = validations[config]
        for key in prepared.ordered_keys:
            journal_record = validation.records_by_key[key]
            prepared_record = prepared.records_by_key[key]
            observed_hash = journal_record[FINAL_INPUT_HASH_FIELD]
            expected_hash = prepared_record[PREPARED_INPUT_HASH_FIELD]
            if observed_hash != expected_hash:
                raise JoinValidationError(
                    f"{key}: {config} {FINAL_INPUT_HASH_FIELD} does not match "
                    "the prepared index"
                )
            for field in common_fields:
                observed = _canonical_json(
                    journal_record[field], f"{key} {config} {field}"
                )
                expected = _canonical_json(
                    prepared_record[field], f"{key} prepared {field}"
                )
                if observed != expected:
                    raise JoinValidationError(
                        f"{key}: {config} field {field!r} does not match "
                        "the prepared index"
                    )
            for field in OPTIONAL_SHARED_FIELDS:
                if field in journal_record and field in prepared_record:
                    observed = _canonical_json(
                        journal_record[field], f"{key} {config} {field}"
                    )
                    expected = _canonical_json(
                        prepared_record[field], f"{key} prepared {field}"
                    )
                    if observed != expected:
                        raise JoinValidationError(
                            f"{key}: {config} field {field!r} does not match "
                            "the prepared index"
                        )


def _config_values(
    record: Mapping[str, Any],
    config: str,
    key: tuple[str, str],
    *,
    verify_score: bool,
) -> dict[str, Any]:
    prediction = record.get("prediction")
    if not isinstance(prediction, str):
        raise JoinValidationError(f"{key}: {config} prediction must be a string")
    score = _finite_number(record.get("score"), f"{key} {config} score")
    if not 0 <= score <= 1:
        raise JoinValidationError(f"{key}: {config} score must be in [0, 1]")
    if verify_score:
        recomputed_score = score_prediction(
            key[0],
            prediction,
            record["references"],
            all_classes=record.get("all_classes"),
        )
        if not math.isclose(
            score,
            recomputed_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise JoinValidationError(
                f"{key}: {config} stored score {score} differs from the "
                f"official recomputed score {recomputed_score}"
            )
    kv_bytes = _finite_number(record.get("kv_bytes"), f"{key} {config} kv_bytes")
    kv_bytes_fp16 = _finite_number(
        record.get("kv_bytes_fp16"), f"{key} {config} kv_bytes_fp16"
    )
    compression = _finite_number(
        record.get("compression"), f"{key} {config} compression"
    )
    if kv_bytes <= 0 or kv_bytes_fp16 <= 0 or compression <= 0:
        raise JoinValidationError(
            f"{key}: {config} KV byte fields and compression must be positive"
        )
    derived_compression = kv_bytes_fp16 / kv_bytes
    if not math.isclose(
        compression,
        derived_compression,
        rel_tol=COMPRESSION_REL_TOLERANCE,
        abs_tol=COMPRESSION_ABS_TOLERANCE,
    ):
        raise JoinValidationError(
            f"{key}: {config} compression={compression} does not match "
            f"kv_bytes_fp16/kv_bytes={derived_compression}"
        )
    spec = TASK_SPECS[key[0]]
    generated_count = _integer(
        record.get("generated_token_count"),
        f"{key} {config} generated_token_count",
        minimum=spec.minimum_new_tokens,
        maximum=spec.max_new_tokens,
    )
    kv_accounting = record.get("kv_accounting")
    if kv_accounting != KV_ACCOUNTING_BY_CONFIG[config]:
        raise JoinValidationError(
            f"{key}: {config} KV accounting semantics differ from the "
            "locked protocol"
        )
    if config == "C0":
        if not math.isclose(
            kv_bytes,
            kv_bytes_fp16,
            rel_tol=0.0,
            abs_tol=COMPRESSION_ABS_TOLERANCE,
        ) or not math.isclose(
            derived_compression,
            1.0,
            rel_tol=0.0,
            abs_tol=COMPRESSION_ABS_TOLERANCE,
        ):
            raise JoinValidationError(
                f"{key}: C0 must remain the exact FP16 byte reference"
            )
    return {
        "score": score,
        "prediction": prediction,
        "kv_bytes": kv_bytes,
        "kv_bytes_fp16": kv_bytes_fp16,
        "compression": derived_compression,
        "generated_token_count": generated_count,
        "kv_accounting": copy.deepcopy(dict(kv_accounting)),
    }


def _join_one(
    key: tuple[str, str],
    records: Mapping[str, Mapping[str, Any]],
    prepared_record: Mapping[str, Any] | None = None,
    *,
    verify_scores: bool,
) -> dict[str, Any]:
    for config in FIXED_CONFIGS:
        _validate_shared_record(records[config], config, key)

    shared = {
        field: _require_equal(records, field, key, required=True)
        for field in REQUIRED_SHARED_FIELDS
    }
    for field in OPTIONAL_SHARED_FIELDS:
        value = _require_equal(records, field, key, required=False)
        if value is not None or all(field in records[c] for c in FIXED_CONFIGS):
            shared[field] = value
        elif prepared_record is not None and field in prepared_record:
            shared[field] = copy.deepcopy(prepared_record[field])

    task, benchmark_id = key
    row: dict[str, Any] = {
        "schema_version": JOIN_SCHEMA,
        "task": task,
        "category": TASK_SPECS[task].category,
        "benchmark_id": benchmark_id,
        **shared,
    }
    # Category is registry-derived even when it was absent from raw records.
    row["category"] = TASK_SPECS[task].category
    for config in FIXED_CONFIGS:
        values = _config_values(
            records[config],
            config,
            key,
            verify_score=verify_scores,
        )
        for field in CONFIG_OUTPUT_FIELDS:
            row[f"{config}_{field}"] = values[field]
    return row


def _validate_source_indices(
    rows: Sequence[Mapping[str, Any]], *, require_complete: bool
) -> None:
    by_task: OrderedDict[str, list[int]] = OrderedDict(
        (task, []) for task in TASK_ORDER
    )
    for row in rows:
        by_task[row["task"]].append(row["source_index"])
    for task, indices in by_task.items():
        if len(indices) != len(set(indices)):
            raise JoinValidationError(f"{task} contains duplicate source_index values")
        if require_complete:
            expected = list(range(TASK_SPECS[task].expected_test_examples))
            if sorted(indices) != expected:
                raise JoinValidationError(
                    f"{task} source_index values are not exactly 0..{len(expected) - 1}"
                )


def join_journals(
    journal_paths: Mapping[str, Path | str],
    *,
    prepared_dir: Path | str | None = None,
    expected_keys: Iterable[Sequence[Any] | Mapping[str, Any]] | None = None,
    require_complete: bool = True,
) -> JoinResult:
    """Validate and join C0--C5 completed journals by ``(task, benchmark_id)``.

    Production callers must supply ``prepared_dir`` and leave
    ``require_complete=True``.  Its validated ``index.jsonl`` is the authority
    for all 3,750 keys and final token hashes.  ``expected_keys`` is an
    intentionally separate injection boundary for CPU tests and smoke runs.
    """

    if prepared_dir is not None and expected_keys is not None:
        raise JoinValidationError(
            "provide prepared_dir for production or expected_keys for a "
            "test/smoke join, not both"
        )
    if prepared_dir is None and expected_keys is None:
        raise JoinValidationError(
            "expected IDs may not be derived from C0; provide prepared_dir "
            "(production) or explicit expected_keys (test/smoke)"
        )
    missing = set(FIXED_CONFIGS) - set(journal_paths)
    extra = set(journal_paths) - set(FIXED_CONFIGS)
    if missing or extra:
        raise JoinValidationError(
            f"journal_paths must contain exactly C0-C5; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    paths = {config: Path(journal_paths[config]) for config in FIXED_CONFIGS}
    for config, path in paths.items():
        if not path.is_file():
            raise JoinValidationError(f"{config} journal does not exist: {path}")

    prepared = (
        load_prepared_expectations(
            prepared_dir, require_complete=require_complete
        )
        if prepared_dir is not None
        else None
    )
    normalized_keys = (
        prepared.ordered_keys
        if prepared is not None
        else _normalize_expected_keys(expected_keys or ())
    )
    _validate_expected_panel(normalized_keys, require_complete=require_complete)

    validations: dict[str, ConfigValidation] = {
        config: validate_config_records(config, paths[config], normalized_keys)
        for config in FIXED_CONFIGS
    }
    cross = validate_cross_config_hashes(validations)
    if prepared is not None:
        _validate_journals_against_prepared(validations, prepared)
    indexed = {
        config: validations[config].records_by_key for config in FIXED_CONFIGS
    }
    rows = [
        _join_one(
            key,
            {config: indexed[config][key] for config in FIXED_CONFIGS},
            prepared.records_by_key[key] if prepared is not None else None,
            verify_scores=require_complete,
        )
        for key in normalized_keys
    ]
    _validate_source_indices(rows, require_complete=require_complete)
    task_position = {task: index for index, task in enumerate(TASK_ORDER)}
    rows.sort(
        key=lambda row: (
            task_position[row["task"]],
            row["source_index"],
            row["benchmark_id"],
        )
    )
    if require_complete and len(rows) != EXPECTED_TOTAL_EXAMPLES:
        raise JoinValidationError(
            f"joined {len(rows)} rows; expected {EXPECTED_TOTAL_EXAMPLES}"
        )
    # Detect any mutation after per-record validation before returning evidence.
    cross = validate_cross_config_hashes(validations)
    if prepared is not None:
        try:
            current_prepared_hashes = (
                file_sha256(prepared.manifest_path),
                file_sha256(prepared.status_path),
                file_sha256(prepared.index_path),
            )
        except OSError as exc:
            raise JoinValidationError(
                "prepared artifact disappeared during the join"
            ) from exc
        if current_prepared_hashes != (
            prepared.manifest_sha256,
            prepared.status_sha256,
            prepared.index_sha256,
        ):
            raise JoinValidationError(
                "prepared artifact changed during the join"
            )
        try:
            current_token_hashes = tuple(
                (path, file_sha256(path))
                for path, _ in prepared.token_file_hashes
            )
        except OSError as exc:
            raise JoinValidationError(
                "prepared token files disappeared during the join"
            ) from exc
        if current_token_hashes != prepared.token_file_hashes:
            raise JoinValidationError(
                "prepared token files changed during the join"
            )
    return JoinResult(tuple(rows), validations, cross, prepared)


def validate_locked_oracle_policy(
    path: Path | str = ORACLE_CONFIG_PATH,
) -> dict[str, Any]:
    """Fail closed unless the repository config contains the locked policy."""

    config_path = Path(path)
    try:
        raw_config = config_path.read_bytes()
        payload = json.loads(raw_config)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OraclePolicyError(
            f"cannot read locked oracle policy from {config_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise OraclePolicyError("LongBench16 run config must be an object")
    policy = payload.get("oracle_analysis")
    expected_primary = {
        "name": PRIMARY_ORACLE_OBJECTIVE,
        "quality_threshold": (
            "actual_candidate_score_gte_0.99_times_actual_c0_score"
        ),
        "eligible_ranking": (
            "highest_actual_per_prompt_compression_then_fixed_config_order"
        ),
        "fallback": (
            "highest_actual_quality_then_actual_compression_then_fixed_config_order"
        ),
    }
    if (
        not isinstance(policy, dict)
        or policy.get("interpretation")
        != "post_hoc_upper_bound_not_a_deployable_router"
        or tuple(policy.get("candidate_configurations", ()))
        != ORACLE_CANDIDATES
        or policy.get("primary") != expected_primary
        or policy.get("required_diagnostic") != QUALITY_ORACLE_DIAGNOSTIC
    ):
        raise OraclePolicyError(
            "configs/longbench16_24k.json does not contain the exact locked "
            "oracle_analysis policy"
        )
    return {
        "path": str(config_path.resolve()),
        "sha256": hashlib.sha256(raw_config).hexdigest(),
        "policy": copy.deepcopy(policy),
    }


def _require_oracle_objective(objective: str | None) -> str:
    validate_locked_oracle_policy()
    selected = PRIMARY_ORACLE_OBJECTIVE if objective is None else objective
    if selected not in ORACLE_OBJECTIVES:
        raise OraclePolicyError(
            f"unsupported oracle objective {selected!r}; "
            f"select one of {ORACLE_OBJECTIVES}"
        )
    return selected


def _oracle_tie_rule(objective: str) -> list[str]:
    if objective == PRIMARY_ORACLE_OBJECTIVE:
        return [
            "eligible iff actual candidate score >= 0.99 * actual C0 score",
            "among eligible candidates, highest actual per-prompt compression",
            "eligible compression ties use fixed [C1, C2, C3, C4, C5] order",
            (
                "if none eligible, highest actual quality, then actual "
                "compression, then fixed order"
            ),
        ]
    return [
        "highest actual per-prompt quality",
        "then highest actual per-prompt measured compression",
        "then fixed [C1, C2, C3, C4, C5] order",
    ]


def _oracle_decision(
    row: Mapping[str, Any], objective: str
) -> dict[str, Any]:
    config_preference = {
        config: -index for index, config in enumerate(ORACLE_CANDIDATES)
    }
    if objective == QUALITY_ORACLE_DIAGNOSTIC:
        selected = max(
            ORACLE_CANDIDATES,
            key=lambda config: (
                row[f"{config}_score"],
                row[f"{config}_compression"],
                config_preference[config],
            ),
        )
        return {
            "configuration": selected,
            "fallback": False,
            "quality_threshold": None,
            "eligible_configurations": None,
            "actual_threshold_violation": False,
        }

    threshold = ISO_QUALITY_TAU * row["C0_score"]
    eligible = [
        config
        for config in ORACLE_CANDIDATES
        if row[f"{config}_score"] >= threshold
    ]
    fallback = not eligible
    if eligible:
        selected = max(
            eligible,
            key=lambda config: (
                row[f"{config}_compression"],
                config_preference[config],
            ),
        )
    else:
        selected = max(
            ORACLE_CANDIDATES,
            key=lambda config: (
                row[f"{config}_score"],
                row[f"{config}_compression"],
                config_preference[config],
            ),
        )
    return {
        "configuration": selected,
        "fallback": fallback,
        "quality_threshold": threshold,
        "eligible_configurations": eligible,
        "actual_threshold_violation": (
            not fallback and row[f"{selected}_score"] < threshold
        ),
    }


def apply_oracle(
    rows: Iterable[Mapping[str, Any]],
    *,
    objective: str | None,
) -> tuple[dict[str, Any], ...]:
    """Copy joined rows and add deterministic post-hoc oracle columns."""

    selected_objective = _require_oracle_objective(objective)
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise JoinValidationError(f"joined row {index} is not an object")
        row = copy.deepcopy(dict(source))
        decision = _oracle_decision(row, selected_objective)
        selected = decision["configuration"]
        row["oracle_objective"] = selected_objective
        row["oracle_configuration"] = selected
        for field in CONFIG_OUTPUT_FIELDS:
            row[f"oracle_{field}"] = row[f"{selected}_{field}"]
        row["oracle_fallback"] = decision["fallback"]
        row["oracle_quality_threshold"] = decision["quality_threshold"]
        row["oracle_eligible_configurations"] = decision[
            "eligible_configurations"
        ]
        row["oracle_actual_threshold_violation"] = decision[
            "actual_threshold_violation"
        ]
        output.append(row)
    if not output:
        raise JoinValidationError("cannot analyze an empty join")
    return tuple(output)


def _aggregate_one_configuration(
    rows: Sequence[Mapping[str, Any]],
    config: str,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    quality_records = [
        {"task": row["task"], "score": row[f"{config}_score"]} for row in rows
    ]
    compression_records = [
        {
            "task": row["task"],
            "kv_bytes": row[f"{config}_kv_bytes"],
            "kv_bytes_fp16": row[f"{config}_kv_bytes_fp16"],
        }
        for row in rows
    ]
    return {
        "quality": aggregate_quality(
            quality_records, require_complete=require_complete
        ),
        "compression": aggregate_compression(
            compression_records, require_complete=require_complete
        ),
    }


def _oracle_summary(
    rows: Sequence[Mapping[str, Any]],
    objective: str,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    quality = aggregate_quality(
        (
            {"task": row["task"], "score": row["oracle_score"]}
            for row in rows
        ),
        require_complete=require_complete,
    )
    compression = aggregate_compression(
        (
            {
                "task": row["task"],
                "kv_bytes": row["oracle_kv_bytes"],
                "kv_bytes_fp16": row["oracle_kv_bytes_fp16"],
            }
            for row in rows
        ),
        require_complete=require_complete,
    )
    selection_counts = Counter(row["oracle_configuration"] for row in rows)
    selection_counts_by_task = {
        task: {
            config: sum(
                row["task"] == task
                and row["oracle_configuration"] == config
                for row in rows
            )
            for config in ORACLE_CANDIDATES
        }
        for task in TASK_ORDER
    }
    summary: dict[str, Any] = {
        "name": objective,
        "deterministic_tie_rule": _oracle_tie_rule(objective),
        "quality": quality,
        "compression": compression,
        "selection_counts": {
            config: selection_counts.get(config, 0)
            for config in ORACLE_CANDIDATES
        },
        "selection_counts_by_task": selection_counts_by_task,
    }
    if objective == PRIMARY_ORACLE_OBJECTIVE:
        fallback_count = sum(bool(row["oracle_fallback"]) for row in rows)
        nonfallback_count = len(rows) - fallback_count
        violation_count = sum(
            bool(row["oracle_actual_threshold_violation"])
            for row in rows
            if not row["oracle_fallback"]
        )
        if violation_count:
            raise JoinValidationError(
                "iso-quality oracle selected a threshold-violating "
                "nonfallback candidate"
            )
        summary.update(
            {
                "quality_threshold_tau": ISO_QUALITY_TAU,
                "quality_threshold": (
                    "actual_candidate_score >= 0.99 * actual_C0_score"
                ),
                "fallback_count": fallback_count,
                "fallback_fraction": fallback_count / len(rows),
                "nonfallback_count": nonfallback_count,
                "actual_threshold_violation_count_nonfallback": (
                    violation_count
                ),
                "actual_threshold_violation_fraction_nonfallback": (
                    violation_count / nonfallback_count
                    if nonfallback_count
                    else 0.0
                ),
            }
        )
    return summary


def analyze_joined_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    oracle_objective: str | None = None,
    require_complete: bool = True,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Compute fixed methods and both required post-hoc oracle analyses."""

    policy_metadata = validate_locked_oracle_policy()
    selected_objective = _require_oracle_objective(oracle_objective)
    source_rows = tuple(copy.deepcopy(dict(row)) for row in rows)
    selected_rows = apply_oracle(source_rows, objective=selected_objective)
    primary_rows = (
        selected_rows
        if selected_objective == PRIMARY_ORACLE_OBJECTIVE
        else apply_oracle(source_rows, objective=PRIMARY_ORACLE_OBJECTIVE)
    )
    quality_rows = (
        selected_rows
        if selected_objective == QUALITY_ORACLE_DIAGNOSTIC
        else apply_oracle(source_rows, objective=QUALITY_ORACLE_DIAGNOSTIC)
    )
    oracle_rows: list[dict[str, Any]] = []
    oracle_fields = (
        "objective",
        "configuration",
        *CONFIG_OUTPUT_FIELDS,
        "fallback",
        "quality_threshold",
        "eligible_configurations",
        "actual_threshold_violation",
    )
    for selected, primary, quality in zip(
        selected_rows, primary_rows, quality_rows, strict=True
    ):
        output = copy.deepcopy(selected)
        for field in oracle_fields:
            output[f"primary_oracle_{field}"] = copy.deepcopy(
                primary[f"oracle_{field}"]
            )
            output[f"quality_oracle_{field}"] = copy.deepcopy(
                quality[f"oracle_{field}"]
            )
        oracle_rows.append(output)
    fixed = {
        config: _aggregate_one_configuration(
            oracle_rows, config, require_complete=require_complete
        )
        for config in FIXED_CONFIGS
    }
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "row_count": len(oracle_rows),
        "key_fields": list(DEFAULT_KEY_FIELDS),
        "fixed_configurations": fixed,
        "oracle_analysis": {
            "interpretation": "post-hoc upper bounds; not deployable routers",
            "candidate_pool": list(ORACLE_CANDIDATES),
            "policy_source": {
                "path": policy_metadata["path"],
                "sha256": policy_metadata["sha256"],
            },
            "joined_oracle_columns_objective": selected_objective,
            "primary": _oracle_summary(
                primary_rows,
                PRIMARY_ORACLE_OBJECTIVE,
                require_complete=require_complete,
            ),
            "required_quality_first_diagnostic": _oracle_summary(
                quality_rows,
                QUALITY_ORACLE_DIAGNOSTIC,
                require_complete=require_complete,
            ),
        },
    }
    return tuple(oracle_rows), analysis


def build_analysis(
    join_result: JoinResult,
    *,
    oracle_objective: str | None = None,
    require_complete: bool = True,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Analyze a :class:`JoinResult` and attach journal integrity metadata."""

    current_cross = validate_cross_config_hashes(join_result.validations)
    if (
        current_cross.combined_input_hash_sha256
        != join_result.cross_validation.combined_input_hash_sha256
    ):
        raise JoinValidationError(
            "cross-configuration input hashes changed after the join"
        )
    if len(join_result.rows) != current_cross.record_count:
        raise JoinValidationError(
            "join row count differs from validated journal record count"
        )
    if join_result.prepared is not None:
        prepared = join_result.prepared
        try:
            current_prepared_hashes = (
                file_sha256(prepared.manifest_path),
                file_sha256(prepared.status_path),
                file_sha256(prepared.index_path),
            )
        except OSError as exc:
            raise JoinValidationError(
                "prepared artifact disappeared after the join"
            ) from exc
        expected_prepared_hashes = (
            prepared.manifest_sha256,
            prepared.status_sha256,
            prepared.index_sha256,
        )
        if current_prepared_hashes != expected_prepared_hashes:
            raise JoinValidationError(
                "prepared artifact changed after the join"
            )
        try:
            current_token_hashes = tuple(
                (path, file_sha256(path))
                for path, _ in prepared.token_file_hashes
            )
        except OSError as exc:
            raise JoinValidationError(
                "prepared token files disappeared after the join"
            ) from exc
        if current_token_hashes != prepared.token_file_hashes:
            raise JoinValidationError(
                "prepared token files changed after the join"
            )
        _validate_journals_against_prepared(
            join_result.validations, prepared
        )
    rows, analysis = analyze_joined_rows(
        join_result.rows,
        oracle_objective=oracle_objective,
        require_complete=require_complete,
    )
    analysis["integrity"] = {
        "input_journals": {
            config: {
                "path": str(join_result.validations[config].path),
                "sha256": join_result.validations[config].file_sha256,
                "completed_record_count": join_result.validations[
                    config
                ].record_count,
                "journal_event_count": join_result.validations[
                    config
                ].journal_event_count,
                "failed_attempt_count": join_result.validations[
                    config
                ].failed_attempt_count,
                "retried_key_count": len(
                    join_result.validations[config].retried_keys
                ),
            }
            for config in FIXED_CONFIGS
        },
        "cross_config_final_input_hash": {
            "status": "passed",
            "field": join_result.validations["C0"].hash_field,
            "combined_sha256": (
                join_result.cross_validation.combined_input_hash_sha256
            ),
        },
    }
    if join_result.prepared is not None:
        prepared = join_result.prepared
        analysis["integrity"]["prepared_artifact"] = {
            "status": "passed",
            "directory": str(prepared.prepared_dir),
            "record_count": prepared.record_count,
            "manifest": {
                "path": str(prepared.manifest_path),
                "sha256": prepared.manifest_sha256,
            },
            "status_file": {
                "path": str(prepared.status_path),
                "sha256": prepared.status_sha256,
            },
            "index": {
                "path": str(prepared.index_path),
                "sha256": prepared.index_sha256,
            },
            "journal_hash_field": FINAL_INPUT_HASH_FIELD,
            "prepared_hash_field": PREPARED_INPUT_HASH_FIELD,
        }
    return rows, analysis


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    encoded_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"joined row {index} must be a mapping")
        encoded_rows.append(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    if not encoded_rows:
        raise JoinValidationError("refusing to write an empty joined JSONL")
    return ("\n".join(encoded_rows) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_jsonl(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically create a JSONL artifact without replacing existing output."""

    destination = assert_safe_write_path(path)
    payload = _jsonl_bytes(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": payload.count(b"\n"),
    }


def write_analysis_outputs(
    joined_path: Path | str,
    analysis_path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish immutable joined JSONL and analysis JSON artifacts atomically."""

    joined_destination = assert_safe_write_path(joined_path)
    analysis_destination = assert_safe_write_path(analysis_path)
    if joined_destination == analysis_destination:
        raise ValueError("joined_path and analysis_path must differ")
    if joined_destination.exists():
        raise FileExistsError(joined_destination)
    if analysis_destination.exists():
        raise FileExistsError(analysis_destination)

    joined_metadata = atomic_write_jsonl(joined_destination, rows)
    payload = copy.deepcopy(dict(analysis))
    payload["artifacts"] = {
        "joined_jsonl": joined_metadata,
        "analysis_json": {"path": str(analysis_destination)},
    }
    try:
        atomic_write_json(analysis_destination, payload)
    except BaseException:
        joined_destination.unlink(missing_ok=True)
        _fsync_directory(joined_destination.parent)
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly join and aggregate a complete C0-C5 LongBench16 run."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Run root containing C0/per_prompt.jsonl through C5/per_prompt.jsonl.",
    )
    parser.add_argument(
        "--prepared-dir",
        required=True,
        type=Path,
        help=(
            "Validated preparation artifact used by the generation run. "
            "Its index defines the production composite-ID and token-hash set."
        ),
    )
    parser.add_argument(
        "--oracle-objective",
        default=PRIMARY_ORACLE_OBJECTIVE,
        choices=ORACLE_OBJECTIVES,
        help=(
            "Controls the generic oracle_* columns; default is the locked "
            "tau=0.99 primary. Both configured oracle summaries are always "
            "reported."
        ),
    )
    parser.add_argument(
        "--joined-output",
        type=Path,
        help="Default: <run-dir>/analysis/joined.jsonl",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        help="Default: <run-dir>/analysis/summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    journal_paths = {
        config: args.run_dir / config / "per_prompt.jsonl"
        for config in FIXED_CONFIGS
    }
    result = join_journals(journal_paths, prepared_dir=args.prepared_dir)
    rows, analysis = build_analysis(
        result, oracle_objective=args.oracle_objective
    )
    joined_output = (
        args.joined_output
        if args.joined_output is not None
        else args.run_dir / "analysis" / "joined.jsonl"
    )
    analysis_output = (
        args.analysis_output
        if args.analysis_output is not None
        else args.run_dir / "analysis" / "summary.json"
    )
    payload = write_analysis_outputs(
        joined_output, analysis_output, rows, analysis
    )
    print(
        json.dumps(
            {
                "row_count": payload["row_count"],
                "joined_output": str(joined_output),
                "analysis_output": str(analysis_output),
                "oracle_objective": args.oracle_objective,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ANALYSIS_SCHEMA",
    "COMPRESSION_ABS_TOLERANCE",
    "COMPRESSION_REL_TOLERANCE",
    "CONFIG_OUTPUT_FIELDS",
    "FEATURE_KEYS",
    "JOIN_SCHEMA",
    "ISO_QUALITY_TAU",
    "JoinResult",
    "JoinValidationError",
    "ORACLE_CANDIDATES",
    "ORACLE_CONFIG_PATH",
    "ORACLE_OBJECTIVES",
    "PRIMARY_ORACLE_OBJECTIVE",
    "PREPARED_INPUT_HASH_FIELD",
    "PREPARED_SCHEMA",
    "PreparedExpectations",
    "QUALITY_ORACLE_DIAGNOSTIC",
    "OraclePolicyError",
    "analyze_joined_rows",
    "apply_oracle",
    "atomic_write_jsonl",
    "build_analysis",
    "join_journals",
    "load_prepared_expectations",
    "main",
    "validate_locked_oracle_policy",
    "write_analysis_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
