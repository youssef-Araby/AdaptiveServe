#!/usr/bin/env python3
"""Prepare the immutable 3,750-example LongBench16 input set.

This is a CPU-only stage. It validates the pinned LongBench source files,
constructs the official prompts, extracts the router's seven pre-chat features,
applies the Llama chat template where required, and performs exact token-space
middle truncation. The finalized token IDs are persisted once so C0--C5 consume
identical model inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .longbench16_protocol import (
        EXPECTED_TOTAL_EXAMPLES,
        FINAL_INPUT_TOKEN_CAP,
        LONG_BENCH_DATASET_ID,
        LONG_BENCH_DATASET_REVISION,
        LONG_BENCH_RESOLVED_DATASET_ID,
        LONG_BENCH_SPLIT,
        MIDDLE_TRUNCATION_TOKENS_PER_SIDE,
        TASK_SPECS,
        TaskSpec,
        config_file_hashes,
        format_task_prompt,
        protocol_config_hash,
        sha256_file,
        source_manifest,
    )
except ImportError:  # Direct execution: python scripts/longbench16_prepare.py
    from longbench16_protocol import (
        EXPECTED_TOTAL_EXAMPLES,
        FINAL_INPUT_TOKEN_CAP,
        LONG_BENCH_DATASET_ID,
        LONG_BENCH_DATASET_REVISION,
        LONG_BENCH_RESOLVED_DATASET_ID,
        LONG_BENCH_SPLIT,
        MIDDLE_TRUNCATION_TOKENS_PER_SIDE,
        TASK_SPECS,
        TaskSpec,
        config_file_hashes,
        format_task_prompt,
        protocol_config_hash,
        sha256_file,
        source_manifest,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("~/.cache/longbench/data").expanduser()
DEFAULT_RUN_CONFIG = REPO_ROOT / "configs" / "longbench16_24k.json"
PREPARED_SCHEMA = "adaptiveserve-longbench16-prepared/v1"
TOKEN_HASH_ALGORITHM = "sha256-little-endian-uint32"
ORDERED_ID_HASH_ALGORITHM = "sha256-utf8-length-prefixed"
CHAT_TEMPLATE_DATE_STRING = "20 Jul 2026"
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PreparationError(RuntimeError):
    """Raised when source or prepared-input invariants do not hold."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _assert_not_p0(path: Path) -> None:
    p0_root = (REPO_ROOT / "runs" / "p0").resolve()
    candidate = path.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(p0_root)
    except ValueError:
        return
    raise PreparationError(f"Refusing to prepare inputs inside protected {p0_root}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreparationError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreparationError(f"Invalid JSON file: {path}") from exc


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
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
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _validated_clean_source_code(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact clean source snapshot used to prepare the inputs."""

    commit = value.get("commit")
    dirty = value.get("dirty")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise PreparationError(
            "canonical preparation requires a resolved 40-character git commit"
        )
    if dirty is not False:
        raise PreparationError(
            "canonical preparation requires a clean source tree"
        )
    return {"commit": commit, "dirty": False}


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash token IDs as an explicit little-endian unsigned-32-bit byte stream."""

    if isinstance(token_ids, (str, bytes)):
        raise TypeError("token_ids must be a non-string sequence")
    try:
        untyped_array = np.asarray(token_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("token_ids must be one-dimensional") from exc
    if untyped_array.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    values = untyped_array.tolist()
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, (int, np.integer))
        for token_id in values
    ):
        raise TypeError("token IDs must be integers")
    if any(token_id < 0 or token_id > 0xFFFFFFFF for token_id in values):
        raise ValueError("token IDs must fit unsigned 32-bit storage")
    array = np.asarray(values, dtype="<u4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def ordered_ids_sha256(benchmark_ids: Sequence[str]) -> str:
    """Hash ordered IDs without delimiter ambiguity."""

    if isinstance(benchmark_ids, (str, bytes)) or not isinstance(
        benchmark_ids, Sequence
    ):
        raise TypeError("benchmark_ids must be a non-string sequence")
    digest = hashlib.sha256()
    for benchmark_id in benchmark_ids:
        if not isinstance(benchmark_id, str):
            raise TypeError("benchmark IDs must be strings")
        encoded = benchmark_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def middle_truncate_token_ids(
    token_ids: Sequence[int],
    *,
    cap: int = FINAL_INPUT_TOKEN_CAP,
    side: int = MIDDLE_TRUNCATION_TOKENS_PER_SIDE,
) -> tuple[list[int], bool]:
    """Apply exact token-space middle truncation without decode/re-tokenize."""

    if cap <= 0 or side <= 0 or side * 2 != cap:
        raise ValueError("middle truncation requires cap == 2 * side > 0")
    ids = [int(token_id) for token_id in token_ids]
    if any(token_id < 0 or token_id > 0xFFFFFFFF for token_id in ids):
        raise ValueError("token IDs must fit unsigned 32-bit storage")
    if len(ids) <= cap:
        return ids, False
    truncated = ids[:side] + ids[-side:]
    if len(truncated) != cap:
        raise AssertionError("middle truncation produced the wrong token count")
    return truncated, True


def _shannon_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -math.fsum(
        (count / total) * math.log2(count / total)
        for count in counts
        if count > 0
    )


def extract_prompt_features_strict(
    prompt: str, tokenizer: Any
) -> dict[str, int | float | None]:
    """Reproduce the seven-feature router contract without silent fallbacks."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise TypeError("a tokenizer with encode() is required")
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise PreparationError("tokenizer.encode did not return a flat integer list")

    counts = Counter(token_ids)
    raw_bytes = prompt.encode("utf-8", errors="strict")
    compressed_bytes = len(gzip.compress(raw_bytes, compresslevel=6))
    last_question = prompt.rfind("?")
    question_position = (
        last_question / len(prompt) if last_question >= 0 and prompt else None
    )
    n_tokens = len(token_ids)
    return {
        "seq_len_tokens": n_tokens,
        "seq_len_chars": len(prompt),
        "token_entropy": round(_shannon_entropy(list(counts.values())), 4),
        "gzip_ratio": round(compressed_bytes / max(len(raw_bytes), 1), 4),
        "unique_token_ratio": round(len(counts) / max(n_tokens, 1), 4),
        "question_position": (
            round(question_position, 4)
            if question_position is not None
            else None
        ),
        "newline_density": round(
            prompt.count("\n") / max(len(prompt), 1), 4
        ),
    }


def finalize_input_ids(
    tokenizer: Any, spec: TaskSpec, prompt: str
) -> tuple[list[int], int, bool]:
    """Return finalized IDs, their pre-truncation length, and truncation flag."""

    if spec.apply_chat_wrapper:
        if not getattr(tokenizer, "chat_template", None):
            raise PreparationError(
                f"{spec.task_id} requires a tokenizer chat template"
            )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            date_string=CHAT_TEMPLATE_DATE_STRING,
        )
        if hasattr(encoded, "input_ids"):
            token_ids = encoded.input_ids
        elif isinstance(encoded, Mapping):
            token_ids = encoded.get("input_ids")
        else:
            token_ids = encoded
    else:
        # Match the tokenizer's normal model-input construction, including
        # Llama's BOS token, for LongBench's five no-chat tasks.
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)

    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise PreparationError(
            f"{spec.task_id} tokenizer output is not a flat integer list"
        )
    pre_truncation_count = len(token_ids)
    finalized, truncated = middle_truncate_token_ids(token_ids)
    if len(finalized) > FINAL_INPUT_TOKEN_CAP:
        raise AssertionError("finalized input exceeds the 24K cap")
    return finalized, pre_truncation_count, truncated


def load_strict_task_rows(
    data_dir: Path,
    spec: TaskSpec,
    *,
    expected_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and fail-closed validate one official task file."""

    path = data_dir / f"{spec.task_id}.jsonl"
    if not path.is_file():
        raise PreparationError(f"Missing required LongBench source file: {path}")
    observed_sha256 = sha256_file(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise PreparationError(
            f"{spec.task_id} source hash mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    rows: list[dict[str, Any]] = []
    benchmark_ids: list[str] = []
    for source_index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            raise PreparationError(
                f"{path} contains a blank record at source index {source_index}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(
                f"{path} has invalid JSON at source index {source_index}"
            ) from exc
        if not isinstance(row, dict):
            raise PreparationError(f"{path} row {source_index} is not an object")

        required = {
            "_id",
            "input",
            "context",
            "answers",
            "all_classes",
            "dataset",
            "language",
            "length",
        }
        missing = required.difference(row)
        if missing:
            raise PreparationError(
                f"{path} row {source_index} lacks {sorted(missing)}"
            )
        benchmark_id = row["_id"]
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise PreparationError(
                f"{path} row {source_index} has an invalid _id"
            )
        if row["dataset"] != spec.task_id:
            raise PreparationError(
                f"{path} row {source_index} reports dataset={row['dataset']!r}"
            )
        if not isinstance(row["language"], str) or not row["language"]:
            raise PreparationError(
                f"{path} row {source_index} has invalid language"
            )
        if spec.language == "en" and row["language"] != "en":
            raise PreparationError(
                f"{path} row {source_index} is not an English record"
            )
        if (
            isinstance(row["length"], bool)
            or not isinstance(row["length"], int)
            or row["length"] < 0
        ):
            raise PreparationError(
                f"{path} row {source_index} has invalid length"
            )
        if not isinstance(row["input"], str) or not isinstance(
            row["context"], str
        ):
            raise PreparationError(
                f"{path} row {source_index} has non-string prompt fields"
            )
        if not isinstance(row["answers"], list) or not row["answers"]:
            raise PreparationError(
                f"{path} row {source_index} has no reference answers"
            )
        if any(
            not isinstance(answer, str) or not answer
            for answer in row["answers"]
        ):
            raise PreparationError(
                f"{path} row {source_index} has invalid reference answers"
            )
        all_classes = row["all_classes"]
        if all_classes is not None and (
            not isinstance(all_classes, list)
            or not all_classes
            or any(
                not isinstance(class_name, str) or not class_name
                for class_name in all_classes
            )
        ):
            raise PreparationError(
                f"{path} row {source_index} has invalid all_classes"
            )
        if spec.scorer == "classification_score" and all_classes is None:
            raise PreparationError(
                f"{path} row {source_index} requires all_classes"
            )
        if spec.scorer != "classification_score" and all_classes is not None:
            raise PreparationError(
                f"{path} row {source_index} has unexpected all_classes"
            )
        rows.append(row)
        benchmark_ids.append(benchmark_id)

    if len(rows) != spec.expected_test_examples:
        raise PreparationError(
            f"{spec.task_id} has {len(rows)} rows; "
            f"expected {spec.expected_test_examples}"
        )
    if len(set(benchmark_ids)) != len(benchmark_ids):
        duplicates = sorted(
            benchmark_id
            for benchmark_id, count in Counter(benchmark_ids).items()
            if count > 1
        )
        raise PreparationError(
            f"{spec.task_id} has duplicate benchmark IDs: {duplicates[:5]}"
        )

    return rows, {
        "task": spec.task_id,
        "path": str(path),
        "rows": len(rows),
        "sha256": observed_sha256,
        "ordered_benchmark_ids_sha256": ordered_ids_sha256(benchmark_ids),
        "ordered_id_hash_algorithm": ORDERED_ID_HASH_ALGORITHM,
    }


def _tokenizer_metadata(
    tokenizer: Any, *, model_id: str, revision: str
) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": int(getattr(tokenizer, "vocab_size")),
        "eos_token_id": int(getattr(tokenizer, "eos_token_id")),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            if isinstance(chat_template, str)
            else None
        ),
        "no_chat_add_special_tokens": True,
        "chat_template_date_string": CHAT_TEMPLATE_DATE_STRING,
    }


def prepare_inputs(
    *,
    tokenizer: Any,
    data_dir: Path,
    output_dir: Path,
    run_config_path: Path = DEFAULT_RUN_CONFIG,
    model_id: str,
    model_revision: str,
    _expected_task_hashes_for_testing: Mapping[str, str] | None = None,
    _source_code_state_for_testing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare and persist the complete locked input set."""

    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve(strict=False)
    run_config_path = run_config_path.expanduser().resolve()
    _assert_not_p0(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PreparationError(
            f"Prepared output is immutable and already non-empty: {output_dir}"
        )
    source_code = _validated_clean_source_code(
        _git_state()
        if _source_code_state_for_testing is None
        else _source_code_state_for_testing
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = _load_json(run_config_path)
    input_policy = run_config.get("input_policy", {})
    expected_truncations = input_policy.get(
        "expected_primary_truncation_counts"
    )
    if run_config.get("model", {}).get("model_id") != model_id:
        raise PreparationError("run config model_id does not match the tokenizer")
    if run_config.get("model", {}).get("tokenizer_revision") != model_revision:
        raise PreparationError(
            "run config tokenizer revision does not match the requested revision"
        )
    if input_policy.get("final_input_token_cap") != FINAL_INPUT_TOKEN_CAP:
        raise PreparationError("run config does not lock the 24K input cap")
    if (
        input_policy.get("overlength_policy")
        != "first_12000_plus_last_12000_token_ids"
    ):
        raise PreparationError(
            "run config does not lock exact token-space middle truncation"
        )
    if (
        input_policy.get("decode_and_retokenize_after_truncation")
        is not False
    ):
        raise PreparationError(
            "run config must forbid decoding/re-tokenizing truncated inputs"
        )
    if (
        input_policy.get("feature_prompt_stage")
        != "official_task_prompt_before_chat_wrapper_and_truncation"
    ):
        raise PreparationError(
            "run config does not lock the seven-feature extraction stage"
        )
    if input_policy.get("persist_prepared_token_ids") is not True:
        raise PreparationError("run config must persist finalized token IDs")
    if input_policy.get("token_id_hash") != TOKEN_HASH_ALGORITHM:
        raise PreparationError("run config token hash algorithm is not supported")
    if input_policy.get("no_chat_add_special_tokens") is not True:
        raise PreparationError(
            "run config must retain tokenizer special tokens for no-chat tasks"
        )
    if (
        input_policy.get("chat_template_date_string")
        != CHAT_TEMPLATE_DATE_STRING
    ):
        raise PreparationError(
            "run config chat-template date does not match the locked value"
        )

    status_path = output_dir / "status.json"
    _atomic_write_json(
        status_path,
        {
            "schema_version": PREPARED_SCHEMA,
            "status": "running",
            "started_at_utc": utc_now(),
        },
    )

    metadata_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    token_files: list[dict[str, Any]] = []
    truncation_counts: OrderedDict[str, int] = OrderedDict()
    pre_lengths: list[int] = []
    retained_tokens = 0
    total_pre_tokens = 0
    started_at = utc_now()
    pinned_task_hashes: dict[str, str] = source_manifest()["longbench_dataset"][
        "extracted_task_sha256"
    ]
    expected_task_hashes = (
        pinned_task_hashes
        if _expected_task_hashes_for_testing is None
        else dict(_expected_task_hashes_for_testing)
    )
    if set(expected_task_hashes) != set(TASK_SPECS):
        raise PreparationError(
            "expected source hashes do not cover the exact 16-task panel"
        )
    hashes_verified_against_pinned_release = (
        expected_task_hashes == pinned_task_hashes
    )
    try:
        for task_id, spec in TASK_SPECS.items():
            rows, source_metadata = load_strict_task_rows(
                data_dir,
                spec,
                expected_sha256=expected_task_hashes[task_id],
            )
            source_files.append(source_metadata)
            concatenated: list[int] = []
            offsets = [0]
            task_truncations = 0

            for source_index, row in enumerate(rows):
                prompt = format_task_prompt(task_id, row)
                features = extract_prompt_features_strict(prompt, tokenizer)
                input_ids, pre_count, truncated = finalize_input_ids(
                    tokenizer, spec, prompt
                )
                input_hash = token_ids_sha256(input_ids)
                concatenated.extend(input_ids)
                offsets.append(len(concatenated))
                task_truncations += int(truncated)
                pre_lengths.append(pre_count)
                total_pre_tokens += pre_count
                retained_tokens += len(input_ids)

                metadata_rows.append(
                    {
                        "schema_version": PREPARED_SCHEMA,
                        "model": run_config["model"]["alias"],
                        "model_id": model_id,
                        "task": task_id,
                        "category": spec.category,
                        "benchmark_id": row["_id"],
                        "source_index": source_index,
                        "metric": spec.scorer,
                        "references": row["answers"],
                        "all_classes": row["all_classes"],
                        "features": features,
                        "pre_truncation_token_count": pre_count,
                        "post_truncation_token_count": len(input_ids),
                        "truncated": truncated,
                        "final_input_token_sha256": input_hash,
                        "token_hash_algorithm": TOKEN_HASH_ALGORITHM,
                        "max_new_tokens": spec.max_new_tokens,
                        "minimum_new_tokens": spec.minimum_new_tokens,
                        "stop_on_newline_token": spec.stop_on_newline_token,
                        "token_file": f"tokens/{task_id}.npz",
                        "token_offset_index": source_index,
                    }
                )

            truncation_counts[task_id] = task_truncations
            token_path = output_dir / "tokens" / f"{task_id}.npz"
            _atomic_save_npz(
                token_path,
                input_ids=np.asarray(concatenated, dtype="<u4"),
                offsets=np.asarray(offsets, dtype="<u8"),
            )
            token_files.append(
                {
                    "task": task_id,
                    "path": str(token_path.relative_to(output_dir)),
                    "sha256": sha256_file(token_path),
                    "rows": len(rows),
                    "stored_token_ids": len(concatenated),
                }
            )

        if len(metadata_rows) != EXPECTED_TOTAL_EXAMPLES:
            raise PreparationError(
                f"Prepared {len(metadata_rows)} rows; expected "
                f"{EXPECTED_TOTAL_EXAMPLES}"
            )
        composite_keys = [
            (record["task"], record["benchmark_id"])
            for record in metadata_rows
        ]
        if len(set(composite_keys)) != EXPECTED_TOTAL_EXAMPLES:
            raise PreparationError("Prepared composite keys are not unique")
        if expected_truncations is not None and dict(truncation_counts) != dict(
            expected_truncations
        ):
            raise PreparationError(
                "Primary-model truncation audit mismatch: "
                f"{dict(truncation_counts)} != {expected_truncations}"
            )
        expected_total_truncated = input_policy.get(
            "expected_primary_total_truncated"
        )
        if (
            expected_total_truncated is not None
            and sum(truncation_counts.values()) != expected_total_truncated
        ):
            raise PreparationError(
                "Primary-model total truncation count does not match the run config"
            )

        index_path = output_dir / "index.jsonl"
        _atomic_write_text(
            index_path,
            "".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for record in metadata_rows
            ),
        )

        percentiles = {
            percentile: float(np.percentile(pre_lengths, float(percentile)))
            for percentile in ("50", "90", "95", "99")
        }
        manifest = {
            "schema_version": PREPARED_SCHEMA,
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "benchmark": {
                "requested_dataset_id": LONG_BENCH_DATASET_ID,
                "resolved_dataset_id": LONG_BENCH_RESOLVED_DATASET_ID,
                "dataset_revision": LONG_BENCH_DATASET_REVISION,
                "split": LONG_BENCH_SPLIT,
                "source_hashes_verified_against_pinned_release": (
                    hashes_verified_against_pinned_release
                ),
                "source_files": source_files,
            },
            "tokenizer": _tokenizer_metadata(
                tokenizer, model_id=model_id, revision=model_revision
            ),
            "input_policy": {
                "cap": FINAL_INPUT_TOKEN_CAP,
                "middle_truncation_tokens_per_side": (
                    MIDDLE_TRUNCATION_TOKENS_PER_SIDE
                ),
                "token_hash_algorithm": TOKEN_HASH_ALGORITHM,
                "decode_and_retokenize_after_truncation": False,
                "no_chat_add_special_tokens": True,
                "chat_template_date_string": CHAT_TEMPLATE_DATE_STRING,
            },
            "records": len(metadata_rows),
            "truncation_counts": dict(truncation_counts),
            "total_truncated": sum(truncation_counts.values()),
            "pre_truncation_length_percentiles": percentiles,
            "maximum_pre_truncation_tokens": max(pre_lengths),
            "aggregate_tokens_retained_fraction": (
                retained_tokens / total_pre_tokens
            ),
            "prepared_index": {
                "path": str(index_path.relative_to(output_dir)),
                "sha256": sha256_file(index_path),
                "rows": len(metadata_rows),
            },
            "token_files": token_files,
            "protocol_config_hash": protocol_config_hash(),
            "protocol_config_files": config_file_hashes(),
            "run_config": {
                "path": str(run_config_path),
                "sha256": sha256_file(run_config_path),
            },
            "source_code": source_code,
        }
        manifest_path = output_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_json(
            status_path,
            {
                "schema_version": PREPARED_SCHEMA,
                "status": "complete",
                "completed_at_utc": manifest["completed_at_utc"],
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
        return manifest
    except BaseException as exc:
        _atomic_write_json(
            status_path,
            {
                "schema_version": PREPARED_SCHEMA,
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument(
        "--model-id", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--model-revision",
        default="0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow tokenizer network access (default requires the pinned cache)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "ADAPTIVESERVE_LB_N" in os.environ:
        raise SystemExit(
            "ADAPTIVESERVE_LB_N is forbidden by the final 3,750-example protocol"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        local_files_only=not args.allow_download,
    )
    manifest = prepare_inputs(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_config_path=args.run_config,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    print(
        f"Prepared {manifest['records']} inputs at {args.output_dir} "
        f"({manifest['total_truncated']} truncated)"
    )


if __name__ == "__main__":
    main()
