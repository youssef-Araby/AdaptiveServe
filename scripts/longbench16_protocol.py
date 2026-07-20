"""Strict, CPU-only protocol helpers for the 16-task LongBench-v1 panel.

The benchmark assets are pinned to THUDM/LongBench commit
``2e00731f8d0bff23dc4325161044d0ed8af94c1e`` and the dataset revision
``5e628be450b7e67fb7ae6e201bd6d8f7056f7672``.  This module deliberately has
no torch, transformers, datasets, or GPU dependency.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import re
import string
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "longbench_v1"

LONG_BENCH_CODE_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_DATASET_ID = "THUDM/LongBench"
LONG_BENCH_RESOLVED_DATASET_ID = "zai-org/LongBench"
LONG_BENCH_DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONG_BENCH_SPLIT = "test"

FINAL_INPUT_TOKEN_CAP = 24_000
MIDDLE_TRUNCATION_TOKENS_PER_SIDE = 12_000
EXPECTED_TOTAL_EXAMPLES = 3_750
EXPECTED_GENERATIONS_PER_MODEL = 22_500

CATEGORY_ORDER = (
    "single_document_qa",
    "multi_document_qa",
    "summarization",
    "few_shot_learning",
    "synthetic_tasks",
    "code_completion",
)

TASK_ORDER = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

EXCLUDED_CHINESE_TASKS = frozenset(
    {
        "multifieldqa_zh",
        "dureader",
        "vcsum",
        "lsht",
        "passage_retrieval_zh",
    }
)
CHAT_WRAPPER_EXCLUSIONS = frozenset(
    {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}
)
FIRST_LINE_POSTPROCESS_TASKS = frozenset({"trec", "triviaqa", "samsum"})

_SCORERS = frozenset(
    {
        "qa_f1_score",
        "rouge_score",
        "classification_score",
        "count_score",
        "retrieval_score",
        "code_sim_score",
    }
)
_CONFIG_FILES = (
    "official_dataset2prompt.json",
    "official_dataset2maxlen.json",
    "task_registry.json",
    "source_manifest.json",
)


class ProtocolConfigError(ValueError):
    """Raised when pinned protocol configuration is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One immutable LongBench task contract."""

    task_id: str
    display_name: str
    category: str
    language: str
    expected_test_examples: int
    scorer: str
    max_new_tokens: int
    apply_chat_wrapper: bool
    first_line_postprocess: bool
    minimum_new_tokens: int
    stop_on_newline_token: bool
    newline_stop_token_policy: str | None
    prompt_template: str


def _load_json(filename: str) -> Any:
    path = CONFIG_DIR / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolConfigError(f"Missing protocol config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolConfigError(f"Invalid JSON in protocol config: {path}") from exc


_OFFICIAL_PROMPTS: dict[str, str] = _load_json("official_dataset2prompt.json")
_OFFICIAL_MAX_NEW_TOKENS: dict[str, int] = _load_json(
    "official_dataset2maxlen.json"
)
_REGISTRY: dict[str, Any] = _load_json("task_registry.json")
_SOURCE_MANIFEST: dict[str, Any] = _load_json("source_manifest.json")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using a documented canonical UTF-8 serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_file_hashes() -> dict[str, str]:
    """Return exact-byte SHA-256 hashes for all local protocol configs."""

    return {name: sha256_file(CONFIG_DIR / name) for name in _CONFIG_FILES}


def protocol_config_hash() -> str:
    """Return one stable digest over the named local config-file digests."""

    return canonical_json_sha256(config_file_hashes())


def source_manifest() -> dict[str, Any]:
    """Return a defensive copy of pinned source provenance."""

    return copy.deepcopy(_SOURCE_MANIFEST)


def verify_config_integrity() -> dict[str, str]:
    """Verify all locally pinned assets declared by ``source_manifest.json``."""

    actual: dict[str, str] = {}
    local_assets = _SOURCE_MANIFEST.get("local_assets")
    if not isinstance(local_assets, dict):
        raise ProtocolConfigError("source_manifest.json lacks local_assets")
    for filename, metadata in local_assets.items():
        if not isinstance(metadata, dict) or "local_sha256" not in metadata:
            raise ProtocolConfigError(f"Missing local hash metadata for {filename}")
        observed = sha256_file(CONFIG_DIR / filename)
        expected = metadata["local_sha256"]
        if observed != expected:
            raise ProtocolConfigError(
                f"Config hash mismatch for {filename}: {observed} != {expected}"
            )
        actual[filename] = observed
    return actual


def _validate_and_build_specs() -> Mapping[str, TaskSpec]:
    if _REGISTRY.get("schema_version") != 1:
        raise ProtocolConfigError("Unsupported task_registry schema_version")
    if _REGISTRY.get("benchmark") != "LongBench v1":
        raise ProtocolConfigError("Registry benchmark must be LongBench v1")
    if _REGISTRY.get("dataset_id") != LONG_BENCH_DATASET_ID:
        raise ProtocolConfigError("Unexpected requested LongBench dataset ID")
    if _REGISTRY.get("resolved_dataset_id") != LONG_BENCH_RESOLVED_DATASET_ID:
        raise ProtocolConfigError("Unexpected resolved LongBench dataset ID")
    if _REGISTRY.get("dataset_revision") != LONG_BENCH_DATASET_REVISION:
        raise ProtocolConfigError("Unexpected LongBench dataset revision")
    if _REGISTRY.get("split") != LONG_BENCH_SPLIT:
        raise ProtocolConfigError("LongBench split must be test")
    if _REGISTRY.get("final_input_token_cap") != FINAL_INPUT_TOKEN_CAP:
        raise ProtocolConfigError("Final input cap must be exactly 24,000 tokens")
    if (
        _REGISTRY.get("middle_truncation_tokens_per_side")
        != MIDDLE_TRUNCATION_TOKENS_PER_SIDE
    ):
        raise ProtocolConfigError("Middle truncation must retain 12,000 tokens/side")
    if tuple(_REGISTRY.get("category_order", ())) != CATEGORY_ORDER:
        raise ProtocolConfigError("Category order does not match the protocol")

    if set(_OFFICIAL_PROMPTS) != set(_OFFICIAL_MAX_NEW_TOKENS):
        raise ProtocolConfigError("Official prompt and max-length task sets differ")
    if len(_OFFICIAL_PROMPTS) != 21:
        raise ProtocolConfigError("Pinned full LongBench-v1 config must have 21 tasks")
    if set(TASK_ORDER) | EXCLUDED_CHINESE_TASKS != set(_OFFICIAL_PROMPTS):
        raise ProtocolConfigError("16-task panel plus Chinese exclusions is not LongBench-v1")
    if set(TASK_ORDER) & EXCLUDED_CHINESE_TASKS:
        raise ProtocolConfigError("Chinese task leaked into the non-Chinese panel")

    rows = _REGISTRY.get("tasks")
    if not isinstance(rows, list):
        raise ProtocolConfigError("Registry tasks must be an ordered list")
    ids = tuple(row.get("task_id") for row in rows if isinstance(row, dict))
    if ids != TASK_ORDER:
        raise ProtocolConfigError("Registry task order/membership is not the locked panel")

    specs: OrderedDict[str, TaskSpec] = OrderedDict()
    for row in rows:
        task_id = row["task_id"]
        category = row["category"]
        scorer = row["scorer"]
        if category not in CATEGORY_ORDER:
            raise ProtocolConfigError(f"Unknown category for {task_id}: {category}")
        if scorer not in _SCORERS:
            raise ProtocolConfigError(f"Unknown scorer for {task_id}: {scorer}")
        if row["max_new_tokens"] != _OFFICIAL_MAX_NEW_TOKENS[task_id]:
            raise ProtocolConfigError(f"Official max_new_tokens mismatch for {task_id}")

        spec = TaskSpec(
            task_id=task_id,
            display_name=row["display_name"],
            category=category,
            language=row["language"],
            expected_test_examples=row["expected_test_examples"],
            scorer=scorer,
            max_new_tokens=row["max_new_tokens"],
            apply_chat_wrapper=row["apply_chat_wrapper"],
            first_line_postprocess=row["first_line_postprocess"],
            minimum_new_tokens=row["minimum_new_tokens"],
            stop_on_newline_token=row["stop_on_newline_token"],
            newline_stop_token_policy=row.get("newline_stop_token_policy"),
            prompt_template=_OFFICIAL_PROMPTS[task_id],
        )
        specs[task_id] = spec

    if sum(spec.expected_test_examples for spec in specs.values()) != EXPECTED_TOTAL_EXAMPLES:
        raise ProtocolConfigError("Expected task counts do not sum to 3,750")
    if {
        task_id for task_id, spec in specs.items() if not spec.apply_chat_wrapper
    } != CHAT_WRAPPER_EXCLUSIONS:
        raise ProtocolConfigError("Chat-wrapper exclusion set is not official")
    if {
        task_id for task_id, spec in specs.items() if spec.first_line_postprocess
    } != FIRST_LINE_POSTPROCESS_TASKS:
        raise ProtocolConfigError("First-line post-processing set is not official")

    samsum = specs["samsum"]
    if (
        samsum.minimum_new_tokens != 1
        or not samsum.stop_on_newline_token
        or samsum.newline_stop_token_policy
        != 'tokenizer.encode("\\n", add_special_tokens=False)[-1]'
    ):
        raise ProtocolConfigError("SAMSum generation override is incomplete")
    for task_id, spec in specs.items():
        if task_id != "samsum" and (
            spec.minimum_new_tokens != 0
            or spec.stop_on_newline_token
            or spec.newline_stop_token_policy is not None
        ):
            raise ProtocolConfigError(f"Unexpected generation override for {task_id}")

    code_manifest = _SOURCE_MANIFEST.get("longbench_code", {})
    dataset_manifest = _SOURCE_MANIFEST.get("longbench_dataset", {})
    if code_manifest.get("commit") != LONG_BENCH_CODE_COMMIT:
        raise ProtocolConfigError("Official LongBench code commit is not pinned")
    if dataset_manifest.get("requested_id") != LONG_BENCH_DATASET_ID:
        raise ProtocolConfigError("Dataset requested ID provenance mismatch")
    if dataset_manifest.get("resolved_id") != LONG_BENCH_RESOLVED_DATASET_ID:
        raise ProtocolConfigError("Dataset resolved ID provenance mismatch")
    if dataset_manifest.get("revision") != LONG_BENCH_DATASET_REVISION:
        raise ProtocolConfigError("Official LongBench dataset revision is not pinned")
    dataset_assets = dataset_manifest.get("assets", {})
    data_zip = dataset_assets.get("data.zip", {})
    if data_zip.get("raw_upstream_sha256") != (
        "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
    ):
        raise ProtocolConfigError("Official LongBench data.zip hash is not pinned")
    extracted_hashes = dataset_manifest.get("extracted_task_sha256")
    if not isinstance(extracted_hashes, dict) or set(extracted_hashes) != set(
        TASK_ORDER
    ):
        raise ProtocolConfigError(
            "Pinned extracted source hashes must cover the exact 16-task panel"
        )
    if any(
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in extracted_hashes.values()
    ):
        raise ProtocolConfigError("Pinned extracted source hashes are invalid")

    return MappingProxyType(specs)


verify_config_integrity()
TASK_SPECS: Mapping[str, TaskSpec] = _validate_and_build_specs()
CATEGORY_TO_TASKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        category: tuple(
            task_id
            for task_id, spec in TASK_SPECS.items()
            if spec.category == category
        )
        for category in CATEGORY_ORDER
    }
)
EXPECTED_CATEGORY_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        category: sum(
            TASK_SPECS[task_id].expected_test_examples
            for task_id in CATEGORY_TO_TASKS[category]
        )
        for category in CATEGORY_ORDER
    }
)


def _get_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        raise KeyError(
            f"Unknown LongBench-v1 non-Chinese task {task!r}; "
            f"expected one of {tuple(TASK_SPECS)}"
        ) from exc


def format_task_prompt(task: str, row: Mapping[str, Any]) -> str:
    """Format one record with the exact pinned official task template."""

    spec = _get_spec(task)
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    required_fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(spec.prompt_template)
        if field_name is not None
    }
    missing = required_fields.difference(row)
    if missing:
        raise KeyError(f"{task} row is missing prompt fields: {sorted(missing)}")
    non_strings = sorted(
        name for name in required_fields if not isinstance(row[name], str)
    )
    if non_strings:
        raise TypeError(f"{task} prompt fields must be strings: {non_strings}")
    return spec.prompt_template.format_map(row)


def postprocess_prediction(task: str, prediction: str) -> str:
    """Apply official ``eval.py`` prediction post-processing exactly."""

    spec = _get_spec(task)
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if spec.first_line_postprocess:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Official English token-level QA F1."""

    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


_ROUGE: Any = None


def rouge_score(prediction: str, ground_truth: str) -> float:
    """Official ROUGE-L F score via the LongBench ``rouge`` dependency."""

    global _ROUGE
    if _ROUGE is None:
        try:
            from rouge import Rouge
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Official LongBench ROUGE scoring requires the `rouge` package"
            ) from exc
        _ROUGE = Rouge()
    try:
        scores = _ROUGE.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])


def classification_score(
    prediction: str, ground_truth: str, all_classes: Sequence[str]
) -> float:
    """Official class-name substring score, including its ambiguity penalty."""

    em_match_list = [
        class_name for class_name in all_classes if class_name in prediction
    ]
    # Preserve the pinned official implementation's list-mutation semantics.
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def count_score(prediction: str, ground_truth: str) -> float:
    """Official fractional numeric score for PassageCount."""

    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(number == str(ground_truth) for number in numbers)
    return float(right / len(numbers))


def retrieval_score(prediction: str, ground_truth: str) -> float:
    """Official fractional paragraph-ID score for PassageRetrieval-en."""

    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        raise ValueError(
            "PassageRetrieval-en ground truth must contain `Paragraph <number>`"
        )
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(number == ground_truth_id for number in numbers)
    return float(right / len(numbers))


def _fuzzywuzzy_ratio(first: str, second: str) -> int:
    """Pure-Python ``fuzzywuzzy.fuzz.ratio`` semantics (0..100 integer)."""

    ratio = difflib.SequenceMatcher(None, first, second).ratio()
    return int(round(100 * ratio))


def code_sim_score(prediction: str, ground_truth: str) -> float:
    """Official first-valid-line code similarity used by LCC/RepoBench-P."""

    selected = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            selected = line
            break
    return _fuzzywuzzy_ratio(selected, ground_truth) / 100.0


def score_prediction(
    task: str,
    prediction: str,
    answers: Sequence[Any],
    all_classes: Sequence[str] | None = None,
) -> float:
    """Score one prediction, taking the maximum over official references."""

    spec = _get_spec(task)
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise TypeError("answers must be a non-string sequence")
    if not answers:
        raise ValueError("answers must contain at least one reference")
    processed = postprocess_prediction(task, prediction)

    score = 0.0
    for answer in answers:
        ground_truth = str(answer)
        if spec.scorer == "qa_f1_score":
            candidate = qa_f1_score(processed, ground_truth)
        elif spec.scorer == "rouge_score":
            candidate = rouge_score(processed, ground_truth)
        elif spec.scorer == "classification_score":
            if all_classes is None:
                raise ValueError(f"{task} requires all_classes")
            candidate = classification_score(processed, ground_truth, all_classes)
        elif spec.scorer == "count_score":
            candidate = count_score(processed, ground_truth)
        elif spec.scorer == "retrieval_score":
            candidate = retrieval_score(processed, ground_truth)
        elif spec.scorer == "code_sim_score":
            candidate = code_sim_score(processed, ground_truth)
        else:  # guarded by registry validation
            raise ProtocolConfigError(f"Unsupported scorer: {spec.scorer}")
        score = max(score, candidate)
    return float(score)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _harmonic_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    if any(value <= 0 for value in values):
        raise ValueError("compression ratios must be positive")
    return len(values) / math.fsum(1.0 / value for value in values)


def _validate_task_counts(
    values_by_task: Mapping[str, Sequence[float]], *, require_complete: bool
) -> OrderedDict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for task_id, spec in TASK_SPECS.items():
        count = len(values_by_task[task_id])
        if count == 0:
            raise ValueError(f"No records for required task {task_id}")
        if require_complete and count != spec.expected_test_examples:
            raise ValueError(
                f"{task_id} has {count} records; expected "
                f"{spec.expected_test_examples}"
            )
        counts[task_id] = count
    return counts


def aggregate_quality(
    per_record: Iterable[Mapping[str, Any]], *, require_complete: bool = True
) -> dict[str, Any]:
    """Aggregate scores as task means, category means, then six-category mean."""

    values_by_task: OrderedDict[str, list[float]] = OrderedDict(
        (task_id, []) for task_id in TASK_SPECS
    )
    for index, record in enumerate(per_record):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        task_id = record.get("task")
        if task_id not in TASK_SPECS:
            raise KeyError(f"record {index} has unknown task {task_id!r}")
        score = _finite_number(record.get("score"), field=f"record {index} score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"record {index} score must be in [0, 1]")
        values_by_task[task_id].append(score)

    counts = _validate_task_counts(
        values_by_task, require_complete=require_complete
    )
    task_scores: OrderedDict[str, float] = OrderedDict(
        (task_id, _mean(values_by_task[task_id])) for task_id in TASK_SPECS
    )
    category_scores: OrderedDict[str, float] = OrderedDict(
        (
            category,
            _mean([task_scores[task_id] for task_id in CATEGORY_TO_TASKS[category]]),
        )
        for category in CATEGORY_ORDER
    )
    all_scores = [
        score for task_values in values_by_task.values() for score in task_values
    ]
    return {
        "n_records": len(all_scores),
        "task_counts": dict(counts),
        "task_scores": dict(task_scores),
        "category_scores": dict(category_scores),
        "category_balanced_mean": _mean(list(category_scores.values())),
        "prompt_micro_mean": _mean(all_scores),
    }


def aggregate_compression(
    per_record: Iterable[Mapping[str, Any]], *, require_complete: bool = True
) -> dict[str, Any]:
    """Aggregate measured KV compression with primary and category summaries."""

    ratios_by_task: OrderedDict[str, list[float]] = OrderedDict(
        (task_id, []) for task_id in TASK_SPECS
    )
    for index, record in enumerate(per_record):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        task_id = record.get("task")
        if task_id not in TASK_SPECS:
            raise KeyError(f"record {index} has unknown task {task_id!r}")
        fp16_bytes = _finite_number(
            record.get("kv_bytes_fp16"), field=f"record {index} kv_bytes_fp16"
        )
        compressed_bytes = _finite_number(
            record.get("kv_bytes"), field=f"record {index} kv_bytes"
        )
        if fp16_bytes <= 0 or compressed_bytes <= 0:
            raise ValueError(f"record {index} KV byte fields must be positive")
        ratios_by_task[task_id].append(fp16_bytes / compressed_bytes)

    counts = _validate_task_counts(
        ratios_by_task, require_complete=require_complete
    )
    task_hmeans: OrderedDict[str, float] = OrderedDict(
        (task_id, _harmonic_mean(ratios_by_task[task_id]))
        for task_id in TASK_SPECS
    )
    category_means: OrderedDict[str, float] = OrderedDict(
        (
            category,
            _mean(
                [
                    task_hmeans[task_id]
                    for task_id in CATEGORY_TO_TASKS[category]
                ]
            ),
        )
        for category in CATEGORY_ORDER
    )
    all_ratios = [
        ratio for task_values in ratios_by_task.values() for ratio in task_values
    ]
    return {
        "n_records": len(all_ratios),
        "task_counts": dict(counts),
        "task_harmonic_means": dict(task_hmeans),
        "category_task_balanced_means": dict(category_means),
        "harmonic_mean": _harmonic_mean(all_ratios),
        "category_balanced_mean": _mean(list(category_means.values())),
    }


def validate_protocol_config() -> None:
    """Re-run integrity and semantic validation; return only on success."""

    verify_config_integrity()
    _validate_and_build_specs()


__all__ = [
    "CATEGORY_ORDER",
    "CATEGORY_TO_TASKS",
    "CHAT_WRAPPER_EXCLUSIONS",
    "CONFIG_DIR",
    "EXCLUDED_CHINESE_TASKS",
    "EXPECTED_CATEGORY_COUNTS",
    "EXPECTED_GENERATIONS_PER_MODEL",
    "EXPECTED_TOTAL_EXAMPLES",
    "FINAL_INPUT_TOKEN_CAP",
    "FIRST_LINE_POSTPROCESS_TASKS",
    "LONG_BENCH_CODE_COMMIT",
    "LONG_BENCH_DATASET_ID",
    "LONG_BENCH_DATASET_REVISION",
    "LONG_BENCH_RESOLVED_DATASET_ID",
    "LONG_BENCH_SPLIT",
    "MIDDLE_TRUNCATION_TOKENS_PER_SIDE",
    "ProtocolConfigError",
    "TASK_ORDER",
    "TASK_SPECS",
    "TaskSpec",
    "aggregate_compression",
    "aggregate_quality",
    "canonical_json_sha256",
    "code_sim_score",
    "config_file_hashes",
    "count_score",
    "format_task_prompt",
    "postprocess_prediction",
    "protocol_config_hash",
    "qa_f1_score",
    "retrieval_score",
    "rouge_score",
    "score_prediction",
    "sha256_file",
    "source_manifest",
    "validate_protocol_config",
    "verify_config_integrity",
]
