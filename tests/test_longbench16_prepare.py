from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import longbench16_prepare as prepare
from scripts.longbench16_protocol import (
    EXPECTED_TOTAL_EXAMPLES,
    FINAL_INPUT_TOKEN_CAP,
    TASK_SPECS,
    config_file_hashes,
    format_task_prompt,
    protocol_config_hash,
)


MODEL_ID = "test/fake-longbench-model"
MODEL_REVISION = "0123456789abcdef"
LONG_CHAT_MARKER = "[[LONG_CHAT]]"
LONG_NO_CHAT_MARKER = "[[LONG_NO_CHAT]]"
FEATURE_KEYS = (
    "seq_len_tokens",
    "seq_len_chars",
    "token_entropy",
    "gzip_ratio",
    "unique_token_ratio",
    "question_position",
    "newline_density",
)
CLEAN_SOURCE_CODE = {"commit": "f" * 40, "dirty": False}


class _FakeTokenizer:
    """Deterministic CPU tokenizer with observable chat/no-chat boundaries."""

    name_or_path = MODEL_ID
    vocab_size = 1_000_000
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "fake-chat-template-v1"

    def __init__(self, *, chat_result: str = "list") -> None:
        self.chat_result = chat_result
        self.encode_call_count = 0
        self.encode_add_special_tokens: list[bool] = []
        self.chat_call_count = 0
        self.last_chat_call: dict[str, Any] | None = None
        self.marker_encode_counts: Counter[str] = Counter()
        self.marker_chat_counts: Counter[str] = Counter()

    @staticmethod
    def raw_ids(prompt: str) -> list[int]:
        if LONG_NO_CHAT_MARKER in prompt:
            return list(range(100_000, 124_005))
        if LONG_CHAT_MARKER in prompt:
            return list(range(200_000, 224_005))
        raw = prompt.encode("utf-8")
        checksum = int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")
        return [101, len(raw), 1_000 + checksum % 500_000, 102]

    @classmethod
    def chat_ids(cls, prompt: str) -> list[int]:
        return [800_000, *cls.raw_ids(prompt), 800_001]

    def encode(
        self, prompt: str, *, add_special_tokens: bool = False
    ) -> list[int]:
        self.encode_call_count += 1
        self.encode_add_special_tokens.append(add_special_tokens)
        for marker in (LONG_CHAT_MARKER, LONG_NO_CHAT_MARKER):
            if marker in prompt:
                self.marker_encode_counts[marker] += 1
        return self.raw_ids(prompt)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        date_string: str,
    ) -> Any:
        self.chat_call_count += 1
        self.last_chat_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "date_string": date_string,
        }
        prompt = messages[0]["content"]
        if LONG_CHAT_MARKER in prompt:
            self.marker_chat_counts[LONG_CHAT_MARKER] += 1
        ids = self.chat_ids(prompt)
        if self.chat_result == "list":
            return ids
        if self.chat_result == "mapping":
            return {"input_ids": ids}
        if self.chat_result == "attribute":
            return SimpleNamespace(input_ids=ids)
        raise AssertionError(f"unsupported fake chat result: {self.chat_result}")


@dataclass(frozen=True)
class _PreparedFixture:
    root: Path
    data_dir: Path
    output_dir: Path
    run_config_path: Path
    tokenizer: _FakeTokenizer
    manifest: dict[str, Any]


def _source_row(
    task_id: str,
    source_index: int,
    *,
    marker: str | None = None,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    context = f"Context for {task_id} row {source_index}."
    if marker is not None:
        context = f"{marker} {context}"
    return {
        "_id": benchmark_id or f"{task_id}:{source_index:04d}",
        "input": f"What is the answer for row {source_index}?",
        "context": context,
        "answers": [f"answer-{source_index}"],
        "all_classes": (
            ["description", "entity", "location", "number"]
            if task_id == "trec"
            else None
        ),
        "dataset": task_id,
        "language": TASK_SPECS[task_id].language,
        "length": len(context),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_complete_source_fixture(data_dir: Path) -> None:
    for task_id, spec in TASK_SPECS.items():
        rows = []
        for source_index in range(spec.expected_test_examples):
            marker = None
            if source_index == 0 and task_id == "narrativeqa":
                marker = LONG_CHAT_MARKER
            elif source_index == 0 and task_id == "trec":
                marker = LONG_NO_CHAT_MARKER
            rows.append(_source_row(task_id, source_index, marker=marker))
        _write_jsonl(data_dir / f"{task_id}.jsonl", rows)


def _expected_truncation_counts() -> dict[str, int]:
    return {
        task_id: int(task_id in {"narrativeqa", "trec"})
        for task_id in TASK_SPECS
    }


def _write_run_config(
    path: Path,
    *,
    truncation_counts: dict[str, int] | None = None,
    total_truncated: int | None = None,
) -> None:
    counts = (
        _expected_truncation_counts()
        if truncation_counts is None
        else truncation_counts
    )
    total = sum(counts.values()) if total_truncated is None else total_truncated
    path.write_text(
        json.dumps(
            {
                "model": {
                    "alias": "fake_model",
                    "model_id": MODEL_ID,
                    "tokenizer_revision": MODEL_REVISION,
                },
                "input_policy": {
                    "final_input_token_cap": FINAL_INPUT_TOKEN_CAP,
                    "overlength_policy": (
                        "first_12000_plus_last_12000_token_ids"
                    ),
                    "decode_and_retokenize_after_truncation": False,
                    "feature_prompt_stage": (
                        "official_task_prompt_before_chat_wrapper_and_truncation"
                    ),
                    "persist_prepared_token_ids": True,
                    "token_id_hash": prepare.TOKEN_HASH_ALGORITHM,
                    "no_chat_add_special_tokens": True,
                    "chat_template_date_string": (
                        prepare.CHAT_TEMPLATE_DATE_STRING
                    ),
                    "expected_primary_truncation_counts": counts,
                    "expected_primary_total_truncated": total,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_index(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (output_dir / "index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _fixture_task_hashes(data_dir: Path) -> dict[str, str]:
    return {
        task_id: prepare.sha256_file(data_dir / f"{task_id}.jsonl")
        for task_id in TASK_SPECS
    }


@pytest.fixture(scope="module")
def prepared_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> _PreparedFixture:
    root = tmp_path_factory.mktemp("longbench16-prepare")
    data_dir = root / "source"
    output_dir = root / "prepared"
    run_config_path = root / "run-config.json"
    _write_complete_source_fixture(data_dir)
    _write_run_config(run_config_path)
    tokenizer = _FakeTokenizer()
    manifest = prepare.prepare_inputs(
        tokenizer=tokenizer,
        data_dir=data_dir,
        output_dir=output_dir,
        run_config_path=run_config_path,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        _expected_task_hashes_for_testing=_fixture_task_hashes(data_dir),
        _source_code_state_for_testing=CLEAN_SOURCE_CODE,
    )
    return _PreparedFixture(
        root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        run_config_path=run_config_path,
        tokenizer=tokenizer,
        manifest=manifest,
    )


def test_middle_truncation_and_token_hash_are_exact() -> None:
    source = [0, 1, 2, 3, 4, 5, 6, 0xFFFFFFFF]
    finalized, truncated = prepare.middle_truncate_token_ids(
        source, cap=6, side=3
    )

    assert finalized == [0, 1, 2, 5, 6, 0xFFFFFFFF]
    assert truncated is True
    expected_bytes = np.asarray(finalized, dtype="<u4").tobytes(order="C")
    assert prepare.token_ids_sha256(finalized) == hashlib.sha256(
        expected_bytes
    ).hexdigest()

    at_cap, at_cap_truncated = prepare.middle_truncate_token_ids(
        source[:6], cap=6, side=3
    )
    assert at_cap == source[:6]
    assert at_cap_truncated is False

    with pytest.raises(ValueError, match="cap == 2"):
        prepare.middle_truncate_token_ids(source, cap=6, side=2)
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        prepare.middle_truncate_token_ids([-1], cap=2, side=1)
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        prepare.token_ids_sha256([-1])
    with pytest.raises(TypeError, match="integers"):
        prepare.token_ids_sha256([1.5])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="one-dimensional"):
        prepare.token_ids_sha256([[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_ordered_id_hash_is_length_prefixed_and_order_sensitive() -> None:
    benchmark_ids = ["a", "α", "bc"]
    expected = hashlib.sha256()
    for benchmark_id in benchmark_ids:
        encoded = benchmark_id.encode("utf-8")
        expected.update(len(encoded).to_bytes(8, "little"))
        expected.update(encoded)

    assert prepare.ordered_ids_sha256(benchmark_ids) == expected.hexdigest()
    assert prepare.ordered_ids_sha256(benchmark_ids) != (
        prepare.ordered_ids_sha256(list(reversed(benchmark_ids)))
    )
    assert prepare.ordered_ids_sha256(["a", "bc"]) != (
        prepare.ordered_ids_sha256(["ab", "c"])
    )


def test_strict_source_loader_enforces_exact_count_and_unique_ids(
    tmp_path: Path,
) -> None:
    spec = replace(TASK_SPECS["qasper"], expected_test_examples=2)
    rows = [
        _source_row("qasper", 0, benchmark_id="paper-α"),
        _source_row("qasper", 1, benchmark_id="paper-β"),
    ]
    _write_jsonl(tmp_path / "qasper.jsonl", rows)

    loaded, metadata = prepare.load_strict_task_rows(tmp_path, spec)

    assert loaded == rows
    assert metadata == {
        "task": "qasper",
        "path": str(tmp_path / "qasper.jsonl"),
        "rows": 2,
        "sha256": prepare.sha256_file(tmp_path / "qasper.jsonl"),
        "ordered_benchmark_ids_sha256": prepare.ordered_ids_sha256(
            ["paper-α", "paper-β"]
        ),
        "ordered_id_hash_algorithm": prepare.ORDERED_ID_HASH_ALGORITHM,
    }
    with pytest.raises(prepare.PreparationError, match="source hash mismatch"):
        prepare.load_strict_task_rows(
            tmp_path,
            spec,
            expected_sha256="0" * 64,
        )

    _write_jsonl(tmp_path / "qasper.jsonl", rows[:1])
    with pytest.raises(prepare.PreparationError, match="has 1 rows; expected 2"):
        prepare.load_strict_task_rows(tmp_path, spec)

    duplicate_rows = [
        rows[0],
        _source_row("qasper", 1, benchmark_id="paper-α"),
    ]
    _write_jsonl(tmp_path / "qasper.jsonl", duplicate_rows)
    with pytest.raises(prepare.PreparationError, match="duplicate benchmark IDs"):
        prepare.load_strict_task_rows(tmp_path, spec)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row.pop("_id"), "lacks"),
        (lambda row: row.__setitem__("_id", ""), "invalid _id"),
        (
            lambda row: row.__setitem__("dataset", "narrativeqa"),
            "reports dataset",
        ),
        (lambda row: row.__setitem__("language", ""), "invalid language"),
        (lambda row: row.__setitem__("language", "zh"), "not an English"),
        (lambda row: row.__setitem__("length", True), "invalid length"),
        (lambda row: row.__setitem__("context", 7), "non-string prompt"),
        (lambda row: row.__setitem__("answers", []), "no reference answers"),
        (
            lambda row: row.__setitem__("answers", [7]),
            "invalid reference answers",
        ),
        (
            lambda row: row.__setitem__("all_classes", "not-a-list"),
            "invalid all_classes",
        ),
    ],
)
def test_strict_source_loader_rejects_malformed_rows(
    tmp_path: Path,
    mutation: Any,
    error: str,
) -> None:
    spec = replace(TASK_SPECS["qasper"], expected_test_examples=1)
    row = _source_row("qasper", 0)
    mutation(row)
    _write_jsonl(tmp_path / "qasper.jsonl", [row])

    with pytest.raises(prepare.PreparationError, match=error):
        prepare.load_strict_task_rows(tmp_path, spec)


def test_seven_feature_contract_is_exact_and_pre_chat() -> None:
    class FeatureTokenizer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def encode(
            self, prompt: str, *, add_special_tokens: bool
        ) -> list[int]:
            self.calls.append((prompt, add_special_tokens))
            return [1, 1, 2, 3]

    tokenizer = FeatureTokenizer()
    prompt = "ab?\n"
    features = prepare.extract_prompt_features_strict(prompt, tokenizer)

    assert tuple(features) == FEATURE_KEYS
    assert features == {
        "seq_len_tokens": 4,
        "seq_len_chars": 4,
        "token_entropy": 1.5,
        "gzip_ratio": round(
            len(gzip.compress(prompt.encode("utf-8"), compresslevel=6)) / 4,
            4,
        ),
        "unique_token_ratio": 0.75,
        "question_position": 0.5,
        "newline_density": 0.25,
    }
    assert tokenizer.calls == [(prompt, False)]

    no_question = prepare.extract_prompt_features_strict("plain", tokenizer)
    assert no_question["question_position"] is None

    class InvalidTokenizer:
        @staticmethod
        def encode(prompt: str, *, add_special_tokens: bool) -> tuple[int, ...]:
            return (1, 2)

    with pytest.raises(prepare.PreparationError, match="flat integer list"):
        prepare.extract_prompt_features_strict("prompt", InvalidTokenizer())


@pytest.mark.parametrize("chat_result", ["list", "mapping", "attribute"])
def test_finalize_input_ids_uses_chat_template_exactly(
    chat_result: str,
) -> None:
    tokenizer = _FakeTokenizer(chat_result=chat_result)
    prompt = "A chat-wrapped prompt?"

    ids, pre_count, truncated = prepare.finalize_input_ids(
        tokenizer, TASK_SPECS["narrativeqa"], prompt
    )

    assert ids == tokenizer.chat_ids(prompt)
    assert pre_count == len(ids)
    assert truncated is False
    assert tokenizer.encode_call_count == 0
    assert tokenizer.chat_call_count == 1
    assert tokenizer.last_chat_call == {
        "messages": [{"role": "user", "content": prompt}],
        "tokenize": True,
        "add_generation_prompt": True,
        "date_string": prepare.CHAT_TEMPLATE_DATE_STRING,
    }


def test_finalize_input_ids_uses_no_chat_encoding_with_special_tokens() -> None:
    tokenizer = _FakeTokenizer()
    prompt = "A no-chat prompt?"

    ids, pre_count, truncated = prepare.finalize_input_ids(
        tokenizer, TASK_SPECS["trec"], prompt
    )

    assert ids == tokenizer.raw_ids(prompt)
    assert pre_count == len(ids)
    assert truncated is False
    assert tokenizer.encode_call_count == 1
    assert tokenizer.encode_add_special_tokens == [True]
    assert tokenizer.chat_call_count == 0

    tokenizer.chat_template = None
    with pytest.raises(prepare.PreparationError, match="chat template"):
        prepare.finalize_input_ids(
            tokenizer, TASK_SPECS["narrativeqa"], prompt
        )


def test_full_preparation_has_exact_task_counts_ids_and_wrapper_calls(
    prepared_fixture: _PreparedFixture,
) -> None:
    case = prepared_fixture
    records = _read_index(case.output_dir)

    assert len(records) == EXPECTED_TOTAL_EXAMPLES == 3_750
    assert len(
        {(record["task"], record["benchmark_id"]) for record in records}
    ) == EXPECTED_TOTAL_EXAMPLES
    counts = Counter(record["task"] for record in records)
    assert counts == Counter(
        {
            task_id: spec.expected_test_examples
            for task_id, spec in TASK_SPECS.items()
        }
    )
    assert [record["task"] for record in records] == [
        task_id
        for task_id, spec in TASK_SPECS.items()
        for _ in range(spec.expected_test_examples)
    ]
    for task_id, spec in TASK_SPECS.items():
        task_records = [
            record for record in records if record["task"] == task_id
        ]
        assert [record["source_index"] for record in task_records] == list(
            range(spec.expected_test_examples)
        )
        assert [record["token_offset_index"] for record in task_records] == (
            list(range(spec.expected_test_examples))
        )
        assert [record["benchmark_id"] for record in task_records] == [
            f"{task_id}:{source_index:04d}"
            for source_index in range(spec.expected_test_examples)
        ]

    expected_chat_calls = sum(
        spec.expected_test_examples
        for spec in TASK_SPECS.values()
        if spec.apply_chat_wrapper
    )
    expected_no_chat_calls = EXPECTED_TOTAL_EXAMPLES - expected_chat_calls
    assert expected_chat_calls == 2_150
    assert expected_no_chat_calls == 1_600
    assert case.tokenizer.chat_call_count == expected_chat_calls
    assert case.tokenizer.encode_call_count == (
        EXPECTED_TOTAL_EXAMPLES + expected_no_chat_calls
    )
    assert Counter(case.tokenizer.encode_add_special_tokens) == {
        False: EXPECTED_TOTAL_EXAMPLES,
        True: expected_no_chat_calls,
    }
    assert case.tokenizer.marker_chat_counts == Counter(
        {LONG_CHAT_MARKER: 1}
    )
    assert case.tokenizer.marker_encode_counts == Counter(
        {LONG_CHAT_MARKER: 1, LONG_NO_CHAT_MARKER: 2}
    )

    by_key = {
        (record["task"], record["source_index"]): record
        for record in records
    }
    chat_record = by_key[("narrativeqa", 1)]
    no_chat_record = by_key[("trec", 1)]
    assert chat_record["features"]["seq_len_tokens"] == 4
    assert chat_record["pre_truncation_token_count"] == 6
    assert no_chat_record["features"]["seq_len_tokens"] == 4
    assert no_chat_record["pre_truncation_token_count"] == 4
    assert all(
        set(record["features"]) == set(FEATURE_KEYS) for record in records
    )


def test_prepared_npz_round_trip_preserves_every_finalized_input(
    prepared_fixture: _PreparedFixture,
) -> None:
    case = prepared_fixture
    records = _read_index(case.output_dir)

    for task_id, spec in TASK_SPECS.items():
        task_records = [
            record for record in records if record["task"] == task_id
        ]
        token_path = case.output_dir / "tokens" / f"{task_id}.npz"
        with np.load(token_path, allow_pickle=False) as archive:
            assert set(archive.files) == {"input_ids", "offsets"}
            input_ids = archive["input_ids"]
            offsets = archive["offsets"]

        assert input_ids.dtype == np.dtype("<u4")
        assert offsets.dtype == np.dtype("<u8")
        assert offsets.shape == (spec.expected_test_examples + 1,)
        assert offsets[0] == 0
        assert offsets[-1] == input_ids.size
        assert np.all(offsets[1:] >= offsets[:-1])

        for source_index, record in enumerate(task_records):
            start = int(offsets[source_index])
            end = int(offsets[source_index + 1])
            round_tripped = input_ids[start:end]
            assert end - start == record["post_truncation_token_count"]
            assert prepare.token_ids_sha256(round_tripped) == (
                record["final_input_token_sha256"]
            )
            assert record["token_file"] == f"tokens/{task_id}.npz"

    by_key = {
        (record["task"], record["source_index"]): record
        for record in records
    }
    for task_id, raw_ids in (
        (
            "narrativeqa",
            _FakeTokenizer.chat_ids(
                format_task_prompt(
                    "narrativeqa",
                    _source_row(
                        "narrativeqa", 0, marker=LONG_CHAT_MARKER
                    ),
                )
            ),
        ),
        (
            "trec",
            _FakeTokenizer.raw_ids(
                format_task_prompt(
                    "trec",
                    _source_row("trec", 0, marker=LONG_NO_CHAT_MARKER),
                )
            ),
        ),
    ):
        expected_ids, expected_truncated = prepare.middle_truncate_token_ids(
            raw_ids
        )
        record = by_key[(task_id, 0)]
        assert expected_truncated is True
        assert record["pre_truncation_token_count"] == len(raw_ids)
        assert record["post_truncation_token_count"] == (
            FINAL_INPUT_TOKEN_CAP
        )
        assert record["truncated"] is True
        assert record["final_input_token_sha256"] == (
            prepare.token_ids_sha256(expected_ids)
        )

        with np.load(
            case.output_dir / record["token_file"], allow_pickle=False
        ) as archive:
            offsets = archive["offsets"]
            stored = archive["input_ids"][
                int(offsets[0]) : int(offsets[1])
            ]
        assert stored.tolist() == expected_ids


def test_manifest_status_and_truncation_audit_are_self_consistent(
    prepared_fixture: _PreparedFixture,
) -> None:
    case = prepared_fixture
    records = _read_index(case.output_dir)
    manifest_path = case.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(
        (case.output_dir / "status.json").read_text(encoding="utf-8")
    )

    assert manifest == case.manifest
    assert manifest["schema_version"] == prepare.PREPARED_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["records"] == EXPECTED_TOTAL_EXAMPLES
    assert manifest["truncation_counts"] == _expected_truncation_counts()
    assert manifest["total_truncated"] == 2
    assert status == {
        "schema_version": prepare.PREPARED_SCHEMA,
        "status": "complete",
        "completed_at_utc": manifest["completed_at_utc"],
        "manifest_sha256": prepare.sha256_file(manifest_path),
    }
    assert manifest["input_policy"] == {
        "cap": FINAL_INPUT_TOKEN_CAP,
        "middle_truncation_tokens_per_side": 12_000,
        "token_hash_algorithm": prepare.TOKEN_HASH_ALGORITHM,
        "decode_and_retokenize_after_truncation": False,
        "no_chat_add_special_tokens": True,
        "chat_template_date_string": prepare.CHAT_TEMPLATE_DATE_STRING,
    }
    assert manifest["prepared_index"] == {
        "path": "index.jsonl",
        "sha256": prepare.sha256_file(case.output_dir / "index.jsonl"),
        "rows": EXPECTED_TOTAL_EXAMPLES,
    }
    assert manifest["protocol_config_hash"] == protocol_config_hash()
    assert manifest["protocol_config_files"] == config_file_hashes()
    assert manifest["run_config"] == {
        "path": str(case.run_config_path.resolve()),
        "sha256": prepare.sha256_file(case.run_config_path),
    }

    source_files = manifest["benchmark"]["source_files"]
    token_files = manifest["token_files"]
    assert [item["task"] for item in source_files] == list(TASK_SPECS)
    assert [item["task"] for item in token_files] == list(TASK_SPECS)
    for source_metadata, token_metadata in zip(
        source_files, token_files, strict=True
    ):
        task_id = source_metadata["task"]
        spec = TASK_SPECS[task_id]
        source_path = case.data_dir / f"{task_id}.jsonl"
        token_path = case.output_dir / "tokens" / f"{task_id}.npz"
        expected_ids = [
            f"{task_id}:{source_index:04d}"
            for source_index in range(spec.expected_test_examples)
        ]
        assert source_metadata["rows"] == spec.expected_test_examples
        assert source_metadata["sha256"] == prepare.sha256_file(source_path)
        assert source_metadata["ordered_benchmark_ids_sha256"] == (
            prepare.ordered_ids_sha256(expected_ids)
        )
        assert token_metadata["rows"] == spec.expected_test_examples
        assert token_metadata["path"] == f"tokens/{task_id}.npz"
        assert token_metadata["sha256"] == prepare.sha256_file(token_path)
        with np.load(token_path, allow_pickle=False) as archive:
            assert token_metadata["stored_token_ids"] == int(
                archive["input_ids"].size
            )

    pre_lengths = [record["pre_truncation_token_count"] for record in records]
    retained = sum(
        record["post_truncation_token_count"] for record in records
    )
    assert manifest["maximum_pre_truncation_tokens"] == max(pre_lengths)
    assert manifest["aggregate_tokens_retained_fraction"] == pytest.approx(
        retained / sum(pre_lengths)
    )
    assert manifest["pre_truncation_length_percentiles"] == {
        percentile: float(np.percentile(pre_lengths, float(percentile)))
        for percentile in ("50", "90", "95", "99")
    }


def test_prepared_output_is_immutable(
    prepared_fixture: _PreparedFixture,
) -> None:
    case = prepared_fixture
    before = {
        str(path.relative_to(case.output_dir)): prepare.sha256_file(path)
        for path in case.output_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(prepare.PreparationError, match="immutable"):
        prepare.prepare_inputs(
            tokenizer=_FakeTokenizer(),
            data_dir=case.data_dir,
            output_dir=case.output_dir,
            run_config_path=case.run_config_path,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )

    after = {
        str(path.relative_to(case.output_dir)): prepare.sha256_file(path)
        for path in case.output_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "source_code",
    (
        {"commit": "f" * 40, "dirty": True},
        {"commit": "unknown", "dirty": None},
    ),
)
def test_prepare_rejects_dirty_or_unknown_source_before_writing(
    tmp_path: Path,
    prepared_fixture: _PreparedFixture,
    source_code: dict[str, Any],
) -> None:
    output_dir = tmp_path / "unclean-source"

    with pytest.raises(prepare.PreparationError, match="canonical preparation"):
        prepare.prepare_inputs(
            tokenizer=_FakeTokenizer(),
            data_dir=prepared_fixture.data_dir,
            output_dir=output_dir,
            run_config_path=prepared_fixture.run_config_path,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            _source_code_state_for_testing=source_code,
        )

    assert not output_dir.exists()


def test_truncation_audit_mismatch_fails_closed_with_status(
    prepared_fixture: _PreparedFixture,
) -> None:
    case = prepared_fixture
    output_dir = case.root / "bad-truncation-audit"
    run_config_path = case.root / "bad-truncation-run-config.json"
    wrong_counts = _expected_truncation_counts()
    wrong_counts["narrativeqa"] = 0
    _write_run_config(
        run_config_path,
        truncation_counts=wrong_counts,
        total_truncated=1,
    )

    with pytest.raises(
        prepare.PreparationError, match="truncation audit mismatch"
    ):
        prepare.prepare_inputs(
            tokenizer=_FakeTokenizer(),
            data_dir=case.data_dir,
            output_dir=output_dir,
            run_config_path=run_config_path,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            _expected_task_hashes_for_testing=_fixture_task_hashes(
                case.data_dir
            ),
            _source_code_state_for_testing=CLEAN_SOURCE_CODE,
        )

    status = json.loads(
        (output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["error_type"] == "PreparationError"
    assert "truncation audit mismatch" in status["error"]
    assert not (output_dir / "manifest.json").exists()


def test_prepare_rejects_direct_and_symlinked_p0_outputs(
    tmp_path: Path,
    prepared_fixture: _PreparedFixture,
) -> None:
    protected_root = prepare.REPO_ROOT / "runs" / "p0"
    direct_output = protected_root / f"pytest-{tmp_path.name}"

    with pytest.raises(prepare.PreparationError, match="protected"):
        prepare.prepare_inputs(
            tokenizer=_FakeTokenizer(),
            data_dir=prepared_fixture.data_dir,
            output_dir=direct_output,
            run_config_path=prepared_fixture.run_config_path,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )
    assert not direct_output.exists()

    alias = tmp_path / "p0-alias"
    alias.symlink_to(protected_root, target_is_directory=True)
    with pytest.raises(prepare.PreparationError, match="protected"):
        prepare._assert_not_p0(alias / "nested")
