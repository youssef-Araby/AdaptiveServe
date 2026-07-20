from __future__ import annotations

import json
import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import longbench16_protocol as protocol


EXPECTED_COUNTS = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "passage_count": 200,
    "passage_retrieval_en": 200,
    "lcc": 500,
    "repobench-p": 500,
}

EXPECTED_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

EXPECTED_SCORERS = {
    "narrativeqa": "qa_f1_score",
    "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score",
    "hotpotqa": "qa_f1_score",
    "2wikimqa": "qa_f1_score",
    "musique": "qa_f1_score",
    "gov_report": "rouge_score",
    "qmsum": "rouge_score",
    "multi_news": "rouge_score",
    "trec": "classification_score",
    "triviaqa": "qa_f1_score",
    "samsum": "rouge_score",
    "passage_count": "count_score",
    "passage_retrieval_en": "retrieval_score",
    "lcc": "code_sim_score",
    "repobench-p": "code_sim_score",
}


def _full_quality_records(category_values: dict[str, float]):
    for task_id, spec in protocol.TASK_SPECS.items():
        for _ in range(spec.expected_test_examples):
            yield {"task": task_id, "score": category_values[spec.category]}


def _full_compression_records(category_ratios: dict[str, int]):
    for task_id, spec in protocol.TASK_SPECS.items():
        ratio = category_ratios[spec.category]
        for _ in range(spec.expected_test_examples):
            yield {
                "task": task_id,
                "kv_bytes_fp16": ratio * 1_000,
                "kv_bytes": 1_000,
            }


def test_locked_panel_order_counts_limits_and_scorers():
    assert tuple(protocol.TASK_SPECS) == protocol.TASK_ORDER
    assert len(protocol.TASK_SPECS) == 16
    assert {
        task: spec.expected_test_examples
        for task, spec in protocol.TASK_SPECS.items()
    } == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS.values()) == protocol.EXPECTED_TOTAL_EXAMPLES == 3_750
    assert protocol.EXPECTED_GENERATIONS_PER_MODEL == 22_500
    assert {
        task: spec.max_new_tokens for task, spec in protocol.TASK_SPECS.items()
    } == EXPECTED_MAX_NEW_TOKENS
    assert {
        task: spec.scorer for task, spec in protocol.TASK_SPECS.items()
    } == EXPECTED_SCORERS


def test_categories_and_category_counts_are_locked():
    assert protocol.CATEGORY_ORDER == (
        "single_document_qa",
        "multi_document_qa",
        "summarization",
        "few_shot_learning",
        "synthetic_tasks",
        "code_completion",
    )
    assert dict(protocol.EXPECTED_CATEGORY_COUNTS) == {
        "single_document_qa": 550,
        "multi_document_qa": 600,
        "summarization": 600,
        "few_shot_learning": 600,
        "synthetic_tasks": 400,
        "code_completion": 1_000,
    }
    assert set(protocol.TASK_SPECS).isdisjoint(protocol.EXCLUDED_CHINESE_TASKS)


def test_task_specs_are_immutable():
    with pytest.raises(TypeError):
        protocol.TASK_SPECS["extra"] = protocol.TASK_SPECS["qasper"]
    with pytest.raises(FrozenInstanceError):
        protocol.TASK_SPECS["qasper"].max_new_tokens = 999


def test_chat_wrapper_and_first_line_sets_are_exact():
    assert protocol.CHAT_WRAPPER_EXCLUSIONS == {
        "trec",
        "triviaqa",
        "samsum",
        "lcc",
        "repobench-p",
    }
    assert protocol.FIRST_LINE_POSTPROCESS_TASKS == {
        "trec",
        "triviaqa",
        "samsum",
    }
    assert {
        task for task, spec in protocol.TASK_SPECS.items() if not spec.apply_chat_wrapper
    } == protocol.CHAT_WRAPPER_EXCLUSIONS


def test_samsum_generation_override_is_explicit():
    spec = protocol.TASK_SPECS["samsum"]
    assert spec.minimum_new_tokens == 1
    assert spec.stop_on_newline_token is True
    assert (
        spec.newline_stop_token_policy
        == 'tokenizer.encode("\\n", add_special_tokens=False)[-1]'
    )
    assert spec.first_line_postprocess is True
    assert spec.apply_chat_wrapper is False


def test_official_prompt_formatting_preserves_upstream_text():
    prompt = protocol.format_task_prompt(
        "narrativeqa", {"context": "STORY", "input": "QUESTION"}
    )
    assert "asconcisely" in prompt  # typo is present in the pinned official file
    assert "Story: STORY" in prompt
    assert "Question: QUESTION" in prompt

    lcc = protocol.format_task_prompt(
        "lcc", {"context": "def f():\n", "input": "IGNORED"}
    )
    assert lcc == "Please complete the code given below. \ndef f():\nNext line of code:\n"
    repobench = protocol.format_task_prompt(
        "repobench-p", {"context": "FILE\n", "input": "TARGET\n"}
    )
    assert repobench == (
        "Please complete the code given below. \n"
        "FILE\nTARGET\nNext line of code:\n"
    )


def test_prompt_formatting_rejects_missing_or_non_string_fields():
    with pytest.raises(KeyError, match="input"):
        protocol.format_task_prompt("qasper", {"context": "paper"})
    with pytest.raises(TypeError, match="strings"):
        protocol.format_task_prompt("qasper", {"context": "paper", "input": 7})
    with pytest.raises(KeyError, match="Unknown"):
        protocol.format_task_prompt("multifieldqa_zh", {"context": "x", "input": "y"})


@pytest.mark.parametrize("task", ["trec", "triviaqa", "samsum"])
def test_official_first_line_postprocessing(task):
    assert protocol.postprocess_prediction(task, "\n\nanswer\nignored") == "answer"


def test_non_first_line_task_preserves_prediction_verbatim():
    prediction = "\nanswer\nsecond line "
    assert protocol.postprocess_prediction("qasper", prediction) == prediction


def test_qa_f1_normalization_and_max_over_references():
    assert protocol.qa_f1_score("The Eiffel, Tower!", "eiffel tower") == 1.0
    assert (
        protocol.score_prediction(
            "triviaqa",
            "\nParis\nextra",
            ["London", "Paris"],
        )
        == 1.0
    )


def test_rouge_scorer_is_official_rouge_l():
    assert protocol.score_prediction("samsum", "same words", ["same words"]) == pytest.approx(
        0.999999995
    )
    assert protocol.rouge_score("", "nonempty") == 0.0


def test_classification_scorer_and_required_classes():
    assert (
        protocol.score_prediction(
            "trec",
            "A and B\nignored",
            ["A"],
            all_classes=["A", "B"],
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="all_classes"):
        protocol.score_prediction("trec", "A", ["A"])


def test_count_retrieval_and_code_scorers():
    assert protocol.count_score("Maybe 3, 4, or 3.", "3") == pytest.approx(2 / 3)
    assert protocol.retrieval_score("Paragraph 2 or 3", "Paragraph 3") == 0.5
    assert (
        protocol.score_prediction(
            "lcc",
            "\n```python\n# explanation\nvalue = 1\n",
            ["value = 1"],
        )
        == 1.0
    )
    with pytest.raises(ValueError, match="Paragraph"):
        protocol.retrieval_score("3", "3")


def test_score_rejects_invalid_answer_container():
    with pytest.raises(TypeError, match="non-string sequence"):
        protocol.score_prediction("qasper", "answer", "answer")
    with pytest.raises(ValueError, match="at least one"):
        protocol.score_prediction("qasper", "answer", [])


def test_quality_aggregation_is_equal_weight_over_six_categories():
    category_values = {
        category: (index + 1) / 10
        for index, category in enumerate(protocol.CATEGORY_ORDER)
    }
    result = protocol.aggregate_quality(_full_quality_records(category_values))
    assert result["n_records"] == 3_750
    assert result["task_counts"] == EXPECTED_COUNTS
    assert result["category_scores"] == pytest.approx(category_values)
    assert result["category_balanced_mean"] == pytest.approx(0.35)
    expected_micro = sum(
        protocol.EXPECTED_CATEGORY_COUNTS[category] * value
        for category, value in category_values.items()
    ) / 3_750
    assert result["prompt_micro_mean"] == pytest.approx(expected_micro)
    assert result["prompt_micro_mean"] != pytest.approx(
        result["category_balanced_mean"]
    )


def test_compression_aggregation_has_both_required_summaries():
    category_ratios = {
        category: index + 1
        for index, category in enumerate(protocol.CATEGORY_ORDER)
    }
    result = protocol.aggregate_compression(
        _full_compression_records(category_ratios)
    )
    assert result["n_records"] == 3_750
    assert result["category_task_balanced_means"] == pytest.approx(category_ratios)
    assert result["category_balanced_mean"] == pytest.approx(3.5)
    expected_harmonic = 3_750 / math.fsum(
        protocol.EXPECTED_CATEGORY_COUNTS[category] / ratio
        for category, ratio in category_ratios.items()
    )
    assert result["harmonic_mean"] == pytest.approx(expected_harmonic)


def test_aggregators_fail_closed_on_incomplete_or_invalid_records():
    one_per_task = [
        {"task": task_id, "score": 0.5} for task_id in protocol.TASK_SPECS
    ]
    with pytest.raises(ValueError, match="expected"):
        protocol.aggregate_quality(one_per_task)
    result = protocol.aggregate_quality(one_per_task, require_complete=False)
    assert result["n_records"] == 16

    bad_compression = [
        {"task": task_id, "kv_bytes_fp16": 1, "kv_bytes": 1}
        for task_id in protocol.TASK_SPECS
    ]
    bad_compression[0]["kv_bytes"] = 0
    with pytest.raises(ValueError, match="positive"):
        protocol.aggregate_compression(bad_compression, require_complete=False)


def test_source_pins_and_raw_upstream_hashes_are_exact():
    manifest = protocol.source_manifest()
    assert manifest["longbench_code"]["commit"] == (
        "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
    )
    code_assets = manifest["longbench_code"]["assets"]
    assert code_assets["LongBench/config/dataset2prompt.json"][
        "raw_upstream_sha256"
    ] == "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"
    assert code_assets["LongBench/config/dataset2maxlen.json"][
        "raw_upstream_sha256"
    ] == "72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022"
    assert code_assets["LongBench/metrics.py"]["raw_upstream_sha256"] == (
        "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323"
    )
    assert code_assets["LongBench/eval.py"]["raw_upstream_sha256"] == (
        "1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721"
    )
    assert code_assets["LongBench/pred.py"]["raw_upstream_sha256"] == (
        "fdbb0ab2bb68c822fc35126ac050ec60110396bd14fb6d27bed1969f6361835c"
    )
    dataset = manifest["longbench_dataset"]
    assert dataset["requested_id"] == "THUDM/LongBench"
    assert dataset["resolved_id"] == "zai-org/LongBench"
    assert dataset["revision"] == "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
    assert dataset["split"] == "test"
    assert dataset["assets"]["data.zip"]["raw_upstream_sha256"] == (
        "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
    )
    assert set(dataset["extracted_task_sha256"]) == set(protocol.TASK_ORDER)


def test_local_config_hashes_and_combined_hash_are_stable():
    assert protocol.verify_config_integrity() == {
        "official_dataset2prompt.json": (
            "5c1231ee6c3f198b0021e9c1ab9b66cbbfb52c432e0ed821a5afe89eefa17fd8"
        ),
        "official_dataset2maxlen.json": (
            "75301d9cacf912e967c775997c94d9f021d176f940cb2d0318b99567220f52fb"
        ),
        "task_registry.json": (
            "59c2febef3571df380f998ea9ad1518c1e8258fcf4c770ce4afde5ab940958b2"
        ),
    }
    assert protocol.protocol_config_hash() == (
        "2b831635fea021356814c41931ebe63be6ed358c6ab910003763623ba1206d51"
    )
    assert protocol.canonical_json_sha256({"b": 2, "a": 1}) == (
        protocol.canonical_json_sha256({"a": 1, "b": 2})
    )
    protocol.validate_protocol_config()


def test_vendored_official_json_is_complete_and_valid():
    prompts = json.loads(
        (
            protocol.CONFIG_DIR / "official_dataset2prompt.json"
        ).read_text(encoding="utf-8")
    )
    max_lengths = json.loads(
        (
            protocol.CONFIG_DIR / "official_dataset2maxlen.json"
        ).read_text(encoding="utf-8")
    )
    assert len(prompts) == len(max_lengths) == 21
    assert set(prompts) == set(max_lengths)
    assert set(protocol.TASK_SPECS) | protocol.EXCLUDED_CHINESE_TASKS == set(prompts)
