from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import longbench16_c6 as c6
from scripts import longbench16_io as run_io
from scripts import longbench16_join as join
from scripts import longbench16_protocol as protocol


EXPECTED_COUNTS = {task: 2 for task in protocol.TASK_ORDER}
SCORES = {
    "C0": 0.80,
    "C1": 0.82,
    "C2": 0.83,
    "C3": 0.70,
    "C4": 0.60,
    "C5": 0.50,
}
RATIOS = {
    "C0": 1.0,
    "C1": 2.0,
    "C2": 4.0,
    "C3": 3.0,
    "C4": 5.0,
    "C5": 6.0,
}


def _row(
    task: str,
    source_index: int,
    *,
    ratios: dict[str, float] | None = None,
    scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    task_index = protocol.TASK_ORDER.index(task)
    benchmark_id = f"{task}:{source_index:04d}"
    spec = protocol.TASK_SPECS[task]
    selected_ratios = ratios or RATIOS
    selected_scores = scores or SCORES
    row: dict[str, Any] = {
        "schema_version": join.JOIN_SCHEMA,
        "model": "llama31_8b",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "task": task,
        "category": spec.category,
        "benchmark_id": benchmark_id,
        "source_index": source_index,
        "metric": spec.scorer,
        "references": ["answer"],
        "all_classes": None,
        "features": {
            "seq_len_tokens": 100 + task_index * 10 + source_index,
            "seq_len_chars": 400 + task_index * 10 + source_index,
            "token_entropy": 2.0 + source_index / 10,
            "gzip_ratio": 0.5,
            "unique_token_ratio": 0.75,
            "question_position": None if source_index == 0 else 0.5,
            "newline_density": 0.1,
        },
        "pre_truncation_token_count": 100,
        "post_truncation_token_count": 100,
        "truncated": False,
        "final_input_token_sha256": (
            f"{task_index * 10 + source_index + 1:064x}"
        ),
        "token_hash_algorithm": "sha256-little-endian-uint32",
        "max_new_tokens": spec.max_new_tokens,
    }
    for config in c6.LABEL_CONFIGURATIONS:
        ratio = selected_ratios[config]
        row.update(
            {
                f"{config}_score": selected_scores[config],
                f"{config}_prediction": f"{config}-prediction",
                f"{config}_kv_bytes": 1_000.0 / ratio,
                f"{config}_kv_bytes_fp16": 1_000.0,
                f"{config}_compression": ratio,
                f"{config}_generated_token_count": max(
                    1, spec.minimum_new_tokens
                ),
                f"{config}_kv_accounting": copy.deepcopy(
                    run_io.KV_ACCOUNTING_BY_CONFIG[config]
                ),
            }
        )
    return row


def _panel(
    *,
    ratio_overrides: dict[tuple[str, int], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    return [
        _row(
            task,
            source_index,
            ratios=(ratio_overrides or {}).get((task, source_index)),
        )
        for task in protocol.TASK_ORDER
        for source_index in range(2)
    ]


def _two_folds() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    all_indices = np.arange(sum(EXPECTED_COUNTS.values()), dtype=np.int64)
    even = all_indices[::2]
    odd = all_indices[1::2]
    return ((odd, even), (even, odd))


class _RecordingScaler:
    def __init__(
        self,
        config: str,
        fold: int,
        log: list[tuple[str, int, str, np.ndarray]],
    ) -> None:
        self.config = config
        self.fold = fold
        self.log = log

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.log.append((self.config, self.fold, "fit", values.copy()))
        return values

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.log.append((self.config, self.fold, "transform", values.copy()))
        return values


class _ConstantRegressor:
    def __init__(
        self,
        config: str,
        fold: int,
        predictions: dict[str, float],
        fit_log: list[tuple[str, int, np.ndarray]] | None = None,
    ) -> None:
        self.config = config
        self.fold = fold
        self.predictions = predictions
        self.fit_log = fit_log

    def fit(self, _features: np.ndarray, labels: np.ndarray) -> "_ConstantRegressor":
        if self.fit_log is not None:
            self.fit_log.append((self.config, self.fold, labels.copy()))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(
            len(features), self.predictions[self.config], dtype=np.float64
        )


def _factories(
    predictions: dict[str, float],
    *,
    scaler_log: list[tuple[str, int, str, np.ndarray]] | None = None,
    fit_log: list[tuple[str, int, np.ndarray]] | None = None,
):
    observed_scalers = scaler_log if scaler_log is not None else []

    def scaler_factory(config: str, fold: int) -> _RecordingScaler:
        return _RecordingScaler(config, fold, observed_scalers)

    def regressor_factory(config: str, fold: int) -> _ConstantRegressor:
        return _ConstantRegressor(config, fold, predictions, fit_log)

    return scaler_factory, regressor_factory


def _evaluate(
    rows: list[dict[str, Any]],
    predictions: dict[str, float],
    **kwargs: Any,
) -> c6.C6Result:
    scaler_factory, regressor_factory = _factories(predictions)
    return c6.evaluate_c6(
        rows,
        expected_task_counts=EXPECTED_COUNTS,
        folds=_two_folds(),
        scaler_factory=scaler_factory,
        regressor_factory=regressor_factory,
        **kwargs,
    )


def test_locked_policy_and_exact_estimator_parameters(tmp_path: Path) -> None:
    policy = c6.load_locked_policy()
    assert policy.reference_configuration == "C0"
    assert policy.candidate_configurations == ("C1", "C2", "C3", "C4", "C5")
    assert policy.tau == 0.99
    assert policy.folds == 10

    estimator = c6._default_regressor_factory("C0", 0)
    parameters = estimator.get_params()
    assert parameters["max_iter"] == 300
    assert parameters["max_depth"] == 4
    assert parameters["learning_rate"] == 0.05
    assert parameters["random_state"] == 0

    document = json.loads(c6.DEFAULT_RUN_CONFIG.read_text())
    document["router_dataset"]["routing_policy"]["iso_quality_tau"] = 0.98
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(document))
    with pytest.raises(c6.C6PolicyError, match="routing_policy"):
        c6.load_locked_policy(drifted)


def test_panel_is_canonical_and_seven_model_inputs_are_numeric() -> None:
    rows = list(reversed(_panel()))
    ordered, matrix = c6.validate_joined_panel(
        rows, expected_task_counts=EXPECTED_COUNTS
    )
    assert len(ordered) == 32
    assert matrix.shape == (32, 7)
    assert np.isfinite(matrix).all()
    assert ordered[0]["task"] == protocol.TASK_ORDER[0]
    assert ordered[0]["source_index"] == 0
    assert matrix[0, c6.FEATURE_KEYS.index("question_position")] == -1.0

    missing = rows[:-1]
    with pytest.raises(c6.C6ValidationError, match="rows; expected"):
        c6.validate_joined_panel(
            missing, expected_task_counts=EXPECTED_COUNTS
        )

    invalid = copy.deepcopy(rows)
    invalid[0]["features"]["token_entropy"] = float("inf")
    with pytest.raises(c6.C6ValidationError, match="must be finite"):
        c6.validate_joined_panel(
            invalid, expected_task_counts=EXPECTED_COUNTS
        )

    missing_label = copy.deepcopy(rows)
    del missing_label[0]["C5_prediction"]
    with pytest.raises(c6.C6ValidationError, match="lacks schema fields"):
        c6.validate_joined_panel(
            missing_label, expected_task_counts=EXPECTED_COUNTS
        )

    wrong_accounting = copy.deepcopy(rows)
    wrong_accounting[0]["C2_kv_accounting"]["kind"] = "physical_tensor_bytes"
    with pytest.raises(c6.C6ValidationError, match="C2_kv_accounting"):
        c6.validate_joined_panel(
            wrong_accounting, expected_task_counts=EXPECTED_COUNTS
        )


def test_training_only_scaling_and_compression_ranking() -> None:
    overrides: dict[tuple[str, int], dict[str, float]] = {}
    for task in protocol.TASK_ORDER:
        held_out_ratios = dict(RATIOS)
        held_out_ratios["C1"] = 100.0
        held_out_ratios["C2"] = 1.5
        overrides[(task, 0)] = held_out_ratios
    rows = _panel(ratio_overrides=overrides)
    predictions = {
        "C0": 0.80,
        "C1": 0.81,
        "C2": 0.81,
        "C3": 0.70,
        "C4": 0.60,
        "C5": 0.50,
    }
    scaler_log: list[tuple[str, int, str, np.ndarray]] = []
    fit_log: list[tuple[str, int, np.ndarray]] = []
    scaler_factory, regressor_factory = _factories(
        predictions, scaler_log=scaler_log, fit_log=fit_log
    )
    result = c6.evaluate_c6(
        rows,
        expected_task_counts=EXPECTED_COUNTS,
        folds=_two_folds(),
        scaler_factory=scaler_factory,
        regressor_factory=regressor_factory,
    )

    fold_zero_rows = [row for row in result.rows if row["fold"] == 0]
    assert {row["source_index"] for row in fold_zero_rows} == {0}
    # C1 has extreme compression only on fold-zero test rows. Training-only
    # ranking therefore still selects C2 for every one of those rows.
    assert {row["chosen_configuration"] for row in fold_zero_rows} == {"C2"}
    assert all("C0" not in row["eligible_candidates"] for row in result.rows)
    assert all(set(row["predicted_scores"]) == set(c6.LABEL_CONFIGURATIONS)
               for row in result.rows)
    assert all(
        row["kv_accounting"]
        == run_io.KV_ACCOUNTING_BY_CONFIG[row["chosen_configuration"]]
        for row in result.rows
    )

    fold_zero_fits = [
        values
        for _config, fold, operation, values in scaler_log
        if fold == 0 and operation == "fit"
    ]
    assert len(fold_zero_fits) == len(c6.LABEL_CONFIGURATIONS)
    # The last decimal digit identifies source_index in the synthetic panel.
    assert all(
        np.all((values[:, 0].astype(int) - 100) % 10 == 1)
        for values in fold_zero_fits
    )
    assert len(fit_log) == 2 * len(c6.LABEL_CONFIGURATIONS)
    assert len(result.summary["fit_metrics"]["C0"]["folds"]) == 2
    assert result.summary["quality"]["n_records"] == 32
    assert result.summary["compression"]["n_records"] == 32


def test_fixed_ties_and_quality_fallback_never_choose_c0() -> None:
    tied_ratios = dict(RATIOS)
    tied_ratios["C1"] = tied_ratios["C2"] = 4.0
    tied_panel = [
        _row(task, source_index, ratios=tied_ratios)
        for task in protocol.TASK_ORDER
        for source_index in range(2)
    ]
    eligible_predictions = {
        "C0": 0.80,
        "C1": 0.80,
        "C2": 0.80,
        "C3": 0.70,
        "C4": 0.60,
        "C5": 0.50,
    }
    eligible = _evaluate(tied_panel, eligible_predictions)
    assert {row["chosen_configuration"] for row in eligible.rows} == {"C1"}
    assert {
        row["selection_reason"] for row in eligible.rows
    } == {"eligible_train_compression"}

    fallback_predictions = {
        "C0": 0.90,
        "C1": 0.50,
        "C2": 0.50,
        "C3": 0.40,
        "C4": 0.30,
        "C5": 0.20,
    }
    fallback = _evaluate(_panel(), fallback_predictions)
    assert {row["chosen_configuration"] for row in fallback.rows} == {"C1"}
    assert {
        row["selection_reason"] for row in fallback.rows
    } == {"highest_predicted_quality_fallback"}
    assert fallback.summary["selection_counts"] == {
        "C1": 32,
        "C2": 0,
        "C3": 0,
        "C4": 0,
        "C5": 0,
    }


def test_reproducibility_and_fold_guards() -> None:
    predictions = {
        "C0": 0.80,
        "C1": 0.82,
        "C2": 0.83,
        "C3": 0.70,
        "C4": 0.60,
        "C5": 0.50,
    }
    first = _evaluate(_panel(), predictions)
    second = _evaluate(list(reversed(_panel())), predictions)
    assert first.rows == second.rows
    assert first.summary == second.summary

    bad_overlap = list(_two_folds())
    train, test = bad_overlap[0]
    bad_overlap[0] = (np.append(train, test[0]), test)
    scaler_factory, regressor_factory = _factories(predictions)
    with pytest.raises(c6.C6ValidationError, match="leaks test rows"):
        c6.evaluate_c6(
            _panel(),
            expected_task_counts=EXPECTED_COUNTS,
            folds=bad_overlap,
            scaler_factory=scaler_factory,
            regressor_factory=regressor_factory,
        )

    duplicate_test = (_two_folds()[0], _two_folds()[0])
    with pytest.raises(
        c6.C6ValidationError, match="held out exactly once"
    ):
        c6.evaluate_c6(
            _panel(),
            expected_task_counts=EXPECTED_COUNTS,
            folds=duplicate_test,
            scaler_factory=scaler_factory,
            regressor_factory=regressor_factory,
        )


def test_production_identity_rejects_dirty_wrong_commit_and_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = run_io.RunLayout.create(
        tmp_path / "runs" / "longbench16_24k",
        "llama31_8b",
        "provenance-run",
    )
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    source_code = {"commit": "f" * 40, "dirty": False}
    config_digest = run_io.file_sha256(c6.DEFAULT_RUN_CONFIG)
    prepared_manifest_path = prepared_dir / "manifest.json"
    prepared_manifest_path.write_text(
        json.dumps(
            {
                "run_config": {
                    "path": str(c6.DEFAULT_RUN_CONFIG.resolve()),
                    "sha256": config_digest,
                },
                "source_code": source_code,
            },
            sort_keys=True,
        )
        + "\n"
    )
    prepared = SimpleNamespace(
        prepared_dir=prepared_dir.resolve(),
        manifest_path=prepared_manifest_path.resolve(),
        manifest_sha256=run_io.file_sha256(prepared_manifest_path),
        index_sha256="a" * 64,
        record_count=protocol.EXPECTED_TOTAL_EXAMPLES,
    )
    monkeypatch.setattr(
        c6,
        "load_prepared_expectations",
        lambda path: prepared
        if Path(path).resolve() == prepared_dir.resolve()
        else (_ for _ in ()).throw(AssertionError(path)),
    )
    model = json.loads(c6.DEFAULT_RUN_CONFIG.read_text())["model"]
    manifest = {
        "schema_version": c6.RUNNER_SCHEMA,
        "status": "running",
        "started_at": "2026-07-20T12:00:00Z",
        "run_id": layout.run_id,
        "model_alias": model["alias"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "tokenizer_id": model["tokenizer_id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "run_config": {
            "path": str(c6.DEFAULT_RUN_CONFIG.resolve()),
            "sha256": config_digest,
        },
        "prepared_inputs": {
            "path": str(prepared_dir.resolve()),
            "manifest_sha256": prepared.manifest_sha256,
            "index_sha256": prepared.index_sha256,
            "records": protocol.EXPECTED_TOTAL_EXAMPLES,
        },
        "source_code": source_code,
    }
    run_io.atomic_write_json(layout.manifest_path, manifest)

    identity = c6.validate_production_run_identity(
        layout, current_source_state=source_code
    )
    assert identity["run_id"] == layout.run_id
    assert identity["source_code"]["commit"] == "f" * 40
    assert identity["source_code"]["dirty"] is False
    assert identity["prepared_inputs"]["records"] == 3_750

    with pytest.raises(c6.C6ProvenanceError, match="clean source tree"):
        c6.validate_production_run_identity(
            layout,
            current_source_state={"commit": "f" * 40, "dirty": True},
        )
    with pytest.raises(c6.C6ProvenanceError, match="differs from"):
        c6.validate_production_run_identity(
            layout,
            current_source_state={"commit": "e" * 40, "dirty": False},
        )

    completed = copy.deepcopy(manifest)
    completed["status"] = "complete"
    run_io.atomic_write_json(
        layout.manifest_path, completed, overwrite=True
    )
    with pytest.raises(c6.C6ProvenanceError, match="manifest is complete"):
        c6.validate_production_run_identity(
            layout, current_source_state=source_code
        )


def test_actual_c0_violations_and_protocol_aggregates_are_reported() -> None:
    predictions = {
        "C0": 0.90,
        "C1": 0.90,
        "C2": 0.89,
        "C3": 0.70,
        "C4": 0.60,
        "C5": 0.50,
    }
    result = _evaluate(_panel(), predictions)
    violation = result.summary["violation_rates_vs_actual_c0"]
    assert violation["below_tau_times_actual_c0_count"] == 0
    assert violation["below_tau_times_actual_c0_rate"] == 0.0
    assert result.summary["quality"]["category_balanced_mean"] == pytest.approx(
        SCORES["C1"]
    )
    assert result.summary["quality"]["prompt_micro_mean"] == pytest.approx(
        SCORES["C1"]
    )
    assert result.summary["compression"]["harmonic_mean"] == pytest.approx(
        RATIOS["C1"]
    )
    assert result.summary["compression"][
        "category_balanced_mean"
    ] == pytest.approx(RATIOS["C1"])


def test_outputs_are_atomic_immutable_and_live_under_run_analysis(
    tmp_path: Path,
) -> None:
    predictions = {
        "C0": 0.80,
        "C1": 0.82,
        "C2": 0.83,
        "C3": 0.70,
        "C4": 0.60,
        "C5": 0.50,
    }
    result = _evaluate(_panel(), predictions)
    layout = run_io.RunLayout.create(
        tmp_path / "runs" / "longbench16_24k",
        "llama31_8b",
        "test-run",
    )
    published = c6.write_c6_outputs(result, layout)
    rows_path = layout.analysis_dir / "c6_per_prompt.jsonl"
    summary_path = layout.analysis_dir / "c6_summary.json"
    assert rows_path.is_file()
    assert summary_path.is_file()
    assert rows_path.read_bytes().endswith(b"\n")
    assert len(rows_path.read_text().splitlines()) == 32
    assert published["schema_version"] == c6.C6_ANALYSIS_SCHEMA
    assert published["artifacts"]["per_prompt_jsonl"]["sha256"] == (
        run_io.file_sha256(rows_path)
    )
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["artifacts"]["per_prompt_jsonl"]["rows"] == 32
    before = (rows_path.read_bytes(), summary_path.read_bytes())
    with pytest.raises(FileExistsError):
        c6.write_c6_outputs(result, layout)
    assert before == (rows_path.read_bytes(), summary_path.read_bytes())

    run_io.atomic_write_json(
        layout.manifest_path,
        {"status": "complete"},
    )
    with pytest.raises(c6.C6ProvenanceError, match="manifest is complete"):
        c6.write_c6_outputs(result, layout)
