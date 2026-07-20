from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import longbench16_io as run_io
from scripts import longbench16_join as join
from scripts import longbench16_protocol as protocol


SCORES = {
    "C0": 0.70,
    "C1": 0.80,
    "C2": 0.90,
    "C3": 0.90,
    "C4": 0.60,
    "C5": 0.50,
}
RATIOS = {
    "C0": 1.0,
    "C1": 2.0,
    "C2": 3.0,
    "C3": 4.0,
    "C4": 5.0,
    "C5": 6.0,
}
EXPECTED_KEYS = tuple(
    (task, f"{task}:0000") for task in protocol.TASK_ORDER
)


def _hash(task: str, benchmark_id: str) -> str:
    return hashlib.sha256(f"{task}:{benchmark_id}".encode()).hexdigest()


def _record(
    config: str,
    task: str,
    benchmark_id: str,
    *,
    source_index: int = 0,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    ratio = RATIOS[config]
    record: dict[str, object] = {
        "model": "llama31_8b",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "configuration": config,
        "task": task,
        "category": protocol.TASK_SPECS[task].category,
        "benchmark_id": benchmark_id,
        "source_index": source_index,
        "metric": protocol.TASK_SPECS[task].scorer,
        "references": [f"answer-{source_index}"],
        "all_classes": (
            ["description and abstract concept", "entity"]
            if task == "trec"
            else None
        ),
        "features": {
            "seq_len_tokens": 10,
            "seq_len_chars": 40,
            "token_entropy": 2.0,
            "gzip_ratio": 1.2,
            "unique_token_ratio": 0.8,
            "question_position": None,
            "newline_density": 0.05,
        },
        "pre_truncation_token_count": 12,
        "post_truncation_token_count": 12,
        "truncated": False,
        "final_input_token_sha256": _hash(task, benchmark_id),
        "token_hash_algorithm": "sha256-little-endian-uint32",
        "max_new_tokens": protocol.TASK_SPECS[task].max_new_tokens,
        "minimum_new_tokens": protocol.TASK_SPECS[task].minimum_new_tokens,
        "stop_on_newline_token": protocol.TASK_SPECS[
            task
        ].stop_on_newline_token,
        "prediction": f"{config}-prediction",
        "score": SCORES[config],
        "kv_bytes": 600.0 / ratio,
        "kv_bytes_fp16": 600.0,
        "compression": ratio,
        "generated_token_count": 1,
        "kv_accounting": run_io.KV_ACCOUNTING_BY_CONFIG[config],
        "status": "completed",
    }
    if overrides:
        record.update(overrides)
    return record


def _write_panel(
    root: Path,
    *,
    overrides: dict[tuple[str, str], dict[str, object]] | None = None,
    benchmark_id_overrides: dict[str, str] | None = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for config in run_io.FIXED_CONFIGS:
        path = root / config / "per_prompt.jsonl"
        journal = run_io.AtomicJsonlJournal(path)
        for task, benchmark_id in EXPECTED_KEYS:
            benchmark_id = (benchmark_id_overrides or {}).get(
                task, benchmark_id
            )
            journal.append(
                _record(
                    config,
                    task,
                    benchmark_id,
                    overrides=(overrides or {}).get((config, task)),
                )
            )
        paths[config] = path
    return paths


def _write_prepared(
    directory: Path,
    *,
    keys: tuple[tuple[str, str], ...] = EXPECTED_KEYS,
) -> Path:
    directory.mkdir(parents=True)
    token_files = []
    for task in dict.fromkeys(task for task, _ in keys):
        token_path = directory / "tokens" / f"{task}.npz"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_bytes(f"synthetic-token-file:{task}".encode())
        token_files.append(
            {
                "task": task,
                "path": f"tokens/{task}.npz",
                "sha256": run_io.file_sha256(token_path),
                "rows": sum(key_task == task for key_task, _ in keys),
                "stored_token_ids": sum(
                    key_task == task for key_task, _ in keys
                ),
            }
        )
    records = []
    for task, benchmark_id in keys:
        source = _record("C0", task, benchmark_id)
        records.append(
            {
                "schema_version": join.PREPARED_SCHEMA,
                "model": source["model"],
                "model_id": source["model_id"],
                "task": task,
                "category": source["category"],
                "benchmark_id": benchmark_id,
                "source_index": source["source_index"],
                "metric": source["metric"],
                "references": source["references"],
                "all_classes": source["all_classes"],
                "features": source["features"],
                "pre_truncation_token_count": source[
                    "pre_truncation_token_count"
                ],
                "post_truncation_token_count": source[
                    "post_truncation_token_count"
                ],
                "truncated": source["truncated"],
                "final_input_token_sha256": source[
                    "final_input_token_sha256"
                ],
                "token_hash_algorithm": source["token_hash_algorithm"],
                "max_new_tokens": source["max_new_tokens"],
                "minimum_new_tokens": source["minimum_new_tokens"],
                "stop_on_newline_token": source["stop_on_newline_token"],
                "token_file": f"tokens/{task}.npz",
                "token_offset_index": source["source_index"],
            }
        )
    index_path = directory / "index.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in records
        )
    )
    manifest = {
        "schema_version": join.PREPARED_SCHEMA,
        "status": "complete",
        "completed_at_utc": "2026-07-20T12:00:00+00:00",
        "records": len(records),
        "benchmark": {
            "requested_dataset_id": protocol.LONG_BENCH_DATASET_ID,
            "resolved_dataset_id": protocol.LONG_BENCH_RESOLVED_DATASET_ID,
            "dataset_revision": protocol.LONG_BENCH_DATASET_REVISION,
            "split": protocol.LONG_BENCH_SPLIT,
            "source_hashes_verified_against_pinned_release": True,
            "source_files": [
                {
                    "task": task,
                    "rows": protocol.TASK_SPECS[
                        task
                    ].expected_test_examples,
                    "sha256": protocol.source_manifest()[
                        "longbench_dataset"
                    ]["extracted_task_sha256"][task],
                }
                for task in protocol.TASK_ORDER
            ],
        },
        "input_policy": {
            "cap": protocol.FINAL_INPUT_TOKEN_CAP,
            "middle_truncation_tokens_per_side": 12_000,
            "token_hash_algorithm": "sha256-little-endian-uint32",
            "decode_and_retokenize_after_truncation": False,
            "no_chat_add_special_tokens": True,
            "chat_template_date_string": "20 Jul 2026",
        },
        "prepared_index": {
            "path": "index.jsonl",
            "sha256": run_io.file_sha256(index_path),
            "rows": len(records),
        },
        "token_files": token_files,
        "protocol_config_hash": protocol.protocol_config_hash(),
        "protocol_config_files": protocol.config_file_hashes(),
        "run_config": {
            "path": str(join.ORACLE_CONFIG_PATH),
            "sha256": run_io.file_sha256(join.ORACLE_CONFIG_PATH),
        },
        "tokenizer": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        },
        "source_code": {
            "commit": "f" * 40,
            "dirty": False,
        },
    }
    manifest_path = directory / "manifest.json"
    run_io.atomic_write_json(manifest_path, manifest)
    run_io.atomic_write_json(
        directory / "status.json",
        {
            "schema_version": join.PREPARED_SCHEMA,
            "status": "complete",
            "completed_at_utc": manifest["completed_at_utc"],
            "manifest_sha256": run_io.file_sha256(manifest_path),
        },
    )
    return directory


def _small_join(
    root: Path,
    *,
    overrides: dict[tuple[str, str], dict[str, object]] | None = None,
) -> join.JoinResult:
    return join.join_journals(
        _write_panel(root, overrides=overrides),
        expected_keys=EXPECTED_KEYS,
        require_complete=False,
    )


def test_config_validation_recomputes_scores_and_locks_c0_reference() -> None:
    task = "narrativeqa"
    key = (task, f"{task}:0000")
    record = _record("C0", *key)

    with pytest.raises(join.JoinValidationError, match="recomputed score"):
        join._config_values(
            record,
            "C0",
            key,
            verify_score=True,
        )

    record["score"] = 0.0
    record["kv_bytes"] = 300.0
    record["compression"] = 2.0
    with pytest.raises(join.JoinValidationError, match="exact FP16"):
        join._config_values(
            record,
            "C0",
            key,
            verify_score=True,
        )


def test_strict_join_explicit_columns_and_protocol_aggregates(
    tmp_path: Path,
) -> None:
    result = _small_join(tmp_path / "run")
    rows, analysis = join.build_analysis(
        result,
        require_complete=False,
    )

    assert result.row_count == len(protocol.TASK_ORDER) == 16
    assert [row["task"] for row in rows] == list(protocol.TASK_ORDER)
    first = rows[0]
    assert first["source_index"] == 0
    assert first["references"] == ["answer-0"]
    assert set(first["features"]) == set(join.FEATURE_KEYS)
    assert first["final_input_token_sha256"] == _hash(
        "narrativeqa", "narrativeqa:0000"
    )
    for config in run_io.FIXED_CONFIGS:
        assert all(
            f"{config}_{field}" in first for field in join.CONFIG_OUTPUT_FIELDS
        )
        assert first[f"{config}_compression"] == pytest.approx(RATIOS[config])

    fixed_c3 = analysis["fixed_configurations"]["C3"]
    assert fixed_c3["quality"]["n_records"] == 16
    assert fixed_c3["quality"]["task_scores"] == pytest.approx(
        {task: SCORES["C3"] for task in protocol.TASK_ORDER}
    )
    assert fixed_c3["quality"]["category_scores"] == pytest.approx(
        {category: SCORES["C3"] for category in protocol.CATEGORY_ORDER}
    )
    assert fixed_c3["quality"]["category_balanced_mean"] == pytest.approx(0.9)
    assert fixed_c3["quality"]["prompt_micro_mean"] == pytest.approx(0.9)
    assert fixed_c3["compression"]["harmonic_mean"] == pytest.approx(4.0)
    assert fixed_c3["compression"]["category_balanced_mean"] == pytest.approx(
        4.0
    )

    oracle_analysis = analysis["oracle_analysis"]
    assert oracle_analysis["candidate_pool"] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    ]
    primary = oracle_analysis["primary"]
    assert primary["name"] == (
        "iso_quality_tau_0.99_max_compression"
    )
    assert primary["selection_counts"] == {
        "C1": 0,
        "C2": 0,
        "C3": 16,
        "C4": 0,
        "C5": 0,
    }
    assert primary["quality"]["category_balanced_mean"] == pytest.approx(0.9)
    assert primary["compression"]["harmonic_mean"] == pytest.approx(4.0)
    assert primary["fallback_count"] == 0
    assert primary["fallback_fraction"] == 0.0
    assert primary["actual_threshold_violation_count_nonfallback"] == 0
    assert primary["actual_threshold_violation_fraction_nonfallback"] == 0.0
    diagnostic = oracle_analysis["required_quality_first_diagnostic"]
    assert diagnostic["name"] == "max_quality_then_compression_then_config"
    assert diagnostic["selection_counts"]["C3"] == 16
    assert all(row["oracle_configuration"] == "C3" for row in rows)
    assert all(row["oracle_fallback"] is False for row in rows)
    assert all(
        row["quality_oracle_configuration"] == "C3" for row in rows
    )
    assert (
        analysis["integrity"]["cross_config_final_input_hash"]["status"]
        == "passed"
    )


def test_locked_oracle_default_and_tie_rules_are_deterministic(
    tmp_path: Path,
) -> None:
    overrides = {
        ("C2", task): {"compression": 4.0, "kv_bytes": 150.0}
        for task in protocol.TASK_ORDER
    }
    result = _small_join(tmp_path / "run", overrides=overrides)

    primary_rows, analysis = join.analyze_joined_rows(
        result.rows,
        require_complete=False,
    )
    assert all(row["oracle_configuration"] == "C2" for row in primary_rows)
    assert all(
        row["quality_oracle_configuration"] == "C2"
        for row in primary_rows
    )
    assert "fixed [C1, C2, C3, C4, C5] order" in analysis[
        "oracle_analysis"
    ]["primary"][
        "deterministic_tie_rule"
    ][2]


def test_iso_quality_oracle_reports_deterministic_fallbacks(
    tmp_path: Path,
) -> None:
    overrides = {
        (config, task): {"score": 0.5}
        for config in join.ORACLE_CANDIDATES
        for task in ("qasper",)
    }
    result = _small_join(tmp_path / "run", overrides=overrides)
    rows, analysis = join.analyze_joined_rows(
        result.rows, require_complete=False
    )

    # No candidate reaches 0.99 * C0 (0.693). All fallback qualities tie,
    # so measured compression chooses C5 deterministically.
    qasper = next(row for row in rows if row["task"] == "qasper")
    assert qasper["oracle_configuration"] == "C5"
    assert qasper["oracle_fallback"] is True
    assert all(
        row["oracle_fallback"] is False
        for row in rows
        if row["task"] != "qasper"
    )
    primary = analysis["oracle_analysis"]["primary"]
    assert primary["fallback_count"] == 1
    assert primary["fallback_fraction"] == pytest.approx(1 / 16)
    assert primary["nonfallback_count"] == 15
    assert primary["actual_threshold_violation_count_nonfallback"] == 0
    assert primary["actual_threshold_violation_fraction_nonfallback"] == 0.0


def test_full_mode_refuses_a_small_expected_key_injection(
    tmp_path: Path,
) -> None:
    paths = _write_panel(tmp_path / "run")
    with pytest.raises(join.JoinValidationError, match="exactly 3750"):
        join.join_journals(paths, expected_keys=EXPECTED_KEYS)


def test_join_never_derives_production_expectations_from_c0(
    tmp_path: Path,
) -> None:
    paths = _write_panel(tmp_path / "run")
    with pytest.raises(join.JoinValidationError, match="may not be derived from C0"):
        join.join_journals(paths, require_complete=False)


def test_prepared_artifact_is_the_key_and_token_hash_authority(
    tmp_path: Path,
) -> None:
    prepared_dir = _write_prepared(tmp_path / "prepared")
    paths = _write_panel(tmp_path / "run")
    result = join.join_journals(
        paths, prepared_dir=prepared_dir, require_complete=False
    )

    assert result.prepared is not None
    assert result.prepared.record_count == 16
    assert result.prepared.ordered_keys == EXPECTED_KEYS
    rows, analysis = join.build_analysis(
        result,
        oracle_objective="max_quality_then_compression_then_config",
        require_complete=False,
    )
    assert len(rows) == 16
    evidence = analysis["integrity"]["prepared_artifact"]
    assert evidence["status"] == "passed"
    assert evidence["record_count"] == 16
    assert evidence["index"]["sha256"] == run_io.file_sha256(
        prepared_dir / "index.jsonl"
    )


def test_mutually_consistent_wrong_benchmark_id_is_rejected_by_preparation(
    tmp_path: Path,
) -> None:
    prepared_dir = _write_prepared(tmp_path / "prepared")
    paths = _write_panel(
        tmp_path / "run",
        benchmark_id_overrides={"qasper": "qasper:wrong-but-consistent"},
    )

    with pytest.raises(run_io.ValidationError, match="unexpected record keys"):
        join.join_journals(
            paths, prepared_dir=prepared_dir, require_complete=False
        )


def test_mutually_consistent_wrong_token_hash_is_rejected_by_preparation(
    tmp_path: Path,
) -> None:
    prepared_dir = _write_prepared(tmp_path / "prepared")
    overrides = {
        (config, "qasper"): {"final_input_token_sha256": "f" * 64}
        for config in run_io.FIXED_CONFIGS
    }
    paths = _write_panel(tmp_path / "run", overrides=overrides)

    with pytest.raises(
        join.JoinValidationError, match="does not match the prepared index"
    ):
        join.join_journals(
            paths, prepared_dir=prepared_dir, require_complete=False
        )


def test_prepared_manifest_status_index_hash_and_rows_are_verified(
    tmp_path: Path,
) -> None:
    failed_status = _write_prepared(tmp_path / "failed-status")
    status_path = failed_status / "status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "failed"
    run_io.atomic_write_json(status_path, status, overwrite=True)
    with pytest.raises(join.JoinValidationError, match="both declare complete"):
        join.load_prepared_expectations(
            failed_status, require_complete=False
        )

    changed_index = _write_prepared(tmp_path / "changed-index")
    with (changed_index / "index.jsonl").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(join.JoinValidationError, match="SHA-256"):
        join.load_prepared_expectations(
            changed_index, require_complete=False
        )

    bad_rows = _write_prepared(tmp_path / "bad-rows")
    manifest_path = bad_rows / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prepared_index"]["rows"] -= 1
    run_io.atomic_write_json(manifest_path, manifest, overwrite=True)
    status_path = bad_rows / "status.json"
    status = json.loads(status_path.read_text())
    status["manifest_sha256"] = run_io.file_sha256(manifest_path)
    run_io.atomic_write_json(status_path, status, overwrite=True)
    with pytest.raises(join.JoinValidationError, match="row counts"):
        join.load_prepared_expectations(bad_rows, require_complete=False)


def test_join_rejects_shared_metadata_or_hash_mismatch(tmp_path: Path) -> None:
    mismatched_references = _write_panel(
        tmp_path / "references",
        overrides={
            ("C4", "qasper"): {"references": ["different underlying prompt"]}
        },
    )
    with pytest.raises(join.JoinValidationError, match="references.*differs"):
        join.join_journals(
            mismatched_references,
            expected_keys=EXPECTED_KEYS,
            require_complete=False,
        )

    mismatched_hash = _write_panel(
        tmp_path / "hash",
        overrides={
            ("C5", "qasper"): {"final_input_token_sha256": "f" * 64}
        },
    )
    with pytest.raises(run_io.ValidationError, match="token hash mismatch"):
        join.join_journals(
            mismatched_hash,
            expected_keys=EXPECTED_KEYS,
            require_complete=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {("C3", "qasper"): {"kv_bytes": 0.0}},
            "must be positive",
        ),
        (
            {("C3", "qasper"): {"compression": 99.0}},
            "does not match",
        ),
        (
            {("C3", "qasper"): {"score": 1.1}},
            r"score must be in \[0, 1\]",
        ),
    ],
)
def test_join_rejects_invalid_numeric_labels(
    tmp_path: Path,
    overrides: dict[tuple[str, str], dict[str, object]],
    message: str,
) -> None:
    paths = _write_panel(tmp_path / "run", overrides=overrides)
    with pytest.raises(join.JoinValidationError, match=message):
        join.join_journals(
            paths, expected_keys=EXPECTED_KEYS, require_complete=False
        )


def test_join_rejects_nonfinite_json_number(tmp_path: Path) -> None:
    paths = _write_panel(tmp_path / "run")
    path = paths["C0"]
    text = path.read_text()
    assert '"kv_bytes":600.0' in text
    path.write_text(text.replace('"kv_bytes":600.0', '"kv_bytes":1e999', 1))

    with pytest.raises(join.JoinValidationError, match="kv_bytes must be finite"):
        join.join_journals(
            paths, expected_keys=EXPECTED_KEYS, require_complete=False
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "", "model_id must be a non-empty string"),
        (
            "references",
            [{"nested": "not an answer scalar"}],
            r"references\[0\] must be a string or finite number",
        ),
        (
            "all_classes",
            [7],
            "all_classes must be null or a list",
        ),
    ],
)
def test_join_requires_typed_identity_and_reference_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    paths = _write_panel(
        tmp_path / "run",
        overrides={("C4", "qasper"): {field: value}},
    )
    with pytest.raises(join.JoinValidationError, match=message):
        join.join_journals(
            paths, expected_keys=EXPECTED_KEYS, require_complete=False
        )


def test_join_rejects_duplicate_source_index(tmp_path: Path) -> None:
    paths = _write_panel(tmp_path / "run")
    extra_key = ("qasper", "qasper:duplicate")
    for config, path in paths.items():
        run_io.AtomicJsonlJournal(path).append(
            _record(
                config,
                extra_key[0],
                extra_key[1],
                source_index=0,
            )
        )

    with pytest.raises(join.JoinValidationError, match="duplicate source_index"):
        join.join_journals(
            paths,
            expected_keys=(*EXPECTED_KEYS, extra_key),
            require_complete=False,
        )


def test_atomic_outputs_include_join_hash_and_never_overwrite(
    tmp_path: Path,
) -> None:
    result = _small_join(tmp_path / "run")
    rows, analysis = join.build_analysis(
        result,
        oracle_objective="max_quality_then_compression_then_config",
        require_complete=False,
    )
    joined_path = tmp_path / "outputs" / "joined.jsonl"
    analysis_path = tmp_path / "outputs" / "summary.json"

    payload = join.write_analysis_outputs(
        joined_path, analysis_path, rows, analysis
    )

    lines = joined_path.read_text().splitlines()
    assert len(lines) == 16
    assert all(isinstance(json.loads(line), dict) for line in lines)
    joined_bytes = joined_path.read_bytes()
    assert payload["artifacts"]["joined_jsonl"]["sha256"] == hashlib.sha256(
        joined_bytes
    ).hexdigest()
    assert json.loads(analysis_path.read_text()) == payload
    assert not list(joined_path.parent.glob(".*.tmp"))

    with pytest.raises(FileExistsError):
        join.write_analysis_outputs(joined_path, analysis_path, rows, analysis)
    with pytest.raises(run_io.UnsafeOutputPathError):
        join.atomic_write_jsonl(tmp_path / "runs" / "p0" / "joined.jsonl", rows)


def test_cli_requires_prepared_artifact_and_defaults_locked_oracle(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc:
        join.main(["--run-dir", str(tmp_path)])
    assert exc.value.code == 2
    args = join._parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--prepared-dir",
            str(tmp_path / "prepared"),
        ]
    )
    assert args.oracle_objective == join.PRIMARY_ORACLE_OBJECTIVE
