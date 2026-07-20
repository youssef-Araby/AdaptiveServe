from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import longbench16_finalize as finalize
from scripts.longbench16_io import FIXED_CONFIGS, atomic_write_json, file_sha256


COMMIT = "a" * 40
INPUT_HASH = "b" * 64


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _declared_artifact(path: Path, *, rows: int) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "rows": rows,
    }


def _environment(
    *,
    run_id: str,
    model_alias: str,
    config_sha256: str,
    prepared_sha256: str,
) -> dict[str, object]:
    packages = {
        package: "1.0"
        for package in finalize._REQUIRED_PACKAGES
    }
    return {
        "captured_at": "2026-07-20T12:00:01Z",
        "python": {
            "version": "3.13.0",
            "implementation": "CPython",
            "executable": "/usr/bin/python",
        },
        "platform": {
            "system": "Linux",
            "release": "test",
            "version": "test",
            "machine": "x86_64",
        },
        "packages": packages,
        "git": {"commit": COMMIT, "dirty": False},
        "gpu": {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "uuid": "GPU-test",
                    "driver_version": "1",
                    "memory_total_mib": 24564,
                    "memory_used_mib": 1024,
                    "memory_free_mib": 23540,
                    "compute_capability": "8.6",
                }
            ],
        },
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        "extra": {
            "run_id": run_id,
            "model_alias": model_alias,
            "run_config_sha256": config_sha256,
            "prepared_manifest_sha256": prepared_sha256,
        },
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    # Keep orchestration tests tiny.  The production entry point has no count,
    # fold, model, or validator injection boundary; the strict component
    # validators have their own full-contract tests.
    monkeypatch.setattr(finalize, "EXPECTED_TOTAL_EXAMPLES", 1)
    monkeypatch.setattr(finalize, "EXPECTED_GENERATIONS_PER_MODEL", 6)

    model_alias = "llama31_8b"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    revision = "0" * 40
    run_id = "run-a"
    run_dir = (
        tmp_path
        / "runs"
        / "longbench16_24k"
        / model_alias
        / run_id
    )
    (run_dir / "analysis").mkdir(parents=True)
    (run_dir / "pilots").mkdir()

    config_path = tmp_path / "longbench16_24k.json"
    atomic_write_json(config_path, {"locked": True})
    config_sha256 = file_sha256(config_path)
    locked = SimpleNamespace(
        path=config_path.resolve(),
        sha256=config_sha256,
        model_alias=model_alias,
        model={
            "model_id": model_id,
            "model_revision": revision,
            "tokenizer_id": model_id,
            "tokenizer_revision": revision,
        },
    )

    prepared_dir = tmp_path / "prepared"
    token_dir = prepared_dir / "tokens"
    token_dir.mkdir(parents=True)
    index_path = prepared_dir / "index.jsonl"
    _write_jsonl(index_path, [{"task": "toy", "benchmark_id": "toy:0"}])
    token_files: list[dict[str, object]] = []
    for task_index in range(16):
        task = f"task-{task_index:02d}"
        token_path = token_dir / f"{task}.npz"
        token_path.write_bytes(f"tokens-{task}".encode())
        token_files.append(
            {
                "task": task,
                "path": f"tokens/{task}.npz",
                "rows": 1,
                "sha256": file_sha256(token_path),
            }
        )
    prepared_manifest = {
        "source_code": {"commit": COMMIT, "dirty": False},
        "benchmark": {
            "source_hashes_verified_against_pinned_release": True,
        },
        "token_files": token_files,
    }
    prepared_manifest_path = prepared_dir / "manifest.json"
    atomic_write_json(prepared_manifest_path, prepared_manifest)
    prepared_status_path = prepared_dir / "status.json"
    atomic_write_json(prepared_status_path, {"status": "complete"})
    prepared_sha256 = file_sha256(prepared_manifest_path)
    index_sha256 = file_sha256(index_path)
    prepared = SimpleNamespace(
        root=prepared_dir.resolve(),
        manifest=prepared_manifest,
        manifest_sha256=prepared_sha256,
        index_path=index_path.resolve(),
        index_sha256=index_sha256,
    )
    prepared_expectations = SimpleNamespace(
        manifest_sha256=prepared_sha256,
        index_sha256=index_sha256,
        record_count=1,
    )

    manifest = {
        "schema_version": finalize.RUNNER_SCHEMA,
        "status": "running",
        "started_at": "2026-07-20T12:00:00Z",
        "run_id": run_id,
        "model_alias": model_alias,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_id": model_id,
        "tokenizer_revision": revision,
        "run_config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
        },
        "prepared_inputs": {
            "path": str(prepared_dir.resolve()),
            "manifest_sha256": prepared_sha256,
            "index_sha256": index_sha256,
            "records": 1,
        },
        "source_code": {"commit": COMMIT, "dirty": False},
    }
    manifest_path = run_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    environment_path = run_dir / "environment.json"
    atomic_write_json(
        environment_path,
        _environment(
            run_id=run_id,
            model_alias=model_alias,
            config_sha256=config_sha256,
            prepared_sha256=prepared_sha256,
        ),
    )

    validation_by_config: dict[str, SimpleNamespace] = {}
    journal_paths: dict[str, Path] = {}
    pilot_paths: dict[str, Path] = {}
    for config in FIXED_CONFIGS:
        path = run_dir / config / "per_prompt.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "status": "completed",
                    "configuration": config,
                    "task": "toy",
                    "benchmark_id": "toy:0",
                    "final_input_token_sha256": INPUT_HASH,
                }
            ],
        )
        journal_paths[config] = path.resolve()
        validation_by_config[config] = SimpleNamespace(
            path=path.resolve(),
            record_count=1,
            journal_event_count=1,
            failed_attempt_count=0,
            retried_keys=frozenset(),
        )
        pilot_path = (
            run_dir
            / "pilots"
            / finalize.PILOT_PROFILE_WORST_CASE_24K_512
            / config
            / "per_prompt.jsonl"
        )
        _write_jsonl(
            pilot_path,
            [
                {
                    "status": "completed",
                    "configuration": config,
                    "task": "toy",
                    "benchmark_id": "toy:0",
                    "pilot_profile": (
                        finalize.PILOT_PROFILE_WORST_CASE_24K_512
                    ),
                    "generated_token_count": 512,
                    "max_new_tokens": 512,
                    "method_metadata": {
                        "pilot_profile": (
                            finalize.PILOT_PROFILE_WORST_CASE_24K_512
                        ),
                        "forced_full_decode": True,
                        "cuda_feasibility": {
                            "headroom_sufficient": True,
                        },
                    },
                }
            ],
        )
        pilot_paths[config] = pilot_path.resolve()
    join_result = SimpleNamespace(
        validations=validation_by_config,
        cross_validation=SimpleNamespace(
            combined_input_hash_sha256=hashlib.sha256(b"cross").hexdigest()
        ),
    )

    joined_row = {
        "schema_version": finalize.JOIN_SCHEMA,
        "task": "toy",
        "benchmark_id": "toy:0",
    }
    rebuilt_analysis = {
        "schema_version": finalize.ANALYSIS_SCHEMA,
        "row_count": 1,
        "oracle_analysis": {
            "joined_oracle_columns_objective": (
                finalize.PRIMARY_ORACLE_OBJECTIVE
            ),
            "primary": {"name": finalize.PRIMARY_ORACLE_OBJECTIVE},
            "required_quality_first_diagnostic": {
                "name": finalize.QUALITY_ORACLE_DIAGNOSTIC
            },
        },
    }
    joined_path = (run_dir / "analysis" / finalize.JOINED_FILENAME).resolve()
    _write_jsonl(joined_path, [joined_row])
    analysis_path = (
        run_dir / "analysis" / finalize.ANALYSIS_FILENAME
    ).resolve()
    joined_summary = copy.deepcopy(rebuilt_analysis)
    joined_summary["artifacts"] = {
        "joined_jsonl": _declared_artifact(joined_path, rows=1),
        "analysis_json": {"path": str(analysis_path)},
    }
    atomic_write_json(analysis_path, joined_summary)

    c6_row = {
        "schema_version": finalize.C6_ROW_SCHEMA,
        "configuration": "C6",
        "task": "toy",
        "benchmark_id": "toy:0",
    }
    c6_summary_base = {
        "schema_version": finalize.C6_ANALYSIS_SCHEMA,
        "status": "complete",
        "evaluation_mode": "primary",
        "row_count": 1,
        "source_artifact": _declared_artifact(joined_path, rows=1),
        "cross_validation": {
            "fold_count": 10,
            "folds_injected": False,
            "each_example_held_out_exactly_once": True,
        },
        "run_identity": {
            "run_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256_at_c6": file_sha256(manifest_path),
                "schema_version": finalize.RUNNER_SCHEMA,
                "status_at_c6": "running",
            },
            "run_id": run_id,
            "started_at": manifest["started_at"],
            "model_alias": model_alias,
            "model_id": model_id,
            "model_revision": revision,
            "tokenizer_id": model_id,
            "tokenizer_revision": revision,
            "run_config": {
                "path": str(config_path.resolve()),
                "sha256": config_sha256,
            },
            "prepared_inputs": {
                "path": str(prepared_dir.resolve()),
                "manifest_sha256": prepared_sha256,
                "index_sha256": index_sha256,
                "records": 1,
            },
            "source_code": {
                "commit": COMMIT,
                "dirty": False,
                "dirty_check_ignored_only": "runs/longbench16_24k/**",
            },
        },
    }
    c6_rows_path = (
        run_dir / "analysis" / finalize.C6_ROWS_FILENAME
    ).resolve()
    _write_jsonl(c6_rows_path, [c6_row])
    c6_summary_path = (
        run_dir / "analysis" / finalize.C6_SUMMARY_FILENAME
    ).resolve()
    c6_summary = copy.deepcopy(c6_summary_base)
    c6_summary["artifacts"] = {
        "per_prompt_jsonl": _declared_artifact(c6_rows_path, rows=1),
        "summary_json": {"path": str(c6_summary_path)},
    }
    atomic_write_json(c6_summary_path, c6_summary)
    pure_c6_summary = copy.deepcopy(c6_summary_base)
    pure_c6_summary.pop("run_identity")
    c6_result = SimpleNamespace(rows=(c6_row,), summary=pure_c6_summary)

    monkeypatch.setattr(finalize, "load_locked_run_config", lambda path: locked)
    monkeypatch.setattr(
        finalize,
        "load_prepared_dataset",
        lambda root, received_locked: prepared,
    )
    monkeypatch.setattr(
        finalize,
        "load_prepared_expectations",
        lambda root, require_complete: prepared_expectations,
    )
    monkeypatch.setattr(
        finalize,
        "validate_runtime_source_code",
        lambda received_prepared: {"commit": COMMIT, "dirty": False},
    )

    def fake_pilot_gate(layout, received_prepared):
        assert layout.run_dir == run_dir.resolve()
        assert received_prepared is prepared
        return {
            config: file_sha256(pilot_paths[config])
            for config in FIXED_CONFIGS
        }

    monkeypatch.setattr(
        finalize,
        "validate_worst_case_pilot_gate",
        fake_pilot_gate,
    )

    def fake_join(paths, *, prepared_dir, require_complete):
        assert require_complete is True
        assert Path(prepared_dir).resolve() == prepared.root
        assert {key: Path(value).resolve() for key, value in paths.items()} == (
            journal_paths
        )
        return join_result

    def fake_analysis(result, *, oracle_objective, require_complete):
        assert result is join_result
        assert oracle_objective == finalize.PRIMARY_ORACLE_OBJECTIVE
        assert require_complete is True
        return (joined_row,), copy.deepcopy(rebuilt_analysis)

    def fake_c6(joined, *, run_config):
        assert Path(joined).resolve() == joined_path
        assert Path(run_config).resolve() == config_path.resolve()
        return c6_result

    monkeypatch.setattr(finalize, "join_journals", fake_join)
    monkeypatch.setattr(finalize, "build_analysis", fake_analysis)
    monkeypatch.setattr(finalize, "evaluate_c6_file", fake_c6)

    return SimpleNamespace(
        run_dir=run_dir.resolve(),
        prepared_dir=prepared_dir.resolve(),
        config_path=config_path.resolve(),
        manifest_path=manifest_path.resolve(),
        environment_path=environment_path.resolve(),
        joined_path=joined_path,
        analysis_path=analysis_path,
        c6_rows_path=c6_rows_path,
        c6_summary_path=c6_summary_path,
        pilot_paths=pilot_paths,
    )


def test_finalization_is_atomic_idempotent_and_records_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(finalize, "_utc_now", lambda: "2026-07-20T13:00:00Z")

    completed = finalize.finalize_run(
        case.run_dir,
        case.prepared_dir,
        run_config=case.config_path,
    )
    first_bytes = case.manifest_path.read_bytes()

    assert completed["status"] == "complete"
    assert completed["started_at"] == "2026-07-20T12:00:00Z"
    assert completed["completed_at"] == "2026-07-20T13:00:00Z"
    finalization = completed["finalization"]
    assert finalization["schema_version"] == finalize.FINALIZATION_SCHEMA
    assert finalization["validation_status"] == "passed"
    assert (
        finalization["validations"]["fixed_configurations"][
            "completed_generation_records"
        ]
        == 6
    )
    assert finalization["validations"]["fixed_configurations"][
        "pilot_outputs_used_in_3750_example_metrics"
    ] is False
    assert (
        finalization["validations"]["worst_case_24k_512_pilot_gate"][
            "status"
        ]
        == "passed"
    )
    artifacts = finalization["artifacts"]
    assert set(artifacts["fixed_per_prompt_journals"]) == set(FIXED_CONFIGS)
    assert set(artifacts["worst_case_24k_512_pilot_journals"]) == set(
        FIXED_CONFIGS
    )
    assert len(artifacts["prepared_inputs"]["token_npz"]) == 16
    assert artifacts["joined_per_prompt"]["sha256"] == file_sha256(
        case.joined_path
    )
    assert artifacts["c6_per_prompt"]["sha256"] == file_sha256(
        case.c6_rows_path
    )
    assert not list(case.run_dir.glob(".manifest.json.*.tmp"))

    again = finalize.finalize_run(
        case.run_dir,
        case.prepared_dir,
        run_config=case.config_path,
    )
    assert again == completed
    assert case.manifest_path.read_bytes() == first_bytes


def test_finalizer_rejects_pilot_only_or_partial_canonical_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    missing = case.run_dir / "C5" / "per_prompt.jsonl"
    missing.unlink()
    pilot = case.run_dir / "pilots" / "C5" / "per_prompt.jsonl"
    _write_jsonl(
        pilot,
        [
            {
                "status": "completed",
                "configuration": "C5",
                "task": "toy",
                "benchmark_id": "toy:0",
                "pilot_profile": "worst_case_24k_512",
            }
        ],
    )

    with pytest.raises(
        finalize.FinalizationError,
        match="pilot-only or partial",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )
    assert json.loads(case.manifest_path.read_text())["status"] == "running"


def test_finalizer_requires_all_six_worst_case_pilot_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    case.pilot_paths["C3"].unlink()

    with pytest.raises(
        finalize.FinalizationError,
        match=r"C3 worst-case 24K\+512 pilot journal",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )
    assert json.loads(case.manifest_path.read_text())["status"] == "running"


def test_finalizer_rejects_stale_oracle_or_c6_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    summary = json.loads(case.analysis_path.read_text())
    summary["oracle_analysis"]["primary"]["name"] = "stale"
    atomic_write_json(case.analysis_path, summary, overwrite=True)

    with pytest.raises(
        finalize.FinalizationError,
        match="differs from independent recomputation",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )

    # Restore the join summary, then make C6 explicitly non-primary.
    case = _fixture(tmp_path / "second", monkeypatch)
    c6_summary = json.loads(case.c6_summary_path.read_text())
    c6_summary["evaluation_mode"] = "test_or_smoke"
    atomic_write_json(case.c6_summary_path, c6_summary, overwrite=True)
    with pytest.raises(
        finalize.FinalizationError,
        match="differs from an independent held-out recomputation",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )


def test_completed_manifest_fails_closed_after_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(finalize, "_utc_now", lambda: "2026-07-20T13:00:00Z")
    finalize.finalize_run(
        case.run_dir,
        case.prepared_dir,
        run_config=case.config_path,
    )

    _write_jsonl(
        case.c6_rows_path,
        [
            {
                "schema_version": finalize.C6_ROW_SCHEMA,
                "configuration": "C6",
                "task": "toy",
                "benchmark_id": "mutated",
            }
        ],
    )
    with pytest.raises(
        finalize.FinalizationError,
        match="differs from independent recomputation",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )
    assert json.loads(case.manifest_path.read_text())["status"] == "complete"


def test_c6_pretransition_run_identity_hash_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    summary = json.loads(case.c6_summary_path.read_text())
    summary["run_identity"]["run_manifest"]["sha256_at_c6"] = "d" * 64
    atomic_write_json(case.c6_summary_path, summary, overwrite=True)

    with pytest.raises(
        finalize.FinalizationError,
        match="C6 run_identity is missing, stale",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )


def test_completed_manifest_timestamp_mismatch_is_not_rewritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(finalize, "_utc_now", lambda: "2026-07-20T13:00:00Z")
    finalize.finalize_run(
        case.run_dir,
        case.prepared_dir,
        run_config=case.config_path,
    )
    manifest = json.loads(case.manifest_path.read_text())
    manifest["completed_at"] = "2026-07-20T14:00:00Z"
    atomic_write_json(case.manifest_path, manifest, overwrite=True)
    tampered_bytes = case.manifest_path.read_bytes()

    with pytest.raises(
        finalize.FinalizationError,
        match="stale, mutated, or not reproducible",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )
    assert case.manifest_path.read_bytes() == tampered_bytes


def test_environment_identity_and_canonical_manifest_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    environment = json.loads(case.environment_path.read_text())
    environment["git"]["commit"] = "c" * 40
    atomic_write_json(case.environment_path, environment, overwrite=True)
    with pytest.raises(
        finalize.FinalizationError,
        match="git state must be clean and match",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )

    case = _fixture(tmp_path / "noncanonical", monkeypatch)
    manifest = json.loads(case.manifest_path.read_text())
    case.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        finalize.FinalizationError,
        match="canonical atomic JSON encoding",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )


def test_canonical_journal_rejects_pilot_profile_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _fixture(tmp_path, monkeypatch)
    journal = case.run_dir / "C2" / "per_prompt.jsonl"
    row = json.loads(journal.read_text())
    row["method_metadata"] = {
        "pilot_profile": "worst_case_24k_512",
        "cuda_feasibility": {"headroom_sufficient": True},
    }
    _write_jsonl(journal, [row])

    with pytest.raises(
        finalize.FinalizationError,
        match="pilot-only metadata",
    ):
        finalize.finalize_run(
            case.run_dir,
            case.prepared_dir,
            run_config=case.config_path,
        )
