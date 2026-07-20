from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import longbench16_io as run_io


EXPECTED_KEYS = (("qasper", 0), ("narrativeqa", 17))


def _input_hash(key: tuple[str, int]) -> str:
    return hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest()


def _completed_record(
    config: str,
    key: tuple[str, int],
    *,
    digest: str | None = None,
) -> dict[str, object]:
    return {
        "configuration": config,
        "task": key[0],
        "benchmark_id": key[1],
        "status": "completed",
        "final_input_token_sha256": digest or _input_hash(key),
        "prediction": "answer",
    }


def _build_validations(
    directory: Path,
    *,
    mismatch: tuple[str, tuple[str, int]] | None = None,
    retry: tuple[str, tuple[str, int]] | None = None,
) -> dict[str, run_io.ConfigValidation]:
    validations: dict[str, run_io.ConfigValidation] = {}
    for config in run_io.FIXED_CONFIGS:
        path = directory / config / "per_prompt.jsonl"
        journal = run_io.AtomicJsonlJournal(path, ("task", "benchmark_id"))
        for key in EXPECTED_KEYS:
            if retry == (config, key):
                journal.append(
                    {
                        "configuration": config,
                        "task": key[0],
                        "benchmark_id": key[1],
                        "status": "failed",
                        "error": "synthetic transient failure",
                    }
                )
            digest = "f" * 64 if mismatch == (config, key) else _input_hash(key)
            journal.append(
                _completed_record(config, key, digest=digest),
                allow_retry=retry == (config, key),
            )
        validations[config] = run_io.validate_config_records(
            config, path, EXPECTED_KEYS
        )
    return validations


def test_run_layout_create_open_and_run_id_immutability(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "longbench16_24k"
    layout = run_io.RunLayout.create(root, "llama-3.1-8b", "20260720T120000Z")

    assert layout.root == root.resolve() / "llama-3.1-8b" / "20260720T120000Z"
    assert layout.manifest_path == layout.root / "manifest.json"
    assert layout.environment_path == layout.root / "environment.json"
    assert layout.records_path("C0") == layout.root / "C0" / "per_prompt.jsonl"
    assert all(layout.config_dir(config).is_dir() for config in run_io.ALL_CONFIGS)
    assert layout.logs_dir.is_dir()
    assert layout.pilots_dir.is_dir()
    assert layout.analysis_dir.is_dir()

    reopened = run_io.RunLayout.open(root, "llama-3.1-8b", "20260720T120000Z")
    assert reopened == layout
    with pytest.raises(run_io.ImmutableRunError):
        run_io.RunLayout.create(root, "llama-3.1-8b", "20260720T120000Z")
    with pytest.raises(ValueError):
        run_io.RunLayout.create(root, "../escape", "another-run")


def test_every_write_api_rejects_runs_p0_including_symlinks(
    tmp_path: Path,
) -> None:
    p0 = tmp_path / "runs" / "p0"

    with pytest.raises(run_io.UnsafeOutputPathError):
        run_io.RunLayout.create(p0, "model", "run")
    with pytest.raises(run_io.UnsafeOutputPathError):
        run_io.atomic_write_json(p0 / "metadata.json", {"unsafe": True})
    with pytest.raises(run_io.UnsafeOutputPathError):
        run_io.AtomicJsonlJournal(p0 / "C0" / "records.jsonl").append(
            {
                "task": "qasper",
                "benchmark_id": 0,
                "status": "completed",
            }
        )

    p0.mkdir(parents=True)
    alias = tmp_path / "archive-alias"
    alias.symlink_to(p0, target_is_directory=True)
    with pytest.raises(run_io.UnsafeOutputPathError):
        run_io.atomic_write_json(alias / "metadata.json", {"unsafe": True})


def test_atomic_json_create_replace_and_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    run_io.atomic_write_json(path, {"value": 1})

    first_bytes = path.read_bytes()
    assert json.loads(first_bytes) == {"value": 1}
    assert run_io.file_sha256(path) == hashlib.sha256(first_bytes).hexdigest()
    with pytest.raises(FileExistsError):
        run_io.atomic_write_json(path, {"value": 2})

    run_io.atomic_write_json(path, {"value": 2}, overwrite=True)
    assert json.loads(path.read_text()) == {"value": 2}
    assert not list(tmp_path.glob(".metadata.json.*.tmp"))
    with pytest.raises(ValueError):
        run_io.file_sha256(path, chunk_size=0)


def test_journal_resume_duplicate_status_and_retry_contract(tmp_path: Path) -> None:
    path = tmp_path / "C0" / "per_prompt.jsonl"
    journal = run_io.AtomicJsonlJournal(path, ("task", "benchmark_id"))
    first = journal.append(
        {
            "task": "qasper",
            "benchmark_id": 0,
            "status": "completed",
            "payload": "preserve me",
        }
    )
    before_resume = path.read_bytes()
    journal.append(
        {
            "task": "narrativeqa",
            "benchmark_id": 17,
            "status": "failed",
            "error": "out of memory",
        }
    )

    resumed = run_io.AtomicJsonlJournal(path, ("task", "benchmark_id"))
    assert path.read_bytes().startswith(before_resume)
    assert first["attempt"] == 1
    assert resumed.should_process("qasper", 0) is False
    assert resumed.should_process("narrativeqa", 17) is True
    assert resumed.completed_keys() == {("qasper", 0)}
    assert resumed.failed_keys() == {("narrativeqa", 17)}

    with pytest.raises(run_io.DuplicateRecordError):
        resumed.append(
            {
                "task": "qasper",
                "benchmark_id": 0,
                "status": "completed",
            }
        )
    with pytest.raises(run_io.DuplicateRecordError):
        resumed.append(
            {
                "task": "narrativeqa",
                "benchmark_id": 17,
                "status": "completed",
            }
        )
    retry = resumed.append(
        {
            "task": "narrativeqa",
            "benchmark_id": 17,
            "status": "completed",
        },
        allow_retry=True,
    )
    assert retry["attempt"] == 2

    snapshot = resumed.snapshot()
    snapshot.ensure_consistent()
    assert snapshot.status_counts == {"completed": 2, "failed": 1}
    assert snapshot.retried_keys == {("narrativeqa", 17)}
    assert snapshot.completed_keys == set(EXPECTED_KEYS)
    assert len(snapshot.records) == 3
    with pytest.raises(run_io.InvalidStatusError):
        resumed.append(
            {
                "task": "musique",
                "benchmark_id": 2,
                "status": "running",
            }
        )


def test_partial_tail_is_detected_and_only_partial_bytes_are_repaired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "per_prompt.jsonl"
    journal = run_io.AtomicJsonlJournal(path)
    journal.append(
        {"task": "qasper", "benchmark_id": 0, "status": "completed"}
    )
    complete_prefix = path.read_bytes()
    partial = b'{"task":"narrativeqa"'
    with path.open("ab") as handle:
        handle.write(partial)

    with pytest.raises(run_io.JournalCorruptionError, match="partial trailing"):
        journal.snapshot()
    assert journal.repair_trailing_partial() == len(partial)
    assert path.read_bytes() == complete_prefix
    assert journal.snapshot().completed_keys == {("qasper", 0)}
    assert journal.repair_trailing_partial() == 0


def test_repair_preserves_a_complete_event_missing_only_its_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "per_prompt.jsonl"
    journal = run_io.AtomicJsonlJournal(path)
    event = journal.append(
        {"task": "qasper", "benchmark_id": 0, "status": "completed"}
    )
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))

    with pytest.raises(run_io.JournalCorruptionError, match="partial trailing"):
        journal.snapshot()
    assert journal.repair_trailing_partial() == 0
    assert path.read_bytes().endswith(b"\n")
    assert journal.snapshot().completed[("qasper", 0)] == event


def test_validation_requires_exact_completed_keys_and_valid_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "C0.jsonl"
    journal = run_io.AtomicJsonlJournal(path)
    journal.append(_completed_record("C0", EXPECTED_KEYS[0]))

    with pytest.raises(run_io.ValidationError, match="not completed"):
        run_io.validate_config_records("C0", path, EXPECTED_KEYS)

    journal.append(
        _completed_record("C0", EXPECTED_KEYS[1], digest="not-a-sha256")
    )
    with pytest.raises(run_io.ValidationError, match="lacks valid"):
        run_io.validate_config_records("C0", path, EXPECTED_KEYS)

    journal.append(
        _completed_record("C0", ("unexpected", 99)),
    )
    with pytest.raises(run_io.ValidationError, match="unexpected record keys"):
        run_io.validate_config_records("C0", path, EXPECTED_KEYS)


def test_cross_config_validation_accepts_retry_history_and_rejects_hash_drift(
    tmp_path: Path,
) -> None:
    valid = _build_validations(
        tmp_path / "valid", retry=("C1", EXPECTED_KEYS[1])
    )
    assert valid["C1"].journal_event_count == 3
    assert valid["C1"].failed_attempt_count == 1
    assert valid["C1"].retried_keys == {EXPECTED_KEYS[1]}

    cross = run_io.validate_cross_config_hashes(valid)
    assert cross.configs == run_io.FIXED_CONFIGS
    assert cross.record_count == len(EXPECTED_KEYS)
    assert cross.input_hashes == {
        key: _input_hash(key) for key in EXPECTED_KEYS
    }
    assert len(cross.combined_input_hash_sha256) == 64
    assert set(cross.record_file_sha256) == set(run_io.FIXED_CONFIGS)

    with valid["C2"].path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(run_io.ValidationError, match="changed after validation"):
        run_io.validate_cross_config_hashes(valid)

    mismatched = _build_validations(
        tmp_path / "mismatched", mismatch=("C5", EXPECTED_KEYS[0])
    )
    with pytest.raises(run_io.ValidationError, match="token hash mismatch"):
        run_io.validate_cross_config_hashes(mismatched)
    with pytest.raises(run_io.ValidationError, match="exactly C0-C5"):
        run_io.validate_cross_config_hashes(
            {key: value for key, value in valid.items() if key != "C5"}
        )


def test_finalize_manifest_is_validated_atomic_and_immutable(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "longbench16_24k" / "model" / "run"
    validations = _build_validations(
        run_dir, retry=("C1", EXPECTED_KEYS[1])
    )
    manifest_path = run_dir / "manifest.json"
    run_io.atomic_write_json(manifest_path, {"status": "running"})
    environment = {"python": {"version": "test"}, "gpu": {"name": "test"}}

    payload = run_io.finalize_manifest(
        manifest_path,
        {"model_alias": "model", "run_id": "run"},
        validations,
        environment=environment,
        completed_at="2026-07-20T12:00:00Z",
    )

    assert json.loads(manifest_path.read_text()) == payload
    assert payload["status"] == "complete"
    assert payload["completed_at"] == "2026-07-20T12:00:00Z"
    assert payload["environment"] == environment
    assert payload["cross_config_validation"]["status"] == "passed"
    assert payload["cross_config_validation"]["record_count"] == len(EXPECTED_KEYS)
    assert set(payload["fixed_configurations"]) == set(run_io.FIXED_CONFIGS)
    assert payload["fixed_configurations"]["C1"]["journal_event_count"] == 3
    assert payload["fixed_configurations"]["C1"]["failed_attempt_count"] == 1
    assert payload["fixed_configurations"]["C1"]["retried_key_count"] == 1
    for config in run_io.FIXED_CONFIGS:
        artifact = payload["fixed_configurations"][config]["per_prompt"]
        assert artifact["path"] == f"{config}/per_prompt.jsonl"
        assert artifact["sha256"] == run_io.file_sha256(
            validations[config].path
        )

    with pytest.raises(run_io.ImmutableRunError):
        run_io.finalize_manifest(manifest_path, {}, validations)


def test_environment_metadata_does_not_require_ml_imports_or_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    metadata = run_io.collect_environment_metadata(
        repo_root=None,
        package_names=("package-that-does-not-exist-adaptiveserve",),
        include_gpu=False,
        extra={"seed": 42},
    )

    assert metadata["packages"] == {
        "package-that-does-not-exist-adaptiveserve": None
    }
    assert metadata["environment"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert metadata["git"] == {}
    assert metadata["extra"] == {"seed": 42}
    assert "gpu" not in metadata
    assert metadata["python"]["version"]
