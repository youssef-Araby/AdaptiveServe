from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import longbench16_run_config as runner
from scripts.longbench16_io import AtomicJsonlJournal, file_sha256
from scripts.longbench16_prepare import (
    PREPARED_SCHEMA,
    TOKEN_HASH_ALGORITHM,
    token_ids_sha256,
)
from scripts.longbench16_protocol import (
    LONG_BENCH_DATASET_ID,
    LONG_BENCH_DATASET_REVISION,
    LONG_BENCH_RESOLVED_DATASET_ID,
    LONG_BENCH_SPLIT,
    config_file_hashes,
    protocol_config_hash,
)


def _small_locked_config() -> runner.LockedRunConfig:
    strict = runner.load_locked_run_config(runner.DEFAULT_RUN_CONFIG)
    payload = copy.deepcopy(dict(strict.payload))
    payload["input_policy"].pop("expected_primary_truncation_counts", None)
    payload["input_policy"].pop("expected_primary_total_truncated", None)
    return runner.LockedRunConfig(
        path=strict.path,
        sha256=strict.sha256,
        payload=payload,
    )


def _toy_spec(*, stop_on_newline: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        category="summarization",
        scorer="rouge_score",
        max_new_tokens=2,
        minimum_new_tokens=1 if stop_on_newline else 0,
        stop_on_newline_token=stop_on_newline,
        expected_test_examples=1,
    )


def _write_prepared_fixture(
    tmp_path: Path,
    *,
    task: str = "toy",
    stop_on_newline: bool = False,
    hash_field: str = "final_input_token_sha256",
) -> tuple[Path, runner.LockedRunConfig, dict[str, SimpleNamespace]]:
    locked = _small_locked_config()
    root = tmp_path / f"prepared-{task}-{hash_field}"
    token_dir = root / "tokens"
    token_dir.mkdir(parents=True)
    token_ids = np.asarray([7, 2, 8], dtype="<u4")
    offsets = np.asarray([0, 3], dtype="<u8")
    token_path = token_dir / f"{task}.npz"
    np.savez_compressed(token_path, input_ids=token_ids, offsets=offsets)
    digest = token_ids_sha256(token_ids)
    spec = _toy_spec(stop_on_newline=stop_on_newline)
    specs = {task: spec}

    row = {
        "schema_version": PREPARED_SCHEMA,
        "model": locked.model_alias,
        "model_id": locked.model["model_id"],
        "task": task,
        "category": spec.category,
        "benchmark_id": f"{task}-0",
        "source_index": 0,
        "metric": spec.scorer,
        "references": ["reference"],
        "all_classes": None,
        "features": {
            "seq_len_tokens": 3,
            "seq_len_chars": 12,
            "token_entropy": 1.5,
            "gzip_ratio": 1.1,
            "unique_token_ratio": 1.0,
            "question_position": None,
            "newline_density": 0.0,
        },
        "pre_truncation_token_count": 3,
        "post_truncation_token_count": 3,
        "truncated": False,
        hash_field: digest,
        "token_hash_algorithm": TOKEN_HASH_ALGORITHM,
        "max_new_tokens": spec.max_new_tokens,
        "minimum_new_tokens": spec.minimum_new_tokens,
        "stop_on_newline_token": spec.stop_on_newline_token,
        "token_file": f"tokens/{task}.npz",
        "token_offset_index": 0,
    }
    index_path = root / "index.jsonl"
    index_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": PREPARED_SCHEMA,
        "status": "complete",
        "benchmark": {
            "requested_dataset_id": LONG_BENCH_DATASET_ID,
            "resolved_dataset_id": LONG_BENCH_RESOLVED_DATASET_ID,
            "dataset_revision": LONG_BENCH_DATASET_REVISION,
            "split": LONG_BENCH_SPLIT,
            "source_hashes_verified_against_pinned_release": True,
        },
        "tokenizer": {
            "model_id": locked.model["tokenizer_id"],
            "requested_revision": locked.model["tokenizer_revision"],
            "tokenizer_class": "FakeTokenizer",
            "name_or_path": locked.model["tokenizer_id"],
            "vocab_size": 32,
            "eos_token_id": 9,
            "pad_token_id": None,
            "chat_template_sha256": hashlib.sha256(b"chat").hexdigest(),
        },
        "input_policy": {
            "cap": runner.FINAL_INPUT_TOKEN_CAP,
            "token_hash_algorithm": TOKEN_HASH_ALGORITHM,
            "decode_and_retokenize_after_truncation": False,
            "no_chat_add_special_tokens": True,
            "chat_template_date_string": "20 Jul 2026",
        },
        "records": 1,
        "truncation_counts": {task: 0},
        "total_truncated": 0,
        "prepared_index": {
            "path": "index.jsonl",
            "sha256": file_sha256(index_path),
            "rows": 1,
        },
        "token_files": [
            {
                "task": task,
                "path": f"tokens/{task}.npz",
                "sha256": file_sha256(token_path),
                "rows": 1,
                "stored_token_ids": 3,
            }
        ],
        "protocol_config_hash": protocol_config_hash(),
        "protocol_config_files": config_file_hashes(),
        "run_config": {
            "path": str(locked.path),
            "sha256": locked.sha256,
        },
        "source_code": {
            "commit": "f" * 40,
            "dirty": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    status = {
        "schema_version": PREPARED_SCHEMA,
        "status": "complete",
        "manifest_sha256": file_sha256(manifest_path),
    }
    (root / "status.json").write_text(
        json.dumps(status, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return root, locked, specs


def _load_small_prepared(
    root: Path,
    locked: runner.LockedRunConfig,
    specs: dict[str, SimpleNamespace],
) -> runner.PreparedDataset:
    return runner.load_prepared_dataset(
        root,
        locked,
        task_specs=specs,
        expected_total=1,
    )


def test_locked_run_config_accepts_repository_config_and_rejects_revision(
    tmp_path: Path,
) -> None:
    locked = runner.load_locked_run_config(runner.DEFAULT_RUN_CONFIG)
    assert locked.model_alias == "llama31_8b"
    assert locked.model["dtype"] == "float16"

    payload = copy.deepcopy(dict(locked.payload))
    payload["model"]["model_revision"] = "0" * 40
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.RunnerValidationError, match="model_revision"):
        runner.load_locked_run_config(path)

    payload = copy.deepcopy(dict(locked.payload))
    payload["input_policy"]["no_chat_add_special_tokens"] = False
    path = tmp_path / "tampered-input-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        runner.RunnerValidationError,
        match="no_chat_add_special_tokens",
    ):
        runner.load_locked_run_config(path)


@pytest.mark.parametrize(
    "hash_field",
    ["final_input_token_sha256", "final_input_ids_sha256"],
)
def test_prepared_loader_validates_canonical_and_legacy_input_hashes(
    tmp_path: Path,
    hash_field: str,
) -> None:
    root, locked, specs = _write_prepared_fixture(
        tmp_path, hash_field=hash_field
    )
    prepared = _load_small_prepared(root, locked, specs)

    assert prepared.examples[0].input_ids.tolist() == [7, 2, 8]
    assert prepared.examples[0].final_input_token_sha256 == token_ids_sha256(
        [7, 2, 8]
    )
    assert prepared.expected_keys == (("toy", "toy-0"),)


def test_prepared_loader_requires_verified_pinned_release_source_hashes(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark"]["source_hashes_verified_against_pinned_release"] = False
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["manifest_sha256"] = file_sha256(manifest_path)
    status_path.write_text(
        json.dumps(status, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        runner.PreparedInputValidationError,
        match="source hashes were not verified",
    ):
        _load_small_prepared(root, locked, specs)


def test_prepared_loader_fails_closed_on_index_and_npz_hash_drift(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    with (root / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(runner.RunnerValidationError, match="index sha256"):
        _load_small_prepared(root, locked, specs)

    root2, locked2, specs2 = _write_prepared_fixture(
        tmp_path, task="other"
    )
    with (root2 / "tokens" / "other.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(runner.RunnerValidationError, match="NPZ sha256"):
        _load_small_prepared(root2, locked2, specs2)


def test_cli_rejects_nonpilot_limit_and_legacy_environment() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--configuration",
                "C0",
                "--prepared-dir",
                "prepared",
                "--run-id",
                "run",
                "--limit",
                "1",
            ]
        )
    args = runner.parse_args(
        [
            "--configuration",
            "C0",
            "--prepared-dir",
            "prepared",
            "--run-id",
            "run",
            "--pilot",
            "--limit",
            "1",
        ]
    )
    assert args.pilot is True and args.limit == 1
    profile_args = runner.parse_args(
        [
            "--configuration",
            "C0",
            "--prepared-dir",
            "prepared",
            "--run-id",
            "run",
            "--pilot",
            "--pilot-profile",
            runner.PILOT_PROFILE_WORST_CASE_24K_512,
        ]
    )
    assert (
        profile_args.pilot_profile
        == runner.PILOT_PROFILE_WORST_CASE_24K_512
    )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--configuration",
                "C0",
                "--prepared-dir",
                "prepared",
                "--run-id",
                "run",
                "--pilot-profile",
                runner.PILOT_PROFILE_WORST_CASE_24K_512,
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--configuration",
                "C0",
                "--prepared-dir",
                "prepared",
                "--run-id",
                "run",
                "--pilot",
                "--pilot-profile",
                runner.PILOT_PROFILE_WORST_CASE_24K_512,
                "--limit",
                "1",
            ]
        )
    with pytest.raises(runner.RunnerValidationError, match="ADAPTIVESERVE_LB_N"):
        runner.reject_legacy_sample_limit({"ADAPTIVESERVE_LB_N": "100"})


def test_layout_resume_and_pilot_journal_never_touch_final_namespace(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    run_root = tmp_path / "runs" / "longbench16_24k"
    layout = runner.open_or_create_run_layout(
        run_root=run_root,
        run_id="test-run",
        locked=locked,
        prepared=prepared,
        environment_factory=lambda **kwargs: {"environment": "test"},
    )
    reopened = runner.open_or_create_run_layout(
        run_root=run_root,
        run_id="test-run",
        locked=locked,
        prepared=prepared,
        environment_factory=lambda **kwargs: {"unused": True},
    )

    assert reopened == layout
    pilot = runner.journal_for_configuration(layout, "C0", pilot=True)
    final = runner.journal_for_configuration(layout, "C0", pilot=False)
    assert pilot.path == layout.pilots_dir / "C0" / "per_prompt.jsonl"
    assert final.path == layout.records_path("C0")
    assert pilot.path != final.path
    profiled = runner.journal_for_configuration(
        layout,
        "C0",
        pilot=True,
        pilot_profile=runner.PILOT_PROFILE_WORST_CASE_24K_512,
    )
    assert profiled.path == (
        layout.pilots_dir
        / runner.PILOT_PROFILE_WORST_CASE_24K_512
        / "C0"
        / "per_prompt.jsonl"
    )
    assert profiled.path not in {pilot.path, final.path}

    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    layout.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        runner.RunnerValidationError,
        match="completed run identities are immutable",
    ):
        runner.open_or_create_run_layout(
            run_root=run_root,
            run_id="test-run",
            locked=locked,
            prepared=prepared,
            environment_factory=lambda **kwargs: {"unused": True},
        )

    with pytest.raises(Exception):
        runner.open_or_create_run_layout(
            run_root=tmp_path / "runs" / "p0",
            run_id="unsafe",
            locked=locked,
            prepared=prepared,
            environment_factory=lambda **kwargs: {},
        )


def test_runtime_source_must_be_clean_and_match_preparation(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    expected = {"commit": "f" * 40, "dirty": False}

    assert runner.validate_runtime_source_code(
        prepared, current_state=expected
    ) == expected
    with pytest.raises(
        runner.RunnerValidationError, match="clean source tree"
    ):
        runner.validate_runtime_source_code(
            prepared,
            current_state={"commit": "f" * 40, "dirty": True},
        )
    with pytest.raises(
        runner.RunnerValidationError, match="differs from"
    ):
        runner.validate_runtime_source_code(
            prepared,
            current_state={"commit": "e" * 40, "dirty": False},
        )


def test_worst_case_pilot_profile_selects_first_exact_24k_512_example(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    with pytest.raises(
        runner.RunnerValidationError,
        match="exactly 24,000 input tokens",
    ):
        runner.select_prepared_examples(
            prepared,
            pilot=True,
            limit=None,
            pilot_profile=runner.PILOT_PROFILE_WORST_CASE_24K_512,
        )

    worst = replace(
        prepared.examples[0],
        pre_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        post_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        truncated=False,
        max_new_tokens=512,
    )
    profiled = replace(prepared, examples=(worst,))
    selected = runner.select_prepared_examples(
        profiled,
        pilot=True,
        limit=None,
        pilot_profile=runner.PILOT_PROFILE_WORST_CASE_24K_512,
    )
    assert selected == (worst,)


class _RuntimeTokenizer:
    eos_token_id = 9
    pad_token_id = None
    vocab_size = 32
    chat_template = "chat"

    def __init__(self) -> None:
        self.newline_calls = 0

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == "\n":
            self.newline_calls += 1
            return [3, 10]
        return list(range(200))

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        return "decoded"


def test_journal_records_canonical_fields_samsum_stop_and_exact_resume(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(
        tmp_path, task="samsum", stop_on_newline=True
    )
    prepared = _load_small_prepared(root, locked, specs)
    tokenizer = _RuntimeTokenizer()
    captured = {}

    def generate(
        model,
        tokenizer_arg,
        input_ids,
        max_new_tokens,
        device,
        **kwargs,
    ):
        captured.update(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
            kwargs=kwargs,
        )
        return "\nraw\nignored", 1, {
            "kv_bytes": 50.0,
            "kv_bytes_fp16": 100.0,
            "compression": 2.0,
        }

    journal = AtomicJsonlJournal(tmp_path / "journal.jsonl")
    dispatch = runner.GenerationDispatch(generate, {}, {"method": "test"})
    summary = runner.run_prepared_journal(
        configuration="C0",
        prepared=prepared,
        journal=journal,
        dispatch=dispatch,
        model=object(),
        tokenizer=tokenizer,
        device="cpu",
        pilot=False,
        score_fn=lambda task, prediction, references, **kwargs: 0.75,
        postprocess_fn=lambda task, prediction: prediction.lstrip().splitlines()[0],
    )

    assert summary.completed_now == 1
    assert captured["input_ids"] == [7, 2, 8]
    assert captured["kwargs"]["stop_token_ids"] == frozenset({9, 10})
    assert captured["kwargs"]["min_new_tokens"] == 1
    record = journal.snapshot().completed[("samsum", "samsum-0")]
    assert record["generated_token_count"] == 1
    assert record["final_input_token_sha256"] == token_ids_sha256([7, 2, 8])
    assert record["model"] == locked.model_alias
    assert record["model_id"] == locked.model["model_id"]
    assert record["category"] == specs["samsum"].category
    assert record["all_classes"] is None
    assert record["token_hash_algorithm"] == TOKEN_HASH_ALGORITHM
    assert record["minimum_new_tokens"] == 1
    assert record["stop_on_newline_token"] is True
    assert record["effective_stop_token_ids"] == [9, 10]
    assert record["prediction"] == "raw"
    assert record["score"] == 0.75
    assert record["kv_bytes"] == 50.0
    assert record["kv_bytes_fp16"] == 100.0
    assert record["compression"] == 2.0
    assert record["status"] == "completed"

    resumed = runner.run_prepared_journal(
        configuration="C0",
        prepared=prepared,
        journal=journal,
        dispatch=runner.GenerationDispatch(
            lambda *args, **kwargs: pytest.fail("completed key reran"),
            {},
            {},
        ),
        model=object(),
        tokenizer=tokenizer,
        device="cpu",
        pilot=False,
        score_fn=lambda *args, **kwargs: 1.0,
        postprocess_fn=lambda task, prediction: prediction,
    )
    assert resumed.completed_now == 0
    assert resumed.skipped_completed == 1
    assert len(journal.snapshot().records) == 1


def test_failed_record_is_append_only_and_successful_resume_is_attempt_two(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    journal = AtomicJsonlJournal(tmp_path / "retry.jsonl")
    tokenizer = _RuntimeTokenizer()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        runner.run_prepared_journal(
            configuration="C3",
            prepared=prepared,
            journal=journal,
            dispatch=runner.GenerationDispatch(fail, {}, {}),
            model=object(),
            tokenizer=tokenizer,
            device="cpu",
            pilot=False,
            score_fn=lambda *args, **kwargs: 1.0,
            postprocess_fn=lambda task, prediction: prediction,
        )
    failed = journal.snapshot().failed[("toy", "toy-0")]
    assert failed["status"] == "failed"
    assert failed["prediction"] is None
    assert failed["generated_token_count"] is None
    assert failed["final_input_token_sha256"] == token_ids_sha256([7, 2, 8])
    assert failed["model"] == locked.model_alias
    assert failed["model_id"] == locked.model["model_id"]
    assert failed["category"] == specs["toy"].category
    assert failed["all_classes"] is None
    assert failed["token_hash_algorithm"] == TOKEN_HASH_ALGORITHM
    assert failed["minimum_new_tokens"] == 0
    assert failed["stop_on_newline_token"] is False

    success = lambda *args, **kwargs: (
        "ok",
        1,
        {"kv_bytes": 100.0, "kv_bytes_fp16": 200.0, "compression": 2.0},
    )
    runner.run_prepared_journal(
        configuration="C3",
        prepared=prepared,
        journal=journal,
        dispatch=runner.GenerationDispatch(success, {}, {}),
        model=object(),
        tokenizer=tokenizer,
        device="cpu",
        pilot=False,
        score_fn=lambda *args, **kwargs: 1.0,
        postprocess_fn=lambda task, prediction: prediction,
    )
    history = journal.snapshot().histories[("toy", "toy-0")]
    assert [record["status"] for record in history] == ["failed", "completed"]
    assert [record["attempt"] for record in history] == [1, 2]


def test_samsum_cannot_complete_below_prepared_minimum_new_tokens(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(
        tmp_path, task="samsum", stop_on_newline=True
    )
    prepared = _load_small_prepared(root, locked, specs)
    journal = AtomicJsonlJournal(tmp_path / "minimum.jsonl")
    result = (
        "",
        0,
        {"kv_bytes": 100.0, "kv_bytes_fp16": 200.0, "compression": 2.0},
    )

    with pytest.raises(
        runner.RunnerValidationError,
        match="minimum/maximum bounds",
    ):
        runner.run_prepared_journal(
            configuration="C0",
            prepared=prepared,
            journal=journal,
            dispatch=runner.GenerationDispatch(
                lambda *args, **kwargs: result, {}, {}
            ),
            model=object(),
            tokenizer=_RuntimeTokenizer(),
            device="cpu",
            pilot=False,
            score_fn=lambda *args, **kwargs: 1.0,
            postprocess_fn=lambda task, prediction: prediction,
        )

    failed = journal.snapshot().failed[("samsum", "samsum-0")]
    assert failed["status"] == "failed"
    assert failed["generated_token_count"] is None


def test_cuda_feasibility_probe_resets_peaks_and_computes_headroom() -> None:
    gib = 1024**3
    calls = []

    class Cuda:
        @staticmethod
        def synchronize(device):
            calls.append(("synchronize", device))

        @staticmethod
        def mem_get_info(device):
            calls.append(("mem_get_info", device))
            return 6 * gib, 10 * gib

        @staticmethod
        def memory_reserved(device):
            calls.append(("memory_reserved", device))
            return 3 * gib

        @staticmethod
        def reset_peak_memory_stats(device):
            calls.append(("reset_peak_memory_stats", device))

        @staticmethod
        def max_memory_allocated(device):
            calls.append(("max_memory_allocated", device))
            return 6 * gib

        @staticmethod
        def max_memory_reserved(device):
            calls.append(("max_memory_reserved", device))
            return 7 * gib

    probe = runner.CudaFeasibilityProbe(
        SimpleNamespace(cuda=Cuda), "cuda:0"
    )
    probe.begin()
    measurement = probe.finish()

    assert measurement["total_vram_bytes"] == 10 * gib
    assert measurement["pre_run_non_torch_bytes"] == gib
    assert measurement["peak_torch_allocated_bytes"] == 6 * gib
    assert measurement["peak_torch_reserved_bytes"] == 7 * gib
    assert measurement["conservative_headroom_bytes"] == 2 * gib
    assert measurement["headroom_sufficient"] is True
    assert measurement["feasibility_only"] is True
    assert [name for name, _ in calls] == [
        "synchronize",
        "mem_get_info",
        "memory_reserved",
        "reset_peak_memory_stats",
        "synchronize",
        "synchronize",
        "max_memory_allocated",
        "max_memory_reserved",
    ]


def _profile_measurement(*, headroom_bytes: int) -> dict[str, object]:
    gib = 1024**3
    total = 10 * gib
    non_torch = gib
    peak_reserved = total - non_torch - headroom_bytes
    return {
        "total_vram_bytes": total,
        "pre_run_free_bytes": 6 * gib,
        "pre_run_torch_reserved_bytes": 3 * gib,
        "pre_run_non_torch_bytes": non_torch,
        "peak_torch_allocated_bytes": peak_reserved - gib,
        "peak_torch_reserved_bytes": peak_reserved,
        "conservative_headroom_bytes": headroom_bytes,
        "required_headroom_bytes": runner.WORST_CASE_MIN_HEADROOM_BYTES,
        "headroom_sufficient": (
            headroom_bytes >= runner.WORST_CASE_MIN_HEADROOM_BYTES
        ),
        "feasibility_only": True,
    }


def test_worst_case_profile_records_cuda_feasibility_metadata(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    worst = replace(
        prepared.examples[0],
        pre_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        post_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        max_new_tokens=512,
    )
    prepared = replace(prepared, examples=(worst,))
    state = {"begun": False}

    class Probe:
        @staticmethod
        def begin():
            state["begun"] = True

        @staticmethod
        def finish():
            return _profile_measurement(headroom_bytes=2 * 1024**3)

    def generate(*args, **kwargs):
        assert state["begun"] is True
        assert kwargs["stop_token_ids"] == frozenset()
        return (
            "ok",
            512,
            {
                "kv_bytes": 100.0,
                "kv_bytes_fp16": 200.0,
                "compression": 2.0,
            },
        )

    journal = AtomicJsonlJournal(tmp_path / "profile.jsonl")
    runner.run_prepared_journal(
        configuration="C0",
        prepared=prepared,
        journal=journal,
        dispatch=runner.GenerationDispatch(generate, {}, {}),
        model=object(),
        tokenizer=_RuntimeTokenizer(),
        device="cuda:0",
        pilot=True,
        pilot_profile=runner.PILOT_PROFILE_WORST_CASE_24K_512,
        cuda_feasibility_factory=lambda device: Probe(),
        score_fn=lambda *args, **kwargs: 1.0,
        postprocess_fn=lambda task, prediction: prediction,
    )

    record = journal.snapshot().completed[worst.key]
    assert (
        record["pilot_profile"]
        == runner.PILOT_PROFILE_WORST_CASE_24K_512
    )
    metadata = record["method_metadata"]
    assert (
        metadata["pilot_profile"]
        == runner.PILOT_PROFILE_WORST_CASE_24K_512
    )
    assert metadata["forced_full_decode"] is True
    feasibility = metadata["cuda_feasibility"]
    assert feasibility["conservative_headroom_bytes"] == 2 * 1024**3
    assert feasibility["headroom_sufficient"] is True
    assert "latency" not in json.dumps(metadata).lower()


def test_worst_case_profile_fails_below_required_cuda_headroom(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    worst = replace(
        prepared.examples[0],
        pre_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        post_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        max_new_tokens=512,
    )
    prepared = replace(prepared, examples=(worst,))

    class Probe:
        @staticmethod
        def begin():
            return None

        @staticmethod
        def finish():
            return _profile_measurement(headroom_bytes=1024**3)

    journal = AtomicJsonlJournal(tmp_path / "low-headroom.jsonl")
    with pytest.raises(
        runner.RunnerValidationError,
        match="less than 1.5 GiB",
    ):
        runner.run_prepared_journal(
            configuration="C0",
            prepared=prepared,
            journal=journal,
            dispatch=runner.GenerationDispatch(
                lambda *args, **kwargs: (
                    "ok",
                    1,
                    {
                        "kv_bytes": 100.0,
                        "kv_bytes_fp16": 200.0,
                        "compression": 2.0,
                    },
                ),
                {},
                {},
            ),
            model=object(),
            tokenizer=_RuntimeTokenizer(),
            device="cuda:0",
            pilot=True,
            pilot_profile=runner.PILOT_PROFILE_WORST_CASE_24K_512,
            cuda_feasibility_factory=lambda device: Probe(),
            score_fn=lambda *args, **kwargs: 1.0,
            postprocess_fn=lambda task, prediction: prediction,
        )

    failed = journal.snapshot().failed[worst.key]
    feasibility = failed["method_metadata"]["cuda_feasibility"]
    assert feasibility["headroom_sufficient"] is False
    assert feasibility["conservative_headroom_bytes"] == 1024**3


def test_final_generation_requires_all_six_worst_case_pilots(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    worst = replace(
        prepared.examples[0],
        pre_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        post_truncation_token_count=runner.FINAL_INPUT_TOKEN_CAP,
        max_new_tokens=512,
    )
    prepared = replace(prepared, examples=(worst,))
    layout = runner.RunLayout.create(
        tmp_path / "runs",
        locked.model_alias,
        "pilot-gate",
    )

    with pytest.raises(
        runner.RunnerValidationError, match="pilot passes for C0"
    ):
        runner.validate_worst_case_pilot_gate(layout, prepared)

    for configuration in runner.FIXED_CONFIGS:
        record = runner._base_record(configuration, worst)
        record.update(
            {
                "pilot_profile": (
                    runner.PILOT_PROFILE_WORST_CASE_24K_512
                ),
                "prediction": "feasibility-only",
                "score": 0.0,
                "generated_token_count": 512,
                "kv_bytes": 100.0,
                "kv_bytes_fp16": 100.0,
                "compression": 1.0,
                "status": "completed",
                "method_metadata": {
                    "pilot_profile": (
                        runner.PILOT_PROFILE_WORST_CASE_24K_512
                    ),
                    "forced_full_decode": True,
                    "cuda_feasibility": _profile_measurement(
                        headroom_bytes=2 * 1024**3
                    ),
                },
            }
        )
        runner.journal_for_configuration(
            layout,
            configuration,
            pilot=True,
            pilot_profile=runner.PILOT_PROFILE_WORST_CASE_24K_512,
        ).append(record)

    hashes = runner.validate_worst_case_pilot_gate(layout, prepared)
    assert set(hashes) == set(runner.FIXED_CONFIGS)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_c2_dispatch_validates_reproducible_calibration_hashes() -> None:
    locked = _small_locked_config()
    payload = copy.deepcopy(dict(locked.payload))
    text = "versioned calibration text"
    calibration_ids = list(range(128))
    method = payload["configurations"]["C2"]
    method.update(
        q_norm_calibration_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        q_norm_calibration_token_sha256=token_ids_sha256(calibration_ids),
    )
    locked = runner.LockedRunConfig(locked.path, locked.sha256, payload)
    captured = {}

    class Module:
        SPEED_TEXT = text
        QAQ_OUTLIER_RATIO = method["outlier_fraction"]
        QAQ_N_BITS_MIN = method["minimum_bits"]
        QAQ_N_BITS_MAX = method["maximum_bits"]
        QAQ_LAST_N_ATTENTIONS = method["last_attention_queries"]
        QAQ_TARGET_ERROR = method["target_error"]
        QAQ_Q_NORM_PERCENTILE = method["q_norm_percentile"]

        @staticmethod
        def _precompute_q_norm(
            model, input_ids, device, *, return_metadata
        ):
            assert return_metadata is True
            captured["ids"] = input_ids.clone()
            captured["device"] = device
            return 12.5, {
                "captured_layers": 2,
                "expected_layers": 2,
                "captured_q_norm_values": 256,
            }

        @staticmethod
        def generate_prepared_c2(*args, **kwargs):
            raise AssertionError("not called")

    class Tokenizer:
        @staticmethod
        def encode(value, *, add_special_tokens):
            assert value == text
            assert add_special_tokens is False
            return list(range(200))

    dispatch = runner.build_generation_dispatch(
        configuration="C2",
        locked=locked,
        model=object(),
        tokenizer=Tokenizer(),
        device=torch.device("cpu"),
        torch_module=torch,
        modules={"C2": Module},
    )

    assert captured["ids"].shape == (1, 128)
    assert dispatch.kwargs["q_norm"] == 12.5
    assert dispatch.kwargs["attn_aware_decode"] is True
    assert dispatch.metadata["q_norm_calibration_text_sha256"] == hashlib.sha256(
        text.encode()
    ).hexdigest()
    assert dispatch.metadata["q_norm_calibration_token_sha256"] == (
        token_ids_sha256(calibration_ids)
    )
    assert dispatch.metadata["q_norm_calibration_coverage"] == {
        "captured_layers": 2,
        "expected_layers": 2,
        "captured_q_norm_values": 256,
    }


def test_c4_c5_dispatch_derive_locked_budgets_and_head_count() -> None:
    locked = _small_locked_config()

    class C4:
        DKV_WINDOW_SIZE = 32
        DKV_KERNEL_SIZE = 7
        DKV_RATIO_MAX = 10
        generate_prepared_c4 = staticmethod(lambda *args, **kwargs: None)

    class C5:
        ADA_WINDOW_SIZE = 32
        ADA_KERNEL_SIZE = 7
        generate_prepared_c5 = staticmethod(lambda *args, **kwargs: None)

        @staticmethod
        def _get_n_kv_heads(model):
            return 8

    c4 = runner.build_generation_dispatch(
        configuration="C4",
        locked=locked,
        model=object(),
        tokenizer=object(),
        device="cpu",
        torch_module=torch,
        modules={"C4": C4},
    )
    c5 = runner.build_generation_dispatch(
        configuration="C5",
        locked=locked,
        model=object(),
        tokenizer=object(),
        device="cpu",
        torch_module=torch,
        modules={"C5": C5},
    )
    assert c4.kwargs == {"budget": 1024}
    assert c5.kwargs == {"n_kv_heads": 8, "budget_per_head": 1024}


def test_model_loader_is_offline_fp16_without_device_map_or_offload() -> None:
    locked = _small_locked_config()
    prepared_manifest = {
        "tokenizer": {
            "eos_token_id": 9,
            "vocab_size": 32,
            "chat_template_sha256": hashlib.sha256(b"chat").hexdigest(),
        }
    }
    selected_device = SimpleNamespace(type="cuda", index=1)

    class FakeCuda:
        selected = None

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @classmethod
        def set_device(cls, device):
            cls.selected = device

    class FakeTorch:
        float16 = object()
        cuda = FakeCuda

        @staticmethod
        def device(name):
            assert name == "cuda:1"
            return selected_device

    class FakeTokenizer:
        eos_token_id = 9
        pad_token_id = None
        vocab_size = 32
        chat_template = "chat"
        _commit_hash = locked.model["tokenizer_revision"]

    class TokenizerFactory:
        kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.kwargs = {"model_id": model_id, **kwargs}
            return FakeTokenizer()

    class Tensor:
        device = selected_device

    class Model:
        training = True
        hf_device_map = None
        config = SimpleNamespace(_commit_hash=locked.model["model_revision"])
        generation_config = SimpleNamespace(
            eos_token_id=list(runner.LOCKED_NATIVE_STOP_TOKEN_IDS)
        )

        def to(self, device):
            assert device is selected_device
            return self

        def eval(self):
            self.training = False
            return self

        def named_parameters(self):
            return [("weight", Tensor())]

        def named_buffers(self):
            return [("buffer", Tensor())]

        def modules(self):
            return [self]

    class ModelFactory:
        kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.kwargs = {"model_id": model_id, **kwargs}
            return Model()

    model, tokenizer, device = runner.load_model_and_tokenizer(
        locked=locked,
        prepared_manifest=prepared_manifest,
        configuration="C4",
        device_name="cuda:1",
        allow_download=False,
        torch_module=FakeTorch,
        auto_model_cls=ModelFactory,
        auto_tokenizer_cls=TokenizerFactory,
    )

    assert device is selected_device
    assert model.training is False
    assert tokenizer.pad_token_id == 9
    assert TokenizerFactory.kwargs["local_files_only"] is True
    assert ModelFactory.kwargs["dtype"] is FakeTorch.float16
    assert ModelFactory.kwargs["attn_implementation"] == "sdpa"
    assert "device_map" not in ModelFactory.kwargs
    assert not any("offload" in key for key in ModelFactory.kwargs)


def test_full_native_eos_set_is_used_and_tokenizer_eos_must_be_present(
    tmp_path: Path,
) -> None:
    root, locked, specs = _write_prepared_fixture(tmp_path)
    prepared = _load_small_prepared(root, locked, specs)
    example = prepared.examples[0]
    tokenizer = _RuntimeTokenizer()
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[7, 8, 9])
    )

    assert runner.stop_token_ids_for_example(
        example, tokenizer, model=model
    ) == frozenset({7, 8, 9})
    model.generation_config.eos_token_id = [7, 8]
    with pytest.raises(
        runner.RunnerValidationError, match="tokenizer EOS.*absent"
    ):
        runner.stop_token_ids_for_example(example, tokenizer, model=model)
