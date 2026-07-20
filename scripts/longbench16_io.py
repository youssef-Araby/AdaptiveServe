"""Crash-safe run I/O and integrity checks for the LongBench16 evaluation.

This module deliberately uses only the Python standard library.  It owns the
filesystem contract for new runs under::

    runs/longbench16_24k/<model-alias>/<run-id>/

The corrected P0 evidence under ``runs/p0`` is immutable.  Every write helper
in this module rejects a destination that resolves inside a ``runs/p0`` tree,
including destinations reached through a symlink.

JSONL journals are append-only event logs.  A prompt may have one completed
event, or one or more failed attempts followed by one completed event.  A
completed prompt can never be appended again.  This makes interrupted runs
resumable without truncating already completed records while retaining failed
attempts for provenance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Linux/WSL production environment; kept explicit rather than unsafe.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX systems.
    fcntl = None


FIXED_CONFIGS = ("C0", "C1", "C2", "C3", "C4", "C5")
ALL_CONFIGS = (*FIXED_CONFIGS, "C6")
KV_ACCOUNTING_BY_CONFIG = {
    "C0": {
        "kind": "physical_tensor_bytes",
        "quantization_metadata_included": False,
        "runtime_representation": "native_fp16_cache",
    },
    "C1": {
        "kind": "modeled_packed_bytes",
        "quantization_metadata_included": True,
        "runtime_representation": "round_trip_fp16_quantization_simulation",
    },
    "C2": {
        "kind": "modeled_packed_bytes",
        "quantization_metadata_included": True,
        "runtime_representation": "round_trip_fp16_quantization_simulation",
    },
    "C3": {
        "kind": "modeled_packed_bytes",
        "quantization_metadata_included": True,
        "runtime_representation": "round_trip_fp16_quantization_simulation",
    },
    "C4": {
        "kind": "physical_tensor_bytes",
        "quantization_metadata_included": False,
        "runtime_representation": "retained_native_fp16_cache",
    },
    "C5": {
        "kind": "physical_tensor_bytes",
        "quantization_metadata_included": False,
        "runtime_representation": "retained_native_fp16_cache",
    },
}
DEFAULT_KEY_FIELDS = ("task", "benchmark_id")
FINAL_INPUT_HASH_FIELD = "final_input_token_sha256"
VALID_STATUSES = frozenset({"completed", "failed"})
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_P0_ARCHIVE_ROOT = (_REPO_ROOT / "runs" / "p0").resolve()


class RunIntegrityError(RuntimeError):
    """Base class for LongBench16 run-integrity failures."""


class UnsafeOutputPathError(PermissionError, RunIntegrityError):
    """Raised when a caller attempts to write beneath ``runs/p0``."""


class ImmutableRunError(FileExistsError, RunIntegrityError):
    """Raised when a caller tries to reuse an existing run ID."""


class JournalCorruptionError(RunIntegrityError):
    """Raised when a JSONL journal is malformed or internally inconsistent."""


class DuplicateRecordError(RunIntegrityError):
    """Raised when a completed key or unapproved retry is appended twice."""


class InvalidStatusError(RunIntegrityError):
    """Raised when a journal event has an unsupported status."""


class ValidationError(RunIntegrityError):
    """Raised when records fail completeness or cross-config validation."""


RecordKey = tuple[Any, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_' or '-'."
        )
    if value in {".", ".."}:
        raise ValueError(f"{label} may not be {value!r}")
    return value


def _resolved(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _contains_runs_p0(path: Path) -> bool:
    lowered = tuple(part.lower() for part in path.parts)
    return any(
        lowered[index] == "runs" and lowered[index + 1] == "p0"
        for index in range(len(lowered) - 1)
    )


def assert_safe_write_path(path: Path | str) -> Path:
    """Return a resolved write path, rejecting any destination in ``runs/p0``."""

    resolved = _resolved(path)
    try:
        resolved.relative_to(_P0_ARCHIVE_ROOT)
    except ValueError:
        pass
    else:
        raise UnsafeOutputPathError(
            f"Refusing to write to immutable P0 evidence: {resolved}"
        )
    # Also protect temporary/test repositories and differently rooted clones.
    if _contains_runs_p0(resolved):
        raise UnsafeOutputPathError(
            f"Refusing to write to a runs/p0 namespace: {resolved}"
        )
    return resolved


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync after an atomic create/replace."""

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute a lowercase SHA-256 digest without loading the file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(
    path: Path | str,
    payload: Any,
    *,
    overwrite: bool = False,
    indent: int | None = 2,
) -> Path:
    """Atomically create or replace a JSON file.

    ``overwrite=False`` is race-safe: an existing destination is never
    replaced.  The temporary file is written and fsynced in the destination
    directory, then published with an atomic hard link.  ``overwrite=True``
    uses :func:`os.replace`, which is atomic on one filesystem.
    """

    destination = assert_safe_write_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp_path, destination)
        else:
            # link() fails atomically with FileExistsError if another process
            # published the destination first.
            os.link(temp_path, destination)
            temp_path.unlink()
        _fsync_directory(destination.parent)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True)
class RunLayout:
    """Paths belonging to one immutable model/run-ID namespace."""

    namespace_root: Path
    model_alias: str
    run_id: str

    @classmethod
    def create(
        cls,
        root: Path | str,
        model_alias: str,
        run_id: str,
    ) -> "RunLayout":
        """Create a fresh run directory and its stable subdirectories.

        The run ID is immutable: this method fails if its directory already
        exists.  Use :meth:`open` to resume an existing run.
        """

        namespace_root = assert_safe_write_path(root)
        model_alias = _validate_component(model_alias, "model_alias")
        run_id = _validate_component(run_id, "run_id")
        layout = cls(namespace_root, model_alias, run_id)
        assert_safe_write_path(layout.run_dir)

        layout.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            layout.run_dir.mkdir()
        except FileExistsError as exc:
            raise ImmutableRunError(
                f"run ID already exists and cannot be reused: {layout.run_dir}"
            ) from exc

        for directory in (
            *(layout.config_dir(config) for config in ALL_CONFIGS),
            layout.logs_dir,
            layout.pilots_dir,
            layout.analysis_dir,
        ):
            directory.mkdir()
        _fsync_directory(layout.model_dir)
        return layout

    @classmethod
    def open(
        cls,
        root: Path | str,
        model_alias: str,
        run_id: str,
    ) -> "RunLayout":
        """Open an existing run without creating or modifying anything."""

        namespace_root = _resolved(root)
        model_alias = _validate_component(model_alias, "model_alias")
        run_id = _validate_component(run_id, "run_id")
        layout = cls(namespace_root, model_alias, run_id)
        if not layout.run_dir.is_dir():
            raise FileNotFoundError(layout.run_dir)
        return layout

    @property
    def model_dir(self) -> Path:
        return self.namespace_root / self.model_alias

    @property
    def run_dir(self) -> Path:
        return self.model_dir / self.run_id

    @property
    def root(self) -> Path:
        """Alias for the concrete run directory."""

        return self.run_dir

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def environment_path(self) -> Path:
        return self.run_dir / "environment.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def pilots_dir(self) -> Path:
        return self.run_dir / "pilots"

    @property
    def analysis_dir(self) -> Path:
        return self.run_dir / "analysis"

    def config_dir(self, config: str) -> Path:
        if config not in ALL_CONFIGS:
            raise ValueError(f"unknown configuration {config!r}")
        return self.run_dir / config

    def records_path(self, config: str) -> Path:
        return self.config_dir(config) / "per_prompt.jsonl"

    def journal(self, config: str) -> "AtomicJsonlJournal":
        return AtomicJsonlJournal(self.records_path(config), DEFAULT_KEY_FIELDS)


def _key_from_record(record: Mapping[str, Any], key_fields: Sequence[str]) -> RecordKey:
    values: list[Any] = []
    for field in key_fields:
        if field not in record:
            raise JournalCorruptionError(f"record is missing key field {field!r}")
        value = record[field]
        if value is None or isinstance(value, (list, dict, set)):
            raise JournalCorruptionError(
                f"record key field {field!r} must be a non-null scalar"
            )
        try:
            hash(value)
        except TypeError as exc:
            raise JournalCorruptionError(
                f"record key field {field!r} is not hashable"
            ) from exc
        values.append(value)
    return tuple(values)


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _parse_jsonl_bytes(data: bytes, path: Path) -> list[dict[str, Any]]:
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise JournalCorruptionError(
            f"{path} has a partial trailing record; call "
            "repair_trailing_partial() before resuming"
        )
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), 1):
        if not raw_line.strip():
            raise JournalCorruptionError(
                f"{path}:{line_number} is an unexpected blank line"
            )
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalCorruptionError(
                f"{path}:{line_number} is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(record, dict):
            raise JournalCorruptionError(
                f"{path}:{line_number} must contain a JSON object"
            )
        records.append(record)
    return records


@dataclass(frozen=True)
class JournalSnapshot:
    """Indexed, status-aware view of an append-only journal."""

    path: Path
    key_fields: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    histories: Mapping[RecordKey, tuple[dict[str, Any], ...]]
    latest: Mapping[RecordKey, dict[str, Any]]
    completed: Mapping[RecordKey, dict[str, Any]]
    failed: Mapping[RecordKey, dict[str, Any]]
    duplicate_keys: frozenset[RecordKey]
    duplicate_completed_keys: frozenset[RecordKey]
    invalid_transition_keys: frozenset[RecordKey]
    status_counts: Mapping[str, int]

    @property
    def completed_keys(self) -> frozenset[RecordKey]:
        return frozenset(self.completed)

    @property
    def failed_keys(self) -> frozenset[RecordKey]:
        return frozenset(self.failed)

    @property
    def retried_keys(self) -> frozenset[RecordKey]:
        return self.duplicate_keys

    def ensure_consistent(self) -> None:
        problems: list[str] = []
        if self.duplicate_completed_keys:
            problems.append(
                "multiple completed events for "
                f"{sorted(map(str, self.duplicate_completed_keys))}"
            )
        if self.invalid_transition_keys:
            problems.append(
                "events appended after completion for "
                f"{sorted(map(str, self.invalid_transition_keys))}"
            )
        if problems:
            raise JournalCorruptionError(f"{self.path}: " + "; ".join(problems))


def _snapshot_from_records(
    path: Path,
    key_fields: tuple[str, ...],
    records: Sequence[dict[str, Any]],
) -> JournalSnapshot:
    histories_mut: dict[RecordKey, list[dict[str, Any]]] = {}
    status_counts: Counter[str] = Counter()
    for record in records:
        key = _key_from_record(record, key_fields)
        status = record.get("status")
        if status not in VALID_STATUSES:
            raise InvalidStatusError(
                f"{path}: key {key!r} has invalid status {status!r}; "
                f"expected one of {sorted(VALID_STATUSES)}"
            )
        history = histories_mut.setdefault(key, [])
        expected_attempt = len(history) + 1
        attempt = record.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise JournalCorruptionError(
                f"{path}: key {key!r} has invalid attempt {attempt!r}"
            )
        if attempt != expected_attempt:
            raise JournalCorruptionError(
                f"{path}: key {key!r} has attempt {attempt}, "
                f"expected {expected_attempt}"
            )
        history.append(record)
        status_counts[status] += 1

    histories = {key: tuple(value) for key, value in histories_mut.items()}
    latest = {key: history[-1] for key, history in histories.items()}
    duplicate_keys = frozenset(key for key, history in histories.items() if len(history) > 1)
    duplicate_completed = frozenset(
        key
        for key, history in histories.items()
        if sum(event["status"] == "completed" for event in history) > 1
    )
    invalid_transitions = frozenset(
        key
        for key, history in histories.items()
        if any(event["status"] == "completed" for event in history[:-1])
    )
    completed = {
        key: event
        for key, event in latest.items()
        if event["status"] == "completed"
        and key not in duplicate_completed
        and key not in invalid_transitions
    }
    failed = {
        key: event for key, event in latest.items() if event["status"] == "failed"
    }
    return JournalSnapshot(
        path=path,
        key_fields=key_fields,
        records=tuple(records),
        histories=histories,
        latest=latest,
        completed=completed,
        failed=failed,
        duplicate_keys=duplicate_keys,
        duplicate_completed_keys=duplicate_completed,
        invalid_transition_keys=invalid_transitions,
        status_counts=dict(status_counts),
    )


class AtomicJsonlJournal:
    """Append-only, fsynced JSONL journal keyed by selected record fields."""

    def __init__(
        self,
        path: Path | str,
        key_fields: Sequence[str] = DEFAULT_KEY_FIELDS,
    ):
        if (
            not key_fields
            or any(not isinstance(field, str) or not field for field in key_fields)
            or len(set(key_fields)) != len(key_fields)
        ):
            raise ValueError(
                "key_fields must be a non-empty sequence of unique non-empty strings"
            )
        self.path = Path(path)
        self.key_fields = tuple(key_fields)

    def snapshot(self) -> JournalSnapshot:
        if fcntl is None:  # pragma: no cover - read-only non-POSIX fallback.
            if not self.path.exists():
                return _snapshot_from_records(self.path, self.key_fields, ())
            records = _parse_jsonl_bytes(self.path.read_bytes(), self.path)
            return _snapshot_from_records(self.path, self.key_fields, records)
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return _snapshot_from_records(self.path, self.key_fields, ())
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            records = _parse_jsonl_bytes(_read_fd(fd), self.path)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return _snapshot_from_records(self.path, self.key_fields, records)

    def completed_keys(self) -> frozenset[RecordKey]:
        snapshot = self.snapshot()
        snapshot.ensure_consistent()
        return snapshot.completed_keys

    def failed_keys(self) -> frozenset[RecordKey]:
        snapshot = self.snapshot()
        snapshot.ensure_consistent()
        return snapshot.failed_keys

    def should_process(self, *key_values: Any) -> bool:
        if len(key_values) != len(self.key_fields):
            raise ValueError(
                f"expected {len(self.key_fields)} key values, got {len(key_values)}"
            )
        return tuple(key_values) not in self.completed_keys()

    def append(
        self,
        record: Mapping[str, Any],
        *,
        allow_retry: bool = False,
    ) -> dict[str, Any]:
        """Append one event and fsync it.

        A repeated key is allowed only when its latest event is ``failed`` and
        ``allow_retry=True``.  Nothing may follow a completed event.
        """

        if fcntl is None:  # pragma: no cover - project runs on Linux/WSL.
            raise RuntimeError("AtomicJsonlJournal requires POSIX fcntl locking")
        destination = assert_safe_write_path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate = dict(record)
        key = _key_from_record(candidate, self.key_fields)
        status = candidate.get("status")
        if status not in VALID_STATUSES:
            raise InvalidStatusError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}"
            )

        fd = os.open(destination, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            existing_records = _parse_jsonl_bytes(_read_fd(fd), destination)
            snapshot = _snapshot_from_records(
                destination, self.key_fields, existing_records
            )
            snapshot.ensure_consistent()
            history = snapshot.histories.get(key, ())
            if history:
                if history[-1]["status"] == "completed":
                    raise DuplicateRecordError(
                        f"key {key!r} is already completed in {destination}"
                    )
                if not allow_retry:
                    raise DuplicateRecordError(
                        f"key {key!r} already has a failed attempt; "
                        "set allow_retry=True to append a retry"
                    )

            expected_attempt = len(history) + 1
            supplied_attempt = candidate.get("attempt")
            if supplied_attempt is not None and supplied_attempt != expected_attempt:
                raise JournalCorruptionError(
                    f"key {key!r} supplied attempt {supplied_attempt!r}; "
                    f"expected {expected_attempt}"
                )
            candidate["attempt"] = expected_attempt
            candidate.setdefault("journaled_at", _utc_now())
            encoded = (
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:  # pragma: no cover - defensive OS failure.
                    raise OSError("zero-byte append while writing journal")
                view = view[written:]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        _fsync_directory(destination.parent)
        return candidate

    def repair_trailing_partial(self) -> int:
        """Repair a crash-truncated final line and return bytes removed.

        Earlier complete lines are validated before truncation.  A valid final
        JSON event missing only its newline terminator is preserved by adding
        that terminator.  Corruption in any complete line is never repaired
        automatically.
        """

        if fcntl is None:  # pragma: no cover
            raise RuntimeError("AtomicJsonlJournal requires POSIX fcntl locking")
        destination = assert_safe_write_path(self.path)
        if not destination.exists():
            return 0
        fd = os.open(destination, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            data = _read_fd(fd)
            if not data or data.endswith(b"\n"):
                return 0
            last_newline = data.rfind(b"\n")
            keep = last_newline + 1 if last_newline >= 0 else 0
            prefix = data[:keep]
            prefix_records = _parse_jsonl_bytes(prefix, destination)
            prefix_snapshot = _snapshot_from_records(
                destination, self.key_fields, prefix_records
            )
            prefix_snapshot.ensure_consistent()
            try:
                completed_records = _parse_jsonl_bytes(data + b"\n", destination)
                completed_snapshot = _snapshot_from_records(
                    destination, self.key_fields, completed_records
                )
                completed_snapshot.ensure_consistent()
            except (JournalCorruptionError, InvalidStatusError):
                pass
            else:
                os.lseek(fd, 0, os.SEEK_END)
                os.write(fd, b"\n")
                os.fsync(fd)
                return 0
            removed = len(data) - keep
            os.ftruncate(fd, keep)
            os.fsync(fd)
            return removed
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _normalize_expected_keys(
    expected_keys: Iterable[Sequence[Any] | Mapping[str, Any]],
    key_fields: tuple[str, ...],
) -> tuple[RecordKey, ...]:
    normalized: list[RecordKey] = []
    for item in expected_keys:
        if isinstance(item, Mapping):
            key = _key_from_record(item, key_fields)
        else:
            key = tuple(item)
            if len(key) != len(key_fields):
                raise ValueError(
                    f"expected key with {len(key_fields)} values, got {key!r}"
                )
        normalized.append(key)
    if len(normalized) != len(set(normalized)):
        raise ValidationError("expected_keys contains duplicates")
    return tuple(normalized)


@dataclass(frozen=True)
class ConfigValidation:
    config: str
    path: Path
    key_fields: tuple[str, ...]
    hash_field: str
    ordered_keys: tuple[RecordKey, ...]
    records: tuple[dict[str, Any], ...]
    records_by_key: Mapping[RecordKey, dict[str, Any]]
    file_sha256: str
    journal_event_count: int
    failed_attempt_count: int
    retried_keys: frozenset[RecordKey]

    @property
    def record_count(self) -> int:
        return len(self.records)


def validate_config_records(
    config: str,
    path: Path | str,
    expected_keys: Iterable[Sequence[Any] | Mapping[str, Any]],
    *,
    key_fields: Sequence[str] = DEFAULT_KEY_FIELDS,
    hash_field: str = FINAL_INPUT_HASH_FIELD,
) -> ConfigValidation:
    """Strictly validate one C0-C5 journal against the expected prompt keys."""

    if config not in FIXED_CONFIGS:
        raise ValueError(f"config must be one of {FIXED_CONFIGS}, got {config!r}")
    key_fields_tuple = tuple(key_fields)
    expected = _normalize_expected_keys(expected_keys, key_fields_tuple)
    expected_set = set(expected)
    journal = AtomicJsonlJournal(path, key_fields_tuple)
    snapshot = journal.snapshot()
    snapshot.ensure_consistent()

    actual_set = set(snapshot.histories)
    unexpected = actual_set - expected_set
    if unexpected:
        raise ValidationError(
            f"{config}: unexpected record keys: {sorted(map(str, unexpected))[:10]}"
        )
    missing = expected_set - set(snapshot.completed)
    if missing:
        failed = missing & set(snapshot.failed)
        never_seen = missing - failed
        raise ValidationError(
            f"{config}: {len(missing)} expected keys are not completed "
            f"({len(failed)} failed, {len(never_seen)} absent); "
            f"examples={sorted(map(str, missing))[:10]}"
        )

    records: list[dict[str, Any]] = []
    for key in expected:
        record = snapshot.completed[key]
        if record.get("configuration") != config:
            raise ValidationError(
                f"{config}: key {key!r} declares configuration "
                f"{record.get('configuration')!r}"
            )
        digest = record.get(hash_field)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValidationError(
                f"{config}: key {key!r} lacks valid {hash_field}"
            )
        records.append(record)

    record_path = Path(path)
    return ConfigValidation(
        config=config,
        path=record_path,
        key_fields=key_fields_tuple,
        hash_field=hash_field,
        ordered_keys=expected,
        records=tuple(records),
        records_by_key={
            key: record for key, record in zip(expected, records, strict=True)
        },
        file_sha256=file_sha256(record_path),
        journal_event_count=len(snapshot.records),
        failed_attempt_count=snapshot.status_counts.get("failed", 0),
        retried_keys=snapshot.retried_keys,
    )


@dataclass(frozen=True)
class CrossConfigValidation:
    configs: tuple[str, ...]
    ordered_keys: tuple[RecordKey, ...]
    input_hashes: Mapping[RecordKey, str]
    combined_input_hash_sha256: str
    record_file_sha256: Mapping[str, str]

    @property
    def record_count(self) -> int:
        return len(self.ordered_keys)


def validate_cross_config_hashes(
    validations: Mapping[str, ConfigValidation],
    *,
    configs: Sequence[str] = FIXED_CONFIGS,
) -> CrossConfigValidation:
    """Require identical keys and final-input token hashes across C0-C5."""

    config_tuple = tuple(configs)
    if config_tuple != FIXED_CONFIGS:
        raise ValueError(f"strict validation requires configs {FIXED_CONFIGS}")
    missing_configs = set(FIXED_CONFIGS) - set(validations)
    extra_configs = set(validations) - set(FIXED_CONFIGS)
    if missing_configs or extra_configs:
        raise ValidationError(
            "cross-config validation requires exactly C0-C5; "
            f"missing={sorted(missing_configs)}, extra={sorted(extra_configs)}"
        )

    baseline = validations["C0"]
    expected = baseline.ordered_keys
    for config in FIXED_CONFIGS:
        validation = validations[config]
        if validation.config != config:
            raise ValidationError(
                f"validation mapping key {config} contains {validation.config}"
            )
        if validation.ordered_keys != expected:
            raise ValidationError(f"{config}: ordered key set differs from C0")
        if validation.key_fields != baseline.key_fields:
            raise ValidationError(f"{config}: key fields differ from C0")
        if validation.hash_field != baseline.hash_field:
            raise ValidationError(f"{config}: input hash field differs from C0")
        try:
            current_file_hash = file_sha256(validation.path)
        except OSError as exc:
            raise ValidationError(
                f"{config}: validated journal is no longer readable: "
                f"{validation.path}"
            ) from exc
        if current_file_hash != validation.file_sha256:
            raise ValidationError(
                f"{config}: journal changed after validation: {validation.path}"
            )

    input_hashes: dict[RecordKey, str] = {}
    mismatches: list[str] = []
    for key in expected:
        by_config = {
            config: validations[config].records_by_key[key][baseline.hash_field].lower()
            for config in FIXED_CONFIGS
        }
        unique = set(by_config.values())
        if len(unique) != 1:
            mismatches.append(f"{key!r}: {by_config}")
            continue
        input_hashes[key] = next(iter(unique))
    if mismatches:
        raise ValidationError(
            f"final-input token hash mismatch for {len(mismatches)} keys; "
            + "; ".join(mismatches[:10])
        )

    combined = hashlib.sha256()
    for key in expected:
        canonical_key = json.dumps(
            list(key), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        combined.update(canonical_key.encode("utf-8"))
        combined.update(b"\t")
        combined.update(input_hashes[key].encode("ascii"))
        combined.update(b"\n")

    return CrossConfigValidation(
        configs=FIXED_CONFIGS,
        ordered_keys=expected,
        input_hashes=input_hashes,
        combined_input_hash_sha256=combined.hexdigest(),
        record_file_sha256={
            config: validations[config].file_sha256 for config in FIXED_CONFIGS
        },
    )


_DEFAULT_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "accelerate",
    "safetensors",
    "tokenizers",
    "numpy",
    "scipy",
    "scikit-learn",
    "rouge",
    "fuzzywuzzy",
)


def _git_metadata(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None:
        return {}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                ":(exclude)runs/longbench16_24k/**",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"commit": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"unavailable": True}


def _gpu_metadata() -> dict[str, Any]:
    query = (
        "index,name,uuid,driver_version,memory.total,"
        "memory.used,memory.free,compute_cap"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    devices = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) != 8:
            return {"available": False, "error": f"unexpected nvidia-smi row: {line}"}
        devices.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "driver_version": values[3],
                "memory_total_mib": int(values[4]),
                "memory_used_mib": int(values[5]),
                "memory_free_mib": int(values[6]),
                "compute_capability": values[7],
            }
        )
    return {"available": bool(devices), "devices": devices}


def collect_environment_metadata(
    *,
    repo_root: Path | str | None = _REPO_ROOT,
    package_names: Sequence[str] = _DEFAULT_PACKAGES,
    include_gpu: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without importing ML frameworks."""

    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    selected_environment = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TRANSFORMERS_CACHE",
        )
        if name in os.environ
    }
    metadata: dict[str, Any] = {
        "captured_at": _utc_now(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "environment": selected_environment,
        "git": _git_metadata(Path(repo_root).resolve() if repo_root else None),
    }
    if include_gpu:
        metadata["gpu"] = _gpu_metadata()
    if extra:
        metadata["extra"] = copy.deepcopy(dict(extra))
    return metadata


def artifact_metadata(
    path: Path | str,
    *,
    relative_to: Path | str | None = None,
) -> dict[str, Any]:
    """Return a manifest-ready path, size, and SHA-256 entry."""

    artifact = Path(path)
    display_path: str
    if relative_to is not None:
        try:
            display_path = str(artifact.resolve().relative_to(Path(relative_to).resolve()))
        except ValueError:
            display_path = str(artifact.resolve())
    else:
        display_path = str(artifact)
    return {
        "path": display_path,
        "bytes": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }


def finalize_manifest(
    manifest_path: Path | str,
    manifest: Mapping[str, Any],
    validations: Mapping[str, ConfigValidation],
    *,
    environment: Mapping[str, Any] | None = None,
    completed_at: str | None = None,
    overwrite_incomplete: bool = True,
) -> dict[str, Any]:
    """Validate C0-C5 and atomically publish a completed run manifest.

    An existing completed manifest is immutable.  An existing non-complete
    manifest may be atomically replaced during finalization.
    """

    destination = assert_safe_write_path(manifest_path)
    cross = validate_cross_config_hashes(validations)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(
                f"cannot replace unreadable existing manifest {destination}"
            ) from exc
        if existing.get("status") == "complete":
            raise ImmutableRunError(
                f"completed manifest is immutable: {destination}"
            )
        if not overwrite_incomplete:
            raise FileExistsError(destination)

    payload = copy.deepcopy(dict(manifest))
    payload["status"] = "complete"
    payload["completed_at"] = completed_at or _utc_now()
    if environment is not None:
        payload["environment"] = copy.deepcopy(dict(environment))
    payload["fixed_configurations"] = {
        config: {
            "record_count": validations[config].record_count,
            "journal_event_count": validations[config].journal_event_count,
            "failed_attempt_count": validations[config].failed_attempt_count,
            "retried_key_count": len(validations[config].retried_keys),
            "per_prompt": artifact_metadata(
                validations[config].path, relative_to=destination.parent
            ),
        }
        for config in FIXED_CONFIGS
    }
    payload["cross_config_validation"] = {
        "configurations": list(cross.configs),
        "record_count": cross.record_count,
        "key_fields": list(validations["C0"].key_fields),
        "final_input_hash_field": validations["C0"].hash_field,
        "combined_input_hash_sha256": cross.combined_input_hash_sha256,
        "status": "passed",
    }

    atomic_write_json(
        destination,
        payload,
        overwrite=destination.exists() and overwrite_incomplete,
    )
    return payload


__all__ = [
    "ALL_CONFIGS",
    "AtomicJsonlJournal",
    "ConfigValidation",
    "CrossConfigValidation",
    "DEFAULT_KEY_FIELDS",
    "DuplicateRecordError",
    "FINAL_INPUT_HASH_FIELD",
    "FIXED_CONFIGS",
    "ImmutableRunError",
    "InvalidStatusError",
    "JournalCorruptionError",
    "JournalSnapshot",
    "KV_ACCOUNTING_BY_CONFIG",
    "RunIntegrityError",
    "RunLayout",
    "UnsafeOutputPathError",
    "ValidationError",
    "artifact_metadata",
    "assert_safe_write_path",
    "atomic_write_json",
    "collect_environment_metadata",
    "file_sha256",
    "finalize_manifest",
    "validate_config_records",
    "validate_cross_config_hashes",
]
