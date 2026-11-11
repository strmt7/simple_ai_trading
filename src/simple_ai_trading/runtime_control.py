"""Shared durable lease, pause and Stop ordering for venue execution adapters."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import Lock, RLock
import time
from typing import BinaryIO, Callable, Iterator, Mapping
import uuid
from weakref import WeakValueDictionary

from .runtime_control_storage import (
    FileIdentity,
    RuntimeStorageEnrollment,
    validate_parent,
)


class RuntimeControlError(RuntimeError):
    """The durable runtime control cannot establish a valid state."""


class RuntimeControlBlocked(RuntimeControlError):
    """The current runtime authority does not permit the operation."""


@dataclass(frozen=True, slots=True)
class RuntimeControlNamespace:
    """Adapter-owned storage identity; SQL identifiers never come from a request."""

    table_name: str
    schema_version: str
    label: str
    legacy_schema_version: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.table_name, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,62}", self.table_name) is None
        ):
            raise ValueError("runtime control table name is invalid")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("runtime control schema version is invalid")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("runtime control label is invalid")
        if self.legacy_schema_version is not None and (
            not isinstance(self.legacy_schema_version, str)
            or not self.legacy_schema_version
            or self.legacy_schema_version == self.schema_version
        ):
            raise ValueError("runtime control legacy schema version is invalid")


_CONTROL_STATES = frozenset({"stopped", "running", "stop_requested"})
_LOCAL_LOCKS: WeakValueDictionary[str, RLock] = WeakValueDictionary()
_LOCAL_LOCKS_GUARD = Lock()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _local_lock(path: Path) -> RLock:
    key = os.path.normcase(str(path.resolve()))
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _LOCAL_LOCKS[key] = lock
        return lock


class _CrossProcessFileLock:
    """Small standard-library exclusive lock used only around order dispatch."""

    blocked_error = RuntimeControlBlocked
    label = "Execution"

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self._monotonic = monotonic
        self._sleep = sleep
        self._local = _local_lock(path)
        self._handle: BinaryIO | None = None
        self._cleanup: ExitStack | None = None
        self._expected_identity = expected_identity

    def _verify_identity(self, handle: BinaryIO | None = None) -> None:
        if self._expected_identity is not None:
            try:
                self._expected_identity.verify(self.path, handle)
            except (OSError, ValueError) as exc:
                raise self.blocked_error(
                    "runtime enrolled interlock is missing or changed"
                ) from exc

    def __enter__(self) -> "_CrossProcessFileLock":
        if self._cleanup is not None:
            raise self.blocked_error("runtime interlock instance is already entered")
        deadline = self._monotonic() + self.timeout_seconds
        if not self._local.acquire(timeout=self.timeout_seconds):
            raise self.blocked_error(f"{self.label} runtime interlock timed out")
        with ExitStack() as cleanup:
            cleanup.callback(self._local.release)
            if self._expected_identity is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._verify_identity()
                handle = self.path.open(
                    "a+b" if self._expected_identity is None else "r+b", buffering=0
                )
            except (OSError, ValueError) as exc:
                if self._expected_identity is None:
                    raise
                raise self.blocked_error(
                    "runtime enrolled interlock is missing or changed"
                ) from exc
            cleanup.callback(handle.close)
            self._verify_identity(handle)
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            while True:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(
                            handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                except OSError as exc:
                    if self._monotonic() >= deadline:
                        raise self.blocked_error(
                            f"{self.label} runtime interlock timed out"
                        ) from exc
                    self._sleep(min(0.05, max(0.0, deadline - self._monotonic())))
                    continue
                cleanup.callback(self._unlock, handle)
                self._verify_identity(handle)
                self._handle = handle
                self._cleanup = cleanup.pop_all()
                return self

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        cleanup = self._cleanup
        self._cleanup = None
        self._handle = None
        if cleanup is not None:
            # ExitStack attempts every release, even when unlock or close raises.
            cleanup.close()


@dataclass(frozen=True, slots=True)
class RuntimeControlSnapshot:
    state: str
    lease_id: str
    owner_process_id: int
    started_at_ms: int
    heartbeat_at_ms: int
    stop_epoch: int
    stop_requested_at_ms: int
    stop_reason: str
    paused: bool
    pause_epoch: int
    pause_changed_at_ms: int
    pause_reason: str
    updated_at_ms: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class RuntimeControl:
    """Coordinate one autonomous process and an ordered, persistent Stop latch."""

    namespace = RuntimeControlNamespace(
        "runtime_control", "runtime-control-v1", "Execution"
    )
    error = RuntimeControlError
    blocked_error = RuntimeControlBlocked
    snapshot_type = RuntimeControlSnapshot
    lock_type = _CrossProcessFileLock

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        maximum_heartbeat_age_ms: int = 30_000,
        interlock_timeout_seconds: float = 30.0,
        clock_ms: Callable[[], int] | None = None,
        enrollment_path: str | Path | None = None,
    ) -> None:
        self.ledger_path = Path(os.path.abspath(ledger_path))
        self.lock_path = self.ledger_path.with_name(
            self.ledger_path.name + ".runtime.lock"
        )
        self.close_lock_path = self.ledger_path.with_name(
            self.ledger_path.name + ".runtime.close.lock"
        )
        self.enrollment_path = (
            Path(os.path.abspath(enrollment_path))
            if enrollment_path is not None
            else self.ledger_path.with_name(
                self.ledger_path.name + f".{self.namespace.table_name}.enrollment.json"
            )
        )
        if os.path.normcase(str(self.enrollment_path)) in {
            os.path.normcase(str(path))
            for path in (self.ledger_path, self.lock_path, self.close_lock_path)
            + tuple(
                self.ledger_path.with_name(self.ledger_path.name + suffix)
                for suffix in ("-wal", "-shm", "-journal")
            )
        }:
            raise ValueError("runtime enrollment must use an independent file")
        self.maximum_heartbeat_age_ms = int(maximum_heartbeat_age_ms)
        self.interlock_timeout_seconds = float(interlock_timeout_seconds)
        if not 5_000 <= self.maximum_heartbeat_age_ms <= 300_000:
            raise ValueError("maximum_heartbeat_age_ms must lie in [5000, 300000]")
        if not 1 <= self.interlock_timeout_seconds <= 300:
            raise ValueError("interlock_timeout_seconds must lie in [1, 300]")
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))

    def _interlock(self) -> _CrossProcessFileLock:
        enrollment = self._enrollment()
        return self.lock_type(
            self.lock_path,
            timeout_seconds=self.interlock_timeout_seconds,
            expected_identity=enrollment.submission_lock,
        )

    def _enrollment(self, *, allow_prepared: bool = False) -> RuntimeStorageEnrollment:
        try:
            result = RuntimeStorageEnrollment.read(
                self.enrollment_path, allow_prepared=allow_prepared
            )
            if (
                result.namespace != self.namespace.table_name
                or result.control_schema_version != self.namespace.schema_version
            ):
                raise ValueError("runtime enrollment namespace or schema differs")
            return result
        except (OSError, ValueError, TypeError) as exc:
            raise self.error(
                "runtime storage enrollment is unavailable or invalid; "
                "explicit provisioning or recovery is required"
            ) from exc

    def _verify_resources(self, enrollment: RuntimeStorageEnrollment) -> None:
        try:
            enrollment.database.verify(self.ledger_path)
            enrollment.submission_lock.verify(self.lock_path)
            enrollment.close_lock.verify(self.close_lock_path)
        except (OSError, ValueError) as exc:
            raise self.error(
                "runtime enrolled storage resource is missing or changed"
            ) from exc

    def _verify_store(
        self, connection: sqlite3.Connection, enrollment: RuntimeStorageEnrollment
    ) -> None:
        try:
            row = connection.execute(
                "SELECT store_id FROM runtime_storage_enrollment WHERE namespace = ?",
                (self.namespace.table_name,),
            ).fetchone()
            if row is None or row[0] != enrollment.store_id:
                raise self.error("runtime database enrollment differs")
        except sqlite3.Error as exc:
            raise self.error(
                "runtime database enrollment is missing or invalid"
            ) from exc

    def _existing_record_digest(self, connection: sqlite3.Connection) -> str:
        row = connection.execute(
            f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise self.error("runtime control record is missing")
        if "paused" in row.keys():
            self._verify_row(row)
        elif self.namespace.legacy_schema_version is not None:
            self._verified_legacy_payload(row)
        else:
            raise self.error("runtime control pause fields are missing")
        return str(row["record_sha256"])

    def inspect_storage(self) -> dict[str, object]:
        """Inspect existing control evidence without creating or adopting storage."""
        try:
            FileIdentity.at(self.ledger_path)
            connection = sqlite3.connect(
                self.ledger_path.as_uri() + "?mode=ro", uri=True
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            raise self.error("existing runtime database is unavailable") from exc
        connection.row_factory = sqlite3.Row
        try:
            digest = (
                self._existing_record_digest(connection)
                if self._control_table_exists(connection)
                else None
            )
            enrollment_exists = self.enrollment_path.exists()
            enrollment = (
                self._enrollment(allow_prepared=True) if enrollment_exists else None
            )
            return {
                "control_record_sha256": digest,
                "namespace": self.namespace.table_name,
                "enrollment_exists": enrollment_exists,
                "enrollment_state": "unenrolled"
                if enrollment is None
                else enrollment.state,
                "enrollment_store_id": None
                if enrollment is None
                else enrollment.store_id,
                "new_exposure_authorized": False,
            }
        finally:
            connection.close()

    def provision_storage(
        self, *, expected_record_sha256: str | None = None
    ) -> RuntimeControlSnapshot:
        """Explicitly create/adopt one namespace; never replace an enrollment."""
        if expected_record_sha256 is not None and (
            not isinstance(expected_record_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_record_sha256) is None
        ):
            raise ValueError("expected runtime record digest is invalid")
        if self.enrollment_path.exists() or self.enrollment_path.is_symlink():
            raise self.error(
                "runtime enrollment already exists; restore or explicitly recover it"
            )
        if self.ledger_path.exists():
            # An absent anchor must never silently rebind an already enrolled
            # database or recreate its potentially missing interlock files.
            FileIdentity.at(self.ledger_path)
            probe = sqlite3.connect(self.ledger_path.as_uri() + "?mode=ro", uri=True)
            try:
                if (
                    probe.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_storage_enrollment'"
                    ).fetchone()
                    and probe.execute(
                        "SELECT 1 FROM runtime_storage_enrollment WHERE namespace = ?",
                        (self.namespace.table_name,),
                    ).fetchone()
                ):
                    raise self.error(
                        "runtime database is already enrolled; restore its independent enrollment"
                    )
            finally:
                probe.close()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.enrollment_path.parent.mkdir(parents=True, exist_ok=True)
        validate_parent(self.ledger_path)
        validate_parent(self.enrollment_path)
        with (
            self.lock_type(
                self.lock_path, timeout_seconds=self.interlock_timeout_seconds
            ),
            self.lock_type(
                self.close_lock_path, timeout_seconds=self.interlock_timeout_seconds
            ),
        ):
            if self.enrollment_path.exists() or self.enrollment_path.is_symlink():
                raise self.error(
                    "runtime enrollment already exists; restore or explicitly recover it"
                )
            if not self.ledger_path.exists():
                if expected_record_sha256 is not None:
                    raise self.error("expected runtime database is missing")
                with self.ledger_path.open("xb") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
            FileIdentity.at(self.ledger_path)
            connection = sqlite3.connect(
                self.ledger_path.as_uri() + "?mode=rw",
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")
                exists = self._control_table_exists(connection)
                if exists:
                    if (
                        self._existing_record_digest(connection)
                        != expected_record_sha256
                    ):
                        raise self.error(
                            "explicit adoption requires the exact existing runtime record digest"
                        )
                elif expected_record_sha256 is not None:
                    raise self.error("expected runtime control namespace is missing")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS runtime_storage_enrollment "
                    "(namespace TEXT PRIMARY KEY, store_id TEXT NOT NULL)"
                )
                if connection.execute(
                    "SELECT 1 FROM runtime_storage_enrollment WHERE namespace = ?",
                    (self.namespace.table_name,),
                ).fetchone():
                    raise self.error(
                        "runtime database is already enrolled; restore its independent enrollment"
                    )
                enrollment = RuntimeStorageEnrollment(
                    self.namespace.table_name,
                    self.namespace.schema_version,
                    uuid.uuid4().hex,
                    "prepared",
                    FileIdentity.at(self.ledger_path),
                    FileIdentity.at(self.lock_path),
                    FileIdentity.at(self.close_lock_path),
                )
                enrollment.prepare(self.enrollment_path)
                if not exists:
                    self._bootstrap_control(connection, require_reconciliation=True)
                connection.execute(
                    "INSERT INTO runtime_storage_enrollment VALUES (?, ?)",
                    (self.namespace.table_name, enrollment.store_id),
                )
                connection.execute("COMMIT")
                enrollment.finish(self.enrollment_path)
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        return self.snapshot()

    def recover_storage_enrollment(
        self, *, expected_store_id: str, expected_record_sha256: str
    ) -> RuntimeControlSnapshot:
        """Finish a committed prepared enrollment without resetting its control row."""
        enrollment = self._enrollment(allow_prepared=True)
        if enrollment.state != "prepared" or enrollment.store_id != expected_store_id:
            raise self.error("prepared runtime enrollment identity differs")
        with (
            self.lock_type(
                self.lock_path,
                timeout_seconds=self.interlock_timeout_seconds,
                expected_identity=enrollment.submission_lock,
            ),
            self.lock_type(
                self.close_lock_path,
                timeout_seconds=self.interlock_timeout_seconds,
                expected_identity=enrollment.close_lock,
            ),
        ):
            if self._enrollment(allow_prepared=True) != enrollment:
                raise self.error("prepared runtime enrollment changed")
            self._verify_resources(enrollment)
            connection = sqlite3.connect(
                self.ledger_path.as_uri() + "?mode=rw",
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._verify_store(connection, enrollment)
                if self._existing_record_digest(connection) != expected_record_sha256:
                    raise self.error("prepared runtime control record differs")
                enrollment.finish(self.enrollment_path)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        return self.snapshot()

    @contextmanager
    def close_guard(self, *, timeout_seconds: float) -> Iterator[None]:
        timeout = float(timeout_seconds)
        if not 1 <= timeout <= 300:
            raise ValueError("timeout_seconds must lie in [1, 300]")
        with self.lock_type(
            self.close_lock_path,
            timeout_seconds=timeout,
            expected_identity=self._enrollment().close_lock,
        ):
            yield

    def _connect(self) -> sqlite3.Connection:
        enrollment = self._enrollment()
        self._verify_resources(enrollment)
        try:
            connection = sqlite3.connect(
                self.ledger_path.as_uri() + "?mode=rw",
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise self.error("enrolled runtime database could not be opened") from exc
        connection.row_factory = sqlite3.Row
        try:
            self._verify_resources(enrollment)
            self._verify_store(connection, enrollment)
            self._initialize(connection)
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA trusted_schema=OFF")
            mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            ).lower()
            if mode != "wal":
                raise self.error("runtime control could not enable WAL mode")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA wal_autocheckpoint=100")
            connection.execute("PRAGMA journal_size_limit=1048576")
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise self.error(
                    f"{self.namespace.label} runtime control integrity check failed"
                )
            return connection
        except BaseException:
            connection.close()
            raise

    def _initialize(self, connection: sqlite3.Connection) -> None:
        if not self._control_table_exists(connection):
            raise self.error(
                "runtime control namespace is missing; restore enrolled storage"
            )
        self._require_initialized_record(connection)

    def _control_table_exists(self, connection: sqlite3.Connection) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (self.namespace.table_name,),
            ).fetchone()
            is not None
        )

    def _require_initialized_record(self, connection: sqlite3.Connection) -> None:
        row = connection.execute(
            f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise self.error(
                f"{self.namespace.label} runtime control record is missing"
            )
        if "paused" not in row.keys():
            if self.namespace.legacy_schema_version is None:
                raise self.error("runtime control pause fields are missing")
            self._migrate_v1(connection)
            row = connection.execute(
                f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise self.error(
                    f"{self.namespace.label} runtime control record is missing"
                )
        self._verify_row(row)

    def _bootstrap_control(
        self, connection: sqlite3.Connection, *, require_reconciliation: bool = False
    ) -> None:
        """Create table and first record inside the caller's writer transaction."""
        connection.execute(
            f"""
            CREATE TABLE {self.namespace.table_name} (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL,
                state TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                owner_process_id INTEGER NOT NULL,
                started_at_ms INTEGER NOT NULL,
                heartbeat_at_ms INTEGER NOT NULL,
                stop_epoch INTEGER NOT NULL,
                stop_requested_at_ms INTEGER NOT NULL,
                stop_reason TEXT NOT NULL,
                paused INTEGER NOT NULL CHECK (paused IN (0, 1)),
                pause_epoch INTEGER NOT NULL CHECK (pause_epoch >= 0),
                pause_changed_at_ms INTEGER NOT NULL,
                pause_reason TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL,
                CHECK (state IN ('stopped', 'running', 'stop_requested')),
                CHECK (owner_process_id >= 0),
                CHECK (stop_epoch >= 0)
            )
            """
        )
        now = self._now_ms()
        payload = {
            "schema_version": self.namespace.schema_version,
            "state": "stop_requested" if require_reconciliation else "stopped",
            "lease_id": "",
            "owner_process_id": 0,
            "started_at_ms": 0,
            "heartbeat_at_ms": 0,
            "stop_epoch": int(require_reconciliation),
            "stop_requested_at_ms": now if require_reconciliation else 0,
            "stop_reason": "storage_enrollment_requires_reconciliation"
            if require_reconciliation
            else "never_started",
            "paused": require_reconciliation,
            "pause_epoch": 0,
            "pause_changed_at_ms": now,
            "pause_reason": "storage_enrollment_requires_reconciliation"
            if require_reconciliation
            else "never_started",
            "updated_at_ms": now,
        }
        connection.execute(
            f"""
            INSERT INTO {self.namespace.table_name} (
                singleton, schema_version, state, lease_id,
                owner_process_id, started_at_ms, heartbeat_at_ms,
                stop_epoch, stop_requested_at_ms, stop_reason,
                paused, pause_epoch, pause_changed_at_ms, pause_reason,
                updated_at_ms, record_sha256
            ) VALUES (
                1, :schema_version, :state, :lease_id,
                :owner_process_id, :started_at_ms, :heartbeat_at_ms,
                :stop_epoch, :stop_requested_at_ms, :stop_reason,
                :paused, :pause_epoch, :pause_changed_at_ms, :pause_reason,
                :updated_at_ms, :record_sha256
            )
            """,
            {
                **payload,
                "paused": int(payload["paused"]),
                "record_sha256": _canonical_sha256(payload),
            },
        )

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """Add a durable pause latch without weakening an existing Stop latch."""

        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise self.error(
                    f"{self.namespace.label} runtime control record is missing"
                )
            # A concurrent caller can complete migration before this writer
            # acquires its transaction. Verify that result instead of migrating twice.
            if "paused" in row.keys():
                self._verify_row(row)
                connection.execute("COMMIT")
                return
            legacy = self._verified_legacy_payload(row)
            connection.execute(
                f"ALTER TABLE {self.namespace.table_name} "
                "ADD COLUMN paused INTEGER NOT NULL DEFAULT 0 "
                "CHECK (paused IN (0, 1))"
            )
            connection.execute(
                f"ALTER TABLE {self.namespace.table_name} "
                "ADD COLUMN pause_epoch INTEGER NOT NULL DEFAULT 0 "
                "CHECK (pause_epoch >= 0)"
            )
            connection.execute(
                f"ALTER TABLE {self.namespace.table_name} "
                "ADD COLUMN pause_changed_at_ms INTEGER NOT NULL DEFAULT 0"
            )
            connection.execute(
                f"ALTER TABLE {self.namespace.table_name} "
                "ADD COLUMN pause_reason TEXT NOT NULL DEFAULT ''"
            )
            migrated = {
                **legacy,
                "schema_version": self.namespace.schema_version,
                "paused": False,
                "pause_epoch": 0,
                "pause_changed_at_ms": int(legacy["updated_at_ms"]),
                "pause_reason": "migrated_from_v1",
            }
            connection.execute(
                f"""
                UPDATE {self.namespace.table_name}
                SET schema_version = ?, paused = ?, pause_epoch = ?,
                    pause_changed_at_ms = ?, pause_reason = ?, record_sha256 = ?
                WHERE singleton = 1
                """,
                [
                    migrated["schema_version"],
                    int(migrated["paused"]),
                    migrated["pause_epoch"],
                    migrated["pause_changed_at_ms"],
                    migrated["pause_reason"],
                    _canonical_sha256(migrated),
                ],
            )
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _verified_legacy_payload(self, row: Mapping[str, object]) -> dict[str, object]:
        legacy = {
            "schema_version": str(row["schema_version"]),
            "state": str(row["state"]),
            "lease_id": str(row["lease_id"]),
            "owner_process_id": int(row["owner_process_id"]),
            "started_at_ms": int(row["started_at_ms"]),
            "heartbeat_at_ms": int(row["heartbeat_at_ms"]),
            "stop_epoch": int(row["stop_epoch"]),
            "stop_requested_at_ms": int(row["stop_requested_at_ms"]),
            "stop_reason": str(row["stop_reason"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }
        if (
            legacy["schema_version"] != self.namespace.legacy_schema_version
            or legacy["state"] not in _CONTROL_STATES
            or str(row["record_sha256"]) != _canonical_sha256(legacy)
        ):
            raise self.error(
                f"{self.namespace.label} runtime control v1 record differs"
            )
        return legacy

    @staticmethod
    def _row_payload(row: Mapping[str, object]) -> dict[str, object]:
        return {
            "schema_version": str(row["schema_version"]),
            "state": str(row["state"]),
            "lease_id": str(row["lease_id"]),
            "owner_process_id": int(row["owner_process_id"]),
            "started_at_ms": int(row["started_at_ms"]),
            "heartbeat_at_ms": int(row["heartbeat_at_ms"]),
            "stop_epoch": int(row["stop_epoch"]),
            "stop_requested_at_ms": int(row["stop_requested_at_ms"]),
            "stop_reason": str(row["stop_reason"]),
            "paused": bool(row["paused"]),
            "pause_epoch": int(row["pause_epoch"]),
            "pause_changed_at_ms": int(row["pause_changed_at_ms"]),
            "pause_reason": str(row["pause_reason"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }

    def _verify_row(self, row: Mapping[str, object]) -> None:
        payload = self._row_payload(row)
        if (
            payload["schema_version"] != self.namespace.schema_version
            or payload["state"] not in _CONTROL_STATES
            or str(row["record_sha256"]) != _canonical_sha256(payload)
        ):
            raise self.error(f"{self.namespace.label} runtime control record differs")

    def _snapshot_row(
        self,
        row: Mapping[str, object],
    ) -> RuntimeControlSnapshot:
        self._verify_row(row)
        payload = self._row_payload(row)
        payload.pop("schema_version")
        return self.snapshot_type(**payload)

    def _write(
        self,
        connection: sqlite3.Connection,
        payload: Mapping[str, object],
    ) -> None:
        body = dict(payload)
        connection.execute(
            f"""
            UPDATE {self.namespace.table_name}
            SET schema_version = ?, state = ?, lease_id = ?,
                owner_process_id = ?, started_at_ms = ?,
                heartbeat_at_ms = ?, stop_epoch = ?,
                stop_requested_at_ms = ?, stop_reason = ?,
                paused = ?, pause_epoch = ?, pause_changed_at_ms = ?,
                pause_reason = ?,
                updated_at_ms = ?, record_sha256 = ?
            WHERE singleton = 1
            """,
            [
                body["schema_version"],
                body["state"],
                body["lease_id"],
                body["owner_process_id"],
                body["started_at_ms"],
                body["heartbeat_at_ms"],
                body["stop_epoch"],
                body["stop_requested_at_ms"],
                body["stop_reason"],
                int(body["paused"]),
                body["pause_epoch"],
                body["pause_changed_at_ms"],
                body["pause_reason"],
                body["updated_at_ms"],
                _canonical_sha256(body),
            ],
        )

    def _now_ms(self) -> int:
        value = int(self._clock_ms())
        if value <= 0:
            raise self.error(f"{self.namespace.label} runtime clock is invalid")
        return value

    def snapshot(self) -> RuntimeControlSnapshot:
        connection = self._connect()
        try:
            row = connection.execute(
                f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise self.error(
                    f"{self.namespace.label} runtime control record is missing"
                )
            return self._snapshot_row(row)
        finally:
            connection.close()

    def acquire(self, *, owner_process_id: int | None = None) -> str:
        owner = os.getpid() if owner_process_id is None else int(owner_process_id)
        if owner <= 0:
            raise ValueError("owner_process_id must be positive")
        with self._interlock():
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"""
                    SELECT * FROM {self.namespace.table_name}
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    raise self.error(
                        f"{self.namespace.label} runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                if current.state != "stopped":
                    raise self.blocked_error(
                        f"{self.namespace.label} autonomous runtime is already active "
                        "or awaiting Stop completion"
                    )
                now = self._now_ms()
                lease_id = uuid.uuid4().hex
                payload = {
                    "schema_version": self.namespace.schema_version,
                    "state": "running",
                    "lease_id": lease_id,
                    "owner_process_id": owner,
                    "started_at_ms": now,
                    "heartbeat_at_ms": now,
                    "stop_epoch": current.stop_epoch,
                    "stop_requested_at_ms": 0,
                    "stop_reason": "",
                    "paused": False,
                    "pause_epoch": current.pause_epoch,
                    "pause_changed_at_ms": now,
                    "pause_reason": "runtime_started",
                    "updated_at_ms": now,
                }
                self._write(connection, payload)
                connection.execute("COMMIT")
                return lease_id
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def _assert_snapshot_allows_open(
        self,
        snapshot: RuntimeControlSnapshot,
        *,
        lease_id: str,
    ) -> None:
        now = self._now_ms()
        if (
            snapshot.state != "running"
            or snapshot.lease_id != lease_id
            or snapshot.paused
            or snapshot.heartbeat_at_ms <= 0
            or snapshot.heartbeat_at_ms > now + 5_000
            or now - snapshot.heartbeat_at_ms > self.maximum_heartbeat_age_ms
        ):
            raise self.blocked_error(
                f"{self.namespace.label} runtime lease does not permit new exposure"
            )

    def assert_opening_allowed(self, lease_id: str) -> None:
        self._assert_snapshot_allows_open(
            self.snapshot(),
            lease_id=str(lease_id),
        )

    @contextmanager
    def submission_guard(self, lease_id: str) -> Iterator[None]:
        with self._interlock():
            self.assert_opening_allowed(str(lease_id))
            yield

    def heartbeat(self, lease_id: str) -> bool:
        """Renew a fresh lease; persist Stop if its worker missed the deadline."""
        # Order expiry against the same last-mile interlock as order submission.
        with self._interlock():
            return self._heartbeat_locked(lease_id)

    def _heartbeat_locked(self, lease_id: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise self.error(
                    f"{self.namespace.label} runtime control record is missing"
                )
            current = self._snapshot_row(row)
            if current.state != "running" or current.lease_id != str(lease_id):
                connection.execute("COMMIT")
                return False
            now = self._now_ms()
            if (
                current.heartbeat_at_ms <= 0
                or current.heartbeat_at_ms > now + 5_000
                or now - current.heartbeat_at_ms > self.maximum_heartbeat_age_ms
            ):
                # Do not refresh away evidence of a freeze or clock discontinuity.
                # Keep ownership and the old heartbeat for exact recovery checks.
                self._write(
                    connection,
                    {
                        "schema_version": self.namespace.schema_version,
                        **current.asdict(),
                        "state": "stop_requested",
                        "stop_epoch": current.stop_epoch + 1,
                        "stop_requested_at_ms": now,
                        "stop_reason": "heartbeat_invalid_or_expired",
                        "paused": True,
                        "pause_changed_at_ms": now,
                        "pause_reason": "stop_requested",
                        "updated_at_ms": now,
                    },
                )
                connection.execute("COMMIT")
                return False
            payload = {
                "schema_version": self.namespace.schema_version,
                **current.asdict(),
                "heartbeat_at_ms": now,
                "updated_at_ms": now,
            }
            self._write(connection, payload)
            connection.execute("COMMIT")
            return True
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def set_paused(
        self,
        paused: bool,
        *,
        reason: str,
    ) -> RuntimeControlSnapshot:
        """Order an external pause transition against the final open interlock."""

        if not isinstance(paused, bool):
            raise TypeError("paused must be a boolean")
        normalized_reason = str(reason or "").strip()
        if not normalized_reason or len(normalized_reason) > 160:
            raise ValueError("pause reason is invalid")
        with self._interlock():
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
                ).fetchone()
                if row is None:
                    raise self.error(
                        f"{self.namespace.label} runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                if current.state != "running":
                    raise self.blocked_error(
                        f"{self.namespace.label} Pause requires an active autonomous runtime"
                    )
                now = self._now_ms()
                if not paused and (
                    current.heartbeat_at_ms <= 0
                    or current.heartbeat_at_ms > now + 5_000
                    or now - current.heartbeat_at_ms > self.maximum_heartbeat_age_ms
                ):
                    raise self.blocked_error(
                        f"{self.namespace.label} runtime heartbeat is stale; resume is blocked"
                    )
                if current.paused == paused:
                    connection.execute("COMMIT")
                    return current
                payload = {
                    "schema_version": self.namespace.schema_version,
                    **current.asdict(),
                    "paused": paused,
                    "pause_epoch": current.pause_epoch + 1,
                    "pause_changed_at_ms": now,
                    "pause_reason": normalized_reason,
                    "updated_at_ms": now,
                }
                self._write(connection, payload)
                connection.execute("COMMIT")
                return self.snapshot_type(
                    **{
                        key: value
                        for key, value in payload.items()
                        if key != "schema_version"
                    }
                )
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def request_stop(
        self,
        *,
        reason: str = "operator_stop",
    ) -> RuntimeControlSnapshot:
        normalized_reason = str(reason or "").strip()
        if not normalized_reason or len(normalized_reason) > 160:
            raise ValueError("stop reason is invalid")
        with self._interlock():
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"""
                    SELECT * FROM {self.namespace.table_name}
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    raise self.error(
                        f"{self.namespace.label} runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                now = self._now_ms()
                if current.state == "stop_requested":
                    connection.execute("COMMIT")
                    return current
                running = current.state == "running"
                payload = {
                    "schema_version": self.namespace.schema_version,
                    "state": "stop_requested",
                    "lease_id": current.lease_id if running else "",
                    "owner_process_id": (current.owner_process_id if running else 0),
                    "started_at_ms": current.started_at_ms if running else 0,
                    "heartbeat_at_ms": (current.heartbeat_at_ms if running else 0),
                    "stop_epoch": current.stop_epoch + 1,
                    "stop_requested_at_ms": now,
                    "stop_reason": normalized_reason,
                    "paused": True,
                    "pause_epoch": current.pause_epoch,
                    "pause_changed_at_ms": now,
                    "pause_reason": "stop_requested",
                    "updated_at_ms": now,
                }
                self._write(connection, payload)
                connection.execute("COMMIT")
                return self.snapshot_type(
                    **{
                        key: value
                        for key, value in payload.items()
                        if key != "schema_version"
                    }
                )
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def release(self, lease_id: str, *, reason: str) -> None:
        if (
            not isinstance(lease_id, str)
            or not lease_id.strip()
            or lease_id != lease_id.strip()
        ):
            raise self.blocked_error("runtime release requires an exact nonempty lease")
        normalized_reason = str(reason or "").strip()
        if not normalized_reason or len(normalized_reason) > 160:
            raise ValueError("release reason is invalid")
        with self._interlock():
            self._release_locked(lease_id, reason=normalized_reason)

    def _release_locked(self, lease_id: str, *, reason: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT * FROM {self.namespace.table_name} WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise self.error(
                    f"{self.namespace.label} runtime control record is missing"
                )
            current = self._snapshot_row(row)
            if current.state == "stopped" and not current.lease_id:
                connection.execute("COMMIT")
                return
            if current.lease_id != str(lease_id):
                raise self.blocked_error(
                    f"{self.namespace.label} runtime lease changed before release"
                )
            now = self._now_ms()
            payload = {
                "schema_version": self.namespace.schema_version,
                "state": "stopped",
                "lease_id": "",
                "owner_process_id": 0,
                "started_at_ms": 0,
                "heartbeat_at_ms": 0,
                "stop_epoch": current.stop_epoch,
                "stop_requested_at_ms": current.stop_requested_at_ms,
                "stop_reason": reason,
                "paused": False,
                "pause_epoch": current.pause_epoch,
                "pause_changed_at_ms": now,
                "pause_reason": "runtime_stopped",
                "updated_at_ms": now,
            }
            self._write(connection, payload)
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def complete_stale_stop(self, *, exposure_closed: bool) -> bool:
        if not exposure_closed:
            raise self.blocked_error(
                "stale runtime Stop cannot complete with owned exposure"
            )
        with self._interlock():
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"""
                    SELECT * FROM {self.namespace.table_name}
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    raise self.error(
                        f"{self.namespace.label} runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                if current.state == "stopped":
                    connection.execute("COMMIT")
                    return True
                now = self._now_ms()
                ownerless_stop = (
                    current.state == "stop_requested"
                    and not current.lease_id
                    and current.owner_process_id == 0
                    and current.started_at_ms == 0
                    and current.heartbeat_at_ms == 0
                )
                if ownerless_stop:
                    payload = {
                        "schema_version": self.namespace.schema_version,
                        "state": "stopped",
                        "lease_id": "",
                        "owner_process_id": 0,
                        "started_at_ms": 0,
                        "heartbeat_at_ms": 0,
                        "stop_epoch": current.stop_epoch,
                        "stop_requested_at_ms": current.stop_requested_at_ms,
                        "stop_reason": "ownerless_stop_completed",
                        "paused": False,
                        "pause_epoch": current.pause_epoch,
                        "pause_changed_at_ms": now,
                        "pause_reason": "runtime_stopped",
                        "updated_at_ms": now,
                    }
                    self._write(connection, payload)
                    connection.execute("COMMIT")
                    return True
                if (
                    current.state != "stop_requested"
                    or current.heartbeat_at_ms <= 0
                    or current.heartbeat_at_ms > now + 5_000
                    or now - current.heartbeat_at_ms <= self.maximum_heartbeat_age_ms
                ):
                    connection.execute("COMMIT")
                    return False
                payload = {
                    "schema_version": self.namespace.schema_version,
                    "state": "stopped",
                    "lease_id": "",
                    "owner_process_id": 0,
                    "started_at_ms": 0,
                    "heartbeat_at_ms": 0,
                    "stop_epoch": current.stop_epoch,
                    "stop_requested_at_ms": current.stop_requested_at_ms,
                    "stop_reason": "stale_runtime_stop_completed",
                    "paused": False,
                    "pause_epoch": current.pause_epoch,
                    "pause_changed_at_ms": now,
                    "pause_reason": "runtime_stopped",
                    "updated_at_ms": now,
                }
                self._write(connection, payload)
                connection.execute("COMMIT")
                return True
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    def wait_until_stopped(
        self,
        *,
        timeout_seconds: float,
        poll_seconds: float = 0.1,
    ) -> RuntimeControlSnapshot:
        timeout = float(timeout_seconds)
        poll = float(poll_seconds)
        if not 0 <= timeout <= 300:
            raise ValueError("timeout_seconds must lie in [0, 300]")
        if not 0.05 <= poll <= 1:
            raise ValueError("poll_seconds must lie in [0.05, 1]")
        deadline = time.monotonic() + timeout
        while True:
            current = self.snapshot()
            if current.state == "stopped" or time.monotonic() >= deadline:
                return current
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))


@dataclass(frozen=True, slots=True)
class RuntimeLeaseInterlock:
    control: RuntimeControl
    lease_id: str

    def assert_opening_allowed(self) -> None:
        self.control.assert_opening_allowed(self.lease_id)

    @contextmanager
    def submission_guard(self) -> Iterator[None]:
        with self.control.submission_guard(self.lease_id):
            yield


class RuntimeControlService:
    """Heartbeat the lease and turn any external Stop into supervisor shutdown."""

    trading_authority = False

    def __init__(
        self,
        control: RuntimeControl,
        *,
        lease_id: str,
        heartbeat_interval_seconds: float = 5.0,
        stop_poll_interval_seconds: float = 0.5,
    ) -> None:
        interval = float(heartbeat_interval_seconds)
        poll = float(stop_poll_interval_seconds)
        if not 1 <= interval <= 30:
            raise ValueError("heartbeat_interval_seconds must lie in [1, 30]")
        if not 0.1 <= poll <= 1:
            raise ValueError("stop_poll_interval_seconds must lie in [0.1, 1]")
        if poll > interval:
            raise ValueError(
                "stop_poll_interval_seconds cannot exceed heartbeat interval"
            )
        self.control = control
        self.lease_id = str(lease_id)
        self.heartbeat_interval_seconds = interval
        self.stop_poll_interval_seconds = poll

    async def run(
        self,
        stop: asyncio.Event,
        *,
        request_stop: Callable[[], None],
        pause_runtime: Callable[[], None] | None = None,
        resume_runtime: Callable[[], None] | None = None,
    ) -> None:
        next_heartbeat = 0.0
        pause_applied = False
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_heartbeat:
                valid = await asyncio.to_thread(
                    self.control.heartbeat,
                    self.lease_id,
                )
                next_heartbeat = now + self.heartbeat_interval_seconds
                snapshot = (
                    await asyncio.to_thread(self.control.snapshot) if valid else None
                )
            else:
                snapshot = await asyncio.to_thread(self.control.snapshot)
                valid = (
                    snapshot.state == "running" and snapshot.lease_id == self.lease_id
                )
            if not valid:
                request_stop()
                return
            if snapshot is None:  # pragma: no cover - guarded by valid
                request_stop()
                return
            if snapshot.paused != pause_applied:
                callback = pause_runtime if snapshot.paused else resume_runtime
                if callback is None:
                    request_stop()
                    return
                callback()
                pause_applied = snapshot.paused
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=min(
                        self.stop_poll_interval_seconds,
                        max(0.0, next_heartbeat - time.monotonic()),
                    ),
                )
            except TimeoutError:
                continue


__all__ = [
    "RuntimeControl",
    "RuntimeControlBlocked",
    "RuntimeControlError",
    "RuntimeControlNamespace",
    "RuntimeControlService",
    "RuntimeControlSnapshot",
    "RuntimeLeaseInterlock",
]
