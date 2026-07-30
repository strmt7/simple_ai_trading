"""Durable single-writer and Stop control for autonomous Polymarket execution."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock, RLock
import time
from typing import Callable, Iterator, Mapping
import uuid

from .polymarket_live import PolymarketLiveBlocked, PolymarketLiveError


POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION = "polymarket-runtime-control-v1"
_CONTROL_STATES = frozenset({"stopped", "running", "stop_requested"})
_LOCAL_LOCKS: dict[str, RLock] = {}
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

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self._monotonic = monotonic
        self._sleep = sleep
        self._local = _local_lock(path)
        self._handle: object | None = None

    def __enter__(self) -> "_CrossProcessFileLock":
        deadline = self._monotonic() + self.timeout_seconds
        if not self._local.acquire(timeout=self.timeout_seconds):
            raise PolymarketLiveBlocked("Polymarket runtime interlock timed out")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b", buffering=0)
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
                    self._handle = handle
                    return self
                except OSError as exc:
                    if self._monotonic() >= deadline:
                        handle.close()
                        raise PolymarketLiveBlocked(
                            "Polymarket runtime interlock timed out"
                        ) from exc
                    self._sleep(min(0.05, max(0.0, deadline - self._monotonic())))
        except Exception:
            self._local.release()
            raise

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        handle = self._handle
        self._handle = None
        try:
            if handle is not None:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        finally:
            self._local.release()


@dataclass(frozen=True, slots=True)
class PolymarketRuntimeControlSnapshot:
    state: str
    lease_id: str
    owner_process_id: int
    started_at_ms: int
    heartbeat_at_ms: int
    stop_epoch: int
    stop_requested_at_ms: int
    stop_reason: str
    updated_at_ms: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class PolymarketRuntimeControl:
    """Coordinate one autonomous process and an ordered, persistent Stop latch."""

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        maximum_heartbeat_age_ms: int = 30_000,
        interlock_timeout_seconds: float = 30.0,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.lock_path = self.ledger_path.with_name(
            self.ledger_path.name + ".runtime.lock"
        )
        self.close_lock_path = self.ledger_path.with_name(
            self.ledger_path.name + ".runtime.close.lock"
        )
        self.maximum_heartbeat_age_ms = int(maximum_heartbeat_age_ms)
        self.interlock_timeout_seconds = float(interlock_timeout_seconds)
        if not 5_000 <= self.maximum_heartbeat_age_ms <= 300_000:
            raise ValueError("maximum_heartbeat_age_ms must lie in [5000, 300000]")
        if not 1 <= self.interlock_timeout_seconds <= 300:
            raise ValueError("interlock_timeout_seconds must lie in [1, 300]")
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))

    def _interlock(self) -> _CrossProcessFileLock:
        return _CrossProcessFileLock(
            self.lock_path,
            timeout_seconds=self.interlock_timeout_seconds,
        )

    @contextmanager
    def close_guard(self, *, timeout_seconds: float) -> Iterator[None]:
        timeout = float(timeout_seconds)
        if not 1 <= timeout <= 300:
            raise ValueError("timeout_seconds must lie in [1, 300]")
        with _CrossProcessFileLock(
            self.close_lock_path,
            timeout_seconds=timeout,
        ):
            yield

    def _connect(self) -> sqlite3.Connection:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.ledger_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA trusted_schema=OFF")
            mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            ).lower()
            if mode != "wal":
                raise PolymarketLiveError("runtime control could not enable WAL mode")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA wal_autocheckpoint=100")
            connection.execute("PRAGMA journal_size_limit=1048576")
            self._initialize(connection)
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise PolymarketLiveError(
                    "Polymarket runtime control integrity check failed"
                )
            return connection
        except Exception:
            connection.close()
            raise

    def _initialize(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS polymarket_runtime_control (
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
                updated_at_ms INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL,
                CHECK (state IN ('stopped', 'running', 'stop_requested')),
                CHECK (owner_process_id >= 0),
                CHECK (stop_epoch >= 0)
            )
            """
        )
        row = connection.execute(
            "SELECT * FROM polymarket_runtime_control WHERE singleton = 1"
        ).fetchone()
        if row is None:
            now = self._now_ms()
            payload = {
                "schema_version": POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
                "state": "stopped",
                "lease_id": "",
                "owner_process_id": 0,
                "started_at_ms": 0,
                "heartbeat_at_ms": 0,
                "stop_epoch": 0,
                "stop_requested_at_ms": 0,
                "stop_reason": "never_started",
                "updated_at_ms": now,
            }
            connection.execute(
                """
                INSERT OR IGNORE INTO polymarket_runtime_control (
                    singleton, schema_version, state, lease_id,
                    owner_process_id, started_at_ms, heartbeat_at_ms,
                    stop_epoch, stop_requested_at_ms, stop_reason,
                    updated_at_ms, record_sha256
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    payload["schema_version"],
                    payload["state"],
                    payload["lease_id"],
                    payload["owner_process_id"],
                    payload["started_at_ms"],
                    payload["heartbeat_at_ms"],
                    payload["stop_epoch"],
                    payload["stop_requested_at_ms"],
                    payload["stop_reason"],
                    payload["updated_at_ms"],
                    _canonical_sha256(payload),
                ],
            )
            row = connection.execute(
                "SELECT * FROM polymarket_runtime_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise PolymarketLiveError(
                    "Polymarket runtime control initialization failed"
                )
        self._verify_row(row)

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
            "updated_at_ms": int(row["updated_at_ms"]),
        }

    @classmethod
    def _verify_row(cls, row: Mapping[str, object]) -> None:
        payload = cls._row_payload(row)
        if (
            payload["schema_version"] != POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION
            or payload["state"] not in _CONTROL_STATES
            or str(row["record_sha256"]) != _canonical_sha256(payload)
        ):
            raise PolymarketLiveError("Polymarket runtime control record differs")

    @classmethod
    def _snapshot_row(
        cls,
        row: Mapping[str, object],
    ) -> PolymarketRuntimeControlSnapshot:
        cls._verify_row(row)
        payload = cls._row_payload(row)
        payload.pop("schema_version")
        return PolymarketRuntimeControlSnapshot(**payload)

    @staticmethod
    def _write(
        connection: sqlite3.Connection,
        payload: Mapping[str, object],
    ) -> None:
        body = dict(payload)
        connection.execute(
            """
            UPDATE polymarket_runtime_control
            SET schema_version = ?, state = ?, lease_id = ?,
                owner_process_id = ?, started_at_ms = ?,
                heartbeat_at_ms = ?, stop_epoch = ?,
                stop_requested_at_ms = ?, stop_reason = ?,
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
                body["updated_at_ms"],
                _canonical_sha256(body),
            ],
        )

    def _now_ms(self) -> int:
        value = int(self._clock_ms())
        if value <= 0:
            raise PolymarketLiveError("Polymarket runtime clock is invalid")
        return value

    def snapshot(self) -> PolymarketRuntimeControlSnapshot:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM polymarket_runtime_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise PolymarketLiveError(
                    "Polymarket runtime control record is missing"
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
                    """
                    SELECT * FROM polymarket_runtime_control
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    raise PolymarketLiveError(
                        "Polymarket runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                if current.state != "stopped":
                    raise PolymarketLiveBlocked(
                        "Polymarket autonomous runtime is already active "
                        "or awaiting Stop completion"
                    )
                now = self._now_ms()
                lease_id = uuid.uuid4().hex
                payload = {
                    "schema_version": POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
                    "state": "running",
                    "lease_id": lease_id,
                    "owner_process_id": owner,
                    "started_at_ms": now,
                    "heartbeat_at_ms": now,
                    "stop_epoch": current.stop_epoch,
                    "stop_requested_at_ms": 0,
                    "stop_reason": "",
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
        snapshot: PolymarketRuntimeControlSnapshot,
        *,
        lease_id: str,
    ) -> None:
        now = self._now_ms()
        if (
            snapshot.state != "running"
            or snapshot.lease_id != lease_id
            or snapshot.heartbeat_at_ms <= 0
            or snapshot.heartbeat_at_ms > now + 5_000
            or now - snapshot.heartbeat_at_ms > self.maximum_heartbeat_age_ms
        ):
            raise PolymarketLiveBlocked(
                "Polymarket runtime lease does not permit new exposure"
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
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM polymarket_runtime_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise PolymarketLiveError(
                    "Polymarket runtime control record is missing"
                )
            current = self._snapshot_row(row)
            if current.state != "running" or current.lease_id != str(lease_id):
                connection.execute("COMMIT")
                return False
            now = self._now_ms()
            payload = {
                "schema_version": POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
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

    def request_stop(
        self,
        *,
        reason: str = "operator_stop",
    ) -> PolymarketRuntimeControlSnapshot:
        normalized_reason = str(reason or "").strip()
        if not normalized_reason or len(normalized_reason) > 160:
            raise ValueError("stop reason is invalid")
        with self._interlock():
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM polymarket_runtime_control
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    raise PolymarketLiveError(
                        "Polymarket runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                now = self._now_ms()
                if current.state == "stop_requested":
                    connection.execute("COMMIT")
                    return current
                running = current.state == "running"
                payload = {
                    "schema_version": POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
                    "state": "stop_requested" if running else "stopped",
                    "lease_id": current.lease_id if running else "",
                    "owner_process_id": (current.owner_process_id if running else 0),
                    "started_at_ms": current.started_at_ms if running else 0,
                    "heartbeat_at_ms": (current.heartbeat_at_ms if running else 0),
                    "stop_epoch": current.stop_epoch + 1,
                    "stop_requested_at_ms": now,
                    "stop_reason": normalized_reason,
                    "updated_at_ms": now,
                }
                self._write(connection, payload)
                connection.execute("COMMIT")
                return PolymarketRuntimeControlSnapshot(
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
        normalized_reason = str(reason or "").strip()
        if not normalized_reason or len(normalized_reason) > 160:
            raise ValueError("release reason is invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM polymarket_runtime_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise PolymarketLiveError(
                    "Polymarket runtime control record is missing"
                )
            current = self._snapshot_row(row)
            if current.state == "stopped" and not current.lease_id:
                connection.execute("COMMIT")
                return
            if current.lease_id != str(lease_id):
                raise PolymarketLiveBlocked(
                    "Polymarket runtime lease changed before release"
                )
            now = self._now_ms()
            payload = {
                "schema_version": POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
                "state": "stopped",
                "lease_id": "",
                "owner_process_id": 0,
                "started_at_ms": 0,
                "heartbeat_at_ms": 0,
                "stop_epoch": current.stop_epoch,
                "stop_requested_at_ms": current.stop_requested_at_ms,
                "stop_reason": normalized_reason,
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
            raise PolymarketLiveBlocked(
                "stale runtime Stop cannot complete with owned exposure"
            )
        with self._interlock():
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM polymarket_runtime_control
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    raise PolymarketLiveError(
                        "Polymarket runtime control record is missing"
                    )
                current = self._snapshot_row(row)
                if current.state == "stopped":
                    connection.execute("COMMIT")
                    return True
                now = self._now_ms()
                if (
                    current.state != "stop_requested"
                    or current.heartbeat_at_ms <= 0
                    or current.heartbeat_at_ms > now + 5_000
                    or now - current.heartbeat_at_ms <= self.maximum_heartbeat_age_ms
                ):
                    connection.execute("COMMIT")
                    return False
                payload = {
                    "schema_version": POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
                    "state": "stopped",
                    "lease_id": "",
                    "owner_process_id": 0,
                    "started_at_ms": 0,
                    "heartbeat_at_ms": 0,
                    "stop_epoch": current.stop_epoch,
                    "stop_requested_at_ms": current.stop_requested_at_ms,
                    "stop_reason": "stale_runtime_stop_completed",
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
    ) -> PolymarketRuntimeControlSnapshot:
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
class PolymarketRuntimeLeaseInterlock:
    control: PolymarketRuntimeControl
    lease_id: str

    def assert_opening_allowed(self) -> None:
        self.control.assert_opening_allowed(self.lease_id)

    @contextmanager
    def submission_guard(self) -> Iterator[None]:
        with self.control.submission_guard(self.lease_id):
            yield


class PolymarketRuntimeControlService:
    """Heartbeat the lease and turn any external Stop into supervisor shutdown."""

    trading_authority = False

    def __init__(
        self,
        control: PolymarketRuntimeControl,
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
    ) -> None:
        next_heartbeat = 0.0
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_heartbeat:
                valid = await asyncio.to_thread(
                    self.control.heartbeat,
                    self.lease_id,
                )
                next_heartbeat = now + self.heartbeat_interval_seconds
            else:
                snapshot = await asyncio.to_thread(self.control.snapshot)
                valid = (
                    snapshot.state == "running" and snapshot.lease_id == self.lease_id
                )
            if not valid:
                request_stop()
                return
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
    "POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION",
    "PolymarketRuntimeControl",
    "PolymarketRuntimeControlService",
    "PolymarketRuntimeControlSnapshot",
    "PolymarketRuntimeLeaseInterlock",
]
