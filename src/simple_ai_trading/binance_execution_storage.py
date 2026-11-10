"""Explicit enrollment of Binance execution storage, independent of its ledger.

The existing runtime enrollment format binds the journal and the controller's
two interlocks. This detects loss/replacement, not coordinated local rollback or
an actor able to rewrite every authority file. No account or worker is created.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sqlite3
import uuid

from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import BinanceOpenIntentJournal, OpenIntentError
from .binance_runtime_control import BinanceRuntimeControl
from .position_transactions import position_transaction
from .runtime_control_storage import (
    FileIdentity,
    RuntimeStorageEnrollment,
    validate_parent,
)

_NAMESPACE = "binance_execution"
_SCHEMA = "binance-execution-storage-v1"
_MAX_INSPECTION_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BinanceExecutionStorage:
    journal: BinanceOpenIntentJournal
    control: BinanceRuntimeControl
    enrollment_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.journal, BinanceOpenIntentJournal) or not isinstance(
            self.control, BinanceRuntimeControl
        ):
            raise OpenIntentError("execution enrollment requires Binance storage")
        path = Path(self.journal.path).absolute()
        object.__setattr__(self, "journal", BinanceOpenIntentJournal(path))
        object.__setattr__(
            self,
            "enrollment_path",
            Path(self.enrollment_path).absolute()
            if self.enrollment_path is not None
            else path.with_name(path.name + ".binance_execution.enrollment.json"),
        )

    def _record(self, *, prepared: bool = False) -> RuntimeStorageEnrollment:
        record = RuntimeStorageEnrollment.read(
            self.enrollment_path, allow_prepared=prepared
        )
        if (record.namespace, record.control_schema_version) != (_NAMESPACE, _SCHEMA):
            raise OpenIntentError("execution enrollment namespace differs")
        return record

    @staticmethod
    def _uuid(connection: sqlite3.Connection) -> str | None:
        columns = tuple(
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(runtime_storage_enrollment)"
            )
        )
        if not columns:
            return None
        if columns != ("namespace", "store_id"):
            raise OpenIntentError("execution storage identity schema differs")
        rows = connection.execute(
            "SELECT namespace, store_id FROM runtime_storage_enrollment LIMIT 2"
        ).fetchall()
        if (
            len(rows) != 1
            or rows[0][0] != _NAMESPACE
            or not isinstance(rows[0][1], str)
            or re.fullmatch(r"[0-9a-f]{32}", rows[0][1]) is None
        ):
            raise OpenIntentError("execution storage identity is missing or ambiguous")
        return rows[0][1]

    @staticmethod
    def _digest(connection: sqlite3.Connection) -> str:
        if not connection.in_transaction:
            connection.execute("BEGIN")
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        size = connection.execute("PRAGMA page_size").fetchone()[0]
        if pages * size > _MAX_INSPECTION_BYTES:
            raise OpenIntentError(
                "execution adoption snapshot exceeds inspection budget"
            )
        return hashlib.sha256(connection.serialize()).hexdigest()

    def inspect(self) -> dict:
        """Return local adoption evidence without creating or repairing storage."""
        result = {"new_exposure_authorized": False, "journal_exists": False}
        if self.journal.path.exists():
            FileIdentity.at(self.journal.path)
            with closing(self.journal._connect()) as connection:
                result.update(
                    journal_exists=True,
                    journal_snapshot_sha256=self._digest(connection),
                    execution_store_id=self._uuid(connection),
                )
        if self.enrollment_path.exists():
            record = self._record(prepared=True)
            result.update(
                enrollment_state=record.state,
                enrollment_store_id=record.store_id,
                enrollment_record_sha256=hashlib.sha256(record.encoded()).hexdigest(),
            )
        else:
            result["enrollment_state"] = "unenrolled"
        return result

    def _resources(self, record: RuntimeStorageEnrollment) -> None:
        control = self.control._enrollment()
        self.control._verify_resources(control)
        if (record.submission_lock, record.close_lock) != (
            control.submission_lock,
            control.close_lock,
        ):
            raise OpenIntentError(
                "execution enrollment belongs to different interlocks"
            )
        record.database.verify(self.journal.path)

    def _verify_record(
        self, record: RuntimeStorageEnrollment, scope: BinanceExecutionScope
    ) -> None:
        self._resources(record)
        with closing(self.journal._connect()) as connection:
            if self._uuid(connection) != record.store_id:
                raise OpenIntentError("execution storage UUID differs from enrollment")
            self.journal._bind_scope(connection, scope)
            self._resources(record)

    def verify(self, *, scope: BinanceExecutionScope) -> None:
        """Opening-only gate; never initialize missing metadata or namespace."""
        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("execution enrollment requires a validated scope")
        try:
            self._verify_record(self._record(), scope)
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise OpenIntentError(
                "execution storage enrollment is unavailable or changed"
            ) from exc

    def _stopped(self) -> None:
        snapshot = self.control.snapshot()
        if (
            snapshot.state not in {"stopped", "stop_requested"}
            or snapshot.lease_id
            or snapshot.owner_process_id != 0
            or not snapshot.paused
        ):
            raise OpenIntentError(
                "execution enrollment requires ownerless, Stop-latched control"
            )

    def provision(
        self,
        *,
        scope: BinanceExecutionScope,
        expected_snapshot_sha256: str | None = None,
    ) -> dict:
        """Explicit fresh setup or exact-snapshot adoption; never clear Stop."""
        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("execution enrollment requires a validated scope")
        if expected_snapshot_sha256 is not None and (
            not isinstance(expected_snapshot_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_snapshot_sha256) is None
        ):
            raise OpenIntentError("execution adoption digest is invalid")
        with self.control._interlock():
            self._stopped()
            if self.enrollment_path.exists() or self.enrollment_path.is_symlink():
                raise OpenIntentError(
                    "execution enrollment exists; restore or recover it"
                )
            existed = self.journal.path.exists()
            if existed != (expected_snapshot_sha256 is not None):
                raise OpenIntentError(
                    "existing execution storage requires exact adoption evidence"
                )
            if existed:
                FileIdentity.at(self.journal.path)
            else:
                validate_parent(self.journal.path)
            with position_transaction(self.journal, write=True) as transaction:
                connection = transaction.connection
                if connection is None or self._uuid(connection) is not None:
                    raise OpenIntentError("execution storage already has an identity")
                if (
                    not existed
                    and connection.execute("PRAGMA user_version").fetchone()[0] != 2
                ):
                    raise OpenIntentError(
                        "execution storage appeared during fresh setup"
                    )
                if existed and self._digest(connection) != expected_snapshot_sha256:
                    raise OpenIntentError("execution adoption snapshot changed")
                self.journal._prepare_scope(transaction, scope)
                control = self.control._enrollment()
                self.control._verify_resources(control)
                record = RuntimeStorageEnrollment(
                    _NAMESPACE,
                    _SCHEMA,
                    uuid.uuid4().hex,
                    "prepared",
                    FileIdentity.at(self.journal.path),
                    control.submission_lock,
                    control.close_lock,
                )
                connection.execute(
                    "CREATE TABLE runtime_storage_enrollment ("
                    "namespace TEXT PRIMARY KEY, store_id TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO runtime_storage_enrollment VALUES (?, ?)",
                    (_NAMESPACE, record.store_id),
                )
                record.prepare(self.enrollment_path)
            self._verify_record(record, scope)
            record.finish(self.enrollment_path)
        return self.inspect()

    def recover(
        self,
        *,
        scope: BinanceExecutionScope,
        expected_store_id: str,
        expected_record_sha256: str,
    ) -> dict:
        """Finish only a matching committed prepared enrollment; do not adopt."""
        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("execution enrollment requires a validated scope")
        with self.control._interlock():
            self._stopped()
            record = self._record(prepared=True)
            if (
                record.state != "prepared"
                or record.store_id != expected_store_id
                or hashlib.sha256(record.encoded()).hexdigest()
                != expected_record_sha256
            ):
                raise OpenIntentError("execution prepared enrollment evidence differs")
            self._verify_record(record, scope)
            record.finish(self.enrollment_path)
        return self.inspect()
