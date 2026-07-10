"""Durable, non-resettable-by-API terminal-holdout governance."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping

from .assets import is_supported_major_symbol, normalize_symbol
from .types import config_paths

LEDGER_SCHEMA_VERSION = "terminal-holdout-ledger-v1"
RESERVATION_SCHEMA_VERSION = "terminal-holdout-reservation-v1"
SUPPORTED_RISK_OBJECTIVES = frozenset({"conservative", "regular", "aggressive"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_RESULT_STATUSES = frozenset({"accepted", "rejected", "evaluation_error"})


class TerminalHoldoutLedgerError(RuntimeError):
    """Base class for terminal-governance failures."""


class TerminalHoldoutReuseError(TerminalHoldoutLedgerError):
    """Raised before evaluation when a terminal market period was already reserved."""


def default_terminal_holdout_ledger_path() -> Path:
    override = str(os.environ.get("SIMPLE_AI_TRADING_TERMINAL_LEDGER") or "").strip()
    if override:
        return Path(override).expanduser()
    return config_paths()["base"] / "terminal_holdouts.sqlite3"


def _sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def terminal_model_fingerprint(model: object) -> str:
    """Bind the evaluated model while excluding later governance stamps."""

    if not is_dataclass(model):
        raise TypeError("terminal model fingerprint requires a dataclass model")
    payload = asdict(model)
    if not isinstance(payload, dict):  # pragma: no cover - dataclasses always return mappings
        raise TypeError("terminal model fingerprint payload is invalid")
    payload.pop("selection_risk", None)
    payload.pop("execution_validation", None)
    canonical = json.dumps(
        {
            "schema_version": RESERVATION_SCHEMA_VERSION,
            "model": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def terminal_result_fingerprint(raw: object) -> str:
    if not isinstance(raw, Mapping):
        raise TypeError("terminal result fingerprint requires a mapping")
    payload = dict(raw)
    payload.pop("reservation", None)
    canonical = json.dumps(
        {
            "schema_version": RESERVATION_SCHEMA_VERSION,
            "terminal_holdout": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reservation_evidence_passed(
    raw: object,
    *,
    expected_dataset_fingerprint: str | None = None,
    expected_model_fingerprint: str | None = None,
    expected_result_fingerprint: str | None = None,
    expected_rows: int | None = None,
    expected_first_timestamp: int | None = None,
    expected_last_timestamp: int | None = None,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_objective: str | None = None,
) -> bool:
    if not isinstance(raw, Mapping):
        return False
    try:
        reservation_id = _sha256(raw.get("reservation_id"), label="reservation_id")
        ledger_id = _sha256(raw.get("ledger_id"), label="ledger_id")
        dataset_fingerprint = _sha256(
            raw.get("dataset_fingerprint"),
            label="dataset_fingerprint",
        )
        model_fingerprint = _sha256(raw.get("model_fingerprint"), label="model_fingerprint")
        result_fingerprint = _sha256(raw.get("result_fingerprint"), label="result_fingerprint")
        symbol = normalize_symbol(str(raw.get("symbol") or ""), default="")
        market_type = str(raw.get("market_type") or "").strip().lower()
        objective = str(raw.get("objective") or "").strip().lower()
        first_timestamp = _strict_int(raw.get("first_timestamp"), label="first_timestamp")
        last_timestamp = _strict_int(raw.get("last_timestamp"), label="last_timestamp")
        rows = _strict_int(raw.get("rows"), label="rows")
        reserved_at_ms = _strict_int(raw.get("reserved_at_ms"), label="reserved_at_ms")
        completed_at_ms = _strict_int(raw.get("completed_at_ms"), label="completed_at_ms")
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        raw.get("schema_version") != RESERVATION_SCHEMA_VERSION
        or not reservation_id
        or not ledger_id
        or raw.get("status") != "complete"
        or raw.get("result_status") != "accepted"
        or raw.get("error") not in (None, "")
        or not is_supported_major_symbol(symbol)
        or market_type not in {"spot", "futures"}
        or objective not in SUPPORTED_RISK_OBJECTIVES
        or first_timestamp < 0
        or last_timestamp < first_timestamp
        or rows <= 0
        or reserved_at_ms <= 0
        or completed_at_ms < reserved_at_ms
    ):
        return False
    expected_digests = (
        (dataset_fingerprint, expected_dataset_fingerprint),
        (model_fingerprint, expected_model_fingerprint),
        (result_fingerprint, expected_result_fingerprint),
    )
    for actual, expected in expected_digests:
        if expected is not None:
            try:
                if actual != _sha256(expected, label="expected fingerprint"):
                    return False
            except ValueError:
                return False
    expected_values = (
        (rows, expected_rows),
        (first_timestamp, expected_first_timestamp),
        (last_timestamp, expected_last_timestamp),
    )
    try:
        if any(expected is not None and actual != int(expected) for actual, expected in expected_values):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    if expected_symbol is not None and symbol != normalize_symbol(expected_symbol, default=""):
        return False
    if expected_market_type is not None and market_type != str(expected_market_type).strip().lower():
        return False
    if expected_objective is not None and objective != str(expected_objective).strip().lower():
        return False
    return True


class TerminalHoldoutLedger:
    """SQLite audit ledger that atomically rejects reused terminal periods."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_terminal_holdout_ledger_path()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            current_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if current_mode != "delete":
                current_mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
            if current_mode != "delete":
                raise TerminalHoldoutLedgerError(
                    f"terminal ledger must use rollback journal mode, got {current_mode!r}"
                )
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            self._initialize_schema(connection)
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or str(quick_check[0]).lower() != "ok":
                raise TerminalHoldoutLedgerError("terminal ledger integrity check failed")
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _initialize_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS governance_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS terminal_holdout_reservations (
                reservation_id TEXT PRIMARY KEY,
                ledger_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                objective TEXT NOT NULL,
                first_timestamp INTEGER NOT NULL,
                last_timestamp INTEGER NOT NULL,
                rows INTEGER NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                model_fingerprint TEXT NOT NULL,
                result_fingerprint TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                result_status TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                reserved_at_ms INTEGER NOT NULL,
                completed_at_ms INTEGER,
                CHECK (market_type IN ('spot', 'futures')),
                CHECK (objective IN ('conservative', 'regular', 'aggressive')),
                CHECK (first_timestamp >= 0 AND last_timestamp >= first_timestamp),
                CHECK (rows > 0),
                CHECK (status IN ('reserved', 'complete', 'failed')),
                UNIQUE (symbol, market_type, objective, dataset_fingerprint)
            );
            CREATE INDEX IF NOT EXISTS terminal_holdout_overlap_idx
            ON terminal_holdout_reservations (
                symbol, market_type, objective, first_timestamp, last_timestamp
            );
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info('terminal_holdout_reservations')")
        }
        if "result_fingerprint" not in columns:
            connection.execute(
                "ALTER TABLE terminal_holdout_reservations "
                "ADD COLUMN result_fingerprint TEXT NOT NULL DEFAULT ''"
            )
        connection.execute(
            "INSERT OR IGNORE INTO governance_metadata (key, value) VALUES ('schema_version', ?)",
            [LEDGER_SCHEMA_VERSION],
        )
        schema_row = connection.execute(
            "SELECT value FROM governance_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if schema_row is None or str(schema_row[0]) != LEDGER_SCHEMA_VERSION:
            raise TerminalHoldoutLedgerError("unsupported terminal ledger schema")
        ledger_row = connection.execute(
            "SELECT value FROM governance_metadata WHERE key = 'ledger_id'"
        ).fetchone()
        if ledger_row is None:
            ledger_id = hashlib.sha256(os.urandom(32)).hexdigest()
            connection.execute(
                "INSERT OR IGNORE INTO governance_metadata (key, value) VALUES ('ledger_id', ?)",
                [ledger_id],
            )
            ledger_row = connection.execute(
                "SELECT value FROM governance_metadata WHERE key = 'ledger_id'"
            ).fetchone()
        if ledger_row is None:
            raise TerminalHoldoutLedgerError("terminal ledger identity was not initialized")
        _sha256(ledger_row[0], label="ledger_id")

    @staticmethod
    def _validate_contract(
        *,
        symbol: str,
        market_type: str,
        objective: str,
        first_timestamp: int,
        last_timestamp: int,
        rows: int,
        dataset_fingerprint: str,
        model_fingerprint: str,
    ) -> dict[str, object]:
        normalized_symbol = normalize_symbol(symbol, default="")
        normalized_market = str(market_type).strip().lower()
        normalized_objective = str(objective).strip().lower()
        first = int(first_timestamp)
        last = int(last_timestamp)
        row_count = int(rows)
        if not is_supported_major_symbol(normalized_symbol):
            raise ValueError(f"unsupported terminal holdout symbol: {normalized_symbol}")
        if normalized_market not in {"spot", "futures"}:
            raise ValueError("terminal holdout market_type must be spot or futures")
        if normalized_objective not in SUPPORTED_RISK_OBJECTIVES:
            raise ValueError("terminal holdout objective is unsupported")
        if first < 0 or last < first:
            raise ValueError("terminal holdout timestamp range is invalid")
        if row_count <= 0:
            raise ValueError("terminal holdout rows must be positive")
        return {
            "symbol": normalized_symbol,
            "market_type": normalized_market,
            "objective": normalized_objective,
            "first_timestamp": first,
            "last_timestamp": last,
            "rows": row_count,
            "dataset_fingerprint": _sha256(
                dataset_fingerprint,
                label="dataset_fingerprint",
            ),
            "model_fingerprint": _sha256(model_fingerprint, label="model_fingerprint"),
        }

    @staticmethod
    def _row_payload(row: sqlite3.Row) -> dict[str, object]:
        return {
            "schema_version": RESERVATION_SCHEMA_VERSION,
            "reservation_id": str(row["reservation_id"]),
            "ledger_id": str(row["ledger_id"]),
            "symbol": str(row["symbol"]),
            "market_type": str(row["market_type"]),
            "objective": str(row["objective"]),
            "first_timestamp": int(row["first_timestamp"]),
            "last_timestamp": int(row["last_timestamp"]),
            "rows": int(row["rows"]),
            "dataset_fingerprint": str(row["dataset_fingerprint"]),
            "model_fingerprint": str(row["model_fingerprint"]),
            "result_fingerprint": str(row["result_fingerprint"]),
            "status": str(row["status"]),
            "result_status": str(row["result_status"]),
            "error": str(row["error"]),
            "reserved_at_ms": int(row["reserved_at_ms"]),
            "completed_at_ms": (
                int(row["completed_at_ms"])
                if row["completed_at_ms"] is not None
                else None
            ),
        }

    def reserve(
        self,
        *,
        symbol: str,
        market_type: str,
        objective: str,
        first_timestamp: int,
        last_timestamp: int,
        rows: int,
        dataset_fingerprint: str,
        model_fingerprint: str,
    ) -> dict[str, object]:
        contract = self._validate_contract(
            symbol=symbol,
            market_type=market_type,
            objective=objective,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            rows=rows,
            dataset_fingerprint=dataset_fingerprint,
            model_fingerprint=model_fingerprint,
        )
        now_ms = time.time_ns() // 1_000_000
        seed = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")
        reservation_id = hashlib.sha256(seed + os.urandom(32) + str(time.time_ns()).encode("ascii")).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            ledger_row = connection.execute(
                "SELECT value FROM governance_metadata WHERE key = 'ledger_id'"
            ).fetchone()
            if ledger_row is None:
                raise TerminalHoldoutLedgerError("terminal ledger identity is missing")
            ledger_id = _sha256(ledger_row[0], label="ledger_id")
            overlap = connection.execute(
                """
                SELECT reservation_id, first_timestamp, last_timestamp, status
                FROM terminal_holdout_reservations
                WHERE symbol = ? AND market_type = ? AND objective = ?
                  AND first_timestamp <= ? AND last_timestamp >= ?
                ORDER BY reserved_at_ms, reservation_id
                LIMIT 1
                """,
                [
                    contract["symbol"],
                    contract["market_type"],
                    contract["objective"],
                    contract["last_timestamp"],
                    contract["first_timestamp"],
                ],
            ).fetchone()
            if overlap is not None:
                raise TerminalHoldoutReuseError(
                    "terminal holdout overlaps a previously reserved period: "
                    f"reservation={overlap['reservation_id']} "
                    f"timestamps={overlap['first_timestamp']}..{overlap['last_timestamp']} "
                    f"status={overlap['status']}"
                )
            connection.execute(
                """
                INSERT INTO terminal_holdout_reservations (
                    reservation_id, ledger_id, symbol, market_type, objective,
                    first_timestamp, last_timestamp, rows,
                    dataset_fingerprint, model_fingerprint,
                    result_fingerprint, status, result_status, error,
                    reserved_at_ms, completed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 'reserved', '', '', ?, NULL)
                """,
                [
                    reservation_id,
                    ledger_id,
                    contract["symbol"],
                    contract["market_type"],
                    contract["objective"],
                    contract["first_timestamp"],
                    contract["last_timestamp"],
                    contract["rows"],
                    contract["dataset_fingerprint"],
                    contract["model_fingerprint"],
                    now_ms,
                ],
            )
            row = connection.execute(
                "SELECT * FROM terminal_holdout_reservations WHERE reservation_id = ?",
                [reservation_id],
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:  # pragma: no cover - guarded by the inserted primary key
                raise TerminalHoldoutLedgerError("terminal reservation insert was not readable")
            return self._row_payload(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finalize(
        self,
        reservation_id: str,
        *,
        result_status: str,
        result_fingerprint: str,
        error: str = "",
    ) -> dict[str, object]:
        reservation = _sha256(reservation_id, label="reservation_id")
        result = str(result_status).strip().lower()
        if result not in _RESULT_STATUSES:
            raise ValueError("terminal holdout result_status is invalid")
        result_sha256 = _sha256(result_fingerprint, label="result_fingerprint")
        detail = str(error).strip()[:2_000]
        status = "failed" if result == "evaluation_error" else "complete"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status, reserved_at_ms FROM terminal_holdout_reservations WHERE reservation_id = ?",
                [reservation],
            ).fetchone()
            if current is None:
                raise TerminalHoldoutLedgerError("terminal reservation does not exist")
            if str(current["status"]) != "reserved":
                raise TerminalHoldoutLedgerError("terminal reservation was already finalized")
            completed_at_ms = max(
                int(current["reserved_at_ms"]),
                time.time_ns() // 1_000_000,
            )
            connection.execute(
                """
                UPDATE terminal_holdout_reservations
                SET status = ?, result_status = ?, result_fingerprint = ?,
                    error = ?, completed_at_ms = ?
                WHERE reservation_id = ? AND status = 'reserved'
                """,
                [status, result, result_sha256, detail, completed_at_ms, reservation],
            )
            row = connection.execute(
                "SELECT * FROM terminal_holdout_reservations WHERE reservation_id = ?",
                [reservation],
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:  # pragma: no cover - guarded by the initial lookup
                raise TerminalHoldoutLedgerError("terminal reservation disappeared")
            return self._row_payload(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def reservation(self, reservation_id: str) -> dict[str, object] | None:
        reservation = _sha256(reservation_id, label="reservation_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM terminal_holdout_reservations WHERE reservation_id = ?",
                [reservation],
            ).fetchone()
            return self._row_payload(row) if row is not None else None
        finally:
            connection.close()

    def evidence_matches(self, raw: object) -> bool:
        if not reservation_evidence_passed(raw):
            return False
        assert isinstance(raw, Mapping)
        try:
            stored = self.reservation(str(raw.get("reservation_id")))
        except (OSError, sqlite3.Error, TerminalHoldoutLedgerError, ValueError):
            return False
        if stored is None:
            return False
        keys = (
            "schema_version",
            "reservation_id",
            "ledger_id",
            "symbol",
            "market_type",
            "objective",
            "first_timestamp",
            "last_timestamp",
            "rows",
            "dataset_fingerprint",
            "model_fingerprint",
            "result_fingerprint",
            "status",
            "result_status",
            "error",
            "reserved_at_ms",
            "completed_at_ms",
        )
        return all(stored.get(key) == raw.get(key) for key in keys)


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "RESERVATION_SCHEMA_VERSION",
    "TerminalHoldoutLedger",
    "TerminalHoldoutLedgerError",
    "TerminalHoldoutReuseError",
    "default_terminal_holdout_ledger_path",
    "reservation_evidence_passed",
    "terminal_model_fingerprint",
    "terminal_result_fingerprint",
]
