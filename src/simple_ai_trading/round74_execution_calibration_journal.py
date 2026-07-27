"""Crash-recoverable intent journal for Round 74 testnet calibration orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

import duckdb

from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS
from .impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX,
)


ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION = (
    "round-074-execution-calibration-journal-v1"
)
ROUND74_EXECUTION_JOURNAL_STATES = frozenset(
    {
        "PREPARED",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "FILLED",
        "REJECTED",
        "UNKNOWN",
        "FLAT_VERIFIED",
    }
)
ROUND74_EXECUTION_JOURNAL_BLOCKING_STATES = frozenset(
    {"PREPARED", "SUBMITTED", "ACKNOWLEDGED", "UNKNOWN"}
)
_ALLOWED_TRANSITIONS = {
    "PREPARED": frozenset(
        {"SUBMITTED", "ACKNOWLEDGED", "FILLED", "REJECTED", "UNKNOWN"}
    ),
    "SUBMITTED": frozenset(
        {"ACKNOWLEDGED", "FILLED", "REJECTED", "UNKNOWN"}
    ),
    "ACKNOWLEDGED": frozenset({"FILLED", "REJECTED", "UNKNOWN"}),
    "UNKNOWN": frozenset(
        {"ACKNOWLEDGED", "FILLED", "REJECTED", "UNKNOWN"}
    ),
    "FILLED": frozenset({"FLAT_VERIFIED"}),
    "REJECTED": frozenset(
        {"SUBMITTED", "ACKNOWLEDGED", "FILLED", "REJECTED", "UNKNOWN"}
    ),
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    selected = str(value).strip()
    if not _IDENTIFIER.fullmatch(selected):
        raise ValueError(f"Round 74 execution journal {label} differs")
    return selected


def _sha256(value: object, *, label: str) -> str:
    selected = str(value)
    if not _SHA256.fullmatch(selected):
        raise ValueError(f"Round 74 execution journal {label} differs")
    return selected


def _decimal(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 74 execution journal {label} differs")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"Round 74 execution journal {label} differs"
        ) from exc
    if not selected.is_finite() or (positive and selected <= 0):
        raise ValueError(f"Round 74 execution journal {label} differs")
    return selected


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Round 74 execution journal {label} differs")
    return value


@dataclass(frozen=True)
class Round74ExecutionCalibrationIntent:
    calibration_run_id: str
    round_trip_id: str
    path: str
    symbol: str
    side: str
    client_order_id: str
    quantity: Decimal
    reference_quote_notional: Decimal
    reduce_only: bool
    created_wall_ns: int
    book_source_sha256: str

    def validate(self) -> None:
        _identifier(self.calibration_run_id, label="calibration run ID")
        _identifier(self.round_trip_id, label="round trip ID")
        _identifier(self.client_order_id, label="client order ID")
        if (
            self.path not in {"entry", "exit"}
            or self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or self.side not in {"BUY", "SELL"}
            or not self.client_order_id.startswith(
                ROUND74_EXECUTION_CALIBRATION_CLIENT_PREFIX
            )
            or len(self.client_order_id) > 36
            or self.reduce_only is not (self.path == "exit")
        ):
            raise ValueError("Round 74 execution journal intent differs")
        _decimal(self.quantity, label="quantity", positive=True)
        _decimal(
            self.reference_quote_notional,
            label="reference quote notional",
            positive=True,
        )
        _positive_integer(self.created_wall_ns, label="created wall time")
        _sha256(self.book_source_sha256, label="book source digest")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION,
            "calibration_run_id": self.calibration_run_id,
            "round_trip_id": self.round_trip_id,
            "path": self.path,
            "symbol": self.symbol,
            "side": self.side,
            "client_order_id": self.client_order_id,
            "quantity": format(self.quantity, "f"),
            "reference_quote_notional": format(
                self.reference_quote_notional,
                "f",
            ),
            "reduce_only": self.reduce_only,
            "created_wall_ns": self.created_wall_ns,
            "book_source_sha256": self.book_source_sha256,
        }

    @property
    def intent_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class Round74ExecutionCalibrationTransition:
    state: str
    occurred_wall_ns: int
    source_payload_sha256: str
    order_id: int = 0
    executed_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal = Decimal("0")
    reason: str = ""

    def validate(self) -> None:
        if self.state not in ROUND74_EXECUTION_JOURNAL_STATES:
            raise ValueError("Round 74 execution journal state differs")
        _positive_integer(self.occurred_wall_ns, label="event wall time")
        _sha256(self.source_payload_sha256, label="event source digest")
        if (
            isinstance(self.order_id, bool)
            or not isinstance(self.order_id, int)
            or self.order_id < 0
        ):
            raise ValueError("Round 74 execution journal order ID differs")
        quantity = _decimal(
            self.executed_quantity,
            label="executed quantity",
        )
        average = _decimal(
            self.average_fill_price,
            label="average fill price",
        )
        if (
            quantity < 0
            or average < 0
            or (self.state in {"FILLED", "FLAT_VERIFIED"})
            and (self.order_id <= 0 or quantity <= 0 or average <= 0)
            or len(self.reason) > 512
            or any(ord(character) < 0x20 for character in self.reason)
        ):
            raise ValueError("Round 74 execution journal transition differs")


@dataclass(frozen=True)
class Round74ExecutionCalibrationSnapshot:
    intent: Round74ExecutionCalibrationIntent
    state: str
    sequence_number: int
    event_sha256: str
    occurred_wall_ns: int
    order_id: int
    executed_quantity: Decimal
    average_fill_price: Decimal
    reason: str


class Round74ExecutionCalibrationJournal:
    """Append-only recovery state in the shared DuckDB evidence database."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS round74_execution_calibration_intent (
                client_order_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                calibration_run_id VARCHAR NOT NULL,
                round_trip_id VARCHAR NOT NULL,
                path VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                quantity VARCHAR NOT NULL,
                reference_quote_notional VARCHAR NOT NULL,
                reduce_only BOOLEAN NOT NULL,
                created_wall_ns UBIGINT NOT NULL,
                book_source_sha256 VARCHAR NOT NULL,
                intent_sha256 VARCHAR NOT NULL,
                UNIQUE(calibration_run_id, round_trip_id, path)
            );

            CREATE TABLE IF NOT EXISTS round74_execution_calibration_event (
                event_sha256 VARCHAR PRIMARY KEY,
                client_order_id VARCHAR NOT NULL,
                sequence_number UINTEGER NOT NULL,
                state VARCHAR NOT NULL,
                occurred_wall_ns UBIGINT NOT NULL,
                source_payload_sha256 VARCHAR NOT NULL,
                order_id UBIGINT NOT NULL,
                executed_quantity VARCHAR NOT NULL,
                average_fill_price VARCHAR NOT NULL,
                reason VARCHAR NOT NULL,
                previous_event_sha256 VARCHAR NOT NULL,
                UNIQUE(client_order_id, sequence_number)
            );
            """
        )

    @staticmethod
    def _initial_event(intent: Round74ExecutionCalibrationIntent) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION,
            "client_order_id": intent.client_order_id,
            "sequence_number": 1,
            "state": "PREPARED",
            "occurred_wall_ns": intent.created_wall_ns,
            "source_payload_sha256": intent.book_source_sha256,
            "order_id": 0,
            "executed_quantity": "0",
            "average_fill_price": "0",
            "reason": "",
            "previous_event_sha256": intent.intent_sha256,
        }
        value["event_sha256"] = _canonical_sha256(value)
        return value

    def record_intent(
        self,
        intent: Round74ExecutionCalibrationIntent,
    ) -> Round74ExecutionCalibrationSnapshot:
        intent.validate()
        existing = self.connection.execute(
            """
            SELECT intent_sha256
            FROM round74_execution_calibration_intent
            WHERE client_order_id = ?
            """,
            [intent.client_order_id],
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != intent.intent_sha256:
                raise ValueError(
                    "Round 74 execution journal intent identity differs"
                )
            return self.current(intent.client_order_id)
        initial = self._initial_event(intent)
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                """
                INSERT INTO round74_execution_calibration_intent
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    intent.client_order_id,
                    ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION,
                    intent.calibration_run_id,
                    intent.round_trip_id,
                    intent.path,
                    intent.symbol,
                    intent.side,
                    format(intent.quantity, "f"),
                    format(intent.reference_quote_notional, "f"),
                    intent.reduce_only,
                    intent.created_wall_ns,
                    intent.book_source_sha256,
                    intent.intent_sha256,
                ],
            )
            self.connection.execute(
                """
                INSERT INTO round74_execution_calibration_event
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    initial["event_sha256"],
                    initial["client_order_id"],
                    initial["sequence_number"],
                    initial["state"],
                    initial["occurred_wall_ns"],
                    initial["source_payload_sha256"],
                    initial["order_id"],
                    initial["executed_quantity"],
                    initial["average_fill_price"],
                    initial["reason"],
                    initial["previous_event_sha256"],
                ],
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return self.current(intent.client_order_id)

    def _intent(self, row: tuple[object, ...]) -> Round74ExecutionCalibrationIntent:
        return Round74ExecutionCalibrationIntent(
            calibration_run_id=str(row[0]),
            round_trip_id=str(row[1]),
            path=str(row[2]),
            symbol=str(row[3]),
            side=str(row[4]),
            client_order_id=str(row[5]),
            quantity=Decimal(str(row[6])),
            reference_quote_notional=Decimal(str(row[7])),
            reduce_only=bool(row[8]),
            created_wall_ns=int(row[9]),
            book_source_sha256=str(row[10]),
        )

    def current(
        self,
        client_order_id: str,
    ) -> Round74ExecutionCalibrationSnapshot:
        selected = _identifier(client_order_id, label="client order ID")
        row = self.connection.execute(
            """
            SELECT
                i.calibration_run_id, i.round_trip_id, i.path, i.symbol,
                i.side, i.client_order_id, i.quantity,
                i.reference_quote_notional, i.reduce_only, i.created_wall_ns,
                i.book_source_sha256, e.state, e.sequence_number,
                e.event_sha256, e.occurred_wall_ns, e.order_id,
                e.executed_quantity, e.average_fill_price, e.reason
            FROM round74_execution_calibration_intent AS i
            JOIN round74_execution_calibration_event AS e
              ON e.client_order_id = i.client_order_id
            WHERE i.client_order_id = ?
            ORDER BY e.sequence_number DESC
            LIMIT 1
            """,
            [selected],
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Round 74 execution journal intent is unknown: {selected}"
            )
        intent = self._intent(tuple(row[:11]))
        intent.validate()
        return Round74ExecutionCalibrationSnapshot(
            intent=intent,
            state=str(row[11]),
            sequence_number=int(row[12]),
            event_sha256=str(row[13]),
            occurred_wall_ns=int(row[14]),
            order_id=int(row[15]),
            executed_quantity=Decimal(str(row[16])),
            average_fill_price=Decimal(str(row[17])),
            reason=str(row[18]),
        )

    def transition(
        self,
        client_order_id: str,
        transition: Round74ExecutionCalibrationTransition,
    ) -> Round74ExecutionCalibrationSnapshot:
        transition.validate()
        current = self.current(client_order_id)
        allowed = _ALLOWED_TRANSITIONS.get(current.state, frozenset())
        if transition.state not in allowed:
            raise ValueError(
                "Round 74 execution journal state transition differs"
            )
        if (
            transition.occurred_wall_ns < current.occurred_wall_ns
            or transition.executed_quantity < current.executed_quantity
            or (
                current.order_id > 0
                and transition.order_id != current.order_id
            )
            or (
                transition.executed_quantity > current.intent.quantity
            )
            or current.state == "REJECTED"
            and current.intent.path != "exit"
            or transition.state == "FLAT_VERIFIED"
            and current.intent.path != "exit"
        ):
            raise ValueError(
                "Round 74 execution journal transition accounting differs"
            )
        sequence_number = current.sequence_number + 1
        value: dict[str, object] = {
            "schema_version": ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION,
            "client_order_id": current.intent.client_order_id,
            "sequence_number": sequence_number,
            "state": transition.state,
            "occurred_wall_ns": transition.occurred_wall_ns,
            "source_payload_sha256": transition.source_payload_sha256,
            "order_id": transition.order_id,
            "executed_quantity": format(
                transition.executed_quantity,
                "f",
            ),
            "average_fill_price": format(
                transition.average_fill_price,
                "f",
            ),
            "reason": transition.reason,
            "previous_event_sha256": current.event_sha256,
        }
        event_sha256 = _canonical_sha256(value)
        self.connection.execute(
            """
            INSERT INTO round74_execution_calibration_event
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event_sha256,
                current.intent.client_order_id,
                sequence_number,
                transition.state,
                transition.occurred_wall_ns,
                transition.source_payload_sha256,
                transition.order_id,
                value["executed_quantity"],
                value["average_fill_price"],
                transition.reason,
                current.event_sha256,
            ],
        )
        return self.current(current.intent.client_order_id)

    def current_snapshots(
        self,
    ) -> tuple[Round74ExecutionCalibrationSnapshot, ...]:
        client_order_ids = self.connection.execute(
            """
            SELECT client_order_id
            FROM round74_execution_calibration_intent
            ORDER BY calibration_run_id, round_trip_id, path
            """
        ).fetchall()
        return tuple(self.current(str(row[0])) for row in client_order_ids)

    def verify(self) -> None:
        intent_rows = self.connection.execute(
            """
            SELECT calibration_run_id, round_trip_id, path, symbol, side,
                   client_order_id, quantity, reference_quote_notional,
                   reduce_only, created_wall_ns, book_source_sha256,
                   intent_sha256
            FROM round74_execution_calibration_intent
            ORDER BY client_order_id
            """
        ).fetchall()
        for row in intent_rows:
            intent = self._intent(tuple(row[:11]))
            intent.validate()
            if str(row[11]) != intent.intent_sha256:
                raise ValueError(
                    "Round 74 execution journal intent digest differs"
                )
            previous = intent.intent_sha256
            event_rows = self.connection.execute(
                """
                SELECT event_sha256, sequence_number, state, occurred_wall_ns,
                       source_payload_sha256, order_id, executed_quantity,
                       average_fill_price, reason, previous_event_sha256
                FROM round74_execution_calibration_event
                WHERE client_order_id = ?
                ORDER BY sequence_number
                """,
                [intent.client_order_id],
            ).fetchall()
            if not event_rows:
                raise ValueError(
                    "Round 74 execution journal event chain is missing"
                )
            current_state = ""
            for index, event in enumerate(event_rows, start=1):
                if int(event[1]) != index or str(event[9]) != previous:
                    raise ValueError(
                        "Round 74 execution journal event chain differs"
                    )
                value = {
                    "schema_version": ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION,
                    "client_order_id": intent.client_order_id,
                    "sequence_number": int(event[1]),
                    "state": str(event[2]),
                    "occurred_wall_ns": int(event[3]),
                    "source_payload_sha256": str(event[4]),
                    "order_id": int(event[5]),
                    "executed_quantity": str(event[6]),
                    "average_fill_price": str(event[7]),
                    "reason": str(event[8]),
                    "previous_event_sha256": str(event[9]),
                }
                if str(event[0]) != _canonical_sha256(value):
                    raise ValueError(
                        "Round 74 execution journal event digest differs"
                    )
                if index == 1:
                    if str(event[2]) != "PREPARED":
                        raise ValueError(
                            "Round 74 execution journal initial state differs"
                        )
                elif str(event[2]) not in _ALLOWED_TRANSITIONS.get(
                    current_state,
                    frozenset(),
                ):
                    raise ValueError(
                        "Round 74 execution journal event state differs"
                    )
                current_state = str(event[2])
                previous = str(event[0])

    def blocking_round_trip_ids(self) -> tuple[str, ...]:
        """Return every pair that could still own or create exposure."""

        self.verify()
        rows = self.connection.execute(
            """
            WITH latest AS (
                SELECT
                    i.calibration_run_id, i.round_trip_id, i.path,
                    e.state,
                    row_number() OVER (
                        PARTITION BY i.client_order_id
                        ORDER BY e.sequence_number DESC
                    ) AS current_rank
                FROM round74_execution_calibration_intent AS i
                JOIN round74_execution_calibration_event AS e
                  ON e.client_order_id = i.client_order_id
            )
            SELECT calibration_run_id, round_trip_id, path, state
            FROM latest
            WHERE current_rank = 1
            ORDER BY calibration_run_id, round_trip_id, path
            """
        ).fetchall()
        pairs: dict[tuple[str, str], dict[str, str]] = {}
        for run_id, pair_id, path, state in rows:
            pairs.setdefault((str(run_id), str(pair_id)), {})[
                str(path)
            ] = str(state)
        blocking: list[str] = []
        for (run_id, pair_id), states in sorted(pairs.items()):
            entry = states.get("entry")
            exit_state = states.get("exit")
            if (
                entry in ROUND74_EXECUTION_JOURNAL_BLOCKING_STATES
                or exit_state in ROUND74_EXECUTION_JOURNAL_BLOCKING_STATES
                or entry == "FILLED"
                and exit_state != "FLAT_VERIFIED"
                or exit_state == "FILLED"
            ):
                blocking.append(f"{run_id}:{pair_id}")
        return tuple(blocking)


__all__ = [
    "ROUND74_EXECUTION_JOURNAL_BLOCKING_STATES",
    "ROUND74_EXECUTION_JOURNAL_SCHEMA_VERSION",
    "ROUND74_EXECUTION_JOURNAL_STATES",
    "Round74ExecutionCalibrationIntent",
    "Round74ExecutionCalibrationJournal",
    "Round74ExecutionCalibrationSnapshot",
    "Round74ExecutionCalibrationTransition",
]
