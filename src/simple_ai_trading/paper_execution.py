"""Venue-neutral paper order lifecycle and conservative execution simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import re
from typing import Callable, Iterable, Mapping

import duckdb

from .positions import BOT_OWNER


PAPER_JOURNAL_SCHEMA_VERSION = "paper-order-journal-v1"
PAPER_EXECUTION_SCHEMA_VERSION = "conservative-paper-execution-v1"
ORDER_STATES = frozenset(
    {
        "INTENT",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "PARTIAL",
        "FILLED",
        "CANCEL_PENDING",
        "CANCELLED",
        "EXPIRED",
        "REJECTED",
        "UNKNOWN",
        "CLOSE_PENDING",
    }
)
TERMINAL_ORDER_STATES = frozenset({"FILLED", "CANCELLED", "EXPIRED", "REJECTED"})
BLOCKING_ORDER_STATES = frozenset(
    {
        "INTENT",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "PARTIAL",
        "CANCEL_PENDING",
        "UNKNOWN",
        "CLOSE_PENDING",
    }
)
_ORDER_TYPES = frozenset({"GTC", "GTD", "FOK", "FAK"})
_SIDES = frozenset({"BUY", "SELL"})
_SOURCES = frozenset({"execution", "reconciliation", "operator", "risk", "simulator"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TRANSITIONS = {
    "INTENT": frozenset({"SUBMITTED", "REJECTED", "CANCELLED"}),
    "SUBMITTED": frozenset(
        {
            "ACKNOWLEDGED",
            "PARTIAL",
            "FILLED",
            "CANCEL_PENDING",
            "CANCELLED",
            "EXPIRED",
            "REJECTED",
            "UNKNOWN",
            "CLOSE_PENDING",
        }
    ),
    "ACKNOWLEDGED": frozenset(
        {
            "PARTIAL",
            "FILLED",
            "CANCEL_PENDING",
            "CANCELLED",
            "EXPIRED",
            "UNKNOWN",
            "CLOSE_PENDING",
        }
    ),
    "PARTIAL": frozenset(
        {
            "PARTIAL",
            "FILLED",
            "CANCEL_PENDING",
            "CANCELLED",
            "EXPIRED",
            "UNKNOWN",
            "CLOSE_PENDING",
        }
    ),
    "CANCEL_PENDING": frozenset(
        {"PARTIAL", "FILLED", "CANCELLED", "UNKNOWN", "CLOSE_PENDING"}
    ),
    "UNKNOWN": frozenset(
        {
            "ACKNOWLEDGED",
            "PARTIAL",
            "FILLED",
            "CANCEL_PENDING",
            "CANCELLED",
            "EXPIRED",
            "REJECTED",
            "UNKNOWN",
            "CLOSE_PENDING",
        }
    ),
    "CLOSE_PENDING": frozenset({"PARTIAL", "FILLED", "UNKNOWN", "CLOSE_PENDING"}),
}


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


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        qualifier = "positive " if positive else "finite "
        raise ValueError(f"{name} must be a {qualifier}decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _identifier(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{name} is invalid")
    return text


@dataclass(frozen=True)
class PaperOrderIntent:
    """Immutable venue-neutral order intent owned by this bot."""

    intent_id: str
    venue: str
    market_id: str
    asset_id: str
    symbol: str
    outcome: str
    side: str
    order_type: str
    limit_price: Decimal
    quantity: Decimal
    created_at_ms: int
    expires_at_ms: int
    owner: str = BOT_OWNER
    parent_inventory_id: str = ""

    def validated(self) -> "PaperOrderIntent":
        intent_id = _identifier(self.intent_id, name="intent_id")
        venue = _identifier(self.venue.lower(), name="venue")
        market_id = _identifier(self.market_id, name="market_id")
        asset_id = _identifier(self.asset_id, name="asset_id")
        symbol = _identifier(self.symbol.upper(), name="symbol")
        outcome = str(self.outcome or "").strip()
        if len(outcome) > 64 or any(ord(character) < 0x20 for character in outcome):
            raise ValueError("outcome is invalid")
        side = str(self.side or "").strip().upper()
        order_type = str(self.order_type or "").strip().upper()
        if side not in _SIDES:
            raise ValueError("side must be BUY or SELL")
        if order_type not in _ORDER_TYPES:
            raise ValueError("order_type must be GTC, GTD, FOK, or FAK")
        limit_price = _decimal(self.limit_price, name="limit_price", positive=True)
        quantity = _decimal(self.quantity, name="quantity", positive=True)
        created_at_ms = int(self.created_at_ms)
        expires_at_ms = int(self.expires_at_ms)
        if created_at_ms < 0 or expires_at_ms <= created_at_ms:
            raise ValueError("order intent timestamps are invalid")
        if self.owner != BOT_OWNER:
            raise ValueError("paper intents must use the shared bot owner")
        parent = str(self.parent_inventory_id or "").strip()
        if parent:
            parent = _identifier(parent, name="parent_inventory_id")
        return replace(
            self,
            intent_id=intent_id,
            venue=venue,
            market_id=market_id,
            asset_id=asset_id,
            symbol=symbol,
            outcome=outcome,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            quantity=quantity,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
            parent_inventory_id=parent,
        )

    def payload(self) -> dict[str, object]:
        item = self.validated()
        return {
            "schema_version": PAPER_JOURNAL_SCHEMA_VERSION,
            "intent_id": item.intent_id,
            "venue": item.venue,
            "market_id": item.market_id,
            "asset_id": item.asset_id,
            "symbol": item.symbol,
            "outcome": item.outcome,
            "side": item.side,
            "order_type": item.order_type,
            "limit_price": _decimal_text(item.limit_price),
            "quantity": _decimal_text(item.quantity),
            "created_at_ms": item.created_at_ms,
            "expires_at_ms": item.expires_at_ms,
            "owner": item.owner,
            "parent_inventory_id": item.parent_inventory_id,
        }


@dataclass(frozen=True)
class PaperOrderTransition:
    """One immutable state transition for an existing paper order intent."""

    event_id: str
    state: str
    occurred_at_ms: int
    cumulative_filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal = Decimal("0")
    cumulative_fee_quote: Decimal = Decimal("0")
    reason: str = ""
    source: str = "simulator"
    source_event_id: str = ""
    source_payload_sha256: str = ""

    def validated(self) -> "PaperOrderTransition":
        event_id = _identifier(self.event_id, name="event_id")
        state = str(self.state or "").strip().upper()
        if state not in ORDER_STATES - {"INTENT"}:
            raise ValueError(f"unsupported paper order state: {state}")
        occurred_at_ms = int(self.occurred_at_ms)
        if occurred_at_ms < 0:
            raise ValueError("occurred_at_ms must be non-negative")
        filled = _decimal(
            self.cumulative_filled_quantity,
            name="cumulative_filled_quantity",
        )
        average = _decimal(self.average_fill_price, name="average_fill_price")
        fee = _decimal(self.cumulative_fee_quote, name="cumulative_fee_quote")
        if filled < 0 or average < 0 or fee < 0:
            raise ValueError("fill and fee values must be non-negative")
        if (filled == 0) != (average == 0):
            raise ValueError(
                "average fill price must be zero exactly when filled quantity is zero"
            )
        reason = str(self.reason or "").strip()
        if len(reason) > 500 or any(ord(character) < 0x20 for character in reason):
            raise ValueError("transition reason is invalid")
        source = str(self.source or "").strip().lower()
        if source not in _SOURCES:
            raise ValueError(f"unsupported transition source: {source}")
        source_event_id = str(self.source_event_id or "").strip()
        if source_event_id:
            source_event_id = _identifier(source_event_id, name="source_event_id")
        source_sha = str(self.source_payload_sha256 or "").strip().lower()
        if source_sha and not _SHA256.fullmatch(source_sha):
            raise ValueError("source_payload_sha256 is invalid")
        return replace(
            self,
            event_id=event_id,
            state=state,
            occurred_at_ms=occurred_at_ms,
            cumulative_filled_quantity=filled,
            average_fill_price=average,
            cumulative_fee_quote=fee,
            reason=reason,
            source=source,
            source_event_id=source_event_id,
            source_payload_sha256=source_sha,
        )


@dataclass(frozen=True)
class PaperOrderSnapshot:
    intent_id: str
    state: str
    sequence_number: int
    occurred_at_ms: int
    cumulative_filled_quantity: Decimal
    average_fill_price: Decimal
    cumulative_fee_quote: Decimal
    event_sha256: str

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_ORDER_STATES

    @property
    def blocks_new_exposure(self) -> bool:
        return self.state in BLOCKING_ORDER_STATES


@dataclass(frozen=True)
class PaperInventory:
    """Bot-owned paper inventory derived only from immutable order events."""

    opening_intent_id: str
    venue: str
    market_id: str
    asset_id: str
    symbol: str
    outcome: str
    opening_side: str
    opened_quantity: Decimal
    closed_quantity: Decimal
    remaining_quantity: Decimal


@dataclass(frozen=True)
class PaperReconciliationReport:
    venue: str
    inventory: tuple[PaperInventory, ...]
    blocking_intent_ids: tuple[str, ...]
    integrity_errors: tuple[str, ...]
    ownership_errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.integrity_errors and not self.ownership_errors

    @property
    def can_open(self) -> bool:
        return self.ok and not self.blocking_intent_ids

    @property
    def can_close(self) -> bool:
        return self.ok


class PaperOrderJournal:
    """Hash-chained append-only intent journal usable by every venue adapter."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_order_intent (
                intent_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                venue VARCHAR NOT NULL,
                market_id VARCHAR NOT NULL,
                asset_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                outcome VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                order_type VARCHAR NOT NULL,
                limit_price VARCHAR NOT NULL,
                quantity VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL,
                expires_at_ms BIGINT NOT NULL,
                owner VARCHAR NOT NULL,
                parent_inventory_id VARCHAR NOT NULL,
                intent_sha256 VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_order_event (
                event_id VARCHAR PRIMARY KEY,
                intent_id VARCHAR NOT NULL,
                sequence_number UINTEGER NOT NULL,
                state VARCHAR NOT NULL,
                occurred_at_ms BIGINT NOT NULL,
                cumulative_filled_quantity VARCHAR NOT NULL,
                average_fill_price VARCHAR NOT NULL,
                cumulative_fee_quote VARCHAR NOT NULL,
                reason VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                source_event_id VARCHAR NOT NULL,
                source_payload_sha256 VARCHAR NOT NULL,
                previous_event_sha256 VARCHAR NOT NULL,
                event_sha256 VARCHAR NOT NULL,
                UNIQUE(intent_id, sequence_number)
            );
            """
        )

    @staticmethod
    def _initial_event(
        intent: PaperOrderIntent, intent_sha256: str
    ) -> dict[str, object]:
        payload = {
            "schema_version": PAPER_JOURNAL_SCHEMA_VERSION,
            "event_id": _canonical_sha256(
                {"intent_id": intent.intent_id, "intent_sha256": intent_sha256}
            ),
            "intent_id": intent.intent_id,
            "sequence_number": 1,
            "state": "INTENT",
            "occurred_at_ms": intent.created_at_ms,
            "cumulative_filled_quantity": "0",
            "average_fill_price": "0",
            "cumulative_fee_quote": "0",
            "reason": "",
            "source": "operator",
            "source_event_id": "",
            "source_payload_sha256": "",
            "previous_event_sha256": intent_sha256,
        }
        payload["event_sha256"] = _canonical_sha256(payload)
        return payload

    @staticmethod
    def _transition_payload(
        intent_id: str,
        sequence_number: int,
        previous_event_sha256: str,
        transition: PaperOrderTransition,
    ) -> dict[str, object]:
        payload = {
            "schema_version": PAPER_JOURNAL_SCHEMA_VERSION,
            "event_id": transition.event_id,
            "intent_id": intent_id,
            "sequence_number": int(sequence_number),
            "state": transition.state,
            "occurred_at_ms": transition.occurred_at_ms,
            "cumulative_filled_quantity": _decimal_text(
                transition.cumulative_filled_quantity
            ),
            "average_fill_price": _decimal_text(transition.average_fill_price),
            "cumulative_fee_quote": _decimal_text(transition.cumulative_fee_quote),
            "reason": transition.reason,
            "source": transition.source,
            "source_event_id": transition.source_event_id,
            "source_payload_sha256": transition.source_payload_sha256,
            "previous_event_sha256": previous_event_sha256,
        }
        payload["event_sha256"] = _canonical_sha256(payload)
        return payload

    def record_intent(self, intent: PaperOrderIntent) -> PaperOrderSnapshot:
        item = intent.validated()
        intent_payload = item.payload()
        intent_sha = _canonical_sha256(intent_payload)
        existing = self.connection.execute(
            "SELECT intent_sha256 FROM paper_order_intent WHERE intent_id = ?",
            [item.intent_id],
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != intent_sha:
                raise ValueError(
                    "intent_id already exists with a different immutable payload"
                )
            return self.current(item.intent_id)
        initial = self._initial_event(item, intent_sha)
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                """
                INSERT INTO paper_order_intent VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    item.intent_id,
                    PAPER_JOURNAL_SCHEMA_VERSION,
                    item.venue,
                    item.market_id,
                    item.asset_id,
                    item.symbol,
                    item.outcome,
                    item.side,
                    item.order_type,
                    _decimal_text(item.limit_price),
                    _decimal_text(item.quantity),
                    item.created_at_ms,
                    item.expires_at_ms,
                    item.owner,
                    item.parent_inventory_id,
                    intent_sha,
                ],
            )
            self._insert_event(initial)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return self.current(item.intent_id)

    def transition(
        self,
        intent_id: str,
        transition: PaperOrderTransition,
    ) -> PaperOrderSnapshot:
        normalized_id = _identifier(intent_id, name="intent_id")
        item = transition.validated()
        intent_row = self.connection.execute(
            "SELECT quantity FROM paper_order_intent WHERE intent_id = ?",
            [normalized_id],
        ).fetchone()
        if intent_row is None:
            raise KeyError(f"unknown paper intent: {normalized_id}")
        existing = self.connection.execute(
            """
            SELECT intent_id, state, occurred_at_ms, cumulative_filled_quantity,
                   average_fill_price, cumulative_fee_quote, reason, source,
                   source_event_id, source_payload_sha256
            FROM paper_order_event WHERE event_id = ?
            """,
            [item.event_id],
        ).fetchone()
        if existing is not None:
            semantic = (
                normalized_id,
                item.state,
                item.occurred_at_ms,
                _decimal_text(item.cumulative_filled_quantity),
                _decimal_text(item.average_fill_price),
                _decimal_text(item.cumulative_fee_quote),
                item.reason,
                item.source,
                item.source_event_id,
                item.source_payload_sha256,
            )
            if tuple(
                str(value) if index not in {2} else int(value)
                for index, value in enumerate(existing)
            ) != tuple(
                str(value) if index not in {2} else int(value)
                for index, value in enumerate(semantic)
            ):
                raise ValueError(
                    "event_id already exists with a different immutable payload"
                )
            return self._snapshot_for_event(item.event_id)
        latest = self.current(normalized_id)
        payload = self._transition_payload(
            normalized_id,
            latest.sequence_number + 1,
            latest.event_sha256,
            item,
        )
        self._validate_transition(
            previous=latest,
            transition=item,
            order_quantity=_decimal(intent_row[0], name="quantity", positive=True),
        )
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self._insert_event(payload)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return self._snapshot_for_event(item.event_id)

    @staticmethod
    def _validate_transition(
        *,
        previous: PaperOrderSnapshot,
        transition: PaperOrderTransition,
        order_quantity: Decimal,
    ) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(previous.state, frozenset())
        if transition.state not in allowed:
            raise ValueError(
                f"invalid paper order transition {previous.state}->{transition.state}"
            )
        if previous.state == "UNKNOWN" and transition.state != "UNKNOWN":
            if transition.source != "reconciliation":
                raise ValueError("only reconciliation may resolve an UNKNOWN order")
        if previous.state == "CLOSE_PENDING" and transition.state in {
            "PARTIAL",
            "FILLED",
        }:
            if transition.source != "reconciliation":
                raise ValueError(
                    "only reconciliation may apply a late CLOSE_PENDING fill"
                )
        if transition.occurred_at_ms < previous.occurred_at_ms:
            raise ValueError("paper order transition time regressed")
        filled = transition.cumulative_filled_quantity
        tolerance = max(Decimal("0.000000000001"), order_quantity * Decimal("1e-12"))
        if filled + tolerance < previous.cumulative_filled_quantity:
            raise ValueError("cumulative filled quantity regressed")
        if filled > order_quantity + tolerance:
            raise ValueError("cumulative filled quantity exceeds order quantity")
        if transition.cumulative_fee_quote + tolerance < previous.cumulative_fee_quote:
            raise ValueError("cumulative fee regressed")
        if abs(filled - previous.cumulative_filled_quantity) <= tolerance:
            if transition.average_fill_price != previous.average_fill_price:
                raise ValueError("average fill price changed without a new fill")
            if transition.cumulative_fee_quote != previous.cumulative_fee_quote:
                raise ValueError("cumulative fee changed without a new fill")
        if transition.state == "PARTIAL" and not (
            Decimal("0") < filled < order_quantity
        ):
            raise ValueError("PARTIAL requires a nonzero incomplete fill")
        if transition.state == "FILLED" and abs(filled - order_quantity) > tolerance:
            raise ValueError("FILLED requires the complete order quantity")
        if transition.state == "CLOSE_PENDING" and filled >= order_quantity - tolerance:
            raise ValueError("CLOSE_PENDING requires an unfilled remainder")

    def _insert_event(self, payload: dict[str, object]) -> None:
        self.connection.execute(
            """
            INSERT INTO paper_order_event VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                payload["event_id"],
                payload["intent_id"],
                payload["sequence_number"],
                payload["state"],
                payload["occurred_at_ms"],
                payload["cumulative_filled_quantity"],
                payload["average_fill_price"],
                payload["cumulative_fee_quote"],
                payload["reason"],
                payload["source"],
                payload["source_event_id"],
                payload["source_payload_sha256"],
                payload["previous_event_sha256"],
                payload["event_sha256"],
            ],
        )

    def current(self, intent_id: str) -> PaperOrderSnapshot:
        normalized = _identifier(intent_id, name="intent_id")
        row = self.connection.execute(
            """
            SELECT event_id, state, sequence_number, occurred_at_ms,
                   cumulative_filled_quantity, average_fill_price,
                   cumulative_fee_quote, event_sha256
            FROM paper_order_event
            WHERE intent_id = ?
            ORDER BY sequence_number DESC
            LIMIT 1
            """,
            [normalized],
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown paper intent: {normalized}")
        return self._snapshot(row, normalized)

    def _snapshot_for_event(self, event_id: str) -> PaperOrderSnapshot:
        row = self.connection.execute(
            """
            SELECT event_id, state, sequence_number, occurred_at_ms,
                   cumulative_filled_quantity, average_fill_price,
                   cumulative_fee_quote, event_sha256, intent_id
            FROM paper_order_event WHERE event_id = ?
            """,
            [event_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown paper event: {event_id}")
        return self._snapshot(row[:8], str(row[8]))

    @staticmethod
    def _snapshot(row: tuple[object, ...], intent_id: str) -> PaperOrderSnapshot:
        return PaperOrderSnapshot(
            intent_id=intent_id,
            state=str(row[1]),
            sequence_number=int(row[2]),
            occurred_at_ms=int(row[3]),
            cumulative_filled_quantity=Decimal(str(row[4])),
            average_fill_price=Decimal(str(row[5])),
            cumulative_fee_quote=Decimal(str(row[6])),
            event_sha256=str(row[7]),
        )

    def integrity_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        intents = self.connection.execute(
            """
            SELECT intent_id, schema_version, venue, market_id, asset_id, symbol,
                   outcome, side, order_type, limit_price, quantity, created_at_ms,
                   expires_at_ms, owner, parent_inventory_id, intent_sha256
            FROM paper_order_intent ORDER BY intent_id
            """
        ).fetchall()
        for row in intents:
            payload = {
                "schema_version": str(row[1]),
                "intent_id": str(row[0]),
                "venue": str(row[2]),
                "market_id": str(row[3]),
                "asset_id": str(row[4]),
                "symbol": str(row[5]),
                "outcome": str(row[6]),
                "side": str(row[7]),
                "order_type": str(row[8]),
                "limit_price": str(row[9]),
                "quantity": str(row[10]),
                "created_at_ms": int(row[11]),
                "expires_at_ms": int(row[12]),
                "owner": str(row[13]),
                "parent_inventory_id": str(row[14]),
            }
            intent_sha = _canonical_sha256(payload)
            if intent_sha != str(row[15]):
                errors.append(f"intent_hash_mismatch:{row[0]}")
            events = self.connection.execute(
                """
                SELECT event_id, intent_id, sequence_number, state, occurred_at_ms,
                       cumulative_filled_quantity, average_fill_price,
                       cumulative_fee_quote, reason, source, source_event_id,
                       source_payload_sha256, previous_event_sha256, event_sha256
                FROM paper_order_event WHERE intent_id = ?
                ORDER BY sequence_number
                """,
                [row[0]],
            ).fetchall()
            if not events:
                errors.append(f"intent_has_no_events:{row[0]}")
            previous_sha = intent_sha
            for expected_sequence, event in enumerate(events, start=1):
                event_payload = {
                    "schema_version": PAPER_JOURNAL_SCHEMA_VERSION,
                    "event_id": str(event[0]),
                    "intent_id": str(event[1]),
                    "sequence_number": int(event[2]),
                    "state": str(event[3]),
                    "occurred_at_ms": int(event[4]),
                    "cumulative_filled_quantity": str(event[5]),
                    "average_fill_price": str(event[6]),
                    "cumulative_fee_quote": str(event[7]),
                    "reason": str(event[8]),
                    "source": str(event[9]),
                    "source_event_id": str(event[10]),
                    "source_payload_sha256": str(event[11]),
                    "previous_event_sha256": str(event[12]),
                }
                if int(event[2]) != expected_sequence:
                    errors.append(f"event_sequence_gap:{row[0]}:{event[2]}")
                if str(event[12]) != previous_sha:
                    errors.append(f"event_chain_mismatch:{row[0]}:{event[2]}")
                actual = _canonical_sha256(event_payload)
                if actual != str(event[13]):
                    errors.append(f"event_hash_mismatch:{row[0]}:{event[2]}")
                previous_sha = str(event[13])
        return tuple(errors)

    def intent(self, intent_id: str) -> PaperOrderIntent:
        """Load one immutable intent from the journal."""

        normalized = _identifier(intent_id, name="intent_id")
        row = self.connection.execute(
            """
            SELECT intent_id, venue, market_id, asset_id, symbol, outcome, side,
                   order_type, limit_price, quantity, created_at_ms, expires_at_ms,
                   owner, parent_inventory_id
            FROM paper_order_intent WHERE intent_id = ?
            """,
            [normalized],
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown paper intent: {normalized}")
        return PaperOrderIntent(
            intent_id=str(row[0]),
            venue=str(row[1]),
            market_id=str(row[2]),
            asset_id=str(row[3]),
            symbol=str(row[4]),
            outcome=str(row[5]),
            side=str(row[6]),
            order_type=str(row[7]),
            limit_price=Decimal(str(row[8])),
            quantity=Decimal(str(row[9])),
            created_at_ms=int(row[10]),
            expires_at_ms=int(row[11]),
            owner=str(row[12]),
            parent_inventory_id=str(row[13]),
        ).validated()

    def _current_rows(self, venue: str | None = None) -> list[tuple[object, ...]]:
        normalized_venue = None
        if venue is not None:
            normalized_venue = _identifier(str(venue).lower(), name="venue")
        return self.connection.execute(
            """
            WITH ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY intent_id ORDER BY sequence_number DESC
                ) AS current_rank
                FROM paper_order_event
            )
            SELECT i.intent_id, i.venue, i.market_id, i.asset_id, i.symbol,
                   i.outcome, i.side, i.order_type, i.limit_price, i.quantity,
                   i.created_at_ms, i.expires_at_ms, i.owner,
                   i.parent_inventory_id, e.state, e.occurred_at_ms,
                   e.cumulative_filled_quantity, e.average_fill_price,
                   e.cumulative_fee_quote, e.event_sha256
            FROM paper_order_intent AS i
            JOIN ranked AS e ON e.intent_id = i.intent_id AND e.current_rank = 1
            WHERE (? IS NULL OR i.venue = ?)
            ORDER BY i.intent_id
            """,
            [normalized_venue, normalized_venue],
        ).fetchall()

    def reconcile(self, venue: str | None = None) -> PaperReconciliationReport:
        """Derive bot-owned inventory and unresolved orders from the hash chain."""

        rows = self._current_rows(venue)
        by_id = {str(row[0]): row for row in rows}
        ownership_errors: list[str] = []
        closed_by_parent: dict[str, Decimal] = {}
        for row in rows:
            child_id = str(row[0])
            parent_id = str(row[13])
            if not parent_id:
                continue
            parent = by_id.get(parent_id)
            if parent is None:
                ownership_errors.append(f"close_parent_missing:{child_id}:{parent_id}")
                continue
            if str(parent[13]):
                ownership_errors.append(
                    f"close_parent_is_not_opening_intent:{child_id}"
                )
                continue
            if any(str(row[index]) != str(parent[index]) for index in (1, 2, 3, 4)):
                ownership_errors.append(f"close_parent_identity_mismatch:{child_id}")
                continue
            if str(row[6]) == str(parent[6]):
                ownership_errors.append(
                    f"close_side_does_not_reverse_parent:{child_id}"
                )
                continue
            closed_by_parent[parent_id] = closed_by_parent.get(
                parent_id, Decimal("0")
            ) + Decimal(str(row[16]))

        inventory: list[PaperInventory] = []
        for row in rows:
            opening_id = str(row[0])
            if str(row[13]):
                continue
            opened = Decimal(str(row[16]))
            closed = closed_by_parent.get(opening_id, Decimal("0"))
            tolerance = max(Decimal("0.000000000001"), opened * Decimal("1e-12"))
            if closed > opened + tolerance:
                ownership_errors.append(f"inventory_overclosed:{opening_id}")
            if opened <= 0:
                continue
            inventory.append(
                PaperInventory(
                    opening_intent_id=opening_id,
                    venue=str(row[1]),
                    market_id=str(row[2]),
                    asset_id=str(row[3]),
                    symbol=str(row[4]),
                    outcome=str(row[5]),
                    opening_side=str(row[6]),
                    opened_quantity=opened,
                    closed_quantity=min(opened, closed),
                    remaining_quantity=max(Decimal("0"), opened - closed),
                )
            )
        remaining_by_opening = {
            item.opening_intent_id: item.remaining_quantity for item in inventory
        }
        blocking_items: list[str] = []
        for row in rows:
            if str(row[14]) not in BLOCKING_ORDER_STATES:
                continue
            parent_id = str(row[13])
            if (
                str(row[14]) == "CLOSE_PENDING"
                and parent_id
                and remaining_by_opening.get(parent_id, Decimal("0")) <= 0
            ):
                continue
            blocking_items.append(str(row[0]))
        blocking = tuple(blocking_items)
        report_venue = "*" if venue is None else str(venue).lower()
        return PaperReconciliationReport(
            venue=report_venue,
            inventory=tuple(inventory),
            blocking_intent_ids=blocking,
            integrity_errors=self.integrity_errors(),
            ownership_errors=tuple(sorted(set(ownership_errors))),
        )

    def owned_quantity(self, opening_intent_id: str) -> Decimal:
        """Return remaining bot-owned quantity or fail on inconsistent ownership."""

        normalized = _identifier(opening_intent_id, name="opening_intent_id")
        report = self.reconcile()
        if not report.ok:
            detail = "; ".join((*report.integrity_errors, *report.ownership_errors))
            raise ValueError(f"paper ownership reconciliation failed: {detail}")
        for inventory in report.inventory:
            if inventory.opening_intent_id == normalized:
                return inventory.remaining_quantity
        return Decimal("0")


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    quantity: Decimal

    def validated(self) -> "BookLevel":
        return BookLevel(
            price=_decimal(self.price, name="book price", positive=True),
            quantity=_decimal(self.quantity, name="book quantity", positive=True),
        )


@dataclass(frozen=True)
class PaperBookSnapshot:
    venue: str
    market_id: str
    asset_id: str
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    source_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    source_payload_sha256: str
    connected: bool = True
    gap_free: bool = True

    def validated(self) -> "PaperBookSnapshot":
        venue = _identifier(self.venue.lower(), name="venue")
        market_id = _identifier(self.market_id, name="market_id")
        asset_id = _identifier(self.asset_id, name="asset_id")
        bids = tuple(level.validated() for level in self.bids)
        asks = tuple(level.validated() for level in self.asks)
        if tuple(sorted(bids, key=lambda level: level.price, reverse=True)) != bids:
            raise ValueError("bids must be sorted from best to worst")
        if tuple(sorted(asks, key=lambda level: level.price)) != asks:
            raise ValueError("asks must be sorted from best to worst")
        if len({level.price for level in bids}) != len(bids):
            raise ValueError("bid prices must be unique")
        if len({level.price for level in asks}) != len(asks):
            raise ValueError("ask prices must be unique")
        if bids and asks and bids[0].price >= asks[0].price:
            raise ValueError("book is crossed or locked")
        source_sha = str(self.source_payload_sha256 or "").strip().lower()
        if not _SHA256.fullmatch(source_sha):
            raise ValueError("book source_payload_sha256 is invalid")
        source_time_ms = int(self.source_time_ms)
        received_wall_ms = int(self.received_wall_ms)
        monotonic_ns = int(self.received_monotonic_ns)
        if min(source_time_ms, received_wall_ms, monotonic_ns) < 0:
            raise ValueError("book timestamps must be non-negative")
        return replace(
            self,
            venue=venue,
            market_id=market_id,
            asset_id=asset_id,
            bids=bids,
            asks=asks,
            source_time_ms=source_time_ms,
            received_wall_ms=received_wall_ms,
            received_monotonic_ns=monotonic_ns,
            source_payload_sha256=source_sha,
        )


@dataclass(frozen=True)
class PaperFill:
    price: Decimal
    quantity: Decimal
    fee_quote: Decimal
    liquidity_role: str


@dataclass(frozen=True)
class PaperExecutionResult:
    state: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_fill_price: Decimal
    fee_quote: Decimal
    fills: tuple[PaperFill, ...]
    reason: str
    source_payload_sha256: str


FeeFunction = Callable[[Decimal, Decimal, str], Decimal]


@dataclass(frozen=True)
class BinanceBpsFeeModel:
    """Binance fee adapter for the shared execution simulator."""

    maker_fee_bps: Decimal
    taker_fee_bps: Decimal

    def __call__(self, price: Decimal, quantity: Decimal, role: str) -> Decimal:
        maker = _decimal(self.maker_fee_bps, name="maker_fee_bps")
        taker = _decimal(self.taker_fee_bps, name="taker_fee_bps")
        if maker < 0 or taker < 0:
            raise ValueError("Binance fee rates must be non-negative")
        rate = maker if role == "maker" else taker
        return price * quantity * rate / Decimal("10000")


@dataclass(frozen=True)
class PolymarketFeeModel:
    """Current documented Polymarket fee curve with conservative precision."""

    enabled: bool
    rate: Decimal
    exponent: int
    taker_only: bool

    def __call__(self, price: Decimal, quantity: Decimal, role: str) -> Decimal:
        if not self.enabled or (role == "maker" and self.taker_only):
            return Decimal("0")
        rate = _decimal(self.rate, name="Polymarket fee rate")
        if rate < 0 or rate > 1:
            raise ValueError("Polymarket fee rate is outside [0, 1]")
        if int(self.exponent) != 1:
            raise ValueError(
                "unsupported Polymarket fee exponent; no documented simulator formula"
            )
        if price <= 0 or price >= 1:
            raise ValueError("Polymarket match price must lie strictly between 0 and 1")
        raw = quantity * rate * price * (Decimal("1") - price)
        if raw < Decimal("0.00001"):
            return Decimal("0")
        return raw.quantize(Decimal("0.00001"), rounding=ROUND_CEILING)


def simulate_aggressive_order(
    intent: PaperOrderIntent,
    book: PaperBookSnapshot,
    *,
    execution_time_ms: int,
    submission_latency_ms: int,
    maximum_book_age_ms: int,
    fee: FeeFunction,
    owned_quantity: Decimal | None = None,
    closing_position: bool = False,
) -> PaperExecutionResult:
    """Walk one observed book without midpoint, hidden, or future fill credit."""

    order = intent.validated()
    snapshot = book.validated()
    execution_time = int(execution_time_ms)
    latency = int(submission_latency_ms)
    max_age = int(maximum_book_age_ms)
    if order.order_type not in {"FOK", "FAK"}:
        raise ValueError("aggressive simulation requires FOK or FAK")
    if snapshot.venue != order.venue or snapshot.market_id != order.market_id:
        raise ValueError("book venue or market does not match the order intent")
    if snapshot.asset_id != order.asset_id:
        raise ValueError("book asset does not match the order intent")
    if latency <= 0:
        return _empty_execution(
            order, "REJECTED", "zero_or_negative_latency_prohibited", snapshot
        )
    if execution_time < order.created_at_ms + latency:
        return _empty_execution(
            order, "REJECTED", "submission_latency_not_elapsed", snapshot
        )
    if execution_time >= order.expires_at_ms:
        return _empty_execution(
            order, "EXPIRED", "order_expired_before_execution", snapshot
        )
    if not snapshot.connected or not snapshot.gap_free:
        return _empty_execution(
            order, "UNKNOWN", "book_connection_or_gap_unproven", snapshot
        )
    if snapshot.received_wall_ms > execution_time:
        return _empty_execution(
            order, "REJECTED", "future_book_snapshot_prohibited", snapshot
        )
    if max_age < 0 or execution_time - snapshot.received_wall_ms > max_age:
        return _empty_execution(order, "EXPIRED", "stale_book_snapshot", snapshot)
    if owned_quantity is not None:
        owned = _decimal(owned_quantity, name="owned_quantity")
        if owned < order.quantity:
            return _empty_execution(
                order, "REJECTED", "order_exceeds_bot_owned_inventory", snapshot
            )

    levels = snapshot.asks if order.side == "BUY" else snapshot.bids
    eligible = tuple(
        level
        for level in levels
        if (
            level.price <= order.limit_price
            if order.side == "BUY"
            else level.price >= order.limit_price
        )
    )
    available = sum((level.quantity for level in eligible), start=Decimal("0"))
    if order.order_type == "FOK" and available < order.quantity:
        state = "CLOSE_PENDING" if closing_position else "CANCELLED"
        return _empty_execution(
            order, state, "insufficient_displayed_depth_for_fok", snapshot
        )

    remaining = order.quantity
    fills: list[PaperFill] = []
    for level in eligible:
        if remaining <= 0:
            break
        fill_quantity = min(remaining, level.quantity)
        fill_fee = _decimal(
            fee(level.price, fill_quantity, "taker"),
            name="calculated fee",
        )
        if fill_fee < 0:
            raise ValueError("fee model returned a negative fee")
        fills.append(PaperFill(level.price, fill_quantity, fill_fee, "taker"))
        remaining -= fill_quantity

    filled = order.quantity - remaining
    notional = sum((fill.price * fill.quantity for fill in fills), start=Decimal("0"))
    average = notional / filled if filled > 0 else Decimal("0")
    total_fee = sum((fill.fee_quote for fill in fills), start=Decimal("0"))
    if remaining <= 0:
        state = "FILLED"
        reason = "displayed_depth_walk_complete"
    elif closing_position:
        state = "CLOSE_PENDING"
        reason = "unfilled_bot_owned_close_remainder"
    else:
        state = "CANCELLED"
        reason = "fak_unfilled_remainder_cancelled"
    return PaperExecutionResult(
        state=state,
        filled_quantity=filled,
        remaining_quantity=max(Decimal("0"), remaining),
        average_fill_price=average,
        fee_quote=total_fee,
        fills=tuple(fills),
        reason=reason,
        source_payload_sha256=snapshot.source_payload_sha256,
    )


def _empty_execution(
    intent: PaperOrderIntent,
    state: str,
    reason: str,
    book: PaperBookSnapshot,
) -> PaperExecutionResult:
    return PaperExecutionResult(
        state=state,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        average_fill_price=Decimal("0"),
        fee_quote=Decimal("0"),
        fills=(),
        reason=reason,
        source_payload_sha256=book.source_payload_sha256,
    )


def paper_intent_id(
    venue: str,
    inventory_id: str,
    action: str,
    *,
    attempt: int = 1,
) -> str:
    """Return one deterministic cross-venue paper intent identifier."""

    normalized_venue = _identifier(str(venue).lower(), name="venue")
    normalized_inventory = _identifier(inventory_id, name="inventory_id")
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"open", "close"}:
        raise ValueError("paper intent action must be open or close")
    attempt_number = int(attempt)
    if attempt_number < 1:
        raise ValueError("paper intent attempt must be positive")
    digest = _canonical_sha256(
        {
            "venue": normalized_venue,
            "inventory_id": normalized_inventory,
            "action": normalized_action,
            "attempt": attempt_number,
        }
    )
    return f"paper-{normalized_venue}-{normalized_action}-{digest[:32]}"


def binance_book_ticker_snapshot(
    payload: Mapping[str, object],
    *,
    market_type: str,
    symbol: str,
    received_wall_ms: int,
    received_monotonic_ns: int,
) -> PaperBookSnapshot:
    """Normalize one real Binance BBO response without inventing depth or time."""

    normalized_market = str(market_type or "").strip().lower()
    if normalized_market not in {"spot", "futures"}:
        raise ValueError("Binance paper market_type must be spot or futures")
    normalized_symbol = _identifier(str(symbol).upper(), name="symbol")
    raw = dict(payload)
    payload_symbol = str(raw.get("symbol") or raw.get("s") or "").upper()
    if payload_symbol != normalized_symbol:
        raise ValueError("Binance book ticker symbol does not match the request")

    def level(price_name: str, quantity_name: str) -> BookLevel:
        return BookLevel(
            price=_decimal(raw.get(price_name), name=price_name, positive=True),
            quantity=_decimal(
                raw.get(quantity_name), name=quantity_name, positive=True
            ),
        )

    source_time_ms = 0
    for name in ("T", "E", "time"):
        value = raw.get(name)
        if value is None:
            continue
        try:
            candidate = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if candidate >= 0:
            source_time_ms = candidate
            break
    payload_json = _canonical_json(raw)
    return PaperBookSnapshot(
        venue=f"binance-{normalized_market}",
        market_id=normalized_symbol,
        asset_id=normalized_symbol,
        bids=(level("bidPrice", "bidQty"),),
        asks=(level("askPrice", "askQty"),),
        source_time_ms=source_time_ms,
        received_wall_ms=int(received_wall_ms),
        received_monotonic_ns=int(received_monotonic_ns),
        source_payload_sha256=hashlib.sha256(payload_json.encode("ascii")).hexdigest(),
    ).validated()


class ConservativePaperExecutionAdapter:
    """Single execution path shared by Binance and Polymarket paper orders."""

    execution_schema_version = PAPER_EXECUTION_SCHEMA_VERSION

    def __init__(
        self,
        journal: PaperOrderJournal,
        *,
        venue: str,
        fee: FeeFunction,
    ) -> None:
        self.journal = journal
        self.venue = _identifier(str(venue).lower(), name="venue")
        self.fee = fee

    def _validate_venue_order(
        self,
        intent: PaperOrderIntent,
        *,
        closing_position: bool,
    ) -> None:
        del intent, closing_position

    def execute_aggressive(
        self,
        intent: PaperOrderIntent,
        book: PaperBookSnapshot,
        *,
        execution_time_ms: int,
        submission_latency_ms: int,
        maximum_book_age_ms: int,
        closing_position: bool = False,
    ) -> PaperExecutionResult:
        order = intent.validated()
        snapshot = book.validated()
        if order.venue != self.venue:
            raise ValueError("paper intent venue does not match the adapter")
        if bool(order.parent_inventory_id) != bool(closing_position):
            raise ValueError(
                "closing paper intents require exactly one parent inventory"
            )
        self._validate_venue_order(order, closing_position=closing_position)
        owned_quantity = None
        if closing_position:
            parent = self.journal.intent(order.parent_inventory_id)
            if parent.parent_inventory_id:
                raise ValueError("paper close parent must be an opening intent")
            if any(
                current != expected
                for current, expected in (
                    (order.venue, parent.venue),
                    (order.market_id, parent.market_id),
                    (order.asset_id, parent.asset_id),
                    (order.symbol, parent.symbol),
                )
            ):
                raise ValueError("paper close identity does not match parent inventory")
            if order.side == parent.side:
                raise ValueError("paper close side does not reverse parent inventory")
            owned_quantity = self.journal.owned_quantity(order.parent_inventory_id)
            if owned_quantity < order.quantity:
                raise ValueError("paper close exceeds bot-owned inventory")
        initial = self.journal.record_intent(order)
        if initial.state != "INTENT":
            raise ValueError(
                f"paper intent is already active or resolved: {initial.state}"
            )
        submitted_event_id = _canonical_sha256(
            {
                "intent_id": order.intent_id,
                "stage": "submitted",
                "book_sha256": snapshot.source_payload_sha256,
            }
        )
        submitted = self.journal.transition(
            order.intent_id,
            PaperOrderTransition(
                event_id=submitted_event_id,
                state="SUBMITTED",
                occurred_at_ms=order.created_at_ms,
                source="simulator",
                source_event_id=snapshot.source_payload_sha256,
                source_payload_sha256=snapshot.source_payload_sha256,
            ),
        )
        result = simulate_aggressive_order(
            order,
            snapshot,
            execution_time_ms=execution_time_ms,
            submission_latency_ms=submission_latency_ms,
            maximum_book_age_ms=maximum_book_age_ms,
            fee=self.fee,
            owned_quantity=owned_quantity,
            closing_position=closing_position,
        )
        if (
            closing_position
            and result.remaining_quantity > 0
            and result.state != "UNKNOWN"
        ):
            result = replace(
                result,
                state="CLOSE_PENDING",
                reason=f"close_unresolved:{result.reason}",
            )
        cumulative_filled = (
            submitted.cumulative_filled_quantity + result.filled_quantity
        )
        if result.filled_quantity > 0:
            previous_notional = (
                submitted.cumulative_filled_quantity * submitted.average_fill_price
            )
            added_notional = result.filled_quantity * result.average_fill_price
            cumulative_average = (
                previous_notional + added_notional
            ) / cumulative_filled
        else:
            cumulative_average = submitted.average_fill_price
        cumulative_fee = submitted.cumulative_fee_quote + result.fee_quote
        final_event_id = _canonical_sha256(
            {
                "intent_id": order.intent_id,
                "stage": "execution",
                "execution_time_ms": int(execution_time_ms),
                "book_sha256": snapshot.source_payload_sha256,
                "state": result.state,
            }
        )
        self.journal.transition(
            order.intent_id,
            PaperOrderTransition(
                event_id=final_event_id,
                state=result.state,
                occurred_at_ms=int(execution_time_ms),
                cumulative_filled_quantity=cumulative_filled,
                average_fill_price=cumulative_average,
                cumulative_fee_quote=cumulative_fee,
                reason=result.reason,
                source="simulator",
                source_event_id=snapshot.source_payload_sha256,
                source_payload_sha256=snapshot.source_payload_sha256,
            ),
        )
        return result


class BinancePaperExecutionAdapter(ConservativePaperExecutionAdapter):
    def __init__(
        self,
        journal: PaperOrderJournal,
        *,
        market_type: str,
        maker_fee_bps: Decimal,
        taker_fee_bps: Decimal,
    ) -> None:
        normalized_market = str(market_type or "").strip().lower()
        if normalized_market not in {"spot", "futures"}:
            raise ValueError("Binance paper market_type must be spot or futures")
        self.market_type = normalized_market
        super().__init__(
            journal,
            venue=f"binance-{normalized_market}",
            fee=BinanceBpsFeeModel(maker_fee_bps, taker_fee_bps),
        )

    def _validate_venue_order(
        self,
        intent: PaperOrderIntent,
        *,
        closing_position: bool,
    ) -> None:
        if self.market_type == "spot" and not closing_position and intent.side != "BUY":
            raise ValueError("Binance spot paper execution prohibits naked short opens")


class PolymarketPaperExecutionAdapter(ConservativePaperExecutionAdapter):
    def __init__(
        self,
        journal: PaperOrderJournal,
        *,
        fee: PolymarketFeeModel,
    ) -> None:
        super().__init__(journal, venue="polymarket", fee=fee)

    def _validate_venue_order(
        self,
        intent: PaperOrderIntent,
        *,
        closing_position: bool,
    ) -> None:
        if intent.limit_price >= 1:
            raise ValueError("Polymarket paper price must be below one")
        expected_side = "SELL" if closing_position else "BUY"
        if intent.side != expected_side:
            raise ValueError("Polymarket paper execution prohibits naked token shorts")


@dataclass(frozen=True)
class PassiveQueueState:
    intent_id: str
    asset_id: str
    side: str
    price: Decimal
    queue_ahead_quantity: Decimal
    remaining_quantity: Decimal
    filled_quantity: Decimal
    activated_at_ms: int
    expires_at_ms: int

    def validated(self) -> "PassiveQueueState":
        side = str(self.side or "").upper()
        if side not in _SIDES:
            raise ValueError("passive queue side must be BUY or SELL")
        queue = _decimal(self.queue_ahead_quantity, name="queue_ahead_quantity")
        remaining = _decimal(self.remaining_quantity, name="remaining_quantity")
        filled = _decimal(self.filled_quantity, name="filled_quantity")
        if min(queue, remaining, filled) < 0 or remaining <= 0:
            raise ValueError("passive queue quantities are invalid")
        activated = int(self.activated_at_ms)
        expires = int(self.expires_at_ms)
        if activated < 0 or expires <= activated:
            raise ValueError("passive queue timestamps are invalid")
        return replace(
            self,
            intent_id=_identifier(self.intent_id, name="intent_id"),
            asset_id=_identifier(self.asset_id, name="asset_id"),
            side=side,
            price=_decimal(self.price, name="passive price", positive=True),
            queue_ahead_quantity=queue,
            remaining_quantity=remaining,
            filled_quantity=filled,
            activated_at_ms=activated,
            expires_at_ms=expires,
        )


@dataclass(frozen=True)
class AggressiveTradePrint:
    asset_id: str
    side: str
    price: Decimal
    quantity: Decimal
    occurred_at_ms: int
    source_payload_sha256: str


def apply_passive_trade_print(
    state: PassiveQueueState,
    trade: AggressiveTradePrint,
) -> tuple[PassiveQueueState, Decimal]:
    """Consume queue only with a matching post-arrival aggressive print."""

    current = state.validated()
    trade_asset = _identifier(trade.asset_id, name="trade asset_id")
    trade_side = str(trade.side or "").upper()
    price = _decimal(trade.price, name="trade price", positive=True)
    quantity = _decimal(trade.quantity, name="trade quantity", positive=True)
    occurred = int(trade.occurred_at_ms)
    source_sha = str(trade.source_payload_sha256 or "").lower()
    if not _SHA256.fullmatch(source_sha):
        raise ValueError("trade source_payload_sha256 is invalid")
    required_taker_side = "SELL" if current.side == "BUY" else "BUY"
    if (
        trade_asset != current.asset_id
        or trade_side != required_taker_side
        or price != current.price
        or occurred <= current.activated_at_ms
        or occurred > current.expires_at_ms
    ):
        return current, Decimal("0")
    queue_consumed = min(current.queue_ahead_quantity, quantity)
    queue_remaining = current.queue_ahead_quantity - queue_consumed
    residual_print = quantity - queue_consumed
    filled_now = min(current.remaining_quantity, residual_print)
    return (
        replace(
            current,
            queue_ahead_quantity=queue_remaining,
            remaining_quantity=current.remaining_quantity - filled_now,
            filled_quantity=current.filled_quantity + filled_now,
        ),
        filled_now,
    )


def assert_shared_venue_execution_contract(adapters: Iterable[object]) -> None:
    """Fail when a venue adapter does not expose the shared simulator identity."""

    for adapter in adapters:
        if (
            getattr(adapter, "execution_schema_version", None)
            != PAPER_EXECUTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "venue adapter does not use the shared paper execution contract"
            )


__all__ = [
    "AggressiveTradePrint",
    "BLOCKING_ORDER_STATES",
    "BinancePaperExecutionAdapter",
    "BinanceBpsFeeModel",
    "BookLevel",
    "ConservativePaperExecutionAdapter",
    "ORDER_STATES",
    "PAPER_EXECUTION_SCHEMA_VERSION",
    "PAPER_JOURNAL_SCHEMA_VERSION",
    "PaperBookSnapshot",
    "PaperExecutionResult",
    "PaperFill",
    "PaperInventory",
    "PaperOrderIntent",
    "PaperOrderJournal",
    "PaperOrderSnapshot",
    "PaperOrderTransition",
    "PaperReconciliationReport",
    "PassiveQueueState",
    "PolymarketFeeModel",
    "PolymarketPaperExecutionAdapter",
    "TERMINAL_ORDER_STATES",
    "apply_passive_trade_print",
    "assert_shared_venue_execution_contract",
    "binance_book_ticker_snapshot",
    "paper_intent_id",
    "simulate_aggressive_order",
]
