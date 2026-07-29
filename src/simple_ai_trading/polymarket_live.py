"""Fail-closed live Polymarket order ownership and reconciliation.

This module contains no Binance execution imports. A Polymarket venue adapter
may consume external market signals elsewhere, but orders, positions, P&L, and
recovery remain entirely owned by this subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Mapping, Protocol, Sequence


POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION = "polymarket-live-ledger-v1"
POLYMARKET_LIVE_ORDER_SCHEMA_VERSION = "polymarket-live-order-v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_ORDER_ID = re.compile(r"^0x[0-9a-f]{64}$")
_BYTES32 = re.compile(r"^0x[0-9a-f]{64}$")
_ZERO_SHA256 = "0" * 64
_POSITION_TOLERANCE = Decimal("0.000001")

_OPEN_STATES = frozenset(
    {
        "prepared",
        "submitting",
        "unknown",
        "live",
        "partial",
        "matched_pending",
        "cancel_pending",
        "cancel_unknown",
    }
)
_TERMINAL_STATES = frozenset({"rejected", "cancelled", "expired", "failed", "filled"})
_FILL_ACTIVE_STATUSES = frozenset({"MATCHED", "MINED", "CONFIRMED", "RETRYING"})
_FILL_TERMINAL_STATUSES = frozenset({"CONFIRMED", "FAILED"})


class PolymarketLiveError(RuntimeError):
    """Base error for the independent live Polymarket boundary."""


class PolymarketLiveBlocked(PolymarketLiveError):
    """Raised when a deterministic safety or compliance gate blocks an action."""


class PolymarketLiveUnknownState(PolymarketLiveError):
    """Raised when the venue may have accepted an operation but did not prove it."""


class PolymarketVenueRejected(PolymarketLiveError):
    """Raised only when the venue proves that an order was rejected."""


class PolymarketStateConflict(PolymarketLiveBlocked):
    """Raised when another reconciliation source advanced the order first."""


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


def _decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    if positive and parsed <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip()
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{name} is invalid")
    return normalized


def _condition_id(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if _CONDITION_ID.fullmatch(normalized) is None:
        raise ValueError("market_id is invalid")
    return normalized


def _token_id(value: object) -> str:
    normalized = str(value or "").strip()
    if _TOKEN_ID.fullmatch(normalized) is None:
        raise ValueError("token_id is invalid")
    return normalized


def _order_id(value: object, *, name: str = "order_id") -> str:
    normalized = str(value or "").strip().lower()
    if _ORDER_ID.fullmatch(normalized) is None:
        raise ValueError(f"{name} is invalid")
    return normalized


def polymarket_live_metadata(bot_id: str, intent_id: str) -> str:
    """Return a non-secret bytes32 strategy marker for a signed V2 order."""

    payload = {
        "schema_version": POLYMARKET_LIVE_ORDER_SCHEMA_VERSION,
        "bot_id": _identifier(bot_id, name="bot_id"),
        "intent_id": _identifier(intent_id, name="intent_id"),
    }
    return "0x" + _canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class PolymarketLiveOrderIntent:
    intent_id: str
    bot_id: str
    market_id: str
    token_id: str
    symbol: str
    outcome: str
    side: str
    order_type: str
    limit_price: Decimal
    quantity: Decimal
    fee_reserve_quote: Decimal
    created_at_ms: int
    expires_at_ms: int
    parent_intent_id: str = ""
    closing_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _identifier(self.intent_id, name="intent_id"))
        object.__setattr__(self, "bot_id", _identifier(self.bot_id, name="bot_id"))
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        symbol = str(self.symbol or "").strip().upper()
        if symbol != "BTC":
            raise ValueError("live Polymarket execution is BTC-only")
        object.__setattr__(self, "symbol", symbol)
        outcome = str(self.outcome or "").strip().title()
        if outcome not in {"Up", "Down"}:
            raise ValueError("outcome must be Up or Down")
        object.__setattr__(self, "outcome", outcome)
        side = str(self.side or "").strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        object.__setattr__(self, "side", side)
        order_type = str(self.order_type or "").strip().upper()
        if order_type not in {"FAK", "FOK", "GTD"}:
            raise ValueError("order_type must be FAK, FOK, or GTD")
        object.__setattr__(self, "order_type", order_type)
        limit_price = _decimal(self.limit_price, name="limit_price", positive=True)
        if limit_price >= 1:
            raise ValueError("limit_price must be below one")
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(
            self,
            "quantity",
            _decimal(self.quantity, name="quantity", positive=True),
        )
        object.__setattr__(
            self,
            "fee_reserve_quote",
            _decimal(
                self.fee_reserve_quote,
                name="fee_reserve_quote",
                nonnegative=True,
            ),
        )
        created_at_ms = int(self.created_at_ms)
        expires_at_ms = int(self.expires_at_ms)
        if created_at_ms <= 0 or expires_at_ms <= created_at_ms:
            raise ValueError("order lifetime is invalid")
        if expires_at_ms - created_at_ms > 300_000:
            raise ValueError("order lifetime exceeds the five-minute market horizon")
        object.__setattr__(self, "created_at_ms", created_at_ms)
        object.__setattr__(self, "expires_at_ms", expires_at_ms)
        parent = str(self.parent_intent_id or "").strip()
        if parent:
            parent = _identifier(parent, name="parent_intent_id")
        if self.closing_only:
            if side != "SELL" or not parent:
                raise ValueError("closing_only requires a SELL with parent_intent_id")
        elif side == "SELL":
            raise ValueError("SELL orders must close bot-owned inventory")
        elif parent:
            raise ValueError("parent_intent_id is reserved for closing orders")
        object.__setattr__(self, "parent_intent_id", parent)

    @property
    def metadata(self) -> str:
        return polymarket_live_metadata(self.bot_id, self.intent_id)

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_LIVE_ORDER_SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "bot_id": self.bot_id,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "symbol": self.symbol,
            "outcome": self.outcome,
            "side": self.side,
            "order_type": self.order_type,
            "limit_price": format(self.limit_price, "f"),
            "quantity": format(self.quantity, "f"),
            "fee_reserve_quote": format(self.fee_reserve_quote, "f"),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "parent_intent_id": self.parent_intent_id,
            "closing_only": self.closing_only,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PolymarketPreparedOrder:
    intent: PolymarketLiveOrderIntent
    expected_order_id: str
    metadata: str
    opaque_signed_order: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_order_id",
            _order_id(self.expected_order_id, name="expected_order_id"),
        )
        metadata = str(self.metadata or "").strip().lower()
        if _BYTES32.fullmatch(metadata) is None or metadata != self.intent.metadata:
            raise ValueError("prepared order metadata differs")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class PolymarketSubmission:
    accepted: bool
    order_id: str
    status: str
    rejection_code: str = ""

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if self.accepted:
            object.__setattr__(self, "order_id", _order_id(self.order_id))
            if status not in {"live", "matched", "delayed", "unmatched"}:
                raise ValueError("accepted submission status is invalid")
            if self.rejection_code:
                raise ValueError("accepted submission cannot have a rejection code")
        else:
            if self.order_id:
                raise ValueError("rejected submission cannot have an order_id")
            if not self.rejection_code:
                raise ValueError("rejected submission must have a rejection code")
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class PolymarketRemoteOrder:
    order_id: str
    market_id: str
    token_id: str
    side: str
    status: str
    original_quantity: Decimal
    matched_quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _order_id(self.order_id))
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        side = str(self.side or "").strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("remote order side is invalid")
        object.__setattr__(self, "side", side)
        status = str(self.status or "").strip().upper()
        if not status:
            raise ValueError("remote order status is missing")
        object.__setattr__(self, "status", status)
        original = _decimal(
            self.original_quantity,
            name="original_quantity",
            positive=True,
        )
        matched = _decimal(
            self.matched_quantity,
            name="matched_quantity",
            nonnegative=True,
        )
        if matched > original:
            raise ValueError("remote matched quantity exceeds original quantity")
        object.__setattr__(self, "original_quantity", original)
        object.__setattr__(self, "matched_quantity", matched)


@dataclass(frozen=True, slots=True)
class PolymarketRemoteFill:
    trade_id: str
    order_id: str
    market_id: str
    token_id: str
    side: str
    quantity: Decimal
    price: Decimal
    status: str
    observed_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "trade_id", _identifier(self.trade_id, name="trade_id"))
        object.__setattr__(self, "order_id", _order_id(self.order_id))
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        side = str(self.side or "").strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("fill side is invalid")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "quantity",
            _decimal(self.quantity, name="fill quantity", positive=True),
        )
        price = _decimal(self.price, name="fill price", positive=True)
        if price >= 1:
            raise ValueError("fill price must be below one")
        object.__setattr__(self, "price", price)
        status = str(self.status or "").strip().upper()
        if status not in _FILL_ACTIVE_STATUSES | {"FAILED"}:
            raise ValueError("fill status is invalid")
        object.__setattr__(self, "status", status)
        observed_at_ms = int(self.observed_at_ms)
        if observed_at_ms <= 0:
            raise ValueError("fill observation time is invalid")
        object.__setattr__(self, "observed_at_ms", observed_at_ms)


@dataclass(frozen=True, slots=True)
class PolymarketRemotePosition:
    market_id: str
    token_id: str
    quantity: Decimal
    redeemable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        object.__setattr__(
            self,
            "quantity",
            _decimal(self.quantity, name="position quantity", positive=True),
        )


@dataclass(frozen=True, slots=True)
class PolymarketVenuePreflight:
    protocol_version: int
    server_time_ms: int
    observed_at_ms: int
    geoblocked: bool
    country: str
    region: str
    closed_only: bool
    wallet_address: str
    open_orders: tuple[PolymarketRemoteOrder, ...]
    positions: tuple[PolymarketRemotePosition, ...]

    @property
    def clock_skew_ms(self) -> int:
        return abs(self.observed_at_ms - self.server_time_ms)


@dataclass(frozen=True, slots=True)
class PolymarketFundingPreflight:
    asset_type: str
    token_id: str
    available_balance: Decimal
    available_allowance: Decimal

    def __post_init__(self) -> None:
        asset_type = str(self.asset_type or "").strip().upper()
        if asset_type not in {"COLLATERAL", "CONDITIONAL"}:
            raise ValueError("funding asset type is invalid")
        object.__setattr__(self, "asset_type", asset_type)
        token_id = str(self.token_id or "").strip()
        if asset_type == "COLLATERAL":
            if token_id:
                raise ValueError("collateral funding cannot have a token ID")
        else:
            token_id = _token_id(token_id)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(
            self,
            "available_balance",
            _decimal(
                self.available_balance,
                name="available_balance",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "available_allowance",
            _decimal(
                self.available_allowance,
                name="available_allowance",
                nonnegative=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class PolymarketCancelResult:
    cancelled_order_ids: tuple[str, ...]
    failed_order_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cancelled_order_ids",
            tuple(_order_id(value) for value in self.cancelled_order_ids),
        )
        object.__setattr__(
            self,
            "failed_order_ids",
            tuple(_order_id(value) for value in self.failed_order_ids),
        )
        if set(self.cancelled_order_ids) & set(self.failed_order_ids):
            raise ValueError("cancel result sets overlap")


class PolymarketLiveVenue(Protocol):
    """Authenticated venue boundary used by the coordinator."""

    def preflight(self) -> PolymarketVenuePreflight: ...

    def prepare_order(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketPreparedOrder: ...

    def submit_order(self, prepared: PolymarketPreparedOrder) -> PolymarketSubmission: ...

    def open_orders(self) -> tuple[PolymarketRemoteOrder, ...]: ...

    def fills_for_orders(
        self,
        order_ids: Sequence[str],
        *,
        market_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteFill, ...]: ...

    def positions(self) -> tuple[PolymarketRemotePosition, ...]: ...

    def funding(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        neg_risk: bool,
    ) -> PolymarketFundingPreflight: ...

    def cancel_orders(self, order_ids: Sequence[str]) -> PolymarketCancelResult: ...


class PolymarketRuntimeAuthority(Protocol):
    """Runtime liveness gate implemented independently of strategy logic."""

    def note_reconciliation(self, result: "PolymarketReconciliation") -> None: ...

    def note_reconciliation_failure(self, failure_code: str) -> None: ...

    def assert_submission_allowed(self, *, closing_only: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class PolymarketLiveOrderRecord:
    intent: PolymarketLiveOrderIntent
    expected_order_id: str
    state: str
    remote_status: str
    matched_quantity: Decimal
    failure_code: str
    updated_at_ms: int

    @property
    def blocks_new_exposure(self) -> bool:
        return self.state in _OPEN_STATES


@dataclass(frozen=True, slots=True)
class PolymarketOwnedInventory:
    market_id: str
    token_id: str
    quantity: Decimal
    provisional: bool


@dataclass(frozen=True, slots=True)
class PolymarketOrderFillEvidence:
    order_id: str
    quantity: Decimal
    has_active_fills: bool
    all_active_fills_confirmed: bool


@dataclass(frozen=True, slots=True)
class PolymarketLiveRiskLimits:
    """Hard execution ceilings supplied by the independent risk service."""

    maximum_order_quote: Decimal
    maximum_token_quantity: Decimal
    maximum_intent_age_ms: int = 2_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_order_quote",
            _decimal(
                self.maximum_order_quote,
                name="maximum_order_quote",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_token_quantity",
            _decimal(
                self.maximum_token_quantity,
                name="maximum_token_quantity",
                positive=True,
            ),
        )
        maximum_intent_age_ms = int(self.maximum_intent_age_ms)
        if not 100 <= maximum_intent_age_ms <= 30_000:
            raise ValueError("maximum_intent_age_ms must lie in [100, 30000]")
        object.__setattr__(self, "maximum_intent_age_ms", maximum_intent_age_ms)


@dataclass(frozen=True, slots=True)
class PolymarketReconciliation:
    ok: bool
    can_open: bool
    can_close: bool
    foreign_order_ids: tuple[str, ...]
    foreign_position_token_ids: tuple[str, ...]
    missing_position_token_ids: tuple[str, ...]
    blocking_intent_ids: tuple[str, ...]
    errors: tuple[str, ...]


class PolymarketLiveOrderLedger:
    """Small durable ownership ledger; it never stores signed orders or secrets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            if mode != "wal":
                raise PolymarketLiveError("live ledger could not enable WAL mode")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA wal_autocheckpoint=100")
            connection.execute("PRAGMA journal_size_limit=1048576")
            self._initialize(connection)
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise PolymarketLiveError("live ledger integrity check failed")
            self._verify_event_chain(connection)
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS polymarket_live_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS polymarket_live_orders (
                intent_id TEXT PRIMARY KEY,
                bot_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                outcome TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                limit_price TEXT NOT NULL,
                quantity TEXT NOT NULL,
                fee_reserve_quote TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                parent_intent_id TEXT NOT NULL,
                closing_only INTEGER NOT NULL,
                metadata TEXT NOT NULL,
                expected_order_id TEXT NOT NULL UNIQUE,
                intent_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                remote_status TEXT NOT NULL,
                matched_quantity TEXT NOT NULL,
                failure_code TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL,
                CHECK (side IN ('BUY', 'SELL')),
                CHECK (order_type IN ('FAK', 'FOK', 'GTD')),
                CHECK (closing_only IN (0, 1))
            );
            CREATE TABLE IF NOT EXISTS polymarket_live_fills (
                trade_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT NOT NULL,
                status TEXT NOT NULL,
                observed_at_ms INTEGER NOT NULL,
                fill_sha256 TEXT NOT NULL,
                PRIMARY KEY (trade_id, order_id),
                FOREIGN KEY (order_id)
                    REFERENCES polymarket_live_orders(expected_order_id)
            );
            CREATE TABLE IF NOT EXISTS polymarket_live_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                previous_record_sha256 TEXT NOT NULL,
                record_sha256 TEXT NOT NULL UNIQUE,
                observed_at_ms INTEGER NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO polymarket_live_metadata (key, value)
            VALUES ('schema_version', ?)
            """,
            [POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION],
        )
        row = connection.execute(
            "SELECT value FROM polymarket_live_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or str(row[0]) != POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION:
            raise PolymarketLiveError("live ledger schema differs")

    @staticmethod
    def _order_row_payload(row: Mapping[str, object]) -> dict[str, object]:
        return {
            "intent_id": str(row["intent_id"]),
            "bot_id": str(row["bot_id"]),
            "market_id": str(row["market_id"]),
            "token_id": str(row["token_id"]),
            "symbol": str(row["symbol"]),
            "outcome": str(row["outcome"]),
            "side": str(row["side"]),
            "order_type": str(row["order_type"]),
            "limit_price": str(row["limit_price"]),
            "quantity": str(row["quantity"]),
            "fee_reserve_quote": str(row["fee_reserve_quote"]),
            "created_at_ms": int(row["created_at_ms"]),
            "expires_at_ms": int(row["expires_at_ms"]),
            "parent_intent_id": str(row["parent_intent_id"]),
            "closing_only": int(row["closing_only"]),
            "metadata": str(row["metadata"]),
            "expected_order_id": str(row["expected_order_id"]),
            "intent_sha256": str(row["intent_sha256"]),
            "state": str(row["state"]),
            "remote_status": str(row["remote_status"]),
            "matched_quantity": str(row["matched_quantity"]),
            "failure_code": str(row["failure_code"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }

    @classmethod
    def _verify_order_row(cls, row: Mapping[str, object]) -> None:
        if str(row["record_sha256"]) != _canonical_sha256(
            cls._order_row_payload(row)
        ):
            raise PolymarketLiveError("live order snapshot hash differs")

    @classmethod
    def _write_order_row_hash(
        cls,
        connection: sqlite3.Connection,
        intent_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM polymarket_live_orders WHERE intent_id = ?",
            [intent_id],
        ).fetchone()
        if row is None:
            raise KeyError(intent_id)
        digest = _canonical_sha256(cls._order_row_payload(row))
        connection.execute(
            """
            UPDATE polymarket_live_orders SET record_sha256 = ?
            WHERE intent_id = ?
            """,
            [digest, intent_id],
        )

    @staticmethod
    def _fill_row_payload(row: Mapping[str, object]) -> dict[str, object]:
        return {
            "trade_id": str(row["trade_id"]),
            "order_id": str(row["order_id"]),
            "market_id": str(row["market_id"]),
            "token_id": str(row["token_id"]),
            "side": str(row["side"]),
            "quantity": str(row["quantity"]),
            "price": str(row["price"]),
            "status": str(row["status"]),
            "observed_at_ms": int(row["observed_at_ms"]),
        }

    @classmethod
    def _verify_fill_row(cls, row: Mapping[str, object]) -> None:
        if str(row["fill_sha256"]) != _canonical_sha256(
            cls._fill_row_payload(row)
        ):
            raise PolymarketLiveError("live fill snapshot hash differs")

    @staticmethod
    def _verify_event_chain(connection: sqlite3.Connection) -> None:
        previous = _ZERO_SHA256
        rows = connection.execute(
            """
            SELECT sequence, intent_id, event_type, payload_json, payload_sha256,
                   previous_record_sha256, record_sha256, observed_at_ms
            FROM polymarket_live_audit ORDER BY sequence
            """
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError as exc:
                raise PolymarketLiveError("live audit JSON is invalid") from exc
            if _canonical_json(payload) != str(row["payload_json"]):
                raise PolymarketLiveError("live audit JSON is not canonical")
            payload_sha = _canonical_sha256(payload)
            record_payload = {
                "sequence": int(row["sequence"]),
                "intent_id": str(row["intent_id"]),
                "event_type": str(row["event_type"]),
                "payload_sha256": payload_sha,
                "previous_record_sha256": previous,
                "observed_at_ms": int(row["observed_at_ms"]),
            }
            record_sha = _canonical_sha256(record_payload)
            if (
                str(row["payload_sha256"]) != payload_sha
                or str(row["previous_record_sha256"]) != previous
                or str(row["record_sha256"]) != record_sha
            ):
                raise PolymarketLiveError("live audit hash chain differs")
            previous = record_sha

    @staticmethod
    def _append_audit(
        connection: sqlite3.Connection,
        *,
        intent_id: str,
        event_type: str,
        payload: Mapping[str, object],
        observed_at_ms: int,
    ) -> None:
        last = connection.execute(
            """
            SELECT sequence, record_sha256
            FROM polymarket_live_audit ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        sequence = 1 if last is None else int(last["sequence"]) + 1
        previous = _ZERO_SHA256 if last is None else str(last["record_sha256"])
        payload_json = _canonical_json(dict(payload))
        payload_sha = hashlib.sha256(payload_json.encode("ascii")).hexdigest()
        record = {
            "sequence": sequence,
            "intent_id": intent_id,
            "event_type": event_type,
            "payload_sha256": payload_sha,
            "previous_record_sha256": previous,
            "observed_at_ms": int(observed_at_ms),
        }
        connection.execute(
            """
            INSERT INTO polymarket_live_audit (
                sequence, intent_id, event_type, payload_json, payload_sha256,
                previous_record_sha256, record_sha256, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                sequence,
                intent_id,
                event_type,
                payload_json,
                payload_sha,
                previous,
                _canonical_sha256(record),
                int(observed_at_ms),
            ],
        )

    def reserve(self, prepared: PolymarketPreparedOrder, *, observed_at_ms: int) -> None:
        intent = prepared.intent
        intent_payload = intent.asdict()
        now = int(observed_at_ms)
        if now <= 0:
            raise ValueError("observed_at_ms must be positive")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM polymarket_live_orders WHERE intent_id = ?",
                [intent.intent_id],
            ).fetchone()
            intent_sha = _canonical_sha256(intent_payload)
            if existing is not None:
                self._verify_order_row(existing)
                if (
                    str(existing["intent_sha256"]) != intent_sha
                    or str(existing["expected_order_id"]) != prepared.expected_order_id
                ):
                    raise PolymarketLiveBlocked("intent_id was already bound differently")
                connection.execute("COMMIT")
                return
            connection.execute(
                """
                INSERT INTO polymarket_live_orders (
                    intent_id, bot_id, market_id, token_id, symbol, outcome,
                    side, order_type, limit_price, quantity, fee_reserve_quote,
                    created_at_ms,
                    expires_at_ms, parent_intent_id, closing_only, metadata,
                    expected_order_id, intent_sha256, state, remote_status,
                    matched_quantity, failure_code, updated_at_ms, record_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'prepared', '', '0', '', ?, '')
                """,
                [
                    intent.intent_id,
                    intent.bot_id,
                    intent.market_id,
                    intent.token_id,
                    intent.symbol,
                    intent.outcome,
                    intent.side,
                    intent.order_type,
                    format(intent.limit_price, "f"),
                    format(intent.quantity, "f"),
                    format(intent.fee_reserve_quote, "f"),
                    intent.created_at_ms,
                    intent.expires_at_ms,
                    intent.parent_intent_id,
                    int(intent.closing_only),
                    prepared.metadata,
                    prepared.expected_order_id,
                    intent_sha,
                    now,
                ],
            )
            self._write_order_row_hash(connection, intent.intent_id)
            self._append_audit(
                connection,
                intent_id=intent.intent_id,
                event_type="prepared",
                payload={
                    "intent_sha256": intent_sha,
                    "expected_order_id": prepared.expected_order_id,
                    "metadata": prepared.metadata,
                },
                observed_at_ms=now,
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition(
        self,
        intent_id: str,
        *,
        expected_states: Sequence[str],
        state: str,
        observed_at_ms: int,
        remote_status: str = "",
        matched_quantity: Decimal | str = Decimal("0"),
        failure_code: str = "",
    ) -> None:
        normalized_intent = _identifier(intent_id, name="intent_id")
        expected = tuple(str(value) for value in expected_states)
        if not expected:
            raise ValueError("expected_states cannot be empty")
        if state not in _OPEN_STATES | _TERMINAL_STATES:
            raise ValueError("live order state is invalid")
        matched = _decimal(
            matched_quantity,
            name="matched_quantity",
            nonnegative=True,
        )
        now = int(observed_at_ms)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM polymarket_live_orders WHERE intent_id = ?",
                [normalized_intent],
            ).fetchone()
            if row is None:
                raise KeyError(normalized_intent)
            self._verify_order_row(row)
            if str(row["state"]) not in expected:
                raise PolymarketStateConflict(
                    f"intent state {row['state']} does not permit {state}"
                )
            quantity = _decimal(row["quantity"], name="quantity", positive=True)
            if matched > quantity:
                raise PolymarketLiveBlocked("matched quantity exceeds intent quantity")
            connection.execute(
                """
                UPDATE polymarket_live_orders
                SET state = ?, remote_status = ?, matched_quantity = ?,
                    failure_code = ?, updated_at_ms = ?
                WHERE intent_id = ?
                """,
                [
                    state,
                    str(remote_status or "").strip().upper(),
                    format(matched, "f"),
                    str(failure_code or "").strip(),
                    now,
                    normalized_intent,
                ],
            )
            self._write_order_row_hash(connection, normalized_intent)
            self._append_audit(
                connection,
                intent_id=normalized_intent,
                event_type=state,
                payload={
                    "prior_state": str(row["state"]),
                    "state": state,
                    "remote_status": str(remote_status or "").strip().upper(),
                    "matched_quantity": format(matched, "f"),
                    "failure_code": str(failure_code or "").strip(),
                },
                observed_at_ms=now,
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def record_fill(self, fill: PolymarketRemoteFill) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            order = connection.execute(
                """
                SELECT intent_id, market_id, token_id, side, quantity
                FROM polymarket_live_orders WHERE expected_order_id = ?
                """,
                [fill.order_id],
            ).fetchone()
            if order is None:
                raise PolymarketLiveBlocked("fill is not tied to a bot-owned order")
            if (
                str(order["market_id"]) != fill.market_id
                or str(order["token_id"]) != fill.token_id
                or str(order["side"]) != fill.side
            ):
                raise PolymarketLiveBlocked("fill identity differs from owned order")
            payload = {
                "trade_id": fill.trade_id,
                "order_id": fill.order_id,
                "market_id": fill.market_id,
                "token_id": fill.token_id,
                "side": fill.side,
                "quantity": format(fill.quantity, "f"),
                "price": format(fill.price, "f"),
                "status": fill.status,
                "observed_at_ms": fill.observed_at_ms,
            }
            digest = _canonical_sha256(payload)
            existing = connection.execute(
                """
                SELECT *
                FROM polymarket_live_fills
                WHERE trade_id = ? AND order_id = ?
                """,
                [fill.trade_id, fill.order_id],
            ).fetchone()
            if existing is not None:
                self._verify_fill_row(existing)
                if (
                    str(existing["market_id"]) != fill.market_id
                    or str(existing["token_id"]) != fill.token_id
                    or str(existing["side"]) != fill.side
                    or _decimal(
                        existing["quantity"],
                        name="stored fill quantity",
                        positive=True,
                    )
                    != fill.quantity
                    or _decimal(
                        existing["price"],
                        name="stored fill price",
                        positive=True,
                    )
                    != fill.price
                ):
                    raise PolymarketLiveBlocked("existing fill economics differ")
                prior_status = str(existing["status"])
                if prior_status in _FILL_TERMINAL_STATUSES:
                    if str(existing["fill_sha256"]) != digest:
                        raise PolymarketLiveBlocked("terminal fill evidence differs")
                    connection.execute("COMMIT")
                    return
                allowed = {
                    "MATCHED": {"MATCHED", "MINED", "CONFIRMED", "RETRYING", "FAILED"},
                    "MINED": {"MINED", "CONFIRMED", "RETRYING", "FAILED"},
                    "RETRYING": {"RETRYING", "MINED", "CONFIRMED", "FAILED"},
                }
                if fill.status not in allowed.get(prior_status, set()):
                    raise PolymarketLiveBlocked("fill status regressed")
                connection.execute(
                    """
                    UPDATE polymarket_live_fills
                    SET status = ?, observed_at_ms = ?, fill_sha256 = ?
                    WHERE trade_id = ? AND order_id = ?
                    """,
                    [
                        fill.status,
                        fill.observed_at_ms,
                        digest,
                        fill.trade_id,
                        fill.order_id,
                    ],
                )
            else:
                connection.execute(
                    """
                    INSERT INTO polymarket_live_fills (
                        trade_id, order_id, market_id, token_id, side, quantity,
                        price, status, observed_at_ms, fill_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        fill.trade_id,
                        fill.order_id,
                        fill.market_id,
                        fill.token_id,
                        fill.side,
                        format(fill.quantity, "f"),
                        format(fill.price, "f"),
                        fill.status,
                        fill.observed_at_ms,
                        digest,
                    ],
                )
            active_rows = connection.execute(
                """
                SELECT * FROM polymarket_live_fills
                WHERE order_id = ? AND status != 'FAILED'
                """,
                [fill.order_id],
            ).fetchall()
            cumulative_quantity = sum(
                (
                    _decimal(
                        row["quantity"],
                        name="stored fill quantity",
                        positive=True,
                    )
                    for row in active_rows
                ),
                Decimal("0"),
            )
            for row in active_rows:
                self._verify_fill_row(row)
            order_quantity = _decimal(
                order["quantity"],
                name="order quantity",
                positive=True,
            )
            if cumulative_quantity > order_quantity:
                raise PolymarketLiveBlocked(
                    "cumulative fills exceed the signed order quantity"
                )
            self._append_audit(
                connection,
                intent_id=str(order["intent_id"]),
                event_type="fill",
                payload=payload,
                observed_at_ms=fill.observed_at_ms,
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> PolymarketLiveOrderRecord:
        PolymarketLiveOrderLedger._verify_order_row(row)
        intent = PolymarketLiveOrderIntent(
            intent_id=str(row["intent_id"]),
            bot_id=str(row["bot_id"]),
            market_id=str(row["market_id"]),
            token_id=str(row["token_id"]),
            symbol=str(row["symbol"]),
            outcome=str(row["outcome"]),
            side=str(row["side"]),
            order_type=str(row["order_type"]),
            limit_price=_decimal(row["limit_price"], name="limit_price", positive=True),
            quantity=_decimal(row["quantity"], name="quantity", positive=True),
            fee_reserve_quote=_decimal(
                row["fee_reserve_quote"],
                name="fee_reserve_quote",
                nonnegative=True,
            ),
            created_at_ms=int(row["created_at_ms"]),
            expires_at_ms=int(row["expires_at_ms"]),
            parent_intent_id=str(row["parent_intent_id"]),
            closing_only=bool(row["closing_only"]),
        )
        if str(row["metadata"]) != intent.metadata:
            raise PolymarketLiveError("stored live order metadata differs")
        if str(row["intent_sha256"]) != _canonical_sha256(intent.asdict()):
            raise PolymarketLiveError("stored live intent hash differs")
        state = str(row["state"])
        if state not in _OPEN_STATES | _TERMINAL_STATES:
            raise PolymarketLiveError("stored live order state is invalid")
        matched_quantity = _decimal(
            row["matched_quantity"],
            name="matched_quantity",
            nonnegative=True,
        )
        if matched_quantity > intent.quantity:
            raise PolymarketLiveError("stored matched quantity exceeds order quantity")
        updated_at_ms = int(row["updated_at_ms"])
        if updated_at_ms <= 0:
            raise PolymarketLiveError("stored live order update time is invalid")
        return PolymarketLiveOrderRecord(
            intent=intent,
            expected_order_id=_order_id(row["expected_order_id"]),
            state=state,
            remote_status=str(row["remote_status"]),
            matched_quantity=matched_quantity,
            failure_code=str(row["failure_code"]),
            updated_at_ms=updated_at_ms,
        )

    def records(self) -> tuple[PolymarketLiveOrderRecord, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM polymarket_live_orders ORDER BY created_at_ms, intent_id"
            ).fetchall()
            return tuple(self._record(row) for row in rows)
        finally:
            connection.close()

    def record(self, intent_id: str) -> PolymarketLiveOrderRecord:
        normalized = _identifier(intent_id, name="intent_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM polymarket_live_orders WHERE intent_id = ?",
                [normalized],
            ).fetchone()
            if row is None:
                raise KeyError(normalized)
            return self._record(row)
        finally:
            connection.close()

    def owned_order_ids(self) -> tuple[str, ...]:
        return tuple(record.expected_order_id for record in self.records())

    def open_owned_order_ids(self) -> tuple[str, ...]:
        return tuple(
            record.expected_order_id
            for record in self.records()
            if record.state in _OPEN_STATES
        )

    def reconciliation_targets(
        self,
        *,
        observed_at_ms: int,
        terminal_lookback_ms: int = 600_000,
    ) -> tuple[PolymarketLiveOrderRecord, ...]:
        now = int(observed_at_ms)
        lookback = int(terminal_lookback_ms)
        if now <= 0 or not 60_000 <= lookback <= 86_400_000:
            raise ValueError("reconciliation target window is invalid")
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT DISTINCT order_id FROM polymarket_live_fills
                WHERE status NOT IN ('CONFIRMED', 'FAILED')
                """
            ).fetchall()
        finally:
            connection.close()
        provisional_order_ids = {str(row["order_id"]) for row in rows}
        cutoff = now - lookback
        return tuple(
            record
            for record in self.records()
            if record.state in _OPEN_STATES
            or record.updated_at_ms >= cutoff
            or record.expected_order_id in provisional_order_ids
        )

    def order_fill_evidence(self, order_id: str) -> PolymarketOrderFillEvidence:
        normalized = _order_id(order_id)
        connection = self._connect()
        try:
            owned = connection.execute(
                """
                SELECT 1 FROM polymarket_live_orders
                WHERE expected_order_id = ?
                """,
                [normalized],
            ).fetchone()
            if owned is None:
                raise KeyError(normalized)
            rows = connection.execute(
                """
                SELECT * FROM polymarket_live_fills
                WHERE order_id = ? AND status != 'FAILED'
                """,
                [normalized],
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            self._verify_fill_row(row)
        quantity = sum(
            (
                _decimal(row["quantity"], name="fill quantity", positive=True)
                for row in rows
            ),
            Decimal("0"),
        )
        return PolymarketOrderFillEvidence(
            order_id=normalized,
            quantity=quantity,
            has_active_fills=bool(rows),
            all_active_fills_confirmed=bool(rows)
            and all(str(row["status"]) == "CONFIRMED" for row in rows),
        )

    def owned_inventory(self) -> tuple[PolymarketOwnedInventory, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT f.*
                FROM polymarket_live_fills AS f
                JOIN polymarket_live_orders AS o
                  ON o.expected_order_id = f.order_id
                WHERE f.status != 'FAILED'
                """
            ).fetchall()
        finally:
            connection.close()
        quantities: dict[tuple[str, str], Decimal] = {}
        provisional: dict[tuple[str, str], bool] = {}
        for row in rows:
            self._verify_fill_row(row)
            key = (str(row["market_id"]), str(row["token_id"]))
            quantity = _decimal(row["quantity"], name="fill quantity", positive=True)
            signed = quantity if str(row["side"]) == "BUY" else -quantity
            quantities[key] = quantities.get(key, Decimal("0")) + signed
            if str(row["status"]) != "CONFIRMED":
                provisional[key] = True
        output: list[PolymarketOwnedInventory] = []
        for (market_id, token_id), quantity in sorted(quantities.items()):
            if quantity < -_POSITION_TOLERANCE:
                raise PolymarketLiveError("owned inventory became negative")
            if quantity > _POSITION_TOLERANCE:
                output.append(
                    PolymarketOwnedInventory(
                        market_id=market_id,
                        token_id=token_id,
                        quantity=quantity,
                        provisional=provisional.get((market_id, token_id), False),
                    )
                )
        return tuple(output)


class PolymarketLiveCoordinator:
    """Coordinate live orders without touching account-wide or foreign state."""

    def __init__(
        self,
        venue: PolymarketLiveVenue,
        ledger: PolymarketLiveOrderLedger,
        *,
        risk_limits: PolymarketLiveRiskLimits,
        runtime_authority: PolymarketRuntimeAuthority | None = None,
        maximum_clock_skew_ms: int = 5_000,
        require_dedicated_wallet: bool = True,
    ) -> None:
        self.venue = venue
        self.ledger = ledger
        if not isinstance(risk_limits, PolymarketLiveRiskLimits):
            raise TypeError("risk_limits must be PolymarketLiveRiskLimits")
        self.risk_limits = risk_limits
        self.runtime_authority = runtime_authority
        self.maximum_clock_skew_ms = int(maximum_clock_skew_ms)
        if not 0 <= self.maximum_clock_skew_ms <= 60_000:
            raise ValueError("maximum_clock_skew_ms must lie in [0, 60000]")
        if not require_dedicated_wallet:
            raise ValueError("live Polymarket execution requires a dedicated wallet")

    def preflight(self) -> PolymarketReconciliation:
        try:
            venue = self.venue.preflight()
        except Exception as exc:
            if self.runtime_authority is not None:
                self.runtime_authority.note_reconciliation_failure(
                    exc.__class__.__name__
                )
            raise
        errors: list[str] = []
        if venue.protocol_version != 2:
            errors.append("unsupported_clob_protocol")
        if venue.geoblocked:
            errors.append("geoblocked")
        if venue.clock_skew_ms > self.maximum_clock_skew_ms:
            errors.append("clock_skew")
        if venue.closed_only:
            errors.append("closed_only")
        result = self._reconcile_snapshot(
            open_orders=venue.open_orders,
            positions=venue.positions,
            base_errors=errors,
        )
        if self.runtime_authority is not None:
            self.runtime_authority.note_reconciliation(result)
        return result

    def _transition_reconciled(
        self,
        record: PolymarketLiveOrderRecord,
        *,
        state: str,
        observed_at_ms: int,
        remote_status: str,
        matched_quantity: Decimal,
        failure_code: str = "",
    ) -> None:
        try:
            self.ledger.transition(
                record.intent.intent_id,
                expected_states=(record.state,),
                state=state,
                observed_at_ms=observed_at_ms,
                remote_status=remote_status,
                matched_quantity=matched_quantity,
                failure_code=failure_code,
            )
        except PolymarketStateConflict:
            # The authenticated stream advanced the same order concurrently.
            return

    def reconcile(self) -> PolymarketReconciliation:
        now = int(time.time() * 1_000)
        targets = self.ledger.reconciliation_targets(observed_at_ms=now)
        owned_ids = tuple(record.expected_order_id for record in targets)
        market_ids = tuple(
            sorted({record.intent.market_id for record in targets})
        )
        for fill in self.venue.fills_for_orders(owned_ids, market_ids=market_ids):
            self.ledger.record_fill(fill)
        open_orders = self.venue.open_orders()
        remote_by_id = {order.order_id: order for order in open_orders}
        for record in self.ledger.records():
            remote = remote_by_id.get(record.expected_order_id)
            if remote is not None:
                if (
                    remote.market_id != record.intent.market_id
                    or remote.token_id != record.intent.token_id
                    or remote.side != record.intent.side
                    or remote.original_quantity != record.intent.quantity
                ):
                    self._transition_reconciled(
                        record,
                        state="unknown",
                        observed_at_ms=now,
                        remote_status=remote.status,
                        matched_quantity=record.matched_quantity,
                        failure_code="remote_order_identity_mismatch",
                    )
                    continue
                state = "partial" if remote.matched_quantity > 0 else "live"
                self._transition_reconciled(
                    record,
                    state=state,
                    observed_at_ms=now,
                    remote_status=remote.status,
                    matched_quantity=remote.matched_quantity,
                )
                continue
            fill_evidence = self.ledger.order_fill_evidence(
                record.expected_order_id
            )
            if record.state in {
                "submitting",
                "unknown",
                "live",
                "partial",
                "matched_pending",
                "cancel_pending",
                "cancel_unknown",
            }:
                if fill_evidence.has_active_fills:
                    state = (
                        "filled"
                        if fill_evidence.all_active_fills_confirmed
                        else "matched_pending"
                    )
                    self._transition_reconciled(
                        record,
                        state=state,
                        observed_at_ms=now,
                        remote_status=(
                            "CONFIRMED"
                            if fill_evidence.all_active_fills_confirmed
                            else "MATCHED"
                        ),
                        matched_quantity=fill_evidence.quantity,
                    )
                else:
                    self._transition_reconciled(
                        record,
                        state="unknown",
                        observed_at_ms=now,
                        remote_status="MISSING",
                        matched_quantity=record.matched_quantity,
                        failure_code="remote_order_absent_without_terminal_evidence",
                    )
        result = self._reconcile_snapshot(
            open_orders=open_orders,
            positions=self.venue.positions(),
            base_errors=[],
        )
        if self.runtime_authority is not None:
            self.runtime_authority.note_reconciliation(result)
        return result

    def _reconcile_snapshot(
        self,
        *,
        open_orders: Sequence[PolymarketRemoteOrder],
        positions: Sequence[PolymarketRemotePosition],
        base_errors: Sequence[str],
    ) -> PolymarketReconciliation:
        errors = list(base_errors)
        records = self.ledger.records()
        owned_ids = {record.expected_order_id for record in records}
        foreign_orders = tuple(
            sorted(order.order_id for order in open_orders if order.order_id not in owned_ids)
        )
        if foreign_orders:
            errors.append("foreign_open_orders")
        owned_inventory = {item.token_id: item for item in self.ledger.owned_inventory()}
        remote_positions = {item.token_id: item for item in positions}
        foreign_positions = tuple(
            sorted(
                token_id
                for token_id, remote in remote_positions.items()
                if token_id not in owned_inventory
                or remote.quantity
                > owned_inventory[token_id].quantity + _POSITION_TOLERANCE
            )
        )
        missing_positions = tuple(
            sorted(
                token_id
                for token_id, owned in owned_inventory.items()
                if token_id not in remote_positions
                or remote_positions[token_id].quantity
                < owned.quantity - _POSITION_TOLERANCE
            )
        )
        if foreign_positions:
            errors.append("foreign_positions")
        if missing_positions:
            errors.append("owned_position_mismatch")
        blocking = tuple(
            record.intent.intent_id
            for record in records
            if record.blocks_new_exposure
        )
        if any(
            record.state in {"submitting", "unknown", "cancel_unknown"}
            for record in records
        ):
            errors.append("unknown_order_state")
        if any(item.provisional for item in owned_inventory.values()):
            errors.append("provisional_fill_state")
        unique_errors = tuple(dict.fromkeys(errors))
        ownership_ok = not foreign_orders and not foreign_positions and not missing_positions
        can_close = ownership_ok and "geoblocked" not in unique_errors
        can_open = not unique_errors and not blocking
        return PolymarketReconciliation(
            ok=not unique_errors,
            can_open=can_open,
            can_close=can_close,
            foreign_order_ids=foreign_orders,
            foreign_position_token_ids=foreign_positions,
            missing_position_token_ids=missing_positions,
            blocking_intent_ids=blocking,
            errors=unique_errors,
        )

    def submit(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketLiveOrderRecord:
        now = int(time.time() * 1_000)
        if intent.created_at_ms - now > self.maximum_clock_skew_ms:
            raise PolymarketLiveBlocked("live intent timestamp is in the future")
        if now - intent.created_at_ms > self.risk_limits.maximum_intent_age_ms:
            raise PolymarketLiveBlocked("live intent exceeded its execution TTL")
        if intent.expires_at_ms <= now:
            raise PolymarketLiveBlocked("live intent has already expired")
        gate = self.preflight()
        if self.runtime_authority is not None:
            self.runtime_authority.assert_submission_allowed(
                closing_only=intent.closing_only
            )
        if intent.closing_only:
            if not gate.can_close:
                raise PolymarketLiveBlocked(
                    f"live close blocked by reconciliation: {gate.errors}"
                )
            try:
                parent = self.ledger.record(intent.parent_intent_id)
            except KeyError as exc:
                raise PolymarketLiveBlocked(
                    "closing intent parent is not bot-owned"
                ) from exc
            if (
                parent.intent.bot_id != intent.bot_id
                or parent.intent.market_id != intent.market_id
                or parent.intent.token_id != intent.token_id
                or parent.intent.side != "BUY"
                or parent.intent.closing_only
            ):
                raise PolymarketLiveBlocked(
                    "closing intent differs from its bot-owned parent"
                )
            inventory = {
                item.token_id: item.quantity
                for item in self.ledger.owned_inventory()
            }
            if inventory.get(intent.token_id, Decimal("0")) < intent.quantity:
                raise PolymarketLiveBlocked(
                    "closing intent exceeds confirmed bot-owned inventory"
                )
        elif not gate.can_open:
            raise PolymarketLiveBlocked(
                f"live open blocked by reconciliation: {gate.errors}"
            )
        else:
            order_quote = (
                intent.limit_price * intent.quantity + intent.fee_reserve_quote
            )
            if order_quote > self.risk_limits.maximum_order_quote:
                raise PolymarketLiveBlocked("live order exceeds its quote ceiling")
            inventory = {
                item.token_id: item.quantity
                for item in self.ledger.owned_inventory()
            }
            if (
                inventory.get(intent.token_id, Decimal("0")) + intent.quantity
                > self.risk_limits.maximum_token_quantity
            ):
                raise PolymarketLiveBlocked(
                    "live order exceeds its token inventory ceiling"
                )
        tick = _decimal(tick_size, name="tick_size", positive=True)
        if intent.limit_price % tick:
            raise PolymarketLiveBlocked("limit price is not aligned to venue tick")
        funding = self.venue.funding(intent, neg_risk=bool(neg_risk))
        if intent.side == "BUY":
            if funding.asset_type != "COLLATERAL":
                raise PolymarketLiveBlocked("venue returned the wrong funding asset")
            required = (
                intent.limit_price * intent.quantity + intent.fee_reserve_quote
            )
        else:
            if (
                funding.asset_type != "CONDITIONAL"
                or funding.token_id != intent.token_id
            ):
                raise PolymarketLiveBlocked("venue returned the wrong funding asset")
            required = intent.quantity
        if (
            funding.available_balance < required
            or funding.available_allowance < required
        ):
            raise PolymarketLiveBlocked(
                "live order has insufficient balance or exchange allowance"
            )
        prepared = self.venue.prepare_order(
            intent,
            tick_size=tick,
            neg_risk=bool(neg_risk),
        )
        now = int(time.time() * 1_000)
        self.ledger.reserve(prepared, observed_at_ms=now)
        self.ledger.transition(
            intent.intent_id,
            expected_states=("prepared",),
            state="submitting",
            observed_at_ms=now,
        )
        try:
            response = self.venue.submit_order(prepared)
        except PolymarketVenueRejected as exc:
            current = self.ledger.record(intent.intent_id)
            if current.state != "submitting":
                self.ledger.transition(
                    intent.intent_id,
                    expected_states=(current.state,),
                    state="unknown",
                    observed_at_ms=int(time.time() * 1_000),
                    remote_status=current.remote_status,
                    matched_quantity=current.matched_quantity,
                    failure_code="rejection_conflicts_with_stream_evidence",
                )
                raise PolymarketLiveUnknownState(
                    "venue rejection conflicts with authenticated stream evidence"
                ) from exc
            self.ledger.transition(
                intent.intent_id,
                expected_states=("submitting",),
                state="rejected",
                observed_at_ms=int(time.time() * 1_000),
                failure_code=exc.__class__.__name__,
            )
            return self.ledger.record(intent.intent_id)
        except Exception as exc:
            current = self.ledger.record(intent.intent_id)
            if current.state in {
                "live",
                "partial",
                "matched_pending",
                "cancel_pending",
                "cancel_unknown",
                "cancelled",
                "filled",
            }:
                return current
            self.ledger.transition(
                intent.intent_id,
                expected_states=(current.state,),
                state="unknown",
                observed_at_ms=int(time.time() * 1_000),
                remote_status=current.remote_status,
                matched_quantity=current.matched_quantity,
                failure_code=exc.__class__.__name__,
            )
            raise PolymarketLiveUnknownState(
                "Polymarket submission outcome is unknown; no retry is permitted"
            ) from exc
        if not response.accepted:
            current = self.ledger.record(intent.intent_id)
            if current.state != "submitting":
                self.ledger.transition(
                    intent.intent_id,
                    expected_states=(current.state,),
                    state="unknown",
                    observed_at_ms=int(time.time() * 1_000),
                    remote_status=current.remote_status,
                    matched_quantity=current.matched_quantity,
                    failure_code="rejection_conflicts_with_stream_evidence",
                )
                raise PolymarketLiveUnknownState(
                    "venue rejection conflicts with authenticated stream evidence"
                )
            self.ledger.transition(
                intent.intent_id,
                expected_states=("submitting",),
                state="rejected",
                observed_at_ms=int(time.time() * 1_000),
                remote_status=response.status,
                failure_code=response.rejection_code,
            )
            return self.ledger.record(intent.intent_id)
        if response.order_id != prepared.expected_order_id:
            current = self.ledger.record(intent.intent_id)
            self.ledger.transition(
                intent.intent_id,
                expected_states=(current.state,),
                state="unknown",
                observed_at_ms=int(time.time() * 1_000),
                remote_status=response.status,
                matched_quantity=current.matched_quantity,
                failure_code="venue_order_id_mismatch",
            )
            raise PolymarketLiveUnknownState("venue order ID differs from signed order hash")
        state = (
            "live"
            if response.status in {"live", "delayed", "unmatched"}
            else "matched_pending"
        )
        current = self.ledger.record(intent.intent_id)
        if current.state in {
            "partial",
            "matched_pending",
            "cancel_pending",
            "cancel_unknown",
            "cancelled",
            "filled",
        }:
            return current
        if current.state == "live" and state == "live":
            return current
        if current.state in {"rejected", "expired", "failed"}:
            self.ledger.transition(
                intent.intent_id,
                expected_states=(current.state,),
                state="unknown",
                observed_at_ms=int(time.time() * 1_000),
                remote_status=response.status,
                matched_quantity=current.matched_quantity,
                failure_code="acceptance_conflicts_with_terminal_state",
            )
            raise PolymarketLiveUnknownState(
                "venue acceptance conflicts with terminal local evidence"
            )
        self.ledger.transition(
            intent.intent_id,
            expected_states=(current.state,),
            state=state,
            observed_at_ms=int(time.time() * 1_000),
            remote_status=response.status,
            matched_quantity=current.matched_quantity,
        )
        return self.ledger.record(intent.intent_id)

    def cancel_owned_open_orders(self) -> PolymarketCancelResult:
        owned = self.ledger.open_owned_order_ids()
        if not owned:
            return PolymarketCancelResult((), ())
        remote = {order.order_id for order in self.venue.open_orders()}
        targets = tuple(order_id for order_id in owned if order_id in remote)
        missing = tuple(order_id for order_id in owned if order_id not in remote)
        now = int(time.time() * 1_000)
        by_order = {
            record.expected_order_id: record for record in self.ledger.records()
        }
        for order_id in missing:
            record = by_order[order_id]
            self.ledger.transition(
                record.intent.intent_id,
                expected_states=(record.state,),
                state="cancel_unknown",
                observed_at_ms=now,
                remote_status=record.remote_status or "MISSING",
                matched_quantity=record.matched_quantity,
                failure_code="remote_order_absent_without_terminal_evidence",
            )
        if not targets:
            raise PolymarketLiveUnknownState(
                "owned orders are absent without terminal cancellation evidence"
            )
        for order_id in targets:
            record = by_order[order_id]
            self.ledger.transition(
                record.intent.intent_id,
                expected_states=(record.state,),
                state="cancel_pending",
                observed_at_ms=now,
                remote_status=record.remote_status,
                matched_quantity=record.matched_quantity,
            )
        try:
            result = self.venue.cancel_orders(targets)
        except Exception as exc:
            for order_id in targets:
                record = by_order[order_id]
                current = self.ledger.record(record.intent.intent_id)
                self.ledger.transition(
                    record.intent.intent_id,
                    expected_states=("cancel_pending",),
                    state="cancel_unknown",
                    observed_at_ms=int(time.time() * 1_000),
                    remote_status=current.remote_status,
                    matched_quantity=current.matched_quantity,
                    failure_code=exc.__class__.__name__,
                )
            raise PolymarketLiveUnknownState(
                "Polymarket cancellation outcome is unknown"
            ) from exc
        for order_id in result.cancelled_order_ids:
            if order_id not in targets:
                raise PolymarketLiveBlocked("venue cancelled an unrequested order")
            record = by_order[order_id]
            current = self.ledger.record(record.intent.intent_id)
            self.ledger.transition(
                record.intent.intent_id,
                expected_states=("cancel_pending",),
                state="cancelled",
                observed_at_ms=int(time.time() * 1_000),
                remote_status="CANCELLED",
                matched_quantity=current.matched_quantity,
            )
        for order_id in result.failed_order_ids:
            if order_id not in targets:
                raise PolymarketLiveBlocked("venue reported an unrequested order")
            record = by_order[order_id]
            current = self.ledger.record(record.intent.intent_id)
            self.ledger.transition(
                record.intent.intent_id,
                expected_states=("cancel_pending",),
                state="cancel_unknown",
                observed_at_ms=int(time.time() * 1_000),
                remote_status=current.remote_status,
                matched_quantity=current.matched_quantity,
                failure_code="venue_cancel_failed",
            )
        accounted = set(result.cancelled_order_ids) | set(result.failed_order_ids)
        if accounted != set(targets):
            for order_id in set(targets) - accounted:
                record = by_order[order_id]
                current = self.ledger.record(record.intent.intent_id)
                self.ledger.transition(
                    record.intent.intent_id,
                    expected_states=("cancel_pending",),
                    state="cancel_unknown",
                    observed_at_ms=int(time.time() * 1_000),
                    remote_status=current.remote_status,
                    matched_quantity=current.matched_quantity,
                    failure_code="venue_cancel_response_incomplete",
                )
            raise PolymarketLiveUnknownState("venue cancellation response was incomplete")
        if missing:
            raise PolymarketLiveUnknownState(
                "one or more owned orders were absent during cancellation"
            )
        return result


__all__ = [
    "POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION",
    "POLYMARKET_LIVE_ORDER_SCHEMA_VERSION",
    "PolymarketCancelResult",
    "PolymarketFundingPreflight",
    "PolymarketLiveBlocked",
    "PolymarketLiveCoordinator",
    "PolymarketLiveError",
    "PolymarketLiveOrderIntent",
    "PolymarketLiveOrderLedger",
    "PolymarketLiveOrderRecord",
    "PolymarketLiveRiskLimits",
    "PolymarketLiveUnknownState",
    "PolymarketLiveVenue",
    "PolymarketRuntimeAuthority",
    "PolymarketOrderFillEvidence",
    "PolymarketOwnedInventory",
    "PolymarketPreparedOrder",
    "PolymarketReconciliation",
    "PolymarketRemoteFill",
    "PolymarketRemoteOrder",
    "PolymarketRemotePosition",
    "PolymarketSubmission",
    "PolymarketStateConflict",
    "PolymarketVenuePreflight",
    "PolymarketVenueRejected",
    "polymarket_live_metadata",
]
