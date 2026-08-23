"""Fail-closed live Polymarket order ownership and reconciliation.

This module contains no Binance execution imports. A Polymarket venue adapter
may consume external market signals elsewhere, but orders, positions, P&L, and
recovery remain entirely owned by this subsystem.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import ContextManager, Mapping, Protocol, Sequence


POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION = "polymarket-live-ledger-v3"
POLYMARKET_LIVE_ORDER_SCHEMA_VERSION = "polymarket-live-order-v1"
_POLYMARKET_LIVE_LEDGER_V1 = "polymarket-live-ledger-v1"
_POLYMARKET_LIVE_LEDGER_V2 = "polymarket-live-ledger-v2"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_ORDER_ID = re.compile(r"^0x[0-9a-f]{64}$")
_BYTES32 = re.compile(r"^0x[0-9a-f]{64}$")
_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")
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
_FILL_ACTIVE_STATUSES = frozenset(
    {"MATCHED_NOT_BROADCASTED", "MATCHED", "MINED", "CONFIRMED", "RETRYING"}
)
_FILL_TERMINAL_STATUSES = frozenset({"CONFIRMED", "FAILED"})
_REMOTE_ORDER_ACTIVE_STATUSES = frozenset({"LIVE"})
_REMOTE_ORDER_TERMINAL_STATUSES = frozenset(
    {"INVALID", "CANCELED_MARKET_RESOLVED", "CANCELED", "MATCHED"}
)
_REDEMPTION_TRANSITIONS = {
    "prepared": frozenset({"submitting", "failed"}),
    "submitting": frozenset({"submitted", "unknown", "failed"}),
    "submitted": frozenset({"confirmed", "unknown", "failed"}),
    "unknown": frozenset({"confirmed", "failed"}),
    "confirmed": frozenset(),
    "failed": frozenset(),
}


class PolymarketLiveError(RuntimeError):
    """Base error for the independent live Polymarket boundary."""


class PolymarketLiveBlocked(PolymarketLiveError):
    """Raised when a deterministic safety or compliance gate blocks an action."""


class PolymarketLiveUnknownState(PolymarketLiveError):
    """Raised when the venue may have accepted an operation but did not prove it."""


class PolymarketVenueRejected(PolymarketLiveError):
    """Raised only when the venue proves that an order was rejected."""


class PolymarketVenueTemporarilyUnavailable(PolymarketLiveBlocked):
    """Raised when the venue proves an endpoint is temporarily unavailable."""


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


def _remote_order_status(value: object) -> str:
    status = str(value or "").strip().upper()
    if status.startswith("ORDER_STATUS_"):
        status = status.removeprefix("ORDER_STATUS_")
    return status


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


def _required_integer(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return int(value)


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
        object.__setattr__(
            self, "intent_id", _identifier(self.intent_id, name="intent_id")
        )
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
    maker_address: str
    side: str
    order_type: str
    price: Decimal
    status: str
    original_quantity: Decimal
    matched_quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _order_id(self.order_id))
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        maker_address = str(self.maker_address or "").strip().lower()
        if _ADDRESS.fullmatch(maker_address) is None:
            raise ValueError("remote order maker address is invalid")
        object.__setattr__(self, "maker_address", maker_address)
        side = str(self.side or "").strip().upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("remote order side is invalid")
        object.__setattr__(self, "side", side)
        order_type = str(self.order_type or "").strip().upper()
        if order_type not in {"GTC", "FOK", "GTD", "FAK"}:
            raise ValueError("remote order type is invalid")
        object.__setattr__(self, "order_type", order_type)
        price = _decimal(self.price, name="remote order price", positive=True)
        if price >= 1:
            raise ValueError("remote order price must be below one")
        object.__setattr__(self, "price", price)
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
    role: str
    reported_fee_rate_bps: int
    fee_rate: Decimal | None
    fee_exponent: int | None
    fee_quote: Decimal | None
    fee_schedule_sha256: str
    transaction_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trade_id", _identifier(self.trade_id, name="trade_id")
        )
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
        role = str(self.role or "").strip().upper()
        if role not in {"MAKER", "TAKER"}:
            raise ValueError("fill role is invalid")
        object.__setattr__(self, "role", role)
        reported_fee_rate_bps = int(self.reported_fee_rate_bps)
        if not -1 <= reported_fee_rate_bps <= 10_000:
            raise ValueError("reported fill fee rate is invalid")
        object.__setattr__(
            self,
            "reported_fee_rate_bps",
            reported_fee_rate_bps,
        )
        fee_schedule_sha256 = str(self.fee_schedule_sha256 or "").strip().lower()
        accounting_values = (
            self.fee_rate is not None,
            self.fee_exponent is not None,
            self.fee_quote is not None,
            bool(fee_schedule_sha256),
        )
        if any(accounting_values) and not all(accounting_values):
            raise ValueError("fill fee accounting evidence is incomplete")
        if all(accounting_values):
            fee_rate = _decimal(
                self.fee_rate,
                name="fill fee rate",
                nonnegative=True,
            )
            fee_exponent = _required_integer(
                self.fee_exponent,
                name="fill fee exponent",
            )
            fee_quote = _decimal(
                self.fee_quote,
                name="fill fee quote",
                nonnegative=True,
            )
            if (
                fee_rate > 1
                or fee_exponent <= 0
                or fee_quote > self.quantity
                or re.fullmatch(r"[0-9a-f]{64}", fee_schedule_sha256) is None
            ):
                raise ValueError("fill fee accounting evidence is invalid")
            if role == "MAKER" and fee_quote:
                raise ValueError("maker fills cannot carry a Polymarket fee")
            if fee_rate == 0 and fee_quote:
                raise ValueError("zero-rate fill cannot carry a Polymarket fee")
            object.__setattr__(self, "fee_rate", fee_rate)
            object.__setattr__(self, "fee_exponent", fee_exponent)
            object.__setattr__(self, "fee_quote", fee_quote)
            object.__setattr__(
                self,
                "fee_schedule_sha256",
                fee_schedule_sha256,
            )
        else:
            object.__setattr__(self, "fee_rate", None)
            object.__setattr__(self, "fee_exponent", None)
            object.__setattr__(self, "fee_quote", None)
            object.__setattr__(self, "fee_schedule_sha256", "")
        transaction_hash = str(self.transaction_hash or "").strip().lower()
        if transaction_hash and _BYTES32.fullmatch(transaction_hash) is None:
            raise ValueError("fill transaction hash is invalid")
        if status in {"MINED", "CONFIRMED"} and not transaction_hash:
            raise ValueError("mined or confirmed fill requires a transaction hash")
        object.__setattr__(self, "transaction_hash", transaction_hash)

    @property
    def accounting_verified(self) -> bool:
        return self.fee_rate is not None


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

    def __post_init__(self) -> None:
        wallet_address = str(self.wallet_address or "").strip().lower()
        if _ADDRESS.fullmatch(wallet_address) is None:
            raise ValueError("preflight wallet address is invalid")
        object.__setattr__(self, "wallet_address", wallet_address)

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
class PolymarketCloseQuote:
    market_id: str
    token_id: str
    quantity: Decimal
    limit_price: Decimal
    average_price: Decimal
    fee_quote: Decimal
    net_quote: Decimal
    fee_rate: Decimal
    fee_exponent: int
    tick_size: Decimal
    minimum_order_size: Decimal
    neg_risk: bool
    source_time_ms: int
    observed_at_ms: int
    book_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        quantity = _decimal(self.quantity, name="close quantity", positive=True)
        price = _decimal(self.limit_price, name="close limit price", positive=True)
        average_price = _decimal(
            self.average_price,
            name="close average price",
            positive=True,
        )
        fee_quote = _decimal(
            self.fee_quote,
            name="close fee quote",
            nonnegative=True,
        )
        net_quote = _decimal(
            self.net_quote,
            name="close net quote",
            positive=True,
        )
        fee_rate = _decimal(
            self.fee_rate,
            name="close fee rate",
            nonnegative=True,
        )
        fee_exponent = int(self.fee_exponent)
        tick = _decimal(self.tick_size, name="close tick size", positive=True)
        minimum = _decimal(
            self.minimum_order_size,
            name="close minimum order size",
            positive=True,
        )
        if price >= 1 or price % tick or average_price >= 1 or average_price < price:
            raise ValueError("close limit price is invalid for the venue tick")
        if quantity < minimum:
            raise ValueError("close quantity is below the venue minimum")
        if fee_rate > 1 or fee_exponent <= 0:
            raise ValueError("close fee parameters are invalid")
        if net_quote != average_price * quantity - fee_quote:
            raise ValueError("close quote proceeds do not reconcile")
        source_time = int(self.source_time_ms)
        observed_at = int(self.observed_at_ms)
        if source_time <= 0 or observed_at <= 0:
            raise ValueError("close quote chronology is invalid")
        digest = str(self.book_payload_sha256 or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("close quote payload hash is invalid")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "limit_price", price)
        object.__setattr__(self, "average_price", average_price)
        object.__setattr__(self, "fee_quote", fee_quote)
        object.__setattr__(self, "net_quote", net_quote)
        object.__setattr__(self, "fee_rate", fee_rate)
        object.__setattr__(self, "fee_exponent", fee_exponent)
        object.__setattr__(self, "tick_size", tick)
        object.__setattr__(self, "minimum_order_size", minimum)
        object.__setattr__(self, "source_time_ms", source_time)
        object.__setattr__(self, "observed_at_ms", observed_at)
        object.__setattr__(self, "book_payload_sha256", digest)

    @property
    def source_age_ms(self) -> int:
        return self.observed_at_ms - self.source_time_ms


@dataclass(frozen=True, slots=True)
class PolymarketOpenQuote:
    market_id: str
    token_id: str
    outcome: str
    quantity: Decimal
    limit_price: Decimal
    average_price: Decimal
    fee_quote: Decimal
    total_quote: Decimal
    fee_rate: Decimal
    fee_exponent: int
    tick_size: Decimal
    minimum_order_size: Decimal
    neg_risk: bool
    source_time_ms: int
    observed_at_ms: int
    book_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        outcome = str(self.outcome or "").strip().title()
        if outcome not in {"Up", "Down"}:
            raise ValueError("open quote outcome must be Up or Down")
        object.__setattr__(self, "outcome", outcome)
        quantity = _decimal(self.quantity, name="open quantity", positive=True)
        limit_price = _decimal(
            self.limit_price,
            name="open limit price",
            positive=True,
        )
        average_price = _decimal(
            self.average_price,
            name="open average price",
            positive=True,
        )
        fee_quote = _decimal(
            self.fee_quote,
            name="open fee quote",
            nonnegative=True,
        )
        total_quote = _decimal(
            self.total_quote,
            name="open total quote",
            positive=True,
        )
        fee_rate = _decimal(
            self.fee_rate,
            name="open fee rate",
            nonnegative=True,
        )
        fee_exponent = int(self.fee_exponent)
        tick = _decimal(self.tick_size, name="open tick size", positive=True)
        minimum = _decimal(
            self.minimum_order_size,
            name="open minimum order size",
            positive=True,
        )
        if (
            limit_price >= 1
            or limit_price % tick
            or average_price >= 1
            or average_price > limit_price
        ):
            raise ValueError("open price is invalid for the venue tick")
        if quantity < minimum:
            raise ValueError("open quantity is below the venue minimum")
        if fee_rate > 1 or fee_exponent <= 0:
            raise ValueError("open fee parameters are invalid")
        if total_quote != average_price * quantity + fee_quote:
            raise ValueError("open quote cost does not reconcile")
        if type(self.neg_risk) is not bool:
            raise ValueError("open quote neg-risk flag is invalid")
        source_time = int(self.source_time_ms)
        observed_at = int(self.observed_at_ms)
        if source_time <= 0 or observed_at <= 0:
            raise ValueError("open quote chronology is invalid")
        digest = str(self.book_payload_sha256 or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("open quote payload hash is invalid")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "average_price", average_price)
        object.__setattr__(self, "fee_quote", fee_quote)
        object.__setattr__(self, "total_quote", total_quote)
        object.__setattr__(self, "fee_rate", fee_rate)
        object.__setattr__(self, "fee_exponent", fee_exponent)
        object.__setattr__(self, "tick_size", tick)
        object.__setattr__(self, "minimum_order_size", minimum)
        object.__setattr__(self, "source_time_ms", source_time)
        object.__setattr__(self, "observed_at_ms", observed_at)
        object.__setattr__(self, "book_payload_sha256", digest)

    @property
    def source_age_ms(self) -> int:
        return self.observed_at_ms - self.source_time_ms

    @property
    def fee_per_share(self) -> Decimal:
        return self.fee_quote / self.quantity


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
        if len(set(self.cancelled_order_ids)) != len(self.cancelled_order_ids):
            raise ValueError("cancelled order IDs contain duplicates")
        if len(set(self.failed_order_ids)) != len(self.failed_order_ids):
            raise ValueError("failed order IDs contain duplicates")
        if set(self.cancelled_order_ids) & set(self.failed_order_ids):
            raise ValueError("cancel result sets overlap")


class PolymarketLiveVenue(Protocol):
    """Authenticated venue boundary used by the coordinator."""

    @property
    def wallet_address(self) -> str: ...

    def preflight(self) -> PolymarketVenuePreflight: ...

    def prepare_order(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketPreparedOrder: ...

    def submit_order(
        self, prepared: PolymarketPreparedOrder
    ) -> PolymarketSubmission: ...

    def open_orders(self) -> tuple[PolymarketRemoteOrder, ...]: ...

    def orders_by_id(
        self,
        order_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteOrder, ...]: ...

    def open_quote(
        self,
        *,
        market_id: str,
        token_id: str,
        outcome: str,
        quantity: Decimal,
        maximum_book_age_ms: int,
    ) -> PolymarketOpenQuote: ...

    def close_quote(
        self,
        *,
        market_id: str,
        token_id: str,
        quantity: Decimal,
        maximum_book_age_ms: int,
    ) -> PolymarketCloseQuote: ...

    def fills_for_orders(
        self,
        order_ids: Sequence[str],
        *,
        market_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteFill, ...]: ...

    def positions(self) -> tuple[PolymarketRemotePosition, ...]: ...

    def collateral_balance(self) -> Decimal: ...

    def funding(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        neg_risk: bool,
    ) -> PolymarketFundingPreflight: ...

    def cancel_orders(self, order_ids: Sequence[str]) -> PolymarketCancelResult: ...


class PolymarketRuntimeAuthority(Protocol):
    """Runtime liveness gate implemented independently of strategy logic."""

    def reconciliation_checkpoint(self) -> int: ...

    def note_reconciliation(
        self,
        result: "PolymarketReconciliation",
        *,
        checkpoint: int | None = None,
    ) -> None: ...

    def note_reconciliation_failure(
        self,
        failure_code: str,
        *,
        checkpoint: int | None = None,
    ) -> None: ...

    def assert_submission_allowed(self, *, closing_only: bool) -> None: ...

    def submission_guard(
        self,
        *,
        closing_only: bool,
    ) -> ContextManager[None]: ...


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
class PolymarketOwnedLot:
    parent_intent_id: str
    market_id: str
    token_id: str
    quantity: Decimal
    reserved_close_quantity: Decimal
    provisional: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_intent_id",
            _identifier(self.parent_intent_id, name="parent_intent_id"),
        )
        object.__setattr__(self, "market_id", _condition_id(self.market_id))
        object.__setattr__(self, "token_id", _token_id(self.token_id))
        quantity = _decimal(self.quantity, name="lot quantity", positive=True)
        reserved = _decimal(
            self.reserved_close_quantity,
            name="reserved close quantity",
            nonnegative=True,
        )
        if reserved > quantity:
            raise ValueError("reserved close quantity exceeds lot quantity")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "reserved_close_quantity", reserved)

    @property
    def available_quantity(self) -> Decimal:
        if self.provisional:
            return Decimal("0")
        return self.quantity - self.reserved_close_quantity


@dataclass(frozen=True, slots=True)
class PolymarketConditionAccounting:
    """Verified bot-only cash flow and inventory for one condition."""

    condition_id: str
    gross_buy_cost_quote: Decimal
    gross_sell_proceeds_quote: Decimal
    confirmed_redemption_payout_quote: Decimal
    up_quantity: Decimal
    down_quantity: Decimal
    up_cost_basis_quote: Decimal
    down_cost_basis_quote: Decimal
    confirmed_fill_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _condition_id(self.condition_id))
        for field_name in (
            "gross_buy_cost_quote",
            "gross_sell_proceeds_quote",
            "confirmed_redemption_payout_quote",
            "up_quantity",
            "down_quantity",
            "up_cost_basis_quote",
            "down_cost_basis_quote",
        ):
            object.__setattr__(
                self,
                field_name,
                _decimal(
                    getattr(self, field_name),
                    name=field_name,
                    nonnegative=True,
                ),
            )
        count = int(self.confirmed_fill_count)
        if count < 0:
            raise ValueError("confirmed fill count cannot be negative")
        object.__setattr__(self, "confirmed_fill_count", count)

    @property
    def guaranteed_payout_quote(self) -> Decimal:
        return min(self.up_quantity, self.down_quantity)

    def quantity(self, outcome: str) -> Decimal:
        if outcome == "Up":
            return self.up_quantity
        if outcome == "Down":
            return self.down_quantity
        raise ValueError("condition accounting outcome is invalid")

    def cost_basis_quote(self, outcome: str) -> Decimal:
        if outcome == "Up":
            return self.up_cost_basis_quote
        if outcome == "Down":
            return self.down_cost_basis_quote
        raise ValueError("condition accounting outcome is invalid")

    @property
    def net_cash_outflow_quote(self) -> Decimal:
        return (
            self.gross_buy_cost_quote
            - self.gross_sell_proceeds_quote
            - self.confirmed_redemption_payout_quote
        )

    @property
    def maximum_loss_quote(self) -> Decimal:
        return max(
            Decimal("0"),
            self.net_cash_outflow_quote - self.guaranteed_payout_quote,
        )

    @property
    def inventory_downside_quote(self) -> Decimal:
        """Return remaining cost basis not protected by paired payout."""

        return max(
            Decimal("0"),
            self.up_cost_basis_quote
            + self.down_cost_basis_quote
            - self.guaranteed_payout_quote,
        )


@dataclass(frozen=True, slots=True)
class PolymarketLedgerRevision:
    """A cheap immutable tip used to reject torn multi-query snapshots."""

    sequence: int
    record_sha256: str

    def __post_init__(self) -> None:
        sequence = int(self.sequence)
        digest = str(self.record_sha256 or "").strip().lower()
        if (
            sequence < 0
            or (sequence == 0 and digest != _ZERO_SHA256)
            or (sequence > 0 and re.fullmatch(r"[0-9a-f]{64}", digest) is None)
        ):
            raise ValueError("Polymarket ledger revision is invalid")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "record_sha256", digest)


@dataclass(frozen=True, slots=True)
class PolymarketRealizedPnlEvent:
    """Exact fee-inclusive realized PnL from one owned close or redemption."""

    event_id: str
    condition_id: str
    source: str
    observed_at_ms: int
    proceeds_quote: Decimal
    consumed_cost_basis_quote: Decimal
    pnl_quote: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            _identifier(self.event_id, name="realized event_id"),
        )
        object.__setattr__(self, "condition_id", _condition_id(self.condition_id))
        source = str(self.source or "").strip().lower()
        if source not in {"sell_fill", "redemption"}:
            raise ValueError("Polymarket realized PnL source is invalid")
        observed = int(self.observed_at_ms)
        if observed <= 0:
            raise ValueError("Polymarket realized PnL time is invalid")
        proceeds = _decimal(
            self.proceeds_quote,
            name="realized proceeds",
            nonnegative=True,
        )
        basis = _decimal(
            self.consumed_cost_basis_quote,
            name="realized cost basis",
            nonnegative=True,
        )
        pnl = _decimal(self.pnl_quote, name="realized PnL")
        if pnl != proceeds - basis:
            raise ValueError("Polymarket realized PnL accounting differs")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "observed_at_ms", observed)
        object.__setattr__(self, "proceeds_quote", proceeds)
        object.__setattr__(self, "consumed_cost_basis_quote", basis)
        object.__setattr__(self, "pnl_quote", pnl)


@dataclass(frozen=True, slots=True)
class PolymarketOrderFillEvidence:
    order_id: str
    quantity: Decimal
    has_active_fills: bool
    all_active_fills_confirmed: bool


@dataclass(frozen=True, slots=True)
class PolymarketRedemptionRecord:
    redemption_id: str
    condition_id: str
    attempt: int
    inventory: tuple[PolymarketOwnedInventory, ...]
    preflight_json: str
    state: str
    transaction_id: str
    transaction_hash: str
    failure_code: str
    created_at_ms: int
    updated_at_ms: int
    payout_quote: Decimal = Decimal("0")
    payout_proof_sha256: str = ""
    payout_accounting_state: str = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PolymarketLiveRiskLimits:
    """Hard execution ceilings supplied by the independent risk service."""

    maximum_order_quote: Decimal
    maximum_token_quantity: Decimal
    maximum_total_at_risk_quote: Decimal
    maximum_active_markets: int
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
        total_at_risk = _decimal(
            self.maximum_total_at_risk_quote,
            name="maximum_total_at_risk_quote",
            positive=True,
        )
        if total_at_risk < self.maximum_order_quote:
            raise ValueError(
                "maximum_total_at_risk_quote cannot be below maximum_order_quote"
            )
        object.__setattr__(
            self,
            "maximum_total_at_risk_quote",
            total_at_risk,
        )
        maximum_active_markets = int(self.maximum_active_markets)
        if not 1 <= maximum_active_markets <= 10:
            raise ValueError("maximum_active_markets must lie in [1, 10]")
        object.__setattr__(
            self,
            "maximum_active_markets",
            maximum_active_markets,
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
            mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            ).lower()
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
                role TEXT NOT NULL,
                reported_fee_rate_bps INTEGER NOT NULL,
                fee_rate TEXT NOT NULL,
                fee_exponent INTEGER NOT NULL,
                fee_quote TEXT NOT NULL,
                fee_schedule_sha256 TEXT NOT NULL,
                transaction_hash TEXT NOT NULL,
                accounting_state TEXT NOT NULL,
                fill_sha256 TEXT NOT NULL,
                PRIMARY KEY (trade_id, order_id),
                FOREIGN KEY (order_id)
                    REFERENCES polymarket_live_orders(expected_order_id),
                CHECK (role IN ('UNKNOWN', 'MAKER', 'TAKER')),
                CHECK (accounting_state IN ('UNKNOWN', 'VERIFIED'))
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
            CREATE TABLE IF NOT EXISTS polymarket_live_redemptions (
                redemption_id TEXT PRIMARY KEY,
                condition_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                inventory_json TEXT NOT NULL,
                preflight_json TEXT NOT NULL,
                state TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                transaction_hash TEXT NOT NULL,
                failure_code TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                payout_quote TEXT NOT NULL,
                payout_proof_sha256 TEXT NOT NULL,
                payout_accounting_state TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                UNIQUE (condition_id, attempt),
                CHECK (attempt > 0),
                CHECK (payout_accounting_state IN ('UNKNOWN', 'VERIFIED'))
            );
            CREATE UNIQUE INDEX IF NOT EXISTS
                polymarket_live_redemptions_transaction_id
                ON polymarket_live_redemptions (transaction_id)
                WHERE transaction_id != '';
            CREATE UNIQUE INDEX IF NOT EXISTS
                polymarket_live_redemptions_transaction_hash
                ON polymarket_live_redemptions (transaction_hash)
                WHERE transaction_hash != '';
            """
        )
        row = connection.execute(
            "SELECT value FROM polymarket_live_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            populated = any(
                int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"  # nosec B608
                    ).fetchone()[0]
                )
                for table in (
                    "polymarket_live_orders",
                    "polymarket_live_fills",
                    "polymarket_live_audit",
                    "polymarket_live_redemptions",
                )
            )
            if populated:
                raise PolymarketLiveError("populated live ledger has no schema version")
            connection.execute(
                """
                INSERT INTO polymarket_live_metadata (key, value)
                VALUES ('schema_version', ?)
                """,
                [POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION],
            )
            return
        version = str(row[0])
        if version == _POLYMARKET_LIVE_LEDGER_V1:
            PolymarketLiveOrderLedger._migrate_v1_fill_accounting(connection)
            version = _POLYMARKET_LIVE_LEDGER_V2
        if version == _POLYMARKET_LIVE_LEDGER_V2:
            PolymarketLiveOrderLedger._migrate_v2_redemption_accounting(connection)
            return
        if version != POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION:
            raise PolymarketLiveError("live ledger schema differs")

    @staticmethod
    def _fill_row_payload_v1(row: Mapping[str, object]) -> dict[str, object]:
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

    @staticmethod
    def _migrate_v1_fill_accounting(connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT * FROM polymarket_live_fills").fetchall()
        for row in rows:
            expected = _canonical_sha256(
                PolymarketLiveOrderLedger._fill_row_payload_v1(row)
            )
            if str(row["fill_sha256"]) != expected:
                raise PolymarketLiveError("legacy live fill snapshot hash differs")
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(polymarket_live_fills)"
            ).fetchall()
        }
        additions = (
            ("role", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
            ("reported_fee_rate_bps", "INTEGER NOT NULL DEFAULT -1"),
            ("fee_rate", "TEXT NOT NULL DEFAULT ''"),
            ("fee_exponent", "INTEGER NOT NULL DEFAULT 0"),
            ("fee_quote", "TEXT NOT NULL DEFAULT ''"),
            ("fee_schedule_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("transaction_hash", "TEXT NOT NULL DEFAULT ''"),
            ("accounting_state", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
        )
        try:
            connection.execute("BEGIN IMMEDIATE")
            for name, definition in additions:
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE polymarket_live_fills "
                        f"ADD COLUMN {name} {definition}"
                    )
            migrated = connection.execute(
                "SELECT * FROM polymarket_live_fills"
            ).fetchall()
            for row in migrated:
                connection.execute(
                    """
                    UPDATE polymarket_live_fills SET fill_sha256 = ?
                    WHERE trade_id = ? AND order_id = ?
                    """,
                    [
                        _canonical_sha256(
                            PolymarketLiveOrderLedger._fill_row_payload(row)
                        ),
                        str(row["trade_id"]),
                        str(row["order_id"]),
                    ],
                )
            updated = connection.execute(
                """
                UPDATE polymarket_live_metadata SET value = ?
                WHERE key = 'schema_version' AND value = ?
                """,
                [
                    _POLYMARKET_LIVE_LEDGER_V2,
                    _POLYMARKET_LIVE_LEDGER_V1,
                ],
            )
            if updated.rowcount != 1:
                raise PolymarketLiveError(
                    "legacy live ledger version changed during migration"
                )
            for row in connection.execute(
                "SELECT * FROM polymarket_live_fills"
            ).fetchall():
                PolymarketLiveOrderLedger._verify_fill_row(row)
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _redemption_row_payload_v2(
        row: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "redemption_id": str(row["redemption_id"]),
            "condition_id": str(row["condition_id"]),
            "attempt": int(row["attempt"]),
            "inventory_json": str(row["inventory_json"]),
            "preflight_json": str(row["preflight_json"]),
            "state": str(row["state"]),
            "transaction_id": str(row["transaction_id"]),
            "transaction_hash": str(row["transaction_hash"]),
            "failure_code": str(row["failure_code"]),
            "created_at_ms": int(row["created_at_ms"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }

    @staticmethod
    def _migrate_v2_redemption_accounting(
        connection: sqlite3.Connection,
    ) -> None:
        rows = connection.execute(
            "SELECT * FROM polymarket_live_redemptions"
        ).fetchall()
        for row in rows:
            expected = _canonical_sha256(
                PolymarketLiveOrderLedger._redemption_row_payload_v2(row)
            )
            if str(row["record_sha256"]) != expected:
                raise PolymarketLiveError(
                    "legacy live redemption snapshot hash differs"
                )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(polymarket_live_redemptions)"
            ).fetchall()
        }
        additions = (
            ("payout_quote", "TEXT NOT NULL DEFAULT '0'"),
            ("payout_proof_sha256", "TEXT NOT NULL DEFAULT ''"),
            (
                "payout_accounting_state",
                "TEXT NOT NULL DEFAULT 'UNKNOWN'",
            ),
        )
        try:
            connection.execute("BEGIN IMMEDIATE")
            for name, definition in additions:
                if name not in columns:
                    connection.execute(
                        "ALTER TABLE polymarket_live_redemptions "
                        f"ADD COLUMN {name} {definition}"
                    )
            migrated = connection.execute(
                "SELECT * FROM polymarket_live_redemptions"
            ).fetchall()
            for row in migrated:
                connection.execute(
                    """
                    UPDATE polymarket_live_redemptions SET record_sha256 = ?
                    WHERE redemption_id = ?
                    """,
                    [
                        _canonical_sha256(
                            PolymarketLiveOrderLedger._redemption_row_payload(row)
                        ),
                        str(row["redemption_id"]),
                    ],
                )
            updated = connection.execute(
                """
                UPDATE polymarket_live_metadata SET value = ?
                WHERE key = 'schema_version' AND value = ?
                """,
                [
                    POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION,
                    _POLYMARKET_LIVE_LEDGER_V2,
                ],
            )
            if updated.rowcount != 1:
                raise PolymarketLiveError(
                    "legacy live ledger version changed during migration"
                )
            for row in connection.execute(
                "SELECT * FROM polymarket_live_redemptions"
            ).fetchall():
                PolymarketLiveOrderLedger._verify_redemption_row(row)
                PolymarketLiveOrderLedger._redemption_record(row)
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _order_row_payload(
        row: Mapping[str, object] | sqlite3.Row,
    ) -> dict[str, object]:
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
    def _verify_order_row(
        cls,
        row: Mapping[str, object] | sqlite3.Row,
    ) -> None:
        if str(row["record_sha256"]) != _canonical_sha256(cls._order_row_payload(row)):
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
            "role": str(row["role"]),
            "reported_fee_rate_bps": int(row["reported_fee_rate_bps"]),
            "fee_rate": str(row["fee_rate"]),
            "fee_exponent": int(row["fee_exponent"]),
            "fee_quote": str(row["fee_quote"]),
            "fee_schedule_sha256": str(row["fee_schedule_sha256"]),
            "transaction_hash": str(row["transaction_hash"]),
            "accounting_state": str(row["accounting_state"]),
        }

    @classmethod
    def _verify_fill_row(cls, row: Mapping[str, object]) -> None:
        if str(row["fill_sha256"]) != _canonical_sha256(cls._fill_row_payload(row)):
            raise PolymarketLiveError("live fill snapshot hash differs")
        accounting_state = str(row["accounting_state"])
        role = str(row["role"])
        reported_fee_rate_bps = int(row["reported_fee_rate_bps"])
        transaction_hash = str(row["transaction_hash"])
        if (
            accounting_state not in {"UNKNOWN", "VERIFIED"}
            or role not in {"UNKNOWN", "MAKER", "TAKER"}
            or not -1 <= reported_fee_rate_bps <= 10_000
            or transaction_hash
            and _BYTES32.fullmatch(transaction_hash) is None
        ):
            raise PolymarketLiveError("stored fill accounting metadata is invalid")
        if accounting_state == "UNKNOWN":
            if (
                str(row["fee_rate"])
                or int(row["fee_exponent"])
                or str(row["fee_quote"])
                or str(row["fee_schedule_sha256"])
            ):
                raise PolymarketLiveError(
                    "unverified fill carries fee accounting values"
                )
            return
        if role == "UNKNOWN":
            raise PolymarketLiveError("verified fill role is unknown")
        try:
            PolymarketRemoteFill(
                trade_id=str(row["trade_id"]),
                order_id=str(row["order_id"]),
                market_id=str(row["market_id"]),
                token_id=str(row["token_id"]),
                side=str(row["side"]),
                quantity=_decimal(
                    row["quantity"],
                    name="stored fill quantity",
                    positive=True,
                ),
                price=_decimal(
                    row["price"],
                    name="stored fill price",
                    positive=True,
                ),
                status=str(row["status"]),
                observed_at_ms=int(row["observed_at_ms"]),
                role=role,
                reported_fee_rate_bps=reported_fee_rate_bps,
                fee_rate=_decimal(
                    row["fee_rate"],
                    name="stored fill fee rate",
                    nonnegative=True,
                ),
                fee_exponent=int(row["fee_exponent"]),
                fee_quote=_decimal(
                    row["fee_quote"],
                    name="stored fill fee quote",
                    nonnegative=True,
                ),
                fee_schedule_sha256=str(row["fee_schedule_sha256"]),
                transaction_hash=transaction_hash,
            )
        except ValueError as exc:
            raise PolymarketLiveError(
                "stored verified fill accounting is invalid"
            ) from exc

    @staticmethod
    def _redemption_row_payload(
        row: Mapping[str, object] | sqlite3.Row,
    ) -> dict[str, object]:
        return {
            "redemption_id": str(row["redemption_id"]),
            "condition_id": str(row["condition_id"]),
            "attempt": int(row["attempt"]),
            "inventory_json": str(row["inventory_json"]),
            "preflight_json": str(row["preflight_json"]),
            "state": str(row["state"]),
            "transaction_id": str(row["transaction_id"]),
            "transaction_hash": str(row["transaction_hash"]),
            "failure_code": str(row["failure_code"]),
            "created_at_ms": int(row["created_at_ms"]),
            "updated_at_ms": int(row["updated_at_ms"]),
            "payout_quote": str(row["payout_quote"]),
            "payout_proof_sha256": str(row["payout_proof_sha256"]),
            "payout_accounting_state": str(row["payout_accounting_state"]),
        }

    @classmethod
    def _verify_redemption_row(
        cls,
        row: Mapping[str, object] | sqlite3.Row,
    ) -> None:
        if str(row["record_sha256"]) != _canonical_sha256(
            cls._redemption_row_payload(row)
        ):
            raise PolymarketLiveError("live redemption snapshot hash differs")

    @classmethod
    def _write_redemption_row_hash(
        cls,
        connection: sqlite3.Connection,
        redemption_id: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT * FROM polymarket_live_redemptions
            WHERE redemption_id = ?
            """,
            [redemption_id],
        ).fetchone()
        if row is None:
            raise KeyError(redemption_id)
        connection.execute(
            """
            UPDATE polymarket_live_redemptions SET record_sha256 = ?
            WHERE redemption_id = ?
            """,
            [_canonical_sha256(cls._redemption_row_payload(row)), redemption_id],
        )

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

    def reserve(
        self, prepared: PolymarketPreparedOrder, *, observed_at_ms: int
    ) -> None:
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
                    raise PolymarketLiveBlocked(
                        "intent_id was already bound differently"
                    )
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
                SELECT intent_id, market_id, token_id, side, quantity, limit_price
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
            limit_price = _decimal(
                order["limit_price"],
                name="signed limit price",
                positive=True,
            )
            violates_limit = (
                fill.side == "BUY"
                and fill.price > limit_price
                or fill.side == "SELL"
                and fill.price < limit_price
            )
            fee_rate_text = ""
            fee_exponent = 0
            fee_quote_text = ""
            if fill.accounting_verified:
                assert fill.fee_rate is not None
                assert fill.fee_exponent is not None
                assert fill.fee_quote is not None
                fee_rate_text = format(fill.fee_rate, "f")
                fee_exponent = fill.fee_exponent
                fee_quote_text = format(fill.fee_quote, "f")
            payload: dict[str, object] = {
                "trade_id": fill.trade_id,
                "order_id": fill.order_id,
                "market_id": fill.market_id,
                "token_id": fill.token_id,
                "side": fill.side,
                "quantity": format(fill.quantity, "f"),
                "price": format(fill.price, "f"),
                "status": fill.status,
                "observed_at_ms": fill.observed_at_ms,
                "role": fill.role,
                "reported_fee_rate_bps": fill.reported_fee_rate_bps,
                "fee_rate": fee_rate_text,
                "fee_exponent": fee_exponent,
                "fee_quote": fee_quote_text,
                "fee_schedule_sha256": fill.fee_schedule_sha256,
                "transaction_hash": fill.transaction_hash,
                "accounting_state": (
                    "VERIFIED" if fill.accounting_verified else "UNKNOWN"
                ),
            }
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
                if violates_limit:
                    raise PolymarketLiveBlocked(
                        "fill price violates the signed limit price"
                    )
                prior_role = str(existing["role"])
                if prior_role not in {"UNKNOWN", fill.role}:
                    raise PolymarketLiveBlocked("existing fill role differs")
                prior_reported_fee_rate_bps = int(existing["reported_fee_rate_bps"])
                if (
                    prior_reported_fee_rate_bps >= 0
                    and fill.reported_fee_rate_bps >= 0
                    and prior_reported_fee_rate_bps != fill.reported_fee_rate_bps
                ):
                    raise PolymarketLiveBlocked(
                        "existing reported fill fee rate differs"
                    )
                prior_transaction_hash = str(existing["transaction_hash"])
                if (
                    prior_transaction_hash
                    and fill.transaction_hash
                    and prior_transaction_hash != fill.transaction_hash
                ):
                    raise PolymarketLiveBlocked(
                        "existing fill transaction hash differs"
                    )
                prior_accounting_state = str(existing["accounting_state"])
                if prior_accounting_state not in {"UNKNOWN", "VERIFIED"}:
                    raise PolymarketLiveError("stored fill accounting state is invalid")
                if prior_accounting_state == "VERIFIED":
                    prior_fee_economics = (
                        str(existing["fee_rate"]),
                        int(existing["fee_exponent"]),
                        str(existing["fee_quote"]),
                        str(existing["fee_schedule_sha256"]),
                    )
                    if fill.accounting_verified and prior_fee_economics != (
                        fee_rate_text,
                        fee_exponent,
                        fee_quote_text,
                        fill.fee_schedule_sha256,
                    ):
                        raise PolymarketLiveBlocked(
                            "existing fill fee accounting differs"
                        )
                    payload.update(
                        {
                            "fee_rate": prior_fee_economics[0],
                            "fee_exponent": prior_fee_economics[1],
                            "fee_quote": prior_fee_economics[2],
                            "fee_schedule_sha256": prior_fee_economics[3],
                            "accounting_state": "VERIFIED",
                        }
                    )
                payload.update(
                    {
                        "role": (fill.role if prior_role == "UNKNOWN" else prior_role),
                        "reported_fee_rate_bps": (
                            fill.reported_fee_rate_bps
                            if prior_reported_fee_rate_bps < 0
                            else prior_reported_fee_rate_bps
                        ),
                        "transaction_hash": (
                            prior_transaction_hash or fill.transaction_hash
                        ),
                        "observed_at_ms": max(
                            int(existing["observed_at_ms"]),
                            fill.observed_at_ms,
                        ),
                    }
                )
                prior_status = str(existing["status"])
                if prior_status in _FILL_TERMINAL_STATUSES:
                    if fill.status != prior_status:
                        raise PolymarketLiveBlocked("terminal fill status differs")
                    payload["status"] = prior_status
                else:
                    allowed = {
                        "MATCHED_NOT_BROADCASTED": {
                            "MATCHED_NOT_BROADCASTED",
                            "MATCHED",
                            "MINED",
                            "CONFIRMED",
                            "RETRYING",
                            "FAILED",
                        },
                        "MATCHED": {
                            "MATCHED",
                            "MINED",
                            "CONFIRMED",
                            "RETRYING",
                            "FAILED",
                        },
                        "MINED": {"MINED", "CONFIRMED", "RETRYING", "FAILED"},
                        "RETRYING": {"RETRYING", "MINED", "CONFIRMED", "FAILED"},
                    }
                    if fill.status not in allowed.get(prior_status, set()):
                        raise PolymarketLiveBlocked("fill status regressed")
                digest = _canonical_sha256(payload)
                if str(existing["fill_sha256"]) == digest:
                    connection.execute("COMMIT")
                    return
                connection.execute(
                    """
                    UPDATE polymarket_live_fills
                    SET status = ?, observed_at_ms = ?, role = ?,
                        reported_fee_rate_bps = ?, fee_rate = ?,
                        fee_exponent = ?, fee_quote = ?,
                        fee_schedule_sha256 = ?, transaction_hash = ?,
                        accounting_state = ?, fill_sha256 = ?
                    WHERE trade_id = ? AND order_id = ?
                    """,
                    [
                        payload["status"],
                        payload["observed_at_ms"],
                        payload["role"],
                        payload["reported_fee_rate_bps"],
                        payload["fee_rate"],
                        payload["fee_exponent"],
                        payload["fee_quote"],
                        payload["fee_schedule_sha256"],
                        payload["transaction_hash"],
                        payload["accounting_state"],
                        digest,
                        fill.trade_id,
                        fill.order_id,
                    ],
                )
            else:
                if violates_limit:
                    raise PolymarketLiveBlocked(
                        "fill price violates the signed limit price"
                    )
                digest = _canonical_sha256(payload)
                connection.execute(
                    """
                    INSERT INTO polymarket_live_fills (
                        trade_id, order_id, market_id, token_id, side, quantity,
                        price, status, observed_at_ms, role,
                        reported_fee_rate_bps, fee_rate, fee_exponent, fee_quote,
                        fee_schedule_sha256, transaction_hash, accounting_state,
                        fill_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        fill.trade_id,
                        fill.order_id,
                        fill.market_id,
                        fill.token_id,
                        fill.side,
                        format(fill.quantity, "f"),
                        format(fill.price, "f"),
                        payload["status"],
                        payload["observed_at_ms"],
                        payload["role"],
                        payload["reported_fee_rate_bps"],
                        payload["fee_rate"],
                        payload["fee_exponent"],
                        payload["fee_quote"],
                        payload["fee_schedule_sha256"],
                        payload["transaction_hash"],
                        payload["accounting_state"],
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

    def revision(self) -> PolymarketLedgerRevision:
        """Return the audit-chain tip without rescanning immutable history."""

        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT sequence, record_sha256
                FROM polymarket_live_audit
                ORDER BY sequence DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return PolymarketLedgerRevision(0, _ZERO_SHA256)
            return PolymarketLedgerRevision(
                sequence=int(row["sequence"]),
                record_sha256=str(row["record_sha256"]),
            )
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
                   OR accounting_state != 'VERIFIED'
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

    def unverified_fill_accounting_count(self) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM polymarket_live_fills
                WHERE status != 'FAILED' AND accounting_state != 'VERIFIED'
                """
            ).fetchone()
            return 0 if row is None else int(row[0])
        finally:
            connection.close()

    @staticmethod
    def _redemption_inventory_json(
        condition_id: str,
        inventory: Sequence[PolymarketOwnedInventory],
    ) -> str:
        condition = _condition_id(condition_id)
        items = tuple(
            sorted(
                inventory,
                key=lambda item: (item.market_id, item.token_id),
            )
        )
        if not items:
            raise ValueError("redemption inventory cannot be empty")
        if any(item.market_id != condition or item.provisional for item in items):
            raise ValueError(
                "redemption requires confirmed inventory for one condition"
            )
        if len({item.token_id for item in items}) != len(items):
            raise ValueError("redemption inventory contains duplicate tokens")
        return _canonical_json(
            [
                {
                    "market_id": item.market_id,
                    "token_id": item.token_id,
                    "quantity": format(item.quantity, "f"),
                }
                for item in items
            ]
        )

    @staticmethod
    def _parse_redemption_inventory(
        condition_id: str,
        inventory_json: str,
    ) -> tuple[PolymarketOwnedInventory, ...]:
        try:
            payload = json.loads(inventory_json)
        except json.JSONDecodeError as exc:
            raise PolymarketLiveError("redemption inventory JSON is invalid") from exc
        if not isinstance(payload, list) or _canonical_json(payload) != inventory_json:
            raise PolymarketLiveError("redemption inventory JSON is not canonical")
        output: list[PolymarketOwnedInventory] = []
        for raw in payload:
            if not isinstance(raw, Mapping):
                raise PolymarketLiveError("redemption inventory row is invalid")
            item = PolymarketOwnedInventory(
                market_id=_condition_id(raw.get("market_id")),
                token_id=_token_id(raw.get("token_id")),
                quantity=_decimal(
                    raw.get("quantity"),
                    name="redemption quantity",
                    positive=True,
                ),
                provisional=False,
            )
            if item.market_id != condition_id:
                raise PolymarketLiveError("redemption inventory condition differs")
            output.append(item)
        if not output or len({item.token_id for item in output}) != len(output):
            raise PolymarketLiveError("redemption inventory token set is invalid")
        return tuple(output)

    @classmethod
    def _redemption_record(cls, row: sqlite3.Row) -> PolymarketRedemptionRecord:
        cls._verify_redemption_row(row)
        redemption_id = _identifier(
            row["redemption_id"],
            name="redemption_id",
        )
        condition_id = _condition_id(row["condition_id"])
        attempt = int(row["attempt"])
        expected_id = f"redemption:{condition_id[2:]}:{attempt:06d}"
        if attempt <= 0 or redemption_id != expected_id:
            raise PolymarketLiveError("redemption attempt identity is invalid")
        state = str(row["state"])
        if state not in {
            "prepared",
            "submitting",
            "submitted",
            "unknown",
            "confirmed",
            "failed",
        }:
            raise PolymarketLiveError("redemption state is invalid")
        transaction_id = str(row["transaction_id"])
        if transaction_id:
            transaction_id = _identifier(
                transaction_id,
                name="transaction_id",
            )
        transaction_hash = str(row["transaction_hash"]).lower()
        if transaction_hash:
            transaction_hash = _order_id(
                transaction_hash,
                name="transaction_hash",
            )
        created_at_ms = int(row["created_at_ms"])
        updated_at_ms = int(row["updated_at_ms"])
        if created_at_ms <= 0 or updated_at_ms < created_at_ms:
            raise PolymarketLiveError("redemption chronology is invalid")
        preflight_json = str(row["preflight_json"])
        try:
            preflight = json.loads(preflight_json)
        except json.JSONDecodeError as exc:
            raise PolymarketLiveError("redemption preflight JSON is invalid") from exc
        if (
            not isinstance(preflight, Mapping)
            or _canonical_json(preflight) != preflight_json
        ):
            raise PolymarketLiveError("redemption preflight JSON is not canonical")
        try:
            payout_quote = _decimal(
                row["payout_quote"],
                name="redemption payout",
                nonnegative=True,
            )
        except ValueError as exc:
            raise PolymarketLiveError("redemption payout is invalid") from exc
        payout_proof_sha256 = str(row["payout_proof_sha256"]).lower()
        payout_accounting_state = str(row["payout_accounting_state"])
        if payout_accounting_state not in {"UNKNOWN", "VERIFIED"}:
            raise PolymarketLiveError("redemption payout accounting state is invalid")
        if payout_accounting_state == "UNKNOWN":
            if payout_quote != 0 or payout_proof_sha256:
                raise PolymarketLiveError(
                    "unverified redemption cannot carry payout accounting"
                )
        elif (
            state != "confirmed"
            or re.fullmatch(r"[0-9a-f]{64}", payout_proof_sha256) is None
        ):
            raise PolymarketLiveError("verified redemption payout proof is invalid")
        return PolymarketRedemptionRecord(
            redemption_id=redemption_id,
            condition_id=condition_id,
            attempt=attempt,
            inventory=cls._parse_redemption_inventory(
                condition_id,
                str(row["inventory_json"]),
            ),
            preflight_json=preflight_json,
            state=state,
            transaction_id=transaction_id,
            transaction_hash=transaction_hash,
            failure_code=str(row["failure_code"]),
            created_at_ms=created_at_ms,
            updated_at_ms=updated_at_ms,
            payout_quote=payout_quote,
            payout_proof_sha256=payout_proof_sha256,
            payout_accounting_state=payout_accounting_state,
        )

    def redemption_records(self) -> tuple[PolymarketRedemptionRecord, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                ORDER BY created_at_ms, condition_id, attempt
                """
            ).fetchall()
            return tuple(self._redemption_record(row) for row in rows)
        finally:
            connection.close()

    def unverified_redemption_accounting_count(self) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM polymarket_live_redemptions
                WHERE state = 'confirmed'
                  AND payout_accounting_state != 'VERIFIED'
                """
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    def reserve_redemption(
        self,
        condition_id: str,
        inventory: Sequence[PolymarketOwnedInventory],
        *,
        observed_at_ms: int,
        preflight: Mapping[str, object] | None = None,
    ) -> PolymarketRedemptionRecord:
        condition = _condition_id(condition_id)
        inventory_json = self._redemption_inventory_json(condition, inventory)
        preflight_json = _canonical_json(dict(preflight or {}))
        now = int(observed_at_ms)
        if now <= 0:
            raise ValueError("redemption observation time must be positive")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                WHERE condition_id = ?
                ORDER BY attempt DESC
                LIMIT 1
                """,
                [condition],
            ).fetchone()
            if existing is not None:
                self._verify_redemption_row(existing)
                if str(existing["inventory_json"]) != inventory_json:
                    raise PolymarketLiveBlocked(
                        "condition was already bound to different redemption inventory"
                    )
                if str(existing["state"]) != "failed":
                    connection.execute("COMMIT")
                    return self._redemption_record(existing)
                attempt = int(existing["attempt"]) + 1
            else:
                attempt = 1
            redemption_id = f"redemption:{condition[2:]}:{attempt:06d}"
            connection.execute(
                """
                INSERT INTO polymarket_live_redemptions (
                    redemption_id, condition_id, attempt, inventory_json,
                    preflight_json, state, transaction_id, transaction_hash, failure_code,
                    created_at_ms, updated_at_ms, payout_quote,
                    payout_proof_sha256, payout_accounting_state, record_sha256
                ) VALUES (
                    ?, ?, ?, ?, ?, 'prepared', '', '', '', ?, ?,
                    '0', '', 'UNKNOWN', ''
                )
                """,
                [
                    redemption_id,
                    condition,
                    attempt,
                    inventory_json,
                    preflight_json,
                    now,
                    now,
                ],
            )
            self._write_redemption_row_hash(connection, redemption_id)
            self._append_audit(
                connection,
                intent_id=redemption_id,
                event_type="redemption_prepared",
                payload={
                    "condition_id": condition,
                    "attempt": attempt,
                    "inventory_json": inventory_json,
                    "preflight_json": preflight_json,
                },
                observed_at_ms=now,
            )
            row = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                WHERE redemption_id = ?
                """,
                [redemption_id],
            ).fetchone()
            if row is None:
                raise PolymarketLiveError("redemption reservation disappeared")
            record = self._redemption_record(row)
            connection.execute("COMMIT")
            return record
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition_redemption(
        self,
        redemption_id: str,
        *,
        expected_states: Sequence[str],
        state: str,
        observed_at_ms: int,
        transaction_id: str | None = None,
        transaction_hash: str | None = None,
        failure_code: str = "",
        payout_quote: Decimal | str | None = None,
        payout_proof_sha256: str | None = None,
    ) -> PolymarketRedemptionRecord:
        normalized_id = _identifier(redemption_id, name="redemption_id")
        expected = tuple(str(value) for value in expected_states)
        if not expected:
            raise ValueError("expected redemption states cannot be empty")
        if state not in {
            "prepared",
            "submitting",
            "submitted",
            "unknown",
            "confirmed",
            "failed",
        }:
            raise ValueError("redemption state is invalid")
        now = int(observed_at_ms)
        if now <= 0:
            raise ValueError("redemption observation time must be positive")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                WHERE redemption_id = ?
                """,
                [normalized_id],
            ).fetchone()
            if row is None:
                raise KeyError(normalized_id)
            self._verify_redemption_row(row)
            if str(row["state"]) not in expected:
                raise PolymarketStateConflict(
                    f"redemption state {row['state']} does not permit {state}"
                )
            prior_state = str(row["state"])
            proof_requested = (
                payout_quote is not None or payout_proof_sha256 is not None
            )
            if proof_requested and (
                payout_quote is None or payout_proof_sha256 is None
            ):
                raise ValueError(
                    "redemption payout and proof must be supplied together"
                )
            prior_accounting = str(row["payout_accounting_state"])
            prior_payout = _decimal(
                row["payout_quote"],
                name="stored redemption payout",
                nonnegative=True,
            )
            prior_proof = str(row["payout_proof_sha256"]).lower()
            if proof_requested:
                resolved_payout = _decimal(
                    payout_quote,
                    name="redemption payout",
                    nonnegative=True,
                )
                resolved_proof = str(payout_proof_sha256).strip().lower()
                if re.fullmatch(r"[0-9a-f]{64}", resolved_proof) is None:
                    raise ValueError("redemption payout proof hash is invalid")
                if state != "confirmed":
                    raise ValueError(
                        "redemption payout can only accompany confirmation"
                    )
                redeemed_inventory = self._parse_redemption_inventory(
                    str(row["condition_id"]),
                    str(row["inventory_json"]),
                )
                if resolved_payout > sum(
                    (item.quantity for item in redeemed_inventory),
                    start=Decimal("0"),
                ):
                    raise PolymarketLiveBlocked(
                        "redemption payout exceeds reserved inventory"
                    )
                if prior_accounting == "VERIFIED" and (
                    resolved_payout != prior_payout or resolved_proof != prior_proof
                ):
                    raise PolymarketLiveBlocked(
                        "verified redemption payout cannot change"
                    )
                resolved_accounting = "VERIFIED"
            else:
                resolved_payout = prior_payout
                resolved_proof = prior_proof
                resolved_accounting = prior_accounting
            accounting_upgrade = (
                prior_state == "confirmed"
                and state == "confirmed"
                and prior_accounting == "UNKNOWN"
                and resolved_accounting == "VERIFIED"
            )
            accounting_idempotent = (
                prior_state == "confirmed"
                and state == "confirmed"
                and prior_accounting == "VERIFIED"
                and resolved_accounting == "VERIFIED"
                and resolved_payout == prior_payout
                and resolved_proof == prior_proof
            )
            if (
                state not in _REDEMPTION_TRANSITIONS[prior_state]
                and not accounting_upgrade
                and not accounting_idempotent
            ):
                raise PolymarketStateConflict(
                    f"redemption transition {prior_state} -> {state} is invalid"
                )
            if now < int(row["updated_at_ms"]):
                raise ValueError("redemption observation time moved backwards")
            resolved_id = (
                str(row["transaction_id"])
                if transaction_id is None
                else str(transaction_id).strip()
            )
            resolved_hash = (
                str(row["transaction_hash"])
                if transaction_hash is None
                else str(transaction_hash).strip().lower()
            )
            if resolved_id:
                resolved_id = _identifier(resolved_id, name="transaction_id")
            if resolved_hash:
                resolved_hash = _order_id(
                    resolved_hash,
                    name="transaction_hash",
                )
            prior_id = str(row["transaction_id"])
            prior_hash = str(row["transaction_hash"])
            if prior_id and resolved_id != prior_id:
                raise PolymarketLiveBlocked("redemption transaction ID cannot change")
            if prior_hash and resolved_hash != prior_hash:
                raise PolymarketLiveBlocked("redemption transaction hash cannot change")
            duplicate = connection.execute(
                """
                SELECT transaction_id, transaction_hash
                FROM polymarket_live_redemptions
                WHERE redemption_id != ?
                  AND (
                    (? != '' AND transaction_id = ?)
                    OR (? != '' AND transaction_hash = ?)
                  )
                LIMIT 1
                """,
                [
                    normalized_id,
                    resolved_id,
                    resolved_id,
                    resolved_hash,
                    resolved_hash,
                ],
            ).fetchone()
            if duplicate is not None:
                raise PolymarketLiveBlocked(
                    "redemption transaction identity was already used"
                )
            if state in {"submitted", "confirmed"} and not (
                resolved_id or resolved_hash
            ):
                raise ValueError("submitted redemption lacks transaction identity")
            if state == "confirmed" and not resolved_hash:
                raise ValueError("confirmed redemption lacks a transaction hash")
            if state == "confirmed" and resolved_accounting != "VERIFIED":
                raise ValueError(
                    "confirmed redemption lacks verified payout accounting"
                )
            resolved_failure = str(failure_code or "").strip()
            if len(resolved_failure) > 256:
                raise ValueError("redemption failure code is too long")
            connection.execute(
                """
                UPDATE polymarket_live_redemptions
                SET state = ?, transaction_id = ?, transaction_hash = ?,
                    failure_code = ?, updated_at_ms = ?, payout_quote = ?,
                    payout_proof_sha256 = ?, payout_accounting_state = ?
                WHERE redemption_id = ?
                """,
                [
                    state,
                    resolved_id,
                    resolved_hash,
                    resolved_failure,
                    now,
                    format(resolved_payout, "f"),
                    resolved_proof,
                    resolved_accounting,
                    normalized_id,
                ],
            )
            self._write_redemption_row_hash(connection, normalized_id)
            self._append_audit(
                connection,
                intent_id=normalized_id,
                event_type=f"redemption_{state}",
                payload={
                    "prior_state": prior_state,
                    "transaction_id": resolved_id,
                    "transaction_hash": resolved_hash,
                    "failure_code": resolved_failure,
                    "payout_quote": format(resolved_payout, "f"),
                    "payout_proof_sha256": resolved_proof,
                    "payout_accounting_state": resolved_accounting,
                },
                observed_at_ms=now,
            )
            updated = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                WHERE redemption_id = ?
                """,
                [normalized_id],
            ).fetchone()
            if updated is None:
                raise PolymarketLiveError("redemption transition disappeared")
            record = self._redemption_record(updated)
            connection.execute("COMMIT")
            return record
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def owned_lots(self) -> tuple[PolymarketOwnedLot, ...]:
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
        records = self.records()
        by_intent = {record.intent.intent_id: record for record in records}
        fills_by_order: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            self._verify_fill_row(row)
            fills_by_order.setdefault(str(row["order_id"]), []).append(row)
        quantities: dict[str, Decimal] = {}
        reserved: dict[str, Decimal] = {}
        provisional: dict[str, bool] = {}
        for record in records:
            if record.intent.side != "BUY":
                continue
            fill_rows = fills_by_order.get(record.expected_order_id, [])
            quantity = sum(
                (
                    _decimal(row["quantity"], name="fill quantity", positive=True)
                    for row in fill_rows
                ),
                Decimal("0"),
            )
            if quantity > 0:
                quantities[record.intent.intent_id] = quantity
                reserved[record.intent.intent_id] = Decimal("0")
                provisional[record.intent.intent_id] = any(
                    str(row["status"]) != "CONFIRMED" for row in fill_rows
                )
        for record in records:
            if not record.intent.closing_only:
                continue
            parent_id = record.intent.parent_intent_id
            parent = by_intent.get(parent_id)
            if (
                parent is None
                or parent.intent.side != "BUY"
                or parent.intent.closing_only
                or parent.intent.bot_id != record.intent.bot_id
                or parent.intent.market_id != record.intent.market_id
                or parent.intent.token_id != record.intent.token_id
            ):
                raise PolymarketLiveError("closing order parent ownership is invalid")
            fill_rows = fills_by_order.get(record.expected_order_id, [])
            sold = sum(
                (
                    _decimal(row["quantity"], name="fill quantity", positive=True)
                    for row in fill_rows
                ),
                Decimal("0"),
            )
            if sold:
                quantities[parent_id] = quantities.get(parent_id, Decimal("0")) - sold
                if any(str(row["status"]) != "CONFIRMED" for row in fill_rows):
                    provisional[parent_id] = True
            if record.state in _OPEN_STATES:
                outstanding = record.intent.quantity - sold
                if outstanding < 0:
                    raise PolymarketLiveError(
                        "closing fills exceed the closing intent quantity"
                    )
                reserved[parent_id] = (
                    reserved.get(parent_id, Decimal("0")) + outstanding
                )
        for redemption in self.redemption_records():
            if redemption.state != "confirmed":
                continue
            for item in redemption.inventory:
                remaining = item.quantity
                candidates = (
                    record
                    for record in records
                    if record.intent.side == "BUY"
                    and record.intent.market_id == item.market_id
                    and record.intent.token_id == item.token_id
                )
                for parent in candidates:
                    parent_id = parent.intent.intent_id
                    available = max(
                        Decimal("0"),
                        quantities.get(parent_id, Decimal("0")),
                    )
                    consumed = min(available, remaining)
                    quantities[parent_id] = available - consumed
                    remaining -= consumed
                    if remaining <= _POSITION_TOLERANCE:
                        remaining = Decimal("0")
                        break
                if remaining > _POSITION_TOLERANCE:
                    raise PolymarketLiveError(
                        "confirmed redemption exceeds owned inventory"
                    )
        output: list[PolymarketOwnedLot] = []
        for record in records:
            if record.intent.side != "BUY":
                continue
            parent_id = record.intent.intent_id
            quantity = quantities.get(parent_id, Decimal("0"))
            held = reserved.get(parent_id, Decimal("0"))
            if quantity < -_POSITION_TOLERANCE:
                raise PolymarketLiveError("bot-owned lot became negative")
            quantity = max(Decimal("0"), quantity)
            if held > quantity + _POSITION_TOLERANCE:
                raise PolymarketLiveError(
                    "reserved close quantity exceeds bot-owned lot"
                )
            if quantity > _POSITION_TOLERANCE:
                output.append(
                    PolymarketOwnedLot(
                        parent_intent_id=parent_id,
                        market_id=record.intent.market_id,
                        token_id=record.intent.token_id,
                        quantity=quantity,
                        reserved_close_quantity=min(held, quantity),
                        provisional=provisional.get(parent_id, False),
                    )
                )
        return tuple(output)

    def owned_inventory(self) -> tuple[PolymarketOwnedInventory, ...]:
        quantities: dict[tuple[str, str], Decimal] = {}
        provisional: dict[tuple[str, str], bool] = {}
        for lot in self.owned_lots():
            key = (lot.market_id, lot.token_id)
            quantities[key] = quantities.get(key, Decimal("0")) + lot.quantity
            provisional[key] = provisional.get(key, False) or lot.provisional
        return tuple(
            PolymarketOwnedInventory(
                market_id=market_id,
                token_id=token_id,
                quantity=quantity,
                provisional=provisional[(market_id, token_id)],
            )
            for (market_id, token_id), quantity in sorted(quantities.items())
        )

    def realized_pnl_events(self) -> tuple[PolymarketRealizedPnlEvent, ...]:
        """Rebuild fee-inclusive realized PnL from exact owned evidence."""

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            order_rows = connection.execute(
                """
                SELECT * FROM polymarket_live_orders
                ORDER BY created_at_ms, intent_id
                """
            ).fetchall()
            orders_by_id: dict[str, sqlite3.Row] = {}
            orders_by_intent: dict[str, sqlite3.Row] = {}
            for row in order_rows:
                self._verify_order_row(row)
                if str(row["state"]) in _OPEN_STATES:
                    raise PolymarketLiveBlocked(
                        "realized PnL accounting has an active or unknown order"
                    )
                orders_by_id[str(row["expected_order_id"])] = row
                orders_by_intent[str(row["intent_id"])] = row

            fill_rows = connection.execute(
                """
                SELECT * FROM polymarket_live_fills
                WHERE status != 'FAILED'
                ORDER BY observed_at_ms, trade_id, order_id
                """
            ).fetchall()
            parent_quantities: dict[str, Decimal] = {}
            parent_costs: dict[str, Decimal] = {}
            parent_consumed: dict[str, Decimal] = {}
            for row in fill_rows:
                self._verify_fill_row(row)
                if (
                    str(row["status"]) != "CONFIRMED"
                    or str(row["accounting_state"]) != "VERIFIED"
                ):
                    raise PolymarketLiveBlocked(
                        "realized PnL requires confirmed fee-verified fills"
                    )
                order = orders_by_id.get(str(row["order_id"]))
                if order is None:
                    raise PolymarketLiveError(
                        "realized PnL fill lacks its bot-owned order"
                    )
                if (
                    str(row["market_id"]) != str(order["market_id"])
                    or str(row["token_id"]) != str(order["token_id"])
                    or str(row["side"]) != str(order["side"])
                ):
                    raise PolymarketLiveError(
                        "realized PnL fill identity differs from its order"
                    )
                if str(order["side"]) != "BUY":
                    continue
                quantity = _decimal(
                    row["quantity"],
                    name="realized buy quantity",
                    positive=True,
                )
                cost = _decimal(
                    row["price"],
                    name="realized buy price",
                    positive=True,
                ) * quantity + _decimal(
                    row["fee_quote"],
                    name="realized buy fee",
                    nonnegative=True,
                )
                parent = str(order["intent_id"])
                parent_quantities[parent] = (
                    parent_quantities.get(parent, Decimal("0")) + quantity
                )
                parent_costs[parent] = parent_costs.get(parent, Decimal("0")) + cost

            events: list[PolymarketRealizedPnlEvent] = []
            for row in fill_rows:
                order = orders_by_id[str(row["order_id"])]
                if str(order["side"]) != "SELL":
                    continue
                parent = str(order["parent_intent_id"])
                parent_order = orders_by_intent.get(parent)
                if (
                    not parent
                    or parent_order is None
                    or str(parent_order["side"]) != "BUY"
                    or str(parent_order["market_id"]) != str(order["market_id"])
                    or str(parent_order["token_id"]) != str(order["token_id"])
                ):
                    raise PolymarketLiveError(
                        "realized close lacks its exact bot-owned parent"
                    )
                quantity = _decimal(
                    row["quantity"],
                    name="realized sell quantity",
                    positive=True,
                )
                acquired = parent_quantities.get(parent, Decimal("0"))
                consumed = parent_consumed.get(parent, Decimal("0"))
                if (
                    acquired <= 0
                    or consumed + quantity > acquired + _POSITION_TOLERANCE
                ):
                    raise PolymarketLiveError(
                        "realized close exceeds its bot-owned parent quantity"
                    )
                basis = parent_costs[parent] * quantity / acquired
                parent_consumed[parent] = consumed + quantity
                gross = (
                    _decimal(
                        row["price"],
                        name="realized sell price",
                        positive=True,
                    )
                    * quantity
                )
                fee = _decimal(
                    row["fee_quote"],
                    name="realized sell fee",
                    nonnegative=True,
                )
                if fee > gross:
                    raise PolymarketLiveError(
                        "realized sell fee exceeds gross proceeds"
                    )
                proceeds = gross - fee
                event_identity = {
                    "source": "sell_fill",
                    "trade_id": str(row["trade_id"]),
                    "order_id": str(row["order_id"]),
                }
                events.append(
                    PolymarketRealizedPnlEvent(
                        event_id=("sell:" + _canonical_sha256(event_identity)[:32]),
                        condition_id=str(order["market_id"]),
                        source="sell_fill",
                        observed_at_ms=int(row["observed_at_ms"]),
                        proceeds_quote=proceeds,
                        consumed_cost_basis_quote=basis,
                        pnl_quote=proceeds - basis,
                    )
                )

            redemption_rows = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                ORDER BY updated_at_ms, redemption_id
                """
            ).fetchall()
            for row in redemption_rows:
                redemption = self._redemption_record(row)
                if redemption.state in {
                    "prepared",
                    "submitting",
                    "submitted",
                    "unknown",
                }:
                    raise PolymarketLiveBlocked(
                        "realized PnL has an unresolved redemption"
                    )
                if redemption.state != "confirmed":
                    continue
                if redemption.payout_accounting_state != "VERIFIED":
                    raise PolymarketLiveBlocked(
                        "realized PnL has an unverified redemption"
                    )
                consumed_basis = Decimal("0")
                for item in redemption.inventory:
                    remaining = item.quantity
                    for parent_order in order_rows:
                        if (
                            str(parent_order["side"]) != "BUY"
                            or str(parent_order["market_id"]) != item.market_id
                            or str(parent_order["token_id"]) != item.token_id
                        ):
                            continue
                        parent = str(parent_order["intent_id"])
                        acquired = parent_quantities.get(parent, Decimal("0"))
                        consumed = parent_consumed.get(parent, Decimal("0"))
                        available = max(Decimal("0"), acquired - consumed)
                        selected = min(available, remaining)
                        if selected:
                            consumed_basis += parent_costs[parent] * selected / acquired
                            parent_consumed[parent] = consumed + selected
                            remaining -= selected
                        if remaining <= _POSITION_TOLERANCE:
                            remaining = Decimal("0")
                            break
                    if remaining:
                        raise PolymarketLiveError(
                            "realized redemption exceeds bot-owned cost basis"
                        )
                events.append(
                    PolymarketRealizedPnlEvent(
                        event_id=redemption.redemption_id,
                        condition_id=redemption.condition_id,
                        source="redemption",
                        observed_at_ms=redemption.updated_at_ms,
                        proceeds_quote=redemption.payout_quote,
                        consumed_cost_basis_quote=consumed_basis,
                        pnl_quote=redemption.payout_quote - consumed_basis,
                    )
                )
            connection.execute("COMMIT")
            return tuple(
                sorted(
                    events,
                    key=lambda item: (item.observed_at_ms, item.event_id),
                )
            )
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def condition_accounting(
        self,
        condition_id: str,
    ) -> PolymarketConditionAccounting:
        """Rebuild exact event downside from verified bot-owned evidence."""

        condition = _condition_id(condition_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            order_rows = connection.execute(
                """
                SELECT * FROM polymarket_live_orders
                WHERE market_id = ?
                ORDER BY created_at_ms, intent_id
                """,
                [condition],
            ).fetchall()
            by_order_id: dict[str, sqlite3.Row] = {}
            by_intent_id: dict[str, sqlite3.Row] = {}
            token_outcomes: dict[str, str] = {}
            for row in order_rows:
                self._verify_order_row(row)
                if str(row["state"]) in _OPEN_STATES:
                    raise PolymarketLiveBlocked(
                        "condition accounting has an active or unknown order"
                    )
                order_id = str(row["expected_order_id"])
                by_order_id[order_id] = row
                by_intent_id[str(row["intent_id"])] = row
                token = str(row["token_id"])
                outcome = str(row["outcome"])
                prior = token_outcomes.setdefault(token, outcome)
                if prior != outcome:
                    raise PolymarketLiveError(
                        "condition token maps to contradictory outcomes"
                    )

            fill_rows = connection.execute(
                """
                SELECT f.*
                FROM polymarket_live_fills AS f
                JOIN polymarket_live_orders AS o
                  ON o.expected_order_id = f.order_id
                WHERE o.market_id = ? AND f.status != 'FAILED'
                ORDER BY f.observed_at_ms, f.trade_id, f.order_id
                """,
                [condition],
            ).fetchall()
            gross_buy_cost = Decimal("0")
            gross_sell_proceeds = Decimal("0")
            quantities = {"Up": Decimal("0"), "Down": Decimal("0")}
            parent_buy_quantities: dict[str, Decimal] = {}
            parent_buy_costs: dict[str, Decimal] = {}
            parent_consumed_quantities: dict[str, Decimal] = {}
            for row in fill_rows:
                self._verify_fill_row(row)
                if (
                    str(row["status"]) != "CONFIRMED"
                    or str(row["accounting_state"]) != "VERIFIED"
                ):
                    raise PolymarketLiveBlocked(
                        "condition accounting requires confirmed fee-verified fills"
                    )
                order = by_order_id.get(str(row["order_id"]))
                if order is None:
                    raise PolymarketLiveError(
                        "condition fill lacks its bot-owned order"
                    )
                quantity = _decimal(
                    row["quantity"],
                    name="condition fill quantity",
                    positive=True,
                )
                price = _decimal(
                    row["price"],
                    name="condition fill price",
                    positive=True,
                )
                fee = _decimal(
                    row["fee_quote"],
                    name="condition fill fee",
                    nonnegative=True,
                )
                outcome = str(order["outcome"])
                side = str(order["side"])
                gross = price * quantity
                if side == "BUY":
                    fill_cost = gross + fee
                    gross_buy_cost += fill_cost
                    quantities[outcome] += quantity
                    parent_id = str(order["intent_id"])
                    parent_buy_quantities[parent_id] = (
                        parent_buy_quantities.get(parent_id, Decimal("0")) + quantity
                    )
                    parent_buy_costs[parent_id] = (
                        parent_buy_costs.get(parent_id, Decimal("0")) + fill_cost
                    )
                elif side == "SELL":
                    if fee > gross:
                        raise PolymarketLiveError(
                            "condition sell fee exceeds gross proceeds"
                        )
                    gross_sell_proceeds += gross - fee
                    quantities[outcome] -= quantity
                    parent_id = str(order["parent_intent_id"])
                    parent = by_intent_id.get(parent_id)
                    if (
                        not parent_id
                        or parent is None
                        or str(parent["side"]) != "BUY"
                        or str(parent["outcome"]) != outcome
                    ):
                        raise PolymarketLiveError(
                            "condition close lacks its bot-owned parent"
                        )
                    parent_consumed_quantities[parent_id] = (
                        parent_consumed_quantities.get(
                            parent_id,
                            Decimal("0"),
                        )
                        + quantity
                    )
                else:
                    raise PolymarketLiveError("condition fill side is invalid")

            redemption_rows = connection.execute(
                """
                SELECT * FROM polymarket_live_redemptions
                WHERE condition_id = ?
                ORDER BY attempt
                """,
                [condition],
            ).fetchall()
            redemption_payout = Decimal("0")
            for row in redemption_rows:
                redemption = self._redemption_record(row)
                if redemption.state in {
                    "prepared",
                    "submitting",
                    "submitted",
                    "unknown",
                }:
                    raise PolymarketLiveBlocked(
                        "condition accounting has an unresolved redemption"
                    )
                if redemption.state != "confirmed":
                    continue
                if redemption.payout_accounting_state != "VERIFIED":
                    raise PolymarketLiveBlocked(
                        "condition accounting has an unverified redemption"
                    )
                redemption_payout += redemption.payout_quote
                for item in redemption.inventory:
                    redeemed_outcome = token_outcomes.get(item.token_id)
                    if redeemed_outcome is None:
                        raise PolymarketLiveError(
                            "redeemed token lacks a bot-owned outcome"
                        )
                    quantities[redeemed_outcome] -= item.quantity
                    remaining = item.quantity
                    for parent in order_rows:
                        if (
                            str(parent["side"]) != "BUY"
                            or str(parent["token_id"]) != item.token_id
                        ):
                            continue
                        parent_id = str(parent["intent_id"])
                        acquired = parent_buy_quantities.get(
                            parent_id,
                            Decimal("0"),
                        )
                        consumed = parent_consumed_quantities.get(
                            parent_id,
                            Decimal("0"),
                        )
                        available = max(Decimal("0"), acquired - consumed)
                        selected = min(available, remaining)
                        if selected:
                            parent_consumed_quantities[parent_id] = consumed + selected
                            remaining -= selected
                        if remaining <= _POSITION_TOLERANCE:
                            remaining = Decimal("0")
                            break
                    if remaining:
                        raise PolymarketLiveError(
                            "redeemed quantity exceeds bot-owned cost basis"
                        )

            for outcome, quantity in tuple(quantities.items()):
                if quantity < -_POSITION_TOLERANCE:
                    raise PolymarketLiveError(
                        "condition accounting inventory became negative"
                    )
                quantities[outcome] = max(Decimal("0"), quantity)
            remaining_costs = {"Up": Decimal("0"), "Down": Decimal("0")}
            remaining_quantities = {"Up": Decimal("0"), "Down": Decimal("0")}
            for parent_id, acquired in parent_buy_quantities.items():
                consumed = parent_consumed_quantities.get(parent_id, Decimal("0"))
                if consumed > acquired + _POSITION_TOLERANCE:
                    raise PolymarketLiveError(
                        "condition close exceeds bot-owned cost basis"
                    )
                remaining = max(Decimal("0"), acquired - consumed)
                parent = by_intent_id[parent_id]
                outcome = str(parent["outcome"])
                remaining_quantities[outcome] += remaining
                remaining_costs[outcome] += (
                    parent_buy_costs[parent_id] * remaining / acquired
                )
            if any(
                abs(remaining_quantities[outcome] - quantities[outcome])
                > _POSITION_TOLERANCE
                for outcome in ("Up", "Down")
            ):
                raise PolymarketLiveError(
                    "condition inventory and cost basis do not reconcile"
                )
            result = PolymarketConditionAccounting(
                condition_id=condition,
                gross_buy_cost_quote=gross_buy_cost,
                gross_sell_proceeds_quote=gross_sell_proceeds,
                confirmed_redemption_payout_quote=redemption_payout,
                up_quantity=quantities["Up"],
                down_quantity=quantities["Down"],
                up_cost_basis_quote=remaining_costs["Up"],
                down_cost_basis_quote=remaining_costs["Down"],
                confirmed_fill_count=len(fill_rows),
            )
            connection.execute("COMMIT")
            return result
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


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
        checkpoint = (
            None
            if self.runtime_authority is None
            else self.runtime_authority.reconciliation_checkpoint()
        )
        try:
            venue = self.venue.preflight()
        except Exception as exc:
            if self.runtime_authority is not None:
                self.runtime_authority.note_reconciliation_failure(
                    exc.__class__.__name__, checkpoint=checkpoint
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
        if venue.wallet_address != self.venue.wallet_address:
            errors.append("wallet_identity_mismatch")
        result = self._reconcile_snapshot(
            open_orders=venue.open_orders,
            positions=venue.positions,
            base_errors=errors,
        )
        if self.runtime_authority is not None:
            self.runtime_authority.note_reconciliation(result, checkpoint=checkpoint)
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
    ) -> bool:
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
            return False
        return True

    def _remote_identity_matches(
        self,
        record: PolymarketLiveOrderRecord,
        remote: PolymarketRemoteOrder,
    ) -> bool:
        return (
            remote.order_id == record.expected_order_id
            and remote.market_id == record.intent.market_id
            and remote.token_id == record.intent.token_id
            and remote.maker_address == self.venue.wallet_address
            and remote.side == record.intent.side
            and remote.order_type == record.intent.order_type
            and remote.price == record.intent.limit_price
            and remote.original_quantity == record.intent.quantity
        )

    def _apply_remote_order_evidence(
        self,
        record: PolymarketLiveOrderRecord,
        remote: PolymarketRemoteOrder,
        *,
        fill_evidence: PolymarketOrderFillEvidence,
        observed_at_ms: int,
    ) -> bool:
        if not self._remote_identity_matches(record, remote):
            return self._transition_reconciled(
                record,
                state="unknown",
                observed_at_ms=observed_at_ms,
                remote_status=remote.status,
                matched_quantity=record.matched_quantity,
                failure_code="remote_order_identity_mismatch",
            )
        status = _remote_order_status(remote.status)
        if status in _REMOTE_ORDER_ACTIVE_STATUSES:
            if fill_evidence.quantity > remote.matched_quantity:
                return self._transition_reconciled(
                    record,
                    state="unknown",
                    observed_at_ms=observed_at_ms,
                    remote_status=remote.status,
                    matched_quantity=record.matched_quantity,
                    failure_code="remote_order_fill_quantity_mismatch",
                )
            return self._transition_reconciled(
                record,
                state="partial" if remote.matched_quantity > 0 else "live",
                observed_at_ms=observed_at_ms,
                remote_status=remote.status,
                matched_quantity=remote.matched_quantity,
            )
        if status not in _REMOTE_ORDER_TERMINAL_STATUSES:
            return self._transition_reconciled(
                record,
                state="unknown",
                observed_at_ms=observed_at_ms,
                remote_status=remote.status,
                matched_quantity=record.matched_quantity,
                failure_code="unsupported_remote_order_status",
            )
        if status == "INVALID":
            if remote.matched_quantity > 0 or fill_evidence.has_active_fills:
                return self._transition_reconciled(
                    record,
                    state="unknown",
                    observed_at_ms=observed_at_ms,
                    remote_status=remote.status,
                    matched_quantity=record.matched_quantity,
                    failure_code="invalid_order_has_fill_evidence",
                )
            return self._transition_reconciled(
                record,
                state="rejected",
                observed_at_ms=observed_at_ms,
                remote_status=remote.status,
                matched_quantity=Decimal("0"),
            )
        if remote.matched_quantity == 0:
            if fill_evidence.has_active_fills or status == "MATCHED":
                return self._transition_reconciled(
                    record,
                    state="unknown",
                    observed_at_ms=observed_at_ms,
                    remote_status=remote.status,
                    matched_quantity=record.matched_quantity,
                    failure_code="terminal_order_fill_quantity_mismatch",
                )
            terminal_state = (
                "expired" if status == "CANCELED_MARKET_RESOLVED" else "cancelled"
            )
            return self._transition_reconciled(
                record,
                state=terminal_state,
                observed_at_ms=observed_at_ms,
                remote_status=remote.status,
                matched_quantity=Decimal("0"),
            )
        if (
            not fill_evidence.has_active_fills
            or fill_evidence.quantity != remote.matched_quantity
        ):
            return self._transition_reconciled(
                record,
                state="matched_pending",
                observed_at_ms=observed_at_ms,
                remote_status=remote.status,
                matched_quantity=remote.matched_quantity,
                failure_code="terminal_order_awaiting_exact_fill_evidence",
            )
        return self._transition_reconciled(
            record,
            state=(
                "filled"
                if fill_evidence.all_active_fills_confirmed
                else "matched_pending"
            ),
            observed_at_ms=observed_at_ms,
            remote_status=remote.status,
            matched_quantity=remote.matched_quantity,
        )

    def reconcile(self) -> PolymarketReconciliation:
        checkpoint = (
            None
            if self.runtime_authority is None
            else self.runtime_authority.reconciliation_checkpoint()
        )
        now = int(time.time() * 1_000)
        targets = self.ledger.reconciliation_targets(observed_at_ms=now)
        owned_ids = tuple(record.expected_order_id for record in targets)
        market_ids = tuple(sorted({record.intent.market_id for record in targets}))
        for fill in self.venue.fills_for_orders(owned_ids, market_ids=market_ids):
            self.ledger.record_fill(fill)
        open_orders = self.venue.open_orders()
        remote_by_id = {order.order_id: order for order in open_orders}
        records = self.ledger.records()
        missing_active_ids = tuple(
            record.expected_order_id
            for record in records
            if record.state in _OPEN_STATES
            and record.expected_order_id not in remote_by_id
        )
        exact_orders = self.venue.orders_by_id(missing_active_ids)
        requested_exact_ids = set(missing_active_ids)
        if any(order.order_id not in requested_exact_ids for order in exact_orders):
            raise PolymarketLiveBlocked(
                "venue returned an unrequested exact-order record"
            )
        if len({order.order_id for order in exact_orders}) != len(exact_orders):
            raise PolymarketLiveBlocked("venue returned duplicate exact-order records")
        exact_by_id = {order.order_id: order for order in exact_orders}
        for record in records:
            remote = remote_by_id.get(record.expected_order_id)
            if remote is None:
                remote = exact_by_id.get(record.expected_order_id)
            if remote is not None:
                self._apply_remote_order_evidence(
                    record,
                    remote,
                    fill_evidence=self.ledger.order_fill_evidence(
                        record.expected_order_id
                    ),
                    observed_at_ms=now,
                )
                continue
            fill_evidence = self.ledger.order_fill_evidence(record.expected_order_id)
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
            self.runtime_authority.note_reconciliation(result, checkpoint=checkpoint)
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
            sorted(
                order.order_id
                for order in open_orders
                if order.order_id not in owned_ids
            )
        )
        if foreign_orders:
            errors.append("foreign_open_orders")
        owned_inventory = {
            item.token_id: item for item in self.ledger.owned_inventory()
        }
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
            record.intent.intent_id for record in records if record.blocks_new_exposure
        )
        if any(
            record.state in {"submitting", "unknown", "cancel_unknown"}
            for record in records
        ):
            errors.append("unknown_order_state")
        if any(item.provisional for item in owned_inventory.values()):
            errors.append("provisional_fill_state")
        if self.ledger.unverified_fill_accounting_count():
            errors.append("unverified_fill_accounting")
        if self.ledger.unverified_redemption_accounting_count():
            errors.append("unverified_redemption_accounting")
        if any(
            record.state in {"prepared", "submitting", "submitted", "unknown"}
            for record in self.ledger.redemption_records()
        ):
            errors.append("unknown_redemption_state")
        unique_errors = tuple(dict.fromkeys(errors))
        ownership_ok = (
            not foreign_orders and not foreign_positions and not missing_positions
        )
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

    def _owned_at_risk(self) -> tuple[Decimal, frozenset[str]]:
        markets = {lot.market_id for lot in self.ledger.owned_lots()}
        total = sum(
            (
                self.ledger.condition_accounting(market_id).maximum_loss_quote
                for market_id in markets
            ),
            start=Decimal("0"),
        )
        return total, frozenset(markets)

    def _assert_intent_current(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        observed_at_ms: int,
    ) -> None:
        if intent.created_at_ms - observed_at_ms > self.maximum_clock_skew_ms:
            raise PolymarketLiveBlocked("live intent timestamp is in the future")
        if (
            observed_at_ms - intent.created_at_ms
            > self.risk_limits.maximum_intent_age_ms
        ):
            raise PolymarketLiveBlocked("live intent exceeded its execution TTL")
        if intent.expires_at_ms <= observed_at_ms:
            raise PolymarketLiveBlocked("live intent has already expired")

    def _assert_order_dispatch_available(self) -> None:
        checker = getattr(self.venue, "assert_order_dispatch_available", None)
        if checker is None:
            return
        if not callable(checker):
            raise PolymarketLiveBlocked(
                "venue order-dispatch availability gate is invalid"
            )
        checker()

    def submit(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketLiveOrderRecord:
        now = int(time.time() * 1_000)
        self._assert_intent_current(intent, observed_at_ms=now)
        self._assert_order_dispatch_available()
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
                or parent.state not in _TERMINAL_STATES
            ):
                raise PolymarketLiveBlocked(
                    "closing intent differs from its bot-owned parent"
                )
            lots = {item.parent_intent_id: item for item in self.ledger.owned_lots()}
            lot = lots.get(intent.parent_intent_id)
            if (
                lot is None
                or lot.market_id != intent.market_id
                or lot.token_id != intent.token_id
                or lot.provisional
                or lot.available_quantity < intent.quantity
            ):
                raise PolymarketLiveBlocked(
                    "closing intent exceeds its confirmed unreserved bot-owned lot"
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
            current_at_risk, active_markets = self._owned_at_risk()
            if intent.market_id in active_markets or any(
                record.intent.market_id == intent.market_id
                for record in self.ledger.records()
            ):
                condition = self.ledger.condition_accounting(intent.market_id)
                up_quantity = condition.up_quantity
                down_quantity = condition.down_quantity
                if intent.outcome == "Up":
                    up_quantity += intent.quantity
                else:
                    down_quantity += intent.quantity
                projected_condition_loss = max(
                    Decimal("0"),
                    condition.net_cash_outflow_quote
                    + order_quote
                    - min(up_quantity, down_quantity),
                )
                projected_total_at_risk = current_at_risk
                if intent.market_id in active_markets:
                    projected_total_at_risk -= condition.maximum_loss_quote
                projected_total_at_risk += projected_condition_loss
            else:
                projected_total_at_risk = current_at_risk + order_quote
            if projected_total_at_risk > self.risk_limits.maximum_total_at_risk_quote:
                raise PolymarketLiveBlocked(
                    "live order exceeds the aggregate capital-at-risk ceiling"
                )
            if (
                intent.market_id not in active_markets
                and len(active_markets) >= self.risk_limits.maximum_active_markets
            ):
                raise PolymarketLiveBlocked(
                    "live order exceeds the active-market ceiling"
                )
            inventory = {
                item.token_id: item.quantity for item in self.ledger.owned_inventory()
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
            required = intent.limit_price * intent.quantity + intent.fee_reserve_quote
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
        submission_guard = nullcontext()
        if self.runtime_authority is not None:
            guard_factory = getattr(
                self.runtime_authority,
                "submission_guard",
                None,
            )
            if callable(guard_factory):
                submission_guard = guard_factory(closing_only=intent.closing_only)
        dispatch_started = False
        try:
            with submission_guard:
                if self.runtime_authority is not None:
                    self.runtime_authority.assert_submission_allowed(
                        closing_only=intent.closing_only
                    )
                self._assert_order_dispatch_available()
                now = int(time.time() * 1_000)
                self._assert_intent_current(intent, observed_at_ms=now)
                self.ledger.reserve(prepared, observed_at_ms=now)
                self.ledger.transition(
                    intent.intent_id,
                    expected_states=("prepared",),
                    state="submitting",
                    observed_at_ms=now,
                )
                dispatch_started = True
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
            if not dispatch_started:
                raise
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
            raise PolymarketLiveUnknownState(
                "venue order ID differs from signed order hash"
            )
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

    def submit_owned_close_orders(
        self,
        *,
        maximum_book_age_ms: int = 1_500,
    ) -> tuple[PolymarketLiveOrderRecord, ...]:
        maximum_age = int(maximum_book_age_ms)
        if not 100 <= maximum_age <= 5_000:
            raise ValueError("maximum_book_age_ms must lie in [100, 5000]")
        gate = self.reconcile()
        if not gate.can_close:
            raise PolymarketLiveBlocked(
                f"owned close blocked by reconciliation: {gate.errors}"
            )
        if self.ledger.open_owned_order_ids():
            raise PolymarketLiveBlocked(
                "owned close requires all prior bot orders to be terminal"
            )
        records = self.ledger.records()
        by_intent = {record.intent.intent_id: record for record in records}
        output: list[PolymarketLiveOrderRecord] = []
        for lot in self.ledger.owned_lots():
            if lot.provisional or lot.reserved_close_quantity:
                raise PolymarketLiveBlocked(
                    "owned close requires confirmed unreserved lots"
                )
            quantity = lot.available_quantity
            parent = by_intent[lot.parent_intent_id]
            quote = self.venue.close_quote(
                market_id=lot.market_id,
                token_id=lot.token_id,
                quantity=quantity,
                maximum_book_age_ms=maximum_age,
            )
            if (
                quote.market_id != lot.market_id
                or quote.token_id != lot.token_id
                or quote.quantity != quantity
                or quote.source_age_ms < -self.maximum_clock_skew_ms
                or quote.source_age_ms > maximum_age
            ):
                raise PolymarketLiveBlocked(
                    "owned close quote identity or freshness differs"
                )
            attempt = 1 + sum(
                record.intent.closing_only
                and record.intent.parent_intent_id == lot.parent_intent_id
                for record in records
            )
            now = int(time.time() * 1_000)
            identity = _canonical_sha256(
                {
                    "parent_intent_id": lot.parent_intent_id,
                    "attempt": attempt,
                    "quantity": format(quantity, "f"),
                    "limit_price": format(quote.limit_price, "f"),
                    "average_price": format(quote.average_price, "f"),
                    "fee_quote": format(quote.fee_quote, "f"),
                    "net_quote": format(quote.net_quote, "f"),
                    "fee_rate": format(quote.fee_rate, "f"),
                    "fee_exponent": quote.fee_exponent,
                    "book_payload_sha256": quote.book_payload_sha256,
                    "observed_at_ms": quote.observed_at_ms,
                }
            )
            intent = PolymarketLiveOrderIntent(
                intent_id=f"stop-close-{identity[:48]}",
                bot_id=parent.intent.bot_id,
                market_id=lot.market_id,
                token_id=lot.token_id,
                symbol="BTC",
                outcome=parent.intent.outcome,
                side="SELL",
                order_type="FAK",
                limit_price=quote.limit_price,
                quantity=quantity,
                fee_reserve_quote=quote.fee_quote,
                created_at_ms=now,
                expires_at_ms=now + 10_000,
                parent_intent_id=lot.parent_intent_id,
                closing_only=True,
            )
            output.append(
                self.submit(
                    intent,
                    tick_size=quote.tick_size,
                    neg_risk=quote.neg_risk,
                )
            )
        return tuple(output)

    def submit_owned_close_order(
        self,
        *,
        parent_intent_id: str,
        quantity: Decimal,
        action_binding_sha256: str,
        minimum_net_quote: Decimal = Decimal("0"),
        maximum_book_age_ms: int = 1_500,
    ) -> tuple[PolymarketLiveOrderRecord, PolymarketCloseQuote]:
        """Submit one model-requested reduction of an exact bot-owned lot."""

        parent_id = _identifier(parent_intent_id, name="parent_intent_id")
        requested = _decimal(quantity, name="close quantity", positive=True)
        minimum_net = _decimal(
            minimum_net_quote,
            name="minimum close net quote",
            nonnegative=True,
        )
        binding = str(action_binding_sha256 or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", binding) is None or binding == _ZERO_SHA256:
            raise ValueError("action binding must be a nonzero SHA-256 digest")
        maximum_age = int(maximum_book_age_ms)
        if not 100 <= maximum_age <= 5_000:
            raise ValueError("maximum_book_age_ms must lie in [100, 5000]")
        gate = self.reconcile()
        if not gate.can_close:
            raise PolymarketLiveBlocked(
                f"targeted close blocked by reconciliation: {gate.errors}"
            )
        if self.ledger.open_owned_order_ids():
            raise PolymarketLiveBlocked(
                "targeted close requires all prior bot orders to be terminal"
            )
        records = self.ledger.records()
        by_intent = {record.intent.intent_id: record for record in records}
        parent = by_intent.get(parent_id)
        lots = {item.parent_intent_id: item for item in self.ledger.owned_lots()}
        lot = lots.get(parent_id)
        if (
            parent is None
            or parent.intent.side != "BUY"
            or parent.intent.closing_only
            or parent.state not in _TERMINAL_STATES
            or lot is None
            or lot.provisional
            or lot.reserved_close_quantity
            or requested > lot.available_quantity
        ):
            raise PolymarketLiveBlocked(
                "targeted close exceeds its confirmed unreserved bot-owned lot"
            )
        quote = self.venue.close_quote(
            market_id=lot.market_id,
            token_id=lot.token_id,
            quantity=requested,
            maximum_book_age_ms=maximum_age,
        )
        if (
            quote.market_id != lot.market_id
            or quote.token_id != lot.token_id
            or quote.quantity != requested
            or quote.source_age_ms < -self.maximum_clock_skew_ms
            or quote.source_age_ms > maximum_age
        ):
            raise PolymarketLiveBlocked(
                "targeted close quote identity or freshness differs"
            )
        if quote.net_quote < minimum_net:
            raise PolymarketLiveBlocked(
                "targeted close no longer meets its after-cost proceeds floor"
            )
        attempt = 1 + sum(
            record.intent.closing_only and record.intent.parent_intent_id == parent_id
            for record in records
        )
        now = int(time.time() * 1_000)
        identity = _canonical_sha256(
            {
                "action_binding_sha256": binding,
                "minimum_net_quote": format(minimum_net, "f"),
                "parent_intent_id": parent_id,
                "attempt": attempt,
                "quantity": format(requested, "f"),
                "limit_price": format(quote.limit_price, "f"),
                "average_price": format(quote.average_price, "f"),
                "fee_quote": format(quote.fee_quote, "f"),
                "net_quote": format(quote.net_quote, "f"),
                "fee_rate": format(quote.fee_rate, "f"),
                "fee_exponent": quote.fee_exponent,
                "book_payload_sha256": quote.book_payload_sha256,
                "observed_at_ms": quote.observed_at_ms,
            }
        )
        intent = PolymarketLiveOrderIntent(
            intent_id=f"policy-close-{identity[:48]}",
            bot_id=parent.intent.bot_id,
            market_id=lot.market_id,
            token_id=lot.token_id,
            symbol="BTC",
            outcome=parent.intent.outcome,
            side="SELL",
            order_type="FAK",
            limit_price=quote.limit_price,
            quantity=requested,
            fee_reserve_quote=quote.fee_quote,
            created_at_ms=now,
            expires_at_ms=now + 10_000,
            parent_intent_id=parent_id,
            closing_only=True,
        )
        return (
            self.submit(
                intent,
                tick_size=quote.tick_size,
                neg_risk=quote.neg_risk,
            ),
            quote,
        )

    def cancel_owned_open_orders(self) -> PolymarketCancelResult:
        records = tuple(
            record for record in self.ledger.records() if record.state in _OPEN_STATES
        )
        if not records:
            return PolymarketCancelResult((), ())
        open_orders = self.venue.open_orders()
        remote_by_id = {order.order_id: order for order in open_orders}
        missing_ids = tuple(
            record.expected_order_id
            for record in records
            if record.expected_order_id not in remote_by_id
        )
        exact_orders = self.venue.orders_by_id(missing_ids)
        requested_exact_ids = set(missing_ids)
        if any(order.order_id not in requested_exact_ids for order in exact_orders):
            raise PolymarketLiveBlocked(
                "venue returned an unrequested exact-order record"
            )
        if len({order.order_id for order in exact_orders}) != len(exact_orders):
            raise PolymarketLiveBlocked("venue returned duplicate exact-order records")
        exact_by_id = {order.order_id: order for order in exact_orders}
        now = int(time.time() * 1_000)
        missing_evidence = False
        target_records: list[PolymarketLiveOrderRecord] = []
        for record in records:
            remote = remote_by_id.get(record.expected_order_id)
            if remote is None:
                remote = exact_by_id.get(record.expected_order_id)
            if remote is None:
                try:
                    self.ledger.transition(
                        record.intent.intent_id,
                        expected_states=(record.state,),
                        state="cancel_unknown",
                        observed_at_ms=now,
                        remote_status=record.remote_status or "MISSING",
                        matched_quantity=record.matched_quantity,
                        failure_code="remote_order_absent_without_terminal_evidence",
                    )
                    missing_evidence = True
                except PolymarketStateConflict:
                    current = self.ledger.record(record.intent.intent_id)
                    if current.state not in _TERMINAL_STATES:
                        missing_evidence = True
                continue
            if record.state == "cancel_pending":
                # A prior attempt may still be in flight. Reconciliation must first
                # prove that the order remains open before another exact-ID cancel.
                missing_evidence = True
                continue
            evidence_applied = self._apply_remote_order_evidence(
                record,
                remote,
                fill_evidence=self.ledger.order_fill_evidence(record.expected_order_id),
                observed_at_ms=now,
            )
            if not evidence_applied:
                current = self.ledger.record(record.intent.intent_id)
                if current.state not in _TERMINAL_STATES:
                    missing_evidence = True
                continue
            current = self.ledger.record(record.intent.intent_id)
            if current.state in _TERMINAL_STATES:
                continue
            if current.state not in {"live", "partial"}:
                missing_evidence = True
                continue
            try:
                self.ledger.transition(
                    record.intent.intent_id,
                    expected_states=(current.state,),
                    state="cancel_pending",
                    observed_at_ms=now,
                    remote_status=current.remote_status,
                    matched_quantity=current.matched_quantity,
                )
                target_records.append(current)
            except PolymarketStateConflict:
                current = self.ledger.record(record.intent.intent_id)
                if current.state not in _TERMINAL_STATES:
                    missing_evidence = True
        targets = tuple(record.expected_order_id for record in target_records)
        by_order = {record.expected_order_id: record for record in target_records}
        if not targets:
            if missing_evidence:
                raise PolymarketLiveUnknownState(
                    "owned orders lack terminal cancellation evidence"
                )
            return PolymarketCancelResult((), ())
        try:
            result = self.venue.cancel_orders(targets)
        except PolymarketVenueTemporarilyUnavailable as exc:
            for record in target_records:
                current = self.ledger.record(record.intent.intent_id)
                if current.state != "cancel_pending":
                    continue
                restored_state = "partial" if current.matched_quantity > 0 else "live"
                try:
                    self.ledger.transition(
                        record.intent.intent_id,
                        expected_states=("cancel_pending",),
                        state=restored_state,
                        observed_at_ms=int(time.time() * 1_000),
                        remote_status=current.remote_status,
                        matched_quantity=current.matched_quantity,
                        failure_code="cancel_deferred_venue_unavailable",
                    )
                except PolymarketStateConflict:
                    pass
            raise PolymarketLiveBlocked(
                "Polymarket cancellation was deferred while the venue was "
                "temporarily unavailable"
            ) from exc
        except Exception as exc:
            for record in target_records:
                current = self.ledger.record(record.intent.intent_id)
                if current.state != "cancel_pending":
                    continue
                try:
                    self.ledger.transition(
                        record.intent.intent_id,
                        expected_states=("cancel_pending",),
                        state="cancel_unknown",
                        observed_at_ms=int(time.time() * 1_000),
                        remote_status=current.remote_status,
                        matched_quantity=current.matched_quantity,
                        failure_code=exc.__class__.__name__,
                    )
                except PolymarketStateConflict:
                    pass
            raise PolymarketLiveUnknownState(
                "Polymarket cancellation outcome is unknown"
            ) from exc
        reported = set(result.cancelled_order_ids) | set(result.failed_order_ids)
        if not reported <= set(targets):
            raise PolymarketLiveBlocked("venue reported an unrequested order")
        for order_id in result.cancelled_order_ids:
            record = by_order[order_id]
            current = self.ledger.record(record.intent.intent_id)
            if current.state == "cancel_pending":
                fill_evidence = self.ledger.order_fill_evidence(order_id)
                fill_mismatch = fill_evidence.quantity > current.matched_quantity or (
                    current.matched_quantity == 0 and fill_evidence.has_active_fills
                )
                if fill_mismatch:
                    next_state = "cancel_unknown"
                    failure_code = "cancelled_order_fill_quantity_mismatch"
                    missing_evidence = True
                elif current.matched_quantity == 0:
                    next_state = "cancelled"
                    failure_code = ""
                elif (
                    fill_evidence.quantity == current.matched_quantity
                    and fill_evidence.all_active_fills_confirmed
                ):
                    next_state = "filled"
                    failure_code = ""
                else:
                    next_state = "matched_pending"
                    failure_code = "cancelled_order_awaiting_exact_fill_evidence"
                    missing_evidence = True
                try:
                    self.ledger.transition(
                        record.intent.intent_id,
                        expected_states=("cancel_pending",),
                        state=next_state,
                        observed_at_ms=int(time.time() * 1_000),
                        remote_status="CANCELLED",
                        matched_quantity=current.matched_quantity,
                        failure_code=failure_code,
                    )
                except PolymarketStateConflict:
                    missing_evidence = True
            elif current.state not in _TERMINAL_STATES:
                missing_evidence = True
        for order_id in result.failed_order_ids:
            missing_evidence = True
            record = by_order[order_id]
            current = self.ledger.record(record.intent.intent_id)
            if current.state == "cancel_pending":
                try:
                    self.ledger.transition(
                        record.intent.intent_id,
                        expected_states=("cancel_pending",),
                        state="cancel_unknown",
                        observed_at_ms=int(time.time() * 1_000),
                        remote_status=current.remote_status,
                        matched_quantity=current.matched_quantity,
                        failure_code="venue_cancel_failed",
                    )
                except PolymarketStateConflict:
                    missing_evidence = True
            elif current.state not in _TERMINAL_STATES:
                missing_evidence = True
        if reported != set(targets):
            for order_id in set(targets) - reported:
                record = by_order[order_id]
                current = self.ledger.record(record.intent.intent_id)
                if current.state == "cancel_pending":
                    try:
                        self.ledger.transition(
                            record.intent.intent_id,
                            expected_states=("cancel_pending",),
                            state="cancel_unknown",
                            observed_at_ms=int(time.time() * 1_000),
                            remote_status=current.remote_status,
                            matched_quantity=current.matched_quantity,
                            failure_code="venue_cancel_response_incomplete",
                        )
                    except PolymarketStateConflict:
                        pass
            raise PolymarketLiveUnknownState(
                "venue cancellation response was incomplete"
            )
        if missing_evidence:
            raise PolymarketLiveUnknownState(
                "one or more owned orders are absent or lack terminal "
                "cancellation evidence"
            )
        return result


__all__ = [
    "POLYMARKET_LIVE_LEDGER_SCHEMA_VERSION",
    "POLYMARKET_LIVE_ORDER_SCHEMA_VERSION",
    "PolymarketCancelResult",
    "PolymarketCloseQuote",
    "PolymarketConditionAccounting",
    "PolymarketFundingPreflight",
    "PolymarketLedgerRevision",
    "PolymarketLiveBlocked",
    "PolymarketLiveCoordinator",
    "PolymarketLiveError",
    "PolymarketLiveOrderIntent",
    "PolymarketLiveOrderLedger",
    "PolymarketLiveOrderRecord",
    "PolymarketLiveRiskLimits",
    "PolymarketLiveUnknownState",
    "PolymarketLiveVenue",
    "PolymarketOpenQuote",
    "PolymarketRuntimeAuthority",
    "PolymarketOrderFillEvidence",
    "PolymarketOwnedInventory",
    "PolymarketOwnedLot",
    "PolymarketPreparedOrder",
    "PolymarketRealizedPnlEvent",
    "PolymarketRedemptionRecord",
    "PolymarketReconciliation",
    "PolymarketRemoteFill",
    "PolymarketRemoteOrder",
    "PolymarketRemotePosition",
    "PolymarketSubmission",
    "PolymarketStateConflict",
    "PolymarketVenuePreflight",
    "PolymarketVenueRejected",
    "PolymarketVenueTemporarilyUnavailable",
    "polymarket_live_metadata",
]
