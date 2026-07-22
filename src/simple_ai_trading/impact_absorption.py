"""Round 73 prospective L2 event semantics and synchronized local book."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import heapq
import math
from typing import Literal, Mapping, Sequence, cast

from .assets import normalize_symbol


ROUND73_DESIGN_SHA256 = (
    "84b5e6c942d03ebd97b7e120951ed576e3fd8161d65755734c359b7261d6b1fe"
)
ROUND73_EVENT_SCHEMA_VERSION = "round-073-prospective-l2-event-v1"
EXPECTED_STREAM_TYPE = 1
ROUND73_LEVEL_BANDS = (
    "levels_1_5",
    "levels_6_10",
    "levels_11_20",
    "outside_20",
)
Round73LevelBand = Literal[
    "levels_1_5",
    "levels_6_10",
    "levels_11_20",
    "outside_20",
]
_COMBINED_STREAM_SUFFIXES = {
    "depthUpdate": "depth@100ms",
    "bookTicker": "bookTicker",
    "aggTrade": "aggTrade",
    "markPriceUpdate": "markPrice@1s",
    "forceOrder": "forceOrder",
}


class ImpactFeedIntegrityError(ValueError):
    """An event cannot safely extend the current Round 73 evidence segment."""


class DepthSnapshotBridgeError(ImpactFeedIntegrityError):
    """Buffered depth events cannot be joined to the REST snapshot."""


class DepthSequenceGapError(ImpactFeedIntegrityError):
    """A USD-M depth event does not continue the verified ``pu`` chain."""


def _integer(value: object, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool):
        raise ImpactFeedIntegrityError(f"{label} must be an integer")
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ImpactFeedIntegrityError(f"{label} must be an integer") from exc
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ImpactFeedIntegrityError(f"{label} must be an integer")
    parsed = int(numeric)
    if positive and parsed <= 0:
        raise ImpactFeedIntegrityError(f"{label} must be positive")
    if not positive and parsed < 0:
        raise ImpactFeedIntegrityError(f"{label} must be non-negative")
    return parsed


def _decimal(value: object, label: str, *, allow_zero: bool = False) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ImpactFeedIntegrityError(f"{label} must be decimal") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ImpactFeedIntegrityError(f"{label} must be finite and {qualifier}")
    return parsed


def _float(value: object, label: str, *, allow_zero: bool = False) -> float:
    parsed = float(_decimal(value, label, allow_zero=allow_zero))
    if not math.isfinite(parsed):
        raise ImpactFeedIntegrityError(f"{label} must be finite")
    return parsed


def _signed_float(value: object, label: str) -> float:
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ImpactFeedIntegrityError(f"{label} must be decimal") from exc
    if not numeric.is_finite():
        raise ImpactFeedIntegrityError(f"{label} must be finite")
    parsed = float(numeric)
    if not math.isfinite(parsed):
        raise ImpactFeedIntegrityError(f"{label} must be finite")
    return parsed


def _normalized_symbol(value: object, label: str) -> str:
    symbol = normalize_symbol(value, default="")
    if not symbol:
        raise ImpactFeedIntegrityError(f"{label} must be a valid exchange symbol")
    return symbol


def expected_combined_stream_name(*, event_type: str, symbol: str) -> str:
    """Return the exact combined-stream name frozen by the Round 73 capture."""

    expected_symbol = _normalized_symbol(symbol, "expected stream symbol")
    try:
        suffix = _COMBINED_STREAM_SUFFIXES[str(event_type)]
    except KeyError as exc:
        raise ImpactFeedIntegrityError(
            f"unsupported combined-stream event: {event_type or 'missing'}"
        ) from exc
    return f"{expected_symbol.lower()}@{suffix}"


def validate_combined_stream_name(
    stream_name: object, *, event_type: str, symbol: str
) -> str:
    """Bind a typed event to the exact stream that was subscribed for it."""

    observed = str(stream_name)
    expected = expected_combined_stream_name(event_type=event_type, symbol=symbol)
    if observed != expected:
        raise ImpactFeedIntegrityError(
            f"combined stream mismatch: {observed or 'missing'} != {expected}"
        )
    return observed


def _stream_symbol(payload: Mapping[str, object], expected_symbol: str) -> str:
    symbol = _normalized_symbol(payload.get("s"), "stream symbol")
    if symbol != expected_symbol:
        raise ImpactFeedIntegrityError(
            f"stream symbol mismatch: {symbol or 'missing'} != {expected_symbol}"
        )
    return symbol


def _stream_type(payload: Mapping[str, object]) -> int:
    stream_type = _integer(payload.get("st"), "stream type")
    if stream_type != EXPECTED_STREAM_TYPE:
        raise ImpactFeedIntegrityError(
            f"stream type mismatch: {stream_type} != {EXPECTED_STREAM_TYPE}"
        )
    return stream_type


def _event_times(payload: Mapping[str, object]) -> tuple[int, int]:
    event_time_ms = _integer(payload.get("E"), "event time")
    transaction_time_ms = _integer(payload.get("T"), "transaction time")
    if transaction_time_ms > event_time_ms:
        raise ImpactFeedIntegrityError("transaction time cannot exceed event time")
    return event_time_ms, transaction_time_ms


def _event_contract(
    payload: Mapping[str, object],
    *,
    event_type: str,
    symbol: str,
    require_product_symbol: bool,
) -> tuple[int, int]:
    if str(payload.get("e", "")) != event_type:
        raise ImpactFeedIntegrityError(f"expected {event_type} event")
    _stream_symbol(payload, symbol)
    _stream_type(payload)
    if require_product_symbol:
        product_symbol = _normalized_symbol(payload.get("ps"), "public product symbol")
        if product_symbol != symbol:
            raise ImpactFeedIntegrityError(
                "public product symbol does not match event symbol"
            )
    return _event_times(payload)


@dataclass(frozen=True)
class DepthLevelChange:
    side: Literal["bid", "ask"]
    price_ticks: int
    price: float
    previous_qty: float
    new_qty: float
    added_qty: float
    removed_qty: float

    @property
    def added_quote(self) -> float:
        return self.added_qty * self.price

    @property
    def removed_quote(self) -> float:
        return self.removed_qty * self.price


@dataclass(frozen=True)
class DepthUpdate:
    symbol: str
    event_time_ms: int
    transaction_time_ms: int
    first_update_id: int
    final_update_id: int
    previous_update_id: int
    receive_time_ns: int
    changes: tuple[DepthLevelChange, ...]
    best_bid: float
    best_ask: float
    stale: bool = False


@dataclass(frozen=True)
class BookTickerEvent:
    symbol: str
    event_time_ms: int
    transaction_time_ms: int
    update_id: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    receive_time_ns: int


@dataclass(frozen=True)
class AggregateTradeEvent:
    symbol: str
    event_time_ms: int
    transaction_time_ms: int
    aggregate_trade_id: int
    first_trade_id: int
    last_trade_id: int
    price: float
    qty: float
    normalized_qty: float
    buyer_is_maker: bool
    receive_time_ns: int

    @property
    def quote_notional(self) -> float:
        return self.price * self.qty

    @property
    def aggressive_side(self) -> Literal["buy", "sell"]:
        return "sell" if self.buyer_is_maker else "buy"


@dataclass(frozen=True)
class MarkPriceEvent:
    symbol: str
    event_time_ms: int
    mark_price: float
    index_price: float
    estimated_settlement_price: float | None
    funding_rate: float
    next_funding_time_ms: int
    receive_time_ns: int


@dataclass(frozen=True)
class LiquidationSnapshotEvent:
    symbol: str
    event_time_ms: int
    order_time_ms: int
    side: Literal["BUY", "SELL"]
    order_type: str
    time_in_force: str
    original_qty: float
    price: float
    average_price: float
    order_status: str
    last_filled_qty: float
    accumulated_filled_qty: float
    receive_time_ns: int

    @property
    def observed_filled_quote(self) -> float:
        return self.accumulated_filled_qty * self.average_price


@dataclass(frozen=True)
class L2BookState:
    symbol: str
    update_id: int
    best_bid: float
    best_ask: float
    spread_bps: float
    mid: float
    bid_levels: tuple[tuple[float, float], ...]
    ask_levels: tuple[tuple[float, float], ...]
    bid_depth_quote_5: float
    ask_depth_quote_5: float
    bid_depth_quote_10: float
    ask_depth_quote_10: float
    bid_depth_quote_20: float
    ask_depth_quote_20: float
    imbalance_5: float
    imbalance_10: float
    imbalance_20: float


def pre_event_level_band(
    state: L2BookState,
    change: DepthLevelChange,
) -> Round73LevelBand:
    """Classify a changed price by its causal rank before applying the event."""

    levels = state.bid_levels if change.side == "bid" else state.ask_levels
    better = sum(
        price > change.price if change.side == "bid" else price < change.price
        for price, _quantity in levels
    )
    rank = better + 1
    if rank <= 5:
        return "levels_1_5"
    if rank <= 10:
        return "levels_6_10"
    if rank <= 20:
        return "levels_11_20"
    return "outside_20"


class SynchronizedDepthBook:
    """A tick-normalized USD-M local book with strict snapshot/``pu`` continuity."""

    def __init__(self, symbol: str, tick_size: object) -> None:
        self.symbol = _normalized_symbol(symbol, "book symbol")
        self._tick_size = _decimal(tick_size, "tick size")
        self._bids: dict[int, float] = {}
        self._asks: dict[int, float] = {}
        self.snapshot_update_id: int | None = None
        self.last_update_id: int | None = None
        self.bridged = False
        self.stale_event_count = 0

    @property
    def tick_size(self) -> float:
        return float(self._tick_size)

    def _price_ticks(self, value: object) -> int:
        price = _decimal(value, "price")
        ticks = price / self._tick_size
        integral = ticks.to_integral_value()
        if ticks != integral:
            raise ImpactFeedIntegrityError(
                f"price {price} is not aligned to tick size {self._tick_size}"
            )
        parsed = int(integral)
        if parsed <= 0:
            raise ImpactFeedIntegrityError("price tick must be positive")
        return parsed

    def _price(self, ticks: int) -> float:
        return float(self._tick_size * ticks)

    def _levels(
        self,
        value: object,
        *,
        allow_zero: bool,
    ) -> tuple[tuple[int, float], ...]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ImpactFeedIntegrityError("depth levels must be a sequence")
        output: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in value:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes, bytearray))
                or len(item) < 2
            ):
                raise ImpactFeedIntegrityError(
                    "depth level must contain price and quantity"
                )
            ticks = self._price_ticks(item[0])
            if ticks in seen:
                raise ImpactFeedIntegrityError("duplicate price level in one payload")
            seen.add(ticks)
            qty = _float(item[1], "quantity", allow_zero=allow_zero)
            output.append((ticks, qty))
        return tuple(output)

    def initialize(self, snapshot: Mapping[str, object]) -> None:
        update_id = _integer(snapshot.get("lastUpdateId"), "snapshot update id")
        bids = self._levels(snapshot.get("bids"), allow_zero=False)
        asks = self._levels(snapshot.get("asks"), allow_zero=False)
        if not bids or not asks:
            raise ImpactFeedIntegrityError("snapshot must contain both sides")
        bid_book = dict(bids)
        ask_book = dict(asks)
        if max(bid_book) >= min(ask_book):
            raise ImpactFeedIntegrityError("snapshot order book is crossed")
        self._bids = bid_book
        self._asks = ask_book
        self.snapshot_update_id = update_id
        self.last_update_id = update_id
        self.bridged = False
        self.stale_event_count = 0

    def _best_from(
        self, bids: Mapping[int, float], asks: Mapping[int, float]
    ) -> tuple[float, float]:
        if not bids or not asks:
            raise ImpactFeedIntegrityError("local order book side is empty")
        best_bid_ticks = max(bids)
        best_ask_ticks = min(asks)
        if best_bid_ticks >= best_ask_ticks:
            raise ImpactFeedIntegrityError("local order book is crossed")
        return self._price(best_bid_ticks), self._price(best_ask_ticks)

    def _best(self) -> tuple[float, float]:
        return self._best_from(self._bids, self._asks)

    def apply(
        self, payload: Mapping[str, object], *, receive_time_ns: int
    ) -> DepthUpdate:
        if self.last_update_id is None or self.snapshot_update_id is None:
            raise ImpactFeedIntegrityError("depth snapshot has not been initialized")
        event_time_ms, transaction_time_ms = _event_contract(
            payload,
            event_type="depthUpdate",
            symbol=self.symbol,
            require_product_symbol=True,
        )
        received = _integer(receive_time_ns, "receive time")
        first_update_id = _integer(payload.get("U"), "first update id")
        final_update_id = _integer(payload.get("u"), "final update id")
        previous_update_id = _integer(payload.get("pu"), "previous update id")
        if final_update_id < first_update_id:
            raise ImpactFeedIntegrityError("final update id precedes first update id")
        if final_update_id <= self.last_update_id:
            self.stale_event_count += 1
            best_bid, best_ask = self._best()
            return DepthUpdate(
                symbol=self.symbol,
                event_time_ms=event_time_ms,
                transaction_time_ms=transaction_time_ms,
                first_update_id=first_update_id,
                final_update_id=final_update_id,
                previous_update_id=previous_update_id,
                receive_time_ns=received,
                changes=(),
                best_bid=best_bid,
                best_ask=best_ask,
                stale=True,
            )
        if not self.bridged:
            target = self.snapshot_update_id + 1
            if not (
                first_update_id <= target <= final_update_id
                or previous_update_id == self.snapshot_update_id
            ):
                raise DepthSnapshotBridgeError(
                    "depth event does not bridge snapshot: "
                    f"snapshot={self.snapshot_update_id} first={first_update_id} "
                    f"final={final_update_id} previous={previous_update_id}"
                )
        elif previous_update_id != self.last_update_id:
            raise DepthSequenceGapError(
                "depth previous update id mismatch: "
                f"expected={self.last_update_id} actual={previous_update_id}"
            )
        bid_book = self._bids.copy()
        ask_book = self._asks.copy()
        parsed = (
            ("bid", bid_book, self._levels(payload.get("b"), allow_zero=True)),
            ("ask", ask_book, self._levels(payload.get("a"), allow_zero=True)),
        )
        changes: list[DepthLevelChange] = []
        for side, book, levels in parsed:
            for ticks, new_qty in levels:
                previous_qty = float(book.get(ticks, 0.0))
                if new_qty == 0.0:
                    book.pop(ticks, None)
                else:
                    book[ticks] = new_qty
                delta = new_qty - previous_qty
                changes.append(
                    DepthLevelChange(
                        side=cast(Literal["bid", "ask"], side),
                        price_ticks=ticks,
                        price=self._price(ticks),
                        previous_qty=previous_qty,
                        new_qty=new_qty,
                        added_qty=max(delta, 0.0),
                        removed_qty=max(-delta, 0.0),
                    )
                )
        best_bid, best_ask = self._best_from(bid_book, ask_book)
        self._bids = bid_book
        self._asks = ask_book
        self.last_update_id = final_update_id
        self.bridged = True
        return DepthUpdate(
            symbol=self.symbol,
            event_time_ms=event_time_ms,
            transaction_time_ms=transaction_time_ms,
            first_update_id=first_update_id,
            final_update_id=final_update_id,
            previous_update_id=previous_update_id,
            receive_time_ns=received,
            changes=tuple(changes),
            best_bid=best_bid,
            best_ask=best_ask,
        )

    def state(self, levels: int = 20) -> L2BookState:
        count = int(levels)
        if count < 20:
            raise ValueError("Round 73 state requires at least 20 levels")
        best_bid, best_ask = self._best()
        bid_ticks = heapq.nlargest(count, self._bids)
        ask_ticks = heapq.nsmallest(count, self._asks)
        if len(bid_ticks) < 20 or len(ask_ticks) < 20:
            raise ImpactFeedIntegrityError(
                "local book has fewer than 20 levels per side"
            )
        bids = tuple((self._price(tick), self._bids[tick]) for tick in bid_ticks)
        asks = tuple((self._price(tick), self._asks[tick]) for tick in ask_ticks)
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) * 10_000.0 / mid

        def depth(values: tuple[tuple[float, float], ...], size: int) -> float:
            return math.fsum(price * qty for price, qty in values[:size])

        def imbalance(bid_depth: float, ask_depth: float) -> float:
            total = bid_depth + ask_depth
            if total <= 0.0:
                raise ImpactFeedIntegrityError("top-level quote depth must be positive")
            return (bid_depth - ask_depth) / total

        bid5, ask5 = depth(bids, 5), depth(asks, 5)
        bid10, ask10 = depth(bids, 10), depth(asks, 10)
        bid20, ask20 = depth(bids, 20), depth(asks, 20)
        return L2BookState(
            symbol=self.symbol,
            update_id=int(self.last_update_id or 0),
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            mid=mid,
            bid_levels=bids,
            ask_levels=asks,
            bid_depth_quote_5=bid5,
            ask_depth_quote_5=ask5,
            bid_depth_quote_10=bid10,
            ask_depth_quote_10=ask10,
            bid_depth_quote_20=bid20,
            ask_depth_quote_20=ask20,
            imbalance_5=imbalance(bid5, ask5),
            imbalance_10=imbalance(bid10, ask10),
            imbalance_20=imbalance(bid20, ask20),
        )


def parse_book_ticker(
    payload: Mapping[str, object], *, symbol: str, receive_time_ns: int
) -> BookTickerEvent:
    expected = _normalized_symbol(symbol, "expected symbol")
    event_time_ms, transaction_time_ms = _event_contract(
        payload,
        event_type="bookTicker",
        symbol=expected,
        require_product_symbol=True,
    )
    bid = _float(payload.get("b"), "bid")
    ask = _float(payload.get("a"), "ask")
    bid_qty = _float(payload.get("B"), "bid quantity")
    ask_qty = _float(payload.get("A"), "ask quantity")
    if bid >= ask:
        raise ImpactFeedIntegrityError("book ticker is crossed")
    return BookTickerEvent(
        symbol=expected,
        event_time_ms=event_time_ms,
        transaction_time_ms=transaction_time_ms,
        update_id=_integer(payload.get("u"), "book ticker update id"),
        bid=bid,
        bid_qty=bid_qty,
        ask=ask,
        ask_qty=ask_qty,
        receive_time_ns=_integer(receive_time_ns, "receive time"),
    )


def parse_aggregate_trade(
    payload: Mapping[str, object], *, symbol: str, receive_time_ns: int
) -> AggregateTradeEvent:
    expected = _normalized_symbol(symbol, "expected symbol")
    event_time_ms, transaction_time_ms = _event_contract(
        payload,
        event_type="aggTrade",
        symbol=expected,
        require_product_symbol=False,
    )
    maker = payload.get("m")
    if not isinstance(maker, bool):
        raise ImpactFeedIntegrityError("aggregate-trade maker flag must be boolean")
    first_trade_id = _integer(payload.get("f"), "first trade id")
    last_trade_id = _integer(payload.get("l"), "last trade id")
    if last_trade_id < first_trade_id:
        raise ImpactFeedIntegrityError("last trade id precedes first trade id")
    qty = _float(payload.get("q"), "aggregate-trade quantity")
    normalized = _float(payload.get("nq", payload.get("q")), "normalized quantity")
    return AggregateTradeEvent(
        symbol=expected,
        event_time_ms=event_time_ms,
        transaction_time_ms=transaction_time_ms,
        aggregate_trade_id=_integer(payload.get("a"), "aggregate trade id"),
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        price=_float(payload.get("p"), "aggregate-trade price"),
        qty=qty,
        normalized_qty=normalized,
        buyer_is_maker=maker,
        receive_time_ns=_integer(receive_time_ns, "receive time"),
    )


def parse_mark_price(
    payload: Mapping[str, object], *, symbol: str, receive_time_ns: int
) -> MarkPriceEvent:
    expected = _normalized_symbol(symbol, "expected symbol")
    if str(payload.get("e", "")) != "markPriceUpdate":
        raise ImpactFeedIntegrityError("expected markPriceUpdate event")
    _stream_symbol(payload, expected)
    _stream_type(payload)
    event_time_ms = _integer(payload.get("E"), "event time")
    estimated_raw = payload.get("P")
    estimated = None
    if estimated_raw is not None and estimated_raw != "":
        parsed = _float(estimated_raw, "estimated settlement price", allow_zero=True)
        estimated = parsed if parsed > 0.0 else None
    funding_rate = _signed_float(payload.get("r"), "funding rate")
    return MarkPriceEvent(
        symbol=expected,
        event_time_ms=event_time_ms,
        mark_price=_float(payload.get("p"), "mark price"),
        index_price=_float(payload.get("i"), "index price"),
        estimated_settlement_price=estimated,
        funding_rate=funding_rate,
        next_funding_time_ms=_integer(
            payload.get("T"), "next funding time", positive=False
        ),
        receive_time_ns=_integer(receive_time_ns, "receive time"),
    )


def parse_liquidation_snapshot(
    payload: Mapping[str, object], *, symbol: str, receive_time_ns: int
) -> LiquidationSnapshotEvent:
    expected = _normalized_symbol(symbol, "expected symbol")
    if str(payload.get("e", "")) != "forceOrder":
        raise ImpactFeedIntegrityError("expected forceOrder event")
    event_time_ms = _integer(payload.get("E"), "event time")
    order = payload.get("o")
    if not isinstance(order, Mapping):
        raise ImpactFeedIntegrityError(
            "forceOrder payload must contain an order object"
        )
    _stream_symbol(order, expected)
    _stream_type(order)
    product_symbol = _normalized_symbol(order.get("ps"), "liquidation product symbol")
    if product_symbol != expected:
        raise ImpactFeedIntegrityError(
            "liquidation product symbol does not match order symbol"
        )
    side = str(order.get("S", "")).upper()
    if side not in {"BUY", "SELL"}:
        raise ImpactFeedIntegrityError("forceOrder side must be BUY or SELL")
    order_time_ms = _integer(order.get("T"), "forceOrder order time")
    if order_time_ms > event_time_ms:
        raise ImpactFeedIntegrityError("forceOrder order time cannot exceed event time")
    return LiquidationSnapshotEvent(
        symbol=expected,
        event_time_ms=event_time_ms,
        order_time_ms=order_time_ms,
        side=cast(Literal["BUY", "SELL"], side),
        order_type=str(order.get("o", "")),
        time_in_force=str(order.get("f", "")),
        original_qty=_float(order.get("q"), "forceOrder original quantity"),
        price=_float(order.get("p"), "forceOrder price", allow_zero=True),
        average_price=_float(
            order.get("ap"), "forceOrder average price", allow_zero=True
        ),
        order_status=str(order.get("X", "")),
        last_filled_qty=_float(
            order.get("l"), "forceOrder last filled quantity", allow_zero=True
        ),
        accumulated_filled_qty=_float(
            order.get("z"), "forceOrder accumulated filled quantity", allow_zero=True
        ),
        receive_time_ns=_integer(receive_time_ns, "receive time"),
    )


__all__ = [
    "AggregateTradeEvent",
    "BookTickerEvent",
    "DepthLevelChange",
    "DepthSequenceGapError",
    "DepthSnapshotBridgeError",
    "DepthUpdate",
    "ImpactFeedIntegrityError",
    "L2BookState",
    "LiquidationSnapshotEvent",
    "MarkPriceEvent",
    "ROUND73_DESIGN_SHA256",
    "ROUND73_EVENT_SCHEMA_VERSION",
    "SynchronizedDepthBook",
    "expected_combined_stream_name",
    "parse_aggregate_trade",
    "parse_book_ticker",
    "parse_liquidation_snapshot",
    "parse_mark_price",
    "validate_combined_stream_name",
]
