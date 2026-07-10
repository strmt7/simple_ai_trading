"""Bounded live Binance event aggregation for promoted microstructure inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .assets import normalize_symbol
from .microstructure_model import MicrostructureActionPrediction
from .microstructure_runtime import (
    MicrostructureSecond,
    StreamingFeatureRow,
    StreamingMicrostructureFeatureEngine,
)


class MicrostructureFeedIntegrityError(ValueError):
    """The live feed can no longer extend the current causal feature history."""


class LateMicrostructureEventError(MicrostructureFeedIntegrityError):
    """A message arrived after its event-time second was already finalized."""


class MicrostructureScorer(Protocol):
    symbol: str
    decision_cadence_seconds: int
    total_latency_ms: int
    max_quote_age_ms: int

    def score(
        self,
        features: Sequence[float] | np.ndarray,
        *,
        decision_time_ms: int,
        order_notional_quote: float,
        close_bid: float,
        close_ask: float,
        close_bid_qty: float,
        close_ask_qty: float,
        quote_time_ms: int,
        observation_time_ms: int,
    ) -> MicrostructureActionPrediction: ...


@dataclass
class _QuoteAccumulator:
    first_key: tuple[int, int, int] | None = None
    last_key: tuple[int, int, int] | None = None
    open_mid: float = 0.0
    high_mid: float = -math.inf
    low_mid: float = math.inf
    close_mid: float = 0.0
    close_bid: float = 0.0
    close_ask: float = 0.0
    close_bid_qty: float = 0.0
    close_ask_qty: float = 0.0
    spread_sum_bps: float = 0.0
    max_spread_bps: float = 0.0
    imbalance_sum: float = 0.0
    close_imbalance: float = 0.0
    microprice_offset_sum_bps: float = 0.0
    quote_updates: int = 0
    event_delays_ms: list[float] = field(default_factory=list)
    fingerprints: dict[tuple[int, int, int], tuple[float, ...]] = field(default_factory=dict)

    def append(self, payload: Mapping[str, object]) -> bool:
        event_time = int(payload.get("E", 0) or 0)
        transaction_time = int(payload.get("T", event_time) or event_time)
        update_id = int(payload.get("u", 0) or 0)
        bid = float(payload.get("b", 0.0))
        ask = float(payload.get("a", 0.0))
        bid_qty = float(payload.get("B", 0.0))
        ask_qty = float(payload.get("A", 0.0))
        if (
            event_time <= 0
            or transaction_time <= 0
            or transaction_time > event_time
            or update_id <= 0
        ):
            raise MicrostructureFeedIntegrityError("bookTicker identifiers or timestamps are invalid")
        if not all(math.isfinite(value) and value > 0.0 for value in (bid, ask, bid_qty, ask_qty)):
            raise MicrostructureFeedIntegrityError("bookTicker quote values are invalid")
        if bid >= ask:
            raise MicrostructureFeedIntegrityError("bookTicker quote is crossed")
        key = (event_time, transaction_time, update_id)
        fingerprint = (bid, ask, bid_qty, ask_qty)
        existing = self.fingerprints.get(key)
        if existing is not None:
            if existing != fingerprint:
                raise MicrostructureFeedIntegrityError("conflicting duplicate bookTicker update")
            return False
        self.fingerprints[key] = fingerprint
        mid = (bid + ask) / 2.0
        spread = (ask - bid) * 10_000.0 / mid
        imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        microprice = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)
        microprice_offset = (microprice / mid - 1.0) * 10_000.0
        if self.first_key is None or key < self.first_key:
            self.first_key = key
            self.open_mid = mid
        if self.last_key is None or key >= self.last_key:
            self.last_key = key
            self.close_mid = mid
            self.close_bid = bid
            self.close_ask = ask
            self.close_bid_qty = bid_qty
            self.close_ask_qty = ask_qty
            self.close_imbalance = imbalance
        self.high_mid = max(self.high_mid, mid)
        self.low_mid = min(self.low_mid, mid)
        self.spread_sum_bps += spread
        self.max_spread_bps = max(self.max_spread_bps, spread)
        self.imbalance_sum += imbalance
        self.microprice_offset_sum_bps += microprice_offset
        self.quote_updates += 1
        self.event_delays_ms.append(float(event_time - transaction_time))
        return True


@dataclass
class _TradeAccumulator:
    last_key: tuple[int, int, int] | None = None
    close: float = 0.0
    base_volume: float = 0.0
    quote_volume: float = 0.0
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    trade_count: int = 0
    fingerprints: dict[tuple[int, int, int], tuple[float, float, bool, int, int]] = field(
        default_factory=dict
    )

    def append(self, payload: Mapping[str, object]) -> bool:
        event_time = int(payload.get("E", 0) or 0)
        transaction_time = int(payload.get("T", event_time) or event_time)
        event_type = str(payload.get("e", ""))
        id_field = "a" if event_type == "aggTrade" else "t"
        trade_id = int(payload.get(id_field, 0) or 0)
        price = float(payload.get("p", 0.0))
        quantity = float(payload.get("q", 0.0))
        maker_value = payload.get("m")
        if not isinstance(maker_value, bool):
            raise MicrostructureFeedIntegrityError("trade maker flag must be a JSON boolean")
        buyer_is_maker = maker_value
        first_trade_id = trade_id
        last_trade_id = trade_id
        if event_type == "aggTrade":
            first_trade_id = int(payload.get("f", 0) or 0)
            last_trade_id = int(payload.get("l", 0) or 0)
        if (
            event_time <= 0
            or transaction_time <= 0
            or transaction_time > event_time
            or trade_id <= 0
            or first_trade_id <= 0
            or last_trade_id < first_trade_id
        ):
            raise MicrostructureFeedIntegrityError("trade identifiers or timestamps are invalid")
        if not all(math.isfinite(value) and value > 0.0 for value in (price, quantity)):
            raise MicrostructureFeedIntegrityError("trade values are invalid")
        key = (event_time, transaction_time, trade_id)
        fingerprint = (
            price,
            quantity,
            buyer_is_maker,
            first_trade_id,
            last_trade_id,
        )
        existing = self.fingerprints.get(key)
        if existing is not None:
            if existing != fingerprint:
                raise MicrostructureFeedIntegrityError("conflicting duplicate trade update")
            return False
        self.fingerprints[key] = fingerprint
        if self.last_key is None or key >= self.last_key:
            self.last_key = key
            self.close = price
        self.base_volume += quantity
        self.quote_volume += price * quantity
        if buyer_is_maker:
            self.aggressive_sell_volume += quantity
        else:
            self.aggressive_buy_volume += quantity
        self.trade_count += last_trade_id - first_trade_id + 1
        return True


@dataclass(frozen=True)
class LiveTopOfBook:
    symbol: str
    event_time_ms: int
    transaction_time_ms: int
    update_id: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float


@dataclass(frozen=True)
class LiveMicrostructurePrediction:
    feature_row: StreamingFeatureRow
    execution_quote: LiveTopOfBook
    prediction: MicrostructureActionPrediction
    observed_exchange_time_ms: int
    signal_deadline_ms: int
    remaining_latency_budget_ms: int


class LiveMicrostructureSecondAggregator:
    """Aggregate event-time seconds after a bounded local-delivery watermark."""

    def __init__(
        self,
        symbol: str,
        *,
        settlement_delay_ms: int = 250,
        max_pending_seconds: int = 10,
    ) -> None:
        self.symbol = normalize_symbol(symbol)
        self.settlement_delay_ms = int(settlement_delay_ms)
        self.max_pending_seconds = int(max_pending_seconds)
        if self.settlement_delay_ms < 0 or self.settlement_delay_ms > 5_000:
            raise ValueError("settlement_delay_ms must lie in [0, 5000]")
        if self.max_pending_seconds < 2 or self.max_pending_seconds > 120:
            raise ValueError("max_pending_seconds must lie in [2, 120]")
        self._quotes: dict[int, _QuoteAccumulator] = {}
        self._trades: dict[int, _TradeAccumulator] = {}
        self._latest_quote: LiveTopOfBook | None = None
        self._lock = threading.Lock()
        self.last_finalized_second_ms = -1
        self.late_event_count = 0
        self.invalid_event_count = 0
        self.duplicate_event_count = 0
        self.integrity_reset_count = 0
        self.discarded_trade_only_seconds = 0
        self._last_exchange_now_ms = -1

    def _invalidate_pending(self) -> None:
        self._quotes.clear()
        self._trades.clear()
        self._latest_quote = None
        self.integrity_reset_count += 1

    def current_quote(self) -> LiveTopOfBook | None:
        with self._lock:
            return self._latest_quote

    def _event_second(self, payload: Mapping[str, object]) -> int:
        symbol = normalize_symbol(str(payload.get("s", "")))
        if symbol != self.symbol:
            raise MicrostructureFeedIntegrityError(
                f"stream symbol mismatch: {symbol} != {self.symbol}"
            )
        event_time = int(payload.get("E", 0) or 0)
        if event_time <= 0:
            raise MicrostructureFeedIntegrityError("stream event time is missing")
        return (event_time // 1_000) * 1_000

    def ingest(self, payload: Mapping[str, object]) -> None:
        event_type = str(payload.get("e", ""))
        with self._lock:
            try:
                second_ms = self._event_second(payload)
                if second_ms <= self.last_finalized_second_ms:
                    self.late_event_count += 1
                    raise LateMicrostructureEventError(
                        f"late {event_type or 'unknown'} event for finalized second {second_ms}"
                    )
                if event_type == "bookTicker":
                    appended = self._quotes.setdefault(second_ms, _QuoteAccumulator()).append(
                        payload
                    )
                    if appended:
                        quote = LiveTopOfBook(
                            symbol=self.symbol,
                            event_time_ms=int(payload["E"]),
                            transaction_time_ms=int(payload.get("T", payload["E"])),
                            update_id=int(payload["u"]),
                            bid=float(payload["b"]),
                            ask=float(payload["a"]),
                            bid_qty=float(payload["B"]),
                            ask_qty=float(payload["A"]),
                        )
                        current = self._latest_quote
                        if current is None or (
                            quote.event_time_ms,
                            quote.transaction_time_ms,
                            quote.update_id,
                        ) >= (
                            current.event_time_ms,
                            current.transaction_time_ms,
                            current.update_id,
                        ):
                            self._latest_quote = quote
                elif event_type in {"trade", "aggTrade"}:
                    appended = self._trades.setdefault(second_ms, _TradeAccumulator()).append(
                        payload
                    )
                else:
                    raise MicrostructureFeedIntegrityError(
                        f"unsupported live microstructure event: {event_type or 'missing'}"
                    )
                if not appended:
                    self.duplicate_event_count += 1
                pending = set(self._quotes) | set(self._trades)
                if pending and max(pending) - min(pending) > self.max_pending_seconds * 1_000:
                    raise MicrostructureFeedIntegrityError(
                        "live microstructure pending-event window exceeded its bound"
                    )
            except (TypeError, ValueError, OverflowError) as exc:
                self.invalid_event_count += 1
                self._invalidate_pending()
                if isinstance(exc, MicrostructureFeedIntegrityError):
                    raise
                raise MicrostructureFeedIntegrityError(
                    f"live microstructure event could not be parsed: {type(exc).__name__}"
                ) from exc

    def drain(self, exchange_now_ms: int | None = None) -> list[MicrostructureSecond]:
        now_ms = int(time.time() * 1_000) if exchange_now_ms is None else int(exchange_now_ms)
        with self._lock:
            if now_ms < 0 or now_ms < self._last_exchange_now_ms:
                self.invalid_event_count += 1
                self._invalidate_pending()
                raise MicrostructureFeedIntegrityError(
                    "exchange clock must be non-negative and monotonic"
                )
            self._last_exchange_now_ms = now_ms
            complete_before = now_ms - self.settlement_delay_ms
            completed_quote_seconds = sorted(
                second
                for second in self._quotes
                if second + 1_000 <= complete_before
            )
            output: list[MicrostructureSecond] = []
            for second_ms in completed_quote_seconds:
                quote = self._quotes.pop(second_ms)
                trade = self._trades.pop(second_ms, None)
                delays = np.asarray(quote.event_delays_ms, dtype=np.float64)
                trade_close = quote.close_mid if trade is None else trade.close
                buy_volume = 0.0 if trade is None else trade.aggressive_buy_volume
                sell_volume = 0.0 if trade is None else trade.aggressive_sell_volume
                base_volume = 0.0 if trade is None else trade.base_volume
                output.append(
                    MicrostructureSecond(
                        symbol=self.symbol,
                        second_ms=second_ms,
                        open_mid=quote.open_mid,
                        high_mid=quote.high_mid,
                        low_mid=quote.low_mid,
                        close_mid=quote.close_mid,
                        close_bid=quote.close_bid,
                        close_ask=quote.close_ask,
                        close_bid_qty=quote.close_bid_qty,
                        close_ask_qty=quote.close_ask_qty,
                        spread_bps=quote.spread_sum_bps / quote.quote_updates,
                        max_spread_bps=quote.max_spread_bps,
                        l1_imbalance=quote.imbalance_sum / quote.quote_updates,
                        close_l1_imbalance=quote.close_imbalance,
                        microprice_offset_bps=(
                            quote.microprice_offset_sum_bps / quote.quote_updates
                        ),
                        quote_updates=quote.quote_updates,
                        event_delay_p50_ms=float(np.quantile(delays, 0.50)),
                        event_delay_p99_ms=float(np.quantile(delays, 0.99)),
                        trade_close=trade_close,
                        base_volume=base_volume,
                        quote_volume=0.0 if trade is None else trade.quote_volume,
                        aggressive_buy_volume=buy_volume,
                        aggressive_sell_volume=sell_volume,
                        trade_imbalance=(
                            (buy_volume - sell_volume) / base_volume if base_volume > 0.0 else 0.0
                        ),
                        trade_count=0 if trade is None else trade.trade_count,
                    )
                )
                self.last_finalized_second_ms = second_ms
            stale_trade_seconds = [
                second
                for second in self._trades
                if second + 1_000 <= complete_before
            ]
            for second_ms in stale_trade_seconds:
                self._trades.pop(second_ms, None)
                self.discarded_trade_only_seconds += 1
                self.last_finalized_second_ms = max(self.last_finalized_second_ms, second_ms)
            return output


class StreamingMicrostructureCoordinator:
    """Coordinate feed aggregation and inference without owning execution."""

    def __init__(
        self,
        scorer: MicrostructureScorer,
        *,
        settlement_delay_ms: int = 100,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.scorer = scorer
        self.aggregator = LiveMicrostructureSecondAggregator(
            scorer.symbol,
            settlement_delay_ms=settlement_delay_ms,
        )
        self.engine = StreamingMicrostructureFeatureEngine(
            scorer.symbol,
            decision_cadence_seconds=scorer.decision_cadence_seconds,
        )
        if settlement_delay_ms >= int(scorer.total_latency_ms):
            raise ValueError("settlement delay must leave a positive execution-latency budget")
        self._monotonic_ns = monotonic_ns
        self.late_event_resets = 0
        self.feed_integrity_resets = 0
        self.deadline_misses = 0
        self.post_inference_deadline_misses = 0
        self.inference_failures = 0
        self.last_inference_error = ""
        self.missing_current_quote_count = 0

    def ingest(self, payload: Mapping[str, object]) -> None:
        try:
            self.aggregator.ingest(payload)
        except MicrostructureFeedIntegrityError as exc:
            self.engine.reset()
            self.feed_integrity_resets += 1
            if isinstance(exc, LateMicrostructureEventError):
                self.late_event_resets += 1
            raise

    def evaluate_ready(
        self,
        *,
        exchange_now_ms: int,
        order_notional_quote: float,
    ) -> tuple[LiveMicrostructurePrediction, ...]:
        notional = float(order_notional_quote)
        if not math.isfinite(notional) or notional <= 0.0:
            raise ValueError("order_notional_quote must be finite and positive")
        predictions: list[LiveMicrostructurePrediction] = []
        started_ns = int(self._monotonic_ns())
        for second in self.aggregator.drain(exchange_now_ms):
            feature_row = self.engine.append(second)
            if feature_row is None:
                continue
            execution_quote = self.aggregator.current_quote()
            if execution_quote is None:
                self.missing_current_quote_count += 1
                continue
            deadline = feature_row.decision_time_ms + int(self.scorer.total_latency_ms)
            remaining = deadline - int(exchange_now_ms)
            if remaining <= 0:
                self.deadline_misses += 1
                continue
            try:
                prediction = self.scorer.score(
                    feature_row.features,
                    decision_time_ms=feature_row.decision_time_ms,
                    order_notional_quote=notional,
                    close_bid=execution_quote.bid,
                    close_ask=execution_quote.ask,
                    close_bid_qty=execution_quote.bid_qty,
                    close_ask_qty=execution_quote.ask_qty,
                    quote_time_ms=execution_quote.event_time_ms,
                    observation_time_ms=int(exchange_now_ms),
                )
            except Exception as exc:  # noqa: BLE001 - inference failure must fail closed
                self.inference_failures += 1
                self.last_inference_error = f"{type(exc).__name__}: {exc}"[:500]
                continue
            elapsed_ns = max(0, int(self._monotonic_ns()) - started_ns)
            elapsed_ms = (elapsed_ns + 999_999) // 1_000_000
            completed_exchange_time_ms = int(exchange_now_ms) + elapsed_ms
            remaining = deadline - completed_exchange_time_ms
            if remaining <= 0:
                self.deadline_misses += 1
                self.post_inference_deadline_misses += 1
                continue
            predictions.append(
                LiveMicrostructurePrediction(
                    feature_row=feature_row,
                    execution_quote=execution_quote,
                    prediction=prediction,
                    observed_exchange_time_ms=completed_exchange_time_ms,
                    signal_deadline_ms=deadline,
                    remaining_latency_budget_ms=remaining,
                )
            )
        return tuple(predictions)


__all__ = [
    "LateMicrostructureEventError",
    "MicrostructureFeedIntegrityError",
    "LiveMicrostructurePrediction",
    "LiveMicrostructureSecondAggregator",
    "LiveTopOfBook",
    "StreamingMicrostructureCoordinator",
]
