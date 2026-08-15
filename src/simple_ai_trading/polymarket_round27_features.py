"""Receipt-time feature materialization for the Round 27 BTC campaign.

This module is deliberately target blind. It consumes only terminal public-feed
evidence and a target-free condition-replay audit. Resolution labels are joined
by a later, separately frozen experiment runner.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_recorder import DecodedPublicEvent, PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay, PolymarketRecordedBook
from .polymarket_twap60 import (
    PolymarketTwap60FeatureState,
    PolymarketTwap60Tick,
    parse_polymarket_twap60_tick,
)


POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION = (
    "polymarket-round27-receipt-time-features-v1"
)
POLYMARKET_ROUND27_DECISION_STEP_MS = 1_000
POLYMARKET_ROUND27_FIRST_DECISION_OFFSET_MS = 30_000
POLYMARKET_ROUND27_LAST_DECISION_OFFSET_MS = 5_000
POLYMARKET_ROUND27_MAXIMUM_BOOK_AGE_MS = 1_500
POLYMARKET_ROUND27_TRADE_WINDOWS_MS = (
    250,
    1_000,
    5_000,
    15_000,
    60_000,
    300_000,
    600_000,
    1_200_000,
)
POLYMARKET_ROUND27_BOOK_WINDOWS_MS = (1_000, 5_000, 15_000, 60_000)
POLYMARKET_ROUND27_LONG_CONTEXT_WINDOW_MS = 1_200_000
POLYMARKET_ROUND27_LONG_CONTEXT_MINIMUM_COVERAGE = 0.995
POLYMARKET_ROUND27_LONG_CONTEXT_MAXIMUM_RECEIPT_GAP_MS = 5_000.0
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_STATIC_FEATURE_NAMES = (
    "phase.elapsed_fraction",
    "phase.remaining_seconds",
    "market.prior_logit",
    "market.up_midpoint",
    "market.down_midpoint",
    "market.up_spread",
    "market.down_spread",
    "market.up_relative_spread",
    "market.down_relative_spread",
    "market.up_microprice_minus_midpoint",
    "market.down_microprice_minus_midpoint",
    "market.up_top_depth_imbalance",
    "market.down_top_depth_imbalance",
    "market.up_depth5_imbalance",
    "market.down_depth5_imbalance",
    "market.complement_buy_overround",
    "market.complement_sell_underround",
    "market.book_receipt_skew_ms",
    "twap.log_distance_from_open",
    "twap.variance_rate_per_second",
    "twap.path_efficiency",
    "twap.source_age_ms",
)
_BOOK_FLOW_FEATURE_NAMES = tuple(
    f"market.{outcome}_{metric}_{window}ms"
    for outcome in ("up", "down")
    for window in POLYMARKET_ROUND27_BOOK_WINDOWS_MS
    for metric in (
        "mid_log_return",
        "microprice_log_return",
        "top_ofi",
        "update_count",
    )
)
_TRADE_FEATURE_NAMES = tuple(
    f"{venue}.{metric}_{window}ms"
    for venue in ("spot", "usdm")
    for window in POLYMARKET_ROUND27_TRADE_WINDOWS_MS
    for metric in (
        "trade_log_return",
        "realized_variance",
        "signed_quote_imbalance",
        "log1p_quote_notional",
        "trade_count",
        "receipt_coverage_fraction",
        "maximum_receipt_gap_ms",
    )
)
_CROSS_FEATURE_NAMES = (
    "cross.spot_receipt_age_ms",
    "cross.usdm_receipt_age_ms",
    "cross.spot_minus_usdm_receipt_skew_ms",
    "cross.spot_minus_usdm_log_price_basis_bps",
    *(
        f"cross.spot_minus_usdm_log_return_{window}ms"
        for window in POLYMARKET_ROUND27_TRADE_WINDOWS_MS
    ),
    *(
        f"cross.spot_minus_market_up_log_return_{window}ms"
        for window in POLYMARKET_ROUND27_BOOK_WINDOWS_MS
    ),
)
POLYMARKET_ROUND27_FEATURE_NAMES = (
    *_STATIC_FEATURE_NAMES,
    *_BOOK_FLOW_FEATURE_NAMES,
    *_TRADE_FEATURE_NAMES,
    *_CROSS_FEATURE_NAMES,
)
POLYMARKET_ROUND27_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(POLYMARKET_ROUND27_FEATURE_NAMES).encode("ascii")
).hexdigest()


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


def _finite_positive(value: object, *, name: str) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return selected


def _logit(probability: float) -> float:
    selected = min(0.995, max(0.005, float(probability)))
    return math.log(selected / (1.0 - selected))


@dataclass(frozen=True, slots=True)
class Round27TradePoint:
    stream: str
    received_wall_ms: int
    received_monotonic_ns: int
    price: float
    quantity: float
    buyer_is_maker: bool
    event_sha256: str

    @classmethod
    def from_event(cls, event: DecodedPublicEvent) -> "Round27TradePoint":
        expected = {"binance_spot": "trade", "binance_futures": "aggTrade"}
        if (
            event.stream not in expected
            or event.event_type != expected[event.stream]
            or event.symbol != "BTC"
        ):
            raise ValueError("Round 27 Binance trade identity differs")
        payload = event.event.get("data")
        if not isinstance(payload, Mapping) or type(payload.get("m")) is not bool:
            raise ValueError("Round 27 Binance trade payload differs")
        digest = str(event.event_sha256 or "").lower()
        if len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError("Round 27 Binance event hash differs")
        return cls(
            stream=event.stream,
            received_wall_ms=int(event.received_wall_ms),
            received_monotonic_ns=int(event.received_monotonic_ns),
            price=_finite_positive(payload.get("p"), name="trade price"),
            quantity=_finite_positive(payload.get("q"), name="trade quantity"),
            buyer_is_maker=bool(payload["m"]),
            event_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class Round27TradeSeries:
    stream: str
    received_wall_ms: NDArray[np.int64]
    received_monotonic_ns: NDArray[np.int64]
    price: NDArray[np.float64]
    quantity: NDArray[np.float64]
    buyer_is_maker: NDArray[np.bool_]

    @classmethod
    def from_points(
        cls,
        stream: str,
        points: Sequence[Round27TradePoint],
    ) -> "Round27TradeSeries":
        return cls(
            stream=stream,
            received_wall_ms=np.asarray(
                [point.received_wall_ms for point in points], dtype=np.int64
            ),
            received_monotonic_ns=np.asarray(
                [point.received_monotonic_ns for point in points], dtype=np.int64
            ),
            price=np.asarray([point.price for point in points], dtype=np.float64),
            quantity=np.asarray([point.quantity for point in points], dtype=np.float64),
            buyer_is_maker=np.asarray(
                [point.buyer_is_maker for point in points], dtype=np.bool_
            ),
        ).validated()

    @property
    def count(self) -> int:
        return int(self.price.size)

    def validated(self) -> "Round27TradeSeries":
        arrays = (
            self.received_wall_ms,
            self.received_monotonic_ns,
            self.price,
            self.quantity,
            self.buyer_is_maker,
        )
        if (
            self.stream not in {"binance_spot", "binance_futures"}
            or any(value.ndim != 1 for value in arrays)
            or len({int(value.size) for value in arrays}) != 1
            or self.count <= 0
            or self.received_wall_ms.dtype != np.int64
            or self.received_monotonic_ns.dtype != np.int64
            or self.price.dtype != np.float64
            or self.quantity.dtype != np.float64
            or self.buyer_is_maker.dtype != np.bool_
            or np.any(self.received_wall_ms <= 0)
            or np.any(self.received_monotonic_ns <= 0)
            or np.any(np.diff(self.received_wall_ms) < 0)
            or np.any(np.diff(self.received_monotonic_ns) < 0)
            or not np.all(np.isfinite(self.price))
            or not np.all(np.isfinite(self.quantity))
            or np.any(self.price <= 0.0)
            or np.any(self.quantity <= 0.0)
        ):
            raise ValueError("Round 27 compact trade series differs")
        for value in arrays:
            value.flags.writeable = False
        return self


@dataclass(frozen=True, slots=True)
class Round27PublicSourceSeries:
    run_id: str
    spot: Round27TradeSeries
    usdm: Round27TradeSeries
    twap: tuple[PolymarketTwap60Tick, ...]
    source_chain_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    def validated(self) -> "Round27PublicSourceSeries":
        if (
            not self.run_id
            or self.spot.count <= 0
            or self.usdm.count <= 0
            or not self.twap
            or self.target_accessed
            or self.trading_authority
            or len(self.source_chain_sha256) != 64
        ):
            raise ValueError("Round 27 public source series differs")
        if (
            self.spot.validated().stream != "binance_spot"
            or self.usdm.validated().stream != "binance_futures"
        ):
            raise ValueError("Round 27 public source stream differs")
        twap_chronology = tuple(
            (tick.received_monotonic_ns, tick.received_wall_ms) for tick in self.twap
        )
        if twap_chronology != tuple(sorted(twap_chronology)):
            raise ValueError("Round 27 TWAP receipt order differs")
        return self


def load_round27_public_source_series(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> Round27PublicSourceSeries:
    """Load validated public predictor feeds without resolutions or account data."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("Round 27 source run is required")
    wall = {name: array("q") for name in ("binance_spot", "binance_futures")}
    monotonic = {name: array("q") for name in wall}
    price = {name: array("d") for name in wall}
    quantity = {name: array("d") for name in wall}
    maker = {name: array("b") for name in wall}
    twap: list[PolymarketTwap60Tick] = []
    chain = _EMPTY_SHA256
    event_count = 0
    accepted_count = 0
    for event in store.iter_public_events(
        selected,
        streams=("binance_spot", "binance_futures", "polymarket_rtds"),
        ordered=True,
        verified_source=False,
    ):
        event_count += 1
        if event.stream == "polymarket_rtds":
            if event.event_type != "crypto_prices_twap_sixty:update":
                continue
            tick = parse_polymarket_twap60_tick(
                event.event,
                received_wall_ms=event.received_wall_ms,
                received_monotonic_ns=event.received_monotonic_ns,
            )
            twap.append(tick)
            identity = {
                "kind": "twap60",
                "received_monotonic_ns": tick.received_monotonic_ns,
                "source_payload_sha256": tick.source_payload_sha256,
            }
        elif event.symbol == "BTC" and event.event_type in {"trade", "aggTrade"}:
            point = Round27TradePoint.from_event(event)
            wall[point.stream].append(point.received_wall_ms)
            monotonic[point.stream].append(point.received_monotonic_ns)
            price[point.stream].append(point.price)
            quantity[point.stream].append(point.quantity)
            maker[point.stream].append(point.buyer_is_maker)
            identity = {
                "kind": point.stream,
                "received_monotonic_ns": point.received_monotonic_ns,
                "event_sha256": point.event_sha256,
            }
        else:
            continue
        accepted_count += 1
        chain = hashlib.sha256(
            bytes.fromhex(chain) + _canonical_json(identity).encode("ascii")
        ).hexdigest()
        if progress is not None and event_count % 100_000 == 0:
            progress(
                "public-source-replay",
                {
                    "event_count": event_count,
                    "accepted_count": accepted_count,
                    "spot_trade_count": len(wall["binance_spot"]),
                    "usdm_trade_count": len(wall["binance_futures"]),
                    "twap_tick_count": len(twap),
                },
            )

    def compact(stream: str) -> Round27TradeSeries:
        return Round27TradeSeries(
            stream=stream,
            received_wall_ms=np.frombuffer(wall[stream], dtype=np.int64),
            received_monotonic_ns=np.frombuffer(monotonic[stream], dtype=np.int64),
            price=np.frombuffer(price[stream], dtype=np.float64),
            quantity=np.frombuffer(quantity[stream], dtype=np.float64),
            buyer_is_maker=np.frombuffer(maker[stream], dtype=np.int8).astype(
                np.bool_, copy=False
            ),
        ).validated()

    result = Round27PublicSourceSeries(
        run_id=selected,
        spot=compact("binance_spot"),
        usdm=compact("binance_futures"),
        twap=tuple(twap),
        source_chain_sha256=chain,
    ).validated()
    if progress is not None:
        progress(
            "public-source-replay-complete",
            {
                "event_count": event_count,
                "accepted_count": accepted_count,
                "spot_trade_count": result.spot.count,
                "usdm_trade_count": result.usdm.count,
                "twap_tick_count": len(result.twap),
            },
        )
    return result


@dataclass(frozen=True, slots=True)
class _TradeMetrics:
    log_return: float
    realized_variance: float
    signed_quote_imbalance: float
    log1p_quote_notional: float
    count: int
    receipt_coverage_fraction: float
    maximum_receipt_gap_ms: float


def _trade_metrics(
    series: Round27TradeSeries,
    *,
    decision_ms: int,
    window_ms: int,
) -> _TradeMetrics:
    selected = series.validated()
    end = int(np.searchsorted(selected.received_wall_ms, decision_ms, side="left"))
    window_start_ms = decision_ms - window_ms
    start = int(
        np.searchsorted(
            selected.received_wall_ms[:end],
            window_start_ms,
            side="left",
        )
    )
    if start == end:
        return _TradeMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, float(window_ms))
    prices = selected.price[start:end]
    quote = prices * selected.quantity[start:end]
    gross = float(np.sum(quote, dtype=np.float64))
    signs = np.where(selected.buyer_is_maker[start:end], -1.0, 1.0)
    signed = float(np.sum(quote * signs, dtype=np.float64))
    anchor_index = int(
        np.searchsorted(
            selected.received_wall_ms[:end],
            window_start_ms,
            side="right",
        )
    ) - 1
    path_prices = (
        np.concatenate(
            (
                selected.price[anchor_index : anchor_index + 1],
                selected.price[max(start, anchor_index + 1) : end],
            )
        )
        if anchor_index >= 0
        else prices
    )
    returns = np.diff(np.log(path_prices))
    receipt_times = selected.received_wall_ms[start:end]
    boundary_times = np.concatenate(
        (
            np.asarray([window_start_ms], dtype=np.int64),
            receipt_times,
            np.asarray([decision_ms], dtype=np.int64),
        )
    )
    return _TradeMetrics(
        log_return=(
            0.0
            if path_prices.size < 2
            else math.log(path_prices[-1] / path_prices[0])
        ),
        realized_variance=float(np.sum(returns * returns, dtype=np.float64)),
        signed_quote_imbalance=0.0 if gross <= 0.0 else signed / gross,
        log1p_quote_notional=math.log1p(gross),
        count=int(prices.size),
        receipt_coverage_fraction=(
            0.0
            if prices.size < 2
            else min(
                1.0,
                max(
                    0.0,
                    float(receipt_times[-1] - receipt_times[0]) / float(window_ms),
                ),
            )
        ),
        maximum_receipt_gap_ms=float(np.max(np.diff(boundary_times))),
    )


@dataclass(frozen=True, slots=True)
class _BookMetrics:
    midpoint: float
    spread: float
    relative_spread: float
    microprice: float
    top_depth_imbalance: float
    depth5_imbalance: float


def _depth(levels: Sequence[object], count: int) -> float:
    return math.fsum(float(level.quantity) for level in levels[:count])


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return 0.0 if total <= 0.0 else (bid - ask) / total


def _book_metrics(book: PolymarketRecordedBook) -> _BookMetrics:
    snapshot = book.snapshot.validated()
    if not snapshot.bids or not snapshot.asks:
        raise ValueError("Round 27 executable book is one-sided")
    bid = float(snapshot.bids[0].price)
    ask = float(snapshot.asks[0].price)
    bid_quantity = float(snapshot.bids[0].quantity)
    ask_quantity = float(snapshot.asks[0].quantity)
    midpoint = 0.5 * (bid + ask)
    spread = ask - bid
    microprice = (ask * bid_quantity + bid * ask_quantity) / (
        bid_quantity + ask_quantity
    )
    return _BookMetrics(
        midpoint=midpoint,
        spread=spread,
        relative_spread=spread / midpoint,
        microprice=microprice,
        top_depth_imbalance=_imbalance(bid_quantity, ask_quantity),
        depth5_imbalance=_imbalance(_depth(snapshot.bids, 5), _depth(snapshot.asks, 5)),
    )


def _two_sided(book: PolymarketRecordedBook) -> bool:
    return bool(book.snapshot.bids and book.snapshot.asks)


@dataclass(frozen=True, slots=True)
class _BookFlow:
    mid_log_return: float
    microprice_log_return: float
    top_ofi: float
    update_count: int


def _book_flow(
    books: Sequence[PolymarketRecordedBook],
    times: Sequence[int],
    *,
    decision_ms: int,
    window_ms: int,
) -> _BookFlow:
    end = bisect_left(times, decision_ms)
    start = max(0, bisect_left(times, decision_ms - window_ms, 0, end) - 1)
    selected = books[start:end]
    if len(selected) < 2:
        return _BookFlow(0.0, 0.0, 0.0, 0)
    current_segment = selected[-1].segment_id
    current_connection = selected[-1].connection_id
    selected = tuple(
        book
        for book in selected
        if book.segment_id == current_segment
        and book.connection_id == current_connection
        and _two_sided(book)
    )
    if len(selected) < 2:
        return _BookFlow(0.0, 0.0, 0.0, 0)
    first = _book_metrics(selected[0])
    last = _book_metrics(selected[-1])
    ofi = 0.0
    for previous, current in zip(selected, selected[1:], strict=False):
        p = previous.snapshot
        c = current.snapshot
        if not p.bids or not p.asks or not c.bids or not c.asks:
            continue
        previous_bid = float(p.bids[0].price)
        current_bid = float(c.bids[0].price)
        previous_ask = float(p.asks[0].price)
        current_ask = float(c.asks[0].price)
        bid_flow = (
            float(c.bids[0].quantity)
            if current_bid > previous_bid
            else (
                float(c.bids[0].quantity) - float(p.bids[0].quantity)
                if current_bid == previous_bid
                else -float(p.bids[0].quantity)
            )
        )
        ask_flow = (
            -float(c.asks[0].quantity)
            if current_ask < previous_ask
            else (
                float(p.asks[0].quantity) - float(c.asks[0].quantity)
                if current_ask == previous_ask
                else float(p.asks[0].quantity)
            )
        )
        ofi += bid_flow + ask_flow
    return _BookFlow(
        mid_log_return=math.log(last.midpoint / first.midpoint),
        microprice_log_return=math.log(last.microprice / first.microprice),
        top_ofi=ofi,
        update_count=len(selected) - 1,
    )


@dataclass(frozen=True, slots=True)
class Round27FeatureRow:
    schema_version: str
    run_id: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    values: tuple[float, ...]
    feature_names_sha256: str
    maximum_receipt_wall_ms: int
    source_chain_sha256: str
    row_sha256: str
    target_accessed: bool = False
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        condition_id: str,
        event_start_ms: int,
        decision_time_ms: int,
        market_prior_probability: float,
        values: Sequence[float],
        maximum_receipt_wall_ms: int,
        source_chain_sha256: str,
    ) -> "Round27FeatureRow":
        selected_values = tuple(float(value) for value in values)
        payload = {
            "schema_version": POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
            "run_id": str(run_id),
            "condition_id": str(condition_id).lower(),
            "event_start_ms": int(event_start_ms),
            "decision_time_ms": int(decision_time_ms),
            "market_prior_probability": float(market_prior_probability),
            "values": selected_values,
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "maximum_receipt_wall_ms": int(maximum_receipt_wall_ms),
            "source_chain_sha256": str(source_chain_sha256).lower(),
            "target_accessed": False,
            "trading_authority": False,
        }
        row = cls(**payload, row_sha256=_canonical_sha256(payload))
        return row.validated()

    def validated(self) -> "Round27FeatureRow":
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "decision_time_ms": self.decision_time_ms,
            "market_prior_probability": self.market_prior_probability,
            "values": self.values,
            "feature_names_sha256": self.feature_names_sha256,
            "maximum_receipt_wall_ms": self.maximum_receipt_wall_ms,
            "source_chain_sha256": self.source_chain_sha256,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }
        if (
            not self.run_id
            or self.schema_version != POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION
            or not self.condition_id.startswith("0x")
            or len(self.condition_id) != 66
            or self.event_start_ms <= 0
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or self.decision_time_ms % POLYMARKET_ROUND27_DECISION_STEP_MS
            or not 0.0 < self.market_prior_probability < 1.0
            or len(self.values) != len(POLYMARKET_ROUND27_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or self.feature_names_sha256 != POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
            or not 0 < self.maximum_receipt_wall_ms < self.decision_time_ms
            or len(self.source_chain_sha256) != 64
            or self.target_accessed
            or self.trading_authority
            or self.row_sha256 != _canonical_sha256(payload)
        ):
            raise ValueError("Round 27 feature row differs")
        return self


def _latest_index(times: Sequence[int], decision_ms: int) -> int:
    return bisect_left(times, decision_ms) - 1


def _eligible_decision_times(
    market: PolymarketFiveMinuteMarket,
    intervals: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    eligible: list[int] = []
    for interval in intervals:
        if interval.get("eligible") is not True:
            continue
        start = max(
            market.event_start_ms + POLYMARKET_ROUND27_FIRST_DECISION_OFFSET_MS,
            int(interval["interval_start_ms"]),
        )
        end = min(
            market.end_ms - POLYMARKET_ROUND27_LAST_DECISION_OFFSET_MS,
            int(interval["interval_end_ms"]),
        )
        aligned = (
            (start + POLYMARKET_ROUND27_DECISION_STEP_MS - 1)
            // POLYMARKET_ROUND27_DECISION_STEP_MS
            * POLYMARKET_ROUND27_DECISION_STEP_MS
        )
        eligible.extend(range(aligned, end + 1, POLYMARKET_ROUND27_DECISION_STEP_MS))
    return tuple(sorted(set(eligible)))


def build_round27_condition_features(
    *,
    market: PolymarketFiveMinuteMarket,
    books: Sequence[PolymarketRecordedBook],
    source: Round27PublicSourceSeries,
    eligible_intervals: Sequence[Mapping[str, object]],
) -> tuple[Round27FeatureRow, ...]:
    """Build target-blind one-second rows inside audited executable intervals."""

    selected_source = source.validated()
    if market.asset != "BTC" or market.end_ms - market.event_start_ms != 300_000:
        raise ValueError("Round 27 features require a BTC five-minute market")
    by_outcome = {
        outcome: tuple(
            sorted(
                (
                    book
                    for book in books
                    if book.market.condition_id == market.condition_id
                    and book.outcome == outcome
                ),
                key=lambda item: (item.received_wall_ms, item.received_monotonic_ns),
            )
        )
        for outcome in ("Up", "Down")
    }
    if not by_outcome["Up"] or not by_outcome["Down"]:
        raise ValueError("Round 27 condition lacks a two-outcome book")
    book_times = {
        outcome: tuple(item.received_wall_ms for item in values)
        for outcome, values in by_outcome.items()
    }
    trade_series = {"spot": selected_source.spot, "usdm": selected_source.usdm}
    trade_times = {
        name: values.received_wall_ms for name, values in trade_series.items()
    }
    twap_state = PolymarketTwap60FeatureState(
        minimum_return_count=10,
        minimum_coverage_seconds=10,
        maximum_source_age_ms=5_000,
    )
    for tick in selected_source.twap:
        if tick.received_wall_ms < market.end_ms:
            twap_state.observe(tick)
    twap_receipt_times = tuple(tick.received_wall_ms for tick in selected_source.twap)
    rows: list[Round27FeatureRow] = []
    for decision in _eligible_decision_times(market, eligible_intervals):
        indices = {
            outcome: _latest_index(book_times[outcome], decision)
            for outcome in ("Up", "Down")
        }
        if any(index < 0 for index in indices.values()):
            continue
        current_books = {
            outcome: by_outcome[outcome][indices[outcome]] for outcome in ("Up", "Down")
        }
        if (
            current_books["Up"].segment_id != current_books["Down"].segment_id
            or current_books["Up"].connection_id != current_books["Down"].connection_id
            or not _two_sided(current_books["Up"])
            or not _two_sided(current_books["Down"])
            or any(
                decision - book.received_wall_ms
                > POLYMARKET_ROUND27_MAXIMUM_BOOK_AGE_MS
                for book in current_books.values()
            )
        ):
            continue
        up = _book_metrics(current_books["Up"])
        down = _book_metrics(current_books["Down"])
        prior = up.midpoint / (up.midpoint + down.midpoint)
        latest_twap_index = _latest_index(twap_receipt_times, decision)
        if latest_twap_index < 0:
            continue
        latest_twap = selected_source.twap[latest_twap_index]
        twap = twap_state.features(
            market,
            observed_wall_ms=decision - 1,
            observed_monotonic_ns=max(
                current_books["Up"].received_monotonic_ns,
                current_books["Down"].received_monotonic_ns,
                latest_twap.received_monotonic_ns,
            ),
        )
        if not twap.available:
            continue
        assert twap.current_source_age_ms is not None
        values: list[float] = [
            (decision - market.event_start_ms) / 300_000.0,
            (market.end_ms - decision) / 1_000.0,
            _logit(prior),
            up.midpoint,
            down.midpoint,
            up.spread,
            down.spread,
            up.relative_spread,
            down.relative_spread,
            up.microprice - up.midpoint,
            down.microprice - down.midpoint,
            up.top_depth_imbalance,
            down.top_depth_imbalance,
            up.depth5_imbalance,
            down.depth5_imbalance,
            float(current_books["Up"].snapshot.asks[0].price)
            + float(current_books["Down"].snapshot.asks[0].price)
            - 1.0,
            1.0
            - float(current_books["Up"].snapshot.bids[0].price)
            - float(current_books["Down"].snapshot.bids[0].price),
            float(
                abs(
                    current_books["Up"].received_wall_ms
                    - current_books["Down"].received_wall_ms
                )
            ),
            float(twap.log_distance_from_open),
            float(twap.realized_variance_rate_per_second),
            float(twap.path_efficiency),
            float(twap.current_source_age_ms + 1),
        ]
        market_returns: dict[int, float] = {}
        for outcome in ("Up", "Down"):
            for window in POLYMARKET_ROUND27_BOOK_WINDOWS_MS:
                flow = _book_flow(
                    by_outcome[outcome],
                    book_times[outcome],
                    decision_ms=decision,
                    window_ms=window,
                )
                values.extend(
                    (
                        flow.mid_log_return,
                        flow.microprice_log_return,
                        flow.top_ofi,
                        float(flow.update_count),
                    )
                )
                if outcome == "Up":
                    market_returns[window] = flow.mid_log_return
        trade_returns: dict[str, dict[int, float]] = {"spot": {}, "usdm": {}}
        latest_points: dict[str, int] = {}
        long_context_complete = True
        for name in ("spot", "usdm"):
            latest = _latest_index(trade_times[name], decision)
            if latest < 0 or decision - trade_times[name][latest] > 1_500:
                break
            latest_points[name] = latest
            for window in POLYMARKET_ROUND27_TRADE_WINDOWS_MS:
                metrics = _trade_metrics(
                    trade_series[name],
                    decision_ms=decision,
                    window_ms=window,
                )
                trade_returns[name][window] = metrics.log_return
                if window == POLYMARKET_ROUND27_LONG_CONTEXT_WINDOW_MS and (
                    metrics.receipt_coverage_fraction
                    < POLYMARKET_ROUND27_LONG_CONTEXT_MINIMUM_COVERAGE
                    or metrics.maximum_receipt_gap_ms
                    > POLYMARKET_ROUND27_LONG_CONTEXT_MAXIMUM_RECEIPT_GAP_MS
                ):
                    long_context_complete = False
                values.extend(
                    (
                        metrics.log_return,
                        metrics.realized_variance,
                        metrics.signed_quote_imbalance,
                        metrics.log1p_quote_notional,
                        float(metrics.count),
                        metrics.receipt_coverage_fraction,
                        metrics.maximum_receipt_gap_ms,
                    )
                )
        if set(latest_points) != {"spot", "usdm"} or not long_context_complete:
            continue
        values.extend(
            (
                float(
                    decision
                    - trade_series["spot"].received_wall_ms[latest_points["spot"]]
                ),
                float(
                    decision
                    - trade_series["usdm"].received_wall_ms[latest_points["usdm"]]
                ),
                float(
                    trade_series["spot"].received_wall_ms[latest_points["spot"]]
                    - trade_series["usdm"].received_wall_ms[latest_points["usdm"]]
                ),
                10_000.0
                * math.log(
                    trade_series["spot"].price[latest_points["spot"]]
                    / trade_series["usdm"].price[latest_points["usdm"]]
                ),
            )
        )
        values.extend(
            trade_returns["spot"][window] - trade_returns["usdm"][window]
            for window in POLYMARKET_ROUND27_TRADE_WINDOWS_MS
        )
        values.extend(
            trade_returns["spot"][window] - market_returns[window]
            for window in POLYMARKET_ROUND27_BOOK_WINDOWS_MS
        )
        if len(values) != len(POLYMARKET_ROUND27_FEATURE_NAMES):
            raise RuntimeError("Round 27 feature width differs")
        maximum_receipt = max(
            current_books["Up"].received_wall_ms,
            current_books["Down"].received_wall_ms,
            int(trade_series["spot"].received_wall_ms[latest_points["spot"]]),
            int(trade_series["usdm"].received_wall_ms[latest_points["usdm"]]),
            latest_twap.received_wall_ms,
        )
        rows.append(
            Round27FeatureRow.create(
                run_id=selected_source.run_id,
                condition_id=market.condition_id,
                event_start_ms=market.event_start_ms,
                decision_time_ms=decision,
                market_prior_probability=prior,
                values=values,
                maximum_receipt_wall_ms=maximum_receipt,
                source_chain_sha256=selected_source.source_chain_sha256,
            )
        )
    return tuple(rows)


def materialize_round27_target_blind_features(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    condition_audit: Mapping[str, object],
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[tuple[Round27FeatureRow, ...], dict[str, object]]:
    """Materialize all audited conditions while refusing target-bearing evidence."""

    if (
        condition_audit.get("run_id") != run_id
        or condition_audit.get("target_free") is not True
        or condition_audit.get("model_data_eligible") is not False
        or condition_audit.get("edge_claim") is not False
        or condition_audit.get("profitability_claim") is not False
    ):
        raise ValueError("Round 27 condition audit authority differs")
    source = load_round27_public_source_series(
        store,
        run_id=run_id,
        progress=progress,
    )
    markets = {
        market.condition_id: market
        for market in PolymarketEvidenceReplay.load_markets(store, run_id=run_id)
    }
    rows: list[Round27FeatureRow] = []
    rejection_counts: dict[str, int] = {}
    admitted_conditions = 0
    conditions = condition_audit.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("Round 27 condition audit population differs")
    eligible_items = tuple(
        item
        for item in conditions
        if isinstance(item, Mapping) and item.get("eligible") is True
    )
    for batch_start in range(0, len(eligible_items), 32):
        batch = eligible_items[batch_start : batch_start + 32]
        batch_ids = tuple(str(item.get("condition_id") or "").lower() for item in batch)
        if any(condition_id not in markets for condition_id in batch_ids):
            raise ValueError("Round 27 audited condition identity differs")
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=run_id,
            allow_segmented_gaps=True,
            include_resolutions=False,
            book_sample_interval_ms=50,
            condition_ids=batch_ids,
            maximum_received_wall_ms_by_condition={
                condition_id: markets[condition_id].end_ms - 1
                for condition_id in batch_ids
            },
            materialized_minimum_depth_levels=5,
            cap_materialized_depth_to_minimum_order_size=False,
        )
        books_by_condition: dict[str, list[PolymarketRecordedBook]] = {
            condition_id: [] for condition_id in batch_ids
        }
        for book in replay.books:
            books_by_condition[book.market.condition_id].append(book)
        for batch_index, item in enumerate(batch, start=1):
            condition_id = str(item.get("condition_id") or "").lower()
            intervals = item.get("segments")
            if not isinstance(intervals, list):
                raise ValueError("Round 27 audited condition intervals differ")
            condition_rows = build_round27_condition_features(
                market=markets[condition_id],
                books=books_by_condition[condition_id],
                source=source,
                eligible_intervals=intervals,
            )
            if condition_rows:
                admitted_conditions += 1
                rows.extend(condition_rows)
            else:
                rejection_counts["no_complete_causal_feature_rows"] = (
                    rejection_counts.get("no_complete_causal_feature_rows", 0) + 1
                )
            if progress is not None:
                progress(
                    "condition-features",
                    {
                        "completed_condition_count": batch_start + batch_index,
                        "condition_count": len(eligible_items),
                        "condition_id": condition_id,
                        "feature_row_count": len(condition_rows),
                    },
                )
    keys = [(row.condition_id, row.decision_time_ms) for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("Round 27 feature population differs")
    chain = _EMPTY_SHA256
    for row in rows:
        chain = hashlib.sha256(
            bytes.fromhex(chain) + bytes.fromhex(row.row_sha256)
        ).hexdigest()
    report: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
        "run_id": run_id,
        "condition_audit_sha256": condition_audit.get("audit_sha256"),
        "source_chain_sha256": source.source_chain_sha256,
        "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "feature_count": len(POLYMARKET_ROUND27_FEATURE_NAMES),
        "eligible_condition_count": condition_audit.get("eligible_condition_count"),
        "admitted_condition_count": admitted_conditions,
        "feature_row_count": len(rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "row_chain_sha256": chain,
        "receipt_time_causal": True,
        "condition_weight_required": True,
        "official_resolution_accessed": False,
        "target_accessed": False,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return tuple(rows), report


__all__ = [
    "POLYMARKET_ROUND27_BOOK_WINDOWS_MS",
    "POLYMARKET_ROUND27_DECISION_STEP_MS",
    "POLYMARKET_ROUND27_FEATURE_NAMES",
    "POLYMARKET_ROUND27_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_LONG_CONTEXT_MAXIMUM_RECEIPT_GAP_MS",
    "POLYMARKET_ROUND27_LONG_CONTEXT_MINIMUM_COVERAGE",
    "POLYMARKET_ROUND27_LONG_CONTEXT_WINDOW_MS",
    "POLYMARKET_ROUND27_TRADE_WINDOWS_MS",
    "Round27FeatureRow",
    "Round27PublicSourceSeries",
    "Round27TradePoint",
    "Round27TradeSeries",
    "build_round27_condition_features",
    "load_round27_public_source_series",
    "materialize_round27_target_blind_features",
]
