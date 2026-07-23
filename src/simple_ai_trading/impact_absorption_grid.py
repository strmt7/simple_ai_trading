"""Causal one-second feature grid for admitted Round 73 corpus runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Mapping, Sequence


ROUND73_GRID_SCHEMA_VERSION = "round-073-causal-grid-v3"
ROUND73_GRID_CONTRACT_SHA256 = (
    "9dd830896de053fded1468040011fde618c990ad98a5807007c46f2e22553c4a"
)
ROUND73_GRID_WINDOWS_MS = (100, 250, 500, 1_000, 5_000, 15_000, 60_000)
ROUND73_GRID_BANDS = (
    "levels_1_5",
    "levels_6_10",
    "levels_11_20",
    "outside_20",
)
ROUND73_GRID_STEP_NS = 1_000_000_000
ROUND73_GRID_WARMUP_NS = 60_000_000_000

_STATE_FEATURE_NAMES = (
    "spread_bps",
    "bid_quote_notional",
    "ask_quote_notional",
    "l1_imbalance",
    "microprice_offset_bps",
    "bbo_age_ms",
    "bbo_corrected_event_latency_ms",
    "bid_depth_quote_5",
    "ask_depth_quote_5",
    "bid_depth_quote_10",
    "ask_depth_quote_10",
    "bid_depth_quote_20",
    "ask_depth_quote_20",
    "imbalance_5",
    "imbalance_10",
    "imbalance_20",
    "bid_depth_5_share_of_20",
    "ask_depth_5_share_of_20",
    "bid_distance_weighted_depth_20",
    "ask_distance_weighted_depth_20",
    "bid_depth_concentration_20",
    "ask_depth_concentration_20",
    "l2_age_ms",
    "l2_corrected_event_latency_ms",
    "mark_to_mid_bps",
    "index_to_mid_bps",
    "funding_rate",
    "seconds_to_next_funding",
    "mark_age_ms",
    "open_interest",
    "open_interest_age_ms",
    "utc_second_of_day_sine",
    "utc_second_of_day_cosine",
)


def _window_feature_names(window_ms: int) -> tuple[str, ...]:
    prefix = f"w{window_ms}ms_"
    output = [
        "buy_aggressive_quote",
        "sell_aggressive_quote",
        "signed_aggressive_quote",
        "absolute_aggressive_quote",
        "aggregate_trade_count",
        "buyer_taker_share",
    ]
    for side in ("bid", "ask"):
        for action in ("added", "removed"):
            output.extend(
                f"{side}_{action}_quote_{band}" for band in ROUND73_GRID_BANDS
            )
    output.extend(
        f"normalized_order_flow_imbalance_{band}" for band in ROUND73_GRID_BANDS[:3]
    )
    output.extend(
        (
            "mid_log_return",
            "mid_realized_variance",
            "bbo_update_count",
            "mean_spread_bps",
            "maximum_spread_bps",
            "liquidation_snapshot_count",
            "liquidation_observed_quote_notional",
        )
    )
    return tuple(prefix + name for name in output)


ROUND73_GRID_FEATURE_NAMES = _STATE_FEATURE_NAMES + tuple(
    name
    for window_ms in ROUND73_GRID_WINDOWS_MS
    for name in _window_feature_names(window_ms)
)
ROUND73_GRID_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        ROUND73_GRID_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()

_INVALID_BITS = {
    "missing_bbo": 1 << 0,
    "stale_bbo": 1 << 1,
    "invalid_bbo": 1 << 2,
    "negative_bbo_latency": 1 << 3,
    "missing_l2": 1 << 4,
    "stale_l2": 1 << 5,
    "incomplete_l2": 1 << 6,
    "nonmonotone_l2_depth": 1 << 7,
    "negative_l2_latency": 1 << 8,
    "missing_mark": 1 << 9,
    "stale_mark": 1 << 10,
    "invalid_mark": 1 << 11,
    "missing_open_interest": 1 << 12,
    "stale_open_interest": 1 << 13,
    "invalid_open_interest": 1 << 14,
    "future_state": 1 << 15,
    "nonfinite_feature": 1 << 16,
    "invalid_l2_geometry": 1 << 17,
}


def round73_grid_invalid_reasons(mask: int) -> tuple[str, ...]:
    selected = int(mask)
    return tuple(name for name, bit in _INVALID_BITS.items() if selected & bit)


@dataclass(frozen=True)
class Round73BboState:
    received_monotonic_ns: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    corrected_event_latency_ms: float


@dataclass(frozen=True)
class Round73L2State:
    received_monotonic_ns: int
    bid_prices: tuple[float, ...]
    bid_quantities: tuple[float, ...]
    ask_prices: tuple[float, ...]
    ask_quantities: tuple[float, ...]
    bid_depth_quote_5: float
    ask_depth_quote_5: float
    bid_depth_quote_10: float
    ask_depth_quote_10: float
    bid_depth_quote_20: float
    ask_depth_quote_20: float
    imbalance_5: float
    imbalance_10: float
    imbalance_20: float
    corrected_event_latency_ms: float


@dataclass(frozen=True)
class Round73MarkState:
    received_monotonic_ns: int
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time_ms: int


@dataclass(frozen=True)
class Round73OpenInterestState:
    received_monotonic_ns: int
    open_interest: float


@dataclass(frozen=True)
class Round73GridAnchor:
    symbol: str
    anchor_monotonic_ns: int
    anchor_wall_ns: int
    source_max_received_monotonic_ns: int
    valid: bool
    invalid_reason_mask: int
    invalid_reasons: tuple[str, ...]
    signed_aggressive_quote_1s: float
    absolute_aggressive_quote_1s: float
    trailing_median_absolute_aggressive_quote_60s: float | None
    shock_ratio: float | None
    shock_direction: int
    shock_direction_taker_share: float
    feature_values: tuple[float, ...] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": ROUND73_GRID_SCHEMA_VERSION,
            "contract_sha256": ROUND73_GRID_CONTRACT_SHA256,
            "feature_names_sha256": ROUND73_GRID_FEATURE_NAMES_SHA256,
            "symbol": self.symbol,
            "anchor_monotonic_ns": self.anchor_monotonic_ns,
            "anchor_wall_ns": self.anchor_wall_ns,
            "source_max_received_monotonic_ns": (self.source_max_received_monotonic_ns),
            "valid": self.valid,
            "invalid_reason_mask": self.invalid_reason_mask,
            "invalid_reasons": list(self.invalid_reasons),
            "signed_aggressive_quote_1s": self.signed_aggressive_quote_1s,
            "absolute_aggressive_quote_1s": self.absolute_aggressive_quote_1s,
            "trailing_median_absolute_aggressive_quote_60s": (
                self.trailing_median_absolute_aggressive_quote_60s
            ),
            "shock_ratio": self.shock_ratio,
            "shock_direction": self.shock_direction,
            "shock_direction_taker_share": self.shock_direction_taker_share,
            "feature_values": (
                None if self.feature_values is None else list(self.feature_values)
            ),
            "target_constructed": False,
            "model_evaluated": False,
        }


class _WindowState:
    def __init__(self, window_ns: int) -> None:
        self.window_ns = int(window_ns)
        self.trades: deque[tuple[int, float, float]] = deque()
        self.buy_quote = 0.0
        self.sell_quote = 0.0
        self.depth: deque[tuple[int, tuple[float, ...]]] = deque()
        self.depth_sums = [0.0] * 16
        self.bbo: deque[tuple[int, float, float, float]] = deque()
        self.spread_sum = 0.0
        self.spread_max: deque[tuple[int, float]] = deque()
        self.liquidations: deque[tuple[int, float]] = deque()
        self.liquidation_quote = 0.0

    def observe_trade(
        self, timestamp_ns: int, buy_quote: float, sell_quote: float
    ) -> None:
        self.trades.append((timestamp_ns, buy_quote, sell_quote))
        self.buy_quote += buy_quote
        self.sell_quote += sell_quote

    def observe_depth(self, timestamp_ns: int, values: tuple[float, ...]) -> None:
        if len(values) != 16:
            raise ValueError("Round 73 depth-band event width must be 16")
        self.depth.append((timestamp_ns, values))
        for index, value in enumerate(values):
            self.depth_sums[index] += value

    def observe_bbo(
        self,
        timestamp_ns: int,
        log_mid: float,
        cumulative_variation: float,
        spread_bps: float,
    ) -> None:
        self.bbo.append((timestamp_ns, log_mid, cumulative_variation, spread_bps))
        self.spread_sum += spread_bps
        while self.spread_max and self.spread_max[-1][1] <= spread_bps:
            self.spread_max.pop()
        self.spread_max.append((timestamp_ns, spread_bps))

    def observe_liquidation(self, timestamp_ns: int, quote_notional: float) -> None:
        self.liquidations.append((timestamp_ns, quote_notional))
        self.liquidation_quote += quote_notional

    def prune(self, anchor_ns: int) -> None:
        cutoff = int(anchor_ns) - self.window_ns
        while self.trades and self.trades[0][0] < cutoff:
            _timestamp, buy_quote, sell_quote = self.trades.popleft()
            self.buy_quote -= buy_quote
            self.sell_quote -= sell_quote
        while self.depth and self.depth[0][0] < cutoff:
            _timestamp, values = self.depth.popleft()
            for index, value in enumerate(values):
                self.depth_sums[index] -= value
        while self.bbo and self.bbo[0][0] < cutoff:
            _timestamp, _log_mid, _variation, spread = self.bbo.popleft()
            self.spread_sum -= spread
        while self.spread_max and self.spread_max[0][0] < cutoff:
            self.spread_max.popleft()
        while self.liquidations and self.liquidations[0][0] < cutoff:
            _timestamp, quote_notional = self.liquidations.popleft()
            self.liquidation_quote -= quote_notional

    def feature_values(
        self,
        *,
        latest_spread_bps: float,
        band_depths: Sequence[float],
    ) -> tuple[float, ...]:
        absolute_quote = self.buy_quote + self.sell_quote
        signed_quote = self.buy_quote - self.sell_quote
        buyer_share = self.buy_quote / absolute_quote if absolute_quote > 0 else 0.0
        output = [
            self.buy_quote,
            self.sell_quote,
            signed_quote,
            absolute_quote,
            float(len(self.trades)),
            buyer_share,
            *self.depth_sums,
        ]
        for band_index in range(3):
            bid_added = self.depth_sums[band_index]
            bid_removed = self.depth_sums[4 + band_index]
            ask_added = self.depth_sums[8 + band_index]
            ask_removed = self.depth_sums[12 + band_index]
            imbalance = bid_added + ask_removed - bid_removed - ask_added
            output.append(imbalance / max(float(band_depths[band_index]), 1e-12))
        if len(self.bbo) >= 2:
            mid_return = self.bbo[-1][1] - self.bbo[0][1]
            variation = self.bbo[-1][2] - self.bbo[0][2]
        else:
            mid_return = 0.0
            variation = 0.0
        if self.bbo:
            mean_spread = self.spread_sum / len(self.bbo)
            maximum_spread = self.spread_max[0][1]
        else:
            mean_spread = latest_spread_bps
            maximum_spread = latest_spread_bps
        output.extend(
            (
                mid_return,
                variation,
                float(len(self.bbo)),
                mean_spread,
                maximum_spread,
                float(len(self.liquidations)),
                self.liquidation_quote,
            )
        )
        return tuple(output)


class Round73CausalGridAccumulator:
    """Maintain strictly causal state and bounded rolling windows for one symbol."""

    def __init__(self, symbol: str) -> None:
        selected = str(symbol).strip().upper()
        if selected not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
            raise ValueError("Round 73 grid symbol is unsupported")
        self.symbol = selected
        self.windows = {
            window_ms: _WindowState(window_ms * 1_000_000)
            for window_ms in ROUND73_GRID_WINDOWS_MS
        }
        self.latest_bbo: Round73BboState | None = None
        self.latest_l2: Round73L2State | None = None
        self.latest_mark: Round73MarkState | None = None
        self.latest_open_interest: Round73OpenInterestState | None = None
        self.source_max_received_monotonic_ns = 0
        self._last_observed_monotonic_ns = 0
        self._last_log_mid: float | None = None
        self._cumulative_mid_variance = 0.0
        self._prior_anchor_absolute_quote: deque[float] = deque(maxlen=60)
        self._last_anchor_monotonic_ns = 0

    def _observe_timestamp(self, received_monotonic_ns: int) -> int:
        selected = int(received_monotonic_ns)
        if selected < 1:
            raise ValueError("Round 73 grid receipt time must be positive")
        if selected < self._last_observed_monotonic_ns:
            raise ValueError("Round 73 grid events must be ordered by receipt time")
        self._last_observed_monotonic_ns = selected
        self.source_max_received_monotonic_ns = selected
        return selected

    def observe_bbo(
        self,
        *,
        received_monotonic_ns: int,
        bid: float,
        bid_qty: float,
        ask: float,
        ask_qty: float,
        corrected_event_latency_ms: float,
    ) -> None:
        timestamp = self._observe_timestamp(received_monotonic_ns)
        state = Round73BboState(
            received_monotonic_ns=timestamp,
            bid=float(bid),
            bid_qty=float(bid_qty),
            ask=float(ask),
            ask_qty=float(ask_qty),
            corrected_event_latency_ms=float(corrected_event_latency_ms),
        )
        self.latest_bbo = state
        mid = (state.bid + state.ask) / 2.0
        if not math.isfinite(mid) or mid <= 0:
            return
        log_mid = math.log(mid)
        if self._last_log_mid is not None:
            log_return = log_mid - self._last_log_mid
            self._cumulative_mid_variance += log_return * log_return
        self._last_log_mid = log_mid
        spread_bps = (state.ask - state.bid) / mid * 10_000.0
        for window in self.windows.values():
            window.observe_bbo(
                timestamp,
                log_mid,
                self._cumulative_mid_variance,
                spread_bps,
            )

    def observe_l2(
        self,
        *,
        state: Round73L2State,
        depth_band_flow: Mapping[str, Mapping[str, Mapping[str, float]]],
    ) -> None:
        timestamp = self._observe_timestamp(state.received_monotonic_ns)
        if timestamp != state.received_monotonic_ns:
            raise ValueError("Round 73 L2 receipt identity differs")
        self.latest_l2 = state
        values: list[float] = []
        for side in ("bid", "ask"):
            for action in ("added_quote", "removed_quote"):
                for band in ROUND73_GRID_BANDS:
                    values.append(float(depth_band_flow[side][band][action]))
        selected = tuple(values)
        for window in self.windows.values():
            window.observe_depth(timestamp, selected)

    def observe_trade(
        self,
        *,
        received_monotonic_ns: int,
        price: float,
        quantity: float,
        buyer_is_maker: bool,
    ) -> None:
        timestamp = self._observe_timestamp(received_monotonic_ns)
        quote_notional = float(price) * float(quantity)
        buy_quote = 0.0 if buyer_is_maker else quote_notional
        sell_quote = quote_notional if buyer_is_maker else 0.0
        for window in self.windows.values():
            window.observe_trade(timestamp, buy_quote, sell_quote)

    def observe_mark(self, state: Round73MarkState) -> None:
        self._observe_timestamp(state.received_monotonic_ns)
        self.latest_mark = state

    def observe_open_interest(self, state: Round73OpenInterestState) -> None:
        self._observe_timestamp(state.received_monotonic_ns)
        self.latest_open_interest = state

    def observe_liquidation(
        self,
        *,
        received_monotonic_ns: int,
        price: float,
        last_filled_quantity: float,
    ) -> None:
        timestamp = self._observe_timestamp(received_monotonic_ns)
        quote_notional = float(price) * float(last_filled_quantity)
        for window in self.windows.values():
            window.observe_liquidation(timestamp, quote_notional)

    @staticmethod
    def _age_ms(anchor_ns: int, received_ns: int) -> float:
        return (int(anchor_ns) - int(received_ns)) / 1_000_000.0

    def _invalid_mask(self, anchor_ns: int) -> int:
        mask = 0
        bbo = self.latest_bbo
        if bbo is None:
            mask |= _INVALID_BITS["missing_bbo"]
        else:
            age = self._age_ms(anchor_ns, bbo.received_monotonic_ns)
            if age < 0:
                mask |= _INVALID_BITS["future_state"]
            elif age > 1_000:
                mask |= _INVALID_BITS["stale_bbo"]
            if (
                bbo.bid <= 0
                or bbo.ask <= bbo.bid
                or bbo.bid_qty <= 0
                or bbo.ask_qty <= 0
            ):
                mask |= _INVALID_BITS["invalid_bbo"]
            if bbo.corrected_event_latency_ms < 0:
                mask |= _INVALID_BITS["negative_bbo_latency"]
        l2 = self.latest_l2
        if l2 is None:
            mask |= _INVALID_BITS["missing_l2"]
        else:
            age = self._age_ms(anchor_ns, l2.received_monotonic_ns)
            if age < 0:
                mask |= _INVALID_BITS["future_state"]
            elif age > 1_000:
                mask |= _INVALID_BITS["stale_l2"]
            arrays = (
                l2.bid_prices,
                l2.bid_quantities,
                l2.ask_prices,
                l2.ask_quantities,
            )
            complete = all(len(values) >= 20 for values in arrays)
            if not complete or any(
                not math.isfinite(value) or value <= 0
                for values in arrays
                for value in values[:20]
            ):
                mask |= _INVALID_BITS["incomplete_l2"]
            elif (
                any(
                    left <= right
                    for left, right in zip(
                        l2.bid_prices[:20], l2.bid_prices[1:20], strict=False
                    )
                )
                or any(
                    left >= right
                    for left, right in zip(
                        l2.ask_prices[:20], l2.ask_prices[1:20], strict=False
                    )
                )
                or l2.bid_prices[0] >= l2.ask_prices[0]
                or any(
                    not math.isfinite(value) or not -1.0 <= value <= 1.0
                    for value in (l2.imbalance_5, l2.imbalance_10, l2.imbalance_20)
                )
            ):
                mask |= _INVALID_BITS["invalid_l2_geometry"]
            if not (
                0
                < l2.bid_depth_quote_5
                <= l2.bid_depth_quote_10
                <= l2.bid_depth_quote_20
                and 0
                < l2.ask_depth_quote_5
                <= l2.ask_depth_quote_10
                <= l2.ask_depth_quote_20
            ):
                mask |= _INVALID_BITS["nonmonotone_l2_depth"]
            if l2.corrected_event_latency_ms < 0:
                mask |= _INVALID_BITS["negative_l2_latency"]
        mark = self.latest_mark
        if mark is None:
            mask |= _INVALID_BITS["missing_mark"]
        else:
            age = self._age_ms(anchor_ns, mark.received_monotonic_ns)
            if age < 0:
                mask |= _INVALID_BITS["future_state"]
            elif age > 3_000:
                mask |= _INVALID_BITS["stale_mark"]
            if mark.mark_price <= 0 or mark.index_price <= 0:
                mask |= _INVALID_BITS["invalid_mark"]
        open_interest = self.latest_open_interest
        if open_interest is None:
            mask |= _INVALID_BITS["missing_open_interest"]
        else:
            age = self._age_ms(anchor_ns, open_interest.received_monotonic_ns)
            if age < 0:
                mask |= _INVALID_BITS["future_state"]
            elif age > 90_000:
                mask |= _INVALID_BITS["stale_open_interest"]
            if open_interest.open_interest <= 0:
                mask |= _INVALID_BITS["invalid_open_interest"]
        return mask

    @staticmethod
    def _depth_shape(
        prices: Sequence[float],
        quantities: Sequence[float],
    ) -> tuple[float, float]:
        quote = tuple(
            float(price) * float(quantity)
            for price, quantity in zip(prices[:20], quantities[:20])
        )
        total = math.fsum(quote)
        weighted = math.fsum(value / rank for rank, value in enumerate(quote, start=1))
        concentration = (
            math.fsum((value / total) ** 2 for value in quote)
            if total > 0
            else math.nan
        )
        return weighted, concentration

    def _state_values(self, anchor_ns: int, anchor_wall_ns: int) -> tuple[float, ...]:
        bbo = self.latest_bbo
        l2 = self.latest_l2
        mark = self.latest_mark
        open_interest = self.latest_open_interest
        if bbo is None or l2 is None or mark is None or open_interest is None:
            return ()
        mid = (bbo.bid + bbo.ask) / 2.0
        spread_bps = (bbo.ask - bbo.bid) / mid * 10_000.0
        bid_quote = bbo.bid * bbo.bid_qty
        ask_quote = bbo.ask * bbo.ask_qty
        top_total = bbo.bid_qty + bbo.ask_qty
        l1_imbalance = (bbo.bid_qty - bbo.ask_qty) / top_total
        microprice = (bbo.ask * bbo.bid_qty + bbo.bid * bbo.ask_qty) / top_total
        bid_weighted, bid_concentration = self._depth_shape(
            l2.bid_prices,
            l2.bid_quantities,
        )
        ask_weighted, ask_concentration = self._depth_shape(
            l2.ask_prices,
            l2.ask_quantities,
        )
        utc_seconds = (int(anchor_wall_ns) % 86_400_000_000_000) / 1_000_000_000
        phase = math.tau * utc_seconds / 86_400.0
        return (
            spread_bps,
            bid_quote,
            ask_quote,
            l1_imbalance,
            (microprice - mid) / mid * 10_000.0,
            self._age_ms(anchor_ns, bbo.received_monotonic_ns),
            bbo.corrected_event_latency_ms,
            l2.bid_depth_quote_5,
            l2.ask_depth_quote_5,
            l2.bid_depth_quote_10,
            l2.ask_depth_quote_10,
            l2.bid_depth_quote_20,
            l2.ask_depth_quote_20,
            l2.imbalance_5,
            l2.imbalance_10,
            l2.imbalance_20,
            l2.bid_depth_quote_5 / l2.bid_depth_quote_20,
            l2.ask_depth_quote_5 / l2.ask_depth_quote_20,
            bid_weighted,
            ask_weighted,
            bid_concentration,
            ask_concentration,
            self._age_ms(anchor_ns, l2.received_monotonic_ns),
            l2.corrected_event_latency_ms,
            (mark.mark_price - mid) / mid * 10_000.0,
            (mark.index_price - mid) / mid * 10_000.0,
            mark.funding_rate,
            (mark.next_funding_time_ms * 1_000_000 - anchor_wall_ns) / 1_000_000_000.0,
            self._age_ms(anchor_ns, mark.received_monotonic_ns),
            open_interest.open_interest,
            self._age_ms(anchor_ns, open_interest.received_monotonic_ns),
            math.sin(phase),
            math.cos(phase),
        )

    def emit(
        self, *, anchor_monotonic_ns: int, anchor_wall_ns: int
    ) -> Round73GridAnchor:
        anchor_ns = int(anchor_monotonic_ns)
        if self._last_anchor_monotonic_ns and (
            anchor_ns - self._last_anchor_monotonic_ns != ROUND73_GRID_STEP_NS
        ):
            raise ValueError(
                "Round 73 grid anchors must be contiguous one-second steps"
            )
        if self.source_max_received_monotonic_ns >= anchor_ns:
            raise ValueError(
                "Round 73 grid anchor must strictly follow every observed receipt"
            )
        for window in self.windows.values():
            window.prune(anchor_ns)
        one_second = self.windows[1_000]
        signed_quote = one_second.buy_quote - one_second.sell_quote
        absolute_quote = one_second.buy_quote + one_second.sell_quote
        prior_median = (
            float(statistics.median(self._prior_anchor_absolute_quote))
            if len(self._prior_anchor_absolute_quote) == 60
            else None
        )
        shock_ratio = (
            abs(signed_quote) / prior_median
            if prior_median is not None and prior_median > 0
            else None
        )
        shock_direction = 1 if signed_quote > 0 else -1 if signed_quote < 0 else 0
        direction_share = (
            max(one_second.buy_quote, one_second.sell_quote) / absolute_quote
            if absolute_quote > 0
            else 0.0
        )
        invalid_mask = self._invalid_mask(anchor_ns)
        state_values = (
            self._state_values(anchor_ns, int(anchor_wall_ns))
            if invalid_mask == 0
            else ()
        )
        feature_values: tuple[float, ...] | None = None
        if invalid_mask == 0 and state_values:
            l2 = self.latest_l2
            bbo = self.latest_bbo
            if l2 is None or bbo is None:
                raise RuntimeError("Round 73 validated state disappeared")
            band_depths = (
                l2.bid_depth_quote_5 + l2.ask_depth_quote_5,
                (l2.bid_depth_quote_10 - l2.bid_depth_quote_5)
                + (l2.ask_depth_quote_10 - l2.ask_depth_quote_5),
                (l2.bid_depth_quote_20 - l2.bid_depth_quote_10)
                + (l2.ask_depth_quote_20 - l2.ask_depth_quote_10),
            )
            values = list(state_values)
            latest_spread = (bbo.ask - bbo.bid) / ((bbo.ask + bbo.bid) / 2) * 10_000
            for window_ms in ROUND73_GRID_WINDOWS_MS:
                values.extend(
                    self.windows[window_ms].feature_values(
                        latest_spread_bps=latest_spread,
                        band_depths=band_depths,
                    )
                )
            if len(values) != len(ROUND73_GRID_FEATURE_NAMES) or not all(
                math.isfinite(value) for value in values
            ):
                invalid_mask |= _INVALID_BITS["nonfinite_feature"]
            else:
                feature_values = tuple(values)
        self._prior_anchor_absolute_quote.append(absolute_quote)
        self._last_anchor_monotonic_ns = anchor_ns
        reasons = round73_grid_invalid_reasons(invalid_mask)
        return Round73GridAnchor(
            symbol=self.symbol,
            anchor_monotonic_ns=anchor_ns,
            anchor_wall_ns=int(anchor_wall_ns),
            source_max_received_monotonic_ns=(self.source_max_received_monotonic_ns),
            valid=invalid_mask == 0,
            invalid_reason_mask=invalid_mask,
            invalid_reasons=reasons,
            signed_aggressive_quote_1s=signed_quote,
            absolute_aggressive_quote_1s=absolute_quote,
            trailing_median_absolute_aggressive_quote_60s=prior_median,
            shock_ratio=shock_ratio,
            shock_direction=shock_direction,
            shock_direction_taker_share=direction_share,
            feature_values=feature_values,
        )


__all__ = [
    "ROUND73_GRID_BANDS",
    "ROUND73_GRID_CONTRACT_SHA256",
    "ROUND73_GRID_FEATURE_NAMES",
    "ROUND73_GRID_FEATURE_NAMES_SHA256",
    "ROUND73_GRID_SCHEMA_VERSION",
    "ROUND73_GRID_STEP_NS",
    "ROUND73_GRID_WARMUP_NS",
    "ROUND73_GRID_WINDOWS_MS",
    "Round73BboState",
    "Round73CausalGridAccumulator",
    "Round73GridAnchor",
    "Round73L2State",
    "Round73MarkState",
    "Round73OpenInterestState",
    "round73_grid_invalid_reasons",
]
