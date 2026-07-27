"""Causal subsecond event sequences for the Round 74 v10 research path.

This module is deliberately feature-only. It replays exact public receipts in
local arrival order and never constructs targets, fits a model, or authorizes an
order. UTC is used only for a continuous-market cyclic covariate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterator, Mapping, Sequence

from .assets import normalize_symbol
from .impact_absorption import (
    L2BookState,
    MarkPriceEvent,
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
    pre_event_level_band,
    validate_combined_stream_name,
)
from .impact_capture_frame import ImpactCaptureFrameRecord


ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION = "round-074-causal-event-sequence-v4"
ROUND74_EVENT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ROUND74_EVENT_SEQUENCE_LENGTH = 128
ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS = (1, 5, 30, 300)
ROUND74_EVENT_PAYOFF_SIDES = ("long", "short")
ROUND74_EVENT_PAYOFF_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
ROUND74_EVENT_STATE_HALF_LIVES_SECONDS = (5, 30, 300)
ROUND74_EVENT_TYPES = (
    "depthUpdate",
    "bookTicker",
    "aggTrade",
    "markPriceUpdate",
    "forceOrder",
)
ROUND74_EVENT_BANDS = (
    "levels_1_5",
    "levels_6_10",
    "levels_11_20",
    "outside_20",
)
ROUND74_EVENT_MAX_TIME_SINCE_MS = 60_000.0
ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS = 5_000_000_000

ROUND74_EVENT_FEATURE_NAMES = (
    "event_is_depth_update",
    "event_is_book_ticker",
    "event_is_aggregate_trade",
    "event_is_mark_price",
    "event_is_liquidation",
    "symbol_is_btcusdt",
    "symbol_is_ethusdt",
    "symbol_is_solusdt",
    "depth_update_is_stale",
    "log1p_interarrival_us",
    "spread_bps",
    "l1_imbalance",
    "microprice_offset_bps",
    "l2_imbalance_5",
    "l2_imbalance_10",
    "l2_imbalance_20",
    "bid_depth_5_share_of_20",
    "ask_depth_5_share_of_20",
    "log1p_bid_depth_quote_20",
    "log1p_ask_depth_quote_20",
    "mid_log_return_bps",
    *(f"depth_signed_pressure_{band}_scaled" for band in ROUND74_EVENT_BANDS),
    *(f"depth_absolute_flow_{band}_scaled" for band in ROUND74_EVENT_BANDS),
    "trade_signed_quote_scaled",
    "trade_absolute_quote_scaled",
    "trade_price_to_mid_bps",
    "mark_to_mid_bps",
    "index_to_mid_bps",
    "funding_rate_bps",
    "liquidation_signed_quote_scaled",
    "bbo_bid_qty_change_scaled",
    "bbo_ask_qty_change_scaled",
    *(
        feature_name
        for half_life in ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
        for feature_name in (
            f"ewm_return_projection_{half_life}s_bps",
            f"ewm_realized_volatility_{half_life}s_bps",
            f"ewm_signed_trade_pressure_per_second_{half_life}s",
            f"ewm_signed_depth_pressure_per_second_{half_life}s",
            f"ewm_signed_liquidation_pressure_per_second_{half_life}s",
            f"ewm_spread_{half_life}s_bps",
            f"ewm_l1_imbalance_{half_life}s",
        )
    ),
    "log1p_ms_since_depth_update",
    "log1p_ms_since_book_ticker",
    "log1p_ms_since_aggregate_trade",
    "log1p_ms_since_mark_price",
    "log1p_ms_since_liquidation",
    "utc_second_of_day_sine",
    "utc_second_of_day_cosine",
)
ROUND74_EVENT_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        ROUND74_EVENT_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


def _strict_json_object(raw_text: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("event-sequence payload is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("event-sequence payload must be a JSON object")
    return parsed


def _decode_websocket_record(
    record: ImpactCaptureFrameRecord,
) -> tuple[str, str, Mapping[str, object]]:
    if record.stream not in {
        "binance_futures_public",
        "binance_futures_market",
    }:
        raise ValueError("event-sequence record is not a WebSocket market receipt")
    root = _strict_json_object(record.raw_text)
    stream_name = root.get("stream")
    payload = root.get("data")
    if not isinstance(stream_name, str) or not isinstance(payload, Mapping):
        raise ValueError("event-sequence WebSocket wrapper is incomplete")
    event_type = str(payload.get("e", ""))
    symbol_value = payload.get("s")
    if event_type == "forceOrder" and isinstance(payload.get("o"), Mapping):
        symbol_value = payload["o"].get("s")
    symbol = normalize_symbol(symbol_value, default="")
    if not symbol:
        raise ValueError("event-sequence symbol is missing or invalid")
    expected_lane = (
        "binance_futures_public"
        if event_type in {"depthUpdate", "bookTicker"}
        else "binance_futures_market"
    )
    if event_type not in ROUND74_EVENT_TYPES or record.stream != expected_lane:
        raise ValueError("event-sequence type is unsupported or on the wrong lane")
    validate_combined_stream_name(
        stream_name,
        event_type=event_type,
        symbol=symbol,
    )
    return event_type, symbol, payload


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("event-sequence scaling denominator must be positive")
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        raise ValueError("event-sequence ratio is nonfinite")
    return value


def _signed_log1p(value: float) -> float:
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError("event-sequence signed transform input is nonfinite")
    return math.copysign(math.log1p(abs(selected)), selected)


def _bps(numerator: float, denominator: float) -> float:
    return _safe_ratio(numerator, denominator) * 10_000.0


def _depth_total(state: L2BookState) -> float:
    total = float(state.bid_depth_quote_20 + state.ask_depth_quote_20)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("event-sequence top-20 depth is invalid")
    return total


def _time_since_feature(now_ns: int, prior_ns: int | None) -> float:
    if prior_ns is None:
        elapsed_ms = ROUND74_EVENT_MAX_TIME_SINCE_MS
    else:
        elapsed_ms = max(0.0, (int(now_ns) - int(prior_ns)) / 1_000_000.0)
        elapsed_ms = min(ROUND74_EVENT_MAX_TIME_SINCE_MS, elapsed_ms)
    return math.log1p(elapsed_ms)


class _Round74TimeScaleState:
    """Continuous-time causal state with no fixed event-rate assumption."""

    def __init__(self) -> None:
        self._prior_ns: int | None = None
        self._prior_spread_bps = 0.0
        self._prior_l1_imbalance = 0.0
        self._clear_accumulators()

    def _clear_accumulators(self) -> None:
        count = len(ROUND74_EVENT_STATE_HALF_LIVES_SECONDS)
        self._exposure_seconds = [0.0] * count
        self._return_bps = [0.0] * count
        self._return_squared_bps = [0.0] * count
        self._trade_pressure = [0.0] * count
        self._depth_pressure = [0.0] * count
        self._liquidation_pressure = [0.0] * count
        self._spread_integral = [0.0] * count
        self._l1_imbalance_integral = [0.0] * count

    def update(
        self,
        *,
        now_ns: int,
        mid_log_return_bps: float,
        trade_pressure: float,
        depth_pressure: float,
        liquidation_pressure: float,
        spread_bps: float,
        l1_imbalance: float,
    ) -> tuple[float, ...]:
        selected_ns = int(now_ns)
        values = (
            float(mid_log_return_bps),
            float(trade_pressure),
            float(depth_pressure),
            float(liquidation_pressure),
            float(spread_bps),
            float(l1_imbalance),
        )
        if selected_ns < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("Round 74 time-scale state input is invalid")
        gap_reset = self._prior_ns is None
        if self._prior_ns is not None and selected_ns < self._prior_ns:
            raise ValueError("Round 74 time-scale state receipt order regressed")
        if (
            self._prior_ns is not None
            and selected_ns - self._prior_ns > ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS
        ):
            gap_reset = True
        if gap_reset:
            self._clear_accumulators()
            self._prior_ns = selected_ns
            self._prior_spread_bps = values[4]
            self._prior_l1_imbalance = values[5]
        prior_ns = self._prior_ns
        if prior_ns is None:
            raise RuntimeError("Round 74 time-scale state did not initialize")
        elapsed_seconds = (selected_ns - prior_ns) / 1_000_000_000.0
        return_bps = 0.0 if gap_reset else values[0]
        output: list[float] = []
        for index, half_life in enumerate(ROUND74_EVENT_STATE_HALF_LIVES_SECONDS):
            beta = math.log(2.0) / float(half_life)
            decay = math.exp(-beta * elapsed_seconds)
            interval_weight = -math.expm1(-beta * elapsed_seconds) / beta
            exposure = decay * self._exposure_seconds[index] + interval_weight
            self._exposure_seconds[index] = exposure
            self._return_bps[index] = decay * self._return_bps[index] + return_bps
            self._return_squared_bps[index] = (
                decay * self._return_squared_bps[index] + return_bps * return_bps
            )
            self._trade_pressure[index] = (
                decay * self._trade_pressure[index] + values[1]
            )
            self._depth_pressure[index] = (
                decay * self._depth_pressure[index] + values[2]
            )
            self._liquidation_pressure[index] = (
                decay * self._liquidation_pressure[index] + values[3]
            )
            self._spread_integral[index] = (
                decay * self._spread_integral[index]
                + self._prior_spread_bps * interval_weight
            )
            self._l1_imbalance_integral[index] = (
                decay * self._l1_imbalance_integral[index]
                + self._prior_l1_imbalance * interval_weight
            )
            if exposure <= 0.0:
                output.extend((0.0,) * 7)
                continue
            projected_return = self._return_bps[index] / exposure * float(half_life)
            projected_variance = (
                self._return_squared_bps[index] / exposure * float(half_life)
            )
            output.extend(
                (
                    projected_return,
                    math.sqrt(max(0.0, projected_variance)),
                    _signed_log1p(self._trade_pressure[index] / exposure),
                    _signed_log1p(self._depth_pressure[index] / exposure),
                    _signed_log1p(self._liquidation_pressure[index] / exposure),
                    self._spread_integral[index] / exposure,
                    self._l1_imbalance_integral[index] / exposure,
                )
            )
        self._prior_ns = selected_ns
        self._prior_spread_bps = values[4]
        self._prior_l1_imbalance = values[5]
        if not all(math.isfinite(value) for value in output):
            raise ValueError("Round 74 time-scale state is nonfinite")
        return tuple(output)


@dataclass(frozen=True)
class Round74EventToken:
    """One financially normalized feature token tied to an exact receipt."""

    symbol: str
    event_type: str
    frame_index: int
    message_index: int
    received_monotonic_ns: int
    received_wall_ns: int
    exchange_event_time_ms: int
    source_sequence_number: int
    feature_values: tuple[float, ...]

    def validate(self) -> None:
        if self.symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
            raise ValueError("Round 74 event token symbol is unsupported")
        if self.event_type not in ROUND74_EVENT_TYPES:
            raise ValueError("Round 74 event token type is unsupported")
        if (
            min(
                self.frame_index,
                self.message_index,
                self.received_monotonic_ns,
                self.received_wall_ns,
                self.exchange_event_time_ms,
                self.source_sequence_number,
            )
            < 0
        ):
            raise ValueError("Round 74 event token metadata is negative")
        if len(self.feature_values) != len(ROUND74_EVENT_FEATURE_NAMES):
            raise ValueError("Round 74 event token feature count differs")
        if not all(math.isfinite(value) for value in self.feature_values):
            raise ValueError("Round 74 event token contains a nonfinite feature")
        one_hot = self.feature_values[:5]
        if sum(value == 1.0 for value in one_hot) != 1 or any(
            value not in {0.0, 1.0} for value in one_hot
        ):
            raise ValueError("Round 74 event token event encoding is invalid")
        symbol_one_hot = self.feature_values[5:8]
        expected_symbol_index = ROUND74_EVENT_SYMBOLS.index(self.symbol)
        if any(value not in {0.0, 1.0} for value in symbol_one_hot) or tuple(
            index for index, value in enumerate(symbol_one_hot) if value == 1.0
        ) != (expected_symbol_index,):
            raise ValueError("Round 74 event token symbol encoding is invalid")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
            "feature_names_sha256": ROUND74_EVENT_FEATURE_NAMES_SHA256,
            "symbol": self.symbol,
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            "message_index": self.message_index,
            "received_monotonic_ns": self.received_monotonic_ns,
            "received_wall_ns": self.received_wall_ns,
            "exchange_event_time_ms": self.exchange_event_time_ms,
            "source_sequence_number": self.source_sequence_number,
            "feature_values": list(self.feature_values),
            "target_constructed": False,
            "model_evaluated": False,
        }


@dataclass(frozen=True)
class Round74ReplayObservation:
    """One exact replay result with optional causal token and depth state."""

    symbol: str
    event_type: str
    frame_index: int
    message_index: int
    received_monotonic_ns: int
    received_wall_ns: int
    token: Round74EventToken | None
    depth_state: L2BookState | None
    depth_update_is_stale: bool

    def validate(self) -> None:
        if self.symbol not in ROUND74_EVENT_SYMBOLS:
            raise ValueError("Round 74 observation symbol differs")
        if self.event_type not in ROUND74_EVENT_TYPES:
            raise ValueError("Round 74 observation event type differs")
        if (
            min(
                self.frame_index,
                self.message_index,
                self.received_monotonic_ns,
                self.received_wall_ns,
            )
            < 0
        ):
            raise ValueError("Round 74 observation metadata is negative")
        is_depth = self.event_type == "depthUpdate"
        if is_depth != (self.depth_state is not None):
            raise ValueError("Round 74 observation depth state differs")
        if not is_depth and self.depth_update_is_stale:
            raise ValueError("Round 74 non-depth observation is marked stale")
        if self.depth_state is not None and self.depth_state.symbol != self.symbol:
            raise ValueError("Round 74 observation depth symbol differs")
        if self.token is not None:
            self.token.validate()
            if (
                self.token.symbol != self.symbol
                or self.token.event_type != self.event_type
                or self.token.frame_index != self.frame_index
                or self.token.message_index != self.message_index
                or self.token.received_monotonic_ns != self.received_monotonic_ns
                or self.token.received_wall_ns != self.received_wall_ns
            ):
                raise ValueError("Round 74 observation token identity differs")


class Round74EventSequenceEncoder:
    """Replay one symbol's exact receipts into causal event-time tokens."""

    def __init__(
        self,
        *,
        symbol: str,
        tick_size: object,
        depth_snapshot: Mapping[str, object],
        feature_ready_wall_ns: int,
    ) -> None:
        normalized = normalize_symbol(symbol, default="")
        if normalized not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
            raise ValueError("Round 74 event encoder symbol is unsupported")
        if int(feature_ready_wall_ns) <= 0:
            raise ValueError("Round 74 feature-ready time must be positive")
        self.symbol = normalized
        self.feature_ready_wall_ns = int(feature_ready_wall_ns)
        self.book = SynchronizedDepthBook(normalized, tick_size)
        self.book.initialize(depth_snapshot)
        initial = self.book.state()
        self._l2_state = initial
        self._bid = float(initial.best_bid)
        self._ask = float(initial.best_ask)
        self._bid_qty = float(initial.bid_levels[0][1])
        self._ask_qty = float(initial.ask_levels[0][1])
        self._mark: MarkPriceEvent | None = None
        self._last_depth_update_is_stale = False
        self._prior_record_ns: int | None = None
        self._prior_mid: float | None = None
        self._time_scale_state = _Round74TimeScaleState()
        self._last_event_ns: dict[str, int | None] = {
            event_type: None for event_type in ROUND74_EVENT_TYPES
        }

    def _market_state(self) -> tuple[float, float, float, float]:
        if self._bid <= 0.0 or self._ask <= self._bid:
            raise ValueError("Round 74 event encoder BBO is invalid")
        if self._bid_qty <= 0.0 or self._ask_qty <= 0.0:
            raise ValueError("Round 74 event encoder BBO quantity is invalid")
        mid = (self._bid + self._ask) / 2.0
        spread_bps = _bps(self._ask - self._bid, mid)
        quantity_total = self._bid_qty + self._ask_qty
        l1_imbalance = (self._bid_qty - self._ask_qty) / quantity_total
        microprice = (
            self._ask * self._bid_qty + self._bid * self._ask_qty
        ) / quantity_total
        microprice_offset_bps = _bps(microprice - mid, mid)
        return mid, spread_bps, l1_imbalance, microprice_offset_bps

    def consume(
        self,
        *,
        frame_index: int,
        message_index: int,
        record: ImpactCaptureFrameRecord,
    ) -> Round74EventToken | None:
        event_type, symbol, payload = _decode_websocket_record(record)
        return self._consume_decoded(
            frame_index=frame_index,
            message_index=message_index,
            record=record,
            event_type=event_type,
            symbol=symbol,
            payload=payload,
        )

    def _consume_decoded(
        self,
        *,
        frame_index: int,
        message_index: int,
        record: ImpactCaptureFrameRecord,
        event_type: str,
        symbol: str,
        payload: Mapping[str, object],
    ) -> Round74EventToken | None:
        if symbol != self.symbol:
            raise ValueError("Round 74 event encoder received a different symbol")
        received_ns = int(record.received_monotonic_ns)
        if self._prior_record_ns is not None and received_ns < self._prior_record_ns:
            raise ValueError("Round 74 event receipts are not monotone")
        interarrival_us = (
            0.0
            if self._prior_record_ns is None
            else max(0.0, (received_ns - self._prior_record_ns) / 1_000.0)
        )
        self._prior_record_ns = received_ns

        event_flags = tuple(
            1.0 if event_type == candidate else 0.0 for candidate in ROUND74_EVENT_TYPES
        )
        symbol_flags = tuple(
            1.0 if self.symbol == candidate else 0.0
            for candidate in ROUND74_EVENT_SYMBOLS
        )
        stale_depth = 0.0
        depth_signed = {band: 0.0 for band in ROUND74_EVENT_BANDS}
        depth_absolute = {band: 0.0 for band in ROUND74_EVENT_BANDS}
        trade_signed = 0.0
        trade_absolute = 0.0
        trade_pressure = 0.0
        trade_price_to_mid_bps = 0.0
        liquidation_signed = 0.0
        liquidation_pressure = 0.0
        bid_qty_change = 0.0
        ask_qty_change = 0.0
        pre_depth = _depth_total(self._l2_state)
        exchange_event_time_ms: int

        if event_type == "depthUpdate":
            pre_state = self._l2_state
            depth_event = self.book.apply(payload, receive_time_ns=received_ns)
            exchange_event_time_ms = int(depth_event.event_time_ms)
            stale_depth = 1.0 if depth_event.stale else 0.0
            self._last_depth_update_is_stale = depth_event.stale
            for change in depth_event.changes:
                band = pre_event_level_band(pre_state, change)
                signed = (
                    change.added_quote if change.side == "bid" else -change.added_quote
                )
                signed += (
                    -change.removed_quote
                    if change.side == "bid"
                    else change.removed_quote
                )
                depth_signed[band] += signed
                depth_absolute[band] += change.added_quote + change.removed_quote
            self._l2_state = self.book.state()
            self._bid = float(self._l2_state.best_bid)
            self._ask = float(self._l2_state.best_ask)
            self._bid_qty = float(self._l2_state.bid_levels[0][1])
            self._ask_qty = float(self._l2_state.ask_levels[0][1])
        elif event_type == "bookTicker":
            ticker = parse_book_ticker(
                payload,
                symbol=self.symbol,
                receive_time_ns=received_ns,
            )
            exchange_event_time_ms = int(ticker.event_time_ms)
            bid_qty_change = _signed_log1p(
                _safe_ratio(ticker.bid_qty - self._bid_qty, self._bid_qty)
            )
            ask_qty_change = _signed_log1p(
                _safe_ratio(ticker.ask_qty - self._ask_qty, self._ask_qty)
            )
            self._bid = float(ticker.bid)
            self._ask = float(ticker.ask)
            self._bid_qty = float(ticker.bid_qty)
            self._ask_qty = float(ticker.ask_qty)
        elif event_type == "aggTrade":
            trade = parse_aggregate_trade(
                payload,
                symbol=self.symbol,
                receive_time_ns=received_ns,
            )
            exchange_event_time_ms = int(trade.event_time_ms)
            sign = 1.0 if trade.aggressive_side == "buy" else -1.0
            trade_absolute = _signed_log1p(_safe_ratio(trade.quote_notional, pre_depth))
            trade_signed = sign * trade_absolute
            trade_pressure = sign * _safe_ratio(
                trade.quote_notional,
                pre_depth,
            )
            current_mid = (self._bid + self._ask) / 2.0
            trade_price_to_mid_bps = _bps(trade.price - current_mid, current_mid)
        elif event_type == "markPriceUpdate":
            self._mark = parse_mark_price(
                payload,
                symbol=self.symbol,
                receive_time_ns=received_ns,
            )
            exchange_event_time_ms = int(self._mark.event_time_ms)
        else:
            liquidation = parse_liquidation_snapshot(
                payload,
                symbol=self.symbol,
                receive_time_ns=received_ns,
            )
            exchange_event_time_ms = int(liquidation.event_time_ms)
            sign = 1.0 if liquidation.side == "BUY" else -1.0
            liquidation_pressure = sign * _safe_ratio(
                liquidation.observed_filled_quote,
                pre_depth,
            )
            liquidation_signed = _signed_log1p(liquidation_pressure)

        mid, spread_bps, l1_imbalance, microprice_offset_bps = self._market_state()
        mid_log_return_bps = (
            0.0
            if self._prior_mid is None
            else math.log(mid / self._prior_mid) * 10_000.0
        )
        self._prior_mid = mid
        state = self._l2_state
        mark_to_mid_bps = 0.0
        index_to_mid_bps = 0.0
        funding_rate_bps = 0.0
        if self._mark is not None:
            mark_to_mid_bps = _bps(self._mark.mark_price - mid, mid)
            index_to_mid_bps = _bps(self._mark.index_price - mid, mid)
            funding_rate_bps = float(self._mark.funding_rate) * 10_000.0
        depth_pressure = _safe_ratio(sum(depth_signed.values()), pre_depth)
        time_scale = self._time_scale_state.update(
            now_ns=received_ns,
            mid_log_return_bps=mid_log_return_bps,
            trade_pressure=trade_pressure,
            depth_pressure=depth_pressure,
            liquidation_pressure=liquidation_pressure,
            spread_bps=spread_bps,
            l1_imbalance=l1_imbalance,
        )
        temporal = tuple(
            _time_since_feature(received_ns, self._last_event_ns[candidate])
            for candidate in ROUND74_EVENT_TYPES
        )
        self._last_event_ns[event_type] = received_ns
        second_of_day = (int(record.received_wall_ns) // 1_000_000_000) % 86_400
        angle = 2.0 * math.pi * second_of_day / 86_400.0
        values = (
            *event_flags,
            *symbol_flags,
            stale_depth,
            math.log1p(interarrival_us),
            spread_bps,
            l1_imbalance,
            microprice_offset_bps,
            state.imbalance_5,
            state.imbalance_10,
            state.imbalance_20,
            _safe_ratio(state.bid_depth_quote_5, state.bid_depth_quote_20),
            _safe_ratio(state.ask_depth_quote_5, state.ask_depth_quote_20),
            math.log1p(state.bid_depth_quote_20),
            math.log1p(state.ask_depth_quote_20),
            mid_log_return_bps,
            *(
                _signed_log1p(_safe_ratio(depth_signed[band], pre_depth))
                for band in ROUND74_EVENT_BANDS
            ),
            *(
                math.log1p(_safe_ratio(depth_absolute[band], pre_depth))
                for band in ROUND74_EVENT_BANDS
            ),
            trade_signed,
            trade_absolute,
            trade_price_to_mid_bps,
            mark_to_mid_bps,
            index_to_mid_bps,
            funding_rate_bps,
            liquidation_signed,
            bid_qty_change,
            ask_qty_change,
            *time_scale,
            *temporal,
            math.sin(angle),
            math.cos(angle),
        )
        token = Round74EventToken(
            symbol=self.symbol,
            event_type=event_type,
            frame_index=int(frame_index),
            message_index=int(message_index),
            received_monotonic_ns=received_ns,
            received_wall_ns=int(record.received_wall_ns),
            exchange_event_time_ms=exchange_event_time_ms,
            source_sequence_number=int(record.sequence_number),
            feature_values=tuple(float(value) for value in values),
        )
        token.validate()
        if token.received_wall_ns < self.feature_ready_wall_ns:
            return None
        return token

    @property
    def latest_l2_state(self) -> L2BookState:
        return self._l2_state

    @property
    def last_depth_update_is_stale(self) -> bool:
        return self._last_depth_update_is_stale


class Round74MultiSymbolEventReplay:
    """Route one globally ordered exact stream into per-symbol encoders."""

    def __init__(
        self,
        *,
        tick_sizes: Mapping[str, object],
        depth_snapshots: Mapping[str, Mapping[str, object]],
        feature_ready_wall_ns: int,
    ) -> None:
        symbols = ROUND74_EVENT_SYMBOLS
        if tuple(sorted(tick_sizes)) != symbols:
            raise ValueError("Round 74 replay tick-size universe differs")
        if tuple(sorted(depth_snapshots)) != symbols:
            raise ValueError("Round 74 replay snapshot universe differs")
        self._encoders = {
            symbol: Round74EventSequenceEncoder(
                symbol=symbol,
                tick_size=tick_sizes[symbol],
                depth_snapshot=depth_snapshots[symbol],
                feature_ready_wall_ns=feature_ready_wall_ns,
            )
            for symbol in symbols
        }
        self._prior_received_monotonic_ns = -1

    def consume(
        self,
        *,
        frame_index: int,
        message_index: int,
        record: ImpactCaptureFrameRecord,
    ) -> Round74EventToken | None:
        observation = self.consume_observation(
            frame_index=frame_index,
            message_index=message_index,
            record=record,
        )
        return None if observation is None else observation.token

    def consume_observation(
        self,
        *,
        frame_index: int,
        message_index: int,
        record: ImpactCaptureFrameRecord,
    ) -> Round74ReplayObservation | None:
        received_ns = int(record.received_monotonic_ns)
        if received_ns < self._prior_received_monotonic_ns:
            raise ValueError("Round 74 exact replay global receipt order regressed")
        self._prior_received_monotonic_ns = received_ns
        if record.stream == "binance_futures_rest":
            return None
        event_type, symbol, payload = _decode_websocket_record(record)
        encoder = self._encoders.get(symbol)
        if encoder is None:
            raise ValueError("Round 74 exact replay symbol is unsupported")
        token = encoder._consume_decoded(
            frame_index=frame_index,
            message_index=message_index,
            record=record,
            event_type=event_type,
            symbol=symbol,
            payload=payload,
        )
        observation = Round74ReplayObservation(
            symbol=symbol,
            event_type=event_type,
            frame_index=int(frame_index),
            message_index=int(message_index),
            received_monotonic_ns=received_ns,
            received_wall_ns=int(record.received_wall_ns),
            token=token,
            depth_state=(
                encoder.latest_l2_state if event_type == "depthUpdate" else None
            ),
            depth_update_is_stale=(
                encoder.last_depth_update_is_stale
                if event_type == "depthUpdate"
                else False
            ),
        )
        observation.validate()
        return observation


@dataclass(frozen=True)
class Round74EventWindow:
    """A fixed-length causal sequence ending at one exact event receipt."""

    symbol: str
    endpoint_frame_index: int
    endpoint_message_index: int
    endpoint_received_monotonic_ns: int
    feature_values: tuple[tuple[float, ...], ...]

    def validate(self, sequence_length: int) -> None:
        if len(self.feature_values) != int(sequence_length):
            raise ValueError("Round 74 event window length differs")
        if any(
            len(row) != len(ROUND74_EVENT_FEATURE_NAMES)
            or not all(math.isfinite(value) for value in row)
            for row in self.feature_values
        ):
            raise ValueError("Round 74 event window features are invalid")


def iter_round74_event_windows(
    tokens: Sequence[Round74EventToken] | Iterator[Round74EventToken],
    *,
    sequence_length: int = 128,
    stride: int = 16,
    maximum_gap_ns: int = ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS,
) -> Iterator[Round74EventWindow]:
    """Yield per-symbol windows without crossing a long receipt gap."""

    length = int(sequence_length)
    selected_stride = int(stride)
    gap_limit = int(maximum_gap_ns)
    if length < 2:
        raise ValueError("Round 74 event window requires at least two tokens")
    if selected_stride < 1:
        raise ValueError("Round 74 event window stride must be positive")
    if gap_limit < 1:
        raise ValueError("Round 74 event window gap limit must be positive")
    buffers: dict[str, deque[Round74EventToken]] = {}
    prior_receipts: dict[str, int] = {}
    since_yield: dict[str, int] = {}
    global_prior = -1
    for token in tokens:
        token.validate()
        if token.received_monotonic_ns < global_prior:
            raise ValueError("Round 74 global event token order regressed")
        global_prior = token.received_monotonic_ns
        buffer = buffers.setdefault(token.symbol, deque(maxlen=length))
        prior = prior_receipts.get(token.symbol)
        if prior is not None and token.received_monotonic_ns - prior > gap_limit:
            buffer.clear()
            since_yield[token.symbol] = 0
        prior_receipts[token.symbol] = token.received_monotonic_ns
        buffer.append(token)
        if len(buffer) < length:
            continue
        counter = since_yield.get(token.symbol, 0)
        if counter:
            since_yield[token.symbol] = counter - 1
            continue
        window = Round74EventWindow(
            symbol=token.symbol,
            endpoint_frame_index=token.frame_index,
            endpoint_message_index=token.message_index,
            endpoint_received_monotonic_ns=token.received_monotonic_ns,
            feature_values=tuple(item.feature_values for item in buffer),
        )
        window.validate(length)
        yield window
        since_yield[token.symbol] = selected_stride - 1


def iter_round74_v10_event_observations(
    store: object,
    *,
    run_id: str,
) -> Iterator[Round74ReplayObservation]:
    """Audit and stream exact observations from one qualified v10 run.

    The stored capture report and a fresh exact-frame audit must both pass.
    This function still grants no target, model, deployment, or order authority.
    """

    from .impact_absorption_store import (
        IMPACT_CAPTURE_SYMBOLS,
        IMPACT_CAPTURE_V10_CONTRACT_SHA256,
        IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
        IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        ImpactAbsorptionStore,
        iter_impact_capture_v10_records,
        load_impact_capture_v10_preflight,
    )

    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 v10 replay requires an ImpactAbsorptionStore")
    if not store.read_only:
        raise ValueError("Round 74 v10 replay requires a read-only store")
    audit = store.audit_run(run_id)
    if not audit.passed:
        raise ValueError("Round 74 v10 replay requires a passing fresh frame audit")
    connection = store.connect()
    run = connection.execute(
        """
        SELECT status, schema_version, capture_contract_sha256
        FROM impact_capture_run WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run != (
        "completed",
        IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    ):
        raise ValueError("Round 74 v10 replay run identity or status differs")
    report_row = connection.execute(
        """
        SELECT schema_version, capture_contract_sha256, report_json, report_sha256
        FROM impact_capture_report WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if report_row is None:
        raise ValueError("Round 74 v10 replay capture report is missing")
    report_text = str(report_row[2])
    if (
        str(report_row[0]) != IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
        or str(report_row[1]) != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or hashlib.sha256(report_text.encode("ascii")).hexdigest() != str(report_row[3])
    ):
        raise ValueError("Round 74 v10 replay capture report identity differs")
    report = _strict_json_object(report_text)
    if not all(
        report.get(field) is True
        for field in (
            "capture_gate_passed",
            "data_qualification_passed",
            "resource_safety_passed",
            "qualification_passed",
        )
    ):
        raise ValueError("Round 74 v10 replay capture gates did not all pass")
    preflight = load_impact_capture_v10_preflight(
        connection,
        run_id=run_id,
    )
    segment_rows = connection.execute(
        """
        SELECT symbol, status, tick_size
        FROM impact_capture_segment WHERE run_id = ? ORDER BY symbol
        """,
        [run_id],
    ).fetchall()
    if tuple(str(row[0]) for row in segment_rows) != IMPACT_CAPTURE_SYMBOLS or any(
        str(row[1]) != "valid" for row in segment_rows
    ):
        raise ValueError("Round 74 v10 replay symbol segments are not valid")
    tick_sizes = {str(row[0]): float(row[2]) for row in segment_rows}
    snapshots = {
        symbol: _strict_json_object(record.raw_text)
        for symbol, record in preflight.snapshot_records
    }
    replay = Round74MultiSymbolEventReplay(
        tick_sizes=tick_sizes,
        depth_snapshots=snapshots,
        feature_ready_wall_ns=preflight.ready_wall_ns,
    )
    for frame_index, message_index, record in iter_impact_capture_v10_records(
        connection,
        run_id=run_id,
    ):
        observation = replay.consume_observation(
            frame_index=frame_index,
            message_index=message_index,
            record=record,
        )
        if observation is not None:
            yield observation


def iter_round74_v10_event_tokens(
    store: object,
    *,
    run_id: str,
) -> Iterator[Round74EventToken]:
    """Stream feature tokens while retaining the exact observation audit."""

    for observation in iter_round74_v10_event_observations(
        store,
        run_id=run_id,
    ):
        if observation.token is not None:
            yield observation.token


__all__ = [
    "ROUND74_EVENT_BANDS",
    "ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS",
    "ROUND74_EVENT_FEATURE_NAMES",
    "ROUND74_EVENT_FEATURE_NAMES_SHA256",
    "ROUND74_EVENT_MAX_TIME_SINCE_MS",
    "ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS",
    "ROUND74_EVENT_PAYOFF_QUANTILES",
    "ROUND74_EVENT_PAYOFF_SIDES",
    "ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION",
    "ROUND74_EVENT_SEQUENCE_LENGTH",
    "ROUND74_EVENT_STATE_HALF_LIVES_SECONDS",
    "ROUND74_EVENT_SYMBOLS",
    "ROUND74_EVENT_TYPES",
    "Round74MultiSymbolEventReplay",
    "Round74ReplayObservation",
    "Round74EventSequenceEncoder",
    "Round74EventToken",
    "Round74EventWindow",
    "iter_round74_event_windows",
    "iter_round74_v10_event_observations",
    "iter_round74_v10_event_tokens",
]
