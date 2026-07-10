"""Aggregate-trade microstructure feature engineering.

The candle pipeline keeps compatibility with existing workflows, but profitable
day-trading research needs signals that were visible before the candle closed:
taker-side imbalance, trade bursts, large-print concentration, and whether
price movement confirms or rejects the flow.  This module builds those features
from raw Binance aggregate trades without using any future rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .api import Candle
from .market_store import AggTrade, AggTradeBucket


TRADE_TAPE_FEATURES_PER_WINDOW = 16


@dataclass
class _TapeBucket:
    open_time: int
    first_time_ms: int
    last_time_ms: int
    first_price: float
    last_price: float
    high_price: float
    low_price: float
    total_quantity: float = 0.0
    total_notional: float = 0.0
    buy_quantity: float = 0.0
    buy_notional: float = 0.0
    sell_quantity: float = 0.0
    sell_notional: float = 0.0
    aggregate_count: int = 0
    buyer_taker_count: int = 0
    seller_taker_count: int = 0
    max_trade_notional: float = 0.0

    def add(self, trade: AggTrade) -> None:
        price = float(trade.price)
        quantity = float(trade.quantity)
        notional = price * quantity
        timestamp = int(trade.trade_time_ms)
        if timestamp < self.first_time_ms:
            self.first_time_ms = timestamp
            self.first_price = price
        if timestamp >= self.last_time_ms:
            self.last_time_ms = timestamp
            self.last_price = price
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.total_quantity += quantity
        self.total_notional += notional
        self.aggregate_count += 1
        self.max_trade_notional = max(self.max_trade_notional, notional)
        if trade.is_buyer_maker:
            self.sell_quantity += quantity
            self.sell_notional += notional
            self.seller_taker_count += 1
        else:
            self.buy_quantity += quantity
            self.buy_notional += notional
            self.buyer_taker_count += 1


@dataclass(frozen=True)
class TradeTapeFeatureCache:
    close_times: tuple[int, ...]
    total_notional: tuple[float, ...]
    buy_notional: tuple[float, ...]
    sell_notional: tuple[float, ...]
    aggregate_count: tuple[float, ...]
    buyer_taker_count: tuple[float, ...]
    seller_taker_count: tuple[float, ...]
    large_notional: tuple[float, ...]
    no_tape: tuple[float, ...]
    micro_range: tuple[float, ...]
    micro_drift: tuple[float, ...]
    vwap_gap: tuple[float, ...]
    signed_notional_ratio: tuple[float, ...]
    returns: tuple[float, ...]
    notional_prefix: tuple[float, ...]
    buy_notional_prefix: tuple[float, ...]
    sell_notional_prefix: tuple[float, ...]
    aggregate_count_prefix: tuple[float, ...]
    buyer_taker_count_prefix: tuple[float, ...]
    seller_taker_count_prefix: tuple[float, ...]
    large_notional_prefix: tuple[float, ...]
    no_tape_prefix: tuple[float, ...]
    micro_range_prefix: tuple[float, ...]
    micro_drift_prefix: tuple[float, ...]
    signed_ratio_prefix: tuple[float, ...]
    signed_ratio_square_prefix: tuple[float, ...]
    return_prefix: tuple[float, ...]
    return_square_prefix: tuple[float, ...]
    signed_return_product_prefix: tuple[float, ...]


def _safe(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not denominator or not math.isfinite(denominator):
        return default
    return _safe(numerator / denominator)


def _prefix(values: Sequence[float]) -> tuple[float, ...]:
    total = 0.0
    out = [0.0]
    for value in values:
        total += _safe(float(value))
        out.append(total)
    return tuple(out)


def _window_sum(prefix: Sequence[float], start: int, end: int) -> float:
    if not prefix or end < start:
        return 0.0
    start = max(0, int(start))
    end = min(len(prefix) - 2, int(end))
    if end < start:
        return 0.0
    return float(prefix[end + 1] - prefix[start])


def _correlation_from_prefixes(
    count: int,
    x_prefix: Sequence[float],
    y_prefix: Sequence[float],
    x_square_prefix: Sequence[float],
    y_square_prefix: Sequence[float],
    xy_prefix: Sequence[float],
    start: int,
    end: int,
) -> float:
    if count < 3:
        return 0.0
    sum_x = _window_sum(x_prefix, start, end)
    sum_y = _window_sum(y_prefix, start, end)
    sum_x2 = _window_sum(x_square_prefix, start, end)
    sum_y2 = _window_sum(y_square_prefix, start, end)
    sum_xy = _window_sum(xy_prefix, start, end)
    numerator = (count * sum_xy) - (sum_x * sum_y)
    var_x = (count * sum_x2) - (sum_x * sum_x)
    var_y = (count * sum_y2) - (sum_y * sum_y)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / math.sqrt(var_x * var_y)))


def _bucket_open_time(timestamp_ms: int, bucket_ms: int) -> int:
    return (int(timestamp_ms) // int(bucket_ms)) * int(bucket_ms)


def _build_buckets(trades: Sequence[AggTrade], bucket_ms: int) -> dict[int, _TapeBucket]:
    buckets: dict[int, _TapeBucket] = {}
    for trade in sorted(trades, key=lambda item: (int(item.trade_time_ms), int(item.agg_trade_id))):
        price = float(trade.price)
        quantity = float(trade.quantity)
        timestamp = int(trade.trade_time_ms)
        if price <= 0.0 or quantity <= 0.0 or timestamp <= 0:
            continue
        open_time = _bucket_open_time(timestamp, bucket_ms)
        bucket = buckets.get(open_time)
        if bucket is None:
            bucket = _TapeBucket(
                open_time=open_time,
                first_time_ms=timestamp,
                last_time_ms=timestamp,
                first_price=price,
                last_price=price,
                high_price=price,
                low_price=price,
            )
            buckets[open_time] = bucket
        bucket.add(trade)
    return buckets


def _bucket_map_from_sql_buckets(buckets: Sequence[AggTradeBucket]) -> dict[int, _TapeBucket]:
    out: dict[int, _TapeBucket] = {}
    for bucket in buckets:
        if bucket.total_notional <= 0.0 or bucket.total_quantity <= 0.0:
            continue
        out[int(bucket.open_time)] = _TapeBucket(
            open_time=int(bucket.open_time),
            first_time_ms=int(bucket.first_time_ms),
            last_time_ms=int(bucket.last_time_ms),
            first_price=float(bucket.first_price),
            last_price=float(bucket.last_price),
            high_price=float(bucket.high_price),
            low_price=float(bucket.low_price),
            total_quantity=float(bucket.total_quantity),
            total_notional=float(bucket.total_notional),
            buy_quantity=float(bucket.buy_quantity),
            buy_notional=float(bucket.buy_notional),
            sell_quantity=float(bucket.sell_quantity),
            sell_notional=float(bucket.sell_notional),
            aggregate_count=int(bucket.aggregate_count),
            buyer_taker_count=int(bucket.buyer_taker_count),
            seller_taker_count=int(bucket.seller_taker_count),
            max_trade_notional=float(bucket.max_trade_notional),
        )
    return out


def build_trade_tape_feature_cache(
    candles: Sequence[Candle],
    trades: Sequence[AggTrade] | None = None,
    *,
    buckets: Sequence[AggTradeBucket] | None = None,
    bucket_ms: int = 1000,
) -> TradeTapeFeatureCache:
    bucket_width = max(1, int(bucket_ms))
    candle_list = list(candles)
    bucket_map = (
        _bucket_map_from_sql_buckets(list(buckets))
        if buckets is not None
        else _build_buckets(list(trades or ()), bucket_width)
    )
    total_notional: list[float] = []
    buy_notional: list[float] = []
    sell_notional: list[float] = []
    aggregate_count: list[float] = []
    buyer_taker_count: list[float] = []
    seller_taker_count: list[float] = []
    large_notional: list[float] = []
    no_tape: list[float] = []
    micro_range: list[float] = []
    micro_drift: list[float] = []
    vwap_gap: list[float] = []
    signed_notional_ratio: list[float] = []
    returns: list[float] = []

    previous_close: float | None = None
    for candle in candle_list:
        open_time = _bucket_open_time(int(candle.open_time), bucket_width)
        bucket = bucket_map.get(open_time)
        close = float(candle.close)
        if previous_close is None or previous_close <= 0.0 or close <= 0.0:
            returns.append(0.0)
        else:
            returns.append(_safe((close - previous_close) / previous_close))
        previous_close = close
        if bucket is None or bucket.total_notional <= 0.0:
            total_notional.append(0.0)
            buy_notional.append(0.0)
            sell_notional.append(0.0)
            aggregate_count.append(0.0)
            buyer_taker_count.append(0.0)
            seller_taker_count.append(0.0)
            large_notional.append(0.0)
            no_tape.append(1.0)
            micro_range.append(0.0)
            micro_drift.append(0.0)
            vwap_gap.append(0.0)
            signed_notional_ratio.append(0.0)
            continue
        total = bucket.total_notional
        vwap = _safe_ratio(bucket.total_notional, bucket.total_quantity, default=close)
        total_notional.append(total)
        buy_notional.append(bucket.buy_notional)
        sell_notional.append(bucket.sell_notional)
        aggregate_count.append(float(bucket.aggregate_count))
        buyer_taker_count.append(float(bucket.buyer_taker_count))
        seller_taker_count.append(float(bucket.seller_taker_count))
        large_notional.append(bucket.max_trade_notional)
        no_tape.append(0.0)
        micro_range.append(_safe_ratio(bucket.high_price - bucket.low_price, bucket.last_price))
        micro_drift.append(_safe_ratio(bucket.last_price - bucket.first_price, bucket.first_price))
        vwap_gap.append(_safe_ratio(close - vwap, vwap))
        signed_notional_ratio.append(_safe_ratio(bucket.buy_notional - bucket.sell_notional, total))

    signed_square = [value * value for value in signed_notional_ratio]
    return_square = [value * value for value in returns]
    signed_return = [left * right for left, right in zip(signed_notional_ratio, returns, strict=True)]
    return TradeTapeFeatureCache(
        close_times=tuple(int(candle.close_time) for candle in candle_list),
        total_notional=tuple(total_notional),
        buy_notional=tuple(buy_notional),
        sell_notional=tuple(sell_notional),
        aggregate_count=tuple(aggregate_count),
        buyer_taker_count=tuple(buyer_taker_count),
        seller_taker_count=tuple(seller_taker_count),
        large_notional=tuple(large_notional),
        no_tape=tuple(no_tape),
        micro_range=tuple(micro_range),
        micro_drift=tuple(micro_drift),
        vwap_gap=tuple(vwap_gap),
        signed_notional_ratio=tuple(signed_notional_ratio),
        returns=tuple(returns),
        notional_prefix=_prefix(total_notional),
        buy_notional_prefix=_prefix(buy_notional),
        sell_notional_prefix=_prefix(sell_notional),
        aggregate_count_prefix=_prefix(aggregate_count),
        buyer_taker_count_prefix=_prefix(buyer_taker_count),
        seller_taker_count_prefix=_prefix(seller_taker_count),
        large_notional_prefix=_prefix(large_notional),
        no_tape_prefix=_prefix(no_tape),
        micro_range_prefix=_prefix(micro_range),
        micro_drift_prefix=_prefix(micro_drift),
        signed_ratio_prefix=_prefix(signed_notional_ratio),
        signed_ratio_square_prefix=_prefix(signed_square),
        return_prefix=_prefix(returns),
        return_square_prefix=_prefix(return_square),
        signed_return_product_prefix=_prefix(signed_return),
    )


def trade_tape_features_at(
    cache: TradeTapeFeatureCache | None,
    end: int,
    windows: Sequence[int],
    *,
    features_per_window: int = TRADE_TAPE_FEATURES_PER_WINDOW,
) -> list[float]:
    width = max(0, int(features_per_window))
    if width == 0 or not windows:
        return []
    if cache is None or end < 0 or end >= len(cache.close_times):
        return [0.0] * width * len(windows)
    features: list[float] = []
    current_notional = float(cache.total_notional[end])
    current_aggregate_count = float(cache.aggregate_count[end])
    current_vwap_gap = float(cache.vwap_gap[end])
    for window in windows:
        lookback = max(2, int(window))
        if end < lookback - 1:
            features.extend([0.0] * width)
            continue
        start = end + 1 - lookback
        total = _window_sum(cache.notional_prefix, start, end)
        buy = _window_sum(cache.buy_notional_prefix, start, end)
        sell = _window_sum(cache.sell_notional_prefix, start, end)
        aggregate_count = _window_sum(cache.aggregate_count_prefix, start, end)
        buyer_count = _window_sum(cache.buyer_taker_count_prefix, start, end)
        seller_count = _window_sum(cache.seller_taker_count_prefix, start, end)
        large = _window_sum(cache.large_notional_prefix, start, end)
        no_tape_ratio = _safe_ratio(_window_sum(cache.no_tape_prefix, start, end), float(lookback))
        mean_notional = _safe_ratio(total, float(lookback))
        mean_aggregate_count = _safe_ratio(aggregate_count, float(lookback))
        midpoint = start + max(1, lookback // 2)
        first_count = max(1, midpoint - start)
        second_count = max(1, end - midpoint + 1)
        first_signed = _safe_ratio(_window_sum(cache.signed_ratio_prefix, start, midpoint - 1), float(first_count))
        second_signed = _safe_ratio(_window_sum(cache.signed_ratio_prefix, midpoint, end), float(second_count))
        flow_return_alignment = _correlation_from_prefixes(
            lookback,
            cache.signed_ratio_prefix,
            cache.return_prefix,
            cache.signed_ratio_square_prefix,
            cache.return_square_prefix,
            cache.signed_return_product_prefix,
            start,
            end,
        )
        signed_sum = _window_sum(cache.signed_ratio_prefix, start, end)
        signed_square_sum = _window_sum(cache.signed_ratio_square_prefix, start, end)
        signed_mean = _safe_ratio(signed_sum, float(lookback))
        signed_variance = max(0.0, _safe_ratio(signed_square_sum, float(lookback)) - signed_mean * signed_mean)
        signed_std = math.sqrt(signed_variance)
        current_signed = float(cache.signed_notional_ratio[end])
        first_notional = _window_sum(cache.notional_prefix, start, midpoint - 1)
        second_notional = _window_sum(cache.notional_prefix, midpoint, end)
        first_notional_rate = _safe_ratio(first_notional, float(first_count))
        second_notional_rate = _safe_ratio(second_notional, float(second_count))
        large_pressure = _safe_ratio(large, total) * math.tanh(signed_mean * 3.0)
        base = [
            _safe_ratio(buy, total, default=0.5),
            _safe_ratio(buy - sell, total),
            _safe_ratio(buyer_count - seller_count, aggregate_count),
            math.tanh(_safe_ratio(current_notional - mean_notional, mean_notional)),
            math.tanh(_safe_ratio(current_aggregate_count - mean_aggregate_count, mean_aggregate_count)),
            _safe_ratio(large, total),
            math.tanh(current_vwap_gap * 5000.0),
            math.tanh(_safe_ratio(_window_sum(cache.micro_range_prefix, start, end), float(lookback)) * 5000.0),
            math.tanh(_safe_ratio(_window_sum(cache.micro_drift_prefix, start, end), float(lookback)) * 5000.0),
            no_tape_ratio,
            _safe(second_signed - first_signed),
            _safe(flow_return_alignment),
            _safe(current_signed - signed_mean),
            math.tanh(_safe_ratio(current_signed - signed_mean, signed_std) / 3.0),
            math.tanh(_safe_ratio(second_notional_rate - first_notional_rate, mean_notional)),
            _safe(large_pressure),
        ]
        features.extend(_safe(value) for value in base[:width])
        if width > TRADE_TAPE_FEATURES_PER_WINDOW:
            features.extend([0.0] * (width - TRADE_TAPE_FEATURES_PER_WINDOW))
    return features


def trade_tape_has_data(cache: TradeTapeFeatureCache | None) -> bool:
    return bool(cache and any(value > 0.0 for value in cache.total_notional))


__all__ = [
    "TRADE_TAPE_FEATURES_PER_WINDOW",
    "TradeTapeFeatureCache",
    "build_trade_tape_feature_cache",
    "trade_tape_features_at",
    "trade_tape_has_data",
]
