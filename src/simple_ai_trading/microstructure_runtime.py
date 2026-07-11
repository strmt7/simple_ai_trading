"""Streaming parity engine for the causal L1/tape feature contract."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .assets import normalize_symbol
from .microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_TRADE_EMBARGO_MS,
    MICROSTRUCTURE_FEATURE_VERSION,
)


MICROSTRUCTURE_STREAM_WARMUP_SECONDS = 3_600
_HISTORY_SECONDS = MICROSTRUCTURE_STREAM_WARMUP_SECONDS
_REQUIRED_SECONDS = _HISTORY_SECONDS + 1


@dataclass(frozen=True)
class MicrostructureSecond:
    """One closed UTC second of top-of-book and aggressive trade evidence."""

    symbol: str
    second_ms: int
    open_mid: float
    high_mid: float
    low_mid: float
    close_mid: float
    close_bid: float
    close_ask: float
    close_bid_qty: float
    close_ask_qty: float
    spread_bps: float
    max_spread_bps: float
    l1_imbalance: float
    close_l1_imbalance: float
    microprice_offset_bps: float
    quote_updates: int
    event_delay_p50_ms: float
    event_delay_p99_ms: float
    trade_close: float
    base_volume: float
    quote_volume: float
    aggressive_buy_volume: float
    aggressive_sell_volume: float
    trade_imbalance: float
    trade_count: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "MicrostructureSecond":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})  # type: ignore[arg-type]


@dataclass(frozen=True)
class StreamingFeatureRow:
    symbol: str
    feature_version: str
    feature_names: tuple[str, ...]
    source_second_ms: int
    decision_time_ms: int
    close_bid: float
    close_ask: float
    close_bid_qty: float
    close_ask_qty: float
    features: np.ndarray

    def as_mapping(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.feature_names, self.features, strict=True)
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-12)


def _tail_mean(values: np.ndarray, size: int) -> float:
    return float(np.mean(values[-size:]))


def _tail_sum(values: np.ndarray, size: int) -> float:
    return float(np.sum(values[-size:]))


def _validate_second(row: MicrostructureSecond, expected_symbol: str) -> None:
    if normalize_symbol(row.symbol) != expected_symbol:
        raise ValueError(
            f"streaming feature symbol mismatch: {row.symbol} != {expected_symbol}"
        )
    if int(row.second_ms) < 0 or int(row.second_ms) % 1_000 != 0:
        raise ValueError("second_ms must be a non-negative UTC-second boundary")
    finite_values = (
        row.open_mid,
        row.high_mid,
        row.low_mid,
        row.close_mid,
        row.close_bid,
        row.close_ask,
        row.close_bid_qty,
        row.close_ask_qty,
        row.spread_bps,
        row.max_spread_bps,
        row.l1_imbalance,
        row.close_l1_imbalance,
        row.microprice_offset_bps,
        row.event_delay_p50_ms,
        row.event_delay_p99_ms,
        row.trade_close,
        row.base_volume,
        row.quote_volume,
        row.aggressive_buy_volume,
        row.aggressive_sell_volume,
        row.trade_imbalance,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("microstructure second contains a non-finite value")
    if (
        min(
            row.open_mid,
            row.high_mid,
            row.low_mid,
            row.close_mid,
            row.close_bid,
            row.close_ask,
        )
        <= 0.0
    ):
        raise ValueError("microstructure prices must be positive")
    if row.low_mid > min(row.open_mid, row.close_mid) or row.high_mid < max(
        row.open_mid, row.close_mid
    ):
        raise ValueError("microstructure OHLC bounds are inconsistent")
    if row.close_bid >= row.close_ask:
        raise ValueError("microstructure close quote is crossed")
    if min(row.close_bid_qty, row.close_ask_qty) <= 0.0:
        raise ValueError("microstructure close quantities must be positive")
    if (
        min(
            row.spread_bps,
            row.max_spread_bps,
            row.event_delay_p50_ms,
            row.event_delay_p99_ms,
            row.base_volume,
            row.quote_volume,
            row.aggressive_buy_volume,
            row.aggressive_sell_volume,
        )
        < 0.0
    ):
        raise ValueError(
            "microstructure costs, delays, and volumes must be non-negative"
        )
    if row.max_spread_bps < row.spread_bps:
        raise ValueError("max_spread_bps cannot be below event-weighted spread")
    if row.event_delay_p99_ms < row.event_delay_p50_ms:
        raise ValueError("event delay p99 cannot be below p50")
    if row.quote_updates <= 0 or row.trade_count < 0:
        raise ValueError("quote_updates must be positive and trade_count non-negative")
    if not all(
        -1.0 <= value <= 1.0
        for value in (row.l1_imbalance, row.close_l1_imbalance, row.trade_imbalance)
    ):
        raise ValueError("imbalance values must lie in [-1, 1]")


class StreamingMicrostructureFeatureEngine:
    """Emit the exact offline feature order after one hour of clean warmup."""

    def __init__(self, symbol: str, *, decision_cadence_seconds: int = 5) -> None:
        self.symbol = normalize_symbol(symbol)
        self.decision_cadence_seconds = int(decision_cadence_seconds)
        if self.decision_cadence_seconds <= 0 or self.decision_cadence_seconds > 60:
            raise ValueError("decision_cadence_seconds must lie in [1, 60]")
        self._history: deque[MicrostructureSecond] = deque(maxlen=_REQUIRED_SECONDS)
        self.gap_resets = 0

    @property
    def warmup_remaining_seconds(self) -> int:
        return max(0, _REQUIRED_SECONDS - len(self._history))

    @property
    def ready(self) -> bool:
        return self.warmup_remaining_seconds == 0

    def reset(self) -> None:
        self._history.clear()

    def append(self, row: MicrostructureSecond) -> StreamingFeatureRow | None:
        _validate_second(row, self.symbol)
        if self._history:
            delta = int(row.second_ms) - int(self._history[-1].second_ms)
            if delta <= 0:
                raise ValueError("microstructure seconds must be strictly increasing")
            if delta != 1_000:
                self._history.clear()
                self.gap_resets += 1
        self._history.append(row)
        if not self.ready:
            return None
        decision_time_ms = int(row.second_ms) + 1_000
        if (decision_time_ms // 1_000) % self.decision_cadence_seconds != 0:
            return None
        return self._build_row()

    def _build_row(self) -> StreamingFeatureRow:
        rows = list(self._history)
        previous_rows = rows[:-1]
        current_rows = rows[1:]

        def vector(name: str) -> np.ndarray:
            return np.asarray(
                [float(getattr(item, name)) for item in current_rows], dtype=np.float64
            )

        def embargoed_trade_vector(name: str) -> np.ndarray:
            if MICROSTRUCTURE_TRADE_EMBARGO_MS != 1_000:
                raise RuntimeError(
                    "streaming tape embargo implementation requires one second"
                )
            return np.asarray(
                [float(getattr(item, name)) for item in previous_rows],
                dtype=np.float64,
            )

        mids = vector("close_mid")
        previous_mids = np.asarray(
            [float(item.close_mid) for item in previous_rows], dtype=np.float64
        )
        log_returns = np.log(mids / previous_mids)
        highs = vector("high_mid")
        lows = vector("low_mid")
        bids = vector("close_bid")
        asks = vector("close_ask")
        bid_qty = vector("close_bid_qty")
        ask_qty = vector("close_ask_qty")
        previous_bids = np.asarray(
            [float(item.close_bid) for item in previous_rows], dtype=np.float64
        )
        previous_asks = np.asarray(
            [float(item.close_ask) for item in previous_rows], dtype=np.float64
        )
        previous_bid_qty = np.asarray(
            [float(item.close_bid_qty) for item in previous_rows], dtype=np.float64
        )
        previous_ask_qty = np.asarray(
            [float(item.close_ask_qty) for item in previous_rows], dtype=np.float64
        )
        normalized_ofi = (
            np.where(bids >= previous_bids, bid_qty, 0.0)
            - np.where(bids <= previous_bids, previous_bid_qty, 0.0)
            - np.where(asks <= previous_asks, ask_qty, 0.0)
            + np.where(asks >= previous_asks, previous_ask_qty, 0.0)
        ) / np.maximum(
            (bid_qty + ask_qty + previous_bid_qty + previous_ask_qty) / 2.0,
            1e-12,
        )
        spreads = vector("spread_bps")
        l1_imbalance = vector("l1_imbalance")
        close_l1_imbalance = vector("close_l1_imbalance")
        microprice = vector("microprice_offset_bps")
        quote_updates = vector("quote_updates")
        trade_imbalance = embargoed_trade_vector("trade_imbalance")
        buy_volume = embargoed_trade_vector("aggressive_buy_volume")
        sell_volume = embargoed_trade_vector("aggressive_sell_volume")
        signed_flow = buy_volume - sell_volume
        base_volume = embargoed_trade_vector("base_volume")
        delay_p99 = vector("event_delay_p99_ms")
        bid_depth_quote = bids * bid_qty
        ask_depth_quote = asks * ask_qty
        total_depth_quote = bid_depth_quote + ask_depth_quote

        return_bps = {
            size: _tail_sum(log_returns, size) * 10_000.0
            for size in (1, 5, 15, 30, 60, 120, 300, 900, 1_800, 3_600)
        }
        volatility = {
            size: float(np.std(log_returns[-size:], ddof=0))
            for size in (10, 30, 60, 120, 300, 900, 1_800, 3_600)
        }
        ranges = {
            size: (float(np.max(highs[-size:])) - float(np.min(lows[-size:])))
            * 10_000.0
            / mids[-1]
            for size in (60, 300, 900, 1_800, 3_600)
        }
        spread_mean_60 = _tail_mean(spreads, 60)
        spread_mean_300 = _tail_mean(spreads, 300)
        quote_mean_60 = _tail_mean(quote_updates, 60)
        quote_mean_300 = _tail_mean(quote_updates, 300)
        quote_mean_900 = _tail_mean(quote_updates, 900)
        base_volume_mean_60 = _tail_mean(base_volume, 60)
        base_volume_mean_300 = _tail_mean(base_volume, 300)
        base_volume_mean_900 = _tail_mean(base_volume, 900)
        trade_count = embargoed_trade_vector("trade_count")
        trade_count_mean_900 = _tail_mean(trade_count, 900)
        current = current_rows[-1]
        current_trade = previous_rows[-1]

        def delta(values: np.ndarray, lag: int) -> float:
            return float(values[-1] - values[-1 - lag])

        def signed_pressure_to_opposing_depth(size: int) -> float:
            flow = _tail_sum(signed_flow, size)
            capacity = ask_qty[-1] if flow >= 0.0 else bid_qty[-1]
            magnitude = math.log1p(abs(flow) / max(float(capacity), 1e-12))
            return math.copysign(magnitude, flow) if flow else 0.0

        epoch_second = current.second_ms // 1_000
        week_second = (epoch_second + 3 * 86_400) % 604_800
        utc_weekday = ((epoch_second // 86_400) + 3) % 7

        values = [
            *[return_bps[size] for size in (1, 5, 15, 30, 60, 120, 300, 900)],
            *[volatility[size] * 10_000.0 for size in (10, 30, 60, 120, 300, 900)],
            (current.high_mid - current.low_mid) * 10_000.0 / current.close_mid,
            ranges[60],
            ranges[300],
            ranges[900],
            current.spread_bps,
            current.max_spread_bps,
            _safe_ratio(current.spread_bps, spread_mean_60),
            _safe_ratio(current.spread_bps, spread_mean_300),
            current.l1_imbalance,
            current.close_l1_imbalance,
            _tail_mean(l1_imbalance, 10),
            _tail_mean(l1_imbalance, 60),
            _tail_mean(l1_imbalance, 300),
            current.microprice_offset_bps,
            normalized_ofi[-1],
            _tail_mean(normalized_ofi, 10),
            _tail_mean(normalized_ofi, 60),
            _tail_mean(normalized_ofi, 300),
            *[delta(normalized_ofi, lag) for lag in (5, 15, 30, 60)],
            math.log1p(current.quote_updates),
            _safe_ratio(current.quote_updates, quote_mean_60),
            _safe_ratio(current.quote_updates, quote_mean_300),
            current_trade.trade_imbalance,
            _tail_mean(trade_imbalance, 10),
            _tail_mean(trade_imbalance, 60),
            _tail_mean(trade_imbalance, 300),
            *[delta(trade_imbalance, lag) for lag in (5, 15, 30, 60)],
            _safe_ratio(_tail_sum(signed_flow, 10), _tail_sum(base_volume, 10)),
            _safe_ratio(_tail_sum(signed_flow, 60), _tail_sum(base_volume, 60)),
            _safe_ratio(_tail_sum(signed_flow, 300), _tail_sum(base_volume, 300)),
            math.log1p(current_trade.base_volume),
            _safe_ratio(current_trade.base_volume, base_volume_mean_60),
            _safe_ratio(current_trade.base_volume, base_volume_mean_300),
            math.log1p(current_trade.trade_count),
            (current_trade.trade_close / current.close_mid - 1.0) * 10_000.0,
            current.event_delay_p50_ms,
            current.event_delay_p99_ms,
            _safe_ratio(current.event_delay_p99_ms, _tail_mean(delay_p99, 60)),
            _safe_ratio(
                abs(_tail_sum(log_returns, 60)), _tail_sum(np.abs(log_returns), 60)
            ),
            _safe_ratio(
                abs(_tail_sum(log_returns, 300)), _tail_sum(np.abs(log_returns), 300)
            ),
            *[delta(close_l1_imbalance, lag) for lag in (5, 15, 30, 60)],
            *[delta(microprice, lag) for lag in (5, 15, 30, 60)],
            _safe_ratio(return_bps[60] / 10_000.0, volatility[60] * math.sqrt(60.0)),
            _safe_ratio(return_bps[300] / 10_000.0, volatility[300] * math.sqrt(300.0)),
            _safe_ratio(return_bps[900] / 10_000.0, volatility[900] * math.sqrt(900.0)),
            _safe_ratio(volatility[10], volatility[300]),
            _safe_ratio(volatility[60], volatility[900]),
            _safe_ratio(current.spread_bps, volatility[10] * 10_000.0),
            normalized_ofi[-1] * current_trade.trade_imbalance,
            math.log1p(current.quote_updates) - math.log1p(current_trade.trade_count),
            (
                2.0
                * (current.close_mid - current.low_mid)
                / (current.high_mid - current.low_mid)
                - 1.0
                if current.high_mid > current.low_mid
                else 0.0
            ),
            math.sin(
                2.0 * math.pi * ((current.second_ms // 1_000) % 86_400) / 86_400.0
            ),
            math.cos(
                2.0 * math.pi * ((current.second_ms // 1_000) % 86_400) / 86_400.0
            ),
            math.sin(
                2.0 * math.pi * ((current.second_ms // 1_000) % 28_800) / 28_800.0
            ),
            math.cos(
                2.0 * math.pi * ((current.second_ms // 1_000) % 28_800) / 28_800.0
            ),
            return_bps[1_800],
            return_bps[3_600],
            volatility[1_800] * 10_000.0,
            volatility[3_600] * 10_000.0,
            ranges[1_800],
            ranges[3_600],
            _safe_ratio(current.spread_bps, _tail_mean(spreads, 900)),
            _safe_ratio(current.quote_updates, quote_mean_900),
            _safe_ratio(current_trade.base_volume, base_volume_mean_900),
            _safe_ratio(current_trade.trade_count, trade_count_mean_900),
            _safe_ratio(
                abs(_tail_sum(log_returns, 900)), _tail_sum(np.abs(log_returns), 900)
            ),
            _safe_ratio(
                abs(_tail_sum(log_returns, 3_600)),
                _tail_sum(np.abs(log_returns), 3_600),
            ),
            _safe_ratio(
                return_bps[1_800] / 10_000.0,
                volatility[1_800] * math.sqrt(1_800.0),
            ),
            _safe_ratio(
                return_bps[3_600] / 10_000.0,
                volatility[3_600] * math.sqrt(3_600.0),
            ),
            _safe_ratio(volatility[300], volatility[3_600]),
            _safe_ratio(volatility[900], volatility[3_600]),
            math.sin(2.0 * math.pi * week_second / 604_800.0),
            math.cos(2.0 * math.pi * week_second / 604_800.0),
            1.0 if utc_weekday >= 5 else 0.0,
            math.log1p(bid_depth_quote[-1]),
            math.log1p(ask_depth_quote[-1]),
            _safe_ratio(total_depth_quote[-1], _tail_mean(total_depth_quote, 60)),
            _safe_ratio(total_depth_quote[-1], _tail_mean(total_depth_quote, 300)),
            *[signed_pressure_to_opposing_depth(size) for size in (10, 60, 300)],
        ]
        features = np.asarray(values, dtype=np.float32)
        if features.shape != (len(MICROSTRUCTURE_FEATURE_NAMES),):
            raise RuntimeError(
                "streaming microstructure feature count drifted from the offline contract"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError(
                "streaming microstructure features contain non-finite values"
            )
        return StreamingFeatureRow(
            symbol=self.symbol,
            feature_version=MICROSTRUCTURE_FEATURE_VERSION,
            feature_names=MICROSTRUCTURE_FEATURE_NAMES,
            source_second_ms=int(current.second_ms),
            decision_time_ms=int(current.second_ms) + 1_000,
            close_bid=float(current.close_bid),
            close_ask=float(current.close_ask),
            close_bid_qty=float(current.close_bid_qty),
            close_ask_qty=float(current.close_ask_qty),
            features=features,
        )


__all__ = [
    "MICROSTRUCTURE_STREAM_WARMUP_SECONDS",
    "MicrostructureSecond",
    "StreamingFeatureRow",
    "StreamingMicrostructureFeatureEngine",
]
