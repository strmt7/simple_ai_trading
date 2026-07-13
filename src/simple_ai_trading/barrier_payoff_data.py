"""Path-bounded long/short payoff targets from verified minute futures data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .cross_asset_cost_data import MINUTE_MS, MinuteSeries, SYMBOLS
from .derivatives_hurdle_data import (
    DerivativesHurdleDataset,
    FundingState,
)
from .minute_logistic_mixture_tcn_model import MinuteTemporalDataset


STOP_EVENT = 0
TIMEOUT_EVENT = 1
TAKE_PROFIT_EVENT = 2
EVENT_NAMES = ("stop_loss", "timeout", "take_profit")
SIDE_NAMES = ("short", "long")
REALIZED_VOLATILITY_FEATURE = "target_realized_volatility_60m_bps"


@dataclass(frozen=True)
class BarrierSpecification:
    horizon_minutes: int
    stop_volatility_multiple: float
    take_profit_to_stop_ratio: float
    minimum_stop_bps: float
    maximum_stop_bps: float
    round_trip_execution_charge_bps: float

    def validate(self) -> None:
        values = np.asarray(
            [
                self.stop_volatility_multiple,
                self.take_profit_to_stop_ratio,
                self.minimum_stop_bps,
                self.maximum_stop_bps,
                self.round_trip_execution_charge_bps,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("barrier specification contains nonfinite values")
        if self.horizon_minutes <= 0:
            raise ValueError("barrier horizon must be positive")
        if self.stop_volatility_multiple <= 0.0:
            raise ValueError("barrier volatility multiple must be positive")
        if self.take_profit_to_stop_ratio <= 1.0:
            raise ValueError("take-profit/stop ratio must exceed one")
        if self.minimum_stop_bps <= self.round_trip_execution_charge_bps:
            raise ValueError("minimum stop must exceed the round-trip execution charge")
        if self.maximum_stop_bps < self.minimum_stop_bps:
            raise ValueError("maximum stop must not be below the minimum stop")
        if self.round_trip_execution_charge_bps < 0.0:
            raise ValueError("round-trip execution charge must be nonnegative")

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BarrierPayoffDataset:
    timestamps_ms: np.ndarray
    stop_bps: np.ndarray
    take_profit_bps: np.ndarray
    event_code: np.ndarray
    event_minute: np.ndarray
    price_return_bps: np.ndarray
    funding_cash_flow_bps: np.ndarray
    net_payoff_bps: np.ndarray
    gap_through_slippage_bps: np.ndarray
    ambiguous_stop_first: np.ndarray
    role_masks: Mapping[str, np.ndarray]
    specification: BarrierSpecification
    dataset_sha256: str

    @property
    def timestamps(self) -> int:
        return int(self.timestamps_ms.size)

    @property
    def rows(self) -> int:
        return self.timestamps * len(SYMBOLS)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stream_array_sha256(values: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(label.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    byte_view = memoryview(array).cast("B")
    chunk_bytes = 8 * 1024 * 1024
    for offset in range(0, byte_view.nbytes, chunk_bytes):
        digest.update(byte_view[offset : offset + chunk_bytes])
    return digest.hexdigest()


def _funding_in_holding_window(
    state: FundingState,
    entry_time_ms: np.ndarray,
    exit_time_ms: np.ndarray,
) -> np.ndarray:
    rates_bps = state.event_rate.astype(np.float64, copy=False) * 10_000.0
    prefix = np.concatenate(([0.0], np.cumsum(rates_bps, dtype=np.float64)))
    after_entry = np.searchsorted(state.event_time_ms, entry_time_ms, side="right")
    through_exit = np.searchsorted(state.event_time_ms, exit_time_ms, side="right")
    return prefix[through_exit] - prefix[after_entry]


def _source_role_masks(
    source: DerivativesHurdleDataset,
    *,
    horizon_minutes: int,
    timestamps: int,
) -> dict[str, np.ndarray]:
    if horizon_minutes not in source.role_masks:
        raise ValueError(f"source has no {horizon_minutes}-minute role masks")
    symbol_count = len(SYMBOLS)
    roles: dict[str, np.ndarray] = {}
    for role in ("training", "early_stop", "calibration", "viability"):
        blocks = source.role_masks[horizon_minutes][role].reshape(
            symbol_count, timestamps
        )
        if not np.all(blocks == blocks[:1]):
            raise ValueError(f"barrier source role differs across symbols: {role}")
        roles[role] = blocks[0].copy()
        if not np.any(roles[role]):
            raise ValueError(f"barrier source role is empty: {role}")
    if any(
        np.any(roles[left] & roles[right])
        for left_index, left in enumerate(roles)
        for right in tuple(roles)[left_index + 1 :]
    ):
        raise ValueError("barrier chronological roles overlap")
    return roles


def _simulate_side(
    series: MinuteSeries,
    funding: FundingState,
    decision_indices: np.ndarray,
    stop_bps: np.ndarray,
    take_profit_bps: np.ndarray,
    *,
    side: int,
    specification: BarrierSpecification,
) -> tuple[np.ndarray, ...]:
    if side not in (-1, 1):
        raise ValueError("barrier side must be -1 or 1")
    horizon = specification.horizon_minutes
    entry_indices = decision_indices + 1
    timeout_indices = entry_indices + horizon
    if np.any(timeout_indices >= series.open.size):
        raise ValueError("barrier horizon exceeds the verified minute series")
    entry = series.open[entry_indices].astype(np.float64, copy=False)
    if side == 1:
        stop_price = entry * (1.0 - stop_bps / 10_000.0)
        take_price = entry * (1.0 + take_profit_bps / 10_000.0)
    else:
        stop_price = entry * (1.0 + stop_bps / 10_000.0)
        take_price = entry * (1.0 - take_profit_bps / 10_000.0)

    event_code = np.full(entry.size, TIMEOUT_EVENT, dtype=np.int8)
    event_minute = np.full(entry.size, horizon, dtype=np.int16)
    fill_price = series.open[timeout_indices].astype(np.float64, copy=True)
    gap_slippage = np.zeros(entry.size, dtype=np.float64)
    ambiguous = np.zeros(entry.size, dtype=bool)
    unresolved = np.ones(entry.size, dtype=bool)
    for offset in range(horizon):
        if not np.any(unresolved):
            break
        bar_indices = entry_indices + offset
        bar_open = series.open[bar_indices]
        if side == 1:
            stop_hit = unresolved & (series.low[bar_indices] <= stop_price)
            take_hit = unresolved & (series.high[bar_indices] >= take_price)
        else:
            stop_hit = unresolved & (series.high[bar_indices] >= stop_price)
            take_hit = unresolved & (series.low[bar_indices] <= take_price)
        both = stop_hit & take_hit
        ambiguous |= both
        if np.any(stop_hit):
            if side == 1:
                stop_fill = np.minimum(stop_price[stop_hit], bar_open[stop_hit])
                gap_slippage[stop_hit] = np.maximum(
                    0.0,
                    10_000.0 * (stop_price[stop_hit] - stop_fill) / entry[stop_hit],
                )
            else:
                stop_fill = np.maximum(stop_price[stop_hit], bar_open[stop_hit])
                gap_slippage[stop_hit] = np.maximum(
                    0.0,
                    10_000.0 * (stop_fill - stop_price[stop_hit]) / entry[stop_hit],
                )
            fill_price[stop_hit] = stop_fill
            event_code[stop_hit] = STOP_EVENT
            event_minute[stop_hit] = offset + 1
        take_only = take_hit & ~stop_hit
        if np.any(take_only):
            fill_price[take_only] = take_price[take_only]
            event_code[take_only] = TAKE_PROFIT_EVENT
            event_minute[take_only] = offset + 1
        unresolved &= ~(stop_hit | take_only)

    entry_time = series.open_time_ms[entry_indices]
    exit_time = entry_time + event_minute.astype(np.int64) * MINUTE_MS
    funding_bps = _funding_in_holding_window(funding, entry_time, exit_time)
    if side == 1:
        price_return_bps = 10_000.0 * (fill_price / entry - 1.0)
        net_payoff_bps = (
            price_return_bps
            - specification.round_trip_execution_charge_bps
            - funding_bps
        )
    else:
        price_return_bps = 10_000.0 * (1.0 - fill_price / entry)
        net_payoff_bps = (
            price_return_bps
            - specification.round_trip_execution_charge_bps
            + funding_bps
        )
    outputs = (
        event_code,
        event_minute,
        price_return_bps,
        funding_bps,
        net_payoff_bps,
        gap_slippage,
        ambiguous,
    )
    if not all(np.isfinite(value).all() for value in outputs[2:-1]):
        raise ValueError("barrier simulation produced nonfinite values")
    return outputs


def build_barrier_payoff_dataset(
    panel: Mapping[str, MinuteSeries],
    funding: Mapping[str, FundingState],
    source: DerivativesHurdleDataset,
    temporal: MinuteTemporalDataset,
    specification: BarrierSpecification,
) -> BarrierPayoffDataset:
    """Build exact stop/take/timeout targets without persisting a feature copy."""

    specification.validate()
    if set(panel) != set(SYMBOLS) or set(funding) != set(SYMBOLS):
        raise ValueError("barrier panel or funding symbols are incomplete")
    if REALIZED_VOLATILITY_FEATURE not in temporal.feature_names:
        raise ValueError("barrier volatility feature is absent")
    timestamps = temporal.timestamps
    if source.rows != timestamps * len(SYMBOLS):
        raise ValueError("barrier source and temporal row counts differ")
    role_masks = _source_role_masks(
        source,
        horizon_minutes=specification.horizon_minutes,
        timestamps=timestamps,
    )
    volatility_index = temporal.feature_names.index(REALIZED_VOLATILITY_FEATURE)
    one_minute_volatility = temporal.features[..., volatility_index].astype(
        np.float64, copy=False
    )
    stop_bps = np.clip(
        one_minute_volatility
        * math.sqrt(float(specification.horizon_minutes))
        * specification.stop_volatility_multiple,
        specification.minimum_stop_bps,
        specification.maximum_stop_bps,
    )
    take_profit_bps = stop_bps * specification.take_profit_to_stop_ratio
    if not np.isfinite(stop_bps).all() or not np.isfinite(take_profit_bps).all():
        raise ValueError("barrier widths are nonfinite")

    shape = (timestamps, len(SYMBOLS), len(SIDE_NAMES))
    event_code = np.empty(shape, dtype=np.int8)
    event_minute = np.empty(shape, dtype=np.int16)
    price_return_bps = np.empty(shape, dtype=np.float32)
    funding_cash_flow_bps = np.empty(shape, dtype=np.float32)
    net_payoff_bps = np.empty(shape, dtype=np.float32)
    gap_through_slippage_bps = np.empty(shape, dtype=np.float32)
    ambiguous_stop_first = np.empty(shape, dtype=bool)
    for symbol_index, symbol in enumerate(SYMBOLS):
        series = panel[symbol]
        decision_indices = np.searchsorted(series.open_time_ms, temporal.timestamps_ms)
        if np.any(decision_indices >= series.open_time_ms.size) or not np.array_equal(
            series.open_time_ms[decision_indices], temporal.timestamps_ms
        ):
            raise ValueError(f"barrier decision timestamps differ for {symbol}")
        for side_index, side in enumerate((-1, 1)):
            outputs = _simulate_side(
                series,
                funding[symbol],
                decision_indices,
                stop_bps[:, symbol_index],
                take_profit_bps[:, symbol_index],
                side=side,
                specification=specification,
            )
            event_code[:, symbol_index, side_index] = outputs[0]
            event_minute[:, symbol_index, side_index] = outputs[1]
            price_return_bps[:, symbol_index, side_index] = outputs[2]
            funding_cash_flow_bps[:, symbol_index, side_index] = outputs[3]
            net_payoff_bps[:, symbol_index, side_index] = outputs[4]
            gap_through_slippage_bps[:, symbol_index, side_index] = outputs[5]
            ambiguous_stop_first[:, symbol_index, side_index] = outputs[6]

    for symbol_index in range(len(SYMBOLS)):
        for side_index in range(len(SIDE_NAMES)):
            stopped = event_code[:, symbol_index, side_index] == STOP_EVENT
            taken = event_code[:, symbol_index, side_index] == TAKE_PROFIT_EVENT
            if np.any(stopped) and not np.all(
                price_return_bps[stopped, symbol_index, side_index]
                <= -stop_bps[stopped, symbol_index] + 1e-3
            ):
                raise RuntimeError("barrier stop payoff violates its price bound")
            if np.any(taken) and not np.allclose(
                price_return_bps[taken, symbol_index, side_index],
                take_profit_bps[taken, symbol_index],
                rtol=0.0,
                atol=1e-3,
            ):
                raise RuntimeError(
                    "barrier take-profit payoff violates its price bound"
                )

    streams = {
        "stop_bps": _stream_array_sha256(stop_bps, label="barrier-stop-bps"),
        "take_profit_bps": _stream_array_sha256(
            take_profit_bps, label="barrier-take-profit-bps"
        ),
        "event_code": _stream_array_sha256(event_code, label="barrier-event-code"),
        "event_minute": _stream_array_sha256(
            event_minute, label="barrier-event-minute"
        ),
        "net_payoff_bps": _stream_array_sha256(
            net_payoff_bps, label="barrier-net-payoff-bps"
        ),
    }
    identity = _canonical_sha256(
        {
            "schema": "path-bounded-barrier-payoff-dataset-v1",
            "predecessor_dataset_sha256": temporal.dataset_sha256,
            "symbols": list(SYMBOLS),
            "specification": specification.asdict(),
            "streams": streams,
        }
    )
    return BarrierPayoffDataset(
        timestamps_ms=temporal.timestamps_ms.copy(),
        stop_bps=stop_bps.astype(np.float32),
        take_profit_bps=take_profit_bps.astype(np.float32),
        event_code=event_code,
        event_minute=event_minute,
        price_return_bps=price_return_bps,
        funding_cash_flow_bps=funding_cash_flow_bps,
        net_payoff_bps=net_payoff_bps,
        gap_through_slippage_bps=gap_through_slippage_bps,
        ambiguous_stop_first=ambiguous_stop_first,
        role_masks=role_masks,
        specification=specification,
        dataset_sha256=identity,
    )


__all__ = [
    "BarrierPayoffDataset",
    "BarrierSpecification",
    "EVENT_NAMES",
    "SIDE_NAMES",
    "STOP_EVENT",
    "TAKE_PROFIT_EVENT",
    "TIMEOUT_EVENT",
    "build_barrier_payoff_dataset",
]
