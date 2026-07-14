"""Exact stop-or-timeout payoffs from verified one-minute futures bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .cross_asset_cost_data import MINUTE_MS, MinuteSeries, SYMBOLS
from .derivatives_hurdle_data import FundingState


STOP_EVENT = 0
TIMEOUT_EVENT = 1
EVENT_NAMES = ("stop_loss", "timeout")


@dataclass(frozen=True)
class StopTimeSpecification:
    horizon_minutes: int
    stop_volatility_multiple: float
    minimum_stop_bps: float
    maximum_stop_bps: float
    round_trip_execution_charge_bps: float

    def validate(self) -> None:
        numeric = np.asarray(
            [
                self.stop_volatility_multiple,
                self.minimum_stop_bps,
                self.maximum_stop_bps,
                self.round_trip_execution_charge_bps,
            ],
            dtype=np.float64,
        )
        if (
            self.horizon_minutes <= 0
            or not np.isfinite(numeric).all()
            or self.stop_volatility_multiple <= 0.0
            or self.minimum_stop_bps <= self.round_trip_execution_charge_bps
            or self.maximum_stop_bps < self.minimum_stop_bps
            or self.round_trip_execution_charge_bps < 0.0
        ):
            raise ValueError("stop-time specification is invalid")

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StopTimePayoffDataset:
    timestamps_ms: np.ndarray
    stop_bps: np.ndarray
    long_event_code: np.ndarray
    short_event_code: np.ndarray
    long_event_minute: np.ndarray
    short_event_minute: np.ndarray
    long_exit_time_ms: np.ndarray
    short_exit_time_ms: np.ndarray
    long_price_return_bps: np.ndarray
    short_price_return_bps: np.ndarray
    long_funding_cash_flow_bps: np.ndarray
    short_funding_cash_flow_bps: np.ndarray
    long_net_payoff_bps: np.ndarray
    short_net_payoff_bps: np.ndarray
    long_gap_through_slippage_bps: np.ndarray
    short_gap_through_slippage_bps: np.ndarray
    source_dataset_sha256: str
    specification: StopTimeSpecification
    dataset_sha256: str

    @property
    def timestamps(self) -> int:
        return int(np.asarray(self.timestamps_ms).size)

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
    view = memoryview(array).cast("B")
    chunk_bytes = 8 * 1024 * 1024
    for offset in range(0, view.nbytes, chunk_bytes):
        digest.update(view[offset : offset + chunk_bytes])
    return digest.hexdigest()


def _funding_in_holding_window(
    state: FundingState,
    entry_time_ms: np.ndarray,
    exit_time_ms: np.ndarray,
) -> np.ndarray:
    rates_bps = np.asarray(state.event_rate, dtype=np.float64) * 10_000.0
    prefix = np.concatenate(([0.0], np.cumsum(rates_bps, dtype=np.float64)))
    after_entry = np.searchsorted(state.event_time_ms, entry_time_ms, side="right")
    through_exit = np.searchsorted(state.event_time_ms, exit_time_ms, side="right")
    return prefix[through_exit] - prefix[after_entry]


def _simulate_side(
    series: MinuteSeries,
    funding: FundingState,
    decision_indices: np.ndarray,
    stop_bps: np.ndarray,
    *,
    side: int,
    specification: StopTimeSpecification,
) -> tuple[np.ndarray, ...]:
    if side not in (-1, 1):
        raise ValueError("stop-time side must be -1 or 1")
    horizon = int(specification.horizon_minutes)
    entry_indices = np.asarray(decision_indices, dtype=np.int64) + 1
    timeout_indices = entry_indices + horizon
    if np.any(timeout_indices >= series.open.size):
        raise ValueError("stop-time horizon exceeds the verified minute series")

    entry = np.asarray(series.open[entry_indices], dtype=np.float64)
    stop_price = entry * (1.0 - side * stop_bps / 10_000.0)
    event_code = np.full(entry.size, TIMEOUT_EVENT, dtype=np.int8)
    event_minute = np.full(entry.size, horizon, dtype=np.int16)
    fill_price = np.asarray(series.open[timeout_indices], dtype=np.float64).copy()
    gap_slippage = np.zeros(entry.size, dtype=np.float64)
    unresolved = np.ones(entry.size, dtype=bool)

    for offset in range(horizon):
        if not np.any(unresolved):
            break
        bar_indices = entry_indices + offset
        bar_open = np.asarray(series.open[bar_indices], dtype=np.float64)
        if side == 1:
            stopped = unresolved & (series.low[bar_indices] <= stop_price)
        else:
            stopped = unresolved & (series.high[bar_indices] >= stop_price)
        if not np.any(stopped):
            continue
        if side == 1:
            stop_fill = np.minimum(stop_price[stopped], bar_open[stopped])
            gap_slippage[stopped] = np.maximum(
                0.0,
                10_000.0
                * (stop_price[stopped] - stop_fill)
                / entry[stopped],
            )
        else:
            stop_fill = np.maximum(stop_price[stopped], bar_open[stopped])
            gap_slippage[stopped] = np.maximum(
                0.0,
                10_000.0
                * (stop_fill - stop_price[stopped])
                / entry[stopped],
            )
        fill_price[stopped] = stop_fill
        event_code[stopped] = STOP_EVENT
        event_minute[stopped] = offset + 1
        unresolved[stopped] = False

    entry_time = np.asarray(series.open_time_ms[entry_indices], dtype=np.int64)
    exit_time = entry_time + event_minute.astype(np.int64) * MINUTE_MS
    funding_bps = _funding_in_holding_window(funding, entry_time, exit_time)
    if side == 1:
        price_return_bps = 10_000.0 * (fill_price / entry - 1.0)
        funding_cash_flow_bps = -funding_bps
    else:
        price_return_bps = 10_000.0 * (1.0 - fill_price / entry)
        funding_cash_flow_bps = funding_bps
    net_payoff_bps = (
        price_return_bps
        + funding_cash_flow_bps
        - specification.round_trip_execution_charge_bps
    )
    outputs = (
        event_code,
        event_minute,
        exit_time,
        price_return_bps,
        funding_cash_flow_bps,
        net_payoff_bps,
        gap_slippage,
    )
    if not all(np.isfinite(value).all() for value in outputs[3:]):
        raise ValueError("stop-time simulation produced nonfinite values")
    stopped = event_code == STOP_EVENT
    if np.any(stopped) and not np.all(
        price_return_bps[stopped] <= -stop_bps[stopped] + 1e-7
    ):
        raise RuntimeError("stop-time payoff violates the stop loss")
    return outputs


def build_stop_time_payoff_dataset(
    panel: Mapping[str, MinuteSeries],
    funding: Mapping[str, FundingState],
    decision_timestamps_ms: np.ndarray,
    one_minute_volatility_bps: np.ndarray,
    *,
    source_dataset_sha256: str,
    specification: StopTimeSpecification,
) -> StopTimePayoffDataset:
    """Build side-specific after-cost payoffs without copying feature data."""

    specification.validate()
    timestamps_ms = np.asarray(decision_timestamps_ms, dtype=np.int64)
    volatility = np.asarray(one_minute_volatility_bps, dtype=np.float64)
    expected_shape = (timestamps_ms.size, len(SYMBOLS))
    if (
        set(panel) != set(SYMBOLS)
        or set(funding) != set(SYMBOLS)
        or timestamps_ms.ndim != 1
        or timestamps_ms.size == 0
        or np.any(np.diff(timestamps_ms) <= 0)
        or volatility.shape != expected_shape
        or not np.isfinite(volatility).all()
        or np.any(volatility < 0.0)
        or len(source_dataset_sha256) != 64
    ):
        raise ValueError("stop-time source contract is invalid")

    stop_bps = np.clip(
        volatility
        * math.sqrt(float(specification.horizon_minutes))
        * specification.stop_volatility_multiple,
        specification.minimum_stop_bps,
        specification.maximum_stop_bps,
    )
    outputs: dict[str, np.ndarray] = {
        "long_event_code": np.empty(expected_shape, dtype=np.int8),
        "short_event_code": np.empty(expected_shape, dtype=np.int8),
        "long_event_minute": np.empty(expected_shape, dtype=np.int16),
        "short_event_minute": np.empty(expected_shape, dtype=np.int16),
        "long_exit_time_ms": np.empty(expected_shape, dtype=np.int64),
        "short_exit_time_ms": np.empty(expected_shape, dtype=np.int64),
        "long_price_return_bps": np.empty(expected_shape, dtype=np.float32),
        "short_price_return_bps": np.empty(expected_shape, dtype=np.float32),
        "long_funding_cash_flow_bps": np.empty(expected_shape, dtype=np.float32),
        "short_funding_cash_flow_bps": np.empty(expected_shape, dtype=np.float32),
        "long_net_payoff_bps": np.empty(expected_shape, dtype=np.float32),
        "short_net_payoff_bps": np.empty(expected_shape, dtype=np.float32),
        "long_gap_through_slippage_bps": np.empty(expected_shape, dtype=np.float32),
        "short_gap_through_slippage_bps": np.empty(expected_shape, dtype=np.float32),
    }
    field_order = (
        "event_code",
        "event_minute",
        "exit_time_ms",
        "price_return_bps",
        "funding_cash_flow_bps",
        "net_payoff_bps",
        "gap_through_slippage_bps",
    )
    for symbol_index, symbol in enumerate(SYMBOLS):
        series = panel[symbol]
        decision_indices = np.searchsorted(series.open_time_ms, timestamps_ms)
        if (
            np.any(decision_indices >= series.open_time_ms.size)
            or not np.array_equal(series.open_time_ms[decision_indices], timestamps_ms)
        ):
            raise ValueError(f"stop-time decision timestamps differ for {symbol}")
        for side_name, side in (("long", 1), ("short", -1)):
            values = _simulate_side(
                series,
                funding[symbol],
                decision_indices,
                stop_bps[:, symbol_index],
                side=side,
                specification=specification,
            )
            for field, value in zip(field_order, values, strict=True):
                outputs[f"{side_name}_{field}"][:, symbol_index] = value

    streams = {
        "timestamps_ms": _stream_array_sha256(
            timestamps_ms, label="stop-time-timestamps"
        ),
        "stop_bps": _stream_array_sha256(stop_bps, label="stop-time-stop-bps"),
        "long_event_code": _stream_array_sha256(
            outputs["long_event_code"], label="stop-time-long-event"
        ),
        "short_event_code": _stream_array_sha256(
            outputs["short_event_code"], label="stop-time-short-event"
        ),
        "long_net_payoff_bps": _stream_array_sha256(
            outputs["long_net_payoff_bps"], label="stop-time-long-net"
        ),
        "short_net_payoff_bps": _stream_array_sha256(
            outputs["short_net_payoff_bps"], label="stop-time-short-net"
        ),
    }
    identity = _canonical_sha256(
        {
            "schema": "stop-time-payoff-dataset-v1",
            "source_dataset_sha256": source_dataset_sha256,
            "symbols": list(SYMBOLS),
            "specification": specification.asdict(),
            "streams": streams,
        }
    )
    return StopTimePayoffDataset(
        timestamps_ms=timestamps_ms.copy(),
        stop_bps=stop_bps.astype(np.float32),
        source_dataset_sha256=source_dataset_sha256,
        specification=specification,
        dataset_sha256=identity,
        **outputs,
    )


__all__ = [
    "EVENT_NAMES",
    "STOP_EVENT",
    "TIMEOUT_EVENT",
    "StopTimePayoffDataset",
    "StopTimeSpecification",
    "build_stop_time_payoff_dataset",
]
