"""Causal 100 ms BBO barrier targets with explicit protection-gap stress."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np
from numba import njit, prange

from .microstructure_features import (
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .microstructure_warehouse import MicrostructureWarehouse


ADAPTIVE_BARRIER_SCHEMA_VERSION = "adaptive-bbo-barrier-targets-v1"
ADAPTIVE_BARRIER_TARGET_MODE = (
    "exchange_trigger_market_exit_100ms_base_and_adverse_stress_v1"
)
_DAY_MS = 86_400_000
_OUTCOME_HORIZON = 0
_OUTCOME_STOP = 1
_OUTCOME_TAKE = 2
_OUTCOME_AMBIGUOUS_STOP = 3
_OUTCOME_PROTECTION_GAP_STOP = 4
_OUTCOME_NAMES = {
    _OUTCOME_HORIZON: "horizon",
    _OUTCOME_STOP: "stop",
    _OUTCOME_TAKE: "take",
    _OUTCOME_AMBIGUOUS_STOP: "ambiguous_stop",
    _OUTCOME_PROTECTION_GAP_STOP: "protection_gap_stop",
}


@dataclass(frozen=True)
class AdaptiveBarrierSpec:
    horizon_seconds: int
    volatility_feature_name: str
    stop_volatility_multiple: float
    take_volatility_multiple: float
    minimum_stop_bps: float
    maximum_stop_bps: float
    minimum_take_bps: float
    maximum_take_bps: float
    base_protection_delay_ms: int
    stress_protection_delay_ms: int
    trigger_execution_slippage_bps: float
    path_resolution_ms: int = 100
    same_utc_day_exit: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.stop_volatility_multiple,
            self.take_volatility_multiple,
            self.minimum_stop_bps,
            self.maximum_stop_bps,
            self.minimum_take_bps,
            self.maximum_take_bps,
            self.trigger_execution_slippage_bps,
        )
        if not 60 <= int(self.horizon_seconds) <= 7_200:
            raise ValueError("adaptive barrier horizon is outside the research bounds")
        if not self.volatility_feature_name.startswith("realized_volatility_"):
            raise ValueError("adaptive barrier volatility feature is unsupported")
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("adaptive barrier values must be finite")
        if (
            self.stop_volatility_multiple <= 0.0
            or self.take_volatility_multiple < self.stop_volatility_multiple
            or self.minimum_stop_bps <= 0.0
            or self.maximum_stop_bps < self.minimum_stop_bps
            or self.minimum_take_bps <= self.minimum_stop_bps
            or self.maximum_take_bps <= self.maximum_stop_bps
            or self.trigger_execution_slippage_bps < 0.0
        ):
            raise ValueError("adaptive barrier price bounds are invalid")
        if (
            int(self.base_protection_delay_ms) < 0
            or int(self.stress_protection_delay_ms) < int(self.base_protection_delay_ms)
            or int(self.stress_protection_delay_ms) > 5_000
        ):
            raise ValueError("adaptive barrier protection delays are invalid")
        if int(self.path_resolution_ms) != 100:
            raise ValueError("adaptive barrier path resolution must be 100 ms")
        if self.same_utc_day_exit is not True:
            raise ValueError("adaptive barrier positions must exit within the UTC day")


@dataclass(frozen=True)
class AdaptiveBarrierTargets:
    schema_version: str
    target_mode: str
    spec: AdaptiveBarrierSpec
    source_indexes: np.ndarray
    valid: np.ndarray
    stop_barrier_bps: np.ndarray
    take_barrier_bps: np.ndarray
    base_long_net_bps: np.ndarray
    base_short_net_bps: np.ndarray
    base_long_exit_time_ms: np.ndarray
    base_short_exit_time_ms: np.ndarray
    base_long_outcome: np.ndarray
    base_short_outcome: np.ndarray
    stress_long_net_bps: np.ndarray
    stress_short_net_bps: np.ndarray
    stress_long_exit_time_ms: np.ndarray
    stress_short_exit_time_ms: np.ndarray
    stress_long_outcome: np.ndarray
    stress_short_outcome: np.ndarray
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    @property
    def rows(self) -> int:
        return int(len(self.source_indexes))

    @property
    def valid_rows(self) -> int:
        return int(np.sum(self.valid))

    def summary(self) -> dict[str, object]:
        valid = np.asarray(self.valid, dtype=bool)

        def outcome_counts(values: np.ndarray) -> dict[str, int]:
            return {
                name: int(np.sum(values[valid] == code))
                for code, name in _OUTCOME_NAMES.items()
            }

        def target_summary(values: np.ndarray) -> dict[str, float | int]:
            selected = np.asarray(values[valid], dtype=np.float64)
            return {
                "rows": int(len(selected)),
                "positive_rows": int(np.sum(selected > 0.0)),
                "positive_ratio": float(np.mean(selected > 0.0)),
                "mean_net_bps": float(np.mean(selected)),
                "p01_net_bps": float(np.quantile(selected, 0.01)),
                "p50_net_bps": float(np.quantile(selected, 0.50)),
                "p99_net_bps": float(np.quantile(selected, 0.99)),
            }

        return {
            "schema_version": self.schema_version,
            "target_mode": self.target_mode,
            "spec": asdict(self.spec),
            "rows": self.rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.rows - self.valid_rows,
            "stop_barrier_bps": _finite_quantiles(self.stop_barrier_bps[valid]),
            "take_barrier_bps": _finite_quantiles(self.take_barrier_bps[valid]),
            "base": {
                "long": target_summary(self.base_long_net_bps),
                "short": target_summary(self.base_short_net_bps),
                "long_outcomes": outcome_counts(self.base_long_outcome),
                "short_outcomes": outcome_counts(self.base_short_outcome),
            },
            "stress": {
                "long": target_summary(self.stress_long_net_bps),
                "short": target_summary(self.stress_short_net_bps),
                "long_outcomes": outcome_counts(self.stress_long_outcome),
                "short_outcomes": outcome_counts(self.stress_short_outcome),
            },
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }


def _finite_quantiles(values: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values, dtype=np.float64)
    if selected.size == 0 or not np.all(np.isfinite(selected)):
        raise ValueError("adaptive barrier quantiles require finite values")
    return {
        "minimum": float(np.min(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p50": float(np.quantile(selected, 0.50)),
        "p90": float(np.quantile(selected, 0.90)),
        "maximum": float(np.max(selected)),
    }


def volatility_scaled_barriers(
    dataset: MicrostructureDataset,
    source_indexes: np.ndarray,
    spec: AdaptiveBarrierSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a causal per-second volatility estimate into bounded path barriers."""

    validate_microstructure_dataset(dataset)
    indexes = np.asarray(source_indexes, dtype=np.int64)
    if (
        indexes.ndim != 1
        or indexes.size == 0
        or np.any(np.diff(indexes) <= 0)
        or indexes[0] < 0
        or indexes[-1] >= dataset.rows
        or dataset.horizon_seconds != spec.horizon_seconds
    ):
        raise ValueError("adaptive barrier source indexes are invalid")
    try:
        feature_index = dataset.feature_names.index(spec.volatility_feature_name)
    except ValueError as exc:
        raise ValueError("adaptive barrier volatility feature is absent") from exc
    per_second = np.asarray(dataset.features[indexes, feature_index], dtype=np.float64)
    if np.any(~np.isfinite(per_second)) or np.any(per_second <= 0.0):
        raise ValueError("adaptive barrier volatility values are invalid")
    sigma_horizon = per_second * math.sqrt(float(spec.horizon_seconds))
    stop = np.clip(
        spec.stop_volatility_multiple * sigma_horizon,
        spec.minimum_stop_bps,
        spec.maximum_stop_bps,
    )
    take = np.clip(
        spec.take_volatility_multiple * sigma_horizon,
        spec.minimum_take_bps,
        spec.maximum_take_bps,
    )
    if np.any(take <= stop):
        raise ValueError("adaptive barrier take distance must exceed stop distance")
    return stop, take


@njit(cache=True)
def _extreme_trees(  # pragma: no cover - assertions execute the compiled Numba kernel
    min_bid: np.ndarray,
    max_bid: np.ndarray,
    min_ask: np.ndarray,
    max_ask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(min_bid)
    size = 1
    while size < rows:
        size *= 2
    tree_min_bid = np.full(2 * size, np.inf, dtype=np.float64)
    tree_max_bid = np.full(2 * size, -np.inf, dtype=np.float64)
    tree_min_ask = np.full(2 * size, np.inf, dtype=np.float64)
    tree_max_ask = np.full(2 * size, -np.inf, dtype=np.float64)
    tree_min_bid[size : size + rows] = min_bid
    tree_max_bid[size : size + rows] = max_bid
    tree_min_ask[size : size + rows] = min_ask
    tree_max_ask[size : size + rows] = max_ask
    for node in range(size - 1, 0, -1):
        left = 2 * node
        right = left + 1
        tree_min_bid[node] = min(tree_min_bid[left], tree_min_bid[right])
        tree_max_bid[node] = max(tree_max_bid[left], tree_max_bid[right])
        tree_min_ask[node] = min(tree_min_ask[left], tree_min_ask[right])
        tree_max_ask[node] = max(tree_max_ask[left], tree_max_ask[right])
    return size, tree_min_bid, tree_max_bid, tree_min_ask, tree_max_ask


@njit(cache=True, inline="always")
def _long_crosses(  # pragma: no cover - executed inside compiled Numba kernels
    tree_min_bid: np.ndarray,
    tree_max_bid: np.ndarray,
    size: int,
    start: int,
    end: int,
    stop: float,
    take: float,
) -> bool:
    left = start + size
    right = end + size
    minimum = np.inf
    maximum = -np.inf
    while left < right:
        if left & 1:
            minimum = min(minimum, tree_min_bid[left])
            maximum = max(maximum, tree_max_bid[left])
            left += 1
        if right & 1:
            right -= 1
            minimum = min(minimum, tree_min_bid[right])
            maximum = max(maximum, tree_max_bid[right])
        left //= 2
        right //= 2
    return minimum <= stop or maximum >= take


@njit(cache=True, inline="always")
def _short_crosses(  # pragma: no cover - executed inside compiled Numba kernels
    tree_min_ask: np.ndarray,
    tree_max_ask: np.ndarray,
    size: int,
    start: int,
    end: int,
    stop: float,
    take: float,
) -> bool:
    left = start + size
    right = end + size
    minimum = np.inf
    maximum = -np.inf
    while left < right:
        if left & 1:
            minimum = min(minimum, tree_min_ask[left])
            maximum = max(maximum, tree_max_ask[left])
            left += 1
        if right & 1:
            right -= 1
            minimum = min(minimum, tree_min_ask[right])
            maximum = max(maximum, tree_max_ask[right])
        left //= 2
        right //= 2
    return minimum <= take or maximum >= stop


@njit(cache=True, inline="always")
def _first_long_cross(  # pragma: no cover - assertions execute the compiled Numba kernel
    tree_min_bid: np.ndarray,
    tree_max_bid: np.ndarray,
    size: int,
    start: int,
    end: int,
    stop: float,
    take: float,
) -> int:
    if start >= end or not _long_crosses(
        tree_min_bid, tree_max_bid, size, start, end, stop, take
    ):
        return -1
    low = start
    high = end - 1
    while low < high:
        middle = (low + high) // 2
        if _long_crosses(
            tree_min_bid,
            tree_max_bid,
            size,
            start,
            middle + 1,
            stop,
            take,
        ):
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True, inline="always")
def _first_short_cross(  # pragma: no cover - assertions execute the compiled Numba kernel
    tree_min_ask: np.ndarray,
    tree_max_ask: np.ndarray,
    size: int,
    start: int,
    end: int,
    stop: float,
    take: float,
) -> int:
    if start >= end or not _short_crosses(
        tree_min_ask, tree_max_ask, size, start, end, stop, take
    ):
        return -1
    low = start
    high = end - 1
    while low < high:
        middle = (low + high) // 2
        if _short_crosses(
            tree_min_ask,
            tree_max_ask,
            size,
            start,
            middle + 1,
            stop,
            take,
        ):
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True, inline="always")
def _long_net(  # pragma: no cover - executed inside compiled Numba kernels
    entry_ask: float, exit_bid: float, cost_bps: float
) -> float:
    ratio = exit_bid / entry_ask
    return (ratio - 1.0) * 10_000.0 - cost_bps * (1.0 + ratio)


@njit(cache=True, inline="always")
def _short_net(  # pragma: no cover - executed inside compiled Numba kernels
    entry_bid: float, exit_ask: float, cost_bps: float
) -> float:
    ratio = exit_ask / entry_bid
    return (1.0 - ratio) * 10_000.0 - cost_bps * (1.0 + ratio)


@njit(cache=True, parallel=True)
def _evaluate_path_scenario(  # pragma: no cover - assertions execute the compiled Numba kernel
    path_times_ms: np.ndarray,
    min_bid: np.ndarray,
    max_bid: np.ndarray,
    close_bid: np.ndarray,
    min_ask: np.ndarray,
    max_ask: np.ndarray,
    close_ask: np.ndarray,
    tree_size: int,
    tree_min_bid: np.ndarray,
    tree_max_bid: np.ndarray,
    tree_min_ask: np.ndarray,
    tree_max_ask: np.ndarray,
    protected_start_indexes: np.ndarray,
    end_indexes: np.ndarray,
    gap_start_indexes: np.ndarray,
    check_protection_gap: bool,
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    fixed_exit_bid: np.ndarray,
    fixed_exit_ask: np.ndarray,
    fixed_long_exit_time_ms: np.ndarray,
    fixed_short_exit_time_ms: np.ndarray,
    stop_bps: np.ndarray,
    take_bps: np.ndarray,
    cost_bps_per_side: float,
    trigger_slippage_fraction: float,
    adverse_fill: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = len(entry_bid)
    long_net = np.empty(rows, dtype=np.float64)
    short_net = np.empty(rows, dtype=np.float64)
    long_exit_time = fixed_long_exit_time_ms.copy()
    short_exit_time = fixed_short_exit_time_ms.copy()
    long_outcome = np.zeros(rows, dtype=np.int8)
    short_outcome = np.zeros(rows, dtype=np.int8)
    for index in prange(rows):
        long_stop = entry_ask[index] * (1.0 - stop_bps[index] / 10_000.0)
        long_take = entry_ask[index] * (1.0 + take_bps[index] / 10_000.0)
        short_stop = entry_bid[index] * (1.0 + stop_bps[index] / 10_000.0)
        short_take = entry_bid[index] * (1.0 - take_bps[index] / 10_000.0)
        protected_start = int(protected_start_indexes[index])
        end = int(end_indexes[index])
        gap_start = int(gap_start_indexes[index])

        long_cross = -1
        short_cross = -1
        if check_protection_gap and gap_start < protected_start:
            long_cross = _first_long_cross(
                tree_min_bid,
                tree_max_bid,
                tree_size,
                gap_start,
                protected_start,
                long_stop,
                np.inf,
            )
            short_cross = _first_short_cross(
                tree_min_ask,
                tree_max_ask,
                tree_size,
                gap_start,
                protected_start,
                short_stop,
                -np.inf,
            )

        if long_cross >= 0:
            long_exit = min_bid[long_cross] * (1.0 - trigger_slippage_fraction)
            long_exit_time[index] = path_times_ms[long_cross] + 100
            long_outcome[index] = _OUTCOME_PROTECTION_GAP_STOP
        else:
            long_cross = _first_long_cross(
                tree_min_bid,
                tree_max_bid,
                tree_size,
                protected_start,
                end,
                long_stop,
                long_take,
            )
            if long_cross < 0:
                long_exit = fixed_exit_bid[index]
            else:
                hit_stop = min_bid[long_cross] <= long_stop
                hit_take = max_bid[long_cross] >= long_take
                if hit_stop:
                    long_outcome[index] = (
                        _OUTCOME_AMBIGUOUS_STOP if hit_take else _OUTCOME_STOP
                    )
                    trigger_price = long_stop
                else:
                    long_outcome[index] = _OUTCOME_TAKE
                    trigger_price = long_take
                observed_fill = (
                    min_bid[long_cross] if adverse_fill else close_bid[long_cross]
                )
                long_exit = min(trigger_price, observed_fill) * (
                    1.0 - trigger_slippage_fraction
                )
                long_exit_time[index] = path_times_ms[long_cross] + 100

        if short_cross >= 0:
            short_exit = max_ask[short_cross] * (1.0 + trigger_slippage_fraction)
            short_exit_time[index] = path_times_ms[short_cross] + 100
            short_outcome[index] = _OUTCOME_PROTECTION_GAP_STOP
        else:
            short_cross = _first_short_cross(
                tree_min_ask,
                tree_max_ask,
                tree_size,
                protected_start,
                end,
                short_stop,
                short_take,
            )
            if short_cross < 0:
                short_exit = fixed_exit_ask[index]
            else:
                hit_stop = max_ask[short_cross] >= short_stop
                hit_take = min_ask[short_cross] <= short_take
                if hit_stop:
                    short_outcome[index] = (
                        _OUTCOME_AMBIGUOUS_STOP if hit_take else _OUTCOME_STOP
                    )
                    trigger_price = short_stop
                else:
                    short_outcome[index] = _OUTCOME_TAKE
                    trigger_price = short_take
                observed_fill = (
                    max_ask[short_cross] if adverse_fill else close_ask[short_cross]
                )
                short_exit = max(trigger_price, observed_fill) * (
                    1.0 + trigger_slippage_fraction
                )
                short_exit_time[index] = path_times_ms[short_cross] + 100

        long_net[index] = _long_net(entry_ask[index], long_exit, cost_bps_per_side)
        short_net[index] = _short_net(entry_bid[index], short_exit, cost_bps_per_side)
    return (
        long_net,
        short_net,
        long_exit_time,
        short_exit_time,
        long_outcome,
        short_outcome,
    )


def _empty_targets(rows: int) -> tuple[np.ndarray, ...]:
    return (
        np.full(rows, np.nan, dtype=np.float64),
        np.full(rows, np.nan, dtype=np.float64),
        np.full(rows, -1, dtype=np.int64),
        np.full(rows, -1, dtype=np.int64),
        np.full(rows, -1, dtype=np.int8),
        np.full(rows, -1, dtype=np.int8),
    )


def _day_path(
    warehouse: MicrostructureWarehouse,
    symbol: str,
    day_start_ms: int,
) -> dict[str, np.ndarray]:
    cursor = warehouse.connect().execute(
        """
        SELECT bucket_ms, min_bid, max_bid, close_bid,
               min_ask, max_ask, close_ask
        FROM current_book_ticker_100ms
        WHERE symbol = ? AND bucket_ms >= ? AND bucket_ms < ?
        ORDER BY bucket_ms
        """,
        [symbol, day_start_ms - 5_000, day_start_ms + _DAY_MS],
    )
    raw = cursor.fetchnumpy()
    return {
        name: np.asarray(
            raw[name], dtype=np.int64 if name == "bucket_ms" else np.float64
        )
        for name in (
            "bucket_ms",
            "min_bid",
            "max_bid",
            "close_bid",
            "min_ask",
            "max_ask",
            "close_ask",
        )
    }


def _scenario_ranges(
    path_times: np.ndarray,
    decision_times: np.ndarray,
    *,
    total_latency_ms: int,
    protection_delay_ms: int,
    horizon_seconds: int,
    max_quote_age_ms: int,
    check_gap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrival = decision_times + int(total_latency_ms)
    protected_start = arrival + int(protection_delay_ms)
    end_time = arrival + int(horizon_seconds) * 1_000
    gap_start_time = ((arrival + 99) // 100) * 100 if check_gap else protected_start
    protected = np.searchsorted(path_times, protected_start, side="left")
    gap = np.searchsorted(path_times, gap_start_time, side="left")
    # Only complete buckets can affect a target. A bucket that ends after the
    # lifecycle would otherwise create a fill beyond the requested horizon.
    last_complete_start = end_time - 100
    end = np.searchsorted(path_times, last_complete_start, side="right")
    asof = np.searchsorted(path_times, protected_start, side="right") - 1
    valid = (
        (asof >= 0)
        & (protected < end)
        & (end <= len(path_times))
        & ((protected_start - path_times[np.maximum(asof, 0)]) <= max_quote_age_ms)
        & (end_time <= ((decision_times // _DAY_MS) + 1) * _DAY_MS)
    )
    return protected, end, gap, valid


def build_adaptive_barrier_targets(
    warehouse: MicrostructureWarehouse,
    dataset: MicrostructureDataset,
    source_indexes: np.ndarray,
    spec: AdaptiveBarrierSpec,
    *,
    progress: Callable[[int, int, int], None] | None = None,
) -> AdaptiveBarrierTargets:
    """Build base and adverse-stress targets without loading the full path at once."""

    validate_microstructure_dataset(dataset)
    indexes = np.asarray(source_indexes, dtype=np.int64)
    stop, take = volatility_scaled_barriers(dataset, indexes, spec)
    rows = len(indexes)
    base = list(_empty_targets(rows))
    stress = list(_empty_targets(rows))
    valid_output = np.zeros(rows, dtype=bool)
    decision_times = np.asarray(dataset.decision_time_ms[indexes], dtype=np.int64)
    day_ids = decision_times // _DAY_MS
    unique_days = np.unique(day_ids)
    cost = float(dataset.taker_fee_bps + dataset.additional_slippage_bps_per_side)
    slip_fraction = float(spec.trigger_execution_slippage_bps) / 10_000.0

    for day_offset, day_id in enumerate(unique_days, start=1):
        positions = np.flatnonzero(day_ids == day_id)
        path = _day_path(warehouse, dataset.symbol, int(day_id) * _DAY_MS)
        path_times = path["bucket_ms"]
        if path_times.size == 0 or np.any(np.diff(path_times) <= 0):
            raise ValueError(
                f"adaptive barrier 100 ms path is invalid for day {day_id}"
            )
        tree = _extreme_trees(
            path["min_bid"],
            path["max_bid"],
            path["min_ask"],
            path["max_ask"],
        )
        base_ranges = _scenario_ranges(
            path_times,
            decision_times[positions],
            total_latency_ms=dataset.total_latency_ms,
            protection_delay_ms=spec.base_protection_delay_ms,
            horizon_seconds=spec.horizon_seconds,
            max_quote_age_ms=dataset.max_quote_age_ms,
            check_gap=False,
        )
        stress_ranges = _scenario_ranges(
            path_times,
            decision_times[positions],
            total_latency_ms=dataset.total_latency_ms,
            protection_delay_ms=spec.stress_protection_delay_ms,
            horizon_seconds=spec.horizon_seconds,
            max_quote_age_ms=dataset.max_quote_age_ms,
            check_gap=True,
        )
        day_valid = base_ranges[3] & stress_ranges[3]
        valid_positions = positions[day_valid]
        if valid_positions.size:
            source = indexes[valid_positions]
            local = np.flatnonzero(day_valid)
            common = {
                "path_times_ms": path_times,
                "min_bid": path["min_bid"],
                "max_bid": path["max_bid"],
                "close_bid": path["close_bid"],
                "min_ask": path["min_ask"],
                "max_ask": path["max_ask"],
                "close_ask": path["close_ask"],
                "tree_size": tree[0],
                "tree_min_bid": tree[1],
                "tree_max_bid": tree[2],
                "tree_min_ask": tree[3],
                "tree_max_ask": tree[4],
                "entry_bid": np.asarray(
                    dataset.entry_bid_price[source], dtype=np.float64
                ),
                "entry_ask": np.asarray(
                    dataset.entry_ask_price[source], dtype=np.float64
                ),
                "fixed_exit_bid": np.asarray(
                    dataset.fixed_exit_bid_price[source], dtype=np.float64
                ),
                "fixed_exit_ask": np.asarray(
                    dataset.fixed_exit_ask_price[source], dtype=np.float64
                ),
                "fixed_long_exit_time_ms": np.asarray(
                    dataset.long_exit_time_ms[source], dtype=np.int64
                ),
                "fixed_short_exit_time_ms": np.asarray(
                    dataset.short_exit_time_ms[source], dtype=np.int64
                ),
                "stop_bps": stop[valid_positions],
                "take_bps": take[valid_positions],
                "cost_bps_per_side": cost,
                "trigger_slippage_fraction": slip_fraction,
            }
            base_result = _evaluate_path_scenario(
                protected_start_indexes=base_ranges[0][local],
                end_indexes=base_ranges[1][local],
                gap_start_indexes=base_ranges[2][local],
                check_protection_gap=False,
                adverse_fill=False,
                **common,
            )
            stress_result = _evaluate_path_scenario(
                protected_start_indexes=stress_ranges[0][local],
                end_indexes=stress_ranges[1][local],
                gap_start_indexes=stress_ranges[2][local],
                check_protection_gap=True,
                adverse_fill=True,
                **common,
            )
            for target, values in zip(base, base_result, strict=True):
                target[valid_positions] = values
            for target, values in zip(stress, stress_result, strict=True):
                target[valid_positions] = values
            valid_output[valid_positions] = True
        if progress is not None:
            progress(day_offset, len(unique_days), int(np.sum(valid_output)))

    result = AdaptiveBarrierTargets(
        schema_version=ADAPTIVE_BARRIER_SCHEMA_VERSION,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        spec=spec,
        source_indexes=indexes.copy(),
        valid=valid_output,
        stop_barrier_bps=stop,
        take_barrier_bps=take,
        base_long_net_bps=base[0],
        base_short_net_bps=base[1],
        base_long_exit_time_ms=base[2],
        base_short_exit_time_ms=base[3],
        base_long_outcome=base[4],
        base_short_outcome=base[5],
        stress_long_net_bps=stress[0],
        stress_short_net_bps=stress[1],
        stress_long_exit_time_ms=stress[2],
        stress_short_exit_time_ms=stress[3],
        stress_long_outcome=stress[4],
        stress_short_outcome=stress[5],
    )
    validate_adaptive_barrier_targets(dataset, result)
    return result


def validate_adaptive_barrier_targets(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
) -> None:
    validate_microstructure_dataset(dataset)
    if (
        targets.schema_version != ADAPTIVE_BARRIER_SCHEMA_VERSION
        or targets.target_mode != ADAPTIVE_BARRIER_TARGET_MODE
        or targets.trading_authority
        or targets.execution_claim
        or targets.profitability_claim
        or targets.portfolio_claim
        or targets.leverage_applied
    ):
        raise ValueError("adaptive barrier target authority contract is invalid")
    rows = targets.rows
    arrays = (
        targets.valid,
        targets.stop_barrier_bps,
        targets.take_barrier_bps,
        targets.base_long_net_bps,
        targets.base_short_net_bps,
        targets.base_long_exit_time_ms,
        targets.base_short_exit_time_ms,
        targets.base_long_outcome,
        targets.base_short_outcome,
        targets.stress_long_net_bps,
        targets.stress_short_net_bps,
        targets.stress_long_exit_time_ms,
        targets.stress_short_exit_time_ms,
        targets.stress_long_outcome,
        targets.stress_short_outcome,
    )
    if rows <= 0 or any(len(values) != rows for values in arrays):
        raise ValueError("adaptive barrier target arrays are inconsistent")
    indexes = np.asarray(targets.source_indexes, dtype=np.int64)
    valid = np.asarray(targets.valid, dtype=bool)
    if (
        indexes[0] < 0
        or indexes[-1] >= dataset.rows
        or np.any(np.diff(indexes) <= 0)
        or not np.any(valid)
        or np.any(~np.isfinite(targets.stop_barrier_bps))
        or np.any(~np.isfinite(targets.take_barrier_bps))
        or np.any(targets.take_barrier_bps <= targets.stop_barrier_bps)
    ):
        raise ValueError("adaptive barrier indexes or barriers are invalid")
    for values in (
        targets.base_long_net_bps,
        targets.base_short_net_bps,
        targets.stress_long_net_bps,
        targets.stress_short_net_bps,
    ):
        if np.any(~np.isfinite(values[valid])) or np.any(np.isfinite(values[~valid])):
            raise ValueError("adaptive barrier target validity mask differs from PnL")
    earliest = dataset.decision_time_ms[indexes[valid]] + dataset.total_latency_ms
    latest = earliest + targets.spec.horizon_seconds * 1_000
    for values in (
        targets.base_long_exit_time_ms,
        targets.base_short_exit_time_ms,
        targets.stress_long_exit_time_ms,
        targets.stress_short_exit_time_ms,
    ):
        if np.any(values[valid] < earliest) or np.any(values[valid] > latest):
            raise ValueError("adaptive barrier exit time is outside the lifecycle")
        if np.any(values[~valid] != -1):
            raise ValueError("adaptive barrier invalid rows carry exit times")
    for values in (
        targets.base_long_outcome,
        targets.base_short_outcome,
        targets.stress_long_outcome,
        targets.stress_short_outcome,
    ):
        if np.any(~np.isin(values[valid], tuple(_OUTCOME_NAMES))) or np.any(
            values[~valid] != -1
        ):
            raise ValueError("adaptive barrier outcomes are invalid")


__all__ = [
    "ADAPTIVE_BARRIER_SCHEMA_VERSION",
    "ADAPTIVE_BARRIER_TARGET_MODE",
    "AdaptiveBarrierSpec",
    "AdaptiveBarrierTargets",
    "build_adaptive_barrier_targets",
    "validate_adaptive_barrier_targets",
    "volatility_scaled_barriers",
]
