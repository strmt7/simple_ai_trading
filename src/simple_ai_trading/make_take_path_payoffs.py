"""Causal 100 ms action-path payoffs for Round 57 make/take research."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from numba import njit, prange
import numpy as np

from .microstructure_barriers import (
    _extreme_trees,
    _first_long_cross,
    _first_short_cross,
)


ACTION_PATH_PAYOFF_SCHEMA_VERSION = "queue-censored-action-path-payoff-v1"
ACTION_PATH_HORIZON_SECONDS = 300
ACTION_PATH_MAX_QUOTE_AGE_MS = 1_000
ACTION_PATH_RESOLUTION_MS = 100
ACTION_PATH_SCENARIOS = ("base", "stress")
ACTION_PATH_OUTCOME_LABELS = (
    "horizon",
    "stop",
    "take",
    "ambiguous_stop",
    "protection_gap_stop",
)
_OUTCOME_HORIZON = 0
_OUTCOME_STOP = 1
_OUTCOME_TAKE = 2
_OUTCOME_AMBIGUOUS_STOP = 3
_OUTCOME_PROTECTION_GAP_STOP = 4
_DAY_MS = 86_400_000


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _int_array(
    value: Sequence[int] | np.ndarray,
    *,
    name: str,
    rows: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or (rows is not None and raw.size != rows)
        or raw.dtype.kind not in "iu"
        or (raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max))
    ):
        raise ValueError(f"{name} is invalid")
    output = np.array(raw, dtype=np.int64, order="C", copy=True)
    if np.any(output < 0):
        raise ValueError(f"{name} is invalid")
    return output


def _float_array(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    rows: int,
    positive: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != rows or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} is invalid")
    output = np.array(raw, dtype=np.float64, order="C", copy=True)
    if (
        not np.isfinite(output).all()
        or (positive and np.any(output <= 0.0))
    ):
        raise ValueError(f"{name} is invalid")
    return output


@dataclass(frozen=True)
class ActionPathPayoffBatch:
    schema_version: str
    scenario: str
    horizon_seconds: int
    protection_delay_ms: int
    max_quote_age_ms: int
    path_resolution_ms: int
    check_protection_gap: bool
    adverse_path_fill: bool
    source_path_sha256: str
    valid: np.ndarray
    net_bps: np.ndarray
    exit_time_ms: np.ndarray
    outcome: np.ndarray
    markout_5s_bps: np.ndarray
    markout_15s_bps: np.ndarray
    batch_sha256: str

    @property
    def rows(self) -> int:
        return int(self.valid.size)

    def summary(self) -> dict[str, object]:
        valid = self.valid
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "rows": self.rows,
            "valid_rows": int(np.count_nonzero(valid)),
            "outcome_counts": {
                label: int(np.count_nonzero(self.outcome[valid] == code))
                for code, label in enumerate(ACTION_PATH_OUTCOME_LABELS)
            },
            "net_bps_mean": float(np.mean(self.net_bps[valid]))
            if np.any(valid)
            else None,
            "source_path_sha256": self.source_path_sha256,
            "batch_sha256": self.batch_sha256,
            "trading_authority": False,
            "profitability_claim": False,
        }


@njit(cache=True, inline="always")
def _net_bps(
    side: int,
    entry_price: float,
    exit_price: float,
    entry_cost_bps: float,
    exit_cost_bps: float,
) -> float:
    ratio = exit_price / entry_price
    gross = (ratio - 1.0) * 10_000.0 if side == 1 else (1.0 - ratio) * 10_000.0
    return gross - entry_cost_bps - exit_cost_bps * ratio


@njit(cache=True, parallel=True)
def _evaluate_paths_kernel(
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
    valid: np.ndarray,
    protected_start_indexes: np.ndarray,
    end_indexes: np.ndarray,
    gap_start_indexes: np.ndarray,
    fixed_exit_indexes: np.ndarray,
    lifecycle_end_ms: np.ndarray,
    action_side: np.ndarray,
    entry_price: np.ndarray,
    entry_cost_bps: np.ndarray,
    exit_cost_bps: np.ndarray,
    stop_bps: np.ndarray,
    take_bps: np.ndarray,
    check_protection_gap: bool,
    adverse_path_fill: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = valid.size
    net = np.full(rows, np.nan, dtype=np.float64)
    exit_time = np.full(rows, -1, dtype=np.int64)
    outcome = np.full(rows, -1, dtype=np.int8)
    for row in prange(rows):
        if not valid[row]:
            continue
        side = int(action_side[row])
        entry = entry_price[row]
        stop = (
            entry * (1.0 - stop_bps[row] / 10_000.0)
            if side == 1
            else entry * (1.0 + stop_bps[row] / 10_000.0)
        )
        take = (
            entry * (1.0 + take_bps[row] / 10_000.0)
            if side == 1
            else entry * (1.0 - take_bps[row] / 10_000.0)
        )
        protected_start = int(protected_start_indexes[row])
        end = int(end_indexes[row])
        gap_start = int(gap_start_indexes[row])
        cross = -1
        if check_protection_gap and gap_start < protected_start:
            cross = (
                _first_long_cross(
                    tree_min_bid,
                    tree_max_bid,
                    tree_size,
                    gap_start,
                    protected_start,
                    stop,
                    np.inf,
                )
                if side == 1
                else _first_short_cross(
                    tree_min_ask,
                    tree_max_ask,
                    tree_size,
                    gap_start,
                    protected_start,
                    stop,
                    -np.inf,
                )
            )
        if cross >= 0:
            exit_price = min_bid[cross] if side == 1 else max_ask[cross]
            exit_time[row] = path_times_ms[cross] + ACTION_PATH_RESOLUTION_MS
            outcome[row] = _OUTCOME_PROTECTION_GAP_STOP
        else:
            cross = (
                _first_long_cross(
                    tree_min_bid,
                    tree_max_bid,
                    tree_size,
                    protected_start,
                    end,
                    stop,
                    take,
                )
                if side == 1
                else _first_short_cross(
                    tree_min_ask,
                    tree_max_ask,
                    tree_size,
                    protected_start,
                    end,
                    stop,
                    take,
                )
            )
            if cross < 0:
                fixed = fixed_exit_indexes[row]
                exit_price = close_bid[fixed] if side == 1 else close_ask[fixed]
                exit_time[row] = lifecycle_end_ms[row]
                outcome[row] = _OUTCOME_HORIZON
            else:
                if side == 1:
                    hit_stop = min_bid[cross] <= stop
                    hit_take = max_bid[cross] >= take
                    trigger = stop if hit_stop else take
                    observed = min_bid[cross] if adverse_path_fill else close_bid[cross]
                    exit_price = min(trigger, observed)
                else:
                    hit_stop = max_ask[cross] >= stop
                    hit_take = min_ask[cross] <= take
                    trigger = stop if hit_stop else take
                    observed = max_ask[cross] if adverse_path_fill else close_ask[cross]
                    exit_price = max(trigger, observed)
                outcome[row] = (
                    _OUTCOME_AMBIGUOUS_STOP
                    if hit_stop and hit_take
                    else (_OUTCOME_STOP if hit_stop else _OUTCOME_TAKE)
                )
                exit_time[row] = path_times_ms[cross] + ACTION_PATH_RESOLUTION_MS
        net[row] = _net_bps(
            side,
            entry,
            exit_price,
            entry_cost_bps[row],
            exit_cost_bps[row],
        )
    return net, exit_time, outcome


def _completed_quote_indexes(
    path_times_ms: np.ndarray,
    observation_time_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.searchsorted(
        path_times_ms,
        observation_time_ms - ACTION_PATH_RESOLUTION_MS,
        side="right",
    ) - 1
    safe = np.maximum(indexes, 0)
    age = observation_time_ms - (
        path_times_ms[safe] + ACTION_PATH_RESOLUTION_MS
    )
    valid = (indexes >= 0) & (age >= 0) & (age <= ACTION_PATH_MAX_QUOTE_AGE_MS)
    return indexes, valid


def _markout(
    *,
    path_times_ms: np.ndarray,
    close_bid: np.ndarray,
    close_ask: np.ndarray,
    entry_time_ms: np.ndarray,
    entry_price: np.ndarray,
    action_side: np.ndarray,
    seconds: int,
    path_valid: np.ndarray,
) -> np.ndarray:
    observation = entry_time_ms + int(seconds) * 1_000
    indexes, quote_valid = _completed_quote_indexes(path_times_ms, observation)
    valid = path_valid & quote_valid & (
        observation <= ((entry_time_ms // _DAY_MS) + 1) * _DAY_MS
    )
    output = np.full(entry_time_ms.size, np.nan, dtype=np.float64)
    if np.any(valid):
        selected = np.flatnonzero(valid)
        quote_indexes = indexes[selected]
        ratio = np.where(
            action_side[selected] == 1,
            close_bid[quote_indexes] / entry_price[selected],
            close_ask[quote_indexes] / entry_price[selected],
        )
        output[selected] = np.where(
            action_side[selected] == 1,
            (ratio - 1.0) * 10_000.0,
            (1.0 - ratio) * 10_000.0,
        )
    return output


def build_action_path_payoffs(
    *,
    scenario: str,
    path_time_ms: Sequence[int] | np.ndarray,
    path_min_bid: Sequence[float] | np.ndarray,
    path_max_bid: Sequence[float] | np.ndarray,
    path_close_bid: Sequence[float] | np.ndarray,
    path_min_ask: Sequence[float] | np.ndarray,
    path_max_ask: Sequence[float] | np.ndarray,
    path_close_ask: Sequence[float] | np.ndarray,
    entry_time_ms: Sequence[int] | np.ndarray,
    action_side: Sequence[int] | np.ndarray,
    entry_price: Sequence[float] | np.ndarray,
    entry_cost_bps: Sequence[float] | np.ndarray,
    exit_cost_bps: Sequence[float] | np.ndarray,
    stop_bps: Sequence[float] | np.ndarray,
    take_bps: Sequence[float] | np.ndarray,
) -> ActionPathPayoffBatch:
    """Evaluate one day of executed actions under the frozen base or stress path."""

    selected_scenario = str(scenario)
    if selected_scenario not in ACTION_PATH_SCENARIOS:
        raise ValueError("action-path scenario is unsupported")
    protection_delay_ms = 750 if selected_scenario == "base" else 1_500
    check_gap = selected_scenario == "stress"
    adverse_fill = selected_scenario == "stress"
    path_times = _int_array(path_time_ms, name="action-path times")
    path_rows = path_times.size
    if (
        path_rows == 0
        or np.any(np.diff(path_times) <= 0)
        or np.any(path_times % ACTION_PATH_RESOLUTION_MS != 0)
    ):
        raise ValueError("action-path timestamps are invalid")
    min_bid = _float_array(path_min_bid, name="action-path minimum bids", rows=path_rows, positive=True)
    max_bid = _float_array(path_max_bid, name="action-path maximum bids", rows=path_rows, positive=True)
    close_bid = _float_array(path_close_bid, name="action-path close bids", rows=path_rows, positive=True)
    min_ask = _float_array(path_min_ask, name="action-path minimum asks", rows=path_rows, positive=True)
    max_ask = _float_array(path_max_ask, name="action-path maximum asks", rows=path_rows, positive=True)
    close_ask = _float_array(path_close_ask, name="action-path close asks", rows=path_rows, positive=True)
    if (
        np.any(min_bid > close_bid)
        or np.any(close_bid > max_bid)
        or np.any(min_ask > close_ask)
        or np.any(close_ask > max_ask)
        or np.any(close_bid >= close_ask)
    ):
        raise ValueError("action-path quote geometry is invalid")
    entries = _int_array(entry_time_ms, name="action entry times")
    rows = entries.size
    if (
        rows == 0
        or np.any(np.diff(entries) < 0)
        or np.any(
            entries
            > np.iinfo(np.int64).max - ACTION_PATH_HORIZON_SECONDS * 1_000
        )
    ):
        raise ValueError("action entry times are invalid")
    sides_raw = np.asarray(action_side)
    if sides_raw.ndim != 1 or sides_raw.size != rows or sides_raw.dtype.kind not in "iu":
        raise ValueError("action sides are invalid")
    if np.any((sides_raw != 1) & (sides_raw != -1)):
        raise ValueError("action sides are invalid")
    sides = np.array(sides_raw, dtype=np.int8, order="C", copy=True)
    prices = _float_array(entry_price, name="action entry prices", rows=rows, positive=True)
    entry_cost = _float_array(entry_cost_bps, name="action entry costs", rows=rows, positive=False)
    exit_cost = _float_array(exit_cost_bps, name="action exit costs", rows=rows, positive=False)
    stops = _float_array(stop_bps, name="action stop distances", rows=rows, positive=True)
    takes = _float_array(take_bps, name="action take distances", rows=rows, positive=True)
    if (
        np.any(entry_cost < -10.0)
        or np.any(exit_cost < 0.0)
        or np.any(takes <= stops)
        or np.any(stops < 18.0)
        or np.any(stops > 80.0)
        or np.any(takes < 30.0)
        or np.any(takes > 120.0)
    ):
        raise ValueError("action path risk or cost contract is invalid")

    lifecycle_end = entries + ACTION_PATH_HORIZON_SECONDS * 1_000
    protected_time = entries + protection_delay_ms
    protected = np.searchsorted(path_times, protected_time, side="left")
    gap_time = (
        (entries + ACTION_PATH_RESOLUTION_MS - 1) // ACTION_PATH_RESOLUTION_MS
    ) * ACTION_PATH_RESOLUTION_MS
    gap = np.searchsorted(path_times, gap_time, side="left")
    last_complete_start = lifecycle_end - ACTION_PATH_RESOLUTION_MS
    end = np.searchsorted(path_times, last_complete_start, side="right")
    fixed_exit = end - 1
    _, protected_quote_valid = _completed_quote_indexes(path_times, protected_time)
    _, exit_quote_valid = _completed_quote_indexes(path_times, lifecycle_end)
    _, markout_5s_quote_valid = _completed_quote_indexes(path_times, entries + 5_000)
    _, markout_15s_quote_valid = _completed_quote_indexes(path_times, entries + 15_000)
    valid = (
        (protected < end)
        & (end <= path_rows)
        & (fixed_exit >= 0)
        & protected_quote_valid
        & exit_quote_valid
        & markout_5s_quote_valid
        & markout_15s_quote_valid
        & (lifecycle_end <= ((entries // _DAY_MS) + 1) * _DAY_MS)
    )
    tree = _extreme_trees(min_bid, max_bid, min_ask, max_ask)
    net, exit_time, outcome = _evaluate_paths_kernel(
        path_times,
        min_bid,
        max_bid,
        close_bid,
        min_ask,
        max_ask,
        close_ask,
        tree[0],
        tree[1],
        tree[2],
        tree[3],
        tree[4],
        valid,
        protected,
        end,
        gap,
        fixed_exit,
        lifecycle_end,
        sides,
        prices,
        entry_cost,
        exit_cost,
        stops,
        takes,
        check_gap,
        adverse_fill,
    )
    markout_5s = _markout(
        path_times_ms=path_times,
        close_bid=close_bid,
        close_ask=close_ask,
        entry_time_ms=entries,
        entry_price=prices,
        action_side=sides,
        seconds=5,
        path_valid=valid,
    )
    markout_15s = _markout(
        path_times_ms=path_times,
        close_bid=close_bid,
        close_ask=close_ask,
        entry_time_ms=entries,
        entry_price=prices,
        action_side=sides,
        seconds=15,
        path_valid=valid,
    )
    source_path_sha256 = _sha256(
        {
            "arrays": {
                "time": _array_sha256(path_times),
                "min_bid": _array_sha256(min_bid),
                "max_bid": _array_sha256(max_bid),
                "close_bid": _array_sha256(close_bid),
                "min_ask": _array_sha256(min_ask),
                "max_ask": _array_sha256(max_ask),
                "close_ask": _array_sha256(close_ask),
            }
        }
    )
    payload = {
        "schema_version": ACTION_PATH_PAYOFF_SCHEMA_VERSION,
        "scenario": selected_scenario,
        "source_path_sha256": source_path_sha256,
        "horizon_seconds": ACTION_PATH_HORIZON_SECONDS,
        "protection_delay_ms": protection_delay_ms,
        "max_quote_age_ms": ACTION_PATH_MAX_QUOTE_AGE_MS,
        "path_resolution_ms": ACTION_PATH_RESOLUTION_MS,
        "check_protection_gap": check_gap,
        "adverse_path_fill": adverse_fill,
        "inputs": {
            "entry_time_ms": _array_sha256(entries),
            "action_side": _array_sha256(sides),
            "entry_price": _array_sha256(prices),
            "entry_cost_bps": _array_sha256(entry_cost),
            "exit_cost_bps": _array_sha256(exit_cost),
            "stop_bps": _array_sha256(stops),
            "take_bps": _array_sha256(takes),
        },
        "outputs": {
            "valid": _array_sha256(valid),
            "net_bps": _array_sha256(net),
            "exit_time_ms": _array_sha256(exit_time),
            "outcome": _array_sha256(outcome),
            "markout_5s_bps": _array_sha256(markout_5s),
            "markout_15s_bps": _array_sha256(markout_15s),
        },
    }
    batch_sha256 = _sha256(payload)
    for array in (valid, net, exit_time, outcome, markout_5s, markout_15s):
        array.setflags(write=False)
    return ActionPathPayoffBatch(
        schema_version=ACTION_PATH_PAYOFF_SCHEMA_VERSION,
        scenario=selected_scenario,
        horizon_seconds=ACTION_PATH_HORIZON_SECONDS,
        protection_delay_ms=protection_delay_ms,
        max_quote_age_ms=ACTION_PATH_MAX_QUOTE_AGE_MS,
        path_resolution_ms=ACTION_PATH_RESOLUTION_MS,
        check_protection_gap=check_gap,
        adverse_path_fill=adverse_fill,
        source_path_sha256=source_path_sha256,
        valid=valid,
        net_bps=net,
        exit_time_ms=exit_time,
        outcome=outcome,
        markout_5s_bps=markout_5s,
        markout_15s_bps=markout_15s,
        batch_sha256=batch_sha256,
    )


__all__ = [
    "ACTION_PATH_HORIZON_SECONDS",
    "ACTION_PATH_MAX_QUOTE_AGE_MS",
    "ACTION_PATH_OUTCOME_LABELS",
    "ACTION_PATH_PAYOFF_SCHEMA_VERSION",
    "ACTION_PATH_RESOLUTION_MS",
    "ACTION_PATH_SCENARIOS",
    "ActionPathPayoffBatch",
    "build_action_path_payoffs",
]
