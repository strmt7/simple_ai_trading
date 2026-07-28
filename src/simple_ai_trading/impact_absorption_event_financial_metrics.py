"""Shared realized-P&L ordering for Round 74 financial evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ROUND74_REALIZED_METRICS_SCHEMA_VERSION = "round-074-realized-payoff-metrics-v2"


def round74_realization_order_indices(
    *,
    run_ids: Sequence[str],
    exit_monotonic_ns: Sequence[int],
    expected_run_ids: Sequence[str],
) -> np.ndarray:
    """Order outcomes by cohort run and actual realization time."""

    selected_runs = tuple(str(value) for value in run_ids)
    expected = tuple(str(value) for value in expected_run_ids)
    exits = tuple(exit_monotonic_ns)
    if (
        len(selected_runs) != len(exits)
        or not expected
        or len(set(expected)) != len(expected)
        or any(value not in expected for value in selected_runs)
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in exits
        )
    ):
        raise ValueError("Round 74 realized-payoff ordering differs")
    run_rank = {run_id: index for index, run_id in enumerate(expected)}
    order = np.asarray(
        sorted(
            range(len(selected_runs)),
            key=lambda index: (
                run_rank[selected_runs[index]],
                int(exits[index]),
                index,
            ),
        ),
        dtype=np.int64,
    )
    order.setflags(write=False)
    return order


def round74_maximum_realized_drawdown_bps(
    net_payoff_bps: Sequence[float] | np.ndarray,
    *,
    run_ids: Sequence[str],
    exit_monotonic_ns: Sequence[int],
    expected_run_ids: Sequence[str],
) -> float:
    """Calculate additive research drawdown in actual P&L realization order."""

    values = np.asarray(net_payoff_bps, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(run_ids) or not np.isfinite(values).all():
        raise ValueError("Round 74 realized-payoff values differ")
    order = round74_realization_order_indices(
        run_ids=run_ids,
        exit_monotonic_ns=exit_monotonic_ns,
        expected_run_ids=expected_run_ids,
    )
    if values.size == 0:
        return 0.0
    cumulative = np.cumsum(values[order])
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdowns = peaks[1:] - cumulative
    result = float(drawdowns.max())
    if not np.isfinite(result) or result < 0.0:
        raise RuntimeError("Round 74 realized-payoff drawdown differs")
    return result


def round74_maximum_concurrent_adverse_excursion_bps(
    maximum_adverse_excursion_bps: Sequence[float] | np.ndarray,
    *,
    run_ids: Sequence[str],
    entry_monotonic_ns: Sequence[int],
    exit_monotonic_ns: Sequence[int],
    expected_run_ids: Sequence[str],
) -> float:
    """Bound open-position loss by summing each concurrent trade's own MAE."""

    values = np.asarray(maximum_adverse_excursion_bps, dtype=np.float64)
    selected_runs = tuple(str(value) for value in run_ids)
    entries = tuple(entry_monotonic_ns)
    exits = tuple(exit_monotonic_ns)
    expected = tuple(str(value) for value in expected_run_ids)
    if (
        values.ndim != 1
        or len(values) != len(selected_runs)
        or len(entries) != len(selected_runs)
        or len(exits) != len(selected_runs)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not expected
        or len(set(expected)) != len(expected)
        or any(run_id not in expected for run_id in selected_runs)
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in (*entries, *exits)
        )
        or any(
            int(exit_value) < int(entry) for entry, exit_value in zip(entries, exits)
        )
    ):
        raise ValueError("Round 74 concurrent adverse-excursion values differ")
    run_rank = {run_id: index for index, run_id in enumerate(expected)}
    events: list[tuple[int, int, int, float]] = []
    for run_id, entry, exit_value, adverse_excursion in zip(
        selected_runs, entries, exits, values, strict=True
    ):
        if int(exit_value) == int(entry):
            continue
        rank = run_rank[run_id]
        # Intervals are [entry, exit): release capital before admitting a new
        # position at the same timestamp.
        events.append((rank, int(entry), 1, float(adverse_excursion)))
        events.append((rank, int(exit_value), 0, -float(adverse_excursion)))
    current_run = -1
    active = 0.0
    maximum = 0.0
    for rank, _timestamp, _event_order, delta in sorted(events):
        if rank != current_run:
            current_run = rank
            active = 0.0
        active += delta
        tolerance = max(1e-12, maximum * 1e-12)
        if active < -tolerance:
            raise RuntimeError("Round 74 concurrent adverse-excursion sweep differs")
        active = max(0.0, active)
        maximum = max(maximum, active)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise RuntimeError("Round 74 concurrent adverse-excursion bound differs")
    return float(maximum)


def round74_conservative_maximum_drawdown_bps(
    net_payoff_bps: Sequence[float] | np.ndarray,
    maximum_adverse_excursion_bps: Sequence[float] | np.ndarray,
    *,
    run_ids: Sequence[str],
    entry_monotonic_ns: Sequence[int],
    exit_monotonic_ns: Sequence[int],
    expected_run_ids: Sequence[str],
) -> float:
    """Carry realized losses forward while bounding concurrent open-position risk."""

    payoffs = np.asarray(net_payoff_bps, dtype=np.float64)
    adverse = np.asarray(maximum_adverse_excursion_bps, dtype=np.float64)
    selected_runs = tuple(str(value) for value in run_ids)
    entries = tuple(entry_monotonic_ns)
    exits = tuple(exit_monotonic_ns)
    expected = tuple(str(value) for value in expected_run_ids)
    if (
        payoffs.ndim != 1
        or adverse.ndim != 1
        or len(payoffs) != len(adverse)
        or len(payoffs) != len(selected_runs)
        or len(entries) != len(selected_runs)
        or len(exits) != len(selected_runs)
        or not np.isfinite(payoffs).all()
        or not np.isfinite(adverse).all()
        or np.any(adverse < 0.0)
        or not expected
        or len(set(expected)) != len(expected)
        or any(run_id not in expected for run_id in selected_runs)
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in (*entries, *exits)
        )
        or any(
            int(exit_value) < int(entry) for entry, exit_value in zip(entries, exits)
        )
    ):
        raise ValueError("Round 74 conservative drawdown values differ")
    run_rank = {run_id: index for index, run_id in enumerate(expected)}
    events: list[tuple[int, int, int, int]] = []
    for index, (run_id, entry, exit_value) in enumerate(
        zip(selected_runs, entries, exits, strict=True)
    ):
        rank = run_rank[run_id]
        if int(exit_value) > int(entry):
            events.append((rank, int(entry), 1, index))
        events.append((rank, int(exit_value), 0, index))
    current_run = -1
    realized = 0.0
    realized_peak = 0.0
    active_adverse = 0.0
    maximum = 0.0
    for rank, _timestamp, event_kind, index in sorted(events):
        if rank != current_run:
            current_run = rank
            active_adverse = 0.0
        if event_kind == 0:
            if int(exits[index]) > int(entries[index]):
                active_adverse -= float(adverse[index])
            realized += float(payoffs[index])
            realized_peak = max(realized_peak, realized)
        else:
            active_adverse += float(adverse[index])
        tolerance = max(1e-12, maximum * 1e-12)
        if active_adverse < -tolerance:
            raise RuntimeError("Round 74 conservative drawdown sweep differs")
        active_adverse = max(0.0, active_adverse)
        maximum = max(
            maximum,
            realized_peak - realized + active_adverse,
        )
    if not np.isfinite(maximum) or maximum < 0.0:
        raise RuntimeError("Round 74 conservative drawdown bound differs")
    return float(maximum)


__all__ = [
    "ROUND74_REALIZED_METRICS_SCHEMA_VERSION",
    "round74_conservative_maximum_drawdown_bps",
    "round74_maximum_concurrent_adverse_excursion_bps",
    "round74_maximum_realized_drawdown_bps",
    "round74_realization_order_indices",
]
