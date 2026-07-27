"""Shared realized-P&L ordering for Round 74 financial evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ROUND74_REALIZED_METRICS_SCHEMA_VERSION = "round-074-realized-payoff-metrics-v1"


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


__all__ = [
    "ROUND74_REALIZED_METRICS_SCHEMA_VERSION",
    "round74_maximum_realized_drawdown_bps",
    "round74_realization_order_indices",
]
