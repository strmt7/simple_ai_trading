from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.impact_absorption_event_financial_metrics import (
    round74_conservative_maximum_drawdown_bps,
    round74_maximum_concurrent_adverse_excursion_bps,
    round74_maximum_realized_drawdown_bps,
    round74_realization_order_indices,
)


RUNS = ("1" * 32, "2" * 32)


def test_drawdown_uses_actual_exit_order_not_signal_order() -> None:
    payoffs = np.asarray((10.0, -5.0, -5.0), dtype=np.float64)
    order = round74_realization_order_indices(
        run_ids=(RUNS[0], RUNS[0], RUNS[0]),
        exit_monotonic_ns=(20, 10, 30),
        expected_run_ids=RUNS,
    )

    assert tuple(order) == (1, 0, 2)
    assert round74_maximum_realized_drawdown_bps(
        payoffs,
        run_ids=(RUNS[0], RUNS[0], RUNS[0]),
        exit_monotonic_ns=(20, 10, 30),
        expected_run_ids=RUNS,
    ) == pytest.approx(5.0)
    signal_order_cumulative = np.cumsum(payoffs)
    signal_order_peaks = np.maximum.accumulate(
        np.concatenate(([0.0], signal_order_cumulative))
    )
    assert float((signal_order_peaks[1:] - signal_order_cumulative).max()) == 10.0


def test_concurrent_adverse_excursion_uses_half_open_position_intervals() -> None:
    assert round74_maximum_concurrent_adverse_excursion_bps(
        (7.0, 11.0, 5.0, 100.0),
        run_ids=(RUNS[0], RUNS[0], RUNS[0], RUNS[1]),
        entry_monotonic_ns=(10, 20, 15, 30),
        exit_monotonic_ns=(20, 25, 22, 30),
        expected_run_ids=RUNS,
    ) == pytest.approx(16.0)


def test_concurrent_adverse_excursion_rejects_invalid_intervals() -> None:
    with pytest.raises(ValueError):
        round74_maximum_concurrent_adverse_excursion_bps(
            (1.0,),
            run_ids=(RUNS[0],),
            entry_monotonic_ns=(20,),
            exit_monotonic_ns=(10,),
            expected_run_ids=RUNS,
        )


def test_conservative_drawdown_carries_realized_loss_into_new_open_risk() -> None:
    assert round74_conservative_maximum_drawdown_bps(
        (-10.0, 0.0),
        (0.0, 5.0),
        run_ids=(RUNS[0], RUNS[0]),
        entry_monotonic_ns=(0, 20),
        exit_monotonic_ns=(10, 30),
        expected_run_ids=RUNS,
    ) == pytest.approx(15.0)


def test_conservative_drawdown_carries_equity_but_not_positions_across_runs() -> None:
    assert round74_conservative_maximum_drawdown_bps(
        (-10.0, 0.0),
        (0.0, 5.0),
        run_ids=RUNS,
        entry_monotonic_ns=(0, 0),
        exit_monotonic_ns=(10, 10),
        expected_run_ids=RUNS,
    ) == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("run_ids", "exits", "payoffs"),
    (
        ((RUNS[0],), (), (1.0,)),
        (("unknown",), (1,), (1.0,)),
        ((RUNS[0],), (-1,), (1.0,)),
        ((RUNS[0],), (1,), (float("nan"),)),
    ),
)
def test_realized_drawdown_rejects_invalid_identity_or_values(
    run_ids: tuple[str, ...],
    exits: tuple[int, ...],
    payoffs: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        round74_maximum_realized_drawdown_bps(
            payoffs,
            run_ids=run_ids,
            exit_monotonic_ns=exits,
            expected_run_ids=RUNS,
        )
