from __future__ import annotations

import pytest

from tools.storage_io import IoBudget


def test_aggregate_io_budget_has_no_idle_credit_or_initial_burst():
    current, delays = [0.0], []

    def sleep(delay):
        delays.append(delay)
        current[0] += delay

    budget = IoBudget(64, clock=lambda: current[0], sleep=sleep)
    budget.consume(2 * 2**20)
    assert delays == [2 / 64]
    current[0] += 30
    budget.consume(2**20)
    assert delays[-1] == 1 / 64
    with pytest.raises(ValueError, match="Negative"):
        budget.consume(-1)


@pytest.mark.parametrize("rate", [0, -1, 257, float("nan"), float("inf")])
def test_invalid_io_budget_cannot_start(rate):
    with pytest.raises(ValueError, match="I/O budget"):
        IoBudget(rate)
