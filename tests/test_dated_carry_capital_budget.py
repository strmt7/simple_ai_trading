from decimal import Decimal as D

import pytest

from tools.review_dated_carry_capital_budget import capital_budget


def test_capital_budget_is_not_double_counted_with_reserve():
    # One year, 2x fully committed capital at 3.25% costs 650 bp, independently
    # of the hypothetical 35-bp allocation for every noncapital cost.
    r = capital_budget(D(1000), 31557600000, D(".0325"), D(2), D(35))
    assert D(r["capital_cost_bips"]) == 650
    assert D(r["remaining_noncapital_cost_budget_bips"]) == 350
    assert D(r["headroom_after_separate_noncapital_reserve_bips"]) == 315
    assert D(r["maximum_capital_multiple_after_separate_reserve"]) > 2


def test_negative_gross_or_excess_reserve_is_not_clamped():
    r = capital_budget(D(-10), 31557600000, D(".01"), D(1), D(35))
    assert D(r["remaining_noncapital_cost_budget_bips"]) == -110
    assert D(r["maximum_capital_multiple_after_separate_reserve"]) < 0


def test_exact_boundary_has_no_positive_headroom():
    r = capital_budget(D(135), 31557600000, D(".01"), D(1), D(35))
    assert D(r["headroom_after_separate_noncapital_reserve_bips"]) == 0
    assert D(r["maximum_capital_multiple_after_separate_reserve"]) == 1


@pytest.mark.parametrize(
    "index,value",
    [
        (0, D("NaN")),
        (1, True),
        (1, 0),
        (2, D(0)),
        (2, D("Infinity")),
        (3, D(".5")),
        (4, D(-1)),
        (0, 1.0),
    ],
)
def test_invalid_budget_inputs_reject(index, value):
    args = [D(100), 31557600000, D(".01"), D(1), D(35)]
    args[index] = value
    with pytest.raises(ValueError):
        capital_budget(*args)
