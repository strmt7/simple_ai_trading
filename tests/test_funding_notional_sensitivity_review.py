from decimal import Decimal as D

import pytest

from tools.review_funding_notional_sensitivity import sensitivity


def test_variable_marks_can_reverse_positive_rate_sum():
    rates = [D("0.0011"), D("-0.001")]
    assert sum(rates) > 0
    assert sum(rate * mark for rate, mark in zip(rates, [D(100), D(120)])) == D("-0.01")
    result = sensitivity(rates, D(0))
    assert D(result["unit_weight_gross_bips"]) == 1
    assert D(result["illustrative_outer_bounds"][0]["lower_net_bips"]) == D("-1.1")


def test_independent_coordinate_bounds_include_frozen_cost_once():
    result = sensitivity([D("0.01"), D("-0.02")], D(5))
    assert D(result["unit_weight_net_bips"]) == -105
    row = result["illustrative_outer_bounds"][1]
    assert D(row["lower_net_bips"]) == -165
    assert D(row["upper_net_bips"]) == -45
    assert D(result["symmetric_weight_radius_to_zero_net"]) == D("0.35")


def test_zero_coefficients_do_not_invent_finite_break_even_radius():
    assert sensitivity([D(0)], D(5))["symmetric_weight_radius_to_zero_net"] is None


@pytest.mark.parametrize(
    "rates,hurdle",
    [
        ([], D(0)),
        ([D("NaN")], D(0)),
        ([D("Infinity")], D(0)),
        ([D(1)], D(-1)),
        ([D(1)], D("NaN")),
    ],
)
def test_invalid_economics_rejected(rates, hurdle):
    with pytest.raises(ValueError):
        sensitivity(rates, hurdle)
