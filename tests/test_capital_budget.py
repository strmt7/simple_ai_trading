from fractions import Fraction as F

import pytest

from simple_ai_trading.capital_budget import evaluate_capital_budget


def evaluate(**overrides):
    return evaluate_capital_budget(
        **{
            "valuation_label": "supplied common quote unit; no FX admission",
            "gross_cash": F(3),
            "reserved_cost_cash": F(1),
            "minimum_capital": F(10),
            "annual_capital_charge": F(1, 10),
            "duration_ms": 100,
            "annual_charge_basis_ms": 200,
            **overrides,
        }
    )


def test_strict_upper_bound_and_minimum_do_not_qualify_a_native_edge():
    result = evaluate()
    assert result.charge_per_unit_capital == F(1, 20)
    assert result.strict_capital_upper_bound == F(40)
    assert result.net_at_minimum_capital == F(3, 2)
    assert result.necessary_condition_passed
    assert not result.native_capital_qualified and not result.qualified_edge


@pytest.mark.parametrize(
    "minimum,passed",
    [(F(40) - F(1, 10**40), True), (F(40), False), (F(40) + F(1, 10**40), False)],
)
def test_strict_equality_cannot_be_rounded_into_a_pass(minimum, passed):
    result = evaluate(minimum_capital=minimum)
    assert result.necessary_condition_passed is passed


@pytest.mark.parametrize("gross", [F(1), F(0), F(-1)])
def test_missing_budget_cannot_be_rescued_by_zero_capital(gross):
    result = evaluate(gross_cash=gross, minimum_capital=F(0))
    assert not result.necessary_condition_passed
    assert result.strict_capital_upper_bound <= 0


def test_cash_scaling_preserves_ratio_and_elapsed_time_changes_charge():
    base = evaluate()
    scaled = evaluate(
        gross_cash=F(300), reserved_cost_cash=F(100), minimum_capital=F(1000)
    )
    assert scaled.strict_capital_upper_bound == 100 * base.strict_capital_upper_bound
    assert scaled.net_at_minimum_capital == 100 * base.net_at_minimum_capital
    assert (
        evaluate(duration_ms=200).strict_capital_upper_bound
        == base.strict_capital_upper_bound / 2
    )


@pytest.mark.parametrize(
    "override",
    [
        {"gross_cash": 1.0},
        {"gross_cash": True},
        {"reserved_cost_cash": F(-1)},
        {"minimum_capital": F(-1)},
        {"annual_capital_charge": F(0)},
        {"duration_ms": True},
        {"duration_ms": 0},
        {"annual_charge_basis_ms": -1},
        {"annual_charge_basis_ms": 200.0},
        {"valuation_label": " "},
    ],
)
def test_ambiguous_units_values_and_time_conventions_reject(override):
    with pytest.raises(ValueError):
        evaluate(**override)
