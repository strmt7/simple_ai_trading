"""Supplied-law proofs and counterexamples; no vendor fees or market quotes."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from simple_ai_trading.option_long_floor import (
    CappedExerciseFee,
    LongOptionFloor,
    evaluate_fully_paid_long_option_floor,
)


def _fee(rate="0.1", cap="0.5"):
    return CappedExerciseFee(Decimal(rate), Decimal(cap))


def _floor(**changes):
    inputs = {
        "lower_call_strike": Decimal("100"),
        "upper_put_strike": Decimal("300"),
        "matched_base_quantity": Decimal("1"),
        "call_exercise_fee": _fee(),
        "put_exercise_fee": _fee(),
        "terminal_cost_reserve_quote": Decimal("0"),
    }
    return evaluate_fully_paid_long_option_floor(**{**inputs, **changes})


def test_exercise_fee_minimum_is_between_strikes_not_at_a_sampled_endpoint():
    result = _floor()
    assert result.gross_payout_floor_quote == 200
    assert result.terminal_net_payout_floor_quote == 150
    assert result.minimizing_breakpoints == (Fraction(250),)
    endpoints = [
        point.terminal_net_payout_quote
        for point in result.critical_points
        if point.settlement_price in (100, 300)
    ]
    assert min(endpoints) == 170
    assert result.qualified_edge is False


def test_nonterminating_switch_and_payout_remain_exact_under_low_decimal_precision():
    with localcontext() as context:
        context.prec = 2
        result = _floor(call_exercise_fee=_fee("0.2"), put_exercise_fee=_fee("0.2"))
    assert result.minimizing_breakpoints == (Fraction(1500, 7),)
    assert result.terminal_net_payout_floor_quote == Fraction(800, 7)
    assert Fraction(500, 3) in {
        point.settlement_price for point in result.critical_points
    }


def test_no_fee_law_has_the_exact_strike_width_floor():
    result = _floor(call_exercise_fee=_fee("0", "0"), put_exercise_fee=_fee("0", "0"))
    assert (
        result.terminal_net_payout_floor_quote == result.gross_payout_floor_quote == 200
    )
    assert result.minimizing_breakpoints == (Fraction(100), Fraction(300))


def test_unequal_leg_fee_laws_are_not_averaged():
    result = _floor(call_exercise_fee=_fee("0", "0"))
    assert result.terminal_net_payout_floor_quote == 175
    assert result.minimizing_breakpoints == (Fraction(250),)


def test_quantity_and_explicit_terminal_reserve_have_distinct_units():
    result = _floor(
        matched_base_quantity=Decimal("0.1"), terminal_cost_reserve_quote=Decimal("2")
    )
    assert result.gross_payout_floor_quote == 20
    assert result.terminal_net_payout_floor_quote == 13
    point = next(
        point for point in result.critical_points if point.settlement_price == 250
    )
    assert point.call_gross_payout_quote == 15
    assert point.put_gross_payout_quote == 5
    assert (
        point.call_exercise_fee_quote == point.put_exercise_fee_quote == Fraction(5, 2)
    )


def test_reserve_can_exceed_payout_without_being_floored_to_zero():
    assert (
        _floor(
            terminal_cost_reserve_quote=Decimal("200")
        ).terminal_net_payout_floor_quote
        == -50
    )


@pytest.mark.parametrize(
    "rate,cap", [("0", "0.5"), ("1", "0"), ("0.5", "0.5"), ("1", "1"), ("0.5", "0.1")]
)
def test_degenerate_fee_switches_and_tail_slopes(rate, cap):
    result = _floor(call_exercise_fee=_fee(rate, cap), put_exercise_fee=_fee(rate, cap))
    assert len(result.critical_points) <= 5
    assert result.terminal_net_payout_floor_quote >= 0
    # Independent checks away from vertices include a large-price tail.
    r, c = Fraction(rate), Fraction(cap)
    for price in (
        Fraction(0),
        Fraction(37),
        Fraction(105),
        Fraction(207),
        Fraction(10000),
    ):
        call, put = max(price - 100, 0), max(300 - price, 0)
        net = call + put - min(r * price, c * call) - min(r * price, c * put)
        assert net >= result.terminal_net_payout_floor_quote


def test_result_cannot_be_constructed_or_mutated_to_claim_qualification():
    with pytest.raises(TypeError):
        LongOptionFloor(Fraction(1), Fraction(1), (), (), qualified_edge=True)
    result = _floor()
    with pytest.raises(FrozenInstanceError):
        result.qualified_edge = True


@pytest.mark.parametrize(
    "field,value",
    [
        ("lower_call_strike", Decimal("0")),
        ("lower_call_strike", Decimal("300")),
        ("upper_put_strike", Decimal("99")),
        ("matched_base_quantity", Decimal("0")),
        ("matched_base_quantity", Decimal("-1")),
        ("terminal_cost_reserve_quote", Decimal("-1")),
        ("lower_call_strike", True),
        ("matched_base_quantity", 1.0),
        ("upper_put_strike", Decimal("NaN")),
        ("terminal_cost_reserve_quote", Decimal("Infinity")),
        ("lower_call_strike", Decimal("1e101")),
        ("upper_put_strike", Decimal("1" * 101)),
        ("call_exercise_fee", None),
        ("put_exercise_fee", "unknown"),
    ],
)
def test_invalid_or_unbounded_inputs_do_not_create_a_floor(field, value):
    with pytest.raises(ValueError):
        _floor(**{field: value})


@pytest.mark.parametrize(
    "rate,cap",
    [("-0.1", "0.5"), ("1.1", "0.5"), ("0.1", "-0.1"), ("0.1", "1.1"), ("NaN", "0.5")],
)
def test_fee_coefficients_outside_the_declared_law_are_rejected(rate, cap):
    with pytest.raises(ValueError):
        _floor(call_exercise_fee=_fee(rate, cap))
