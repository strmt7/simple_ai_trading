from dataclasses import replace
from decimal import Decimal as D

import pytest

from simple_ai_trading.collateral_stress import CollateralState, stress_linear_hedge


def evaluate(states=None, **changes):
    arguments = dict(
        owned_asset_quantity=D(1),
        short_base_quantity=D(1),
        future_entry_price=D(100),
        quote_cash=D(0),
        required_quote_buffer=D(0),
        states=(CollateralState("base", D(100), D(100), D(".95"), D(10), D(0)),)
        if states is None
        else states,
    )
    return stress_linear_hedge(**(arguments | changes))


def test_matched_economic_equity_does_not_prove_margin_safety():
    states = tuple(
        CollateralState(str(p), D(p), D(p), D(".95"), D(p) / 10, D(0))
        for p in (100, 200, 1000)
    )
    result = evaluate(states)
    assert [s.economic_equity for s in result.states] == [100, 100, 100]
    assert [s.credited_equity for s in result.states] == [95, 90, 50]
    assert [s.headroom_after_requirement_and_buffer for s in result.states] == [
        85,
        70,
        -50,
    ]
    assert result.extra_fully_credited_cash_to_meet_supplied_states == 50
    assert not result.strictly_positive_in_every_supplied_state
    assert not result.venue_margin_qualified and not result.qualified_edge
    equality = evaluate(states, quote_cash=D(50))
    assert equality.minimum_headroom == 0
    assert not equality.strictly_positive_in_every_supplied_state
    assert evaluate(states, quote_cash=D(51)).strictly_positive_in_every_supplied_state


def test_basis_mismatch_quantity_and_cost_are_not_netted_away():
    state = CollateralState("basis", D(200), D(210), D(".9"), D(20), D(3))
    result = evaluate((state,), required_quote_buffer=D(5))
    value = result.states[0]
    assert value.short_unrealized_pnl == -110
    assert value.economic_equity == 87
    assert value.credited_equity == 67
    assert result.minimum_headroom == 42
    assert result.extra_fully_credited_cash_to_meet_supplied_states == 0
    assert (
        evaluate((state,), short_base_quantity=D(".5")).states[0].economic_equity == 142
    )
    assert (
        evaluate((replace(state, asset_credit_ratio=D(1)),))
        .states[0]
        .collateral_discount
        == 0
    )
    assert (
        evaluate((replace(state, asset_credit_ratio=D(0)),)).states[0].credited_equity
        == -113
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("spot_mark", D(0)),
        ("future_mark", D(0)),
        ("asset_credit_ratio", D("1.01")),
        ("asset_credit_ratio", D(-1)),
        ("total_margin_requirement", D("NaN")),
        ("cumulative_cost_debits", D(-1)),
        ("spot_mark", 100.0),
        ("label", ""),
        ("label", None),
    ],
)
def test_invalid_state_rejects(field, value):
    state = CollateralState("base", D(100), D(100), D(1), D(0), D(0))
    with pytest.raises(ValueError):
        evaluate((replace(state, **{field: value}),))


@pytest.mark.parametrize(
    "changes",
    [
        {"future_entry_price": D(0)},
        {"quote_cash": D(-1)},
        {"required_quote_buffer": D("Infinity")},
        {"owned_asset_quantity": True},
        {"states": ()},
        {"states": []},
        {"states": (None,)},
    ],
)
def test_invalid_portfolio_rejects(changes):
    with pytest.raises(ValueError):
        evaluate(**changes)


def test_duplicate_labels_reject():
    state = CollateralState("base", D(100), D(100), D(1), D(0), D(0))
    with pytest.raises(ValueError):
        evaluate((state, state))
