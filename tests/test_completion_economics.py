from dataclasses import replace
from decimal import Decimal as D

import pytest

from simple_ai_trading.completion_economics import CompletionState, compare_completion


def full():
    return CompletionState("full", D(1), D(100), D(40), D(0), D(1), None, None)


def compare(states=None, **overrides):
    inputs = dict(
        original_net_shares=D(100),
        liquidation_net_proceeds=D(65),
        states=(full(),) if states is None else states,
        historical_acquisition_cost=D(35),
    )
    return compare_completion(**(inputs | overrides))


def test_positive_historical_profit_can_underperform_sale():
    result = compare()
    assert result.states[0].historical_completion_pnl == 25
    assert result.states[0].historical_liquidation_pnl == 30
    assert result.expected_incremental_value == -5
    assert result.probability_underperforming_liquidation == 1
    assert not result.qualified_edge


def test_net_liquidation_can_be_negative_when_costs_exceed_proceeds():
    state = replace(
        full(),
        opposite_net_shares=D(0),
        acquisition_cash=D(0),
        pair_net_realization=None,
        original_residual_net_bid=D("-0.01"),
    )
    result = compare((state,), liquidation_net_proceeds=D("-0.5"))
    assert result.states[0].completion_net_value == -1
    assert result.expected_incremental_value == D("-0.5")


def test_historical_basis_does_not_change_incremental_decision():
    a, b = compare(), compare(historical_acquisition_cost=D(90))
    assert a.expected_incremental_value == b.expected_incremental_value
    assert b.states[0].historical_completion_pnl == -30
    assert (
        compare(historical_acquisition_cost=None).states[0].historical_completion_pnl
        is None
    )


def test_partial_net_shares_and_overfill_are_valued_without_clipping():
    partial = replace(
        full(),
        opposite_net_shares=D(40),
        acquisition_cash=D(16),
        original_residual_net_bid=D("0.5"),
    )
    value = compare((partial,)).states[0]
    assert (value.matched_shares, value.original_residual_shares) == (40, 60)
    assert value.completion_net_value == 54
    assert value.incremental_value == -11
    overfill = replace(
        full(),
        opposite_net_shares=D(110),
        acquisition_cash=D(44),
        opposite_residual_net_bid=D("0.3"),
    )
    value = compare((overfill,)).states[0]
    assert value.opposite_residual_shares == 10
    assert value.completion_net_value == 59


def test_no_fill_is_not_forced_to_zero_cash_or_zero_risk():
    state = replace(
        full(),
        opposite_net_shares=D(0),
        acquisition_cash=D(0),
        pair_net_realization=None,
        original_residual_net_bid=D("0.4"),
        additional_cost=D(1),
    )
    assert compare((state,)).states[0].incremental_value == -26


def test_positive_expected_value_is_not_all_state_profit():
    win = replace(full(), probability=D("0.9"), acquisition_cash=D(30))
    loss = replace(
        full(),
        label="partial",
        probability=D("0.1"),
        opposite_net_shares=D(40),
        acquisition_cash=D(16),
        original_residual_net_bid=D("0.5"),
    )
    result = compare((win, loss))
    assert result.expected_incremental_value == D("3.4")
    assert result.worst_state_incremental_value == -11
    assert result.probability_underperforming_liquidation == D("0.1")
    assert not result.positive_in_every_supplied_state
    assert not result.qualified_edge


def test_strict_positive_state_gate_and_explicit_costs():
    assert compare(
        (replace(full(), acquisition_cash=D(30)),)
    ).positive_in_every_supplied_state
    tie = replace(full(), acquisition_cash=D(35))
    assert compare((tie,)).expected_incremental_value == 0
    assert not compare((tie,)).positive_in_every_supplied_state
    assert (
        compare((replace(tie, additional_cost=D(2)),)).expected_incremental_value == -2
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"original_net_shares": D(0)},
        {"original_net_shares": D("NaN")},
        {"liquidation_net_proceeds": None},
        {"liquidation_net_proceeds": D("NaN")},
        {"historical_acquisition_cost": D("Infinity")},
        {"states": ()},
        {"states": (full(), full())},
        {"states": (replace(full(), label=" "),)},
        {"states": (replace(full(), probability=D(0)),)},
        {"states": (replace(full(), probability=D("0.99")),)},
    ],
)
def test_invalid_decisions_reject(changes):
    with pytest.raises(ValueError):
        compare(**changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("opposite_net_shares", D(-1)),
        ("acquisition_cash", D("NaN")),
        ("additional_cost", D(-1)),
        ("pair_net_realization", None),
        ("pair_net_realization", D("Infinity")),
    ],
)
def test_invalid_state_values_reject(field, value):
    with pytest.raises(ValueError):
        compare((replace(full(), **{field: value}),))


@pytest.mark.parametrize(
    "quantity,field",
    [(D(40), "original_residual_net_bid"), (D(110), "opposite_residual_net_bid")],
)
def test_missing_nonzero_residual_bid_is_unknown_not_free(quantity, field):
    with pytest.raises(ValueError, match=field):
        compare((replace(full(), opposite_net_shares=quantity),))
