from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_inventory import (
    InventoryCashPath,
    replay_fixed_base_inventory,
)
from simple_ai_trading.inventory_cash_actions import (
    InventoryCashAction,
    evaluate_inventory_cash_actions,
)
from simple_ai_trading.stateful_turnover_model import train_stateful_forecasts


def cash_path(*, incumbent=1, exit_price=100, events=()) -> InventoryCashPath:
    return InventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (Fraction(100), Fraction(exit_price)),
        (0,),
        Fraction(100),
        Fraction(6, 10_000),
        "a" * 64,
        events,
        tuple((event.funding_time_ms, event.rate) for event in events),
        0,
        30,
        "b" * 64,
        initial_signed_base_quantity=Fraction(incumbent),
    )


def payment(time, *, rate="0.01") -> LinearFundingSettlement:
    return LinearFundingSettlement(
        "BTCUSDT", "USDT", time, Fraction(rate), Fraction(150), "c" * 64
    )


def evaluate(path, targets=(0, 1, -1)):
    actions = tuple(
        InventoryCashAction(
            "flatten" if target == 0 else f"action_{index}", Fraction(target)
        )
        for index, target in enumerate(targets)
    )
    return evaluate_inventory_cash_actions(
        path,
        actions,
        state_as_of_ms=8,
        decision_time_ms=9,
        state_source_sha256="d" * 64,
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_hold_is_not_a_new_round_trip_and_flatten_surplus_is_zero(side) -> None:
    held = evaluate(cash_path(incumbent=side), (0, side)).values
    opened = evaluate(cash_path(incumbent=0), (0, side)).values
    assert held[0].cash_replay.total_net_cash_lower == Fraction(-3, 50)
    assert held[0].surplus_cash_lower == held[0].surplus_cash_upper == 0
    assert held[1].cash_replay.total_execution_cost == Fraction(3, 50)
    assert held[1].cash_replay.intervals[0].entry_quantity_change == 0
    assert held[1].surplus_cash_lower == held[1].surplus_cash_upper == 0
    assert opened[1].cash_replay.total_execution_cost == Fraction(3, 25)
    assert opened[1].surplus_reference_lower_bps == -12


def test_holding_resizing_and_reversal_retain_actual_units_and_trade_cost() -> None:
    table = evaluate(cash_path(exit_price=200), (0, 1, Fraction(1, 2), -1, 2))
    _, hold, reduce, reverse, increase = table.values
    assert hold.cash_replay.intervals[0].signed_base_quantity == 1
    assert hold.cash_replay.total_traded_quote == 200
    assert hold.surplus_cash_lower == Fraction("99.94")
    assert reduce.cash_replay.total_traded_quote == 150
    assert reduce.surplus_cash_lower == Fraction("49.97")
    assert reverse.cash_replay.intervals[0].entry_quantity_change == -2
    assert reverse.cash_replay.total_traded_quote == 400
    assert reverse.surplus_cash_lower == Fraction("-100.18")
    assert increase.cash_replay.total_traded_quote == 500
    assert increase.surplus_cash_lower == Fraction("199.76")
    assert not table.financially_qualified


@pytest.mark.parametrize("incumbent", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("target", [-2, Fraction(-1, 2), 0, Fraction(1, 2), 2])
@pytest.mark.parametrize("time", [10, 15, 20])
def test_transition_cash_encloses_every_monotone_partial_quantity(
    incumbent, target, time
) -> None:
    path = cash_path(incumbent=incumbent, exit_price=125, events=(payment(time),))
    path = replace(
        path,
        desired_position=((target > 0) - (target < 0),),
        target_signed_base_quantity=(Fraction(target),),
    )
    replay = replay_fixed_base_inventory(path)
    before = Fraction(incumbent) if time == 10 else Fraction(target)
    after = Fraction(0) if time == 20 else Fraction(target)
    price_cash = Fraction(target) * 25
    cost = (abs(Fraction(target) - incumbent) * 100 + abs(target) * 125) * Fraction(
        6, 10_000
    )
    for weight in (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
        quantity = before + weight * (after - before)
        cash = price_cash - quantity * Fraction(3, 2) - cost
        assert replay.total_net_cash_lower <= cash <= replay.total_net_cash_upper
    assert len(replay.funding_events) == 1


@pytest.mark.parametrize(
    "rate,low,high", [("0.01", "-1.5", "-0.75"), ("-0.01", "0.75", "1.5")]
)
def test_same_sign_resize_boundary_does_not_invent_flat_entitlement(rate, low, high):
    table = evaluate(cash_path(events=(payment(10, rate=rate),)), (0, Fraction(1, 2)))
    event = table.values[1].cash_replay.funding_events[0]
    assert event.boundary_quantity_uncertain
    assert (event.quantity_before, event.quantity_after) == (1, Fraction(1, 2))
    assert (event.cash_lower, event.cash_upper) == (Fraction(low), Fraction(high))


def test_unmodified_incumbent_funding_is_held_at_entry_not_a_fresh_open():
    table = evaluate(cash_path(events=(payment(10),)))
    flatten, hold, reverse = table.values
    assert not hold.cash_replay.funding_events[0].boundary_quantity_uncertain
    assert (
        hold.cash_replay.total_net_cash_lower == hold.cash_replay.total_net_cash_upper
    )
    assert hold.surplus_cash_lower == Fraction("-1.5")
    assert hold.surplus_cash_upper == 0
    assert flatten.surplus_cash_lower == flatten.surplus_cash_upper == 0
    assert reverse.surplus_cash_lower == Fraction("-1.62")
    assert reverse.surplus_cash_upper == Fraction("2.88")


def test_incumbent_default_policy_preserves_units_without_explicit_resizing():
    replay = replay_fixed_base_inventory(
        replace(cash_path(exit_price=200), desired_position=(1,))
    )
    assert replay.intervals[0].signed_base_quantity == 1
    assert replay.intervals[0].entry_quantity_change == 0
    assert replay.total_net_cash_lower == Fraction("99.88")


@pytest.mark.parametrize(
    "change",
    [
        {"initial_signed_base_quantity": 1.0},
        {"target_signed_base_quantity": [Fraction(1)]},
        {"target_signed_base_quantity": ()},
        {"target_signed_base_quantity": (Fraction(1),)},
        {"target_signed_base_quantity": (0.0,)},
    ],
)
def test_quantity_path_rejects_wrong_units_or_sign_binding(change):
    with pytest.raises(ValueError, match="contract"):
        replace(cash_path(), **change)


@pytest.mark.parametrize(
    "actions",
    [
        (),
        [],
        (None,),
        (InventoryCashAction("hold", Fraction(1)),),
        (InventoryCashAction("flatten", Fraction(1)),),
        (InventoryCashAction("flat", Fraction(0)),),
        (
            InventoryCashAction("flatten", Fraction(0)),
            InventoryCashAction("duplicate", Fraction(0)),
        ),
        (
            InventoryCashAction("flatten", Fraction(0)),
            InventoryCashAction("hold", Fraction(1)),
            InventoryCashAction("hold", Fraction(-1)),
        ),
    ],
)
def test_action_contract_rejects_missing_or_ambiguous_comparator(actions):
    with pytest.raises(ValueError):
        evaluate_inventory_cash_actions(
            cash_path(),
            actions,
            state_as_of_ms=8,
            decision_time_ms=9,
            state_source_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    "name,quantity", [("", Fraction(1)), ("A", Fraction(1)), ("hold", 1.0)]
)
def test_action_identity_and_units_reject(name, quantity):
    with pytest.raises(ValueError):
        InventoryCashAction(name, quantity)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_as_of_ms": -1},
        {"state_as_of_ms": 10},
        {"state_as_of_ms": True},
        {"decision_time_ms": 10},
        {"decision_time_ms": 7},
        {"decision_time_ms": True},
        {"state_source_sha256": "not-proof"},
    ],
)
def test_causal_state_contract_rejects_missing_or_future_declarations(kwargs):
    arguments = dict(state_as_of_ms=8, decision_time_ms=9, state_source_sha256="d" * 64)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="state, horizon"):
        evaluate_inventory_cash_actions(
            cash_path(), (InventoryCashAction("flatten", Fraction(0)),), **arguments
        )


@pytest.mark.parametrize(
    "path", [None, replace(cash_path(), one_way_cost_fraction=Fraction(0))]
)
def test_action_scope_and_cost_reject_before_replay(path):
    with pytest.raises(ValueError):
        evaluate(path)


def test_multi_interval_continuation_is_not_silently_a_one_step_label():
    path = replace(
        cash_path(),
        boundary_time_ms=(10, 15, 20),
        boundary_price=(Fraction(100),) * 3,
        desired_position=(0, 0),
    )
    with pytest.raises(ValueError, match="horizon"):
        evaluate(path)


@pytest.mark.parametrize(
    "change",
    [
        {"initial_signed_base_quantity": Fraction(2)},
        {"one_way_cost_fraction": Fraction(7, 10_000)},
        {"price_source_sha256": "e" * 64},
        {"population_certificate_sha256": "e" * 64},
        {"coverage_end_exclusive_ms": 31},
        {"boundary_price": (Fraction(100), Fraction(101))},
    ],
)
def test_replay_binding_changes_with_cash_inputs(change):
    assert (
        evaluate(cash_path()).input_sha256
        != evaluate(replace(cash_path(), **change)).input_sha256
    )


def test_state_action_and_funding_bindings_are_not_only_price_labels():
    path = cash_path()
    baseline = evaluate(path)
    assert baseline == evaluate(path)
    assert baseline.input_sha256 != evaluate(path, (0, Fraction(1, 2), -1)).input_sha256
    assert (
        baseline.input_sha256 != evaluate(cash_path(events=(payment(15),))).input_sha256
    )
    changed_state = evaluate_inventory_cash_actions(
        path,
        tuple(row.action for row in baseline.values),
        state_as_of_ms=7,
        decision_time_ms=9,
        state_source_sha256="e" * 64,
    )
    assert baseline.input_sha256 != changed_state.input_sha256


def test_counterfactual_cannot_enter_legacy_signed_trainer(tmp_path: Path):
    model_dir = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="legacy hourly dataset"):
        train_stateful_forecasts(
            evaluate(cash_path()), model_dir=model_dir, compute_backend="cpu"
        )
    assert not model_dir.exists()


def test_financial_admission_flag_cannot_be_set_by_constructor():
    with pytest.raises(ValueError, match="init=False"):
        replace(evaluate(cash_path()), financially_qualified=True)
