from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as F

import pytest

from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_inventory import InventoryCashPath
from simple_ai_trading.hedged_inventory_cash import (
    HedgedInventoryCashPath,
    replay_hedged_inventory_cash,
)
from simple_ai_trading.spot_inventory_cash import (
    SpotCashTrade,
    SpotInventoryCashPath,
    cash_input_digest,
    replay_spot_inventory_cash,
)


def trade(index, side, quantity, cash, *, base_fee=0, quote_fee=0, name=None):
    return SpotCashTrade(
        name or f"fill-{index}-{side}",
        index,
        side,
        F(quantity),
        F(cash),
        F(base_fee),
        F(quote_fee),
        "a" * 64,
    )


def paired(*, quantity=F(1), spot_exit=120, future_exit=121, events=(), fee=F(0)):
    perpetual = InventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (F(101), F(future_exit)),
        (-1,),
        F(100),
        fee,
        "b" * 64,
        events,
        tuple((e.funding_time_ms, e.rate) for e in events),
        0,
        30,
        "c" * 64,
        target_signed_base_quantity=(-quantity,),
    )
    spot = SpotInventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (F(100), F(spot_exit)),
        F(0),
        (trade(0, "BUY", 1, 100), trade(1, "SELL", 1, spot_exit)),
        "d" * 64,
    )
    return HedgedInventoryCashPath(perpetual, spot, (F(0),), (F(0),), (F(0),), "e" * 64)


def settlement(time, rate):
    return LinearFundingSettlement("BTCUSDT", "USDT", time, F(rate), F(110), "f" * 64)


@pytest.mark.parametrize("spot_exit", [80, 120, 1000])
def test_matched_base_cancels_common_price_changes_but_not_boundary_orphans(spot_exit):
    result = replay_hedged_inventory_cash(
        paired(spot_exit=spot_exit, future_exit=spot_exit + 1)
    )
    assert result.total_net_cash_lower == result.total_net_cash_upper == 0
    assert result.base_matched_in_every_supplied_interval
    assert result.boundary_orphan_base_ranges == ((-1, 1), (-1, 1))
    assert not result.financially_qualified and not result.venue_margin_qualified
    assert not result.training_admitted and not result.qualified_edge


def test_equal_quote_notionals_are_not_equal_base_or_a_neutral_hedge():
    result = replay_hedged_inventory_cash(paired(quantity=F(100, 101)))
    assert result.intervals[0].residual_base_quantity == F(1, 101)
    assert result.total_net_cash_lower == F(20, 101)
    assert not result.base_matched_in_every_supplied_interval


def test_flat_abstention_is_zero_and_intragrid_round_trips_retain_orphan_exposure():
    path = paired()
    flat = replace(
        path,
        spot=replace(path.spot, trades=()),
        perpetual=replace(
            path.perpetual, desired_position=(0,), target_signed_base_quantity=(F(0),)
        ),
    )
    assert replay_hedged_inventory_cash(flat).total_net_cash_lower == 0
    round_trip = replace(
        flat,
        spot=replace(
            flat.spot,
            trades=(
                trade(0, "BUY", 1, 100, name="buy"),
                trade(0, "SELL", 1, 99, name="sell"),
            ),
        ),
    )
    result = replay_hedged_inventory_cash(round_trip)
    assert result.total_net_cash_lower == -1
    assert result.base_matched_in_every_supplied_interval
    assert result.boundary_orphan_base_ranges == ((0, 1), (0, 0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cash_input_binding_rejects_nonfinite_json(value):
    with pytest.raises(ValueError):
        cash_input_digest({"value": value})


def test_buy_base_commission_changes_net_quantity_cash_and_directional_residual_once():
    path = paired()
    spot = replace(
        path.spot,
        trades=(
            trade(0, "BUY", 1, 100, base_fee="0.01"),
            trade(1, "SELL", "0.99", "118.8"),
        ),
    )
    result = replay_hedged_inventory_cash(replace(path, spot=spot))
    assert result.spot.quantity_by_interval == (F("0.99"),)
    assert result.spot.incremental_wealth_change == F("18.8")
    assert result.intervals[0].residual_base_quantity == -F("0.01")
    assert result.total_net_cash_lower == -F("1.2")
    matched = replay_hedged_inventory_cash(
        replace(
            path,
            spot=spot,
            perpetual=replace(
                path.perpetual, target_signed_base_quantity=(-F("0.99"),)
            ),
        )
    )
    assert matched.total_net_cash_lower == -1
    assert matched.base_matched_in_every_supplied_interval


def test_both_quote_fees_perpetual_costs_and_all_external_debits_are_charged_once():
    path = paired(events=(settlement(15, "0.001"),), fee=F(6, 10_000))
    spot = replace(
        path.spot,
        trades=(
            trade(0, "BUY", 1, 100, quote_fee="0.1"),
            trade(1, "SELL", 1, 120, quote_fee="0.12"),
        ),
    )
    path = replace(
        path,
        spot=spot,
        financing_debit_quote=(F("0.02"),),
        opportunity_debit_quote=(F("0.03"),),
        other_debit_quote=(F("0.04"),),
    )
    result = replay_hedged_inventory_cash(path)
    assert result.perpetual.total_execution_cost == F("0.1332")
    assert result.perpetual.funding_events[0].cash_lower == F("0.11")
    assert result.total_external_debit_quote == F("0.09")
    assert result.total_net_cash_lower == result.total_net_cash_upper == -F("0.3332")


@pytest.mark.parametrize("time", [10, 20])
@pytest.mark.parametrize("rate", ["0.01", "-0.01"])
def test_boundary_funding_reuses_original_conservative_entitlement(time, rate):
    result = replay_hedged_inventory_cash(paired(events=(settlement(time, rate),)))
    payment = F(110) * F(rate)
    assert result.total_net_cash_lower == min(0, payment)
    assert result.total_net_cash_upper == max(0, payment)


def test_partial_fills_and_resize_preserve_cash_and_each_interval_quantity():
    path = paired()
    perpetual = replace(
        path.perpetual,
        boundary_time_ms=(10, 15, 20),
        boundary_price=(F(101), F(111), F(121)),
        desired_position=(-1, -1),
        target_signed_base_quantity=(-F(1), -F("0.5")),
    )
    spot = replace(
        path.spot,
        boundary_time_ms=(10, 15, 20),
        boundary_price=(F(100), F(110), F(120)),
        trades=(
            trade(0, "BUY", "0.4", 40, name="partial-a"),
            trade(0, "BUY", "0.6", 60, name="partial-b"),
            trade(1, "SELL", "0.5", 55),
            trade(2, "SELL", "0.5", 60),
        ),
    )
    result = replay_hedged_inventory_cash(
        replace(
            path,
            perpetual=perpetual,
            spot=spot,
            financing_debit_quote=(F(0), F(0)),
            opportunity_debit_quote=(F(0), F(0)),
            other_debit_quote=(F(0), F(0)),
        )
    )
    assert result.spot.quantity_by_interval == (1, F("0.5"))
    assert result.spot.recorded_quote_cash_change == 15
    assert result.perpetual.total_price_cash == -15
    assert result.total_net_cash_lower == 0
    assert result.boundary_orphan_base_ranges[1] == (-F("0.5"), F("0.5"))


def test_incumbent_inventory_excludes_sunk_cash_and_does_not_invent_a_fresh_entry():
    path = paired(fee=F(6, 10_000))
    spot = replace(
        path.spot, initial_base_quantity=F(1), trades=(trade(1, "SELL", 1, 120),)
    )
    result = replay_hedged_inventory_cash(
        replace(
            path,
            spot=spot,
            perpetual=replace(path.perpetual, initial_signed_base_quantity=-F(1)),
        )
    )
    assert result.spot.initial_marked_inventory == 100
    assert result.spot.incremental_wealth_change == 20
    assert result.perpetual.total_traded_quote == 121
    assert result.total_net_cash_lower == -F("0.0726")
    assert result.boundary_orphan_base_ranges[0] == (0, 0)


def test_cash_binding_changes_for_fill_source_cost_or_price_changes():
    path = paired()
    original = replay_hedged_inventory_cash(path).input_sha256
    changes = [
        replace(path, cost_source_sha256="1" * 64),
        replace(path, other_debit_quote=(F(1),)),
        replace(path, spot=replace(path.spot, price_source_sha256="2" * 64)),
        replace(
            path,
            spot=replace(
                path.spot,
                trades=(
                    replace(path.spot.trades[0], source_body_sha256="3" * 64),
                    path.spot.trades[1],
                ),
            ),
        ),
    ]
    assert all(
        replay_hedged_inventory_cash(p).input_sha256 != original for p in changes
    )
    with pytest.raises(TypeError, match="unsupported"):
        cash_input_digest(object())


@pytest.mark.parametrize(
    "change",
    [
        {"boundary_index": True},
        {"boundary_index": -1},
        {"side": "SHORT"},
        {"gross_base_quantity": F(0)},
        {"gross_quote_quantity": F(0)},
        {"gross_base_quantity": 1.0},
        {"base_commission": F(1)},
        {"quote_commission": -F(1)},
        {"source_body_sha256": "bad"},
        {"execution_id": ""},
        {"side": "SELL", "base_commission": F("0.01")},
    ],
)
def test_malformed_or_unsupported_spot_fill_rejects(change):
    with pytest.raises(ValueError):
        replace(trade(0, "BUY", 1, 100), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "BTCUSDC"},
        {"symbol": []},
        {"boundary_time_ms": (10, 10)},
        {"boundary_time_ms": (True, 20)},
        {"boundary_price": (F(100), F(0))},
        {"initial_base_quantity": -F(1)},
        {"price_source_sha256": "bad"},
        {"trades": []},
        {"trades": (object(),)},
    ],
)
def test_invalid_spot_path_rejects(change):
    with pytest.raises(ValueError):
        replace(paired().spot, **change)


def test_bad_fill_order_duplicates_oversells_and_unclosed_inventory_reject():
    path = paired().spot
    for trades in [
        tuple(reversed(path.trades)),
        (
            path.trades[0],
            replace(path.trades[1], execution_id=path.trades[0].execution_id),
        ),
        (replace(path.trades[1], boundary_index=2),),
    ]:
        with pytest.raises(ValueError):
            replace(path, trades=trades)
    for trades in [
        (trade(0, "BUY", 1, 100, base_fee="0.01"), path.trades[1]),
        path.trades[:1],
    ]:
        with pytest.raises(ValueError):
            replay_spot_inventory_cash(replace(path, trades=trades))


@pytest.mark.parametrize(
    "change",
    [
        {"perpetual": None},
        {"spot": None},
        {"cost_source_sha256": "bad"},
        {"financing_debit_quote": (F(-1),)},
        {"opportunity_debit_quote": ()},
        {"other_debit_quote": (0.0,)},
    ],
)
def test_invalid_pair_or_missing_cost_declarations_reject(change):
    with pytest.raises(ValueError):
        replace(paired(), **change)


def test_mismatched_units_clocks_and_long_futures_are_not_this_cash_hedge():
    path = paired()
    for perpetual in [
        replace(path.perpetual, symbol="ETHUSDT"),
        replace(path.perpetual, boundary_time_ms=(11, 20)),
        replace(path.perpetual, initial_signed_base_quantity=F(1)),
        replace(
            path.perpetual, desired_position=(1,), target_signed_base_quantity=(F(1),)
        ),
    ]:
        with pytest.raises(ValueError):
            replace(path, perpetual=perpetual)
    with pytest.raises(ValueError):
        replay_spot_inventory_cash(None)
    with pytest.raises(ValueError):
        replay_hedged_inventory_cash(None)
