from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as F

import pytest

from simple_ai_trading.spot_inventory_cash import (
    SpotCashTrade,
    SpotInventoryCashPath,
    replay_spot_inventory_cash,
)
from simple_ai_trading.spot_reserved_cash import (
    SpotQuoteReservations,
    evaluate_spot_reserved_cash,
)


def path(*, partial=False, empty=False, quote_fee=F(0), base_fee=F(0)):
    trades = (
        ()
        if empty
        else (
            SpotCashTrade("buy", 0, "BUY", F(1), F(100), base_fee, quote_fee, "a" * 64),
            SpotCashTrade(
                "sell", 1, "SELL", 1 - base_fee, F(120), F(0), F(0), "b" * 64
            ),
        )
    )
    if partial:
        trades = (
            SpotCashTrade("buy1", 0, "BUY", F(".4"), F(40), F(0), F(0), "a" * 64),
            SpotCashTrade("buy2", 0, "BUY", F(".6"), F(60), F(0), F(0), "b" * 64),
            trades[1],
        )
    return SpotInventoryCashPath(
        "BTCUSDT", (10, 20), (F(100), F(120)), F(0), trades, "c" * 64
    )


def declaration(supplied, locks):
    return SpotQuoteReservations(
        replay_spot_inventory_cash(supplied).input_sha256,
        tuple(F(x) for x in locks),
        "d" * 64,
    )


def evaluate(supplied, locks, *, cash=F(0), buffer=F(0)):
    return evaluate_spot_reserved_cash(
        supplied,
        declaration(supplied, locks),
        initial_quote_cash=cash,
        required_quote_buffer=buffer,
    )


def test_open_order_peak_can_exceed_fill_only_cash_without_any_extra_fill():
    result = evaluate(path(), (160, 0, 0), cash=F(100))
    assert result.fill_only.supplied_cash_covers_every_prefix
    assert result.available_quote_lower_bound_by_prefix == (-60, 0, 120)
    assert result.minimum_initial_quote_cash == 160
    assert result.additional_quote_cash_required == 60
    assert result.reservation_increment_over_fill_only == 60
    assert result.binding_fill_prefix == 0
    assert not result.supplied_cash_covers_declared_locks_and_buffer


def test_partial_fill_reduces_remaining_lock_without_double_charging_original_order():
    result = evaluate(path(partial=True), (100, 60, 0, 0), cash=F(100))
    assert result.available_quote_lower_bound_by_prefix == (0, 0, 0, 120)
    assert result.minimum_initial_quote_cash == 100
    assert result.reservation_increment_over_fill_only == 0
    assert result.supplied_cash_covers_declared_locks_and_buffer


def test_locks_while_cash_is_unchanged_and_terminal_unresolved_locks_are_not_income():
    result = evaluate(path(empty=True), (75,), cash=F(50), buffer=F(5))
    assert result.minimum_available_quote == -25
    assert result.minimum_initial_quote_cash == 80
    assert result.fill_only.initial_marked_wealth == 50
    assert result.fill_only.quote_cash_after_boundary == (50, 50)
    assert result.additional_quote_cash_required == 30


def test_reserve_release_does_not_add_quote_cash_or_erase_a_prior_peak():
    result = evaluate(path(), (100, 80, 0), cash=F(180))
    assert result.fill_only.quote_cash_after_boundary == (80, 200)
    assert result.available_quote_lower_bound_by_prefix == (80, 0, 200)
    assert result.minimum_initial_quote_cash == 180
    assert result.binding_fill_prefix == 1


def test_quote_commission_is_cash_and_base_commission_is_not_a_second_quote_charge():
    result = evaluate(path(quote_fee=F(1), base_fee=F(".01")), (101, 20, 0))
    assert result.minimum_initial_quote_cash == 121
    assert result.fill_only.minimum_initial_quote_cash == 101
    assert result.available_quote_lower_bound_by_prefix == (-101, -121, 19)


@pytest.mark.parametrize("locks", [(0, 0, 0), (100, 40, 0), (60, 40, 150)])
def test_cash_shift_changes_free_cash_but_not_required_initial_capital(locks):
    low = evaluate(path(), locks, buffer=F(3))
    high = evaluate(path(), locks, cash=F(77), buffer=F(3))
    assert high.minimum_initial_quote_cash == low.minimum_initial_quote_cash
    assert high.available_quote_lower_bound_by_prefix == tuple(
        x + 77 for x in low.available_quote_lower_bound_by_prefix
    )


@pytest.mark.parametrize("supplied", [path(), path(partial=True), path(empty=True)])
def test_zero_declared_locks_reduce_exactly_to_existing_fill_only_constraint(supplied):
    result = evaluate(
        supplied, [0] * (len(supplied.trades) + 1), cash=F(17), buffer=F(2)
    )
    assert (
        result.minimum_initial_quote_cash == result.fill_only.minimum_initial_quote_cash
    )
    assert result.minimum_available_quote == result.fill_only.minimum_quote_cash
    assert result.reservation_increment_over_fill_only == 0


@pytest.mark.parametrize("bad", [True, 1, 1.0, Decimal(1), F(-1)])
def test_reserves_reject_coerced_or_negative_amounts(bad):
    with pytest.raises(ValueError, match="declaration is invalid"):
        SpotQuoteReservations("a" * 64, (bad,), "b" * 64)


@pytest.mark.parametrize("bad", [(), [F(1)], (None,)])
def test_missing_or_mutable_reserve_intervals_are_rejected(bad):
    with pytest.raises(ValueError, match="declaration is invalid"):
        SpotQuoteReservations("a" * 64, bad, "b" * 64)


def test_every_prefix_and_exact_spot_binding_are_required():
    supplied = path()
    for locks in [(0,), (0, 0), (0, 0, 0, 0)]:
        with pytest.raises(ValueError, match="every fill prefix"):
            evaluate(supplied, locks)
    wrong = replace(declaration(supplied, (0, 0, 0)), spot_input_sha256="e" * 64)
    with pytest.raises(ValueError, match="different spot path"):
        evaluate_spot_reserved_cash(
            supplied, wrong, initial_quote_cash=F(0), required_quote_buffer=F(0)
        )


def test_binding_covers_reserve_values_source_and_declared_cash():
    supplied = path()
    reserve = declaration(supplied, (100, 60, 0))

    def run(r=reserve, cash=F(0)):
        return evaluate_spot_reserved_cash(
            supplied, r, initial_quote_cash=cash, required_quote_buffer=F(0)
        )

    original = run()
    assert (
        original.input_sha256
        != run(replace(reserve, source_body_sha256="e" * 64)).input_sha256
    )
    assert (
        original.input_sha256
        != run(
            replace(reserve, maximum_locked_quote_by_prefix=(F(100), F(61), F(0)))
        ).input_sha256
    )
    assert original.input_sha256 != run(cash=F(1)).input_sha256
    assert not original.native_reservations_qualified
    assert not original.continuous_solvency_qualified
    assert not original.training_admitted and not original.qualified_edge
