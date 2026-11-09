from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as F

import pytest

from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_inventory import InventoryCashPath
from simple_ai_trading.hedged_inventory_cash import HedgedInventoryCashPath
from simple_ai_trading.spot_inventory_cash import SpotCashTrade, SpotInventoryCashPath
from simple_ai_trading.spot_prefunding import (
    evaluate_hedged_spot_prefunding,
    evaluate_spot_prefunding,
)


def trade(name, boundary, side, quantity, quote, *, base_fee="0", quote_fee="0"):
    return SpotCashTrade(
        name, boundary, side, F(quantity), F(quote), F(base_fee), F(quote_fee), "a" * 64
    )


def path(trades=None, initial=F(0)):
    return SpotInventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (F(100), F(120)),
        initial,
        (trade("buy", 0, "BUY", "1", "100"), trade("sell", 1, "SELL", "1", "120"))
        if trades is None
        else tuple(trades),
        "b" * 64,
    )


def evaluate(supplied=None, cash=F(0), buffer=F(0)):
    return evaluate_spot_prefunding(
        path() if supplied is None else supplied,
        initial_quote_cash=cash,
        required_quote_buffer=buffer,
    )


def test_profitable_end_does_not_pay_for_initial_purchase():
    result = evaluate()
    assert result.quote_cash_after_boundary == (-100, 20)
    assert result.minimum_quote_cash == -100
    assert result.minimum_cash_execution_id == "buy"
    assert (
        result.minimum_initial_quote_cash
        == result.additional_quote_cash_required
        == 100
    )
    assert not result.supplied_cash_covers_every_prefix
    funded = evaluate(cash=F(100))
    assert funded.quote_cash_after_boundary == (0, 120)
    assert funded.minimum_initial_quote_cash == 100
    assert funded.supplied_cash_covers_every_prefix
    assert funded.additional_quote_cash_required == 0
    assert not funded.capital_source_qualified and not funded.venue_margin_qualified
    assert not funded.qualified_edge


def test_same_boundary_net_cash_hides_intermediate_funding():
    result = evaluate(
        path(
            [
                trade("buy", 0, "BUY", "1", "100"),
                trade("sell", 0, "SELL", "1", "100"),
            ]
        )
    )
    assert result.quote_cash_after_boundary == (0, 0)
    assert result.minimum_quote_cash == -100
    assert result.minimum_initial_quote_cash == 100


def test_incumbent_proceeds_and_fill_order_preserve_capital_comparator():
    buy, sell = trade("buy", 0, "BUY", "1", "100"), trade("sell", 0, "SELL", "1", "100")
    final = trade("final", 1, "SELL", "1", "120")
    buy_first = evaluate(path([buy, sell, final], initial=F(1)))
    sell_first = evaluate(path([sell, buy, final], initial=F(1)))
    assert (
        buy_first.quote_cash_after_boundary
        == sell_first.quote_cash_after_boundary
        == (0, 120)
    )
    assert buy_first.minimum_initial_quote_cash == 100
    assert sell_first.minimum_initial_quote_cash == 0
    assert buy_first.initial_marked_wealth == sell_first.initial_marked_wealth == 100
    assert (
        evaluate(
            path([sell, buy, final], initial=F(1)), cash=F(7)
        ).initial_marked_wealth
        == 107
    )


def test_quote_fees_consume_cash_but_base_fee_is_not_charged_twice():
    result = evaluate(
        path(
            [
                trade("buy", 0, "BUY", "1", "100", base_fee=".01", quote_fee="1"),
                trade("sell", 1, "SELL", ".99", "118.8"),
            ]
        )
    )
    assert result.minimum_initial_quote_cash == 101
    assert result.quote_cash_after_boundary == (-101, F("17.8"))


def test_partial_fills_accumulate_quote_fees_before_any_sale():
    result = evaluate(
        path(
            [
                trade("first", 0, "BUY", ".4", "40", quote_fee=".4"),
                trade("second", 0, "BUY", ".6", "60", quote_fee=".6"),
                trade("close", 1, "SELL", "1", "120"),
            ]
        )
    )
    assert result.minimum_cash_execution_id == "second"
    assert result.minimum_initial_quote_cash == 101


def test_buffer_includes_initial_cash_even_if_the_first_fill_raises_cash():
    supplied = path([trade("sell", 0, "SELL", "1", "100")], initial=F(1))
    insufficient = evaluate(supplied, buffer=F(5))
    assert insufficient.minimum_cash_execution_id is None
    assert insufficient.minimum_quote_cash == 0
    assert insufficient.minimum_initial_quote_cash == 5
    assert not insufficient.supplied_cash_covers_every_prefix
    assert evaluate(supplied, cash=F(5), buffer=F(5)).supplied_cash_covers_every_prefix


def test_flat_abstention_needs_only_its_declared_cash_buffer():
    result = evaluate(path([]))
    assert result.minimum_initial_quote_cash == 0
    assert result.supplied_cash_covers_every_prefix
    assert not evaluate(path([]), buffer=F(1)).supplied_cash_covers_every_prefix


@pytest.mark.parametrize("bad", [True, 1, 1.0, Decimal(1), F(-1)])
@pytest.mark.parametrize("field", ["cash", "buffer"])
def test_cash_inputs_reject_coercion_or_borrowed_negative_cash(field, bad):
    with pytest.raises(ValueError, match="nonnegative Fractions"):
        evaluate(**{field: bad})


def test_prefunding_keeps_existing_inventory_validity_checks():
    with pytest.raises(ValueError, match="terminal spot inventory"):
        evaluate(path([trade("buy", 0, "BUY", "1", "100")]))
    with pytest.raises(ValueError, match="exceeds supplied net"):
        evaluate(path([trade("sell", 0, "SELL", "1", "100")]))


def test_prefunding_binding_covers_cash_buffer_and_fill_provenance():
    original = evaluate()
    assert original.input_sha256 != evaluate(cash=F(1)).input_sha256
    assert original.input_sha256 != evaluate(buffer=F(1)).input_sha256
    changed = replace(
        path(),
        trades=(
            replace(path().trades[0], source_body_sha256="c" * 64),
            path().trades[1],
        ),
    )
    assert original.input_sha256 != evaluate(changed).input_sha256


def test_paired_funding_income_cannot_finance_a_spot_wallet_purchase():
    event = LinearFundingSettlement("BTCUSDT", "USDT", 15, F(".1"), F(110), "d" * 64)
    perpetual = InventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (F(101), F(121)),
        (-1,),
        F(100),
        F(0),
        "e" * 64,
        (event,),
        ((15, F(".1")),),
        0,
        30,
        "f" * 64,
        target_signed_base_quantity=(-F(1),),
    )
    paired = HedgedInventoryCashPath(
        perpetual, path(), (F(0),), (F(5),), (F(0),), "1" * 64
    )
    result = evaluate_hedged_spot_prefunding(
        paired, initial_spot_quote_cash=F(0), required_spot_quote_buffer=F(0)
    )
    assert result.cash_replay.total_net_cash_lower == 6
    assert result.spot_funding.minimum_initial_quote_cash == 100
    assert not result.spot_funding.supplied_cash_covers_every_prefix
    assert not result.perpetual_margin_assessed
    assert not result.continuous_solvency_qualified and not result.training_admitted
    assert not result.qualified_edge
    changed = evaluate_hedged_spot_prefunding(
        paired, initial_spot_quote_cash=F(100), required_spot_quote_buffer=F(0)
    )
    assert changed.spot_funding.supplied_cash_covers_every_prefix
    assert result.input_sha256 != changed.input_sha256
