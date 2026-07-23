from __future__ import annotations

from decimal import Decimal
import math

import pytest

from simple_ai_trading.impact_absorption_targets import (
    ROUND73_TARGET_CONTRACT_SHA256,
    ROUND73_TARGET_MINIMUM_CHARGE_BPS,
    Round73MarketQuantityRules,
    round73_target_payoff,
    walk_round73_book,
)


def _rules(symbol: str = "BTCUSDT") -> Round73MarketQuantityRules:
    return Round73MarketQuantityRules.create(
        symbol=symbol,
        step_size="0.001",
        minimum_quantity="0.001",
        maximum_quantity="120",
        minimum_notional="50",
    )


def _asks() -> tuple[tuple[float, float], ...]:
    return tuple((100.1 + index * 0.1, 0.4) for index in range(20))


def _bids() -> tuple[tuple[float, float], ...]:
    return tuple((99.9 - index * 0.1, 0.4) for index in range(20))


def test_target_contract_digest_is_frozen() -> None:
    assert ROUND73_TARGET_CONTRACT_SHA256 == (
        "3c4c6f0f10505897f71930245ae3252eccfe1ccac90217e6c07151a2416e2253"
    )


def test_quantity_is_floored_to_captured_market_step() -> None:
    rules = _rules()

    quantity = rules.quantize_reference_quantity(
        reference_quote_notional=1_000.0,
        decision_mid=118_123.45,
    )

    assert quantity == 0.008
    assert rules.is_step_aligned(quantity)
    assert Decimal(str(quantity)) * Decimal("118123.45") <= Decimal("1000")


def test_quantity_below_market_minimum_is_ineligible_without_rounding_up() -> None:
    assert (
        _rules().quantize_reference_quantity(
            reference_quote_notional=100.0,
            decision_mid=118_123.45,
        )
        is None
    )


def test_book_walk_uses_exact_quantity_and_visible_levels() -> None:
    walk = walk_round73_book(_asks(), base_quantity=0.9, ascending_prices=True)

    assert walk is not None
    assert walk.filled_base_quantity == pytest.approx(0.9)
    assert walk.quote_notional == pytest.approx(0.4 * 100.1 + 0.4 * 100.2 + 0.1 * 100.3)
    assert walk.average_price == pytest.approx(walk.quote_notional / 0.9)
    assert walk.worst_price == pytest.approx(100.3)
    assert walk.level_count == 3
    assert walk.capacity_ratio > 1.0


def test_book_walk_rejects_partial_capacity_and_invalid_geometry() -> None:
    assert walk_round73_book(_bids(), base_quantity=9.0, ascending_prices=False) is None
    with pytest.raises(ValueError, match="ask ladder"):
        walk_round73_book(tuple(reversed(_asks())), base_quantity=0.1, ascending_prices=True)


def test_long_and_short_payoffs_are_symmetric_before_cost() -> None:
    long_result = round73_target_payoff(
        side="long",
        entry_quote_notional=1_000.0,
        exit_quote_notional=1_010.0,
    )
    short_result = round73_target_payoff(
        side="short",
        entry_quote_notional=1_010.0,
        exit_quote_notional=1_000.0,
    )

    assert long_result.gross_payoff_quote == 10.0
    assert short_result.gross_payoff_quote == 10.0
    assert long_result.charge_quote == pytest.approx(
        ROUND73_TARGET_MINIMUM_CHARGE_BPS / 10_000 * 1_005.0
    )
    assert short_result.charge_quote == pytest.approx(long_result.charge_quote)
    assert long_result.net_payoff_quote == pytest.approx(short_result.net_payoff_quote)


def test_cost_can_turn_a_small_gross_move_into_a_negative_target() -> None:
    result = round73_target_payoff(
        side="long",
        entry_quote_notional=1_000.0,
        exit_quote_notional=1_000.5,
    )

    assert result.gross_payoff_quote > 0.0
    assert result.net_payoff_quote < 0.0
    assert result.positive_net_payoff is False
    assert math.isfinite(result.net_payoff_bps)


def test_payoff_rejects_subcontract_cost_or_invalid_side() -> None:
    with pytest.raises(ValueError, match="payoff input"):
        round73_target_payoff(
            side="long",
            entry_quote_notional=1_000.0,
            exit_quote_notional=1_001.0,
            charge_bps=11.99,
        )
    with pytest.raises(ValueError, match="side"):
        round73_target_payoff(
            side="flat",  # type: ignore[arg-type]
            entry_quote_notional=1_000.0,
            exit_quote_notional=1_001.0,
        )
