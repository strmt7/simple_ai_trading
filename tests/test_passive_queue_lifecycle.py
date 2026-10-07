"""Virtual FIFO quantities remain coherent after completion and at exact clocks."""

from dataclasses import replace
from decimal import Decimal

import pytest

from simple_ai_trading.paper_execution import (
    AggressiveTradePrint,
    PassiveQueueState,
    apply_passive_trade_print,
)


def _state(side: str = "BUY") -> PassiveQueueState:
    return PassiveQueueState(
        intent_id="virtual-marker",
        asset_id="BTCUSDT",
        side=side,
        price=Decimal("100"),
        queue_ahead_quantity=Decimal("5"),
        remaining_quantity=Decimal("3"),
        filled_quantity=Decimal("0"),
        activated_at_ms=1000,
        expires_at_ms=2000,
    )


def _trade(
    side: str = "SELL", quantity: str = "8", time: int = 1100
) -> AggressiveTradePrint:
    return AggressiveTradePrint(
        asset_id="BTCUSDT",
        side=side,
        price=Decimal("100"),
        quantity=Decimal(quantity),
        occurred_at_ms=time,
        source_payload_sha256="a" * 64,
    )


@pytest.mark.parametrize("maker,taker", [("BUY", "SELL"), ("SELL", "BUY")])
def test_completed_marker_remains_valid_and_later_print_cannot_overfill(maker, taker):
    initial = _state(maker)
    complete, fill = apply_passive_trade_print(initial, _trade(taker))
    assert fill == Decimal("3")
    assert complete.remaining_quantity == 0
    assert complete.filled_quantity == 3
    assert complete.validated() == complete
    after, extra_fill = apply_passive_trade_print(complete, _trade(taker, "100", 1200))
    assert after == complete
    assert extra_fill == 0
    assert initial.remaining_quantity == 3


@pytest.mark.parametrize("value", [True, 1000.5, "1000", None])
@pytest.mark.parametrize("field", ["activated_at_ms", "expires_at_ms"])
def test_marker_rejects_lossy_or_ambiguous_timestamp_coercion(field, value):
    with pytest.raises(ValueError):
        replace(_state(), **{field: value}).validated()


@pytest.mark.parametrize("value", [True, 1100.5, "1100", None, -1])
def test_print_rejects_lossy_or_ambiguous_timestamp_coercion(value):
    with pytest.raises(ValueError):
        apply_passive_trade_print(_state(), replace(_trade(), occurred_at_ms=value))


def test_zero_total_or_negative_quantity_state_is_not_a_completed_marker():
    for field, value in (
        ("remaining_quantity", Decimal("0")),
        ("remaining_quantity", Decimal("-1")),
        ("filled_quantity", Decimal("-1")),
        ("queue_ahead_quantity", Decimal("-1")),
    ):
        with pytest.raises(ValueError):
            replace(_state(), **{field: value}).validated()


def test_completion_does_not_turn_invalid_print_into_valid_evidence():
    complete = replace(
        _state(),
        queue_ahead_quantity=Decimal("0"),
        remaining_quantity=Decimal("0"),
        filled_quantity=Decimal("3"),
    )
    with pytest.raises(ValueError):
        apply_passive_trade_print(
            complete, replace(_trade(), source_payload_sha256="bad")
        )


def test_exact_activation_and_expiry_boundaries_preserve_declared_behavior():
    state = _state()
    unchanged, fill = apply_passive_trade_print(state, _trade(time=1000))
    assert unchanged == state and fill == 0
    expired, fill = apply_passive_trade_print(state, _trade(time=2001))
    assert expired == state and fill == 0
    complete, fill = apply_passive_trade_print(state, _trade(time=2000))
    assert fill == 3 and complete.remaining_quantity == 0


def test_identical_aggregate_path_can_support_distinct_virtual_fills():
    # Ten older units precede the marker; ten newer units later join behind it.
    # Canceling either background cohort leaves the same displayed ten units.
    initial = replace(
        _state(), queue_ahead_quantity=Decimal("10"), remaining_quantity=Decimal("1")
    )
    front_background = {"ahead": Decimal("0"), "behind": Decimal("10")}
    back_background = {"ahead": Decimal("10"), "behind": Decimal("0")}
    assert sum(front_background.values()) == sum(back_background.values()) == 10
    trade = _trade(quantity="1", time=1400)
    front, front_fill = apply_passive_trade_print(
        replace(initial, queue_ahead_quantity=front_background["ahead"]), trade
    )
    back, back_fill = apply_passive_trade_print(
        replace(initial, queue_ahead_quantity=back_background["ahead"]), trade
    )
    # The shadow marker never diverts the common background execution.
    front_background["behind"] -= trade.quantity
    back_background["ahead"] -= trade.quantity
    assert sum(front_background.values()) == sum(back_background.values()) == 9
    assert front_fill == 1 and front.remaining_quantity == 0
    assert back_fill == 0 and back.remaining_quantity == 1
    assert front.validated() == front and back.validated() == back
