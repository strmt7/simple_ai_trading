from __future__ import annotations

import pytest

from simple_ai_trading.position_lifecycle import evaluate_position_exit


def test_time_barrier_holds_through_neutral_signals_then_closes() -> None:
    before = evaluate_position_exit(
        position_side=1,
        signal_direction=0,
        current_pnl_pct=0.0002,
        bars_held=14,
        flat_signal_streak=13,
        stop_loss_pct=0.001,
        take_profit_pct=0.002,
        min_position_hold_bars=1,
        flat_signal_exit_grace_bars=15,
        max_position_hold_bars=15,
    )
    at_barrier = evaluate_position_exit(
        position_side=1,
        signal_direction=1,
        current_pnl_pct=0.0002,
        bars_held=15,
        flat_signal_streak=14,
        stop_loss_pct=0.001,
        take_profit_pct=0.002,
        min_position_hold_bars=1,
        flat_signal_exit_grace_bars=15,
        max_position_hold_bars=15,
    )

    assert before.should_close is False
    assert before.flat_signal_streak == 14
    assert at_barrier.should_close is True
    assert at_barrier.reason == "time_limit"
    assert at_barrier.time_limit_reached is True


@pytest.mark.parametrize(
    ("pnl", "signal", "reason"),
    [
        (-0.02, 1, "stop_loss_close"),
        (0.03, 1, "take_profit_close"),
        (0.0, -1, "signal_reverse"),
    ],
)
def test_hard_barriers_and_reverse_signal_can_close_before_time_limit(
    pnl: float,
    signal: int,
    reason: str,
) -> None:
    decision = evaluate_position_exit(
        position_side=1,
        signal_direction=signal,
        current_pnl_pct=pnl,
        bars_held=1,
        flat_signal_streak=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        max_position_hold_bars=15,
    )

    assert decision.should_close is True
    assert decision.reason == reason


def test_legacy_flat_exit_and_disabled_flat_exit_are_explicit() -> None:
    enabled = evaluate_position_exit(
        position_side=-1,
        signal_direction=0,
        current_pnl_pct=0.0,
        bars_held=3,
        flat_signal_streak=2,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=2,
    )
    disabled = evaluate_position_exit(
        position_side=-1,
        signal_direction=0,
        current_pnl_pct=float("nan"),
        bars_held=3,
        flat_signal_streak=2,
        stop_loss_pct=-1.0,
        take_profit_pct=float("inf"),
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=2,
        allow_flat_signal_exit=False,
    )

    assert enabled.reason == "signal_flat"
    assert disabled.should_close is False
    assert disabled.flat_signal_streak == 3
