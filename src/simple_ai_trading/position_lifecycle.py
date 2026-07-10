"""Shared position-exit decisions for backtest, paper, and live execution."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionExitDecision:
    """Deterministic exit decision for an already-open position."""

    should_close: bool
    reason: str
    flat_signal_streak: int
    bars_held: int
    reverse_signal: bool
    time_limit_reached: bool


def _finite_nonnegative(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, parsed) if math.isfinite(parsed) else 0.0


def _finite(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def evaluate_position_exit(
    *,
    position_side: int,
    signal_direction: int,
    current_pnl_pct: float,
    bars_held: int,
    flat_signal_streak: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    min_position_hold_bars: int = 0,
    flat_signal_exit_grace_bars: int = 0,
    max_position_hold_bars: int = 0,
    allow_flat_signal_exit: bool = True,
) -> PositionExitDecision:
    """Evaluate one open-position observation with a shared exit contract."""

    side = 1 if int(position_side) > 0 else (-1 if int(position_side) < 0 else 0)
    signal = 1 if int(signal_direction) > 0 else (-1 if int(signal_direction) < 0 else 0)
    held = max(0, int(bars_held))
    streak = max(0, int(flat_signal_streak))
    streak = streak + 1 if signal == 0 else 0
    stop = _finite_nonnegative(stop_loss_pct)
    take = _finite_nonnegative(take_profit_pct)
    pnl = _finite(current_pnl_pct)
    reverse = side != 0 and signal == -side
    max_hold = max(0, int(max_position_hold_bars))
    time_limit = side != 0 and max_hold > 0 and held >= max_hold

    reason = ""
    if side != 0 and take > 0.0 and pnl >= take:
        reason = "take_profit_close"
    elif side != 0 and stop > 0.0 and pnl <= -stop:
        reason = "stop_loss_close"
    elif reverse:
        reason = "signal_reverse"
    elif time_limit:
        reason = "time_limit"
    elif (
        side != 0
        and allow_flat_signal_exit
        and signal == 0
        and held >= max(0, int(min_position_hold_bars))
        and streak > max(0, int(flat_signal_exit_grace_bars))
    ):
        reason = "signal_flat"

    return PositionExitDecision(
        should_close=bool(reason),
        reason=reason,
        flat_signal_streak=streak,
        bars_held=held,
        reverse_signal=reverse,
        time_limit_reached=time_limit,
    )


__all__ = ["PositionExitDecision", "evaluate_position_exit"]
