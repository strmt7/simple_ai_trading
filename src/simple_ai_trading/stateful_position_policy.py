"""Shared position decisions, separate from return and inventory accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PositionSchedule:
    positions: np.ndarray
    ages: np.ndarray
    transitions: np.ndarray
    reasons: np.ndarray
    final_transition: np.ndarray
    holding_durations: tuple[int, ...]


def stateful_position_schedule(
    forecasts: np.ndarray,
    *,
    mode: str,
    cost_bps: float,
    maximum_holding_hours: int,
    cost_filter_multiplier: float,
) -> PositionSchedule:
    """Apply the existing forecast/turnover policy without reading label outcomes."""
    if (
        forecasts.ndim != 2
        or min(forecasts.shape) <= 0
        or not np.isfinite(forecasts).all()
        or mode not in ("long_only", "long_short")
        or not math.isfinite(cost_bps)
        or cost_bps <= 0
        or type(maximum_holding_hours) is not int
        or not 0 < maximum_holding_hours < 32_767
        or not math.isfinite(cost_filter_multiplier)
        or cost_filter_multiplier <= 0
    ):
        raise ValueError("stateful position policy contract is invalid")
    hours, symbols = forecasts.shape
    positions = np.zeros((hours, symbols), dtype=np.int8)
    ages = np.zeros((hours, symbols), dtype=np.int16)
    transitions = np.zeros((hours, symbols), dtype=np.float64)
    reasons = np.zeros((hours, symbols), dtype=np.int8)
    current = np.zeros(symbols, dtype=np.int8)
    current_age = np.zeros(symbols, dtype=np.int16)
    holding_durations: list[int] = []
    for hour in range(hours):
        forecast = forecasts[hour]
        proposed = (
            np.where(forecast > 0.0, 1, 0).astype(np.int8)
            if mode == "long_only"
            else np.sign(forecast).astype(np.int8)
        )
        next_position = current.copy()
        for symbol_index in range(symbols):
            previous = int(current[symbol_index])
            desired = int(proposed[symbol_index])
            if (
                previous != 0
                and int(current_age[symbol_index]) >= maximum_holding_hours
            ):
                next_position[symbol_index] = 0
                reasons[hour, symbol_index] = 4
                holding_durations.append(int(current_age[symbol_index]))
                continue
            if desired == previous:
                continue
            units = abs(desired - previous)
            hurdle = cost_filter_multiplier * cost_bps * units
            if abs(float(forecast[symbol_index])) <= hurdle:
                continue
            next_position[symbol_index] = desired
            if previous == 0 and desired != 0:
                reasons[hour, symbol_index] = 1
            elif previous != 0 and desired == 0:
                reasons[hour, symbol_index] = 2
                holding_durations.append(int(current_age[symbol_index]))
            else:
                reasons[hour, symbol_index] = 3
                holding_durations.append(int(current_age[symbol_index]))
        transitions[hour] = np.abs(next_position - current).astype(np.float64)
        positions[hour] = next_position
        for symbol_index in range(symbols):
            if next_position[symbol_index] == 0:
                current_age[symbol_index] = 0
            elif next_position[symbol_index] == current[symbol_index]:
                current_age[symbol_index] += 1
            else:
                current_age[symbol_index] = 1
        ages[hour] = current_age
        current = next_position
    final_transition = np.abs(current).astype(np.float64)
    for symbol_index in np.flatnonzero(final_transition > 0):
        holding_durations.append(int(current_age[symbol_index]))
    return PositionSchedule(
        positions,
        ages,
        transitions,
        reasons,
        final_transition,
        tuple(holding_durations),
    )
