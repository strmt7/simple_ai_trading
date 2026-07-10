from __future__ import annotations

import pytest

from simple_ai_trading.execution_simulation import (
    ExecutionAssumptions,
    simulate_market_fill,
)
from simple_ai_trading.types import StrategyConfig


def _assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(
        spread_bps=0.5,
        latency_ms=750,
        liquidity_haircut=0.5,
        impact_coefficient=18.0,
        volatility_buffer_bps=2.5,
        testnet_to_live_buffer_bps=2.0,
    )


def test_market_impact_uses_adv_participation_when_available() -> None:
    fill = simulate_market_fill(
        60_000.0,
        1,
        200.0,
        StrategyConfig(),
        bar_volume_notional=0.0,
        daily_volume_notional=9_000_000_000.0,
        assumptions=_assumptions(),
    )

    expected_participation = 200.0 / (9_000_000_000.0 * 0.5)
    assert fill.impact_cost_bps == pytest.approx(18.0 * expected_participation**0.5)
    assert fill.impact_cost_bps < 0.01


def test_missing_adv_keeps_fail_closed_bar_volume_fallback() -> None:
    missing = simulate_market_fill(
        60_000.0,
        1,
        200.0,
        StrategyConfig(),
        bar_volume_notional=0.0,
        assumptions=_assumptions(),
    )
    observed = simulate_market_fill(
        60_000.0,
        1,
        200.0,
        StrategyConfig(),
        bar_volume_notional=20_000.0,
        assumptions=_assumptions(),
    )

    assert missing.impact_cost_bps == pytest.approx(18.0)
    assert observed.impact_cost_bps == pytest.approx(1.8)
