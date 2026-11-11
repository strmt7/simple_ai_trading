"""Exit impact must follow the quantity actually held at the exit reference."""

import math

import pytest

from simple_ai_trading.backtest import _close_position
from simple_ai_trading.execution_simulation import (
    ExecutionAssumptions,
    fixed_base_quote_notional,
)
from simple_ai_trading.features import ModelRow
from simple_ai_trading import hybrid_models
from simple_ai_trading.types import StrategyConfig


ASSUMPTIONS = ExecutionAssumptions(
    spread_bps=0.0,
    latency_ms=0,
    liquidity_haircut=0.0,
    impact_coefficient=100.0,
    volatility_buffer_bps=0.0,
    testnet_to_live_buffer_bps=0.0,
)
ADV = 1_000_000.0
ENTRY_NOTIONAL = 1_000.0


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("exit_reference", [50.0, 200.0])
def test_backtest_exit_impact_prices_the_held_base_quantity(side, exit_reference):
    quantity = 10.0
    impact_bps = 100.0 * math.sqrt(quantity * exit_reference / ADV)
    expected_fill = exit_reference * (1.0 - side * impact_bps / 10_000.0)
    realized = side * (expected_fill - 100.0) * quantity
    fee = expected_fill * quantity * 2.0 / 10_000.0

    cash, actual_realized, actual_fee, actual_impact, actual_fill = _close_position(
        side,
        exit_reference,
        100.0,
        quantity,
        200.0,
        StrategyConfig(taker_fee_bps=2.0),
        fill_volume_notional=100_000.0,
        daily_volume_notional=ADV,
        execution_assumptions=ASSUMPTIONS,
    )

    assert actual_impact == pytest.approx(impact_bps)
    assert actual_fill == pytest.approx(expected_fill)
    assert actual_realized == pytest.approx(realized)
    assert actual_fee == pytest.approx(fee)
    assert cash == pytest.approx(200.0 + realized - fee)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("exit_reference", [50.0, 200.0])
def test_payoff_label_exit_impact_uses_post_entry_base_quantity(
    monkeypatch, side, exit_reference
):
    strategy = StrategyConfig(taker_fee_bps=2.0, stop_loss_pct=0.0, take_profit_pct=0.0)
    rows = [
        ModelRow(
            timestamp=index * 1_000,
            close=price,
            quote_volume=100_000.0,
            features=(0.0,),
            label=0,
        )
        for index, price in enumerate((100.0, 100.0, exit_reference))
    ]
    monkeypatch.setattr(
        hybrid_models, "_reference_notional", lambda *a, **k: ENTRY_NOTIONAL
    )
    monkeypatch.setattr(
        hybrid_models, "market_row_execution_assumptions", lambda *a, **k: ASSUMPTIONS
    )
    monkeypatch.setattr(
        hybrid_models, "market_row_trailing_quote_volume_24h_estimate", lambda row: ADV
    )
    entry_impact = 100.0 * math.sqrt(ENTRY_NOTIONAL / ADV)
    entry_fill = 100.0 * (1.0 + side * entry_impact / 10_000.0)
    quantity = ENTRY_NOTIONAL / entry_fill
    exit_impact = 100.0 * math.sqrt(quantity * exit_reference / ADV)
    exit_fill = exit_reference * (1.0 - side * exit_impact / 10_000.0)
    realized = side * (exit_fill - entry_fill) * quantity
    fees = (ENTRY_NOTIONAL + exit_fill * quantity) * 2.0 / 10_000.0
    expected_net_bps = (realized - fees) / ENTRY_NOTIONAL * 10_000.0

    actual = hybrid_models.path_net_edge_bps(
        rows,
        signal_index=0,
        horizon_bars=1,
        side=side,
        strategy=strategy,
        market_type="futures",
        symbol_profile=None,
    )

    assert actual == pytest.approx(expected_net_bps, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("quantity", [10.0, -10.0])
def test_quote_notional_preserves_signed_base_magnitude(quantity):
    assert fixed_base_quote_notional(quantity, 50.0) == 500.0
    assert fixed_base_quote_notional(quantity, 200.0) == 2000.0


@pytest.mark.parametrize(
    ("quantity", "reference"),
    [
        (0.0, 100.0),
        (True, 100.0),
        (10.0, False),
        (float("nan"), 100.0),
        (10.0, float("inf")),
        (float("inf"), 100.0),
        (10.0, 0.0),
        (10.0, -100.0),
        (10.0, "invalid"),
        (None, 100.0),
        (10**400, 100.0),
        (1e308, 100.0),
        (1e-308, 1e-308),
    ],
)
def test_invalid_fixed_base_notional_is_not_zero_cost_evidence(quantity, reference):
    with pytest.raises(ValueError, match="finite"):
        fixed_base_quote_notional(quantity, reference)


def test_overflowing_payoff_exit_is_unavailable_not_a_training_target(monkeypatch):
    rows = [
        ModelRow(timestamp=index * 1000, close=price, features=(0.0,), label=0)
        for index, price in enumerate((100.0, 100.0, 200.0))
    ]
    monkeypatch.setattr(hybrid_models, "_reference_notional", lambda *a, **k: 1e308)
    assert (
        hybrid_models.path_net_edge_bps(
            rows,
            signal_index=0,
            horizon_bars=1,
            side=1,
            strategy=StrategyConfig(stop_loss_pct=0.0, take_profit_pct=0.0),
            market_type="futures",
            symbol_profile=None,
        )
        is None
    )
