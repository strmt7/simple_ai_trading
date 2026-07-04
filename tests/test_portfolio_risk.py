from __future__ import annotations

import math

from simple_ai_trading.api import Candle
from simple_ai_trading.portfolio_risk import build_portfolio_risk_report, policy_for_strategy
from simple_ai_trading.types import StrategyConfig


def _candles(symbol_offset: float = 0.0, *, count: int = 120, shock: float = 0.0) -> list[Candle]:
    price = 100.0 + symbol_offset
    rows: list[Candle] = []
    for index in range(count):
        move = 0.0008 * math.sin(index / 5.0 + symbol_offset) + 0.0003
        if shock and index == count - 12:
            move -= shock
        price = max(1.0, price * (1.0 + move))
        rows.append(Candle(
            open_time=index * 60_000,
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=1000.0,
            close_time=index * 60_000 + 59_000,
        ))
    return rows


def test_policy_for_strategy_is_stricter_for_conservative() -> None:
    conservative = policy_for_strategy(StrategyConfig(risk_level="conservative"), min_symbols=3)
    aggressive = policy_for_strategy(StrategyConfig(risk_level="aggressive"), min_symbols=3)

    assert conservative.max_cluster_weight < aggressive.max_cluster_weight
    assert conservative.max_portfolio_cvar_95 < aggressive.max_portfolio_cvar_95
    assert conservative.min_effective_symbols > aggressive.min_effective_symbols


def test_portfolio_risk_accepts_diversified_low_tail_risk_set() -> None:
    strategy = StrategyConfig(
        min_diversified_assets=3,
        max_asset_allocation_pct=0.34,
        max_portfolio_risk_pct=0.03,
        max_drawdown_limit=0.20,
    )

    report = build_portfolio_risk_report(
        {
            "AAAUSDC": _candles(0.0),
            "BBBUSDC": _candles(1.7),
            "CCCUSDC": _candles(3.1),
        },
        strategy,
        min_symbols=3,
    )

    assert report.accepted is True
    assert report.reason is None
    assert report.observations >= 100
    assert report.effective_symbol_count >= 2.9
    assert report.portfolio_cvar_95 <= report.policy.max_portfolio_cvar_95
    assert set(report.weights) == {"AAAUSDC", "BBBUSDC", "CCCUSDC"}


def test_portfolio_risk_rejects_single_symbol_when_diversification_required() -> None:
    report = build_portfolio_risk_report(
        {"AAAUSDC": _candles()},
        StrategyConfig(min_diversified_assets=3),
        min_symbols=3,
    )

    assert report.accepted is False
    assert "symbols<3" in str(report.reason)
    assert report.accepted_symbols == []


def test_portfolio_risk_rejects_high_correlation_cluster_weight() -> None:
    shared = _candles(0.0)
    report = build_portfolio_risk_report(
        {
            "AAAUSDC": shared,
            "BBBUSDC": list(shared),
            "CCCUSDC": list(shared),
        },
        StrategyConfig(min_diversified_assets=3, max_asset_allocation_pct=0.20),
        min_symbols=3,
    )

    assert report.accepted is False
    assert "cluster_weight>" in str(report.reason)
    assert report.max_pairwise_correlation >= 0.99
    assert report.max_cluster_weight > report.policy.max_cluster_weight


def test_portfolio_risk_rejects_tail_loss_breach() -> None:
    strategy = StrategyConfig(
        risk_level="conservative",
        min_diversified_assets=3,
        max_asset_allocation_pct=0.34,
        max_portfolio_risk_pct=0.004,
        max_drawdown_limit=0.50,
    )

    report = build_portfolio_risk_report(
        {
            "AAAUSDC": _candles(0.0, shock=0.20),
            "BBBUSDC": _candles(2.0, shock=0.18),
            "CCCUSDC": _candles(4.0, shock=0.16),
        },
        strategy,
        min_symbols=3,
    )

    assert report.accepted is False
    assert "cvar95>" in str(report.reason)
    assert report.portfolio_cvar_95 > report.policy.max_portfolio_cvar_95

