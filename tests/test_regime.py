from __future__ import annotations

from simple_ai_trading.features import ModelRow
from simple_ai_trading.regime import classify_market_regime, summarize_regime_windows


def _rows(prices: list[float]) -> list[ModelRow]:
    return [
        ModelRow(timestamp=index * 60_000, close=price, features=(0.0,), label=1)
        for index, price in enumerate(prices)
    ]


def test_classify_market_regime_detects_trend_up() -> None:
    evidence = classify_market_regime(_rows([100.0, 101.0, 102.0, 103.0, 104.5, 106.0]))

    assert evidence.dominant_regime == "trend_up"
    assert evidence.confidence > 0.5
    assert evidence.trend_return > 0.05
    assert evidence.start_timestamp == 0


def test_classify_market_regime_detects_volatile_chop() -> None:
    evidence = classify_market_regime(_rows([100.0, 103.0, 97.0, 104.0, 96.0, 105.0, 95.0]))

    assert evidence.dominant_regime in {"volatile_chop", "range_bound"}
    assert evidence.reversal_rate >= 0.5
    assert evidence.realized_volatility > 0.0


def test_summarize_regime_windows_flags_concentration() -> None:
    summary = summarize_regime_windows(
        [
            {
                "regime": {"dominant_regime": "trend_up", "confidence": 0.8},
                "accepted": True,
                "result": {"realized_pnl": 10.0, "max_drawdown": 0.01, "profit_factor": 1.4, "expectancy": 2.0},
            },
            {
                "regime": {"dominant_regime": "trend_up", "confidence": 0.7},
                "accepted": True,
                "result": {"realized_pnl": 8.0, "max_drawdown": 0.02, "profit_factor": 1.2, "expectancy": 1.5},
            },
            {
                "regime": {"dominant_regime": "trend_up", "confidence": 0.6},
                "accepted": False,
                "result": {"realized_pnl": -3.0, "max_drawdown": 0.04, "profit_factor": 0.8, "expectancy": -1.0},
            },
        ],
        overall_regime={"dominant_regime": "trend_up"},
    )

    assert summary["dominant_regime"] == "trend_up"
    assert summary["concentration_warning"] is True
    assert summary["by_regime"]["trend_up"]["accepted_windows"] == 2
    assert summary["by_regime"]["trend_up"]["realized_pnl"] == 15.0
