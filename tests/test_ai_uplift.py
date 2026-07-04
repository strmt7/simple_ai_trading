from __future__ import annotations

from simple_ai_trading.ai_uplift import AIUpliftPolicy, assess_ai_uplift
from simple_ai_trading.ai_runtime import estimate_model_parameters_b


def test_estimate_model_parameters_from_local_model_names() -> None:
    assert estimate_model_parameters_b("qwen2.5:7b") == 7.0
    assert estimate_model_parameters_b("tiny-560m") == 0.56
    assert estimate_model_parameters_b("operator-selected-local-llm") is None


def test_ai_uplift_accepts_multibillion_holdout_improvement() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 12.0,
            "roi_pct": 1.2,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
        },
        {
            "realized_pnl": 18.0,
            "roi_pct": 1.8,
            "max_drawdown": 0.035,
            "expectancy": 1.2,
            "closed_trades": 12,
        },
        model_name="qwen2.5:7b",
    )

    assert report.accepted is True
    assert report.advisory_only is False
    assert report.model_parameters_b == 7.0
    assert report.deltas["realized_pnl"] == 6.0


def test_ai_uplift_rejects_small_or_non_improving_models() -> None:
    report = assess_ai_uplift(
        {"realized_pnl": 12.0, "max_drawdown": 0.04, "expectancy": 0.9, "closed_trades": 10},
        {"realized_pnl": 11.0, "max_drawdown": 0.05, "expectancy": 0.8, "closed_trades": 3},
        model_name="tiny-560m",
    )

    assert report.accepted is False
    assert report.advisory_only is True
    assert "model_parameters<2.00B" in report.reasons
    assert "ai_pnl_not_above_baseline" in report.reasons
    assert "ai_drawdown_worse_than_baseline" in report.reasons
    assert "ai_closed_trades<5" in report.reasons


def test_ai_uplift_policy_can_require_stricter_model_size() -> None:
    report = assess_ai_uplift(
        {"realized_pnl": 10.0, "max_drawdown": 0.04, "expectancy": 0.5, "closed_trades": 8},
        {"realized_pnl": 12.0, "max_drawdown": 0.03, "expectancy": 0.7, "closed_trades": 8},
        model_name="qwen2.5:7b",
        policy=AIUpliftPolicy(min_model_parameters_b=13.0),
    )

    assert report.accepted is False
    assert "model_parameters<13.00B" in report.reasons
