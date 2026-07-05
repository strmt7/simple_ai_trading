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
            "trade_returns": (0.010, -0.004, 0.006, 0.011, 0.008, -0.003, 0.012, 0.007, 0.005, 0.009),
        },
        {
            "realized_pnl": 18.0,
            "roi_pct": 1.8,
            "max_drawdown": 0.035,
            "expectancy": 1.2,
            "closed_trades": 12,
            "trade_returns": (0.014, -0.002, 0.009, 0.013, 0.011, -0.001, 0.015, 0.010, 0.006, 0.012),
        },
        model_name="qwen2.5:7b",
    )

    assert report.accepted is True
    assert report.advisory_only is False
    assert report.model_parameters_b == 7.0
    assert report.deltas["realized_pnl"] == 6.0
    assert report.statistical_evidence["accepted"] is True
    assert report.statistical_evidence["sample_count"] == 10
    assert report.statistical_evidence["positive_delta_count"] == 10


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
    assert "ai_uplift_paired_samples<8" in report.reasons


def test_ai_uplift_rejects_statistically_weak_paired_samples() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 12.0,
            "roi_pct": 1.2,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
            "trade_returns": (0.010, 0.011, 0.010, 0.012, 0.011, 0.010, 0.011, 0.010),
        },
        {
            "realized_pnl": 13.0,
            "roi_pct": 1.3,
            "max_drawdown": 0.04,
            "expectancy": 1.0,
            "closed_trades": 10,
            "trade_returns": (0.011, 0.010, 0.011, 0.011, 0.012, 0.009, 0.010, 0.011),
        },
        model_name="qwen2.5:7b",
    )

    assert report.accepted is False
    assert report.statistical_evidence["sample_count"] == 8
    assert report.statistical_evidence["positive_delta_count"] == 4
    assert "ai_uplift_positive_delta_rate<0.55" in report.reasons
    assert "ai_uplift_sign_test_p_value>0.4000" in report.reasons


def test_ai_uplift_rejects_unpaired_sample_lengths() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 12.0,
            "roi_pct": 1.2,
            "max_drawdown": 0.04,
            "expectancy": 0.9,
            "closed_trades": 10,
            "trade_returns": (0.010, 0.011, 0.010, 0.012, 0.011, 0.010, 0.011, 0.010),
        },
        {
            "realized_pnl": 14.0,
            "roi_pct": 1.4,
            "max_drawdown": 0.035,
            "expectancy": 1.1,
            "closed_trades": 10,
            "trade_returns": (0.012, 0.013, 0.012, 0.014, 0.013, 0.012, 0.013, 0.012, 0.011),
        },
        model_name="qwen2.5:7b",
    )

    assert report.accepted is False
    assert report.statistical_evidence["paired_sample_length_mismatch"] is True
    assert "ai_uplift_paired_sample_length_mismatch" in report.reasons


def test_ai_uplift_rejects_tail_risk_deterioration() -> None:
    report = assess_ai_uplift(
        {
            "realized_pnl": 20.0,
            "max_drawdown": 0.04,
            "expectancy": 0.8,
            "profit_factor": 1.8,
            "win_rate": 0.62,
            "closed_trades": 20,
            "max_consecutive_losses": 2,
            "downside_return_risk_ratio": 0.70,
            "liquidation_events": 0,
        },
        {
            "realized_pnl": 25.0,
            "max_drawdown": 0.04,
            "expectancy": 1.0,
            "profit_factor": 1.4,
            "win_rate": 0.55,
            "closed_trades": 22,
            "max_consecutive_losses": 4,
            "downside_return_risk_ratio": 0.60,
            "liquidation_events": 1,
        },
        model_name="qwen2.5:7b",
    )

    assert report.accepted is False
    assert "ai_liquidation_events>0" in report.reasons
    assert "ai_loss_streak_worse_than_baseline" in report.reasons
    assert "ai_profit_factor_worse_than_baseline" in report.reasons
    assert "ai_win_rate_worse_than_baseline" in report.reasons
    assert "ai_downside_return_risk_not_above_baseline" in report.reasons


def test_ai_uplift_policy_can_require_stricter_model_size() -> None:
    report = assess_ai_uplift(
        {"realized_pnl": 10.0, "max_drawdown": 0.04, "expectancy": 0.5, "closed_trades": 8},
        {"realized_pnl": 12.0, "max_drawdown": 0.03, "expectancy": 0.7, "closed_trades": 8},
        model_name="qwen2.5:7b",
        policy=AIUpliftPolicy(min_model_parameters_b=13.0),
    )

    assert report.accepted is False
    assert "model_parameters<13.00B" in report.reasons
