from __future__ import annotations

import pytest

from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import ModelRow
from simple_ai_trading.liquidity_session import LiquiditySessionAdjustment, apply_liquidity_session_meta
from simple_ai_trading.meta_label import apply_meta_label_policy, build_meta_label_report, extract_meta_label_samples
from simple_ai_trading.model import TrainedModel
from simple_ai_trading.types import StrategyConfig


def _model() -> TrainedModel:
    return TrainedModel(
        weights=[8.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.60,
    )


def _rows() -> list[ModelRow]:
    return [
        ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(feature,), label=1)
        for index, feature in enumerate([0.8, 0.7, 0.2, 0.1])
    ]


def _result() -> BacktestResult:
    return BacktestResult(
        starting_cash=1000.0,
        ending_cash=1028.0,
        realized_pnl=28.0,
        win_rate=0.5,
        trades=4,
        max_drawdown=0.02,
        closed_trades=4,
        gross_exposure=400.0,
        total_fees=1.0,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        trade_log=(
            {"opened_at": 0, "side": 1, "net_pnl": 20.0, "return_pct": 0.020},
            {"opened_at": 60_000, "side": 1, "net_pnl": 12.0, "return_pct": 0.012},
            {"opened_at": 120_000, "side": 1, "net_pnl": -3.0, "return_pct": -0.003},
            {"opened_at": 180_000, "side": 1, "net_pnl": -1.0, "return_pct": -0.001},
        ),
    )


def test_extract_meta_label_samples_uses_open_timestamp_scores() -> None:
    samples = extract_meta_label_samples(
        _rows(),
        _model(),
        StrategyConfig(confidence_beta=1.0),
        _result(),
        market_type="spot",
    )

    assert len(samples) == 4
    assert samples[0].profitable is True
    assert samples[0].signal_strength > samples[-1].signal_strength
    assert samples[-1].net_pnl == pytest.approx(-1.0)


def test_build_meta_label_report_rejects_negative_expectancy_downsize_band() -> None:
    report = build_meta_label_report(
        _rows(),
        _model(),
        StrategyConfig(confidence_beta=1.0),
        _result(),
        objective_name="regular",
        market_type="spot",
    )

    assert report.status == "trained"
    assert report.policy["enabled"] is True
    assert report.take_threshold is not None
    assert report.downsize_threshold is not None
    assert report.take_precision >= report.target_precision
    assert report.take_net_pnl > 0.0
    assert report.policy["downsize_evidence_accepted"] is False
    assert report.downsize_threshold == report.take_threshold
    assert report.downsize_count == 0
    assert report.skip_count >= 1


def test_build_meta_label_report_handles_insufficient_samples() -> None:
    sparse = BacktestResult(
        starting_cash=1000.0,
        ending_cash=1001.0,
        realized_pnl=1.0,
        win_rate=1.0,
        trades=1,
        max_drawdown=0.0,
        closed_trades=1,
        gross_exposure=100.0,
        total_fees=0.1,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        trade_log=({"opened_at": 0, "side": 1, "net_pnl": 1.0, "return_pct": 0.001},),
    )

    report = build_meta_label_report(
        _rows(),
        _model(),
        StrategyConfig(confidence_beta=1.0),
        sparse,
        objective_name="regular",
        market_type="spot",
    )

    assert report.status == "insufficient"
    assert report.policy["enabled"] is False


def test_apply_meta_label_policy_classifies_take_downsize_skip_and_invalid() -> None:
    policy = {
        "enabled": True,
        "mode": "take_downsize_skip",
        "take_threshold": 0.20,
        "downsize_threshold": 0.10,
        "downsize_fraction": 0.35,
        "minimum_action_samples": 2,
        "target_precision": 0.60,
        "take_sample_count": 8,
        "take_precision": 0.75,
        "take_mean_return": 0.002,
        "take_net_pnl": 16.0,
        "downsize_sample_count": 6,
        "downsize_precision": 0.50,
        "downsize_mean_return": 0.001,
        "downsize_net_pnl": 4.0,
    }

    take = apply_meta_label_policy(
        policy,
        adjusted_probability=0.82,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert take.action == "take"
    assert take.size_multiplier == pytest.approx(1.0)
    assert take.validation_minimum_sample_count == 2
    assert take.validation_minimum_precision == pytest.approx(0.60)
    assert take.validation_sample_count == 8
    assert take.expected_after_cost_pnl == pytest.approx(16.0)

    downsize = apply_meta_label_policy(
        policy,
        adjusted_probability=0.72,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert downsize.action == "downsize"
    assert downsize.size_multiplier == pytest.approx(0.35)
    assert downsize.validation_minimum_precision == 0.0
    assert downsize.validation_sample_count == 6
    assert downsize.expected_after_cost_return == pytest.approx(0.001)

    skip = apply_meta_label_policy(
        policy,
        adjusted_probability=0.65,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert skip.action == "skip"
    assert skip.size_multiplier == 0.0

    disabled = apply_meta_label_policy(
        {"enabled": False},
        adjusted_probability=0.40,
        threshold=0.60,
        side=-1,
        market_type="futures",
    )
    assert disabled.enabled is False
    assert disabled.action == "take"

    invalid = apply_meta_label_policy(
        {"enabled": True, "mode": "take_downsize_skip", "take_threshold": 0.1, "downsize_threshold": 0.2},
        adjusted_probability=0.99,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert invalid.enabled is True
    assert invalid.action == "skip"
    assert invalid.reason == "invalid_meta_label_thresholds"


def test_liquidity_overlay_preserves_after_cost_bucket_evidence() -> None:
    base = apply_meta_label_policy(
        {
            "enabled": True,
            "mode": "take_downsize_skip",
            "take_threshold": 0.20,
            "downsize_threshold": 0.10,
            "downsize_fraction": 0.5,
            "minimum_action_samples": 5,
            "target_precision": 0.65,
            "take_sample_count": 12,
            "take_precision": 0.75,
            "take_mean_return": 0.002,
            "take_net_pnl": 24.0,
        },
        adjusted_probability=0.85,
        threshold=0.60,
        side=1,
        market_type="spot",
    )

    adjusted = apply_liquidity_session_meta(
        base,
        LiquiditySessionAdjustment(0.70, 0.5, True, False),
    )

    assert adjusted.action == "downsize"
    assert adjusted.size_multiplier == pytest.approx(0.5)
    assert adjusted.validation_minimum_sample_count == 5
    assert adjusted.validation_minimum_precision == pytest.approx(0.65)
    assert adjusted.validation_sample_count == 12
    assert adjusted.expected_after_cost_return == pytest.approx(0.002)
    assert adjusted.expected_after_cost_pnl == pytest.approx(24.0)
