from __future__ import annotations

import pytest

from simple_ai_trading import meta_label as meta_label_module
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
        ModelRow(
            timestamp=index * 60_000,
            close=100.0 + index,
            features=((0.8 if index < 30 else 0.1),),
            label=1,
        )
        for index in range(40)
    ]


def _result() -> BacktestResult:
    trade_log = tuple(
        {
            "opened_at": index * 60_000,
            "side": 1,
            "net_pnl": 4.0 if index < 24 else (-1.0 if index < 30 else -2.0),
            "return_pct": (
                0.004 if index < 24 else (-0.001 if index < 30 else -0.002)
            ),
        }
        for index in range(40)
    )
    return BacktestResult(
        starting_cash=1000.0,
        ending_cash=1070.0,
        realized_pnl=70.0,
        win_rate=0.6,
        trades=40,
        max_drawdown=0.02,
        closed_trades=40,
        gross_exposure=4000.0,
        total_fees=1.0,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        trade_log=trade_log,
    )


def test_extract_meta_label_samples_uses_open_timestamp_scores() -> None:
    samples = extract_meta_label_samples(
        _rows(),
        _model(),
        StrategyConfig(confidence_beta=1.0),
        _result(),
        market_type="spot",
    )

    assert len(samples) == 40
    assert samples[0].profitable is True
    assert samples[0].signal_strength > samples[-1].signal_strength
    assert samples[-1].net_pnl == pytest.approx(-2.0)


def test_return_bootstrap_is_invariant_to_quote_capital_scale() -> None:
    def samples(scale: float):
        return tuple(
            meta_label_module.MetaLabelSample(
                opened_at=index * 60_000,
                side=1,
                probability=0.8,
                adjusted_probability=0.8,
                signal_strength=0.2,
                net_pnl=scale * (2.0 if index < 24 else -1.0),
                return_pct=0.002 if index < 24 else -0.001,
                profitable=index < 24,
            )
            for index in range(30)
        )

    base = meta_label_module._bucket_bootstrap(
        samples(1.0),
        objective="regular",
        action="take",
    )
    scaled = meta_label_module._bucket_bootstrap(
        samples(100.0),
        objective="regular",
        action="take",
    )

    assert scaled == base


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
    assert report.policy["evidence_schema_version"] == "meta-label-after-cost-v2"
    assert report.policy["take_bootstrap_samples"] == 2_000
    assert report.policy["take_bootstrap_confidence"] == pytest.approx(0.95)
    assert report.policy["take_bootstrap_mean_return_lower"] > 0.0


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
    assert report.policy["mode"] == "observe_only"


def test_positive_mean_without_positive_block_bootstrap_stays_observe_only() -> None:
    rows = [
        ModelRow(
            timestamp=index * 60_000,
            close=100.0,
            features=(0.8,),
            label=1,
        )
        for index in range(30)
    ]
    trade_log = tuple(
        {
            "opened_at": index * 60_000,
            "side": 1,
            "net_pnl": 10.0 if index < 20 else -19.0,
            "return_pct": 0.01 if index < 20 else -0.019,
        }
        for index in range(30)
    )
    result = BacktestResult(
        starting_cash=1000.0,
        ending_cash=1010.0,
        realized_pnl=10.0,
        win_rate=2.0 / 3.0,
        trades=30,
        max_drawdown=0.19,
        closed_trades=30,
        gross_exposure=3000.0,
        total_fees=0.0,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        trade_log=trade_log,
    )

    report = build_meta_label_report(
        rows,
        _model(),
        StrategyConfig(confidence_beta=1.0),
        result,
        objective_name="regular",
        market_type="spot",
    )

    assert report.take_mean_return > 0.0
    assert report.take_net_pnl > 0.0
    assert report.policy["take_bootstrap_mean_return_lower"] <= 0.0
    assert report.status == "observe_only"
    assert report.reason == "take_bootstrap_lower_not_positive"
    assert report.policy["enabled"] is False
    blocked = apply_meta_label_policy(
        report.policy,
        adjusted_probability=0.99,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert blocked.action == "skip"
    assert blocked.reason == "meta_label_observe_only"


def test_apply_meta_label_policy_classifies_take_downsize_skip_and_invalid() -> None:
    policy = {
        "enabled": True,
        "evidence_schema_version": "meta-label-after-cost-v2",
        "mode": "take_downsize_skip",
        "take_threshold": 0.20,
        "downsize_threshold": 0.10,
        "downsize_fraction": 0.35,
        "minimum_action_samples": 30,
        "target_precision": 0.60,
        "take_sample_count": 36,
        "take_precision": 0.75,
        "take_mean_return": 0.002,
        "take_net_pnl": 16.0,
        "take_bootstrap_samples": 2_000,
        "take_bootstrap_confidence": 0.95,
        "take_bootstrap_block_length": 6,
        "take_bootstrap_mean_return_lower": 0.0005,
        "downsize_sample_count": 32,
        "downsize_precision": 0.50,
        "downsize_mean_return": 0.001,
        "downsize_net_pnl": 4.0,
        "downsize_bootstrap_samples": 2_000,
        "downsize_bootstrap_confidence": 0.95,
        "downsize_bootstrap_block_length": 6,
        "downsize_bootstrap_mean_return_lower": 0.0002,
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
    assert take.validation_minimum_sample_count == 30
    assert take.validation_minimum_precision == pytest.approx(0.60)
    assert take.validation_sample_count == 36
    assert take.expected_after_cost_pnl == pytest.approx(16.0)
    assert take.validation_bootstrap_samples == 2_000
    assert take.validation_bootstrap_lower_after_cost_return == pytest.approx(
        0.0005
    )

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
    assert downsize.validation_sample_count == 32
    assert downsize.expected_after_cost_return == pytest.approx(0.001)
    assert downsize.validation_bootstrap_block_length == 6

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
        {
            "enabled": True,
            "evidence_schema_version": "meta-label-after-cost-v2",
            "mode": "take_downsize_skip",
            "take_threshold": 0.1,
            "downsize_threshold": 0.2,
        },
        adjusted_probability=0.99,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert invalid.enabled is True
    assert invalid.action == "skip"
    assert invalid.reason == "invalid_meta_label_thresholds"

    legacy = apply_meta_label_policy(
        {
            **policy,
            "evidence_schema_version": "meta-label-after-cost-v1",
        },
        adjusted_probability=0.99,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert legacy.action == "skip"
    assert legacy.reason == "invalid_meta_label_evidence_schema"

    invalid_evidence = apply_meta_label_policy(
        {
            **policy,
            "take_bootstrap_mean_return_lower": 0.0,
        },
        adjusted_probability=0.99,
        threshold=0.60,
        side=1,
        market_type="spot",
    )
    assert invalid_evidence.action == "skip"
    assert invalid_evidence.reason == "invalid_meta_label_take_evidence"


def test_liquidity_overlay_preserves_after_cost_bucket_evidence() -> None:
    base = apply_meta_label_policy(
        {
            "enabled": True,
            "evidence_schema_version": "meta-label-after-cost-v2",
            "mode": "take_downsize_skip",
            "take_threshold": 0.20,
            "downsize_threshold": 0.20,
            "downsize_fraction": 0.5,
            "minimum_action_samples": 30,
            "target_precision": 0.65,
            "take_sample_count": 36,
            "take_precision": 0.75,
            "take_mean_return": 0.002,
            "take_net_pnl": 24.0,
            "take_bootstrap_samples": 2_000,
            "take_bootstrap_confidence": 0.95,
            "take_bootstrap_block_length": 6,
            "take_bootstrap_mean_return_lower": 0.0005,
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
    assert adjusted.validation_minimum_sample_count == 30
    assert adjusted.validation_minimum_precision == pytest.approx(0.65)
    assert adjusted.validation_sample_count == 36
    assert adjusted.expected_after_cost_return == pytest.approx(0.002)
    assert adjusted.expected_after_cost_pnl == pytest.approx(24.0)
    assert adjusted.validation_bootstrap_samples == 2_000
    assert adjusted.validation_bootstrap_lower_after_cost_return == pytest.approx(
        0.0005
    )
