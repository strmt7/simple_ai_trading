from __future__ import annotations

import pytest

from simple_ai_trading.risk_controls import (
    assess_entry_risk,
    build_risk_policy_report,
    market_regime_unpredictability,
    render_risk_policy_report,
    stop_loss_effective_loss_pct,
    stop_loss_sized_notional_pct,
)
from simple_ai_trading.model import TrainedModel, serialize_model
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


def test_risk_policy_report_allows_default_paper_and_renders_summary() -> None:
    report = build_risk_policy_report(RuntimeConfig(), StrategyConfig(), effective_dry_run=True)

    assert report.allowed is True
    assert report.block_count == 0
    assert report.warning_count >= 1
    assert report.notional_cap_pct == 0.08
    assert report.max_loss_per_trade_pct == pytest.approx(
        report.notional_cap_pct * stop_loss_effective_loss_pct(StrategyConfig())
    )
    assert report.max_loss_per_trade_pct > report.notional_cap_pct * StrategyConfig().stop_loss_pct
    assert report.checks[0].asdict()["label"] == "primary symbol"
    assert any(check.label == "regime unpredictability gate" for check in report.checks)
    rendered = render_risk_policy_report(report)
    assert "Risk policy report" in rendered
    assert "allowed=True" in rendered


def test_market_regime_unpredictability_scores_wait_conditions() -> None:
    assert market_regime_unpredictability("volatile_chop", 0.9) > 0.90
    assert market_regime_unpredictability("mixed", 0.2, ("low_regime_separation",)) > 0.85
    assert market_regime_unpredictability("insufficient_data", 0.0) == 1.0
    assert market_regime_unpredictability("trend_up", 0.9) < 0.25


def test_stop_loss_sized_notional_pct_respects_caps_and_leverage() -> None:
    spot = StrategyConfig(risk_per_trade=0.01, max_position_pct=0.5, stop_loss_pct=0.02)
    futures = StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2, stop_loss_pct=0.02, leverage=5.0)

    spot_notional = stop_loss_sized_notional_pct(spot, "spot")
    futures_notional = stop_loss_sized_notional_pct(futures, "futures")
    futures_override = stop_loss_sized_notional_pct(futures, "futures", leverage=20.0)

    assert 0.0 < spot_notional < 0.5
    assert spot_notional * stop_loss_effective_loss_pct(spot) <= spot.risk_per_trade + 1e-12
    assert futures_notional * stop_loss_effective_loss_pct(futures) <= futures.risk_per_trade + 1e-12
    assert futures_override == pytest.approx(futures_notional)


def test_stop_loss_sized_notional_pct_caps_gross_asset_allocation() -> None:
    high_leverage = StrategyConfig(
        risk_per_trade=0.05,
        max_position_pct=0.9,
        max_asset_allocation_pct=0.12,
        stop_loss_pct=0.01,
        leverage=20.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        latency_buffer_ms=0,
    )

    assert stop_loss_sized_notional_pct(high_leverage, "spot") == pytest.approx(0.12)
    assert stop_loss_sized_notional_pct(high_leverage, "futures") == pytest.approx(0.12)


def test_stop_loss_sizing_never_exceeds_per_trade_risk_budget() -> None:
    strategies = [
        StrategyConfig(risk_per_trade=0.003, max_position_pct=0.08, stop_loss_pct=0.010, leverage=5.0),
        StrategyConfig(risk_per_trade=0.006, max_position_pct=0.14, stop_loss_pct=0.020, leverage=10.0),
        StrategyConfig(risk_per_trade=0.010, max_position_pct=0.20, stop_loss_pct=0.025, leverage=15.0),
        StrategyConfig(risk_per_trade=0.020, max_position_pct=0.35, stop_loss_pct=0.015, leverage=20.0),
    ]

    for strategy in strategies:
        for market_type in ("spot", "futures"):
            notional_pct = stop_loss_sized_notional_pct(strategy, market_type, leverage=strategy.leverage)
            estimated_loss_pct = notional_pct * stop_loss_effective_loss_pct(strategy)

            assert estimated_loss_pct <= strategy.risk_per_trade + 1e-12


def test_signed_futures_policy_blocks_disabled_stop_loss_and_liquidation_buffer() -> None:
    report = build_risk_policy_report(
        RuntimeConfig(
            market_type="futures",
            testnet=True,
            dry_run=False,
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        ),
        StrategyConfig(stop_loss_pct=0.0, liquidation_buffer_pct=0.0),
        effective_dry_run=False,
    )

    assert report.allowed is False
    assert any(check.label == "stop loss" and check.status == "block" for check in report.checks)
    assert any(check.label == "liquidation buffer" and check.status == "block" for check in report.checks)

    paper = build_risk_policy_report(
        RuntimeConfig(market_type="futures", dry_run=True),
        StrategyConfig(stop_loss_pct=0.0, liquidation_buffer_pct=0.0),
        effective_dry_run=True,
    )
    assert any(check.label == "stop loss" and check.status == "warn" for check in paper.checks)
    assert any(check.label == "liquidation buffer" and check.status == "warn" for check in paper.checks)


def test_risk_policy_blocks_mainnet_live_missing_credentials_and_zero_cash(tmp_path) -> None:
    report = build_risk_policy_report(
        RuntimeConfig(testnet=False, dry_run=False, managed_usdc=0.0),
        StrategyConfig(risk_per_trade=0.0, max_position_pct=0.0),
        effective_dry_run=False,
        model_path=tmp_path / "missing-model.json",
    )

    labels = {check.label: check.status for check in report.checks}
    assert report.allowed is False
    assert report.block_count >= 5
    assert labels["execution environment"] == "block"
    assert labels["order credentials"] == "block"
    assert labels["managed USDC"] == "block"
    assert labels["risk per trade"] == "block"
    assert labels["model path"] == "block"
    assert report.asdict()["allowed"] is False

    demo = build_risk_policy_report(
        RuntimeConfig(testnet=False, demo=True, dry_run=False, api_key="k", api_secret="s"),
        StrategyConfig(),
        effective_dry_run=False,
    )
    assert any(check.label == "execution environment" and "demo endpoint" in check.detail for check in demo.checks)


def test_risk_policy_blocks_extreme_capital_loss_settings(tmp_path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")
    report = build_risk_policy_report(
        RuntimeConfig(market_type="futures", api_key="k", api_secret="s", dry_run=False, managed_usdc=1000.0),
        StrategyConfig(
            leverage=80.0,
            risk_per_trade=0.04,
            max_position_pct=0.90,
            max_drawdown_limit=0.75,
            max_daily_loss_pct=0.05,
            max_session_loss_pct=0.12,
            max_consecutive_losses=12,
            max_network_errors=12,
            recovery_cooldown_seconds=0,
            slippage_bps=150.0,
            taker_fee_bps=125.0,
        ),
        effective_dry_run=False,
        model_path=model_path,
    )

    assert report.allowed is False
    assert report.leverage == 20.0
    assert report.warning_count >= 3
    assert report.block_count >= 2
    assert any(check.label == "effective leverage" and check.status == "ok" for check in report.checks)
    assert any(check.label == "max position" and check.status == "block" for check in report.checks)
    assert any(check.label == "drawdown stop" and check.status == "block" for check in report.checks)
    assert any(check.label == "daily loss budget" and check.status == "block" for check in report.checks)
    assert any(check.label == "session loss budget" and check.status == "block" for check in report.checks)

    disabled_drawdown = build_risk_policy_report(
        RuntimeConfig(),
        StrategyConfig(max_drawdown_limit=0.0),
        effective_dry_run=True,
    )
    assert any(check.label == "drawdown stop" and check.status == "warn" for check in disabled_drawdown.checks)

    strategy = StrategyConfig()
    strategy.leverage = float("nan")
    strategy.risk_per_trade = object()
    coerced = build_risk_policy_report(
        RuntimeConfig(market_type="futures"),
        strategy,
        effective_dry_run=True,
    )
    assert coerced.leverage == 1.0
    assert any(check.label == "risk per trade" and check.status == "block" for check in coerced.checks)

    disabled_loss = build_risk_policy_report(
        RuntimeConfig(dry_run=False, api_key="k", api_secret="s", managed_usdc=1000.0),
        StrategyConfig(max_daily_loss_pct=0.0, max_session_loss_pct=0.0, recovery_cooldown_seconds=0),
        effective_dry_run=False,
    )
    assert any(check.label == "daily loss budget" and check.status == "block" for check in disabled_loss.checks)
    assert any(check.label == "session loss budget" and check.status == "block" for check in disabled_loss.checks)
    assert any(check.label == "reconnect recovery cooldown" and check.status == "block" for check in disabled_loss.checks)


def test_risk_policy_blocks_live_when_one_stop_can_breach_loss_budget() -> None:
    strategy = StrategyConfig(
        risk_per_trade=0.02,
        max_position_pct=0.35,
        max_asset_allocation_pct=0.35,
        stop_loss_pct=0.02,
        max_daily_loss_pct=0.006,
        max_session_loss_pct=0.012,
        max_portfolio_risk_pct=0.015,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
    )
    report = build_risk_policy_report(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        ),
        strategy,
        effective_dry_run=False,
    )

    coherence = [check for check in report.checks if check.label == "loss budget coherence"]
    assert report.allowed is False
    assert report.max_loss_per_trade_pct == pytest.approx(
        report.notional_cap_pct * stop_loss_effective_loss_pct(strategy)
    )
    assert report.max_loss_per_trade_pct > strategy.max_daily_loss_pct
    assert coherence[0].status == "block"
    assert "daily loss budget 0.60%" in coherence[0].detail


def test_risk_policy_warns_paper_when_one_stop_can_breach_loss_budget() -> None:
    strategy = StrategyConfig(
        risk_per_trade=0.02,
        max_position_pct=0.35,
        max_asset_allocation_pct=0.35,
        stop_loss_pct=0.02,
        max_daily_loss_pct=0.006,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        latency_buffer_ms=0,
        testnet_liquidity_haircut=0.0,
    )

    report = build_risk_policy_report(
        RuntimeConfig(testnet=True, dry_run=True),
        strategy,
        effective_dry_run=True,
    )

    assert report.allowed is True
    assert any(check.label == "loss budget coherence" and check.status == "warn" for check in report.checks)


def test_risk_policy_blocks_live_impossible_stop_loss_geometry() -> None:
    report = build_risk_policy_report(
        RuntimeConfig(
            testnet=True,
            dry_run=False,
            api_key="k",
            api_secret="s",
            managed_usdc=1000.0,
        ),
        StrategyConfig(stop_loss_pct=1.25),
        effective_dry_run=False,
    )

    assert report.allowed is False
    assert any(check.label == "stop-loss geometry" and check.status == "block" for check in report.checks)


def test_risk_policy_warns_paper_impossible_stop_loss_geometry() -> None:
    report = build_risk_policy_report(
        RuntimeConfig(testnet=True, dry_run=True),
        StrategyConfig(stop_loss_pct=1.25),
        effective_dry_run=True,
    )

    assert report.allowed is True
    assert any(check.label == "stop-loss geometry" and check.status == "warn" for check in report.checks)


def test_risk_policy_reports_model_promotion_evidence(tmp_path) -> None:
    promoted_path = tmp_path / "promoted.json"
    serialize_model(
        TrainedModel(
            weights=[0.0],
            bias=0.0,
            feature_dim=1,
            epochs=1,
            feature_means=[0.0],
            feature_stds=[1.0],
            selection_risk={
                "passed": True,
                "effective_trials": 10,
                "selected_score": 0.10,
                "trial_penalty": 0.02,
                "deflated_score": 0.08,
            },
            execution_validation={
                "passed": True,
                "symbol": "BTCUSDC",
                "stress": {"accepted": True},
                "temporal_robustness": {"accepted": True},
                "portfolio": {"accepted": True},
            },
            probability_calibration_size=128,
            probability_log_loss_before=0.62,
            probability_log_loss_after=0.58,
            probability_brier_before=0.24,
            probability_brier_after=0.22,
            probability_ece_before=0.10,
            probability_ece_after=0.08,
        ),
        promoted_path,
    )
    runtime = RuntimeConfig(testnet=True, dry_run=False, api_key="k", api_secret="s", managed_usdc=1000.0)
    promoted = build_risk_policy_report(
        runtime,
        StrategyConfig(),
        effective_dry_run=False,
        model_path=promoted_path,
    )
    assert any(check.label == "model promotion evidence" and check.status == "ok" for check in promoted.checks)

    strict = build_risk_policy_report(
        runtime,
        StrategyConfig(),
        effective_dry_run=False,
        model_path=promoted_path,
        require_model_candidate_search=True,
        require_accelerator_evidence=True,
    )
    assert strict.allowed is False
    strict_detail = "; ".join(check.detail for check in strict.checks if check.label == "model promotion evidence")
    assert "model candidate search" in strict_detail
    assert "training accelerator" in strict_detail

    stale_path = tmp_path / "stale.json"
    serialize_model(
        TrainedModel(
            weights=[0.0],
            bias=0.0,
            feature_dim=1,
            epochs=1,
            feature_means=[0.0],
            feature_stds=[1.0],
        ),
        stale_path,
    )
    stale = build_risk_policy_report(
        runtime,
        StrategyConfig(),
        effective_dry_run=False,
        model_path=stale_path,
    )
    assert stale.allowed is False
    assert any(check.label == "model promotion evidence" and check.status == "block" for check in stale.checks)


def test_entry_risk_decision_explains_each_block() -> None:
    allowed = assess_entry_risk(
        direction=1,
        position_side=0,
        max_open_positions=1,
        max_daily_trades=2,
        daily_trade_count=0,
        cash=1000.0,
        price=50_000.0,
        drawdown=0.0,
        drawdown_limit=0.2,
    )
    assert allowed.allowed is True
    assert allowed.code == "allowed"

    cases = [
        (dict(direction=0), "no_signal"),
        (dict(position_side=1), "position_open"),
        (dict(max_open_positions=0), "max_open_positions"),
        (dict(max_daily_trades=2, daily_trade_count=2), "trade_cap"),
        (dict(cash=0.0), "cash"),
        (dict(price=0.0), "price"),
        (dict(daily_loss=0.02, daily_loss_limit=0.01), "daily_loss"),
        (dict(session_loss=0.02, session_loss_limit=0.01), "session_loss"),
        (dict(consecutive_losses=2, max_consecutive_losses=2), "loss_streak"),
        (dict(recovery_pending=True), "recovery_pending"),
        (
            dict(regime="volatile_chop", regime_confidence=0.9, max_regime_unpredictability=0.60),
            "unpredictable_regime",
        ),
        (dict(regime_cooldown_active=True), "regime_cooldown"),
        (dict(network_errors=3, max_network_errors=3), "network_halt"),
        (dict(drawdown=0.2, drawdown_limit=0.2), "drawdown"),
        (dict(price=float("nan")), "nonfinite"),
        (dict(cash=object()), "nonfinite"),
    ]
    base = dict(
        direction=1,
        position_side=0,
        max_open_positions=1,
        max_daily_trades=2,
        daily_trade_count=0,
        cash=1000.0,
        price=50_000.0,
        drawdown=0.0,
        drawdown_limit=0.2,
    )
    for override, code in cases:
        decision = assess_entry_risk(**{**base, **override})
        assert decision.allowed is False
        assert decision.code == code
        assert decision.asdict()["metrics"]
