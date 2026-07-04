from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

import simple_ai_trading.robust_validation as robust_validation
from simple_ai_trading.advanced_model import advanced_feature_dimension, advanced_feature_signature, default_config_for
from simple_ai_trading.api import Candle
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.execution_simulation import SymbolExecutionProfile
from simple_ai_trading.model import TrainedModel, serialize_model
from simple_ai_trading.training_suite import ObjectiveOutcome, SuiteReport
from simple_ai_trading.types import StrategyConfig


def _result(**overrides) -> BacktestResult:
    payload = dict(
        starting_cash=1000.0,
        ending_cash=1020.0,
        realized_pnl=20.0,
        win_rate=0.7,
        trades=5,
        max_drawdown=0.02,
        closed_trades=5,
        gross_exposure=100.0,
        total_fees=1.0,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        buy_hold_pnl=5.0,
        edge_vs_buy_hold=15.0,
    )
    payload.update(overrides)
    return BacktestResult(**payload)


def _candles(n: int = 240) -> list[Candle]:
    price = 100.0
    candles: list[Candle] = []
    for index in range(n):
        price *= 1.0 + 0.0005 * math.sin(index / 7.0) + 0.0002
        candles.append(Candle(
            open_time=index * 60_000,
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=100.0,
            close_time=index * 60_000 + 59_000,
        ))
    return candles


def _model_for_objective(path: Path, objective: str, strategy: StrategyConfig) -> TrainedModel:
    feature_cfg = default_config_for(objective, strategy.enabled_features)
    dim = advanced_feature_dimension(feature_cfg)
    model = TrainedModel(
        weights=[0.0] * dim,
        bias=10.0,
        feature_dim=dim,
        epochs=1,
        feature_means=[0.0] * dim,
        feature_stds=[1.0] * dim,
        feature_signature=advanced_feature_signature(feature_cfg),
    )
    serialize_model(model, path)
    return model


def test_stress_scenario_scales_strategy_and_symbol_profile() -> None:
    scenario = robust_validation.StressScenario(
        "crunch",
        slippage_multiplier=2.0,
        spread_multiplier=3.0,
        latency_ms=2500,
        liquidity_haircut=0.8,
        fee_multiplier=1.5,
    )
    strategy = StrategyConfig(slippage_bps=4.0, max_spread_bps=6.0, latency_buffer_ms=100, taker_fee_bps=2.0)
    profile = SymbolExecutionProfile("AAAUSDC", 2.0, 10_000_000.0, 50_000, 0.9, 100, 0.2)

    stressed_strategy = scenario.strategy(strategy)
    stressed_profile = scenario.profile(profile)

    assert stressed_strategy.slippage_bps == pytest.approx(8.0)
    assert stressed_strategy.max_spread_bps == pytest.approx(18.0)
    assert stressed_strategy.latency_buffer_ms == 2500
    assert stressed_strategy.taker_fee_bps == pytest.approx(3.0)
    assert stressed_profile is not None
    assert stressed_profile.spread_bps == pytest.approx(6.0)
    assert stressed_profile.liquidity_haircut == pytest.approx(0.8)
    assert stressed_profile.quote_volume < profile.quote_volume


def test_validate_model_under_stress_rejects_failed_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_profiles: list[SymbolExecutionProfile | None] = []

    def fake_backtest(_rows, _model, strategy, **kwargs):
        observed_profiles.append(kwargs.get("symbol_profile"))
        if strategy.slippage_bps > 2.0:
            return _result(realized_pnl=-5.0, ending_cash=995.0, closed_trades=5, edge_vs_buy_hold=-10.0)
        return _result()

    monkeypatch.setattr(robust_validation, "run_backtest", fake_backtest)
    model = TrainedModel(weights=[0.0], bias=10.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])
    profile = SymbolExecutionProfile("AAAUSDC", 1.0, 10_000_000.0, 50_000, 0.9, 100, 0.2)

    report = robust_validation.validate_model_under_stress(
        [object()],
        model,
        StrategyConfig(slippage_bps=1.0, max_spread_bps=1.0),
        objective_name="regular",
        starting_cash=1000.0,
        market_type="spot",
        symbol_profile=profile,
        scenarios=[
            robust_validation.StressScenario("base"),
            robust_validation.StressScenario("bad", slippage_multiplier=3.0),
        ],
    )

    assert report.accepted is False
    assert report.accepted_scenarios == 1
    assert report.worst_realized_pnl == pytest.approx(-5.0)
    assert report.results[1].reject_reason is not None
    assert "profit_factor" in report.results[0].result
    assert "max_consecutive_losses" in report.results[0].result
    assert observed_profiles[0].symbol == "AAAUSDC"


def test_validate_model_temporal_robustness_rejects_bad_latest_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_backtest(_rows, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            return _result(realized_pnl=-2.0, ending_cash=998.0, closed_trades=5, edge_vs_buy_hold=-5.0)
        return _result()

    monkeypatch.setattr(robust_validation, "run_backtest", fake_backtest)
    model = TrainedModel(weights=[0.0], bias=10.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])

    report = robust_validation.validate_model_temporal_robustness(
        [object() for _ in range(40)],
        model,
        StrategyConfig(),
        objective_name="regular",
        starting_cash=1000.0,
        market_type="spot",
        policy=robust_validation.TemporalRobustnessPolicy(
            objective="regular",
            target_windows=4,
            min_windows=3,
            min_accepted_rate=0.75,
            require_latest_window=True,
            min_window_rows=10,
        ),
    )

    assert report.accepted is False
    assert report.reason == "latest_window_failed"
    assert report.window_count == 4
    assert report.accepted_windows == 3
    assert report.windows[-1].reject_reason is not None
    assert report.windows[-1].regime["dominant_regime"]
    assert report.regime_summary["window_count"] == 4
    assert "by_regime" in report.regime_summary
    assert report.statistical_edge["positive_windows"] == 3


def test_validate_model_temporal_robustness_rejects_weak_statistical_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = [20.0, 18.0, 15.0, -80.0]

    def fake_backtest(_rows, *_args, **_kwargs):
        pnl = returns.pop(0)
        return _result(
            realized_pnl=pnl,
            ending_cash=1000.0 + pnl,
            closed_trades=5,
            edge_vs_buy_hold=pnl,
            max_drawdown=0.02 if pnl > 0 else 0.08,
        )

    monkeypatch.setattr(robust_validation, "run_backtest", fake_backtest)
    model = TrainedModel(weights=[0.0], bias=10.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])

    report = robust_validation.validate_model_temporal_robustness(
        [object() for _ in range(40)],
        model,
        StrategyConfig(),
        objective_name="aggressive",
        starting_cash=1000.0,
        market_type="spot",
        policy=robust_validation.TemporalRobustnessPolicy(
            objective="aggressive",
            target_windows=4,
            min_windows=2,
            min_accepted_rate=0.60,
            require_latest_window=False,
            min_window_rows=10,
            max_sign_test_p_value=0.55,
            min_bootstrap_lower_mean_return=0.0,
        ),
    )

    assert report.accepted is False
    assert str(report.reason).startswith("bootstrap_lower_mean_return<")
    assert report.accepted_windows == 3
    assert report.statistical_edge["accepted"] is False
    assert report.statistical_edge["sign_test_p_value"] == pytest.approx(0.3125)
    assert float(report.statistical_edge["bootstrap_lower_mean_return"]) < 0.0


def test_temporal_robustness_prefers_trade_return_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_sets = [
        (-0.02, -0.01, 0.03),
        (-0.03, 0.01, -0.02),
        (-0.01, -0.01, 0.02),
        (-0.02, 0.01, -0.03),
    ]

    def fake_backtest(_rows, *_args, **_kwargs):
        returns = trade_sets.pop(0)
        return _result(
            realized_pnl=10.0,
            ending_cash=1010.0,
            closed_trades=3,
            edge_vs_buy_hold=10.0,
            gross_profit=50.0,
            gross_loss=5.0,
            profit_factor=10.0,
            expectancy=3.0,
            max_consecutive_losses=1,
            trade_returns=returns,
            trade_pnls=tuple(value * 1000.0 for value in returns),
        )

    monkeypatch.setattr(robust_validation, "run_backtest", fake_backtest)
    model = TrainedModel(weights=[0.0], bias=10.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])

    report = robust_validation.validate_model_temporal_robustness(
        [object() for _ in range(40)],
        model,
        StrategyConfig(),
        objective_name="aggressive",
        starting_cash=1000.0,
        market_type="spot",
        policy=robust_validation.TemporalRobustnessPolicy(
            objective="aggressive",
            target_windows=4,
            min_windows=2,
            min_accepted_rate=0.60,
            require_latest_window=False,
            min_window_rows=10,
            max_sign_test_p_value=0.55,
            min_bootstrap_lower_mean_return=-0.005,
        ),
    )

    assert report.accepted is False
    assert report.statistical_edge["evidence_unit"] == "trade"
    assert report.statistical_edge["sample_count"] == 12
    assert report.statistical_edge["trade_return_count"] == 12
    assert str(report.reason).startswith("sign_test_p_value>")


def test_validate_model_temporal_robustness_requires_enough_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robust_validation, "run_backtest", lambda *_a, **_k: _result())
    model = TrainedModel(weights=[0.0], bias=10.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])

    report = robust_validation.validate_model_temporal_robustness(
        [object() for _ in range(20)],
        model,
        StrategyConfig(),
        objective_name="conservative",
        starting_cash=1000.0,
        market_type="spot",
        policy=robust_validation.TemporalRobustnessPolicy(
            objective="conservative",
            target_windows=2,
            min_windows=3,
            min_accepted_rate=1.0,
            require_latest_window=True,
            min_window_rows=10,
        ),
    )

    assert report.accepted is False
    assert report.reason == "window_count<3"
    assert report.accepted_windows == 2


def test_validate_suite_under_stress_loads_saved_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robust_validation, "run_backtest", lambda *_a, **_k: _result())
    strategy = StrategyConfig()
    model_path = tmp_path / "model_regular.json"
    _model_for_objective(model_path, "regular", strategy)
    outcome = ObjectiveOutcome(
        objective="regular",
        model_path=model_path,
        feature_dim=1,
        feature_signature="sig",
        best_score=0.5,
        best_params={"epochs": 1},
        explored_candidates=1,
        rejected_candidates=0,
        epochs=1,
        learning_rate=0.01,
        l2_penalty=0.0,
        row_count=200,
        positive_rate=0.5,
    )
    suite = SuiteReport(
        outcomes=[outcome],
        total_rows=200,
        total_candles=240,
        output_dir=tmp_path,
        summary_path=tmp_path / "summary.json",
        objectives_run=["regular"],
    )

    report = robust_validation.validate_suite_under_stress(
        _candles(),
        strategy,
        suite,
        symbol="AAAUSDC",
        symbol_profile=SymbolExecutionProfile("AAAUSDC", 1.0, 10_000_000.0, 50_000, 0.9, 100, 0.2),
        starting_cash=1000.0,
        market_type="spot",
        scenarios=[robust_validation.StressScenario("base")],
    )

    assert report.accepted is True
    assert report.objective_count == 1
    assert report.scenario_count == 1
    assert report.objectives[0].model_path == str(model_path)
    edge = report.objectives[0].results[0].result["market_edge"]
    assert edge["objective"] == "regular"
    assert edge["net_edge_pct"] == pytest.approx(0.015)


def test_validate_suite_temporal_robustness_loads_saved_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_profiles: list[SymbolExecutionProfile | None] = []

    def fake_backtest(_rows, *_args, **kwargs):
        observed_profiles.append(kwargs.get("symbol_profile"))
        return _result()

    monkeypatch.setattr(robust_validation, "run_backtest", fake_backtest)
    strategy = StrategyConfig()
    model_path = tmp_path / "model_regular.json"
    _model_for_objective(model_path, "regular", strategy)
    outcome = ObjectiveOutcome(
        objective="regular",
        model_path=model_path,
        feature_dim=1,
        feature_signature="sig",
        best_score=0.5,
        best_params={"epochs": 1},
        explored_candidates=1,
        rejected_candidates=0,
        epochs=1,
        learning_rate=0.01,
        l2_penalty=0.0,
        row_count=420,
        positive_rate=0.5,
    )
    suite = SuiteReport(
        outcomes=[outcome],
        total_rows=420,
        total_candles=520,
        output_dir=tmp_path,
        summary_path=tmp_path / "summary.json",
        objectives_run=["regular"],
    )
    profile = SymbolExecutionProfile("AAAUSDC", 1.0, 10_000_000.0, 50_000, 0.9, 100, 0.2)

    report = robust_validation.validate_suite_temporal_robustness(
        _candles(520),
        strategy,
        suite,
        symbol="AAAUSDC",
        symbol_profile=profile,
        starting_cash=1000.0,
        market_type="spot",
    )

    assert report.accepted is True
    assert report.objective_count == 1
    assert report.accepted_windows >= 3
    assert report.objectives[0].model_path == str(model_path)
    assert report.statistical_edge_accepted is True
    assert report.worst_sign_test_p_value < 0.35
    assert report.worst_bootstrap_lower_mean_return > 0.0
    assert observed_profiles
    assert all(item and item.symbol == "AAAUSDC" for item in observed_profiles)


def test_validate_suite_under_stress_uses_model_signature_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_dims: list[int] = []

    def fake_backtest(rows, *_args, **_kwargs):
        observed_dims.append(len(rows[0].features))
        return _result()

    monkeypatch.setattr(robust_validation, "run_backtest", fake_backtest)
    strategy = StrategyConfig()
    candidate_cfg = replace(
        default_config_for("regular", strategy.enabled_features),
        label_lookahead=7,
        label_threshold=0.00168,
        label_stop_threshold=0.00168,
        confluence_windows=(5, 13, 34),
    )
    model_path = tmp_path / "model_regular_candidate.json"
    dim = advanced_feature_dimension(candidate_cfg)
    serialize_model(
        TrainedModel(
            weights=[0.0] * dim,
            bias=10.0,
            feature_dim=dim,
            epochs=1,
            feature_means=[0.0] * dim,
            feature_stds=[1.0] * dim,
            feature_signature=advanced_feature_signature(candidate_cfg),
        ),
        model_path,
    )
    suite = SuiteReport(
        outcomes=[
            ObjectiveOutcome(
                objective="regular",
                model_path=model_path,
                feature_dim=dim,
                feature_signature=advanced_feature_signature(candidate_cfg),
                best_score=0.5,
                best_params={"epochs": 1},
                explored_candidates=1,
                rejected_candidates=0,
                epochs=1,
                learning_rate=0.01,
                l2_penalty=0.0,
                row_count=200,
                positive_rate=0.5,
            )
        ],
        total_rows=200,
        total_candles=240,
        output_dir=tmp_path,
        summary_path=tmp_path / "summary.json",
        objectives_run=["regular"],
    )

    report = robust_validation.validate_suite_under_stress(
        _candles(),
        strategy,
        suite,
        symbol="AAAUSDC",
        symbol_profile=SymbolExecutionProfile("AAAUSDC", 1.0, 10_000_000.0, 50_000, 0.9, 100, 0.2),
        starting_cash=1000.0,
        market_type="spot",
        scenarios=[robust_validation.StressScenario("base")],
    )

    assert report.accepted is True
    assert report.scenario_count == 1
    assert report.objectives[0].error is None
    assert observed_dims == [dim]
