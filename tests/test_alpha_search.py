from __future__ import annotations

from types import SimpleNamespace

from simple_ai_trading.alpha_search import (
    RuleAlphaCandidate,
    RuleAlphaCandidateResult,
    _diagnostic_rank_key,
    model_for_rule_alpha,
    optimize_rule_alpha_model_zoo,
    rule_alpha_feature_params,
    rule_alpha_candidates,
    summarize_rule_alpha_trade_path,
)
from simple_ai_trading.advanced_model import default_config_for
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import ModelRow
from simple_ai_trading.features import FEATURE_NAMES
from simple_ai_trading.model import load_model, serialize_model
from simple_ai_trading.types import StrategyConfig


def _rows(*, rising: bool = True, count: int = 36) -> list[ModelRow]:
    rows: list[ModelRow] = []
    for index in range(count):
        close = 100.0 + index * 0.35 if rising else 100.0 - index * 0.35
        direction = 1.0 if rising else -1.0
        rows.append(ModelRow(
            timestamp=index * 60_000,
            close=close,
            features=(
                0.0030 * direction,
                0.0040 * direction,
                0.0050 * direction,
                0.0060 * direction,
                0.0,
                0.55,
                0.0040 * direction,
                0.0020,
                0.0020,
                1.5,
                0.0040 * direction,
                0.0,
                0.3,
            ),
            label=1,
            volume=10_000.0,
        ))
    return rows


def test_rule_alpha_candidates_are_bounded_and_diverse() -> None:
    candidates = rule_alpha_candidates("conservative", max_candidates=12)

    assert len(candidates) == 12
    assert {candidate.family for candidate in candidates} >= {"momentum_breakout"}
    assert all(0.5 <= candidate.threshold <= 0.95 for candidate in candidates)


def test_rule_alpha_model_roundtrips_and_affects_probability(tmp_path) -> None:
    rows = _rows()
    candidate = RuleAlphaCandidate(
        name="momentum_breakout:unit",
        family="momentum_breakout",
        threshold=0.58,
        sensitivity=8.0,
        deadband=0.02,
        stop_loss_multiplier=0.14,
        take_profit_multiplier=0.10,
        cooldown_multiplier=0.0,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=0,
    )
    model = model_for_rule_alpha(rows, candidate, StrategyConfig(), market_type="futures")

    long_probability = model.predict_proba(rows[0].features)
    short_probability = model.predict_proba(_rows(rising=False)[0].features)

    assert long_probability > 0.5
    assert short_probability < 0.5
    assert model.hybrid_experts[0].params["family"] == "momentum_breakout"

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=13)

    assert loaded.rule_alpha_profile == "momentum_breakout:unit"
    assert loaded.hybrid_experts[0].kind == "rule_alpha"
    assert loaded.hybrid_experts[0].params["deadband"] == 0.02
    assert loaded.predict_proba(rows[0].features) == long_probability


def test_rule_alpha_can_use_advanced_order_flow_features(tmp_path) -> None:
    candidate = RuleAlphaCandidate(
        name="order_flow_momentum:unit",
        family="order_flow_momentum",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.01,
        stop_loss_multiplier=0.18,
        take_profit_multiplier=0.16,
        cooldown_multiplier=0.0,
        min_position_hold_bars=30,
        flat_signal_exit_grace_bars=10,
    )
    cfg = default_config_for("conservative", FEATURE_NAMES)
    feature_params = rule_alpha_feature_params(cfg)
    start = int(feature_params["order_flow_start"])
    feature_count = start + 27
    features = [0.0] * feature_count
    features[0] = 0.001
    features[1] = 0.001
    for group_start in range(start, start + 27, 9):
        features[group_start + 1] = 0.80
        features[group_start + 2] = 0.75
        features[group_start + 3] = 0.45
        features[group_start + 4] = 0.40
        features[group_start + 7] = 0.35
        features[group_start + 8] = 0.30
    row = ModelRow(timestamp=0, close=100.0, features=tuple(features), label=1, volume=1000.0)
    model = model_for_rule_alpha(
        [row],
        candidate,
        StrategyConfig(),
        market_type="futures",
        feature_params=feature_params,
    )

    assert model.hybrid_experts[0].feature_count == feature_count
    assert model.hybrid_experts[0].params["order_flow_start"] == start
    assert model.predict_proba(row.features) > 0.5

    model.rule_alpha_best_win_rate = 0.5
    model.rule_alpha_best_exit_reason_counts = {"take_profit_close": 2}
    path = tmp_path / "flow-model.json"
    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=feature_count)

    assert loaded.hybrid_experts[0].params["order_flow_window_count"] == 3
    assert loaded.rule_alpha_best_exit_reason_counts == {"take_profit_close": 2}
    assert loaded.predict_proba(row.features) == model.predict_proba(row.features)


def test_summarize_rule_alpha_trade_path_counts_exits_and_sides() -> None:
    result = BacktestResult(
        starting_cash=1000.0,
        ending_cash=1001.0,
        realized_pnl=1.0,
        win_rate=0.5,
        trades=2,
        max_drawdown=0.01,
        closed_trades=2,
        gross_exposure=100.0,
        total_fees=0.1,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        profit_factor=1.5,
        average_trade_return=0.001,
        max_consecutive_losses=1,
        trade_log=(
            {"exit_reason": "take_profit_close", "side": 1, "bars_held": 30},
            {"exit_reason": "stop_loss_close", "side": -1, "bars_held": 12},
        ),
    )

    summary = summarize_rule_alpha_trade_path(result)

    assert summary["exit_reason_counts"] == {"stop_loss_close": 1, "take_profit_close": 1}
    assert summary["side_counts"] == {"long": 1, "short": 1}
    assert summary["average_bars_held"] == 21.0


def test_rejected_rule_alpha_diagnostic_prefers_less_negative_pnl_over_activity_score() -> None:
    candidate = RuleAlphaCandidate(
        name="unit",
        family="momentum_breakout",
        threshold=0.54,
        sensitivity=6.0,
        deadband=0.02,
        stop_loss_multiplier=0.18,
        take_profit_multiplier=0.16,
        cooldown_multiplier=0.0,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=0,
    )
    better_loss = BacktestResult(
        starting_cash=1000.0,
        ending_cash=997.0,
        realized_pnl=-3.0,
        win_rate=0.0,
        trades=3,
        max_drawdown=0.003,
        closed_trades=3,
        gross_exposure=100.0,
        total_fees=0.3,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        edge_vs_buy_hold=-1.0,
        profit_factor=0.0,
        max_consecutive_losses=3,
    )
    worse_loss = BacktestResult(
        starting_cash=1000.0,
        ending_cash=990.0,
        realized_pnl=-10.0,
        win_rate=0.0,
        trades=12,
        max_drawdown=0.010,
        closed_trades=12,
        gross_exposure=100.0,
        total_fees=1.2,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        edge_vs_buy_hold=-8.0,
        profit_factor=0.0,
        max_consecutive_losses=12,
    )

    better = RuleAlphaCandidateResult(candidate, False, False, float("-inf"), 0.01, "reject", better_loss)
    worse = RuleAlphaCandidateResult(candidate, False, False, float("-inf"), 0.05, "reject", worse_loss)

    assert _diagnostic_rank_key(better) > _diagnostic_rank_key(worse)

    no_trade = BacktestResult(
        starting_cash=1000.0,
        ending_cash=1000.0,
        realized_pnl=0.0,
        win_rate=0.0,
        trades=0,
        max_drawdown=0.0,
        closed_trades=0,
        gross_exposure=0.0,
        total_fees=0.0,
        stopped_by_drawdown=False,
        max_exposure=0.0,
        trades_per_day_cap_hit=0,
        edge_vs_buy_hold=0.0,
    )
    inactive = RuleAlphaCandidateResult(candidate, False, False, float("-inf"), 0.50, "reject", no_trade)

    assert _diagnostic_rank_key(better) > _diagnostic_rank_key(inactive)


def test_rule_alpha_search_promotes_only_objective_accepted_candidate(monkeypatch) -> None:
    rows = _rows(count=48)
    strategy = StrategyConfig(
        cooldown_minutes=0,
        max_trades_per_day=100,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        liquidity_risk_enabled=False,
        dynamic_liquidity_session_enabled=False,
    )
    objective = SimpleNamespace(
        name="conservative",
        score=lambda result: result.realized_pnl,
        reject_reason=lambda result: None if result.realized_pnl > 0.0 and result.closed_trades > 0 else "no_profit",
    )
    monkeypatch.setattr("simple_ai_trading.alpha_search.get_objective", lambda _name: objective)

    report = optimize_rule_alpha_model_zoo(
        rows,
        strategy,
        objective_name="conservative",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="cpu",
        max_candidates=3,
    )

    assert report.accepted is True
    assert report.model is not None
    assert report.model.rule_alpha_evaluated_candidates == 6
    assert report.best_result is not None
    assert report.best_result.closed_trades > 0


def test_rule_alpha_search_fails_closed_when_objective_rejects(monkeypatch) -> None:
    rows = _rows(count=24)
    objective = SimpleNamespace(
        name="conservative",
        score=lambda result: result.realized_pnl,
        reject_reason=lambda _result: "forced_reject",
    )
    monkeypatch.setattr("simple_ai_trading.alpha_search.get_objective", lambda _name: objective)

    report = optimize_rule_alpha_model_zoo(
        rows,
        StrategyConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        objective_name="conservative",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="cpu",
        max_candidates=2,
    )

    assert report.accepted is False
    assert report.model is None
    assert report.evaluated_candidates == 4
    assert report.best_candidate is not None
