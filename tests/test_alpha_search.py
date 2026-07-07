from __future__ import annotations

from types import SimpleNamespace

from simple_ai_trading.alpha_search import (
    RuleAlphaCandidate,
    model_for_rule_alpha,
    optimize_rule_alpha_model_zoo,
    rule_alpha_candidates,
)
from simple_ai_trading.features import ModelRow
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
