from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import ModelRow
import simple_ai_trading.hybrid_models as hybrid_models
from simple_ai_trading.hybrid_models import optimize_hybrid_model_zoo
from simple_ai_trading.model import (
    HybridExpert,
    HybridPrototype,
    TrainedModel,
    load_model,
    serialize_model,
)
from simple_ai_trading.types import StrategyConfig


def _model() -> TrainedModel:
    return TrainedModel(
        weights=[0.4, -0.2],
        bias=0.0,
        feature_dim=2,
        epochs=3,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
    )


def _rows(count: int = 80) -> list[ModelRow]:
    rows: list[ModelRow] = []
    for index in range(count):
        positive = index % 4 in {1, 2}
        value = 0.05 * index
        rows.append(ModelRow(
            timestamp=index * 60_000,
            close=100.0 + value + (1.0 if positive else -0.5),
            features=(value if positive else -value, -value if positive else value),
            label=1 if positive else 0,
        ))
    return rows


def test_hybrid_experts_roundtrip_and_affect_probability(tmp_path: Path) -> None:
    model = _model()
    base = model.predict_proba((1.0, -1.0))
    model.model_family = "adaptive_hybrid_model_zoo"
    model.hybrid_base_weight = 0.25
    model.hybrid_experts = [
        HybridExpert(
            name="near",
            kind="lorentzian_knn",
            weight=0.5,
            prototypes=[
                HybridPrototype(features=[1.0, -1.0], label=1, timestamp=1, close=101.0),
                HybridPrototype(features=[-1.0, 1.0], label=0, timestamp=2, close=99.0),
            ],
            k=1,
        ),
        HybridExpert(name="rules", kind="technical_confluence", weight=0.25, feature_count=2),
    ]
    hybrid = model.predict_proba((1.0, -1.0))
    assert hybrid > base

    path = tmp_path / "hybrid.json"
    serialize_model(model, path)
    loaded = load_model(path)
    assert loaded.model_family == "adaptive_hybrid_model_zoo"
    assert len(loaded.hybrid_experts) == 2
    assert loaded.predict_proba((1.0, -1.0)) == hybrid


def test_optimize_hybrid_model_zoo_returns_report() -> None:
    rows = _rows()
    model = _model()
    report = optimize_hybrid_model_zoo(
        model,
        rows[:50],
        rows[50:],
        StrategyConfig(
            risk_per_trade=0.01,
            max_position_pct=0.20,
            signal_threshold=0.50,
            min_quote_volume_usdc=1.0,
            min_trade_count_24h=1,
            min_liquidity_score=0.0,
        ),
        objective_name="risky",
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="cpu",
        score_batch_size=16,
        feature_count=2,
    )

    payload = report.asdict()
    assert payload["evaluated_profiles"] >= 1
    assert report.model.predict_proba(rows[-1].features) >= 0.0
    assert report.best_profile


def test_optimize_hybrid_model_zoo_records_expert_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest(_rows, model, _strategy, **_kwargs):
        kinds = {expert.kind for expert in getattr(model, "hybrid_experts", [])}
        pnl = 1.0
        pnl += 3.0 if "lorentzian_knn" in kinds else 0.0
        pnl += 2.0 if "rational_quadratic_kernel" in kinds else 0.0
        pnl += 1.0 if "technical_confluence" in kinds else 0.0
        return BacktestResult(
            starting_cash=1000.0,
            ending_cash=1000.0 + pnl,
            realized_pnl=pnl,
            win_rate=1.0,
            trades=6,
            max_drawdown=0.0,
            closed_trades=6,
            gross_exposure=100.0,
            total_fees=0.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=pnl,
            gross_profit=pnl,
            gross_loss=0.0,
            profit_factor=999.0,
            expectancy=pnl / 6.0,
            trade_pnls=(pnl,),
            trade_returns=(pnl / 1000.0,),
        )

    monkeypatch.setattr(hybrid_models, "run_backtest", fake_run_backtest)

    rows = _rows()
    report = optimize_hybrid_model_zoo(
        _model(),
        rows[:50],
        rows[50:],
        StrategyConfig(signal_threshold=0.50, min_quote_volume_usdc=1.0, min_liquidity_score=0.0),
        objective_name="aggressive",
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="cpu",
        score_batch_size=16,
        feature_count=2,
    )

    payload = report.asdict()
    ablations = {item.removed_expert_kind: item for item in report.ablation_results}
    assert report.accepted is True
    assert "all_hybrid_experts" in ablations
    assert "lorentzian_knn" in ablations
    assert ablations["lorentzian_knn"].delta_vs_best < 0.0
    assert payload["ablation_results"][0]["removed_expert_kind"] == "all_hybrid_experts"
