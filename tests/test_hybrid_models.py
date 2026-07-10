from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.backtest import BacktestResult, run_backtest
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


def test_payoff_internal_validation_split_purges_overlapping_labels() -> None:
    rows = _rows(1_000)
    examples = hybrid_models._PayoffTrainingExamples(
        rows=rows,
        targets=[0.0] * len(rows),
        source_indexes=list(range(len(rows))),
        meta={},
    )

    split = hybrid_models._purged_payoff_train_validation_split(examples, horizon_bars=100)

    assert split is not None
    train_rows, _train_targets, validation_rows, _validation_targets, meta = split
    assert len(validation_rows) == 200
    assert meta["validation_source_start"] == 800
    assert meta["training_source_end"] + 1 + 100 < meta["validation_source_start"]
    assert meta["internal_validation_purged_examples"] == 101
    assert len(train_rows) == 699


def test_intraday_utility_rank_groups_are_chronological_and_below_lightgbm_cap() -> None:
    rows = [
        ModelRow(timestamp=index, close=100.0, features=(0.0, 0.0), label=0)
        for index in range(20_050)
    ]

    groups = hybrid_models._intraday_utility_rank_groups(
        rows,
        duration_ms=1_000_000_000,
        max_group_rows=8192,
    )

    assert groups == [8192, 8192, 3666]
    assert sum(groups) == len(rows)
    assert max(groups) < 10_000


@pytest.mark.parametrize("training_mode", ["binary_hurdle", "daily_utility_rank"])
def test_lightgbm_payoff_expert_serializes_for_dependency_free_inference(
    tmp_path: Path,
    training_mode: str,
) -> None:
    pytest.importorskip("lightgbm")
    rows = _rows(1_000)
    long_action_targets = [
        0.6 if index % 4 in {1, 2} else -0.4
        for index in range(len(rows))
    ]
    short_action_targets = [
        0.6 if index % 4 in {0, 3} else -0.4
        for index in range(len(rows))
    ]
    targets = [
        long_value if long_value > short_value else -short_value
        for long_value, short_value in zip(long_action_targets, short_action_targets, strict=True)
    ]
    examples = hybrid_models._PayoffTrainingExamples(
        rows=rows,
        targets=targets,
        source_indexes=list(range(len(rows))),
        meta={
            "horizon_bars": 10,
            "source_rows": len(rows),
            "sampled_rows": len(rows),
            "training_examples": len(rows),
            "positive_long_rows": 500,
            "positive_short_rows": 500,
            "neutral_rows": 0,
            "clip_bps": 30.0,
            "action_clip_bps": 30.0,
            "long_action_positive_rows": 500,
            "short_action_positive_rows": 500,
            "long_action_mean_bps": 3.0,
            "short_action_mean_bps": 3.0,
            "median_interval_ms": 60_000,
        },
        long_action_targets=long_action_targets,
        short_action_targets=short_action_targets,
    )

    expert = hybrid_models._train_signed_payoff_lightgbm_ranker_expert(
        rows,
        _model(),
        horizon_bars=10,
        objective_name="conservative",
        market_type="futures",
        strategy=StrategyConfig(taker_fee_bps=4.0),
        compute_backend="cpu",
        batch_size=128,
        training_mode=training_mode,
        examples=examples,
    )

    assert expert is not None
    assert expert.kind == "signed_payoff_lightgbm_ranker"
    assert expert.params["payoff_tree_schema"] == "action_value_hurdle_v1"
    assert expert.params["payoff_training_mode"] == training_mode
    assert expert.params["training_backend_kind"] == "cpu"
    assert len(expert.params["long_classifier_tree_info"]) > 0
    assert len(expert.params["short_classifier_tree_info"]) > 0
    assert expert.params["long_enabled"] is True
    assert expert.params["short_enabled"] is True
    assert expert.params["long_gate_diagnostics"]["gate_auc"] > 0.9
    assert expert.params["short_gate_diagnostics"]["gate_auc"] > 0.9
    probability = _model()._expert_probability(expert, rows[-1].features)  # noqa: SLF001
    assert probability is not None
    assert 0.0 <= probability <= 1.0
    expert.weight = 1.0
    action_model = _model()
    action_model.hybrid_base_weight = 0.0
    action_model.hybrid_experts = [expert]
    model_path = tmp_path / "action-payoff-tree.json"
    serialize_model(action_model, model_path)
    loaded = load_model(model_path, expected_feature_dim=2)
    assert loaded.predict_proba(rows[-1].features) == pytest.approx(
        action_model.predict_proba(rows[-1].features),
        abs=1e-12,
    )


def _coherent_backtest_result(pnl: float, *, closed_trades: int = 6) -> BacktestResult:
    trade_pnls = tuple(float(pnl) / closed_trades for _ in range(closed_trades))
    trade_returns = tuple(value / 1000.0 for value in trade_pnls)
    trade_log = tuple(
        {
            "opened_at": int(index * 120_000),
            "closed_at": int(index * 120_000 + 60_000),
            "side": 1,
            "gross_notional": 100.0,
            "entry_price": 100.0,
            "exit_mark_price": 100.0 + max(value, 0.01),
            "realized_pnl": float(value),
            "net_pnl": float(value),
            "return_pct": float(return_pct),
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "exit_reason": "take_profit_close",
        }
        for index, (value, return_pct) in enumerate(zip(trade_pnls, trade_returns, strict=True))
    )
    return BacktestResult(
        starting_cash=1000.0,
        ending_cash=1000.0 + pnl,
        realized_pnl=pnl,
        win_rate=1.0,
        trades=closed_trades,
        max_drawdown=0.0,
        closed_trades=closed_trades,
        gross_exposure=100.0,
        total_fees=0.0,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        buy_hold_pnl=0.0,
        edge_vs_buy_hold=pnl,
        equity_curve=(
            {"timestamp": 0, "equity": 1000.0, "drawdown": 0.0, "position_side": 0},
            {"timestamp": 60_000 * closed_trades, "equity": 1000.0 + pnl, "drawdown": 0.0, "position_side": 0},
        ),
        trade_pnls=trade_pnls,
        trade_returns=trade_returns,
        trade_log=trade_log,
        gross_profit=pnl,
        gross_loss=0.0,
        profit_factor=999.0,
        expectancy=pnl / closed_trades,
        average_trade_return=sum(trade_returns) / len(trade_returns),
        trade_return_stdev=0.0,
        max_consecutive_losses=0,
    )


@pytest.mark.parametrize(
    ("side", "exit_mode", "future_rows", "expected_reason"),
    [
        (1, "time", ((100.05, 100.08, 100.02), (100.20, 100.22, 100.18)), "time_limit"),
        (-1, "time", ((99.95, 99.98, 99.92), (99.80, 99.82, 99.78)), "time_limit"),
        (1, "stop", ((99.00, 100.00, 97.00), (99.00, 99.00, 99.00)), "intrabar_stop_loss"),
        (-1, "stop", ((101.00, 103.00, 100.00), (101.00, 101.00, 101.00)), "intrabar_stop_loss"),
        (1, "take", ((101.00, 103.00, 100.00), (101.00, 101.00, 101.00)), "intrabar_take_profit"),
        (-1, "take", ((99.00, 100.00, 97.00), (99.00, 99.00, 99.00)), "intrabar_take_profit"),
    ],
)
def test_signed_payoff_target_matches_backtest_trade_execution(
    side: int,
    exit_mode: str,
    future_rows: tuple[tuple[float, float, float], tuple[float, float, float]],
    expected_reason: str,
) -> None:
    del exit_mode
    strategy = StrategyConfig(
        leverage=5.0,
        risk_per_trade=0.003,
        max_position_pct=0.08,
        max_asset_allocation_pct=0.20,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        signal_threshold=0.90,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=2,
        max_position_hold_bars=2,
        cooldown_minutes=0,
        unpredictability_cooldown_minutes=0,
        max_regime_unpredictability=1.0,
        max_trades_per_day=10,
        max_drawdown_limit=1.0,
        dynamic_liquidity_session_enabled=False,
        liquidity_risk_enabled=False,
    )
    prices = (
        (100.0, 100.0, 100.0),
        (100.0, 100.0, 100.0),
        future_rows[0],
        future_rows[1],
    )
    rows = [
        ModelRow(
            timestamp=index * 1_000,
            close=close,
            high=high,
            low=low,
            quote_volume=100_000_000.0,
            trade_count=1_000,
            features=(0.0, 0.0),
            label=0,
        )
        for index, (close, high, low) in enumerate(prices)
    ]
    model = _model()
    model.decision_threshold = 0.90
    model.long_decision_threshold = 0.90
    model.short_decision_threshold = 0.10
    trigger_probability = 1.0 if side > 0 else 0.0
    result = run_backtest(
        rows,
        model,
        strategy,
        market_type="futures",
        precomputed_probabilities=(trigger_probability, 0.5, 0.5, 0.5),
        precomputed_regime_scores=(0.0, 0.0, 0.0, 0.0),
        precomputed_liquidity_adjustments=((0.0, 1.0, False, False),) * 4,
    )

    assert result.closed_trades == 1
    trade = result.trade_log[0]
    assert trade["side"] == side
    assert trade["exit_reason"] == expected_reason
    expected_bps = float(trade["net_pnl"]) / float(trade["gross_notional"]) * 10_000.0
    target_bps = hybrid_models.path_net_edge_bps(
        rows,
        signal_index=0,
        horizon_bars=2,
        side=side,
        strategy=strategy,
        market_type="futures",
        symbol_profile=None,
    )

    assert target_bps == pytest.approx(expected_bps, rel=1e-12, abs=1e-12)


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


def test_signed_payoff_ranker_roundtrip_and_scores_both_sides(tmp_path: Path) -> None:
    model = _model()
    model.model_family = "adaptive_hybrid_model_zoo"
    model.hybrid_base_weight = 0.0
    model.hybrid_experts = [
        HybridExpert(
            name="payoff",
            kind="signed_payoff_ranker",
            weight=1.0,
            feature_count=2,
            params={
                "input_dim": 2,
                "weights": [1.0, -1.0],
                "bias": 0.0,
                "clip_bps": 20.0,
                "deadband_bps": 0.0,
                "sensitivity": 5.0,
            },
        )
    ]

    long_probability = model.predict_proba((1.0, -1.0))
    short_probability = model.predict_proba((-1.0, 1.0))
    assert long_probability > 0.90
    assert short_probability < 0.10

    path = tmp_path / "payoff.json"
    serialize_model(model, path)
    loaded = load_model(path)
    assert loaded.hybrid_experts[0].kind == "signed_payoff_ranker"
    assert loaded.predict_proba((1.0, -1.0)) == long_probability


def test_signed_payoff_mlp_ranker_roundtrip_and_scores_both_sides(tmp_path: Path) -> None:
    model = _model()
    model.model_family = "adaptive_hybrid_model_zoo"
    model.hybrid_base_weight = 0.0
    model.hybrid_experts = [
        HybridExpert(
            name="payoff_mlp",
            kind="signed_payoff_mlp_ranker",
            weight=1.0,
            feature_count=2,
            params={
                "input_dim": 2,
                "layers": [
                    {
                        "weights": [[1.0], [-1.0]],
                        "bias": [0.0],
                        "activation": "tanh",
                    },
                ],
                "clip_bps": 20.0,
                "deadband_bps": 0.0,
                "sensitivity": 6.0,
            },
        )
    ]

    long_probability = model.predict_proba((1.0, -1.0))
    short_probability = model.predict_proba((-1.0, 1.0))
    assert long_probability > 0.95
    assert short_probability < 0.05

    path = tmp_path / "payoff_mlp.json"
    serialize_model(model, path)
    loaded = load_model(path)
    assert loaded.hybrid_experts[0].kind == "signed_payoff_mlp_ranker"
    assert loaded.predict_proba((1.0, -1.0)) == long_probability


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
    assert payload["selection_search_rows"] == len(rows[50:])
    assert payload["selection_full_rows"] == len(rows[50:])
    assert report.model.predict_proba(rows[-1].features) >= 0.0
    assert report.best_profile
    assert len(report.profile_results) >= 1
    assert report.profile_results[0].profile


def test_conservative_hybrid_profiles_include_low_base_rescue_models() -> None:
    profiles = {profile.name: profile for profile in hybrid_models._profiles_for("conservative")}

    assert profiles["technical_rescue_core"].base <= 0.10
    assert profiles["technical_rescue_core"].technical >= 0.70
    assert profiles["neighbor_kernel_rescue"].base <= 0.15
    assert profiles["neighbor_kernel_rescue"].lorentzian > profiles["neighbor_kernel_rescue"].base
    assert profiles["balanced_rescue_committee"].base == pytest.approx(0.25)
    assert profiles["neural_guarded_committee"].neural > 0.0
    assert profiles["neural_confirmed_rescue"].base < profiles["neural_confirmed_rescue"].neural
    assert profiles["signed_payoff_mlp_inverse_direct"].invert_probability is True
    assert profiles["signed_payoff_mlp_inverse_direct"].selection_eligible is False
    assert profiles["signed_payoff_tree_direct"].selection_eligible is True


def test_model_specific_payoff_profile_does_not_silently_substitute_expert_family() -> None:
    profile = hybrid_models._WeightProfile(
        name="signed_payoff_tree_direct",
        base=0.0,
        lorentzian=0.0,
        kernel=0.0,
        technical=0.0,
        payoff=1.0,
    )
    linear = HybridExpert(
        name="linear",
        kind="signed_payoff_ranker",
        weight=0.0,
        feature_count=2,
    )

    reason = hybrid_models._skip_profile_for_large_payoff_search(
        profile,
        selection_rows=_rows(),
        payoff_experts=(linear,),
    )
    experts = hybrid_models._experts_for_profile(
        profile,
        (),
        feature_dim=2,
        feature_count=2,
        bandwidth=1.0,
        objective_name="conservative",
        payoff_experts=(linear,),
    )

    assert reason == "required_payoff_expert_unavailable:signed_payoff_lightgbm_ranker"
    assert all(expert.kind != "signed_payoff_ranker" for expert in experts)


def test_neural_profile_skips_when_required_neural_expert_is_unavailable() -> None:
    profile = hybrid_models._WeightProfile(
        name="signed_payoff_neural_gate",
        base=0.0,
        lorentzian=0.0,
        kernel=0.0,
        technical=0.0,
        neural=0.25,
        payoff=0.75,
    )

    reason = hybrid_models._skip_profile_for_large_payoff_search(
        profile,
        selection_rows=_rows(),
        payoff_experts=(),
        neural_expert=None,
    )

    assert reason == "required_neural_expert_unavailable"


def test_inverse_diagnostic_profile_cannot_win_model_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    payoff = HybridExpert(
        name="payoff",
        kind="signed_payoff_ranker",
        weight=0.0,
        feature_count=2,
        params={"input_dim": 2, "weights": [1.0, -1.0], "clip_bps": 20.0},
    )
    profiles = (
        hybrid_models._WeightProfile(
            name="eligible_payoff",
            base=0.0,
            lorentzian=0.0,
            kernel=0.0,
            technical=0.0,
            payoff=1.0,
        ),
        hybrid_models._WeightProfile(
            name="inverse_diagnostic",
            base=0.0,
            lorentzian=0.0,
            kernel=0.0,
            technical=0.0,
            payoff=1.0,
            invert_probability=True,
            selection_eligible=False,
        ),
    )
    monkeypatch.setattr(hybrid_models, "_profiles_for", lambda _objective: profiles)
    monkeypatch.setattr(hybrid_models, "_train_dense_mlp_expert", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        hybrid_models,
        "_train_signed_payoff_ranker_experts",
        lambda *_args, **_kwargs: (payoff,),
    )

    def fake_base(model, *_args, **_kwargs):
        return 1.0, _coherent_backtest_result(1.0), model

    def fake_payoff(model, *_args, **_kwargs):
        score = 100.0 if model.probability_inverted else 2.0
        return score, _coherent_backtest_result(score), model

    monkeypatch.setattr(hybrid_models, "_evaluate_model", fake_base)
    monkeypatch.setattr(hybrid_models, "_evaluate_model_with_threshold_calibration", fake_payoff)

    report = optimize_hybrid_model_zoo(
        _model(),
        _rows()[:50],
        _rows()[50:],
        StrategyConfig(signal_threshold=0.50, min_quote_volume_usdc=1.0, min_liquidity_score=0.0),
        objective_name="conservative",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="cpu",
        score_batch_size=16,
        feature_count=2,
    )

    assert report.best_profile == "eligible_payoff"
    attempts = {item.profile: item for item in report.profile_results}
    assert attempts["inverse_diagnostic"].search_score == pytest.approx(100.0)
    assert attempts["inverse_diagnostic"].selected_search_best is False


def test_hybrid_selection_search_keeps_week_scale_second_level_path() -> None:
    assert hybrid_models._hybrid_selection_search_limit("conservative") >= 151_000
    assert hybrid_models._hybrid_selection_search_limit("regular") >= 151_000
    assert hybrid_models._hybrid_selection_search_limit("aggressive") >= 151_000


def test_payoff_horizon_detection_uses_adjacent_second_level_rows() -> None:
    rows = [
        ModelRow(timestamp=index * 1000, close=100.0, features=(0.0, 0.0), label=0)
        for index in range(302_365)
    ]

    assert hybrid_models._median_row_interval_ms(rows) == 1000
    assert hybrid_models._payoff_horizon_bars(rows, "conservative") == (15, 60, 180)
    assert hybrid_models._payoff_horizon_bars(
        rows,
        "conservative",
        max_position_hold_bars=90,
    ) == (90,)


def test_optimize_hybrid_model_zoo_records_expert_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest(_rows, model, _strategy, **_kwargs):
        kinds = {expert.kind for expert in getattr(model, "hybrid_experts", [])}
        pnl = 1.0
        pnl += 3.0 if "lorentzian_knn" in kinds else 0.0
        pnl += 2.0 if "rational_quadratic_kernel" in kinds else 0.0
        pnl += 1.0 if "technical_confluence" in kinds else 0.0
        return _coherent_backtest_result(pnl)

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


def test_optimize_hybrid_model_zoo_can_select_dense_mlp_expert(monkeypatch: pytest.MonkeyPatch) -> None:
    neural = HybridExpert(
        name="dense_mlp_neural_edge",
        kind="dense_mlp",
        weight=0.0,
        feature_count=2,
        params={
            "input_dim": 2,
            "layers": [
                {"weights": [[1.0], [1.0]], "bias": [0.0], "activation": "sigmoid"},
            ],
            "output_activation": "sigmoid",
            "training_backend_kind": "directml",
        },
    )

    monkeypatch.setattr(hybrid_models, "_train_dense_mlp_expert", lambda *_args, **_kwargs: neural)

    def fake_run_backtest(_rows, model, _strategy, **_kwargs):
        kinds = {expert.kind for expert in getattr(model, "hybrid_experts", [])}
        pnl = 8.0 if "dense_mlp" in kinds else 1.0
        return _coherent_backtest_result(pnl)

    monkeypatch.setattr(hybrid_models, "run_backtest", fake_run_backtest)

    rows = _rows()
    report = optimize_hybrid_model_zoo(
        _model(),
        rows[:50],
        rows[50:],
        StrategyConfig(signal_threshold=0.50, min_quote_volume_usdc=1.0, min_liquidity_score=0.0),
        objective_name="conservative",
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=16,
        feature_count=2,
    )

    assert report.accepted is True
    assert "neural" in report.best_profile
    assert any(expert.kind == "dense_mlp" and expert.weight > 0.0 for expert in report.model.hybrid_experts)


def test_optimize_hybrid_model_zoo_can_select_signed_payoff_ranker(monkeypatch: pytest.MonkeyPatch) -> None:
    payoff = HybridExpert(
        name="signed_payoff_ranker_60s",
        kind="signed_payoff_ranker",
        weight=0.0,
        feature_count=2,
        params={
            "input_dim": 2,
            "weights": [1.0, -1.0],
            "bias": 0.0,
            "clip_bps": 20.0,
            "training_examples": 64,
            "positive_long_rows": 32,
            "positive_short_rows": 28,
            "training_backend_kind": "directml",
        },
    )

    monkeypatch.setattr(hybrid_models, "_train_dense_mlp_expert", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hybrid_models, "_train_signed_payoff_ranker_experts", lambda *_args, **_kwargs: (payoff,))

    def fake_run_backtest(_rows, model, _strategy, **_kwargs):
        kinds = {expert.kind for expert in getattr(model, "hybrid_experts", [])}
        pnl = 9.0 if "signed_payoff_ranker" in kinds else 1.0
        return _coherent_backtest_result(pnl)

    monkeypatch.setattr(hybrid_models, "run_backtest", fake_run_backtest)

    rows = _rows()
    report = optimize_hybrid_model_zoo(
        _model(),
        rows[:50],
        rows[50:],
        StrategyConfig(signal_threshold=0.50, min_quote_volume_usdc=1.0, min_liquidity_score=0.0),
        objective_name="regular",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=16,
        feature_count=2,
    )

    assert report.accepted is True
    assert "signed_payoff" in report.best_profile
    assert any(expert.kind == "signed_payoff_ranker" and expert.weight > 0.0 for expert in report.model.hybrid_experts)
    assert report.payoff_expert_params[0]["training_backend_kind"] == "directml"


def test_payoff_hybrid_threshold_calibration_uses_adaptive_probability_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_calibrate(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            accepted=False,
            best_score=1.0,
            baseline_score=0.0,
            best_realized_pnl=2.0,
            baseline_realized_pnl=0.0,
            best_threshold=0.58,
            best_long_threshold=0.58,
            best_short_threshold=0.42,
        )

    monkeypatch.setattr(hybrid_models, "calibrate_threshold_for_backtest", fake_calibrate)
    monkeypatch.setattr(
        hybrid_models,
        "_evaluate_model",
        lambda model, rows, strategy, **kwargs: (float("-inf"), _coherent_backtest_result(1.0), model),
    )

    score, result, model = hybrid_models._evaluate_model_with_threshold_calibration(
        _model(),
        _rows(),
        StrategyConfig(signal_threshold=0.50),
        objective_name="conservative",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=16,
    )

    assert captured["adaptive_probability_thresholds"] is True
    assert captured["max_adaptive_thresholds"] == 12
    assert score == float("-inf")
    assert result is not None
    assert model.threshold_source == "hybrid_diagnostic_threshold_search"
