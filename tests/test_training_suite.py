"""Comprehensive unit tests for the multi-objective training suite."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import training_suite
from simple_ai_trading.advanced_model import default_config_for
from simple_ai_trading.api import Candle
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import ModelRow
from simple_ai_trading.model import TrainedModel
from simple_ai_trading.objective import (
    ObjectiveSpec,
    ObjectiveTraining,
    get_objective,
)
from simple_ai_trading.training_suite import (
    CandidateParams,
    ObjectiveOutcome,
    SuiteReport,
    TrainingSuiteRejected,
    _calibration_split,
    _candidate_grid,
    _default_training,
    _effective_threshold_for_market,
    _ensemble_seed_pack,
    _calibrate_candidate_threshold,
    _evaluate_candidate,
    _feature_config_for_candidate,
    _feature_ablation_report,
    _local_refinement_candidates,
    _purged_walk_forward_gate,
    _purged_walk_forward_splits,
    _refine_threshold_on_selection_rows,
    _risk_aware_best,
    _risk_non_degrading,
    _strategy_for_candidate,
    _threshold_guard,
    _threshold_values,
    _walk_forward_split,
    describe_candidate_grid,
    preview_candidates,
    rank_report,
    run_training_suite,
    train_for_objective,
)
from simple_ai_trading.types import StrategyConfig


# ----- helpers --------------------------------------------------------------


def _synthetic_candles(n: int = 500, base: float = 100.0) -> list[Candle]:
    candles: list[Candle] = []
    price = base
    for i in range(n):
        open_ = price
        close = price * (1.0 + 0.0005 * math.sin(i / 5.0) + 0.0002)
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        candles.append(Candle(
            open_time=i * 60_000,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1.0 + (i % 5) * 0.1,
            close_time=i * 60_000 + 60_000,
        ))
        price = close
    return candles


def _fake_trained_model(dim: int = 4) -> TrainedModel:
    return TrainedModel(
        weights=[0.0] * dim,
        bias=0.0,
        feature_dim=dim,
        epochs=1,
        feature_means=[0.0] * dim,
        feature_stds=[1.0] * dim,
    )


def _rows(n: int) -> list[ModelRow]:
    return [ModelRow(timestamp=i, close=1.0, features=(0.1, 0.2), label=i % 2) for i in range(n)]


@pytest.fixture
def _passing_selection_risk_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def report(best, _ranked_pool, **counts):
        selected = float(best.get("score", 1.0))
        effective_trials = max(1, sum(int(value) for value in counts.values()))
        return {
            "passed": True,
            "reason": None,
            "reasons": [],
            "effective_trials": effective_trials,
            "selected_score": selected,
            "trial_penalty": 0.0,
            "deflated_score": selected,
            "overfit_diagnostics": {
                "status": "available",
                "method": "two_panel_cscv_proxy",
                "passed": True,
                "reason": None,
                "probability_backtest_overfit": 0.0,
                "max_probability_backtest_overfit": 0.5,
                "candidate_count": 3,
                "split_count": 2,
                "overfit_splits": 0,
                "splits": [],
            },
        }

    monkeypatch.setattr(training_suite, "_selection_risk_report", report)


# ----- CandidateParams ------------------------------------------------------


def test_candidate_params_asdict_keys() -> None:
    params = CandidateParams(
        epochs=10, learning_rate=0.01, l2_penalty=0.001,
        signal_threshold=0.6, stop_loss_pct=0.02, take_profit_pct=0.03,
        risk_per_trade=0.01, confidence_beta=0.9, seed=11,
    )
    d = params.asdict()
    expected_keys = {
        "epochs", "learning_rate", "l2_penalty",
        "signal_threshold", "stop_loss_pct", "take_profit_pct", "risk_per_trade",
        "confidence_beta", "label_threshold_multiplier", "label_lookahead_multiplier",
        "label_mode", "focal_gamma", "seed",
    }
    assert set(d.keys()) == expected_keys


# ----- _candidate_grid ------------------------------------------------------


def test_candidate_grid_returns_unique_deduped_list() -> None:
    training = get_objective("default").training
    grid = _candidate_grid(training)
    assert len(grid) == 2880
    # dedupe check: no two entries share identical tuple of values
    tuples = [tuple(c.asdict().values()) for c in grid]
    assert len(tuples) == len(set(tuples))
    # the grid should include variation in epochs/lr/threshold
    epoch_set = {c.epochs for c in grid}
    lr_set = {c.learning_rate for c in grid}
    l2_set = {c.l2_penalty for c in grid}
    threshold_set = {c.signal_threshold for c in grid}
    confidence_set = {c.confidence_beta for c in grid}
    label_threshold_set = {c.label_threshold_multiplier for c in grid}
    label_lookahead_set = {c.label_lookahead_multiplier for c in grid}
    label_mode_set = {c.label_mode for c in grid}
    focal_gamma_set = {c.focal_gamma for c in grid}
    seed_set = {c.seed for c in grid}
    assert len(epoch_set) >= 2
    assert len(lr_set) >= 2
    assert len(l2_set) >= 2
    assert len(threshold_set) >= 2
    assert min(threshold_set) == pytest.approx(training.signal_threshold - 0.08)
    assert confidence_set == {0.70, 0.85, 1.0}
    assert label_threshold_set == {0.10, 0.75, 1.0, 1.40}
    assert label_lookahead_set == {0.25, 0.75, 1.0, 1.75}
    assert label_mode_set == {
        "downside_event_volatility_triple_barrier",
        "downside_forward_return",
        "event_volatility_triple_barrier",
        "forward_return",
        "triple_barrier",
    }
    assert sum(1 for candidate in grid if candidate.label_mode == "forward_return") == 864
    assert sum(1 for candidate in grid if candidate.label_mode == "triple_barrier") == 864
    assert sum(1 for candidate in grid if candidate.label_mode == "event_volatility_triple_barrier") == 576
    assert sum(1 for candidate in grid if candidate.label_mode == "downside_forward_return") == 288
    assert sum(
        1 for candidate in grid
        if candidate.label_mode == "downside_event_volatility_triple_barrier"
    ) == 288
    assert focal_gamma_set == {0.0, 1.0, 1.5, 2.0}
    assert sum(1 for candidate in grid if candidate.focal_gamma > 0.0) == 2016
    assert seed_set == {7}
    event_candidate = next(
        candidate for candidate in grid
        if candidate.label_mode == "event_volatility_triple_barrier"
    )
    event_cfg = _feature_config_for_candidate(
        default_config_for("conservative", StrategyConfig().enabled_features),
        event_candidate,
    )
    assert event_cfg.label_volatility_window >= 6
    assert event_cfg.label_volatility_multiplier > 0.0


def test_candidate_grid_dedupes_colliding_entries() -> None:
    """Force a collision by zeroing the learning rate so the two lr options collapse."""

    colliding = ObjectiveTraining(
        epochs=200,
        learning_rate=0.0,  # both learning-rate options equal 0.0 -> dedup
        l2_penalty=1e-3,
        signal_threshold=0.5,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        max_position_pct=0.2,
        max_trades_per_day=12,
        leverage=1.0,
        cooldown_minutes=5,
        calibrate_threshold=True,
        walk_forward_train=300,
        walk_forward_test=80,
        walk_forward_step=30,
    )
    grid = _candidate_grid(colliding)
    # All candidates distinct after dedup
    tuples = [tuple(c.asdict().values()) for c in grid]
    assert len(tuples) == len(set(tuples))
    assert len(grid) == 1440


# ----- calibration helpers --------------------------------------------------


def test_calibration_split_small_and_large_rows() -> None:
    train_small, calibration_small = _calibration_split(_rows(12))
    assert len(train_small) == 12
    assert calibration_small == []

    train_large, calibration_large = _calibration_split(_rows(100), ratio=0.2)
    assert len(train_large) == 80
    assert len(calibration_large) == 20
    assert train_large[-1].timestamp == 79
    assert calibration_large[0].timestamp == 80

    purged_train, purged_calibration = _calibration_split(_rows(100), ratio=0.2, purge_gap=9)
    assert len(purged_train) == 71
    assert purged_train[-1].timestamp == 70
    assert purged_calibration[0].timestamp == 80


def test_threshold_guard_accepts_stable_or_sharper_candidates() -> None:
    class Report:
        def __init__(self, *, accuracy: float, f1: float, precision: float):
            self.accuracy = accuracy
            self.f1 = f1
            self.precision = precision

    baseline = Report(accuracy=0.60, f1=0.50, precision=0.45)
    stable = Report(accuracy=0.58, f1=0.46, precision=0.20)
    sharper = Report(accuracy=0.63, f1=0.10, precision=0.44)
    rejected = Report(accuracy=0.50, f1=0.30, precision=0.20)
    zero_hit = SimpleNamespace(accuracy=0.90, f1=0.0, precision=0.0, true_positive=0, false_negative=3)

    assert _threshold_guard(baseline, stable) is True
    assert _threshold_guard(baseline, sharper) is True
    assert _threshold_guard(baseline, rejected) is False
    assert _threshold_guard(baseline, zero_hit) is False


def test_threshold_values_clamps_single_step_and_expands_inverted_range() -> None:
    assert _threshold_values(0.2, 0.8, 1, 1.4) == [1.0]

    values = _threshold_values(0.7, 0.6, 3, 0.65)

    assert values == sorted(values)
    assert values[0] == 0.65
    assert values[-1] == 0.71


def test_effective_threshold_for_market_clamps_futures_to_neutral_band() -> None:
    assert _effective_threshold_for_market(0.05, "futures") == pytest.approx(0.5)
    assert _effective_threshold_for_market(0.61, "futures") == pytest.approx(0.61)
    assert _effective_threshold_for_market(0.05, "spot") == pytest.approx(0.05)


def test_calibrate_candidate_threshold_without_rows_uses_strategy_threshold() -> None:
    model = _fake_trained_model(2)
    strategy = StrategyConfig(signal_threshold=0.62)

    threshold, source, score = _calibrate_candidate_threshold(
        model,
        [],
        strategy,
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.62)
    assert source == "strategy"
    assert score is None


def test_calibrate_candidate_threshold_without_rows_clamps_futures_threshold() -> None:
    model = _fake_trained_model(2)
    strategy = StrategyConfig(signal_threshold=0.12)

    threshold, source, score = _calibrate_candidate_threshold(
        model,
        [],
        strategy,
        market_type="futures",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.5)
    assert source == "strategy"
    assert score is None


def test_calibrate_candidate_threshold_accepts_profit_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.63, f1=0.46, precision=0.44),
    ]
    observed_calibration_kwargs: dict[str, object] = {}

    def fake_profit_calibration(*a, **k):
        observed_calibration_kwargs.update(k)
        return SimpleNamespace(
            accepted=True,
            threshold=0.72,
            best_threshold=0.72,
            score=4.25,
            realized_pnl=4.25,
            closed_trades=3,
        )

    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.41)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(training_suite, "calibrate_threshold_for_backtest", fake_profit_calibration)

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=64,
    )

    assert threshold == pytest.approx(0.72)
    assert source == "profit_backtest"
    assert score == pytest.approx(4.25)
    assert observed_calibration_kwargs["compute_backend"] == "directml"
    assert observed_calibration_kwargs["score_batch_size"] == 64


def test_calibrate_candidate_threshold_rejects_weak_profit_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.50, f1=0.20, precision=0.20),
        SimpleNamespace(accuracy=0.52, f1=0.25, precision=0.22),
    ]

    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.43)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(
        training_suite,
        "calibrate_threshold_for_backtest",
        lambda *a, **k: SimpleNamespace(
            accepted=True,
            threshold=0.81,
            best_threshold=0.81,
            score=1.5,
            baseline_score=-0.5,
            realized_pnl=1.5,
            closed_trades=2,
        ),
    )

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.60)
    assert source == "strategy"
    assert score == pytest.approx(-0.5)


def test_calibrate_candidate_threshold_rejects_unprofitable_profit_report(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.63, f1=0.46, precision=0.44),
        SimpleNamespace(accuracy=0.63, f1=0.46, precision=0.44),
    ]
    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.43)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(
        training_suite,
        "calibrate_threshold_for_backtest",
        lambda *a, **k: SimpleNamespace(
            accepted=True,
            threshold=0.81,
            best_threshold=0.81,
            score=-50.0,
            baseline_score=-100.0,
            realized_pnl=0.0,
            closed_trades=0,
        ),
    )

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.60)
    assert source == "strategy"
    assert score == pytest.approx(-100.0)


def test_calibrate_candidate_threshold_accepts_safe_classification_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.58, f1=0.46, precision=0.44),
        SimpleNamespace(accuracy=0.58, f1=0.46, precision=0.44),
    ]

    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.43)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(
        training_suite,
        "calibrate_threshold_for_backtest",
        lambda *a, **k: SimpleNamespace(
            accepted=False,
            threshold=0.60,
            best_threshold=0.43,
            score=0.2,
            baseline_score=0.2,
            realized_pnl=0.0,
            closed_trades=0,
        ),
    )

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.43)
    assert source == "classification_f1"
    assert score == pytest.approx(0.2)


def test_refine_threshold_on_selection_rows_promotes_positive_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str | None, int]] = []
    probability_calls: list[tuple[int, str | None, int]] = []

    def fake_evaluate(*_args, **kwargs):
        threshold = float(kwargs.get("threshold", 0.5))
        if threshold <= 0.05:
            return SimpleNamespace(accuracy=0.60, f1=0.35, precision=0.30, true_positive=3, false_negative=2)
        return SimpleNamespace(accuracy=0.40, f1=0.20, precision=0.20, true_positive=1, false_negative=3)

    def fake_run_backtest(_rows, model, *_args, **_kwargs):
        observed.append((_kwargs.get("compute_backend"), _kwargs.get("score_batch_size")))
        if float(model.decision_threshold or 0.0) <= 0.05:
            return SimpleNamespace(realized_pnl=1.75, closed_trades=3, max_drawdown=0.01)
        return SimpleNamespace(realized_pnl=-0.05, closed_trades=2, max_drawdown=0.02)

    def fake_probabilities(rows, *_args, **kwargs):
        probability_calls.append((len(rows), kwargs.get("compute_backend"), kwargs.get("batch_size")))
        return [0.90 for _ in rows], SimpleNamespace(kind="directml")

    monkeypatch.setattr(training_suite, "evaluate_classification", fake_evaluate)
    monkeypatch.setattr(training_suite, "_backtest_probabilities", fake_probabilities)
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    refined = _refine_threshold_on_selection_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=64,
    )

    assert refined is not None
    threshold, source, score = refined
    assert threshold == pytest.approx(0.05)
    assert source == "selection_profit_backtest"
    assert score == pytest.approx(1.75)
    assert probability_calls == [(40, "directml", 64)]
    assert observed
    assert all(item == ("directml", 64) for item in observed)


def test_refine_threshold_on_selection_rows_clamps_futures_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_thresholds: list[float] = []

    monkeypatch.setattr(
        training_suite,
        "evaluate_classification",
        lambda *_a, **_k: SimpleNamespace(
            accuracy=0.70,
            f1=0.70,
            precision=0.70,
            true_positive=3,
            false_negative=0,
        ),
    )
    monkeypatch.setattr(
        training_suite,
        "_backtest_probabilities",
        lambda rows, *_a, **_k: ([0.90 for _ in rows], SimpleNamespace(kind="cpu")),
    )

    def fake_run_backtest(_rows, model, *_args, **_kwargs):
        threshold = float(model.decision_threshold or 0.0)
        observed_thresholds.append(threshold)
        return SimpleNamespace(realized_pnl=1.0, closed_trades=3, max_drawdown=0.01)

    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    refined = _refine_threshold_on_selection_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.10),
        market_type="futures",
        starting_cash=1000.0,
    )

    assert refined is not None
    threshold, source, _score = refined
    assert threshold >= 0.5
    assert source == "selection_profit_backtest"
    assert observed_thresholds
    assert min(observed_thresholds) >= 0.5


def test_refine_threshold_on_selection_rows_rejects_losing_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_suite,
        "_backtest_probabilities",
        lambda rows, *_a, **_k: ([0.90 for _ in rows], SimpleNamespace(kind="cpu")),
    )
    monkeypatch.setattr(
        training_suite,
        "run_backtest",
        lambda *_a, **_k: SimpleNamespace(realized_pnl=-0.01, closed_trades=2, max_drawdown=0.01),
    )

    refined = _refine_threshold_on_selection_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert refined is None


def test_refine_threshold_validation_wrapper_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_refine(*_args, **kwargs):
        observed.update(kwargs)
        return 0.5, "selection_profit_backtest", 1.0

    monkeypatch.setattr(training_suite, "_refine_threshold_on_selection_rows", fake_refine)

    refined = training_suite._refine_threshold_on_validation_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=12,
    )

    assert refined == (0.5, "selection_profit_backtest", 1.0)
    assert observed["compute_backend"] == "directml"
    assert observed["score_batch_size"] == 12


# ----- _walk_forward_split --------------------------------------------------


def test_walk_forward_split_small_rows_returns_copies() -> None:
    rows = _rows(5)
    train, test = _walk_forward_split(rows)
    assert train == rows
    assert test == rows
    # independent lists
    assert train is not rows


def test_walk_forward_split_large_rows_splits_properly() -> None:
    rows = _rows(100)
    train, test = _walk_forward_split(rows, eval_ratio=0.25)
    assert len(train) + len(test) == len(rows)
    assert len(test) >= 5


def test_purged_walk_forward_splits_apply_gap() -> None:
    feature_cfg = default_config_for("regular", ())
    training = _default_training(get_objective("regular"))
    folds = _purged_walk_forward_splits(_rows(500), training, feature_cfg)

    assert folds
    first = folds[0]
    assert first["test_start"] - first["train_end"] >= feature_cfg.label_lookahead
    assert len(first["train_rows"]) >= 80
    assert len(first["test_rows"]) >= 30


def test_purged_walk_forward_gate_rejects_failed_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fake_trained_model(2)

    monkeypatch.setattr(
        training_suite,
        "train_advanced",
        lambda rows, cfg, **kwargs: (model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.60, "test", 1.0),
    )
    monkeypatch.setattr(training_suite, "_refine_threshold_on_selection_rows", lambda *a, **k: None)
    results = iter([
        _make_result(realized_pnl=20.0, closed_trades=5, edge_vs_buy_hold=5.0),
        _make_result(realized_pnl=-5.0, ending_cash=995.0, closed_trades=5, edge_vs_buy_hold=-10.0),
    ])

    def fake_backtest(*_args, **_kwargs):
        try:
            return next(results)
        except StopIteration:
            return _make_result(realized_pnl=-5.0, ending_cash=995.0, closed_trades=5, edge_vs_buy_hold=-10.0)

    monkeypatch.setattr(training_suite, "run_backtest", fake_backtest)
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.6,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
    )

    gate = _purged_walk_forward_gate(
        candidate,
        _rows(500),
        StrategyConfig(),
        default_config_for("regular", ()),
        get_objective("regular"),
        _default_training(get_objective("regular")),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert gate["passed"] is False
    assert gate["reason"] == "purged_walk_forward_fold_failed"
    assert gate["accepted_folds"] < gate["fold_count"]
    assert gate["worst_realized_pnl"] == pytest.approx(-5.0)


# ----- _default_training fallback ------------------------------------------


def test_default_training_with_missing_metadata() -> None:
    # ObjectiveSpec requires a scorer; provide a no-op lambda
    spec = ObjectiveSpec(
        name="custom",
        label="Custom",
        summary="s",
        long_description="d",
        scorer=lambda _r: 0.0,
        training=None,
    )
    training = _default_training(spec)
    assert isinstance(training, ObjectiveTraining)
    assert training.epochs == 200


def test_default_training_uses_metadata_when_present() -> None:
    spec = get_objective("default")
    training = _default_training(spec)
    assert training is spec.training


# ----- _strategy_for_candidate ---------------------------------------------


def test_strategy_for_candidate_applies_overlays() -> None:
    base = StrategyConfig()
    params = CandidateParams(
        epochs=77, learning_rate=0.05, l2_penalty=0.002,
        signal_threshold=0.7, stop_loss_pct=0.05, take_profit_pct=0.06,
        risk_per_trade=0.02, confidence_beta=0.77,
    )
    training = get_objective("default").training
    strat = _strategy_for_candidate(base, params, training)
    assert isinstance(strat, StrategyConfig)
    assert strat.training_epochs == 77
    assert strat.signal_threshold == pytest.approx(0.7)
    assert strat.stop_loss_pct == pytest.approx(0.05)
    assert strat.take_profit_pct == pytest.approx(0.06)
    assert strat.risk_per_trade == pytest.approx(0.02)
    assert strat.confidence_beta == pytest.approx(0.77)
    assert strat.leverage == training.leverage
    assert strat.cooldown_minutes == training.cooldown_minutes


# ----- train_for_objective: happy path with fake runner --------------------


def _fake_trade_pnls(realized_pnl: float, closed_trades: int) -> tuple[float, ...]:
    count = max(0, int(closed_trades))
    if count <= 0:
        return ()
    if realized_pnl < 0.0:
        return tuple(float(realized_pnl) / count for _ in range(count))
    weights = [float(index + 1) for index in range(count)]
    pnls = [float(realized_pnl) * weight / sum(weights) for weight in weights]
    pnls[-1] += float(realized_pnl) - sum(pnls)
    return tuple(pnls)


def _fake_equity_curve(starting_cash: float, ending_cash: float, max_drawdown: float) -> tuple[dict[str, float | int], ...]:
    drawdown = max(0.0, min(1.0, float(max_drawdown)))
    if drawdown <= 0.0:
        return (
            {"timestamp": 0, "equity": float(starting_cash), "drawdown": 0.0, "position_side": 0},
            {"timestamp": 60_000, "equity": float(ending_cash), "drawdown": 0.0, "position_side": 0},
        )
    trough = max(0.0, float(starting_cash) * (1.0 - drawdown))
    final_drawdown = 0.0 if ending_cash >= starting_cash else (starting_cash - ending_cash) / max(1.0, starting_cash)
    return (
        {"timestamp": 0, "equity": float(starting_cash), "drawdown": 0.0, "position_side": 0},
        {"timestamp": 60_000, "equity": float(trough), "drawdown": drawdown, "position_side": 0},
        {"timestamp": 120_000, "equity": float(ending_cash), "drawdown": float(final_drawdown), "position_side": 0},
    )


def _fake_trade_log(pnls: tuple[float, ...], returns: tuple[float, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, (net_pnl, return_pct) in enumerate(zip(pnls, returns, strict=True)):
        entry_fee = 0.05
        exit_fee = 0.05
        realized = net_pnl + entry_fee + exit_fee
        rows.append({
            "opened_at": int(index * 120_000),
            "closed_at": int(index * 120_000 + 60_000),
            "side": 1,
            "gross_notional": 100.0,
            "entry_price": 100.0,
            "exit_mark_price": max(0.01, 100.0 + realized),
            "realized_pnl": float(realized),
            "net_pnl": float(net_pnl),
            "return_pct": float(return_pct),
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "exit_reason": "take_profit_close" if net_pnl > 0.0 else "stop_loss_close",
        })
    return tuple(rows)


def _make_result(**overrides) -> BacktestResult:
    starting_cash = float(overrides.get("starting_cash", 1000.0))
    realized_pnl = float(overrides.get("realized_pnl", 50.0))
    ending_cash = float(overrides.get("ending_cash", starting_cash + realized_pnl))
    closed_trades = int(overrides.get("closed_trades", overrides.get("trades", 5)))
    trade_pnls = tuple(float(value) for value in overrides.get("trade_pnls", _fake_trade_pnls(realized_pnl, closed_trades)))
    trade_returns = tuple(float(value) for value in overrides.get("trade_returns", tuple(value / max(1.0, abs(starting_cash)) for value in trade_pnls)))
    gross_profit = sum(value for value in trade_pnls if value > 0.0)
    gross_loss = abs(sum(value for value in trade_pnls if value < 0.0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    average_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
    stdev = 0.0
    if len(trade_returns) >= 2:
        stdev = math.sqrt(sum((value - average_return) ** 2 for value in trade_returns) / (len(trade_returns) - 1))
    max_losses = 0
    current_losses = 0
    for pnl in trade_pnls:
        if pnl < 0.0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    if "buy_hold_pnl" in overrides:
        buy_hold_pnl = float(overrides["buy_hold_pnl"])
    elif "edge_vs_buy_hold" in overrides:
        buy_hold_pnl = realized_pnl - float(overrides["edge_vs_buy_hold"])
    else:
        buy_hold_pnl = 25.0
    max_drawdown = float(overrides.get("max_drawdown", 0.02))
    defaults = dict(
        starting_cash=starting_cash, ending_cash=ending_cash, realized_pnl=realized_pnl,
        win_rate=(sum(1 for value in trade_pnls if value > 0.0) / len(trade_pnls) if trade_pnls else 0.0),
        trades=closed_trades, max_drawdown=max_drawdown, closed_trades=closed_trades,
        gross_exposure=100.0, total_fees=0.1 * len(trade_pnls), stopped_by_drawdown=False,
        max_exposure=100.0, trades_per_day_cap_hit=0,
        buy_hold_pnl=buy_hold_pnl,
        edge_vs_buy_hold=float(overrides.get("edge_vs_buy_hold", realized_pnl - buy_hold_pnl)),
        equity_curve=_fake_equity_curve(starting_cash, ending_cash, max_drawdown),
        trade_pnls=trade_pnls,
        trade_returns=trade_returns,
        trade_log=_fake_trade_log(trade_pnls, trade_returns) if len(trade_pnls) == len(trade_returns) else (),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        expectancy=sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0,
        average_trade_return=average_return,
        trade_return_stdev=stdev,
        max_consecutive_losses=max_losses,
    )
    derived_fields = {
        "win_rate",
        "total_fees",
        "equity_curve",
        "trade_log",
        "gross_profit",
        "gross_loss",
        "profit_factor",
        "expectancy",
        "average_trade_return",
        "trade_return_stdev",
        "max_consecutive_losses",
    }
    defaults.update({key: value for key, value in overrides.items() if key not in derived_fields})
    return BacktestResult(**defaults)


def test_gate_result_payload_includes_objective_reject_reasons() -> None:
    payload = training_suite._gate_result_payload(
        _make_result(realized_pnl=25.0, edge_vs_buy_hold=-10.0, closed_trades=1),
        get_objective("regular"),
    )

    assert payload["accepted"] is False
    assert "closed_trades<3" in payload["reject_reasons"]
    assert "edge_vs_buy_hold<0.0" in payload["reject_reasons"]
    assert "closed_trades<3" in payload["reject_reason"]
    assert payload["market_edge"]["accepted"] is False
    assert "net_edge_pct<0.003000" in payload["market_edge"]["failed_checks"]


def test_candidate_diagnostics_include_probability_inversion_evidence() -> None:
    model = _fake_trained_model(2)
    model.model_family = "advanced:inverted"
    model.probability_inverted = True
    candidate = CandidateParams(
        epochs=4,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        label_threshold_multiplier=0.60,
        label_lookahead_multiplier=0.50,
        seed=11,
    )
    feature_cfg = training_suite._feature_config_for_candidate(default_config_for("default", ()), candidate)

    diagnostics = training_suite._candidate_diagnostics({
        "score": float("-inf"),
        "model": model,
        "candidate": candidate,
        "feature_cfg": feature_cfg,
        "feature_signature": "feature-signature-test",
        "selection_score": 1.0,
        "validation_score": float("-inf"),
        "full_sample_score": float("-inf"),
        "inversion_score": float("-inf"),
        "inversion_selection_score": float("-inf"),
        "selected_internal_variant": "probability_inverted",
        "inversion_source_variant": "full_fit",
        "internal_variant_count": 3,
        "internal_variant_selection": [{"variant": "full_fit", "score": 1.0}],
        "threshold": 0.64,
        "threshold_source": "strategy",
        "threshold_score": None,
        "calibration_rows": 0,
        "validation_rows": 8,
        "selection_result": {"realized_pnl": 1.0},
        "validation_result": {"realized_pnl": -1.0},
        "full_sample_result": {"realized_pnl": -2.0},
        "inversion_selection_result": {"realized_pnl": 2.0},
        "inversion_validation_result": {"realized_pnl": -3.0},
        "inversion_full_sample_result": {"realized_pnl": -4.0},
        "walk_forward_gate": {"passed": False},
    })

    assert diagnostics["score"] is None
    assert diagnostics["model_family"] == "advanced:inverted"
    assert diagnostics["probability_inverted"] is True
    assert diagnostics["feature_signature"] == "feature-signature-test"
    assert diagnostics["label_threshold"] == pytest.approx(feature_cfg.label_threshold)
    assert diagnostics["label_lookahead"] == feature_cfg.label_lookahead
    assert diagnostics["label_mode"] == feature_cfg.label_mode
    assert diagnostics["inversion_score"] is None
    assert diagnostics["inversion_selection_score"] is None
    assert diagnostics["selected_internal_variant"] == "probability_inverted"
    assert diagnostics["inversion_source_variant"] == "full_fit"
    assert diagnostics["internal_variant_count"] == 3
    assert diagnostics["inversion_selection_result"] == {"realized_pnl": 2.0}
    assert diagnostics["inversion_validation_result"] == {"realized_pnl": -3.0}
    assert diagnostics["inversion_full_sample_result"] == {"realized_pnl": -4.0}


def test_candidate_rank_key_orders_rejected_candidates_by_evidence() -> None:
    accepted = {"score": -0.5}
    weak_rejected = {
        "score": float("-inf"),
        "validation_result": {"realized_pnl": -10.0, "edge_vs_buy_hold": -12.0, "max_drawdown": 0.05},
        "full_sample_result": {"realized_pnl": -20.0, "edge_vs_buy_hold": -30.0, "max_drawdown": 0.08},
    }
    better_rejected = {
        "score": float("-inf"),
        "validation_result": {
            "realized_pnl": 4.0,
            "edge_vs_buy_hold": -2.0,
            "max_drawdown": 0.01,
            "closed_trades": 6,
            "win_rate": 0.6,
        },
        "full_sample_result": {
            "realized_pnl": 1.0,
            "edge_vs_buy_hold": -5.0,
            "max_drawdown": 0.02,
            "closed_trades": 18,
            "win_rate": 0.5,
        },
    }

    ranked = sorted([weak_rejected, better_rejected, accepted], key=training_suite._candidate_rank_key, reverse=True)

    assert ranked == [accepted, better_rejected, weak_rejected]


def test_selection_risk_report_deflates_tiny_scores_under_large_trial_count() -> None:
    tiny_best = {"score": 0.0001}
    tiny_report = training_suite._selection_risk_report(
        tiny_best,
        [tiny_best, {"score": 0.00009}],
        base_candidate_count=2000,
        local_refinement_candidates=20,
        ensemble_refinement_candidates=3,
        hybrid_rescue_candidates=0,
    )

    assert tiny_report["passed"] is False
    assert tiny_report["reason"] == "selection_risk_deflated_score<=0"
    assert tiny_report["effective_trials"] == 2023
    assert tiny_report["deflated_score"] < 0.0
    assert tiny_report["overfit_diagnostics"]["status"] == "skipped"

    stronger_best = {
        "score": 0.12,
        "selection_score": 0.12,
        "validation_score": 0.11,
    }
    stronger_report = training_suite._selection_risk_report(
        stronger_best,
        [
            stronger_best,
            {"score": 0.09, "selection_score": 0.09, "validation_score": 0.08},
            {"score": 0.07, "selection_score": 0.07, "validation_score": 0.06},
        ],
        base_candidate_count=2000,
        local_refinement_candidates=20,
        ensemble_refinement_candidates=3,
        hybrid_rescue_candidates=0,
    )

    assert stronger_report["passed"] is True
    assert stronger_report["deflated_score"] > 0.0

    missing_panels = training_suite._selection_risk_report(
        {"score": 1.0},
        [{"score": 1.0}, {"score": 0.8}, {"score": 0.7}],
        base_candidate_count=3,
        local_refinement_candidates=0,
        ensemble_refinement_candidates=0,
        hybrid_rescue_candidates=0,
    )
    assert missing_panels["passed"] is False
    assert missing_panels["reason"] == "requires_selection_and_validation_scores"


def test_selection_risk_report_blocks_severe_pbo_overfit() -> None:
    overfit_best = {"score": 0.90, "selection_score": 0.90, "validation_score": 0.10}
    report = training_suite._selection_risk_report(
        overfit_best,
        [
            overfit_best,
            {"score": 0.80, "selection_score": 0.80, "validation_score": 0.70},
            {"score": 0.70, "selection_score": 0.70, "validation_score": 0.90},
        ],
        base_candidate_count=3,
        local_refinement_candidates=0,
        ensemble_refinement_candidates=0,
        hybrid_rescue_candidates=0,
    )

    assert report["passed"] is False
    assert report["reason"] == "selection_risk_pbo>0.50"
    overfit = report["overfit_diagnostics"]
    assert overfit["status"] == "available"
    assert overfit["passed"] is False
    assert overfit["probability_backtest_overfit"] == pytest.approx(1.0)
    assert overfit["overfit_splits"] == 2
    assert report["deflated_score"] > 0.0


def test_selection_risk_report_allows_stable_selection_validation_ranking() -> None:
    stable_best = {"score": 0.90, "selection_score": 0.90, "validation_score": 0.91}
    report = training_suite._selection_risk_report(
        stable_best,
        [
            stable_best,
            {"score": 0.80, "selection_score": 0.80, "validation_score": 0.75},
            {"score": 0.70, "selection_score": 0.70, "validation_score": 0.72},
        ],
        base_candidate_count=3,
        local_refinement_candidates=0,
        ensemble_refinement_candidates=0,
        hybrid_rescue_candidates=0,
    )

    assert report["passed"] is True
    overfit = report["overfit_diagnostics"]
    assert overfit["status"] == "available"
    assert overfit["passed"] is True
    assert overfit["probability_backtest_overfit"] == pytest.approx(0.0)
    assert overfit["selected_validation_rank_percentile"] == pytest.approx(1.0)


def test_selection_risk_report_counts_hidden_internal_model_trials() -> None:
    best = {
        "score": 0.90,
        "selection_score": 0.90,
        "validation_score": 0.91,
        "internal_variant_count": 3,
    }
    report = training_suite._selection_risk_report(
        best,
        [
            best,
            {
                "score": 0.80,
                "selection_score": 0.80,
                "validation_score": 0.75,
                "internal_variant_count": 2,
            },
        ],
        base_candidate_count=2,
        local_refinement_candidates=0,
        ensemble_refinement_candidates=0,
        hybrid_rescue_candidates=0,
    )

    assert report["internal_variants_evaluated"] == 5
    assert report["internal_variant_extra_trials"] == 3
    assert report["effective_trials"] == 5


def test_evaluate_candidate_without_calibration_uses_strategy_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fake_trained_model(2)

    def fake_train_advanced(rows, cfg, **kwargs):
        assert kwargs["seed"] == 11
        assert kwargs["validation_rows"] == []
        assert kwargs["early_stopping_rounds"] is None
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(training_suite, "run_backtest", lambda *a, **k: _make_result())

    candidate = CandidateParams(
        epochs=4,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(20),
        "rows_eval": _rows(8),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert result["threshold"] == pytest.approx(0.64)
    assert result["threshold_source"] == "strategy"
    assert result["calibration_rows"] == 0
    assert model.strategy_overrides["signal_threshold"] == pytest.approx(0.64)
    assert model.strategy_overrides["risk_per_trade"] == pytest.approx(0.01)
    assert model.strategy_overrides["take_profit_pct"] == pytest.approx(0.03)


def test_evaluate_candidate_can_freeze_probability_inversion_from_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"backtest": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        del cfg, kwargs
        return _fake_trained_model(2), SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    def fake_run_backtest(*args, **kwargs):
        del args, kwargs
        calls["backtest"] += 1
        pnl = 50.0 if calls["backtest"] == 1 else 75.0
        return _make_result(realized_pnl=pnl, closed_trades=5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    candidate = CandidateParams(
        epochs=4,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(20),
        "rows_eval": _rows(8),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert calls["backtest"] == 4
    assert result["selected_internal_variant"] == "probability_inverted"
    assert result["inversion_source_variant"] == "full_fit"
    assert result["internal_variant_count"] == 2
    assert result["model"].probability_inverted is True
    assert result["inversion_validation_result"] == result["validation_result"]
    assert result["inversion_full_sample_result"] == result["full_sample_result"]


def test_evaluate_candidate_failed_probability_calibration_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"train": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        calls["train"] += 1
        if calls["train"] == 1:
            assert kwargs["seed"] == 11
            assert len(kwargs["validation_rows"]) > 0
            assert kwargs["early_stopping_rounds"] is not None
        else:
            assert kwargs["seed"] == 11
            assert "validation_rows" not in kwargs
        return _fake_trained_model(2), SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.67, "classification_f1", 2.5),
    )
    monkeypatch.setattr(training_suite, "run_backtest", lambda *a, **k: _make_result())

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(50),
        "rows_eval": _rows(10),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert result["threshold"] == pytest.approx(0.67)
    assert result["threshold_source"] == "classification_f1"
    assert result["threshold_score"] == pytest.approx(2.5)
    assert result["calibration_rows"] > 0


def test_evaluate_candidate_selects_full_fit_when_calibration_regresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"train": 0, "backtest": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        calls["train"] += 1
        assert kwargs["seed"] == 11
        model = _fake_trained_model(2)
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    def fake_run_backtest(*args, **kwargs):
        calls["backtest"] += 1
        if calls["backtest"] == 1:
            return _make_result(ending_cash=1000.0, realized_pnl=0.0, win_rate=0.2)
        if calls["backtest"] in {2, 4, 5}:
            return _make_result(ending_cash=1075.0, realized_pnl=75.0, win_rate=0.8)
        return _make_result(ending_cash=990.0, realized_pnl=-10.0, win_rate=0.1)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.67, "classification_f1", 2.5),
    )
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(50),
        "rows_eval": _rows(10),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert calls["train"] == 2
    assert calls["backtest"] == 5
    assert result["selected_internal_variant"] == "full_fit"
    assert result["internal_variant_count"] == 3
    assert result["threshold"] == pytest.approx(0.64)
    assert result["threshold_source"] == "strategy_full_fit"
    assert result["threshold_score"] is None
    assert result["calibration_rows"] == 0
    assert result["validation_score"] > 0.0
    assert result["full_sample_score"] > 0.0
    assert result["model"].strategy_overrides["signal_threshold"] == pytest.approx(0.64)


def test_evaluate_candidate_freezes_internal_variant_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated_variants: list[tuple[str, bool, tuple[int, ...]]] = []

    def fake_train_advanced(rows, cfg, **kwargs):
        del cfg
        model = _fake_trained_model(2)
        model.fit_variant = "calibration_fit" if "validation_rows" in kwargs else "full_fit"
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    def fake_run_backtest(rows, model, _strategy, **kwargs):
        del kwargs
        timestamps = tuple(row.timestamp for row in rows)
        variant = str(model.fit_variant)
        inverted = bool(getattr(model, "probability_inverted", False))
        evaluated_variants.append((variant, inverted, timestamps))
        touches_validation = any(timestamp >= 1_000 for timestamp in timestamps)
        if touches_validation:
            assert variant == "calibration_fit"
            assert inverted is False
            return _make_result(realized_pnl=-10.0, closed_trades=5)
        if inverted:
            return _make_result(realized_pnl=30.0, closed_trades=5)
        if variant == "full_fit":
            return _make_result(realized_pnl=40.0, closed_trades=5)
        return _make_result(realized_pnl=50.0, closed_trades=5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.64, "classification_f1", 1.0),
    )
    monkeypatch.setattr(training_suite, "_refine_threshold_on_selection_rows", lambda *a, **k: None)
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(80),
        "rows_eval": [
            ModelRow(timestamp=1_000 + i, close=1.0, features=(0.1, 0.2), label=i % 2)
            for i in range(12)
        ],
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert len(evaluated_variants) == 5
    assert result["selected_internal_variant"] == "calibration_fit"
    assert result["internal_variant_count"] == 3
    assert result["inversion_validation_result"] is None
    assert result["inversion_full_sample_result"] is None
    assert result["score"] == float("-inf")


def test_evaluate_candidate_keeps_final_holdout_out_of_threshold_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threshold_rows: list[list[int]] = []
    backtest_calls: list[tuple[list[int], str | None, int | None]] = []

    def fake_train_advanced(rows, cfg, **kwargs):
        return _fake_trained_model(2), SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    def fake_refine(model, rows, strategy, **kwargs):
        del model, strategy, kwargs
        threshold_rows.append([row.timestamp for row in rows])
        return None

    def fake_run_backtest(rows, _model, _strategy, **kwargs):
        timestamps = [row.timestamp for row in rows]
        backtest_calls.append((
            timestamps,
            kwargs.get("compute_backend"),
            kwargs.get("score_batch_size"),
        ))
        if timestamps and min(timestamps) >= 1_000:
            return _make_result(realized_pnl=-10.0, closed_trades=5)
        return _make_result(realized_pnl=50.0, closed_trades=5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.64, "classification_f1", 1.0),
    )
    monkeypatch.setattr(training_suite, "_refine_threshold_on_selection_rows", fake_refine)
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(80),
        "rows_eval": [
            ModelRow(timestamp=1_000 + i, close=1.0, features=(0.1, 0.2), label=i % 2)
            for i in range(12)
        ],
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
        "compute_backend": "directml",
        "batch_size": 16,
        "score_batch_size": 32,
        "include_full_fit_fallback": False,
    })

    assert threshold_rows
    assert max(threshold_rows[0]) < 1_000
    assert result["score"] == float("-inf")
    assert result["validation_score"] == float("-inf")
    assert backtest_calls
    assert all(call[1:] == ("directml", 32) for call in backtest_calls)


def test_evaluate_candidate_can_skip_full_fit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"train": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        calls["train"] += 1
        model = _fake_trained_model(2)
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.67, "classification_f1", 2.5),
    )
    monkeypatch.setattr(training_suite, "run_backtest", lambda *a, **k: _make_result())

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(50),
        "rows_eval": _rows(10),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
        "include_full_fit_fallback": False,
    })

    assert calls["train"] == 1
    assert result["calibration_rows"] > 0
    assert result["threshold_source"] == "classification_f1"


def test_train_for_objective_happy_with_fake_runner(
    tmp_path: Path,
    _passing_selection_risk_stub: None,
) -> None:
    candles = _synthetic_candles(n=200)
    strategy = StrategyConfig()
    objective = get_objective("default")

    scores_cycle = iter([0.1, 0.9, float("-inf"), 0.5])

    def runner(_obj, candidate, rows, base, feat_cfg, market, cash):
        try:
            score = next(scores_cycle)
        except StopIteration:
            score = 0.0
        return score, base, _fake_trained_model(feat_cfg.polynomial_top_features), 42, 0.5

    outcome = train_for_objective(
        candles, strategy, objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        runner=runner,
    )
    assert isinstance(outcome, ObjectiveOutcome)
    model_file = tmp_path / f"model_{objective.name}.json"
    assert model_file.exists()
    # outcome.asdict conversion covered
    assert outcome.asdict()["model_path"] == str(model_file)
    assert outcome.selection_risk is not None
    assert outcome.selection_risk["passed"] is True
    saved_model = json.loads(model_file.read_text(encoding="utf-8"))
    assert saved_model["selection_risk"]["passed"] is True
    assert saved_model["selection_risk"]["effective_trials"] >= outcome.explored_candidates
    assert saved_model["selection_risk"]["overfit_diagnostics"]["status"] == "available"
    # rejected counts entries scored as -inf
    assert outcome.rejected_candidates >= 1
    assert outcome.validation_rows >= 0
    assert outcome.validation_score is not None
    assert outcome.meta_label_report == {
        "status": "not_run",
        "reason": "runner_path",
        "objective": "conservative",
    }


def test_train_for_objective_rejects_all_rejected_candidates(tmp_path: Path) -> None:
    candles = _synthetic_candles(n=200)
    strategy = StrategyConfig()
    objective = get_objective("default")

    def runner(_obj, candidate, rows, base, feat_cfg, market, cash):
        return float("-inf"), base, _fake_trained_model(feat_cfg.polynomial_top_features), 42, 0.5

    with pytest.raises(TrainingSuiteRejected, match="All conservative training candidates were rejected") as exc:
        train_for_objective(
            candles,
            strategy,
            objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=runner,
        )
    assert exc.value.diagnostics["objective"] == "conservative"
    assert exc.value.diagnostics["row_count"] > 0
    assert exc.value.diagnostics["top_candidates"]
    assert exc.value.diagnostics["top_candidates"][0]["score"] is None
    assert not (tmp_path / f"model_{objective.name}.json").exists()


def test_train_for_objective_rejects_failed_selection_risk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _synthetic_candles(n=200)
    strategy = StrategyConfig()
    objective = get_objective("default")

    def runner(_obj, candidate, rows, base, feat_cfg, market, cash):
        del _obj, candidate, rows, market, cash
        return 1.0, base, _fake_trained_model(feat_cfg.polynomial_top_features), 42, 0.5

    monkeypatch.setattr(
        training_suite,
        "_selection_risk_report",
        lambda *_a, **_k: {
            "passed": False,
            "reason": "selection_risk_deflated_score<=0",
            "effective_trials": 999,
            "finite_candidate_scores": 1,
            "selected_score": 1.0,
            "trial_penalty": 1.5,
            "deflated_score": -0.5,
        },
    )

    with pytest.raises(TrainingSuiteRejected, match="All conservative training candidates were rejected") as exc:
        train_for_objective(
            candles,
            strategy,
            objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=runner,
        )

    assert exc.value.diagnostics["top_candidates"][0]["selection_risk"]["passed"] is False
    assert not (tmp_path / f"model_{objective.name}.json").exists()


def test_train_for_objective_rejects_failed_walk_forward_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    def fake_evaluate(payload):
        return {
            "score": 2.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 2.0,
            "full_sample_score": 2.0,
            "ensemble_refined": False,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(
        training_suite,
        "_purged_walk_forward_gate",
        lambda *_a, **_k: {
            "passed": False,
            "reason": "purged_walk_forward_fold_failed",
            "fold_count": 2,
            "accepted_folds": 1,
            "worst_score": float("-inf"),
            "worst_realized_pnl": -1.0,
            "worst_max_drawdown": 0.2,
            "folds": [],
        },
    )

    with pytest.raises(ValueError, match="All conservative training candidates were rejected"):
        train_for_objective(
            _synthetic_candles(n=420),
            StrategyConfig(),
            objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            max_workers=1,
        )


def test_train_for_objective_rescues_rejected_candidate_with_hybrid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    feature_cfg = default_config_for("conservative", ())
    feature_dim = training_suite.advanced_feature_dimension(feature_cfg)
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    def fake_evaluate(payload):
        return {
            "score": float("-inf"),
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(feature_dim),
            "feature_cfg": payload["feature_cfg"],
            "feature_dim": feature_dim,
            "feature_signature": "base-feature-signature",
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "selection_score": float("-inf"),
            "validation_score": float("-inf"),
            "full_sample_score": float("-inf"),
            "selection_result": {"accepted": False, "reject_reason": "base_failed"},
            "validation_result": {"accepted": False, "reject_reason": "base_failed"},
            "full_sample_result": {"accepted": False, "reject_reason": "base_failed"},
            "ensemble_refined": bool(payload.get("ensemble_seeds")),
        }

    rescue_result = _make_result(realized_pnl=35.0, edge_vs_buy_hold=12.0, closed_trades=6, win_rate=0.75)
    rescue_model = _fake_trained_model(feature_dim)

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(
        training_suite,
        "optimize_hybrid_model_zoo",
        lambda *_a, **_k: SimpleNamespace(
            accepted=True,
            model=rescue_model,
            base_score=float("-inf"),
            best_score=1.25,
            best_profile="technical_rescue",
            evaluated_profiles=4,
            best_result=rescue_result,
            ablation_results=[
                SimpleNamespace(asdict=lambda: {
                    "removed_expert_kind": "all_hybrid_experts",
                    "score": float("-inf"),
                    "delta_vs_best": float("-inf"),
                    "accepted": False,
                    "removed_expert_count": 3,
                    "remaining_expert_count": 0,
                })
            ],
        ),
    )
    monkeypatch.setattr(training_suite, "run_backtest", lambda *_a, **_k: rescue_result)

    outcome = train_for_objective(
        _synthetic_candles(n=260),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert outcome.hybrid_model is True
    assert outcome.hybrid_rescue is True
    assert outcome.hybrid_profile == "technical_rescue"
    assert outcome.hybrid_rescue_candidates >= 1
    assert outcome.hybrid_ablation[0]["removed_expert_kind"] == "all_hybrid_experts"
    assert outcome.best_score > 0.0


def test_train_for_objective_persists_walk_forward_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    gate = {
        "passed": True,
        "reason": None,
        "fold_count": 2,
        "accepted_folds": 2,
        "worst_score": 1.25,
        "worst_realized_pnl": 12.0,
        "worst_max_drawdown": 0.03,
        "folds": [],
    }
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])
    monkeypatch.setattr(training_suite, "optimize_hybrid_model_zoo", lambda *a, **k: None)
    monkeypatch.setattr(training_suite, "_purged_walk_forward_gate", lambda *_a, **_k: gate)
    monkeypatch.setattr(
        training_suite,
        "_evaluate_candidate",
        lambda payload: {
            "score": 2.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 2.0,
            "full_sample_score": 2.0,
            "ensemble_refined": False,
        },
    )

    outcome = train_for_objective(
        _synthetic_candles(n=420),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert outcome.best_score == pytest.approx(1.25)
    assert outcome.walk_forward_gate == gate


def test_train_for_objective_insufficient_candles(tmp_path: Path) -> None:
    objective = get_objective("default")
    with pytest.raises(ValueError, match="Insufficient candles"):
        train_for_objective(
            [],
            StrategyConfig(),
            objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=lambda *a, **k: (0.0, StrategyConfig(), _fake_trained_model(), 0, 0.0),
        )


def test_train_for_objective_empty_candidate_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candles = _synthetic_candles(n=200)

    # Monkeypatch the module-level _candidate_grid to return []
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda training: [])

    def runner(*a, **k):  # pragma: no cover - should not be called
        raise AssertionError("runner should not be invoked for empty grid")

    with pytest.raises(ValueError, match="Candidate grid produced zero"):
        train_for_objective(
            candles, StrategyConfig(), objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=runner,
        )


def test_train_for_objective_promotes_better_seed_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    def fake_evaluate(payload):
        refined = bool(payload.get("ensemble_seeds"))
        return {
            "score": 2.0 if refined else 1.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 2.0 if refined else 1.0,
            "full_sample_score": 2.0 if refined else 1.0,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert _ensemble_seed_pack(11) == (11, 28, 48)
    assert outcome.best_score == 2.0
    assert outcome.ensemble_refined is True
    assert outcome.ensemble_refinement_candidates == 1


def test_train_for_objective_gpu_backend_forces_sequential_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    default_objective = get_objective("default")
    objective = ObjectiveSpec(
        name="default",
        label="Smoke",
        summary="Lenient smoke objective",
        long_description="Lenient real-runner smoke objective for tiny synthetic datasets.",
        scorer=lambda result: 1.0 + result.realized_pnl / max(1.0, result.starting_cash),
        min_closed_trades=0,
        min_realized_pnl=None,
        min_edge_vs_buy_hold=None,
        min_market_edge_pct=None,
        max_drawdown_rejection=1.0,
        training=default_objective.training,
    )
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])
    observed: list[tuple[str, int, int]] = []

    def fake_evaluate(payload):
        observed.append((payload["compute_backend"], payload["batch_size"], payload["score_batch_size"]))
        return {
            "score": 1.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 1.0,
            "full_sample_score": 1.0,
            "ensemble_refined": False,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)
    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=4,
        compute_backend="directml",
        batch_size=64,
        score_batch_size=32,
    )
    assert observed
    assert all(item == ("directml", 64, 32) for item in observed)
    assert outcome.training_backend_kind == "cpu"


def test_train_for_objective_defaults_to_auto_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])
    observed: list[str] = []

    def fake_evaluate(payload):
        observed.append(payload["compute_backend"])
        return {
            "score": 1.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 1.0,
            "full_sample_score": 1.0,
            "ensemble_refined": False,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=4,
    )

    assert observed
    assert set(observed) == {"auto"}


def test_train_for_objective_rejects_weaker_seed_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    def fake_evaluate(payload):
        refined = bool(payload.get("ensemble_seeds"))
        return {
            "score": 0.5 if refined else 1.5,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 0.5 if refined else 1.5,
            "full_sample_score": 0.5 if refined else 1.5,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert outcome.best_score == 1.5
    assert outcome.ensemble_refined is False
    assert outcome.ensemble_refinement_candidates == 1


def test_train_for_objective_promotes_better_local_refinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])

    def fake_evaluate(payload):
        refined = bool(payload.get("ensemble_seeds"))
        candidate_ = payload["candidate"]
        score = 2.0 if candidate_.risk_per_trade < candidate.risk_per_trade and not refined else 1.0
        return {
            "score": score,
            "candidate": candidate_,
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": score,
            "full_sample_score": score,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert len(_local_refinement_candidates(candidate)) == 26
    assert outcome.best_score == 2.0
    assert outcome.best_params["risk_per_trade"] == pytest.approx(0.005)
    assert outcome.local_refinement_candidates == 26
    assert outcome.ensemble_refined is False


def test_risk_non_degrading_blocks_fragile_score_improvement() -> None:
    incumbent = {
        "validation_result": {"realized_pnl": 100.0, "edge_vs_buy_hold": 80.0, "max_drawdown": 0.04},
        "full_sample_result": {"realized_pnl": 90.0, "edge_vs_buy_hold": 70.0, "max_drawdown": 0.05},
    }
    better_and_stable = {
        "validation_result": {"realized_pnl": 103.0, "edge_vs_buy_hold": 78.0, "max_drawdown": 0.051},
        "full_sample_result": {"realized_pnl": 94.0, "edge_vs_buy_hold": 72.0, "max_drawdown": 0.052},
    }
    worse_drawdown = {
        "validation_result": {"realized_pnl": 130.0, "edge_vs_buy_hold": 110.0, "max_drawdown": 0.09},
        "full_sample_result": {"realized_pnl": 125.0, "edge_vs_buy_hold": 105.0, "max_drawdown": 0.10},
    }

    assert _risk_non_degrading(better_and_stable, incumbent) is True
    assert _risk_non_degrading(worse_drawdown, incumbent) is False


def test_risk_aware_best_refuses_fragile_higher_score() -> None:
    incumbent = {
        "score": 1.0,
        "validation_result": {"realized_pnl": 100.0, "edge_vs_buy_hold": 80.0, "max_drawdown": 0.04},
        "full_sample_result": {"realized_pnl": 90.0, "edge_vs_buy_hold": 70.0, "max_drawdown": 0.05},
    }
    fragile = {
        "score": 2.0,
        "validation_result": {"realized_pnl": 130.0, "edge_vs_buy_hold": 110.0, "max_drawdown": 0.12},
        "full_sample_result": {"realized_pnl": 125.0, "edge_vs_buy_hold": 105.0, "max_drawdown": 0.13},
    }
    stable = {
        "score": 1.5,
        "validation_result": {"realized_pnl": 104.0, "edge_vs_buy_hold": 79.0, "max_drawdown": 0.051},
        "full_sample_result": {"realized_pnl": 94.0, "edge_vs_buy_hold": 72.0, "max_drawdown": 0.052},
    }

    assert _risk_aware_best(incumbent, [fragile, stable]) is stable


def test_train_for_objective_checks_top_candidates_for_seed_ensembles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _passing_selection_risk_stub: None,
) -> None:
    objective = get_objective("default")
    candidates = [
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=7),
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=11),
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=13),
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=17),
    ]
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: candidates)
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    base_scores = {7: 3.0, 11: 2.0, 13: 1.0, 17: 0.5}
    ensemble_scores = {7: 2.5, 11: 4.0, 13: 1.1}
    evaluated: list[tuple[int, bool]] = []

    def fake_evaluate(payload):
        seed = int(payload["candidate"].seed)
        refined = bool(payload.get("ensemble_seeds"))
        evaluated.append((seed, refined))
        score = ensemble_scores[seed] if refined else base_scores[seed]
        return {
            "score": score,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": score,
            "full_sample_score": score,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert outcome.best_score == 4.0
    assert outcome.best_params["seed"] == 11
    assert outcome.ensemble_refined is True
    assert outcome.ensemble_refinement_candidates == 3
    assert evaluated[-3:] == [(7, True), (11, True), (13, True)]


def test_feature_ablation_report_replays_masked_feature_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_cfg = default_config_for("regular", ("momentum_1", "momentum_3", "momentum_10", "momentum_20"))
    feature_dim = training_suite.advanced_feature_dimension(feature_cfg)
    rows = [
        ModelRow(timestamp=index, close=100.0 + index, features=tuple([0.1] * feature_dim), label=1)
        for index in range(20)
    ]
    model = _fake_trained_model(feature_dim)
    pnl_by_group = {
        None: 10.0,
        "base_features": 7.0,
        "extra_lookback_windows": 9.0,
        "technical_confluence": 5.0,
        "market_quality_regime": 8.5,
        "higher_timeframe_context": 8.4,
        "order_flow_microstructure": 8.25,
        "nonlinear_transforms": 8.0,
        "polynomial_interactions": 6.0,
    }

    def fake_run_backtest(_rows, scored_model, _strategy, **_kwargs):
        pnl = pnl_by_group[getattr(scored_model, "ablated_feature_group", None)]
        return _make_result(
            realized_pnl=pnl,
            closed_trades=6,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=pnl,
            max_drawdown=0.0,
        )

    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    report = _feature_ablation_report(
        rows,
        model,
        StrategyConfig(),
        feature_cfg,
        get_objective("regular"),
        market_type="spot",
        starting_cash=1000.0,
        score_batch_size=16,
    )

    by_group = {entry["removed_group"]: entry for entry in report}
    assert set(by_group) == set(pnl_by_group) - {None}
    assert by_group["technical_confluence"]["delta_vs_selected"] < 0.0
    assert by_group["technical_confluence"]["realized_pnl"] == pytest.approx(5.0)
    assert by_group["base_features"]["baseline_score"] > by_group["base_features"]["score"]
    assert all(entry["status"] == "evaluated" for entry in report)


# ----- run_training_suite --------------------------------------------------


def test_run_training_suite_with_explicit_objectives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _synthetic_candles(n=100)
    strat = StrategyConfig()

    def fake_train_for_objective(
        candles, base_strategy, objective, *,
        output_dir, market_type, starting_cash, runner=None,
    ):
        # write a placeholder model file so layout matches production
        model_path = output_dir / f"model_{objective.name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        return ObjectiveOutcome(
            objective=objective.name,
            model_path=model_path,
            feature_dim=4,
            feature_signature="sig",
            best_score=0.5,
            best_params={"epochs": 1},
            explored_candidates=1,
            rejected_candidates=0,
            epochs=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            row_count=50,
            positive_rate=0.5,
        )

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)

    report = run_training_suite(
        candles, strat,
        objectives=["default", "conservative"],
        market_type="spot",
        starting_cash=1000.0,
        output_dir=tmp_path,
    )
    assert isinstance(report, SuiteReport)
    assert [o.objective for o in report.outcomes] == ["conservative"]
    summary = tmp_path / "training_suite_summary.json"
    assert summary.exists()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["total_candles"] == len(candles)
    assert data["objectives_run"] == ["conservative"]


def test_run_training_suite_default_objectives_and_summary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _synthetic_candles(n=80)
    strat = StrategyConfig()

    def fake_train_for_objective(
        candles, base_strategy, objective, *,
        output_dir, market_type, starting_cash, runner=None,
    ):
        model_path = output_dir / f"model_{objective.name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        return ObjectiveOutcome(
            objective=objective.name,
            model_path=model_path,
            feature_dim=3,
            feature_signature="sig",
            best_score=1.0,
            best_params={"epochs": 1},
            explored_candidates=1,
            rejected_candidates=0,
            epochs=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            row_count=60,
            positive_rate=0.5,
        )

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)

    summary = tmp_path / "custom_summary.json"
    report = run_training_suite(
        candles, strat,
        objectives=None,
        output_dir=tmp_path,
        summary_path=summary,
    )
    assert summary.exists()
    # All three default objectives should be covered
    assert len(report.outcomes) >= 3
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["summary_path"] == str(summary)


def test_run_training_suite_forwards_optional_gpu_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []

    def fake_train_for_objective(candles, base_strategy, objective, **kwargs):
        observed.append(kwargs)
        return ObjectiveOutcome(
            objective=objective.name,
            model_path=tmp_path / f"{objective.name}.json",
            feature_dim=1,
            feature_signature="sig",
            best_score=1.0,
            best_params={"epochs": 1},
            explored_candidates=1,
            rejected_candidates=0,
            epochs=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            row_count=1,
            positive_rate=0.5,
        )

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)
    run_training_suite(
        _synthetic_candles(n=20),
        StrategyConfig(),
        objectives=["default"],
        output_dir=tmp_path,
        compute_backend="directml",
        batch_size=64,
        score_batch_size=32,
    )
    assert observed[0]["compute_backend"] == "directml"
    assert observed[0]["batch_size"] == 64
    assert observed[0]["score_batch_size"] == 32


# ----- describe_candidate_grid + preview_candidates ------------------------


def test_describe_candidate_grid_keys() -> None:
    grid = describe_candidate_grid(get_objective("default"))
    assert len(grid) > 0
    for item in grid:
        assert "epochs" in item and "learning_rate" in item


def test_preview_candidates_shape() -> None:
    rows = preview_candidates()
    assert len(rows) >= 3
    for row in rows:
        assert "objective" in row
        assert "candidates" in row
        assert "first_candidate" in row


# ----- rank_report ---------------------------------------------------------


def test_rank_report_ranks_precomputed_backtests() -> None:
    ranked = rank_report([
        ({"name": "low"}, _make_result(realized_pnl=5.0, closed_trades=4, win_rate=0.25)),
        ({"name": "high"}, _make_result(realized_pnl=60.0, closed_trades=4, win_rate=0.75)),
        ({"name": "reject"}, _make_result(realized_pnl=80.0, closed_trades=0, win_rate=0.0)),
    ])

    assert ranked[0]["params"] == {"name": "high"}
    assert ranked[0]["accepted"] is True
    assert ranked[-1]["params"] == {"name": "reject"}
    assert ranked[-1]["accepted"] is False
    reject_reasons = str(ranked[-1]["reject_reason"]).split("; ")
    assert "financial_sanity_failed" in reject_reasons
    assert "closed_trades<3" in reject_reasons


# ----- real-runner smoke test for train_for_objective ----------------------


def test_train_for_objective_real_runner_small_dataset(tmp_path: Path) -> None:
    """Exercises the real ``_run_candidate`` path end-to-end with a small dataset.

    We shrink the grid by monkey-patching ``_candidate_grid`` to a single
    lightweight candidate so the test stays fast but still traverses
    ``make_advanced_rows`` + ``train_advanced`` + ``run_backtest``.
    """

    candles = _synthetic_candles(n=260)
    default_objective = get_objective("default")
    objective = ObjectiveSpec(
        name="smoke",
        label="Smoke",
        summary="Lenient smoke objective",
        long_description="Lenient real-runner smoke objective for tiny synthetic datasets.",
        scorer=lambda result: 1.0 + result.realized_pnl / max(1.0, result.starting_cash),
        min_closed_trades=0,
        min_realized_pnl=None,
        min_edge_vs_buy_hold=None,
        min_market_edge_pct=None,
        max_drawdown_rejection=1.0,
        training=default_objective.training,
    )

    single_candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
    )

    # Swap in a tiny grid (monkeypatch via direct assignment on the module).
    original = training_suite._candidate_grid
    original_get_objective = training_suite.get_objective
    training_suite._candidate_grid = lambda _t: [single_candidate]
    training_suite.get_objective = lambda _name: objective
    try:
        outcome = train_for_objective(
            candles, StrategyConfig(), objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            compute_backend="cpu",
        )
    finally:
        training_suite._candidate_grid = original
        training_suite.get_objective = original_get_objective

    assert (tmp_path / f"model_{objective.name}.json").exists()
    assert outcome.feature_dim > 0
    assert outcome.row_count > 0
