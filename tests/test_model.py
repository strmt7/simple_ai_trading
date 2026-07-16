from __future__ import annotations

import json
import pytest

import simple_ai_trading.model as model_module
from simple_ai_trading.features import ModelRow
from pathlib import Path

from simple_ai_trading.model import (
    TrainedModel,
    EnsembleMember,
    HybridExpert,
    ClassificationReport,
    calibrate_probability_temperature,
    confidence_adjusted_probability,
    feature_dimension,
    ModelFeatureMismatchError,
    ModelLoadError,
    load_model,
    market_direction_from_probability,
    model_decision_threshold,
    model_direction_thresholds,
    evaluate_classification,
    evaluate,
    serialize_model,
    temporal_validation_split,
    train,
    effective_training_backend_name,
    walk_forward_report,
)
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.strategy_overrides import clean_strategy_overrides
from simple_ai_trading.trade_tape_features import TRADE_TAPE_FEATURES_PER_WINDOW


def _rows() -> list[ModelRow]:
    out: list[ModelRow] = []
    for i in range(120):
        features = (float(i), float(i * 0.1), 0.5, float(i % 2), 0.01, float(i) / 10.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        label = 1 if i % 2 == 0 else 0
        out.append(ModelRow(timestamp=1000 + i, close=20000.0 + i, features=features, label=label))
    return out


def test_effective_training_backend_defaults_to_gpu_first_auto() -> None:
    assert effective_training_backend_name(None) == "auto"
    assert effective_training_backend_name("") == "auto"
    assert effective_training_backend_name("directml") == "directml"
    assert effective_training_backend_name("cpu") == "cpu"


def test_market_direction_from_probability_fails_closed_on_neutral_and_nonfinite_scores() -> None:
    direction = market_direction_from_probability

    assert direction(0.5, 0.5, market_type="futures", short_threshold=0.5) == 0
    assert direction(float("nan"), 0.5, market_type="futures", short_threshold=0.5) == 0
    assert direction(0.6, 0.55, market_type="futures", short_threshold=0.45) == 1
    assert direction(0.4, 0.55, market_type="futures", short_threshold=0.45) == -1
    assert direction(0.4, 0.55, market_type="futures", infer_symmetric_short=False) == 0
    assert direction(0.4, 0.55, market_type="futures", infer_symmetric_short=True) == -1


def test_collect_feature_stats_uses_population_statistics() -> None:
    rows = [
        ModelRow(timestamp=index, close=1.0, features=(value, value * 2.0), label=0)
        for index, value in enumerate((1.0, 2.0, 3.0))
    ]

    means, stds, backend = model_module.collect_feature_stats(rows, compute_backend="cpu")

    assert backend.kind == "cpu"
    assert means == pytest.approx([2.0, 4.0])
    assert stds == pytest.approx([(2.0 / 3.0) ** 0.5, (8.0 / 3.0) ** 0.5])


def test_train_and_evaluate() -> None:
    model = train(_rows(), epochs=5)
    assert isinstance(model, TrainedModel)
    assert model.feature_dim == 13
    assert model.training_backend_requested == "auto"
    assert model.training_backend_kind in {"directml", "cuda", "rocm", "mps", "cpu"}
    if model.training_backend_kind == "cpu":
        assert model.training_backend_reason
    score = evaluate(_rows(), model, threshold=0.5)
    assert 0.0 <= score <= 1.0


def test_focal_gradient_matches_bce_when_gamma_zero() -> None:
    assert model_module._focal_bce_logit_gradient(0.8, 1, 0.0) == pytest.approx(-0.2)
    assert model_module._focal_bce_logit_gradient(0.2, 0, 0.0) == pytest.approx(0.2)
    assert model_module._focal_bce_logit_gradient(0.99, 0, 2.0) > model_module._focal_bce_logit_gradient(0.2, 0, 2.0)
    assert abs(model_module._focal_bce_logit_gradient(0.99, 1, 2.0)) < abs(model_module._focal_bce_logit_gradient(0.8, 1, 2.0))


def test_class_weights_keep_unit_mean_for_rare_profit_labels() -> None:
    rows = [
        ModelRow(timestamp=i, close=100.0, features=(0.0,), label=0)
        for i in range(99)
    ]
    rows.append(ModelRow(timestamp=99, close=101.0, features=(1.0,), label=1))

    pos_weight, neg_weight = model_module._class_weights(rows)

    assert pos_weight == pytest.approx(50.0)
    assert neg_weight == pytest.approx(100.0 / 198.0)
    mean_weight = (pos_weight + 99 * neg_weight) / 100.0
    assert mean_weight == pytest.approx(1.0)
    assert pos_weight / neg_weight == pytest.approx(99.0)


def test_weighted_focal_loss_penalizes_missing_rare_profit_label() -> None:
    rows = [
        ModelRow(timestamp=i, close=100.0, features=(0.0,), label=0)
        for i in range(99)
    ]
    rows.append(ModelRow(timestamp=99, close=101.0, features=(1.0,), label=1))
    pos_weight, neg_weight = model_module._class_weights(rows)

    all_negative_loss = model_module._weighted_focal_log_loss(
        rows,
        weights=[0.0],
        bias=-4.0,
        means=[0.0],
        stds=[1.0],
        class_weight_pos=pos_weight,
        class_weight_neg=neg_weight,
        focal_gamma=2.0,
    )
    rare_signal_loss = model_module._weighted_focal_log_loss(
        rows,
        weights=[8.0],
        bias=-4.0,
        means=[0.0],
        stds=[1.0],
        class_weight_pos=pos_weight,
        class_weight_neg=neg_weight,
        focal_gamma=2.0,
    )

    assert rare_signal_loss < all_negative_loss


def test_train_records_focal_gamma_and_roundtrips(tmp_path: Path) -> None:
    trained = train(_rows(), epochs=3, compute_backend="cpu", focal_gamma=2.0)
    assert trained.focal_gamma == pytest.approx(2.0)

    path = tmp_path / "focal.json"
    serialize_model(trained, path)
    loaded = load_model(path)

    assert loaded.focal_gamma == pytest.approx(2.0)


def test_probability_inversion_roundtrip(tmp_path: Path) -> None:
    base = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    inverted = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        probability_inverted=True,
    )

    assert inverted.predict_proba((1.0,)) == pytest.approx(1.0 - base.predict_proba((1.0,)))

    path = tmp_path / "model.json"
    serialize_model(inverted, path)
    loaded = load_model(path)
    assert loaded.probability_inverted is True
    assert loaded.predict_proba((1.0,)) == pytest.approx(inverted.predict_proba((1.0,)))


def test_dense_mlp_hybrid_expert_roundtrip_preserves_probability(tmp_path: Path) -> None:
    trained = TrainedModel(
        weights=[0.0, 0.0],
        bias=0.0,
        feature_dim=2,
        epochs=1,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="dense",
                kind="dense_mlp",
                weight=1.0,
                feature_count=2,
                params={
                    "input_dim": 2,
                    "output_activation": "sigmoid",
                    "layers": [
                        {
                            "weights": [[1.0, 0.0], [0.0, 1.0]],
                            "bias": [0.0, 0.0],
                            "activation": "relu",
                        },
                        {
                            "weights": [[1.0], [1.0]],
                            "bias": [0.0],
                            "activation": "sigmoid",
                        },
                    ],
                },
            )
        ],
    )

    before = trained.predict_proba((1.0, 2.0))
    path = tmp_path / "dense.json"
    serialize_model(trained, path)
    loaded = load_model(path, expected_feature_dim=2)

    assert before == pytest.approx(0.9525741268)
    assert loaded.hybrid_experts[0].kind == "dense_mlp"
    assert loaded.predict_proba((1.0, 2.0)) == pytest.approx(before)


def test_rule_alpha_trade_tape_default_width_tracks_feature_schema() -> None:
    width = TRADE_TAPE_FEATURES_PER_WINDOW
    start = 13
    feature_dim = start + width * 2
    features = [0.0] * feature_dim
    features[0] = 0.0012
    features[1] = 0.0008
    for group_start in (start, start + width):
        features[group_start + 0] = 0.77
        features[group_start + 1] = 0.61
        features[group_start + 2] = 0.54
        features[group_start + 3] = 0.33
        features[group_start + 4] = 0.28
        features[group_start + 8] = 0.12
        features[group_start + 9] = 0.0
        features[group_start + 10] = 0.26
        features[group_start + 11] = 0.29
    params = {
        "family": "micro_flow_scalp",
        "sensitivity": 8.0,
        "deadband": 0.01,
        "trade_tape_start": start,
        "trade_tape_window_count": 2,
    }

    implicit = TrainedModel(
        weights=[0.0] * feature_dim,
        bias=0.0,
        feature_dim=feature_dim,
        epochs=0,
        feature_means=[0.0] * feature_dim,
        feature_stds=[1.0] * feature_dim,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="implicit",
                kind="rule_alpha",
                weight=1.0,
                feature_count=feature_dim,
                params=params,
            )
        ],
    )
    explicit = TrainedModel(
        weights=[0.0] * feature_dim,
        bias=0.0,
        feature_dim=feature_dim,
        epochs=0,
        feature_means=[0.0] * feature_dim,
        feature_stds=[1.0] * feature_dim,
        hybrid_base_weight=0.0,
        hybrid_experts=[
            HybridExpert(
                name="explicit",
                kind="rule_alpha",
                weight=1.0,
                feature_count=feature_dim,
                params={**params, "trade_tape_width": width},
            )
        ],
    )

    assert implicit.predict_proba(tuple(features)) == pytest.approx(explicit.predict_proba(tuple(features)))


def test_train_records_requested_backend_fallback_when_unavailable() -> None:
    model = train(_rows(), epochs=2, compute_backend="directml")
    if model.training_backend_kind == "cpu":
        assert model.training_backend_requested == "directml"
        assert "DirectML" in model.training_backend_reason


def test_train_falls_back_when_resolved_gpu_training_errors(monkeypatch) -> None:
    from simple_ai_trading import model as model_mod

    monkeypatch.setattr(
        model_mod,
        "resolve_backend",
        lambda _backend: BackendInfo("cuda", "cuda", "cuda:0", "Fake GPU", ""),
    )

    def fail_gpu(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_mod, "_train_torch", fail_gpu)
    trained = model_mod.train(_rows(), epochs=2, compute_backend="cuda")
    assert trained.training_backend_kind == "cpu"
    assert trained.training_backend_requested == "cuda"
    assert "training failed" in trained.training_backend_reason


def test_torch_training_normalization_matches_population_stats() -> None:
    pytest.importorskip("torch")
    from simple_ai_trading import model as model_mod

    rows = _rows()
    expected_means, expected_stds = model_mod._collect_feature_stats(rows)
    backend = BackendInfo("cpu", "cpu", "cpu", "Torch CPU", "")

    trained = model_mod._train_torch(
        rows,
        epochs=2,
        learning_rate=0.01,
        seed=7,
        l2_penalty=1e-4,
        feature_signature="test",
        validation_rows=rows[:12],
        early_stopping_rounds=None,
        min_delta=1e-6,
        batch_size=32,
        backend=backend,
    )

    assert trained.feature_means == pytest.approx(expected_means, abs=1e-5)
    assert trained.feature_stds == pytest.approx(expected_stds, abs=1e-5)
    assert trained.training_loss is not None


def test_load_model_backwards_compatibility(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy_model.json"
    model_path.write_text(
        """
        {
          "weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
          "feature_version": "v1",
          "bias": 0.01,
          "feature_dim": 13,
          "epochs": 10,
          "feature_means": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          "feature_stds": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    model = load_model(model_path)
    assert isinstance(model, TrainedModel)
    assert model.learning_rate == 0.05
    assert model.l2_penalty == 1e-4
    assert model.class_weight_pos == 1.0
    assert model.class_weight_neg == 1.0
    assert model.focal_gamma == 0.0
    assert model.decision_threshold is None
    assert model.calibration_size == 0
    assert model.validation_size == 0
    assert model.strategy_overrides == {}


def test_load_model_rejects_mismatched_version(tmp_path: Path) -> None:
    model_path = tmp_path / "bad_model.json"
    model_path.write_text(
        """
        {
          "weights": [0.1, 0.2, 0.3],
          "feature_version": "v0",
          "bias": 0.01,
          "feature_dim": 3,
          "epochs": 10,
          "feature_means": [1.0, 1.0, 1.0],
          "feature_stds": [1.0, 1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelFeatureMismatchError, match="Feature version mismatch"):
        load_model(model_path)


def test_load_model_rejects_signature_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "sig_mismatch.json"
    model_payload = {
        "weights": [0.1] * feature_dimension(),
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": feature_dimension(),
        "epochs": 3,
        "feature_means": [1.0] * feature_dimension(),
        "feature_stds": [1.0] * feature_dimension(),
        "feature_signature": "feature_version=v1|feature_count=13|feature_names=momentum_1,momentum_3,momentum_10,momentum_20,ema_spread,rsi,ema_gap,relative_atr,volatility_20,volume_ratio,trend_acceleration,gap_to_vwap,volume_trend|short_window=6|long_window=24|label_threshold=0.001",
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    with pytest.raises(ModelFeatureMismatchError, match="Feature signature mismatch"):
        load_model(
            model_path,
            expected_feature_signature="feature_version=v1|feature_count=13|feature_names=momentum_1,momentum_3,momentum_10,momentum_20,ema_spread,rsi,ema_gap,relative_atr,volatility_20,volume_ratio,trend_acceleration,gap_to_vwap,volume_trend|short_window=4|long_window=8|label_threshold=0.001",
        )


def test_load_model_rejects_missing_signature_when_expected(tmp_path: Path) -> None:
    model_path = tmp_path / "missing_signature.json"
    model_payload = {
        "weights": [0.1] * feature_dimension(),
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": feature_dimension(),
        "epochs": 3,
        "feature_means": [1.0] * feature_dimension(),
        "feature_stds": [1.0] * feature_dimension(),
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    with pytest.raises(ModelFeatureMismatchError, match="missing `feature_signature`"):
        load_model(model_path, expected_feature_signature="runtime-signature")


def test_load_model_allows_subset_feature_dim_when_signature_matches(tmp_path: Path) -> None:
    model_path = tmp_path / "subset_model.json"
    model_payload = {
        "weights": [0.1, 0.2, 0.3],
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": 3,
        "epochs": 3,
        "feature_means": [1.0, 1.0, 1.0],
        "feature_stds": [1.0, 1.0, 1.0],
        "feature_signature": "feature_version=v1|feature_count=3|feature_names=momentum_1,rsi,volume_ratio|short_window=10|long_window=40|label_threshold=0.001",
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    model = load_model(
        model_path,
        expected_feature_signature="feature_version=v1|feature_count=3|feature_names=momentum_1,rsi,volume_ratio|short_window=10|long_window=40|label_threshold=0.001",
        expected_feature_dim=None,
    )
    assert model.feature_dim == 3


def test_evaluate_classification_report() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0, 0.0), label=1),
        ModelRow(timestamp=1, close=101.0, features=(0.0, 0.0), label=0),
        ModelRow(timestamp=2, close=102.0, features=(1.0, 0.0), label=1),
    ]
    model = TrainedModel(
        weights=[1.0, 1.0],
        bias=-0.1,
        feature_dim=2,
        epochs=1,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
    )
    report = evaluate_classification(rows, model, threshold=0.5)
    assert isinstance(report, ClassificationReport)
    assert report.true_positive + report.false_positive + report.true_negative + report.false_negative == len(rows)
    assert 0.0 <= report.accuracy <= 1.0


def test_walk_forward_report_runs() -> None:
    rows = _rows()
    report = walk_forward_report(rows, train_window=60, test_window=20, step=20, epochs=5, calibrate=False)
    assert report["folds"] == 3
    assert report["train_window"] == 60
    assert report["test_window"] == 20
    assert report["step"] == 20
    assert report["average_score"] >= 0.0
    assert report["calibration_sizes"] == [0, 0, 0]


def test_walk_forward_calibrates_inside_training_window() -> None:
    rows = _rows()
    report = walk_forward_report(rows, train_window=50, test_window=10, step=30, epochs=3, calibrate=True)
    assert report["folds"] == 3
    assert report["calibration_sizes"] == [10, 10, 10]
    assert len(report["thresholds"]) == 3


def test_walk_forward_skips_calibration_when_train_window_too_small() -> None:
    rows = _rows()[:20]
    report = walk_forward_report(rows, train_window=8, test_window=3, step=3, epochs=2, calibrate=True)
    assert report["calibration_sizes"]
    assert all(size == 0 for size in report["calibration_sizes"])


def test_temporal_validation_split_keeps_calibration_out_of_validation() -> None:
    rows = list(reversed(_rows()))
    split = temporal_validation_split(rows, calibration_ratio=0.2, validation_ratio=0.2)
    assert len(split.train_rows) == 72
    assert len(split.calibration_rows) == 24
    assert len(split.validation_rows) == 24
    assert split.train_rows[-1].timestamp < split.calibration_rows[0].timestamp
    assert split.calibration_rows[-1].timestamp < split.validation_rows[0].timestamp


def test_temporal_validation_split_caps_oversized_holdouts() -> None:
    rows = [
        ModelRow(timestamp=2, close=102.0, features=(1.0,), label=1),
        ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0),
        ModelRow(timestamp=1, close=101.0, features=(0.5,), label=1),
    ]

    split = temporal_validation_split(rows, calibration_ratio=1.0, validation_ratio=1.0)

    assert [row.timestamp for row in split.train_rows] == [0]
    assert [row.timestamp for row in split.calibration_rows] == [1]
    assert [row.timestamp for row in split.validation_rows] == [2]


def test_decision_threshold_metadata_and_confidence_adjustment(tmp_path: Path) -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.65,
        long_decision_threshold=0.71,
        short_decision_threshold=0.27,
        calibration_size=12,
        validation_size=8,
        training_cutoff_timestamp=123,
        probability_temperature=2.0,
        probability_calibration_size=6,
        probability_log_loss_before=0.8,
        probability_log_loss_after=0.7,
        probability_brier_before=0.25,
        probability_brier_after=0.22,
        probability_ece_before=0.20,
        probability_ece_after=0.15,
        probability_calibration_backend_requested="directml",
        probability_calibration_backend_kind="directml",
        probability_calibration_backend_device="privateuseone:0",
        probability_calibration_backend_reason="",
        threshold_source="profit_backtest",
        threshold_calibration_score=12.3,
        threshold_calibration_pnl=4.5,
        threshold_calibration_trades=6,
        threshold_diagnostic_best_long_threshold=0.69,
        threshold_diagnostic_best_short_threshold=0.31,
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
        training_backend_vendor="DirectML",
        training_backend_reason="",
        model_candidate_count=3,
        model_selected_candidate="triple_barrier_base",
        model_selection_score=0.42,
        round_candidate_diagnostics=[
            {
                "name": "default",
                "selected": False,
                "score": -1.0,
                "closed_trades": 0,
            },
            {
                "name": "triple_barrier_base",
                "selected": True,
                "score": 0.42,
                "closed_trades": 6,
            },
        ],
        strategy_overrides={
            "risk_per_trade": 0.005,
            "signal_threshold": 0.63,
            "take_profit_pct": 0.04,
        },
        meta_label_policy={
            "enabled": True,
            "mode": "take_downsize_skip",
            "take_threshold": 0.12,
            "downsize_threshold": 0.06,
        },
        selection_risk={
            "passed": True,
            "effective_trials": 24,
            "selected_score": 0.12,
            "trial_penalty": 0.01,
            "deflated_score": 0.11,
        },
        execution_validation={
            "passed": True,
            "symbol": "BTCUSDC",
            "walk_forward_gate": {
                "passed": True,
                "reason": None,
                "fold_count": 3,
                "accepted_folds": 3,
                "worst_score": 0.08,
                "worst_realized_pnl": 1.2,
                "worst_max_drawdown": 0.025,
            },
            "stress": {"accepted": True},
            "temporal_robustness": {"accepted": True},
            "portfolio": {"accepted": True},
        },
        ensemble_members=[
            EnsembleMember(
                weights=[0.0],
                bias=2.0,
                feature_means=[0.0],
                feature_stds=[1.0],
                seed=7,
                epochs=3,
                training_loss=0.4,
                validation_loss=0.5,
            ),
            EnsembleMember(
                weights=[0.0],
                bias=-2.0,
                feature_means=[0.0],
                feature_stds=[1.0],
                seed=11,
                epochs=3,
            ),
        ],
    )
    from simple_ai_trading.model import serialize_model

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=1)
    assert model_decision_threshold(loaded, 0.55) == 0.65
    assert model_direction_thresholds(loaded, 0.55, market_type="futures") == pytest.approx((0.71, 0.27))
    assert loaded.threshold_diagnostic_best_long_threshold == pytest.approx(0.69)
    assert loaded.threshold_diagnostic_best_short_threshold == pytest.approx(0.31)
    assert loaded.calibration_size == 12
    assert loaded.validation_size == 8
    assert loaded.training_cutoff_timestamp == 123
    assert loaded.probability_temperature == 2.0
    assert loaded.probability_calibration_size == 6
    assert loaded.probability_brier_after == 0.22
    assert loaded.probability_calibration_backend_requested == "directml"
    assert loaded.probability_calibration_backend_kind == "directml"
    assert loaded.probability_calibration_backend_device == "privateuseone:0"
    assert loaded.threshold_source == "profit_backtest"
    assert loaded.threshold_calibration_score == 12.3
    assert loaded.threshold_calibration_pnl == 4.5
    assert loaded.threshold_calibration_trades == 6
    assert loaded.training_backend_requested == "directml"
    assert loaded.training_backend_kind == "directml"
    assert loaded.training_backend_device == "privateuseone:0"
    assert loaded.training_backend_vendor == "DirectML"
    assert loaded.model_candidate_count == 3
    assert loaded.model_selected_candidate == "triple_barrier_base"
    assert loaded.model_selection_score == pytest.approx(0.42)
    assert loaded.round_candidate_diagnostics[1]["selected"] is True
    assert loaded.round_candidate_diagnostics[1]["closed_trades"] == 6
    assert loaded.strategy_overrides == {
        "risk_per_trade": 0.005,
        "signal_threshold": 0.63,
        "take_profit_pct": 0.04,
    }
    assert loaded.meta_label_policy["enabled"] is True
    assert loaded.meta_label_policy["take_threshold"] == pytest.approx(0.12)
    assert loaded.selection_risk["passed"] is True
    assert loaded.selection_risk["deflated_score"] == pytest.approx(0.11)
    assert loaded.execution_validation["passed"] is True
    assert loaded.execution_validation["symbol"] == "BTCUSDC"
    assert loaded.execution_validation["walk_forward_gate"]["fold_count"] == 3
    assert len(loaded.ensemble_members) == 2
    assert loaded.ensemble_members[0].seed == 7
    assert loaded.ensemble_members[0].training_loss == 0.4
    assert loaded.predict_proba((0.0,)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Feature dimension"):
        loaded.predict_proba(())
    assert confidence_adjusted_probability(0.9, 0.5) == 0.7
    assert confidence_adjusted_probability(0.1, 0.5) == 0.3
    assert confidence_adjusted_probability("bad", 0.5) == 0.5
    assert confidence_adjusted_probability(0.8, None) == 0.8
    assert confidence_adjusted_probability(0.8, "bad") == 0.8
    assert confidence_adjusted_probability(float("nan"), 0.5) == 0.5
    assert confidence_adjusted_probability(0.8, float("nan")) == 0.8
    loaded.decision_threshold = float("nan")
    assert model_decision_threshold(loaded, 0.55) == 0.55
    assert model_decision_threshold(loaded, float("nan")) == 0.5
    loaded.decision_threshold = "bad"
    assert model_decision_threshold(loaded, 0.55) == 0.55


def test_load_model_sanitizes_strategy_overrides(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_payload = {
        "weights": [0.1],
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": 1,
        "epochs": 3,
        "feature_means": [1.0],
        "feature_stds": [1.0],
        "strategy_overrides": {
            "risk_per_trade": 0.004,
            "signal_threshold": "0.7",
            "feature_windows": [1, 2],
            "take_profit_pct": float("nan"),
            "cooldown_minutes": 3,
            "min_position_hold_bars": 4,
            "flat_signal_exit_grace_bars": 2,
            "max_trades_per_day": True,
        },
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    loaded = load_model(model_path, expected_feature_dim=1)

    assert loaded.strategy_overrides == {
        "risk_per_trade": 0.004,
        "cooldown_minutes": 3,
        "min_position_hold_bars": 4,
        "flat_signal_exit_grace_bars": 2,
    }
    assert clean_strategy_overrides(["not", "a", "dict"]) == {}


def test_load_model_rejects_invalid_ensemble_members(tmp_path: Path) -> None:
    model_path = tmp_path / "bad_ensemble.json"
    payload: dict[str, object] = {
        "weights": [0.1],
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": 1,
        "epochs": 3,
        "feature_means": [1.0],
        "feature_stds": [1.0],
        "ensemble_members": [
            {
                "weights": [0.1, 0.2],
                "bias": 0.0,
                "feature_means": [0.0],
                "feature_stds": [1.0],
            }
        ],
    }
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelFeatureMismatchError):
        load_model(model_path, expected_feature_dim=2)
    with pytest.raises(ModelLoadError, match="ensemble member"):
        load_model(model_path, expected_feature_dim=1)

    for ensemble_members, match in [
        ({}, "must be an array"),
        ([1], "is not an object"),
        (
            [
                {
                    "weights": [0.1],
                    "bias": 0.0,
                    "feature_means": [0.0],
                    "feature_stds": [1.0],
                    "training_loss": "nan",
                }
            ],
            "is invalid",
        ),
    ]:
        payload["ensemble_members"] = ensemble_members
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ModelLoadError, match=match):
            load_model(model_path, expected_feature_dim=1)


def test_load_model_rejects_oversized_lightgbm_tree_payload(tmp_path: Path) -> None:
    model_path = tmp_path / "oversized_lightgbm.json"
    payload: dict[str, object] = {
        "weights": [0.1],
        "feature_version": "v1",
        "bias": 0.0,
        "feature_dim": 1,
        "epochs": 1,
        "feature_means": [0.0],
        "feature_stds": [1.0],
        "hybrid_experts": [
            {
                "name": "oversized",
                "kind": "signed_payoff_lightgbm_ranker",
                "weight": 1.0,
                "params": {
                    "input_dim": 1,
                    "tree_info": [
                        {"tree_structure": {"leaf_value": 0.0}}
                        for _ in range(model_module.MAX_SERIALIZED_LIGHTGBM_TREES + 1)
                    ],
                },
            }
        ],
    }
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="tree limit"):
        load_model(model_path, expected_feature_version="v1", expected_feature_dim=1)


def test_temperature_calibration_softens_overconfident_probabilities() -> None:
    rows = [
        ModelRow(timestamp=1, close=1.0, features=(1.0,), label=0),
        ModelRow(timestamp=2, close=1.0, features=(1.0,), label=0),
        ModelRow(timestamp=3, close=1.0, features=(1.0,), label=1),
        ModelRow(timestamp=4, close=1.0, features=(1.0,), label=1),
    ] * 10
    model = TrainedModel(
        weights=[8.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    before = model.predict_proba((1.0,))
    report = calibrate_probability_temperature(rows, model, min_temperature=1.0, max_temperature=6.0, steps=26)
    model.probability_temperature = report.temperature
    after = model.predict_proba((1.0,))
    assert report.improved is True
    assert report.temperature > 1.0
    assert report.log_loss_after < report.log_loss_before
    assert abs(after - 0.5) < abs(before - 0.5)


def test_temperature_calibration_uses_ensemble_predictions() -> None:
    rows = [
        ModelRow(timestamp=index, close=1.0, features=(1.0,), label=index % 2)
        for index in range(40)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        ensemble_members=[
            EnsembleMember(weights=[0.0], bias=8.0, feature_means=[0.0], feature_stds=[1.0]),
            EnsembleMember(weights=[0.0], bias=8.0, feature_means=[0.0], feature_stds=[1.0]),
        ],
    )

    before = model.predict_proba((1.0,))
    report = calibrate_probability_temperature(rows, model, min_temperature=1.0, max_temperature=8.0, steps=29)
    model.probability_temperature = report.temperature
    after = model.predict_proba((1.0,))

    assert report.improved is True
    assert report.temperature > 1.0
    assert report.log_loss_before > 1.0
    assert report.log_loss_after < report.log_loss_before
    assert abs(after - 0.5) < abs(before - 0.5)


def test_temperature_calibration_uses_requested_gpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ModelRow(timestamp=index, close=1.0, features=(1.0,), label=index % 2)
        for index in range(40)
    ]
    model = TrainedModel(
        weights=[8.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    captured: dict[str, object] = {}

    def fake_resolve_backend(requested: str) -> BackendInfo:
        captured["requested"] = requested
        return BackendInfo(
            requested=requested,
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        )

    def fake_temperature_scan(rows_arg, model_arg, candidates, *, backend, batch_size):
        captured["rows"] = len(rows_arg)
        captured["candidate_count"] = len(candidates)
        captured["backend_kind"] = backend.kind
        captured["batch_size"] = batch_size
        return 2.0, 0.2, 0.15

    monkeypatch.setattr(model_module, "resolve_backend", fake_resolve_backend)
    monkeypatch.setattr(model_module, "_temperature_scan_torch", fake_temperature_scan)

    report = calibrate_probability_temperature(
        rows,
        model,
        min_temperature=1.0,
        max_temperature=3.0,
        steps=3,
        compute_backend="directml",
        batch_size=64,
    )

    assert report.improved is True
    assert report.temperature == pytest.approx(2.0)
    assert report.calibration_backend_requested == "directml"
    assert report.calibration_backend_kind == "directml"
    assert report.calibration_backend_device == "privateuseone:0"
    assert captured == {
        "requested": "directml",
        "rows": 40,
        "candidate_count": 3,
        "backend_kind": "directml",
        "batch_size": 64,
    }


def test_temperature_calibration_records_cpu_fallback_when_gpu_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ModelRow(timestamp=index, close=1.0, features=(1.0,), label=index % 2)
        for index in range(40)
    ]
    model = TrainedModel(
        weights=[8.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )

    monkeypatch.setattr(
        model_module,
        "resolve_backend",
        lambda requested: BackendInfo(
            requested=requested,
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        ),
    )

    def fail_temperature_scan(*_args, **_kwargs):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(model_module, "_temperature_scan_torch", fail_temperature_scan)

    report = calibrate_probability_temperature(
        rows,
        model,
        min_temperature=1.0,
        max_temperature=6.0,
        steps=26,
        compute_backend="directml",
        batch_size=64,
    )

    assert report.calibration_backend_requested == "directml"
    assert report.calibration_backend_kind == "cpu"
    assert "temperature calibration failed" in report.calibration_backend_reason
