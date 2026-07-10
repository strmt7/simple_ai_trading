from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from simple_ai_trading import tape_depth_model as tape_model
from simple_ai_trading.tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_FEATURE_VERSION,
    TAPE_DEPTH_TARGET_MODE,
    TapeDepthForecastDataset,
)
from simple_ai_trading.tape_depth_model import (
    TapeDepthSignalPolicy,
    _selected_feature_names,
    load_tape_depth_model_artifact,
    save_tape_depth_model_artifact,
    score_tape_depth_evaluation,
    train_tape_depth_forecaster,
)


def _predictive_dataset(rows: int = 12_000) -> TapeDepthForecastDataset:
    rng = np.random.default_rng(20260710)
    features = rng.normal(0.0, 1.0, size=(rows, len(TAPE_DEPTH_FEATURE_NAMES))).astype(
        np.float32
    )
    signal = features[:, 0] + 0.5 * features[:, 1]
    targets = (3.0 * signal + rng.normal(0.0, 0.35, size=rows)).astype(np.float64)
    depth_available = TAPE_DEPTH_FEATURE_NAMES.index("depth_available")
    depth_age = TAPE_DEPTH_FEATURE_NAMES.index("depth_age_seconds")
    features[:, depth_available] = 1.0
    features[:, depth_age] = np.abs(features[:, depth_age]) * 30.0
    base_ms = 1_700_000_000_000
    times = base_ms + np.arange(rows, dtype=np.int64) * 5_000
    prices = 100.0 + np.arange(rows, dtype=np.float64) * 0.001
    return TapeDepthForecastDataset(
        symbol="BTCUSDT",
        feature_version=TAPE_DEPTH_FEATURE_VERSION,
        feature_names=TAPE_DEPTH_FEATURE_NAMES,
        target_mode=TAPE_DEPTH_TARGET_MODE,
        horizon_seconds=60,
        total_latency_ms=750,
        decision_cadence_seconds=5,
        maximum_depth_age_ms=60_000,
        decision_time_ms=times,
        target_entry_time_ms=times + 1_000,
        target_exit_time_ms=times + 61_000,
        target_entry_price=prices,
        target_exit_price=prices * (1.0 + targets / 10_000.0),
        gross_return_bps=targets,
        features=features,
        source_evidence={
            "verified": True,
            "schema_version": "binance-usdm-tick-v6",
            "manifest_fingerprint": "a" * 64,
        },
    )


def test_tape_depth_forecaster_is_predictive_but_never_executable(tmp_path) -> None:
    dataset = _predictive_dataset()
    artifact = train_tape_depth_forecaster(
        dataset,
        risk_level="conservative",
        compute_backend="cpu",
        minimum_segment_rows=500,
    )

    assert artifact.status == "research_candidate"
    assert artifact.rejection_reasons == ()
    assert artifact.trading_authority is False
    assert artifact.execution_claim is False
    assert artifact.split.purge_ms == 61_000
    assert artifact.evaluation_metrics.direction_auc > 0.90
    assert (
        artifact.evaluation_metrics.mean_absolute_error_bps
        < artifact.evaluation_metrics.zero_baseline_mae_bps
    )
    assert artifact.evaluation_metrics.calibration_threshold_mean_signed_gross_bps > 0.0
    assert set(artifact.model_strings) == {"direction", "mean", "lower", "upper"}
    assert artifact.training_weight_scale_bps > 0.0
    assert artifact.signal_threshold_bps > 0.0
    assert artifact.feature_set == "full"
    assert artifact.model_feature_names == TAPE_DEPTH_FEATURE_NAMES
    assert len(artifact.dataset_fingerprint) == 64

    replay = score_tape_depth_evaluation(artifact, dataset)
    assert replay.rows == artifact.evaluation_metrics.rows
    assert replay.metrics() == artifact.evaluation_metrics
    assert len(replay.fingerprint()) == 64
    no_action_metrics = replace(
        replay,
        signal_policy=replace(
            replay.signal_policy,
            signal_threshold_bps=1e9,
        ),
    ).metrics()
    assert no_action_metrics.calibration_threshold_rows == 0
    assert no_action_metrics.calibration_threshold_signed_gross_bps == 0.0
    assert no_action_metrics.calibration_threshold_mean_signed_gross_bps == 0.0
    assert no_action_metrics.calibration_threshold_positive_rate == 0.0

    path = tmp_path / "tape-depth.json"
    save_tape_depth_model_artifact(artifact, path)
    loaded = load_tape_depth_model_artifact(path)
    assert loaded == artifact


def test_tape_depth_loader_rejects_forged_trading_authority(tmp_path) -> None:
    artifact = train_tape_depth_forecaster(
        _predictive_dataset(6_000),
        compute_backend="cpu",
        minimum_segment_rows=256,
    )
    payload = artifact.asdict()
    payload["trading_authority"] = True
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot authorize trading"):
        load_tape_depth_model_artifact(path)


def test_tape_depth_replay_rejects_feature_drift() -> None:
    dataset = _predictive_dataset(6_000)
    artifact = train_tape_depth_forecaster(
        dataset,
        compute_backend="cpu",
        minimum_segment_rows=256,
    )
    dataset.features[0, 0] += 0.25

    with pytest.raises(ValueError, match="fingerprint differs"):
        score_tape_depth_evaluation(artifact, dataset)


def test_tape_depth_forecaster_honors_timestamp_split_boundaries() -> None:
    dataset = _predictive_dataset(12_000)
    boundaries = tuple(int(dataset.decision_time_ms[index]) for index in (6_000, 8_000, 10_000))

    artifact = train_tape_depth_forecaster(
        dataset,
        compute_backend="cpu",
        minimum_segment_rows=500,
        split_boundaries_ms=boundaries,
    )

    assert artifact.split.tuning_start_ms == boundaries[0]
    assert artifact.split.calibration_start_ms == boundaries[1]
    assert artifact.split.evaluation_start_ms == boundaries[2]
    assert score_tape_depth_evaluation(artifact, dataset).rows == 2_000


def test_tape_depth_predictor_is_shared_but_decision_policy_is_risk_specific() -> None:
    dataset = _predictive_dataset(6_000)
    conservative = train_tape_depth_forecaster(
        dataset,
        risk_level="conservative",
        model_profile="regularized",
        compute_backend="cpu",
        minimum_segment_rows=256,
    )
    aggressive = train_tape_depth_forecaster(
        dataset,
        risk_level="aggressive",
        model_profile="regularized",
        compute_backend="cpu",
        minimum_segment_rows=256,
    )

    assert conservative.risk_level == "conservative"
    assert aggressive.risk_level == "aggressive"
    assert conservative.model_profile == aggressive.model_profile == "regularized"
    assert conservative.best_iterations == aggressive.best_iterations
    assert conservative.model_strings == aggressive.model_strings
    assert conservative.signal_policy.magnitude_quantile == 0.95
    assert aggressive.signal_policy.magnitude_quantile == 0.80
    assert conservative.signal_policy.minimum_direction_probability == 0.60
    assert aggressive.signal_policy.minimum_direction_probability == 0.52
    assert (
        conservative.signal_policy.signal_threshold_bps
        >= aggressive.signal_policy.signal_threshold_bps
    )
    assert (
        conservative.evaluation_metrics.calibration_threshold_rows
        <= aggressive.evaluation_metrics.calibration_threshold_rows
    )
    assert conservative.evaluation_metrics != aggressive.evaluation_metrics


def test_tape_depth_training_statistics_do_not_read_evaluation_targets() -> None:
    dataset = _predictive_dataset(6_000)
    changed_targets = np.asarray(dataset.gross_return_bps, dtype=np.float64).copy()
    changed_targets[int(dataset.rows * 0.85) :] *= 100.0
    changed_evaluation = replace(
        dataset,
        gross_return_bps=changed_targets,
        target_exit_price=(
            np.asarray(dataset.target_entry_price, dtype=np.float64)
            * (1.0 + changed_targets / 10_000.0)
        ),
    )

    original = train_tape_depth_forecaster(
        dataset,
        compute_backend="cpu",
        minimum_segment_rows=256,
    )
    future_changed = train_tape_depth_forecaster(
        changed_evaluation,
        compute_backend="cpu",
        minimum_segment_rows=256,
    )

    assert original.training_weight_scale_bps == future_changed.training_weight_scale_bps
    assert original.best_iterations == future_changed.best_iterations
    assert original.model_strings == future_changed.model_strings
    assert original.probability_calibration == future_changed.probability_calibration
    assert original.signal_policy == future_changed.signal_policy


def test_tape_depth_replay_recomputes_training_scale_and_calibration_threshold() -> None:
    dataset = _predictive_dataset(6_000)
    artifact = train_tape_depth_forecaster(
        dataset,
        compute_backend="cpu",
        minimum_segment_rows=256,
    )

    with pytest.raises(ValueError, match="weight scale differs"):
        score_tape_depth_evaluation(
            replace(
                artifact,
                training_weight_scale_bps=artifact.training_weight_scale_bps + 1.0,
            ),
            dataset,
        )
    with pytest.raises(ValueError, match="signal policy differs at signal_threshold_bps"):
        score_tape_depth_evaluation(
            replace(
                artifact,
                signal_policy=replace(
                    artifact.signal_policy,
                    signal_threshold_bps=artifact.signal_threshold_bps + 1.0,
                ),
            ),
            dataset,
        )


def test_tape_depth_feature_sets_are_ordered_explicit_ablations() -> None:
    core = _selected_feature_names("core")
    derived = _selected_feature_names("tape_derived")
    cross_asset = _selected_feature_names("cross_asset")
    full = _selected_feature_names("full")

    assert set(core) < set(derived) < set(cross_asset) < set(full)
    assert tuple(name for name in full if name in core) == core
    assert tuple(name for name in full if name in derived) == derived
    assert tuple(name for name in full if name in cross_asset) == cross_asset
    assert not any(name.startswith("depth_") for name in derived)
    assert any(name.startswith("vwap_deviation_bps_") for name in derived)
    assert not any(name.startswith("peer_mean_return_bps_") for name in derived)
    assert any(name.startswith("peer_mean_return_bps_") for name in cross_asset)
    assert any(name.startswith("depth_") for name in full)


def test_tape_depth_forecaster_rejects_an_unknown_feature_ablation() -> None:
    with pytest.raises(ValueError, match="feature_set must be"):
        train_tape_depth_forecaster(
            _predictive_dataset(10),
            feature_set="unknown",
            compute_backend="cpu",
        )


def test_tape_depth_learned_statistics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="calibration targets are invalid"):
        tape_model._weight_scale(np.asarray([np.nan]))
    with pytest.raises(ValueError, match="scale must be positive"):
        tape_model._weights(np.asarray([1.0]), scale_bps=0.0)
    with pytest.raises(ValueError, match="signal policy"):
        TapeDepthSignalPolicy(
            risk_level="conservative",
            magnitude_quantile=0.95,
            minimum_direction_probability=0.60,
            interval_width_quantile=0.75,
            signal_threshold_bps=-1.0,
            maximum_interval_width_bps=2.0,
            direction_baseline_probability=0.5,
        )


def test_tape_depth_baselines_are_frozen_from_calibration_not_evaluation() -> None:
    targets = np.asarray([1.0, 2.0, 3.0, -1.0])
    policy = TapeDepthSignalPolicy(
        risk_level="regular",
        magnitude_quantile=0.90,
        minimum_direction_probability=0.56,
        interval_width_quantile=0.90,
        signal_threshold_bps=0.5,
        maximum_interval_width_bps=2.0,
        direction_baseline_probability=0.25,
    )

    metrics = tape_model._evaluation_metrics(
        targets=targets,
        direction_probability=np.asarray([0.8, 0.8, 0.8, 0.2]),
        mean_prediction=targets,
        lower_prediction=targets - 0.5,
        upper_prediction=targets + 0.5,
        signal_policy=policy,
    )

    assert metrics.prevalence_brier == pytest.approx(0.4375)
    assert metrics.majority_accuracy == pytest.approx(0.25)


def test_tape_depth_zero_action_evaluation_is_rejected(monkeypatch) -> None:
    original_metrics = tape_model._evaluation_metrics

    def no_action_metrics(**kwargs):
        metrics = original_metrics(**kwargs)
        return replace(
            metrics,
            calibration_threshold_rows=0,
            calibration_threshold_long_rows=0,
            calibration_threshold_short_rows=0,
            calibration_threshold_selection_rate=0.0,
            calibration_threshold_signed_gross_bps=0.0,
            calibration_threshold_mean_signed_gross_bps=0.0,
            calibration_threshold_positive_rate=0.0,
        )

    monkeypatch.setattr(tape_model, "_evaluation_metrics", no_action_metrics)
    artifact = train_tape_depth_forecaster(
        _predictive_dataset(6_000),
        compute_backend="cpu",
        minimum_segment_rows=256,
    )

    assert artifact.status == "rejected"
    assert "calibration_threshold_signed_gross_return_not_positive" in (
        artifact.rejection_reasons
    )


def test_tape_depth_loader_rejects_missing_or_invalid_learned_statistics(
    tmp_path,
) -> None:
    artifact = train_tape_depth_forecaster(
        _predictive_dataset(6_000),
        compute_backend="cpu",
        minimum_segment_rows=256,
    )
    missing = artifact.asdict()
    missing.pop("signal_policy")
    missing_path = tmp_path / "missing-threshold.json"
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration fields are incomplete"):
        load_tape_depth_model_artifact(missing_path)

    invalid = artifact.asdict()
    invalid["training_weight_scale_bps"] = 0.0
    invalid_path = tmp_path / "invalid-weight-scale.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration fields are invalid"):
        load_tape_depth_model_artifact(invalid_path)
