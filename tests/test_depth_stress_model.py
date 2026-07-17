from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simple_ai_trading.depth_stress_model import (
    DEPTH_STRESS_DESCRIPTOR_NAMES,
    DEPTH_STRESS_MODEL_SCHEMA_VERSION,
    assign_depth_stress_states,
    depth_stress_loss_rows,
    depth_stress_metrics,
    fit_depth_stress_thresholds,
    fit_depth_transition_probabilities,
    orient_depth_stress_descriptors,
    predict_depth_stress_challenger,
    predict_depth_transition_probabilities,
    train_depth_stress_challenger,
)


def _descriptors(rows: int = 120) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    return np.column_stack((axis, axis**2, np.sqrt(axis)))


def test_depth_descriptors_are_oriented_toward_stress() -> None:
    descriptors = orient_depth_stress_descriptors(
        bid_near_depth=[80.0, 20.0],
        ask_near_depth=[80.0, 5.0],
        bid_near_notional=[80.0, 20.0],
        ask_near_notional=[80.0, 5.0],
        bid_far_notional=[800.0, 500.0],
        ask_far_notional=[800.0, 100.0],
    )

    assert descriptors.shape == (2, 3)
    assert tuple(DEPTH_STRESS_DESCRIPTOR_NAMES) == (
        "near_depth_thinness",
        "near_depth_absolute_imbalance",
        "far_to_near_depth_concentration",
    )
    assert descriptors[1, 0] > descriptors[0, 0]
    assert descriptors[1, 1] > descriptors[0, 1]
    assert descriptors[1, 2] > descriptors[0, 2]


@pytest.mark.parametrize(
    "changes",
    [
        {"bid_near_depth": [float("nan")]},
        {"ask_near_depth": [-1.0]},
        {
            "bid_near_notional": [10.0],
            "ask_near_notional": [10.0],
            "bid_far_notional": [5.0],
            "ask_far_notional": [5.0],
        },
    ],
)
def test_depth_descriptors_reject_invalid_provider_values(changes) -> None:
    values = {
        "bid_near_depth": [10.0],
        "ask_near_depth": [10.0],
        "bid_near_notional": [10.0],
        "ask_near_notional": [10.0],
        "bid_far_notional": [100.0],
        "ask_far_notional": [100.0],
    }
    values.update(changes)

    with pytest.raises(ValueError, match="depth-stress"):
        orient_depth_stress_descriptors(**values)


def test_threshold_fit_cannot_see_rows_outside_explicit_fit() -> None:
    first = _descriptors()
    second = first.copy()
    second[60:] += 1_000_000.0

    threshold_a = fit_depth_stress_thresholds(first, np.arange(60))
    threshold_b = fit_depth_stress_thresholds(second, np.arange(60))

    assert threshold_a.upper_tercile == threshold_b.upper_tercile
    assert threshold_a.fit_fingerprint == threshold_b.fit_fingerprint
    assert threshold_a.fitted_rows == 60


def test_state_encoder_counts_severe_descriptors_and_fails_closed() -> None:
    thresholds = fit_depth_stress_thresholds(_descriptors(), np.arange(90))
    cut = np.asarray(thresholds.upper_tercile)
    rows = np.vstack(
        (
            cut - 1.0,
            cut + np.asarray([1.0, -1.0, -1.0]),
            cut + np.asarray([1.0, 1.0, -1.0]),
            np.asarray([np.nan, 0.0, 0.0]),
        )
    )

    assert assign_depth_stress_states(rows, thresholds).tolist() == [0, 1, 2, 2]
    with pytest.raises(ValueError, match="non-finite"):
        assign_depth_stress_states(rows, thresholds, fail_closed=False)


def test_conditional_transition_baseline_and_proper_scores() -> None:
    pre = np.asarray([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int8)
    post = np.asarray([0, 0, 1, 1, 2, 2, 2, 1], dtype=np.int8)
    transition = fit_depth_transition_probabilities(pre, post, alpha=1.0)
    probabilities = predict_depth_transition_probabilities(transition, pre)
    metrics = depth_stress_metrics(post, probabilities)
    losses = depth_stress_loss_rows(post, probabilities)

    assert transition.shape == (3, 3)
    assert np.allclose(np.sum(transition, axis=1), 1.0)
    assert transition[0, 0] > transition[0, 2]
    assert transition[2, 2] > transition[2, 0]
    assert metrics.rows == len(post)
    assert metrics.negative_log_likelihood > 0.0
    assert 0.0 <= metrics.multiclass_brier <= 2.0
    assert 0.0 <= metrics.stressed_brier <= 1.0
    assert metrics.negative_log_likelihood == pytest.approx(
        np.mean(losses["negative_log_likelihood"])
    )
    assert metrics.multiclass_brier == pytest.approx(np.mean(losses["multiclass_brier"]))
    assert metrics.stressed_brier == pytest.approx(np.mean(losses["stressed_brier"]))


def test_marginal_baseline_has_identical_probability_rows() -> None:
    pre = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int8)
    post = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    marginal = fit_depth_transition_probabilities(
        pre,
        post,
        condition_on_pre_state=False,
    )

    assert np.allclose(marginal[0], marginal[1])
    assert np.allclose(marginal[1], marginal[2])


def _model_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = 390
    rng = np.random.default_rng(20260717)
    labels = (np.arange(rows) % 3).astype(np.int8)
    features = np.column_stack(
        (
            labels + rng.normal(0.0, 0.08, rows),
            np.sin(np.arange(rows) / 13.0),
            rng.normal(0.0, 1.0, rows),
        )
    ).astype(np.float32)
    return features, labels, np.arange(270), np.arange(270, 360)


def test_shallow_challenger_reloads_with_no_trading_authority() -> None:
    features, labels, train, tuning = _model_fixture()
    artifact = train_depth_stress_challenger(
        features,
        labels,
        train_rows=train,
        tuning_rows=tuning,
        feature_names=("state_proxy", "cycle", "noise"),
        compute_backend="cpu",
        maximum_iterations=64,
    )
    probabilities = predict_depth_stress_challenger(artifact, features[360:])

    assert artifact.schema_version == DEPTH_STRESS_MODEL_SCHEMA_VERSION
    assert artifact.model_family == "lightgbm_shallow_multiclass"
    assert artifact.backend_kind == "cpu"
    assert artifact.trading_authority is False
    assert artifact.profitability_claim is False
    assert len(artifact.model_sha256) == 64
    assert probabilities.shape == (30, 3)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    assert np.mean(np.argmax(probabilities, axis=1) == labels[360:]) > 0.90


def test_challenger_rejects_overlap_and_prediction_contract_drift() -> None:
    features, labels, train, tuning = _model_fixture()
    with pytest.raises(ValueError, match="overlap"):
        train_depth_stress_challenger(
            features,
            labels,
            train_rows=train,
            tuning_rows=np.arange(250, 340),
            feature_names=("state_proxy", "cycle", "noise"),
            compute_backend="cpu",
            maximum_iterations=32,
        )
    artifact = train_depth_stress_challenger(
        features,
        labels,
        train_rows=train,
        tuning_rows=tuning,
        feature_names=("state_proxy", "cycle", "noise"),
        compute_backend="cpu",
        maximum_iterations=32,
    )
    with pytest.raises(ValueError, match="artifact contract"):
        replace(artifact, trading_authority=True)
    tampered = replace(artifact)
    object.__setattr__(tampered, "model_string", f"{artifact.model_string}\n")
    with pytest.raises(ValueError, match="artifact digest"):
        predict_depth_stress_challenger(tampered, features[:3])
    invalid = features[:3].copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="prediction matrix"):
        predict_depth_stress_challenger(artifact, invalid)
