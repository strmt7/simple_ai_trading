from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from polymarket_round25_support import small_round25_dataset

from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from simple_ai_trading.polymarket_round25_controls import (
    POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND25_L2_GRID,
    POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION,
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
    Round25L2CalibrationScore,
    Round25LogisticResidualArtifact,
    _apply_isotonic,
    _fit_bounded_logistic,
    _phase_index,
    _weighted_isotonic_thresholds,
    fit_round25_feature_transform,
    fit_round25_logistic_residual,
    fit_round25_phase_isotonic,
    predict_round25_logistic_residual_probability,
    round25_bounded_residual_linear,
    round25_condition_equal_scores,
    round25_market_prior_predictions,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _logistic_artifact() -> Round25LogisticResidualArtifact:
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    scores = tuple(
        Round25L2CalibrationScore(
            l2=l2,
            condition_equal_log_loss=0.5 + index * 0.1,
            condition_equal_brier_score=0.2 + index * 0.01,
        )
        for index, l2 in enumerate(POLYMARKET_ROUND25_L2_GRID)
    )
    payload = {
        "calibration_dataset_sha256": "b" * 64,
        "calibration_resolution_authority_sha256": "c" * 64,
        "calibration_scores": [score.payload() for score in scores],
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_id": "l2-logistic-residual-v1",
        "center": [0.0] * width,
        "coefficients": [0.0] * width,
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "fit_role": "train",
        "intercept": 1.0,
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "residual_logit_bound": POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
        "scale": [1.0] * width,
        "schema_version": POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION,
        "selected_l2": POLYMARKET_ROUND25_L2_GRID[0],
        "selection_role": "calibration",
        "trading_authority": False,
        "train_dataset_sha256": "a" * 64,
        "train_resolution_authority_sha256": "c" * 64,
    }
    return Round25LogisticResidualArtifact(
        train_dataset_sha256="a" * 64,
        calibration_dataset_sha256="b" * 64,
        train_resolution_authority_sha256="c" * 64,
        calibration_resolution_authority_sha256="c" * 64,
        center=(0.0,) * width,
        scale=(1.0,) * width,
        selected_l2=POLYMARKET_ROUND25_L2_GRID[0],
        intercept=1.0,
        coefficients=(0.0,) * width,
        calibration_scores=scores,
        artifact_sha256=_canonical_sha256(payload),
    )


def test_control_fit_contract_is_self_hashed_and_frozen_before_capture() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-control-fit-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["status"].startswith("frozen_target_and_outcome_blind")
    assert contract["dataset_gates"]["undersized_fit_allowed"] is False
    assert contract["truth_state"]["edge_verified"] is False
    assert contract["truth_state"]["live_authority"] is False


@pytest.mark.parametrize(
    ("elapsed_ms", "phase"),
    [(0, 0), (74_999, 0), (75_000, 1), (150_000, 2), (225_000, 3), (299_999, 3)],
)
def test_phase_boundaries_are_exact(elapsed_ms: int, phase: int) -> None:
    start = POLYMARKET_ROUND25_CAMPAIGN_START_MS
    assert _phase_index(start, start + elapsed_ms) == phase


def test_weighted_pav_aggregates_ties_and_is_monotone() -> None:
    x = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    w = np.ones(4, dtype=np.float64)

    thresholds_x, thresholds_y = _weighted_isotonic_thresholds(x, y, w)
    prediction = _apply_isotonic(x, thresholds_x, thresholds_y)

    assert thresholds_x == (0.1, 0.2, 0.3, 0.4)
    assert thresholds_y == (0.0, 0.5, 0.5, 1.0)
    assert np.all(np.diff(prediction) >= 0.0)

    tied_x = np.asarray([0.2, 0.2, 0.8, 0.8], dtype=np.float64)
    tied_y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    tied_thresholds = _weighted_isotonic_thresholds(tied_x, tied_y, w)
    assert tied_thresholds == ((0.2, 0.8), (0.5, 0.5))


def test_isotonic_rejects_single_class_and_invalid_prior() -> None:
    with pytest.raises(ValueError, match="population"):
        _weighted_isotonic_thresholds(
            np.asarray([0.2, 0.8]),
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="prediction input"):
        _apply_isotonic(np.asarray([1.0]), (0.2, 0.8), (0.2, 0.8))


def test_training_transform_uses_iqr_and_leaves_availability_unscaled() -> None:
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    matrix = np.tile(np.asarray([1.0, 2.0, 3.0, 4.0])[:, None], (1, width))
    binary_indices = [
        index
        for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        if name.endswith("_available")
    ]
    matrix[:, binary_indices] = np.asarray([0.0, 1.0, 1.0, 0.0])[:, None]
    matrix[:, 0] = 7.0

    center, scale = fit_round25_feature_transform(matrix)

    assert center[0] == 7.0
    assert scale[0] == 1.0
    assert center[1] == 2.5
    assert scale[1] == 1.5
    assert np.all(center[binary_indices] == 0.0)
    assert np.all(scale[binary_indices] == 1.0)

    matrix[0, binary_indices[0]] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        fit_round25_feature_transform(matrix)


def test_bounded_logistic_improves_synthetic_signal_without_exceeding_bound() -> None:
    rng = np.random.default_rng(1729)
    matrix = rng.normal(size=(400, 2))
    prior = np.full(400, 0.5, dtype=np.float64)
    labels = (matrix[:, 0] + 0.25 * matrix[:, 1] > 0.0).astype(np.float64)
    weights = np.ones(400, dtype=np.float64)

    intercept, coefficients = _fit_bounded_logistic(
        matrix,
        labels,
        prior,
        weights,
        l2=0.1,
    )
    linear = round25_bounded_residual_linear(
        matrix,
        prior,
        intercept,
        coefficients,
    )
    fitted_log_loss, _ = round25_condition_equal_scores(labels, linear, weights)
    baseline_log_loss, _ = round25_condition_equal_scores(
        labels,
        np.zeros_like(labels),
        weights,
    )

    assert fitted_log_loss < baseline_log_loss
    assert np.max(np.abs(linear)) <= POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    assert np.mean((expit(linear) >= 0.5) == labels) > 0.9


def test_logistic_artifact_is_hash_bound_and_target_free_at_inference() -> None:
    artifact = _logistic_artifact()
    probability = predict_round25_logistic_residual_probability(
        artifact,
        feature_values=(0.0,) * len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES),
        market_prior_probability=0.5,
    )

    bounded_intercept = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * np.tanh(
        1.0 / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    )
    assert probability == pytest.approx(float(expit(bounded_intercept)))
    assert artifact.trading_authority is False
    with pytest.raises(ValueError, match="artifact differs"):
        replace(artifact, selected_l2=0.1)
    with pytest.raises(ValueError, match="numeric vector"):
        predict_round25_logistic_residual_probability(
            artifact,
            feature_values=(0.0,),
            market_prior_probability=0.5,
        )


def test_all_public_fit_paths_reject_undersized_development_data() -> None:
    train = small_round25_dataset("train")
    calibration = small_round25_dataset("calibration")

    with pytest.raises(ValueError, match="minimum condition gate"):
        round25_market_prior_predictions(train)
    with pytest.raises(ValueError, match="minimum condition gate"):
        fit_round25_phase_isotonic(calibration)
    with pytest.raises(ValueError, match="minimum condition gate"):
        fit_round25_logistic_residual(train=train, calibration=calibration)


def test_control_score_and_phase_timestamp_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="score"):
        Round25L2CalibrationScore(
            l2=0.2,
            condition_equal_log_loss=0.5,
            condition_equal_brier_score=0.2,
        )
    with pytest.raises(ValueError, match="timestamp"):
        _phase_index(
            POLYMARKET_ROUND25_CAMPAIGN_START_MS,
            POLYMARKET_ROUND25_CAMPAIGN_START_MS + 300_000,
        )
