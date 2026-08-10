from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from polymarket_round25_support import small_round25_dataset
from simple_ai_trading.polymarket_round25_controls import (
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
    round25_condition_equal_scores,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round25_lightgbm import (
    POLYMARKET_ROUND25_LIGHTGBM_CONFIGS,
    POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256,
    Round25CompiledLightGBM,
    Round25LightGBMConfig,
    _create_round25_lightgbm_artifact,
    _train_round25_lightgbm,
    fit_round25_lightgbm_residual,
    round25_lightgbm_config,
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


@pytest.fixture(scope="module")
def fitted_mechanics() -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(25_025)
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    train_rows = 1_200
    calibration_rows = 800
    train_matrix = np.zeros((train_rows, width), dtype=np.float64)
    calibration_matrix = np.zeros((calibration_rows, width), dtype=np.float64)
    train_matrix[:, 0] = rng.normal(size=train_rows)
    calibration_matrix[:, 0] = rng.normal(size=calibration_rows)
    train_labels = (train_matrix[:, 0] > 0.0).astype(np.float64)
    calibration_labels = (calibration_matrix[:, 0] > 0.0).astype(np.float64)
    train_prior = np.full(train_rows, 0.5, dtype=np.float64)
    calibration_prior = np.full(calibration_rows, 0.5, dtype=np.float64)
    train_weights = np.full(train_rows, 1.0 / 16.0, dtype=np.float64)
    calibration_weights = np.full(
        calibration_rows,
        1.0 / 16.0,
        dtype=np.float64,
    )
    config = round25_lightgbm_config("lightgbm-residual-depth3-v1")
    fitted = _train_round25_lightgbm(
        config=config,
        train_matrix=train_matrix,
        train_labels=train_labels,
        train_prior=train_prior,
        train_weights=train_weights,
        calibration_matrix=calibration_matrix,
        calibration_labels=calibration_labels,
        calibration_prior=calibration_prior,
        calibration_weights=calibration_weights,
        compute_backend="cpu",
    )
    artifact = _create_round25_lightgbm_artifact(
        config=config,
        train_dataset_sha256="a" * 64,
        calibration_dataset_sha256="b" * 64,
        train_resolution_authority_sha256="c" * 64,
        calibration_resolution_authority_sha256="c" * 64,
        center=np.zeros(width, dtype=np.float64),
        scale=np.ones(width, dtype=np.float64),
        fitted=fitted,
    )
    return artifact, calibration_matrix, calibration_labels, calibration_prior


def test_lightgbm_fit_contract_is_self_hashed_and_claim_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-lightgbm-fit-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["dataset_gates"]["undersized_fit_allowed"] is False
    assert contract["shared_training"]["residual_logit_bound"] == 4.0
    assert contract["truth_state"]["lightgbm_model_fitted"] is False
    assert contract["truth_state"]["edge_verified"] is False
    assert contract["truth_state"]["live_authority"] is False


def test_lightgbm_candidate_configs_are_finite_and_condition_bounded() -> None:
    assert [config.candidate_id for config in POLYMARKET_ROUND25_LIGHTGBM_CONFIGS] == [
        "lightgbm-residual-depth3-v1",
        "lightgbm-residual-depth5-v1",
    ]
    assert all(
        config.minimum_rows_per_leaf
        == config.minimum_conditions_per_leaf * 16
        for config in POLYMARKET_ROUND25_LIGHTGBM_CONFIGS
    )
    with pytest.raises(ValueError, match="configuration differs"):
        Round25LightGBMConfig(
            candidate_id="lightgbm-residual-depth3-v1",
            max_depth=9,
            num_leaves=8,
            learning_rate=0.03,
            maximum_trees=500,
            early_stopping_rounds=50,
            minimum_conditions_per_leaf=25,
            minimum_rows_per_leaf=400,
        )
    with pytest.raises(ValueError, match="not frozen"):
        round25_lightgbm_config("unregistered-candidate")


def test_public_lightgbm_fit_rejects_undersized_data_before_backend_resolution() -> None:
    with pytest.raises(ValueError, match="minimum condition gate"):
        fit_round25_lightgbm_residual(
            candidate_id="lightgbm-residual-depth3-v1",
            train=small_round25_dataset("train"),
            calibration=small_round25_dataset("calibration"),
            compute_backend="directml",
        )


def test_lightgbm_mechanics_serialize_bound_and_improve_ephemeral_signal(
    fitted_mechanics: tuple[object, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    artifact, matrix, labels, prior = fitted_mechanics
    assert hasattr(artifact, "validated")
    selected = artifact.validated()  # type: ignore[union-attr]
    runtime = Round25CompiledLightGBM(selected)
    probabilities = np.asarray(
        runtime.predict_probabilities(matrix, prior),
        dtype=np.float64,
    )
    weights = np.full(len(labels), 1.0 / 16.0, dtype=np.float64)
    fitted_log_loss, fitted_brier = round25_condition_equal_scores(
        labels,
        np.log(probabilities) - np.log1p(-probabilities),
        weights,
    )
    baseline_log_loss, baseline_brier = round25_condition_equal_scores(
        labels,
        np.zeros_like(labels),
        weights,
    )
    lower = float(expit(-POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND))
    upper = float(expit(POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND))

    assert selected.backend_kind == "cpu"
    assert selected.best_iteration <= selected.config.maximum_trees
    assert runtime.artifact_sha256 == selected.artifact_sha256
    assert fitted_log_loss < baseline_log_loss
    assert fitted_brier < baseline_brier
    assert np.min(probabilities) >= lower
    assert np.max(probabilities) <= upper
    with pytest.raises(ValueError, match="artifact differs"):
        replace(selected, backend_device="tampered")

    invalid = matrix[:1].copy()
    availability_index = next(
        index
        for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        if name.endswith("_available")
    )
    invalid[0, availability_index] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        runtime.predict_probabilities(invalid, prior[:1])
