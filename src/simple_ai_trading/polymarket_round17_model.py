"""Train/tune-only candidates for the Round 17 BTC five-minute model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .lightgbm_backend import lightgbm_backend_parameters
from .polymarket_round17_features import (
    POLYMARKET_ROUND17_CONTRACT_SHA256,
    POLYMARKET_ROUND17_FEATURE_NAMES,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
    PolymarketRound17FeatureRow,
)


POLYMARKET_ROUND17_PRETEST_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-development-pretest-v1"
)
_MODEL_ROLES = ("train", "tune_calibration", "tune_selection")
_PANEL_ROLES = (
    *_MODEL_ROLES,
    "tune_uncertainty",
    "tune_economic",
)
_PROBABILITY_FLOOR = 1e-6
_EMBARGO_MS = 3_600_000
_COHORT_PLAN_SHA256 = "37fede4da0d6c504bce7cb763b9bd49032e0252a8cede045f29f05acff67fc00"
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_LOGISTIC_L2_GRID = (0.01, 0.1, 1.0)
_LIGHTGBM_GRID = (
    {
        "max_depth": 2,
        "num_leaves": 3,
        "min_data_in_leaf": 20,
    },
    {
        "max_depth": 3,
        "num_leaves": 7,
        "min_data_in_leaf": 40,
    },
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _probability(values: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(selected)):
        raise ValueError("Round 17 prediction is non-finite")
    return np.clip(selected, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)


@dataclass(frozen=True, slots=True)
class Round17DevelopmentPanel:
    role: str
    condition_ids: np.ndarray
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    dataset_sha256: str
    target_manifest_sha256: str
    feature_names_sha256: str = POLYMARKET_ROUND17_FEATURE_NAMES_SHA256
    cohort_plan_sha256: str = _COHORT_PLAN_SHA256

    def validate(self) -> Round17DevelopmentPanel:
        role = str(self.role or "").strip()
        condition_ids = np.asarray(self.condition_ids, dtype=object)
        event_starts = np.asarray(self.event_start_ms, dtype=np.int64)
        decisions = np.asarray(self.decision_time_ms, dtype=np.int64)
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.float64)
        rows = len(labels)
        if (
            role not in _PANEL_ROLES
            or rows < 1
            or condition_ids.shape != (rows,)
            or event_starts.shape != (rows,)
            or decisions.shape != (rows,)
            or features.shape != (rows, len(POLYMARKET_ROUND17_FEATURE_NAMES))
            or not np.all(np.isfinite(features))
            or not np.all(np.isin(labels, (0.0, 1.0)))
            or len(str(self.dataset_sha256)) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(self.dataset_sha256)
            )
            or len(str(self.target_manifest_sha256)) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(self.target_manifest_sha256)
            )
            or self.feature_names_sha256 != POLYMARKET_ROUND17_FEATURE_NAMES_SHA256
            or self.cohort_plan_sha256 != _COHORT_PLAN_SHA256
            or np.any(event_starts <= 0)
            or np.any(event_starts % 300_000)
            or np.any(decisions < event_starts)
            or np.any(decisions >= event_starts + 300_000)
            or np.any((decisions - event_starts) % 250)
            or any(
                _CONDITION_ID.fullmatch(str(value or "").strip()) is None
                for value in condition_ids
            )
            or len(np.unique(labels)) != 2
        ):
            raise ValueError("Round 17 development panel is invalid")
        ordered = np.lexsort((decisions, event_starts))
        if not np.array_equal(ordered, np.arange(rows)):
            raise ValueError("Round 17 development panel is not chronological")
        for condition in np.unique(condition_ids):
            selected = np.flatnonzero(condition_ids == condition)
            if (
                len(np.unique(event_starts[selected])) != 1
                or len(np.unique(labels[selected])) != 1
            ):
                raise ValueError("Round 17 condition target identity differs")
        return self


def _condition_weights(condition_ids: np.ndarray) -> np.ndarray:
    selected = np.asarray(condition_ids, dtype=object)
    weights = np.zeros(len(selected), dtype=np.float64)
    for condition in np.unique(selected):
        indices = np.flatnonzero(selected == condition)
        weights[indices] = 1.0 / len(indices)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Round 17 condition weights are empty")
    return weights / total


def _weighted_log_loss(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> float:
    probability = _probability(predictions)
    target = np.asarray(labels, dtype=np.float64)
    selected_weights = np.asarray(weights, dtype=np.float64)
    return float(
        -np.sum(
            selected_weights
            * (target * np.log(probability) + (1.0 - target) * np.log1p(-probability))
        )
        / np.sum(selected_weights)
    )


def _condition_losses(
    panel: Round17DevelopmentPanel,
    predictions: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    probability = _probability(predictions)
    conditions = tuple(str(value) for value in np.unique(panel.condition_ids))
    losses: list[float] = []
    for condition in conditions:
        indices = np.flatnonzero(panel.condition_ids == condition)
        weights = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        losses.append(
            _weighted_log_loss(panel.labels[indices], probability[indices], weights)
        )
    return np.asarray(losses, dtype=np.float64), conditions


def _condition_metrics(
    panel: Round17DevelopmentPanel,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    losses, conditions = _condition_losses(panel, predictions)
    probability = _probability(predictions)
    weights = _condition_weights(panel.condition_ids)
    brier = float(np.sum(weights * np.square(probability - panel.labels)))
    labels = panel.labels >= 0.5
    decisions = probability >= 0.5
    positive = labels
    negative = ~labels
    true_positive_rate = (
        0.0
        if not np.any(positive)
        else float(
            np.sum(weights[positive] * decisions[positive]) / np.sum(weights[positive])
        )
    )
    true_negative_rate = (
        0.0
        if not np.any(negative)
        else float(
            np.sum(weights[negative] * (~decisions[negative]))
            / np.sum(weights[negative])
        )
    )
    standard_error = (
        0.0
        if len(losses) < 2
        else float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
    )
    true_positive = float(np.sum(weights * decisions * labels))
    true_negative = float(np.sum(weights * (~decisions) * (~labels)))
    false_positive = float(np.sum(weights * decisions * (~labels)))
    false_negative = float(np.sum(weights * (~decisions) * labels))
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    calibration = _fit_platt(panel.labels, probability, panel.condition_ids)
    expected_calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        if not np.any(selected):
            continue
        bin_weight = float(np.sum(weights[selected]))
        predicted_mean = float(
            np.sum(weights[selected] * probability[selected]) / bin_weight
        )
        observed_mean = float(
            np.sum(weights[selected] * panel.labels[selected]) / bin_weight
        )
        expected_calibration_error += bin_weight * abs(predicted_mean - observed_mean)
    return {
        "condition_count": len(conditions),
        "row_count": len(panel.labels),
        "condition_balanced_log_loss": float(np.mean(losses)),
        "condition_log_loss_standard_error": standard_error,
        "condition_balanced_brier": brier,
        "calibration_intercept": calibration["intercept"],
        "calibration_slope": calibration["slope"],
        "expected_calibration_error": expected_calibration_error,
        "condition_weighted_balanced_accuracy": (
            true_positive_rate + true_negative_rate
        )
        / 2.0,
        "condition_weighted_matthews_correlation": (
            0.0
            if denominator <= 0.0
            else (true_positive * true_negative - false_positive * false_negative)
            / denominator
        ),
    }


def _fit_platt(
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    condition_ids: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.float64)
    raw_logit = logit(_probability(raw_predictions))
    weights = _condition_weights(condition_ids)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        slope = float(parameters[1])
        linear = intercept + slope * raw_logit
        prediction = expit(linear)
        residual = prediction - target
        regularization = 1e-6
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
            + 0.5 * regularization * (intercept**2 + (slope - 1.0) ** 2)
        )
        gradient = np.asarray(
            [
                np.sum(weights * residual) + regularization * intercept,
                np.sum(weights * residual * raw_logit) + regularization * (slope - 1.0),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((None, None), (0.0, 8.0)),
        options={"maxiter": 256, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Round 17 probability calibration failed")
    return {
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
    }


def _apply_platt(
    raw_predictions: np.ndarray,
    calibration: Mapping[str, object],
) -> np.ndarray:
    intercept = float(calibration["intercept"])
    slope = float(calibration["slope"])
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope < 0.0:
        raise ValueError("Round 17 calibration parameters are invalid")
    return _probability(expit(intercept + slope * logit(_probability(raw_predictions))))


def _feature_layers() -> dict[str, tuple[int, ...]]:
    names = POLYMARKET_ROUND17_FEATURE_NAMES
    structural_index = names.index("structural_probability_up")
    core = tuple(
        index
        for index, name in enumerate(names)
        if index != structural_index
        and name != "external_available"
        and not name.startswith("binance_")
    )
    spot = tuple(
        index
        for index, name in enumerate(names)
        if index != structural_index
        and (
            index in core
            or name.startswith("binance_spot_")
            or name == "external_available"
        )
    )
    full = tuple(index for index in range(len(names)) if index != structural_index)
    return {
        "chainlink_clob": core,
        "chainlink_clob_spot": spot,
        "chainlink_clob_spot_perpetual": full,
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights)
    threshold = float(quantile) * float(cumulative[-1])
    index = min(
        len(ordered_values) - 1,
        int(np.searchsorted(cumulative, threshold, side="left")),
    )
    return float(ordered_values[index])


def _feature_transform(
    panel: Round17DevelopmentPanel,
    feature_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = panel.features[:, feature_indices]
    weights = _condition_weights(panel.condition_ids)
    lower = np.asarray(
        [
            _weighted_quantile(matrix[:, index], weights, 0.001)
            for index in range(matrix.shape[1])
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            _weighted_quantile(matrix[:, index], weights, 0.999)
            for index in range(matrix.shape[1])
        ],
        dtype=np.float64,
    )
    clipped = np.clip(matrix, lower, upper)
    mean = np.average(clipped, axis=0, weights=weights)
    variance = np.average(np.square(clipped - mean), axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    return (
        lower,
        upper,
        np.asarray(mean, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )


def _model_matrix(
    model: Mapping[str, object],
    panel: Round17DevelopmentPanel,
) -> tuple[np.ndarray, tuple[int, ...]]:
    return _model_matrix_from_features(model, panel.features)


def _model_matrix_from_features(
    model: Mapping[str, object],
    features: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    indices = tuple(
        int(value)
        for value in model["feature_indices"]  # type: ignore[index]
    )
    lower = np.asarray(model["feature_lower"], dtype=np.float64)
    upper = np.asarray(model["feature_upper"], dtype=np.float64)
    if (
        lower.shape != (len(indices),)
        or upper.shape != (len(indices),)
        or np.any(lower > upper)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise ValueError("Round 17 train-fitted feature support is invalid")
    selected = np.asarray(features, dtype=np.float64)
    if (
        selected.ndim != 2
        or selected.shape[1] != len(POLYMARKET_ROUND17_FEATURE_NAMES)
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 17 inference features are invalid")
    return np.clip(selected[:, indices], lower, upper), indices


def _structural_logit(panel: Round17DevelopmentPanel) -> np.ndarray:
    return _structural_logit_from_features(panel.features)


def _structural_logit_from_features(features: np.ndarray) -> np.ndarray:
    index = POLYMARKET_ROUND17_FEATURE_NAMES.index("structural_probability_up")
    selected = np.asarray(features, dtype=np.float64)
    if (
        selected.ndim != 2
        or selected.shape[1] != len(POLYMARKET_ROUND17_FEATURE_NAMES)
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 17 inference features are invalid")
    return logit(_probability(selected[:, index]))


def _fit_logistic_residual(
    train: Round17DevelopmentPanel,
    feature_indices: tuple[int, ...],
    *,
    l2: float,
) -> dict[str, object]:
    lower, upper, mean, scale = _feature_transform(train, feature_indices)
    matrix = (np.clip(train.features[:, feature_indices], lower, upper) - mean) / scale
    offset = _structural_logit(train)
    weights = _condition_weights(train.condition_ids)
    target = train.labels

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        coefficient = parameters[1:]
        linear = offset + intercept + matrix @ coefficient
        prediction = expit(linear)
        residual = prediction - target
        penalty = 0.5 * float(l2) * float(coefficient @ coefficient)
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, linear) - target * linear)) + penalty
        )
        gradient = np.concatenate(
            (
                np.asarray([np.sum(weights * residual)], dtype=np.float64),
                matrix.T @ (weights * residual) + float(l2) * coefficient,
            )
        )
        return loss, gradient

    initial = np.zeros(len(feature_indices) + 1, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 512, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Round 17 residual logistic fit failed")
    return {
        "family": "logistic_residual",
        "l2": float(l2),
        "feature_indices": list(feature_indices),
        "feature_names": [
            POLYMARKET_ROUND17_FEATURE_NAMES[index] for index in feature_indices
        ],
        "feature_lower": lower.tolist(),
        "feature_upper": upper.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(result.x[0]),
        "coefficient": result.x[1:].tolist(),
    }


def _fit_lightgbm_residual(
    train: Round17DevelopmentPanel,
    calibration: Round17DevelopmentPanel,
    feature_indices: tuple[int, ...],
    *,
    configuration: Mapping[str, int],
    backend_parameters: Mapping[str, object],
    backend_kind: str,
    backend_device: str,
    seed: int,
) -> dict[str, object]:
    lower, upper, _mean, _scale = _feature_transform(train, feature_indices)
    train_matrix = np.clip(train.features[:, feature_indices], lower, upper)
    calibration_matrix = np.clip(
        calibration.features[:, feature_indices],
        lower,
        upper,
    )
    train_weights = _condition_weights(train.condition_ids)
    calibration_weights = _condition_weights(calibration.condition_ids)
    train_offset = _structural_logit(train)
    calibration_offset = _structural_logit(calibration)
    train_set = lgb.Dataset(
        train_matrix,
        label=train.labels,
        weight=train_weights,
        init_score=train_offset,
        free_raw_data=False,
    )
    calibration_set = lgb.Dataset(
        calibration_matrix,
        label=calibration.labels,
        weight=calibration_weights,
        init_score=calibration_offset,
        reference=train_set,
        free_raw_data=False,
    )
    parameters: dict[str, object] = {
        **dict(backend_parameters),
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "max_depth": int(configuration["max_depth"]),
        "num_leaves": int(configuration["num_leaves"]),
        "min_data_in_leaf": int(configuration["min_data_in_leaf"]),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.1,
        "max_bin": 127,
        "seed": int(seed),
        "verbosity": -1,
    }
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=512,
        valid_sets=[calibration_set],
        valid_names=["tune_calibration"],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = max(1, int(booster.best_iteration))
    model_string = booster.model_to_string(num_iteration=best_iteration)
    reloaded = lgb.Booster(model_str=model_string)
    original = np.asarray(
        booster.predict(
            calibration_matrix,
            num_iteration=best_iteration,
            raw_score=True,
        ),
        dtype=np.float64,
    )
    restored = np.asarray(
        reloaded.predict(
            calibration_matrix,
            raw_score=True,
        ),
        dtype=np.float64,
    )
    if float(np.max(np.abs(original - restored), initial=0.0)) > 1e-10:
        raise RuntimeError("Round 17 LightGBM serialization identity failed")
    return {
        "family": "lightgbm_residual",
        "configuration": dict(configuration),
        "feature_indices": list(feature_indices),
        "feature_names": [
            POLYMARKET_ROUND17_FEATURE_NAMES[index] for index in feature_indices
        ],
        "feature_lower": lower.tolist(),
        "feature_upper": upper.tolist(),
        "best_iteration": best_iteration,
        "model_string": model_string,
        "lightgbm_version": str(lgb.__version__),
        "backend_kind": backend_kind,
        "backend_device": backend_device,
    }


def _raw_model_prediction(
    model: Mapping[str, object],
    panel: Round17DevelopmentPanel,
) -> np.ndarray:
    return _raw_model_prediction_from_features(model, panel.features)


def _raw_model_prediction_from_features(
    model: Mapping[str, object],
    features: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(features, dtype=np.float64)
    if (
        selected.ndim != 2
        or selected.shape[1] != len(POLYMARKET_ROUND17_FEATURE_NAMES)
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 17 inference features are invalid")
    family = str(model.get("family") or "")
    if family == "training_prevalence":
        return np.full(
            len(selected),
            float(model["probability_up"]),
            dtype=np.float64,
        )
    if family == "structural_control":
        return _probability(expit(_structural_logit_from_features(selected)))
    if family == "market_prior_control":
        index = POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up")
        return _probability(selected[:, index])
    matrix, _indices = _model_matrix_from_features(model, selected)
    if family == "logistic_residual":
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coefficient = np.asarray(model["coefficient"], dtype=np.float64)
        matrix = (matrix - mean) / scale
        linear = (
            _structural_logit_from_features(selected)
            + float(model["intercept"])
            + matrix @ coefficient
        )
        return _probability(expit(linear))
    if family == "lightgbm_residual":
        booster = lgb.Booster(model_str=str(model["model_string"]))
        residual = np.asarray(
            booster.predict(
                matrix,
                raw_score=True,
            ),
            dtype=np.float64,
        )
        return _probability(expit(_structural_logit_from_features(selected) + residual))
    raise ValueError("Round 17 candidate family is invalid")


def predict_round17_candidate(
    candidate: Mapping[str, object],
    panel: Round17DevelopmentPanel,
) -> np.ndarray:
    selected = panel.validate()
    raw = _raw_model_prediction(candidate, selected)
    calibration = candidate.get("calibration")
    if calibration is None:
        return raw
    if not isinstance(calibration, Mapping):
        raise ValueError("Round 17 candidate calibration is invalid")
    return _apply_platt(raw, calibration)


def predict_round17_feature_rows(
    candidate: Mapping[str, object],
    rows: Sequence[PolymarketRound17FeatureRow],
) -> np.ndarray:
    """Predict exact hash-validated feature rows without accepting target data."""

    selected = tuple(rows)
    if not selected or any(
        not isinstance(row, PolymarketRound17FeatureRow) for row in selected
    ):
        raise TypeError("Round 17 inference rows differ")
    features = np.asarray([row.values for row in selected], dtype=np.float64)
    raw = _raw_model_prediction_from_features(candidate, features)
    calibration = candidate.get("calibration")
    if calibration is None:
        return raw
    if not isinstance(calibration, Mapping):
        raise ValueError("Round 17 candidate calibration is invalid")
    return _apply_platt(raw, calibration)


def _panel_boundaries(
    train: Round17DevelopmentPanel,
    calibration: Round17DevelopmentPanel,
    selection: Round17DevelopmentPanel,
) -> dict[str, object]:
    panels = (train.validate(), calibration.validate(), selection.validate())
    if tuple(panel.role for panel in panels) != _MODEL_ROLES:
        raise ValueError("Round 17 development panel roles differ")
    if len({panel.dataset_sha256 for panel in panels}) != 1:
        raise ValueError("Round 17 development dataset identities differ")
    if len({panel.target_manifest_sha256 for panel in panels}) != 1:
        raise ValueError("Round 17 development target identities differ")
    condition_sets = [set(panel.condition_ids.tolist()) for panel in panels]
    if any(
        left & right
        for index, left in enumerate(condition_sets)
        for right in condition_sets[index + 1 :]
    ):
        raise ValueError("Round 17 development conditions overlap")
    train_end = int(np.max(train.event_start_ms)) + 300_000
    calibration_start = int(np.min(calibration.event_start_ms))
    calibration_end = int(np.max(calibration.event_start_ms)) + 300_000
    selection_start = int(np.min(selection.event_start_ms))
    if (
        calibration_start - train_end < _EMBARGO_MS
        or selection_start - calibration_end < _EMBARGO_MS
    ):
        raise ValueError("Round 17 development embargo is too short")
    return {
        "dataset_sha256": train.dataset_sha256,
        "target_manifest_sha256": train.target_manifest_sha256,
        "cohort_plan_sha256": train.cohort_plan_sha256,
        "roles": {
            panel.role: {
                "row_count": len(panel.labels),
                "condition_count": len(np.unique(panel.condition_ids)),
                "first_event_start_ms": int(np.min(panel.event_start_ms)),
                "last_event_start_ms": int(np.max(panel.event_start_ms)),
                "condition_ids": sorted(
                    str(value) for value in np.unique(panel.condition_ids)
                ),
                "condition_ids_sha256": _canonical_sha256(
                    sorted(str(value) for value in np.unique(panel.condition_ids))
                ),
            }
            for panel in panels
        },
        "train_tune_calibration_embargo_ms": calibration_start - train_end,
        "tune_internal_embargo_ms": selection_start - calibration_end,
        "test_role_accessed": False,
    }


def _valid_pretest_boundaries(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    dataset_sha256 = str(value.get("dataset_sha256") or "")
    target_manifest_sha256 = str(value.get("target_manifest_sha256") or "")
    roles = value.get("roles")
    if (
        len(dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha256)
        or len(target_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in target_manifest_sha256
        )
        or value.get("cohort_plan_sha256") != _COHORT_PLAN_SHA256
        or not isinstance(roles, Mapping)
        or tuple(roles) != _MODEL_ROLES
        or value.get("test_role_accessed") is not False
    ):
        return False
    condition_sets: list[set[str]] = []
    boundaries: list[tuple[int, int]] = []
    for role in _MODEL_ROLES:
        selected = roles.get(role)
        if not isinstance(selected, Mapping):
            return False
        condition_ids = selected.get("condition_ids")
        if (
            not isinstance(condition_ids, list)
            or not condition_ids
            or condition_ids != sorted(condition_ids)
            or len(condition_ids) != len(set(condition_ids))
            or any(
                not isinstance(condition_id, str)
                or _CONDITION_ID.fullmatch(condition_id) is None
                for condition_id in condition_ids
            )
            or selected.get("condition_count") != len(condition_ids)
            or selected.get("condition_ids_sha256") != _canonical_sha256(condition_ids)
        ):
            return False
        row_count = selected.get("row_count")
        first = selected.get("first_event_start_ms")
        last = selected.get("last_event_start_ms")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < len(condition_ids)
            or isinstance(first, bool)
            or not isinstance(first, int)
            or isinstance(last, bool)
            or not isinstance(last, int)
            or first <= 0
            or first % 300_000
            or last < first
            or last % 300_000
        ):
            return False
        condition_sets.append(set(condition_ids))
        boundaries.append((first, last))
    if any(
        left & right
        for index, left in enumerate(condition_sets)
        for right in condition_sets[index + 1 :]
    ):
        return False
    train_calibration_embargo = boundaries[1][0] - (boundaries[0][1] + 300_000)
    calibration_selection_embargo = boundaries[2][0] - (boundaries[1][1] + 300_000)
    return (
        value.get("train_tune_calibration_embargo_ms") == train_calibration_embargo
        and value.get("tune_internal_embargo_ms") == calibration_selection_embargo
        and train_calibration_embargo >= _EMBARGO_MS
        and calibration_selection_embargo >= _EMBARGO_MS
    )


def _control_models(
    train: Round17DevelopmentPanel,
) -> tuple[dict[str, object], ...]:
    weights = _condition_weights(train.condition_ids)
    prevalence = float(np.sum(weights * train.labels))
    market_prior = {
        "family": "market_prior_control",
        "candidate_id": "round17-market-prior-control",
    }
    market_raw = _raw_model_prediction(market_prior, train)
    market_prior["calibration"] = _fit_platt(
        train.labels,
        market_raw,
        train.condition_ids,
    )
    return (
        {
            "family": "training_prevalence",
            "candidate_id": "round17-training-prevalence-control",
            "probability_up": prevalence,
        },
        {
            "family": "structural_control",
            "candidate_id": "round17-chainlink-structural-control",
        },
        market_prior,
    )


def _candidate_record(
    model: dict[str, object],
    calibration_panel: Round17DevelopmentPanel,
    selection_panel: Round17DevelopmentPanel,
    *,
    candidate_id: str,
    layer: str,
    complexity_rank: int,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_calibration = _raw_model_prediction(model, calibration_panel)
    calibration = _fit_platt(
        calibration_panel.labels,
        raw_calibration,
        calibration_panel.condition_ids,
    )
    model["candidate_id"] = candidate_id
    model["layer"] = layer
    model["calibration"] = calibration
    calibration_prediction = predict_round17_candidate(model, calibration_panel)
    selection_prediction = predict_round17_candidate(model, selection_panel)
    metrics = {
        "candidate_id": candidate_id,
        "family": model["family"],
        "layer": layer,
        "complexity_rank": complexity_rank,
        "calibration": _condition_metrics(
            calibration_panel,
            calibration_prediction,
        ),
        "selection": _condition_metrics(
            selection_panel,
            selection_prediction,
        ),
    }
    return model, metrics


def fit_round17_development_pretest(
    train: Round17DevelopmentPanel,
    tune_calibration: Round17DevelopmentPanel,
    tune_selection: Round17DevelopmentPanel,
    *,
    compute_backend: str = "auto",
    seed: int = 17_017,
) -> dict[str, object]:
    """Fit bounded candidates without accepting or loading a test panel."""

    boundaries = _panel_boundaries(train, tune_calibration, tune_selection)
    controls = _control_models(train)
    control_records: list[dict[str, object]] = []
    control_predictions: dict[str, np.ndarray] = {}
    for control in controls:
        candidate_id = str(control["candidate_id"])
        prediction = predict_round17_candidate(control, tune_selection)
        control_predictions[candidate_id] = prediction
        control_records.append(
            {
                "candidate_id": candidate_id,
                "family": control["family"],
                "selection": _condition_metrics(tune_selection, prediction),
                "model": control,
            }
        )
    strongest_control = min(
        control_records,
        key=lambda item: float(
            item["selection"]["condition_balanced_log_loss"]  # type: ignore[index]
        ),
    )
    strongest_control_id = str(strongest_control["candidate_id"])
    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
        pin_opencl_device=True,
    )

    fitted: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for layer, feature_indices in _feature_layers().items():
        for l2 in _LOGISTIC_L2_GRID:
            model = _fit_logistic_residual(
                train,
                feature_indices,
                l2=l2,
            )
            candidate, record = _candidate_record(
                model,
                tune_calibration,
                tune_selection,
                candidate_id=f"round17-logistic-{layer}-l2-{l2:g}",
                layer=layer,
                complexity_rank=1,
            )
            fitted.append(candidate)
            records.append(record)
        for configuration in _LIGHTGBM_GRID:
            model = _fit_lightgbm_residual(
                train,
                tune_calibration,
                feature_indices,
                configuration=configuration,
                backend_parameters=backend_parameters,
                backend_kind=backend_kind,
                backend_device=backend_device,
                seed=seed,
            )
            candidate, record = _candidate_record(
                model,
                tune_calibration,
                tune_selection,
                candidate_id=(
                    f"round17-lightgbm-{layer}-depth-{configuration['max_depth']}"
                ),
                layer=layer,
                complexity_rank=2,
            )
            fitted.append(candidate)
            records.append(record)

    best_record = min(
        records,
        key=lambda item: float(
            item["selection"]["condition_balanced_log_loss"]  # type: ignore[index]
        ),
    )
    best_loss = float(
        best_record["selection"]["condition_balanced_log_loss"]  # type: ignore[index]
    )
    best_standard_error = float(
        best_record["selection"]["condition_log_loss_standard_error"]  # type: ignore[index]
    )
    eligible = [
        record
        for record in records
        if float(
            record["selection"]["condition_balanced_log_loss"]  # type: ignore[index]
        )
        <= best_loss + best_standard_error
    ]
    selected_record = min(
        eligible,
        key=lambda item: (
            int(item["complexity_rank"]),
            float(
                item["selection"]["condition_balanced_log_loss"]  # type: ignore[index]
            ),
            str(item["candidate_id"]),
        ),
    )
    selected_id = str(selected_record["candidate_id"])
    selected_model = next(
        model for model in fitted if model["candidate_id"] == selected_id
    )
    selected_prediction = predict_round17_candidate(
        selected_model,
        tune_selection,
    )
    control_prediction = control_predictions[strongest_control_id]
    selected_losses, selected_conditions = _condition_losses(
        tune_selection,
        selected_prediction,
    )
    control_losses, control_conditions = _condition_losses(
        tune_selection,
        control_prediction,
    )
    if selected_conditions != control_conditions:
        raise RuntimeError("Round 17 paired condition identities differ")
    improvement = control_losses - selected_losses
    improvement_mean = float(np.mean(improvement))
    improvement_standard_error = (
        0.0
        if len(improvement) < 2
        else float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
    )
    lower_95 = improvement_mean - 1.96 * improvement_standard_error
    development_gates = {
        "selected_log_loss_below_strongest_control": (
            float(
                selected_record["selection"][  # type: ignore[index]
                    "condition_balanced_log_loss"
                ]
            )
            < float(
                strongest_control["selection"][  # type: ignore[index]
                    "condition_balanced_log_loss"
                ]
            )
        ),
        "paired_condition_log_loss_improvement_lower_95_positive": lower_95 > 0.0,
    }
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND17_PRETEST_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
        "dataset_and_partition": boundaries,
        "controls": control_records,
        "candidate_ledger": records,
        "selection_rule": {
            "primary_metric": "condition_equal_weighted_tune_selection_log_loss",
            "one_standard_error_reference_candidate": best_record["candidate_id"],
            "one_standard_error": best_standard_error,
            "prefer_simplest_within_one_standard_error": True,
        },
        "selected_candidate": selected_model,
        "selected_candidate_id": selected_id,
        "strongest_control_id": strongest_control_id,
        "paired_condition_log_loss_improvement": {
            "mean": improvement_mean,
            "standard_error": improvement_standard_error,
            "lower_95_normal_approximation": lower_95,
            "condition_count": len(improvement),
        },
        "development_gates": development_gates,
        "development_accepted": all(development_gates.values()),
        "compute": {
            "requested": compute_backend,
            "lightgbm_backend_kind": backend_kind,
            "lightgbm_backend_device": backend_device,
        },
        "causal_tcn_status": (
            "blocked_until_a_simpler_candidate_passes_development_stability_gates"
        ),
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "execution_simulation_completed": False,
        "profitability_claim": False,
        "live_trading_authority": False,
    }
    payload["pretest_sha256"] = _canonical_sha256(payload)
    return payload


def validate_round17_pretest_artifact(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("pretest_sha256", "")).strip().lower()
    candidate_ledger = payload.get("candidate_ledger")
    selected_candidate = payload.get("selected_candidate")
    controls = payload.get("controls")
    development_gates = payload.get("development_gates")
    candidate_ids = (
        []
        if not isinstance(candidate_ledger, list)
        else [
            str(item.get("candidate_id") or "")
            for item in candidate_ledger
            if isinstance(item, Mapping)
        ]
    )
    families = (
        []
        if not isinstance(candidate_ledger, list)
        else [
            str(item.get("family") or "")
            for item in candidate_ledger
            if isinstance(item, Mapping)
        ]
    )
    layers = (
        []
        if not isinstance(candidate_ledger, list)
        else [
            str(item.get("layer") or "")
            for item in candidate_ledger
            if isinstance(item, Mapping)
        ]
    )
    selected_indices = (
        selected_candidate.get("feature_indices", [])
        if isinstance(selected_candidate, Mapping)
        else []
    )
    selected_names = (
        selected_candidate.get("feature_names", [])
        if isinstance(selected_candidate, Mapping)
        else []
    )
    selected_lower = (
        selected_candidate.get("feature_lower", [])
        if isinstance(selected_candidate, Mapping)
        else []
    )
    selected_upper = (
        selected_candidate.get("feature_upper", [])
        if isinstance(selected_candidate, Mapping)
        else []
    )
    if (
        claimed != _canonical_sha256(payload)
        or payload.get("schema_version") != POLYMARKET_ROUND17_PRETEST_SCHEMA_VERSION
        or payload.get("contract_sha256") != POLYMARKET_ROUND17_CONTRACT_SHA256
        or payload.get("feature_names_sha256")
        != POLYMARKET_ROUND17_FEATURE_NAMES_SHA256
        or not _valid_pretest_boundaries(payload.get("dataset_and_partition"))
        or not isinstance(candidate_ledger, list)
        or len(candidate_ledger) != 15
        or len(candidate_ids) != 15
        or len(set(candidate_ids)) != 15
        or families.count("logistic_residual") != 9
        or families.count("lightgbm_residual") != 6
        or any(layers.count(layer) != 5 for layer in _feature_layers())
        or not isinstance(selected_candidate, Mapping)
        or selected_candidate.get("candidate_id")
        != payload.get("selected_candidate_id")
        or payload.get("selected_candidate_id") not in candidate_ids
        or not isinstance(selected_indices, list)
        or not isinstance(selected_names, list)
        or not isinstance(selected_lower, list)
        or not isinstance(selected_upper, list)
        or not (
            len(selected_indices)
            == len(selected_names)
            == len(selected_lower)
            == len(selected_upper)
            > 0
        )
        or any(
            int(index) < 0
            or int(index) >= len(POLYMARKET_ROUND17_FEATURE_NAMES)
            or POLYMARKET_ROUND17_FEATURE_NAMES[int(index)] != str(name)
            for index, name in zip(selected_indices, selected_names, strict=True)
        )
        or not isinstance(controls, list)
        or len(controls) != 3
        or not isinstance(development_gates, Mapping)
        or set(development_gates)
        != {
            "paired_condition_log_loss_improvement_lower_95_positive",
            "selected_log_loss_below_strongest_control",
        }
        or any(type(result) is not bool for result in development_gates.values())
        or payload.get("development_accepted")
        is not all(bool(result) for result in development_gates.values())
        or payload.get("causal_tcn_status")
        != "blocked_until_a_simpler_candidate_passes_development_stability_gates"
        or payload.get("test_features_accessed") is not False
        or payload.get("test_targets_accessed") is not False
        or payload.get("execution_simulation_completed") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("live_trading_authority") is not False
    ):
        raise ValueError("Round 17 pretest artifact integrity differs")
    return {**payload, "pretest_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND17_PRETEST_SCHEMA_VERSION",
    "Round17DevelopmentPanel",
    "fit_round17_development_pretest",
    "predict_round17_candidate",
    "predict_round17_feature_rows",
    "validate_round17_pretest_artifact",
]
