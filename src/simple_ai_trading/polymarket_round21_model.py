"""Matched residual-probability candidates for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .lightgbm_backend import lightgbm_backend_parameters
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256


POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION = (
    "polymarket-round21-matched-residual-development-v1"
)
POLYMARKET_ROUND21_MODEL_SEED = 21_021
POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS = 30
POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES = 2_000
POLYMARKET_ROUND21_DATASET_DESIGN_SHA256 = (
    "089f046fd611e32950381ec4f33a1e6b54a0a0a4d6be161de0643ade94590eba"
)
_ROLES = ("train", "tune_calibration", "tune_selection", "test")
_LAYERS = ("core", "core_spot", "core_spot_usdm")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROBABILITY_FLOOR = 1e-6
_LOGISTIC_L2 = (0.01, 0.1, 1.0)
_LIGHTGBM_GRID = (
    {"max_depth": 2, "num_leaves": 3, "min_data_in_leaf": 20},
    {"max_depth": 3, "num_leaves": 7, "min_data_in_leaf": 40},
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
    if selected.ndim != 1 or not np.all(np.isfinite(selected)):
        raise ValueError("Round 21 probability vector is invalid")
    return np.clip(selected, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)


def _float_matrix(value: object, *, rows: int) -> np.ndarray:
    selected = np.asarray(value)
    if (
        selected.ndim != 2
        or selected.shape[0] != rows
        or selected.shape[1] < 1
        or selected.dtype.kind != "f"
        or selected.dtype.itemsize not in (4, 8)
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 21 feature matrix is invalid")
    return np.asarray(selected, dtype=np.float32, order="C")


def _condition_groups(
    condition_ids: np.ndarray,
) -> tuple[tuple[str, int, int], ...]:
    if condition_ids.ndim != 1 or len(condition_ids) < 1:
        raise ValueError("Round 21 condition identities are invalid")
    boundaries = np.flatnonzero(condition_ids[1:] != condition_ids[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(condition_ids)])))
    groups = tuple(
        (str(condition_ids[start]), int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    )
    if len({condition for condition, _start, _end in groups}) != len(groups):
        raise ValueError("Round 21 condition identities are not contiguous")
    return groups


@dataclass(frozen=True, slots=True)
class Round21DevelopmentPanel:
    role: str
    condition_ids: np.ndarray
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    labels: np.ndarray
    structural_probability: np.ndarray
    market_prior_probability: np.ndarray
    core_features: np.ndarray
    spot_features: np.ndarray
    usdm_features: np.ndarray
    spot_available: np.ndarray
    usdm_available: np.ndarray
    core_feature_names_sha256: str
    spot_feature_names_sha256: str
    usdm_feature_names_sha256: str
    dataset_sha256: str
    target_manifest_sha256: str
    dataset_design_sha256: str

    def validate(self) -> Round21DevelopmentPanel:
        role = str(self.role or "").strip()
        condition_ids = np.asarray(self.condition_ids, dtype=object)
        event_starts = np.asarray(self.event_start_ms, dtype=np.int64)
        decisions = np.asarray(self.decision_time_ms, dtype=np.int64)
        labels = np.asarray(self.labels)
        structural = np.asarray(self.structural_probability)
        market_prior = np.asarray(self.market_prior_probability)
        rows = len(condition_ids)
        spot_available = np.asarray(self.spot_available)
        usdm_available = np.asarray(self.usdm_available)
        if (
            role not in _ROLES
            or rows < 1
            or event_starts.shape != (rows,)
            or decisions.shape != (rows,)
            or labels.shape != (rows,)
            or labels.dtype.kind != "f"
            or structural.shape != (rows,)
            or structural.dtype.kind != "f"
            or market_prior.shape != (rows,)
            or market_prior.dtype.kind != "f"
            or spot_available.shape != (rows,)
            or spot_available.dtype.kind != "b"
            or usdm_available.shape != (rows,)
            or usdm_available.dtype.kind != "b"
            or any(
                _SHA256.fullmatch(str(value or "").strip()) is None
                for value in (
                    self.core_feature_names_sha256,
                    self.spot_feature_names_sha256,
                    self.usdm_feature_names_sha256,
                    self.dataset_sha256,
                    self.target_manifest_sha256,
                    self.dataset_design_sha256,
                )
            )
            or self.dataset_design_sha256
            != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        ):
            raise ValueError("Round 21 development panel is invalid")
        core = _float_matrix(self.core_features, rows=rows)
        spot = _float_matrix(self.spot_features, rows=rows)
        usdm = _float_matrix(self.usdm_features, rows=rows)
        labels64 = np.asarray(labels, dtype=np.float64)
        structural64 = _probability(np.asarray(structural, dtype=np.float64))
        market_prior64 = _probability(np.asarray(market_prior, dtype=np.float64))
        if (
            not np.all(np.isin(labels64, (0.0, 1.0)))
            or set(labels64.tolist()) != {0.0, 1.0}
            or np.any(np.asarray(structural, dtype=np.float64) <= 0.0)
            or np.any(np.asarray(structural, dtype=np.float64) >= 1.0)
            or np.any(np.asarray(market_prior, dtype=np.float64) <= 0.0)
            or np.any(np.asarray(market_prior, dtype=np.float64) >= 1.0)
            or np.any(event_starts <= 0)
            or np.any(event_starts % 300_000)
            or np.any(decisions < event_starts)
            or np.any(decisions >= event_starts + 300_000)
            or np.any((decisions - event_starts) % 250)
            or np.any(event_starts[1:] < event_starts[:-1])
            or np.any(
                (event_starts[1:] == event_starts[:-1])
                & (decisions[1:] < decisions[:-1])
            )
            or np.any(spot[~spot_available] != 0.0)
            or np.any(usdm[~usdm_available] != 0.0)
            or np.any(usdm_available & ~spot_available)
        ):
            raise ValueError("Round 21 development panel is invalid")
        for condition, start, end in _condition_groups(condition_ids):
            if (
                _CONDITION_ID.fullmatch(condition) is None
                or event_starts[start] != event_starts[end - 1]
                or not np.all(labels64[start:end] == labels64[start])
            ):
                raise ValueError("Round 21 condition target identity differs")
        return replace(
            self,
            role=role,
            condition_ids=condition_ids,
            event_start_ms=event_starts,
            decision_time_ms=decisions,
            labels=labels64,
            structural_probability=structural64,
            market_prior_probability=market_prior64,
            core_features=core,
            spot_features=spot,
            usdm_features=usdm,
            spot_available=np.asarray(spot_available, dtype=np.bool_),
            usdm_available=np.asarray(usdm_available, dtype=np.bool_),
        )


def _layer_mask(panel: Round21DevelopmentPanel, layer: str) -> np.ndarray:
    if layer == "core":
        return np.ones(len(panel.labels), dtype=np.bool_)
    if layer == "core_spot":
        return panel.spot_available.copy()
    if layer == "core_spot_usdm":
        return panel.spot_available & panel.usdm_available
    raise ValueError("Round 21 feature layer is invalid")


def _layer_matrix(panel: Round21DevelopmentPanel, layer: str) -> np.ndarray:
    if layer == "core":
        values = panel.core_features
    elif layer == "core_spot":
        values = np.concatenate(
            (panel.core_features, panel.spot_features),
            axis=1,
        )
    elif layer == "core_spot_usdm":
        values = np.concatenate(
            (panel.core_features, panel.spot_features, panel.usdm_features),
            axis=1,
        )
    else:
        raise ValueError("Round 21 feature layer is invalid")
    return np.asarray(values, dtype=np.float32, order="C")


def _selected_indices(
    panel: Round21DevelopmentPanel,
    layer: str,
) -> np.ndarray:
    return np.flatnonzero(_layer_mask(panel, layer))


def _condition_count(condition_ids: np.ndarray) -> int:
    return len(_condition_groups(np.asarray(condition_ids, dtype=object)))


def _condition_weights(condition_ids: np.ndarray) -> np.ndarray:
    groups = _condition_groups(np.asarray(condition_ids, dtype=object))
    weights = np.empty(len(condition_ids), dtype=np.float64)
    for _condition, start, end in groups:
        weights[start:end] = 1.0 / len(groups) / (end - start)
    return weights


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    threshold = float(probability) * float(cumulative[-1])
    index = min(
        len(ordered_values) - 1,
        int(np.searchsorted(cumulative, threshold, side="left")),
    )
    return float(ordered_values[index])


def _fit_transform(
    matrix: np.ndarray,
    weights: np.ndarray,
) -> dict[str, list[float]]:
    lower = np.empty(matrix.shape[1], dtype=np.float64)
    upper = np.empty(matrix.shape[1], dtype=np.float64)
    mean = np.empty(matrix.shape[1], dtype=np.float64)
    scale = np.empty(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        column = np.asarray(matrix[:, index], dtype=np.float64)
        lower[index] = _weighted_quantile(column, weights, 0.001)
        upper[index] = _weighted_quantile(column, weights, 0.999)
        clipped = np.clip(column, lower[index], upper[index])
        mean[index] = float(np.sum(weights * clipped))
        variance = float(np.sum(weights * np.square(clipped - mean[index])))
        scale[index] = math.sqrt(max(variance, 1e-12))
    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _transform_matrix(
    matrix: np.ndarray,
    transform: Mapping[str, object],
    *,
    normalize: bool,
) -> np.ndarray:
    lower = np.asarray(transform["lower"], dtype=np.float32)
    upper = np.asarray(transform["upper"], dtype=np.float32)
    if (
        lower.shape != (matrix.shape[1],)
        or upper.shape != (matrix.shape[1],)
        or np.any(lower > upper)
    ):
        raise ValueError("Round 21 feature transform is invalid")
    output = np.asarray(matrix, dtype=np.float32, order="C").copy()
    np.clip(output, lower, upper, out=output)
    if normalize:
        mean = np.asarray(transform["mean"], dtype=np.float32)
        scale = np.asarray(transform["scale"], dtype=np.float32)
        if (
            mean.shape != lower.shape
            or scale.shape != lower.shape
            or np.any(scale <= 0.0)
        ):
            raise ValueError("Round 21 feature transform is invalid")
        np.subtract(output, mean, out=output)
        np.divide(output, scale, out=output)
    return output


def _fit_logistic_residual(
    matrix: np.ndarray,
    labels: np.ndarray,
    structural_probability: np.ndarray,
    weights: np.ndarray,
    transform: Mapping[str, object],
    *,
    l2: float,
    layer: str,
) -> dict[str, object]:
    normalized = _transform_matrix(matrix, transform, normalize=True)
    offset = logit(_probability(structural_probability))
    target = np.asarray(labels, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        coefficient = parameters[1:]
        linear = offset + intercept + normalized @ coefficient
        residual = expit(linear) - target
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
            + 0.5 * l2 * float(coefficient @ coefficient)
        )
        weighted_residual = weights * residual
        gradient = np.concatenate(
            (
                np.asarray([np.sum(weighted_residual)]),
                np.asarray(normalized.T @ weighted_residual)
                + l2 * coefficient,
            )
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 512, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Round 21 residual logistic fit failed: {str(result.message).strip()}"
        )
    return {
        "candidate_id": f"{layer}-logistic-l2-{format(l2, 'g')}",
        "family": "logistic_residual",
        "layer": layer,
        "l2": float(l2),
        "transform": dict(transform),
        "intercept": float(result.x[0]),
        "coefficient": result.x[1:].tolist(),
    }


def _fit_lightgbm_residual(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_structural: np.ndarray,
    train_weights: np.ndarray,
    stop_matrix: np.ndarray,
    stop_labels: np.ndarray,
    stop_structural: np.ndarray,
    stop_weights: np.ndarray,
    transform: Mapping[str, object],
    configuration: Mapping[str, int],
    backend_parameters: Mapping[str, object],
    *,
    backend_kind: str,
    backend_device: str,
    layer: str,
) -> dict[str, object]:
    train_values = _transform_matrix(train_matrix, transform, normalize=False)
    stop_values = _transform_matrix(stop_matrix, transform, normalize=False)
    train_set = lgb.Dataset(
        train_values,
        label=train_labels,
        weight=train_weights,
        init_score=logit(_probability(train_structural)),
        free_raw_data=False,
    )
    stop_set = lgb.Dataset(
        stop_values,
        label=stop_labels,
        weight=stop_weights,
        init_score=logit(_probability(stop_structural)),
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
        "seed": POLYMARKET_ROUND21_MODEL_SEED,
        "verbosity": -1,
    }
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=512,
        valid_sets=[stop_set],
        valid_names=["tune_early_stop"],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = max(1, int(booster.best_iteration))
    model_string = booster.model_to_string(num_iteration=best_iteration)
    reloaded = lgb.Booster(model_str=model_string)
    original = np.asarray(
        booster.predict(stop_values, num_iteration=best_iteration, raw_score=True),
        dtype=np.float64,
    )
    restored = np.asarray(
        reloaded.predict(stop_values, raw_score=True),
        dtype=np.float64,
    )
    if float(np.max(np.abs(original - restored), initial=0.0)) > 1e-10:
        raise RuntimeError("Round 21 LightGBM serialization identity failed")
    label = (
        f"d{configuration['max_depth']}-l{configuration['num_leaves']}-"
        f"m{configuration['min_data_in_leaf']}"
    )
    return {
        "candidate_id": f"{layer}-lightgbm-{label}",
        "family": "lightgbm_residual",
        "layer": layer,
        "configuration": dict(configuration),
        "transform": dict(transform),
        "best_iteration": best_iteration,
        "model_string": model_string,
        "lightgbm_version": str(lgb.__version__),
        "backend_kind": backend_kind,
        "backend_device": backend_device,
    }


def _raw_prediction(
    model: Mapping[str, object],
    matrix: np.ndarray,
    structural_probability: np.ndarray,
) -> np.ndarray:
    family = str(model.get("family") or "")
    offset = logit(_probability(structural_probability))
    transform = model.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Round 21 model transform is unavailable")
    if family == "logistic_residual":
        values = _transform_matrix(matrix, transform, normalize=True)
        coefficient = np.asarray(model["coefficient"], dtype=np.float32)
        if coefficient.shape != (values.shape[1],):
            raise ValueError("Round 21 logistic coefficient shape differs")
        linear = (
            offset
            + float(model["intercept"])
            + np.asarray(values @ coefficient, dtype=np.float64)
        )
    elif family == "lightgbm_residual":
        values = _transform_matrix(matrix, transform, normalize=False)
        booster = lgb.Booster(model_str=str(model["model_string"]))
        linear = offset + np.asarray(
            booster.predict(values, raw_score=True),
            dtype=np.float64,
        )
    else:
        raise ValueError("Round 21 model family is invalid")
    return _probability(expit(linear))


def _fit_platt(
    labels: np.ndarray,
    predictions: np.ndarray,
    condition_ids: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.float64)
    raw = logit(_probability(predictions))
    weights = _condition_weights(condition_ids)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = float(parameters[0]) + float(parameters[1]) * raw
        residual = expit(linear) - target
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
        )
        gradient = np.asarray(
            [
                np.sum(weights * residual),
                np.sum(weights * residual * raw),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((-20.0, 20.0), (0.0, 20.0)),
        options={"maxiter": 256, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Round 21 Platt calibration failed")
    return {"intercept": float(result.x[0]), "slope": float(result.x[1])}


def _apply_platt(
    predictions: np.ndarray,
    calibration: Mapping[str, object],
) -> np.ndarray:
    intercept = float(calibration["intercept"])
    slope = float(calibration["slope"])
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope < 0.0:
        raise ValueError("Round 21 calibration is invalid")
    return _probability(
        expit(intercept + slope * logit(_probability(predictions)))
    )


def predict_round21_candidate(
    model: Mapping[str, object],
    panel: Round21DevelopmentPanel,
) -> tuple[np.ndarray, np.ndarray]:
    selected = panel.validate()
    layer = str(model.get("layer") or "")
    indices = _selected_indices(selected, layer)
    matrix = _layer_matrix(selected, layer)[indices]
    raw = _raw_prediction(
        model,
        matrix,
        selected.structural_probability[indices],
    )
    calibration = model.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Round 21 model calibration is unavailable")
    return indices, _apply_platt(raw, calibration)


def _condition_losses(
    condition_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    metric: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    groups = _condition_groups(condition_ids)
    losses: list[float] = []
    conditions: list[str] = []
    probability = _probability(predictions)
    for condition, start, end in groups:
        target = labels[start:end]
        selected = probability[start:end]
        if metric == "log_loss":
            value = -np.mean(
                target * np.log(selected)
                + (1.0 - target) * np.log1p(-selected)
            )
        elif metric == "brier":
            value = np.mean(np.square(selected - target))
        else:
            raise ValueError("Round 21 metric is invalid")
        conditions.append(condition)
        losses.append(float(value))
    return np.asarray(losses, dtype=np.float64), tuple(conditions)


def _metrics(
    condition_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    losses, conditions = _condition_losses(
        condition_ids,
        labels,
        predictions,
        metric="log_loss",
    )
    brier, _ = _condition_losses(
        condition_ids,
        labels,
        predictions,
        metric="brier",
    )
    return {
        "condition_count": len(conditions),
        "condition_equal_log_loss": float(np.mean(losses)),
        "condition_equal_brier_score": float(np.mean(brier)),
        "log_loss_standard_error": (
            0.0
            if len(losses) < 2
            else float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
        ),
    }


def _paired_improvement(
    condition_ids: np.ndarray,
    labels: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    metric: str,
    seed_offset: int,
) -> dict[str, float | int]:
    control_loss, control_conditions = _condition_losses(
        condition_ids,
        labels,
        control,
        metric=metric,
    )
    candidate_loss, candidate_conditions = _condition_losses(
        condition_ids,
        labels,
        candidate,
        metric=metric,
    )
    if control_conditions != candidate_conditions:
        raise RuntimeError("Round 21 paired condition identities differ")
    difference = control_loss - candidate_loss
    generator = np.random.default_rng(
        POLYMARKET_ROUND21_MODEL_SEED + int(seed_offset)
    )
    samples = np.empty(POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES):
        selected = generator.integers(0, len(difference), size=len(difference))
        samples[index] = float(np.mean(difference[selected]))
    return {
        "condition_count": len(difference),
        "mean": float(np.mean(difference)),
        "lower_95": float(np.quantile(samples, 0.025, method="linear")),
        "upper_95": float(np.quantile(samples, 0.975, method="linear")),
    }


def _split_calibration_indices(
    panel: Round21DevelopmentPanel,
    layer: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = _selected_indices(panel, layer)
    selected_conditions = panel.condition_ids[indices]
    groups = _condition_groups(selected_conditions)
    if len(groups) < 2 * POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS:
        raise ValueError("Round 21 calibration layer has too few conditions")
    midpoint = len(groups) // 2
    split_row = groups[midpoint][1]
    return indices[:split_row], indices[split_row:]


def _fit_layer_candidates(
    train: Round21DevelopmentPanel,
    calibration: Round21DevelopmentPanel,
    selection: Round21DevelopmentPanel,
    layer: str,
    backend_parameters: Mapping[str, object],
    *,
    backend_kind: str,
    backend_device: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    train_indices = _selected_indices(train, layer)
    selection_indices = _selected_indices(selection, layer)
    stop_indices, platt_indices = _split_calibration_indices(calibration, layer)
    for name, panel, indices in (
        ("train", train, train_indices),
        ("tune early-stop", calibration, stop_indices),
        ("tune calibration", calibration, platt_indices),
        ("tune selection", selection, selection_indices),
    ):
        if _condition_count(panel.condition_ids[indices]) < (
            POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
        ):
            raise ValueError(f"Round 21 {layer} {name} has too few conditions")
    train_matrix = _layer_matrix(train, layer)[train_indices]
    stop_matrix = _layer_matrix(calibration, layer)[stop_indices]
    platt_matrix = _layer_matrix(calibration, layer)[platt_indices]
    selection_matrix = _layer_matrix(selection, layer)[selection_indices]
    train_weights = _condition_weights(train.condition_ids[train_indices])
    stop_weights = _condition_weights(calibration.condition_ids[stop_indices])
    transform = _fit_transform(train_matrix, train_weights)
    models: list[dict[str, object]] = []
    for l2 in _LOGISTIC_L2:
        models.append(
            _fit_logistic_residual(
                train_matrix,
                train.labels[train_indices],
                train.structural_probability[train_indices],
                train_weights,
                transform,
                l2=l2,
                layer=layer,
            )
        )
    for configuration in _LIGHTGBM_GRID:
        models.append(
            _fit_lightgbm_residual(
                train_matrix,
                train.labels[train_indices],
                train.structural_probability[train_indices],
                train_weights,
                stop_matrix,
                calibration.labels[stop_indices],
                calibration.structural_probability[stop_indices],
                stop_weights,
                transform,
                configuration,
                backend_parameters,
                backend_kind=backend_kind,
                backend_device=backend_device,
                layer=layer,
            )
        )
    records: list[dict[str, object]] = []
    for model in models:
        raw_calibration = _raw_prediction(
            model,
            platt_matrix,
            calibration.structural_probability[platt_indices],
        )
        model["calibration"] = _fit_platt(
            calibration.labels[platt_indices],
            raw_calibration,
            calibration.condition_ids[platt_indices],
        )
        prediction = _apply_platt(
            _raw_prediction(
                model,
                selection_matrix,
                selection.structural_probability[selection_indices],
            ),
            model["calibration"],  # type: ignore[arg-type]
        )
        records.append(
            {
                "candidate_id": model["candidate_id"],
                "family": model["family"],
                "layer": layer,
                "model": model,
                "selection_metrics": _metrics(
                    selection.condition_ids[selection_indices],
                    selection.labels[selection_indices],
                    prediction,
                ),
            }
        )
    return records, selection_indices


def _select_candidate(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not records:
        raise ValueError("Round 21 candidate ledger is empty")
    best = min(
        records,
        key=lambda item: float(
            item["selection_metrics"]["condition_equal_log_loss"]  # type: ignore[index]
        ),
    )
    threshold = float(
        best["selection_metrics"]["condition_equal_log_loss"]  # type: ignore[index]
    ) + float(
        best["selection_metrics"]["log_loss_standard_error"]  # type: ignore[index]
    )
    eligible = [
        item
        for item in records
        if float(
            item["selection_metrics"]["condition_equal_log_loss"]  # type: ignore[index]
        )
        <= threshold
    ]
    return min(
        eligible,
        key=lambda item: (
            0 if item["family"] == "logistic_residual" else 1,
            -float(item["model"].get("l2", 0.0)),  # type: ignore[union-attr]
            float(
                item["selection_metrics"][  # type: ignore[index]
                    "condition_equal_log_loss"
                ]
            ),
            str(item["candidate_id"]),
        ),
    )


def _control_predictions(
    train: Round21DevelopmentPanel,
    calibration: Round21DevelopmentPanel,
    selection: Round21DevelopmentPanel,
) -> list[dict[str, object]]:
    _stop_indices, platt_indices = _split_calibration_indices(
        calibration,
        "core",
    )
    controls: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "structural_probability",
            calibration.structural_probability[platt_indices],
            selection.structural_probability,
        ),
        (
            "executable_market_prior",
            calibration.market_prior_probability[platt_indices],
            selection.market_prior_probability,
        ),
    ]
    records: list[dict[str, object]] = []
    for index, (control_id, calibration_raw, selection_raw) in enumerate(controls):
        calibration_parameters = _fit_platt(
            calibration.labels[platt_indices],
            calibration_raw,
            calibration.condition_ids[platt_indices],
        )
        calibrated = _apply_platt(selection_raw, calibration_parameters)
        records.extend(
            (
                {
                    "control_id": f"{control_id}_raw",
                    "prediction": selection_raw.tolist(),
                    "metrics": _metrics(
                        selection.condition_ids,
                        selection.labels,
                        selection_raw,
                    ),
                },
                {
                    "control_id": f"{control_id}_calibrated",
                    "calibration": calibration_parameters,
                    "prediction": calibrated.tolist(),
                    "metrics": _metrics(
                        selection.condition_ids,
                        selection.labels,
                        calibrated,
                    ),
                },
            )
        )
    prevalence = float(
        np.sum(_condition_weights(train.condition_ids) * train.labels)
    )
    prevalence_prediction = np.full(
        len(selection.labels),
        prevalence,
        dtype=np.float64,
    )
    records.append(
        {
            "control_id": "training_prevalence",
            "probability_up": prevalence,
            "prediction": prevalence_prediction.tolist(),
            "metrics": _metrics(
                selection.condition_ids,
                selection.labels,
                prevalence_prediction,
            ),
        }
    )
    return records


def _dataset_identity(panel: Round21DevelopmentPanel) -> dict[str, object]:
    return {
        "role": panel.role,
        "row_count": len(panel.labels),
        "condition_count": _condition_count(panel.condition_ids),
        "first_event_start_ms": int(panel.event_start_ms[0]),
        "last_event_start_ms": int(panel.event_start_ms[-1]),
        "dataset_sha256": panel.dataset_sha256,
        "target_manifest_sha256": panel.target_manifest_sha256,
        "dataset_design_sha256": panel.dataset_design_sha256,
    }


def fit_round21_development(
    *,
    train: Round21DevelopmentPanel,
    tune_calibration: Round21DevelopmentPanel,
    tune_selection: Round21DevelopmentPanel,
    compute_backend: str = "auto",
) -> dict[str, object]:
    train = train.validate()
    tune_calibration = tune_calibration.validate()
    tune_selection = tune_selection.validate()
    if (
        train.role != "train"
        or tune_calibration.role != "tune_calibration"
        or tune_selection.role != "tune_selection"
        or not (
            train.event_start_ms[-1]
            < tune_calibration.event_start_ms[0]
            < tune_selection.event_start_ms[0]
        )
        or len(
            {
                train.core_feature_names_sha256,
                tune_calibration.core_feature_names_sha256,
                tune_selection.core_feature_names_sha256,
            }
        )
        != 1
        or len(
            {
                train.spot_feature_names_sha256,
                tune_calibration.spot_feature_names_sha256,
                tune_selection.spot_feature_names_sha256,
            }
        )
        != 1
        or len(
            {
                train.usdm_feature_names_sha256,
                tune_calibration.usdm_feature_names_sha256,
                tune_selection.usdm_feature_names_sha256,
            }
        )
        != 1
        or len(
            {
                train.core_features.shape[1],
                tune_calibration.core_features.shape[1],
                tune_selection.core_features.shape[1],
            }
        )
        != 1
        or len(
            {
                train.spot_features.shape[1],
                tune_calibration.spot_features.shape[1],
                tune_selection.spot_features.shape[1],
            }
        )
        != 1
        or len(
            {
                train.usdm_features.shape[1],
                tune_calibration.usdm_features.shape[1],
                tune_selection.usdm_features.shape[1],
            }
        )
        != 1
    ):
        raise ValueError("Round 21 development partition boundary differs")
    backend_parameters, backend_kind, backend_device = (
        lightgbm_backend_parameters(
            compute_backend,
            POLYMARKET_ROUND21_MODEL_SEED,
            reproducible=True,
            pin_opencl_device=True,
        )
    )
    controls = _control_predictions(
        train,
        tune_calibration,
        tune_selection,
    )
    strongest_control = min(
        controls,
        key=lambda item: float(
            item["metrics"]["condition_equal_log_loss"]  # type: ignore[index]
        ),
    )
    layer_results: dict[str, object] = {}
    selected_models: dict[str, Mapping[str, object]] = {}
    core_prediction: np.ndarray | None = None
    for layer_index, layer in enumerate(_LAYERS):
        records, selection_indices = _fit_layer_candidates(
            train,
            tune_calibration,
            tune_selection,
            layer,
            backend_parameters,
            backend_kind=backend_kind,
            backend_device=backend_device,
        )
        selected_record = _select_candidate(records)
        selected_model = selected_record["model"]
        if not isinstance(selected_model, Mapping):
            raise RuntimeError("Round 21 selected model is unavailable")
        indices, prediction = predict_round21_candidate(
            selected_model,
            tune_selection,
        )
        if not np.array_equal(indices, selection_indices):
            raise RuntimeError("Round 21 selected prediction population differs")
        selected_models[layer] = selected_model
        if layer == "core":
            core_prediction = prediction
            improvements = {
                str(control["control_id"]): {
                    metric: _paired_improvement(
                        tune_selection.condition_ids,
                        tune_selection.labels,
                        np.asarray(control["prediction"], dtype=np.float64),
                        prediction,
                        metric=metric,
                        seed_offset=layer_index * 100
                        + control_index * 10
                        + metric_index,
                    )
                    for metric_index, metric in enumerate(("log_loss", "brier"))
                }
                for control_index, control in enumerate(controls)
            }
            accepted = all(
                result[metric]["lower_95"] > 0.0
                for result in improvements.values()
                for metric in ("log_loss", "brier")
            )
            comparison = {
                "against_every_control": improvements,
                "predictive_development_accepted": accepted,
            }
        else:
            if core_prediction is None:
                raise RuntimeError("Round 21 core model was not selected first")
            matched_core = core_prediction[selection_indices]
            improvements = {
                metric: _paired_improvement(
                    tune_selection.condition_ids[selection_indices],
                    tune_selection.labels[selection_indices],
                    matched_core,
                    prediction,
                    metric=metric,
                    seed_offset=layer_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("log_loss", "brier"))
            }
            accepted = all(
                improvements[metric]["lower_95"] > 0.0
                for metric in ("log_loss", "brier")
            )
            comparison = {
                "matched_core_candidate_id": selected_models["core"][
                    "candidate_id"
                ],
                "matched_decision_count": len(selection_indices),
                "matched_condition_count": _condition_count(
                    tune_selection.condition_ids[selection_indices]
                ),
                "incremental_improvement": improvements,
                "predictive_development_accepted": accepted,
            }
        layer_results[layer] = {
            "candidate_ledger": records,
            "selected_candidate_id": selected_model["candidate_id"],
            "selection_indices_sha256": hashlib.sha256(
                np.asarray(selection_indices, dtype="<i8").tobytes()
            ).hexdigest(),
            "comparison": comparison,
        }
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
        "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
        "dataset_and_partition": {
            "train": _dataset_identity(train),
            "tune_calibration": _dataset_identity(tune_calibration),
            "tune_selection": _dataset_identity(tune_selection),
            "core_feature_names_sha256": train.core_feature_names_sha256,
            "spot_feature_names_sha256": train.spot_feature_names_sha256,
            "usdm_feature_names_sha256": train.usdm_feature_names_sha256,
        },
        "controls": [
            {key: value for key, value in control.items() if key != "prediction"}
            for control in controls
        ],
        "strongest_control_id": strongest_control["control_id"],
        "layers": layer_results,
        "selection_rule": {
            "primary_metric": "condition_equal_log_loss",
            "prefer_simplest_within_one_standard_error": True,
            "optional_layers_use_exact_matched_core_population": True,
        },
        "compute": {
            "requested": str(compute_backend),
            "lightgbm_backend_kind": backend_kind,
            "lightgbm_backend_device": backend_device,
        },
        "economic_evaluation_completed": False,
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "model_selected": False,
        "ai_edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_round21_development_artifact(payload)


def validate_round21_development_artifact(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("artifact_sha256", "")).strip().lower()
    layers = payload.get("layers")
    controls = payload.get("controls")
    false_fields = (
        "economic_evaluation_completed",
        "test_features_accessed",
        "test_targets_accessed",
        "model_selected",
        "ai_edge_claim",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or payload.get("dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or not isinstance(controls, list)
        or len(controls) != 5
        or not isinstance(layers, Mapping)
        or set(layers) != set(_LAYERS)
        or any(
            not isinstance(layer, Mapping)
            or not isinstance(layer.get("candidate_ledger"), list)
            or len(layer["candidate_ledger"]) != 5
            or layer.get("selected_candidate_id")
            not in {
                record.get("candidate_id")
                for record in layer["candidate_ledger"]
                if isinstance(record, Mapping)
            }
            or _SHA256.fullmatch(
                str(layer.get("selection_indices_sha256") or "")
            )
            is None
            for layer in layers.values()
        )
        or any(payload.get(field) is not False for field in false_fields)
    ):
        raise ValueError("Round 21 development artifact differs")
    return {**payload, "artifact_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND21_BOOTSTRAP_SAMPLES",
    "POLYMARKET_ROUND21_DATASET_DESIGN_SHA256",
    "POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS",
    "POLYMARKET_ROUND21_MODEL_SCHEMA_VERSION",
    "Round21DevelopmentPanel",
    "fit_round21_development",
    "predict_round21_candidate",
    "validate_round21_development_artifact",
]
