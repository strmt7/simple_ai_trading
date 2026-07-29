"""Frozen predictive screen for independent Polymarket BTC development."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .lightgbm_backend import lightgbm_backend_parameters
from .polymarket_historical_dataset import CALENDAR_FEATURE_NAMES, FEATURE_NAMES
from .polymarket_historical_screen import HistoricalScreenStore


PRETEST_SCHEMA_VERSION = "polymarket-historical-btc-pretest-v1"
EVALUATION_SCHEMA_VERSION = "polymarket-historical-btc-evaluation-v1"
MODEL_SEED = 14_074
RIDGE_L2_GRID = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_BLOCK_CONDITIONS = 12
_EPSILON = 1e-6
_LIGHTGBM_GRID = (
    {
        "candidate": "lgbm-depth2-leaves3",
        "learning_rate": 0.03,
        "num_leaves": 3,
        "max_depth": 2,
        "min_data_in_leaf": 128,
        "lambda_l2": 1.0,
    },
    {
        "candidate": "lgbm-depth3-leaves7",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 96,
        "lambda_l2": 5.0,
    },
    {
        "candidate": "lgbm-depth4-leaves15",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 128,
        "lambda_l2": 10.0,
    },
)


ProgressCallback = Callable[[str, Mapping[str, object]], None]


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability(value: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(value, dtype=np.float64), _EPSILON, 1.0 - _EPSILON)
    if output.ndim != 1 or not np.all(np.isfinite(output)):
        raise ValueError("historical model probabilities are invalid")
    return output


@dataclass(frozen=True, slots=True)
class HistoricalModelPanel:
    condition_ids: np.ndarray
    roles: np.ndarray
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    dataset_sha256: str

    def validate(self, *, expected_roles: Sequence[str]) -> None:
        rows = len(self.labels)
        roles = tuple(expected_roles)
        if (
            rows == 0
            or self.condition_ids.shape != (rows,)
            or self.roles.shape != (rows,)
            or self.event_start_ms.shape != (rows,)
            or self.decision_time_ms.shape != (rows,)
            or self.features.shape != (rows, len(FEATURE_NAMES))
            or not np.all(np.isfinite(self.features))
            or not np.all((self.labels == 0.0) | (self.labels == 1.0))
            or not set(np.unique(self.roles)).issubset(set(roles))
            or len(self.dataset_sha256) != 64
        ):
            raise ValueError("historical model panel differs")
        unique, counts = np.unique(self.condition_ids, return_counts=True)
        if len(unique) == 0 or np.any(counts != 8):
            raise ValueError("historical model condition decision coverage differs")
        for condition_id in unique:
            selected = self.condition_ids == condition_id
            if len(np.unique(self.labels[selected])) != 1:
                raise ValueError("historical model target differs within a condition")


def _dataset_identity(store: HistoricalScreenStore) -> str:
    row = (
        store.connect()
        .execute(
            """
        SELECT contract_sha256, feature_names_json, feature_names_sha256,
               row_count, condition_count, dataset_sha256
        FROM feature.dataset_manifest WHERE singleton
        """
        )
        .fetchone()
    )
    names_json = _canonical_json(FEATURE_NAMES)
    names_sha = hashlib.sha256(names_json.encode("ascii")).hexdigest()
    if (
        row is None
        or str(row[0]) != store.contract.contract_sha256
        or str(row[1]) != names_json
        or str(row[2]) != names_sha
        or int(row[3]) < 8_000
        or int(row[4]) < 1_000
        or len(str(row[5])) != 64
    ):
        raise ValueError("historical model dataset manifest differs")
    return str(row[5])


def load_historical_model_panel(
    store: HistoricalScreenStore,
    *,
    roles: Sequence[str],
) -> HistoricalModelPanel:
    selected_roles = tuple(str(role) for role in roles)
    if (
        not selected_roles
        or len(selected_roles) != len(set(selected_roles))
        or any(role not in {"train", "tune", "test"} for role in selected_roles)
    ):
        raise ValueError("historical model panel role selection is invalid")
    if "test" in selected_roles and store.state not in {
        "targets_complete",
        "evaluated",
    }:
        raise ValueError(
            "historical test panel is not authorized before one-use access"
        )
    if any(
        role in {"train", "tune"} for role in selected_roles
    ) and store.state not in {
        "development_targets_complete",
        "pretest_complete",
        "targets_complete",
        "evaluated",
    }:
        raise ValueError("historical development panel is not authorized")
    placeholders = ",".join("?" for _ in selected_roles)
    rows = (
        store.connect()
        .execute(
            f"""
        SELECT row.condition_id, row.role, row.event_start_ms,
               row.decision_time_ms, row.feature_values,
               resolution.winning_outcome
        FROM feature.causal_row AS row
        INNER JOIN target.official_resolution AS resolution
          ON resolution.condition_id = row.condition_id
         AND resolution.role = row.role
        WHERE row.role IN ({placeholders})
        ORDER BY row.event_start_ms, row.decision_time_ms
        """,
            list(selected_roles),
        )
        .fetchall()
    )
    if not rows:
        raise ValueError("historical model panel has no joined target rows")
    panel = HistoricalModelPanel(
        condition_ids=np.asarray([str(row[0]) for row in rows], dtype=object),
        roles=np.asarray([str(row[1]) for row in rows], dtype=object),
        event_start_ms=np.asarray([int(row[2]) for row in rows], dtype=np.int64),
        decision_time_ms=np.asarray([int(row[3]) for row in rows], dtype=np.int64),
        features=np.asarray([row[4] for row in rows], dtype=np.float32),
        labels=np.asarray(
            [1.0 if str(row[5]) == "Up" else 0.0 for row in rows],
            dtype=np.float64,
        ),
        dataset_sha256=_dataset_identity(store),
    )
    if any(str(row[5]) not in {"Up", "Down"} for row in rows):
        raise ValueError("historical model target outcome differs")
    panel.validate(expected_roles=selected_roles)
    return panel


def _condition_weights(condition_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(
        np.asarray(condition_ids, dtype=object),
        return_inverse=True,
        return_counts=True,
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights * (len(weights) / float(np.sum(weights)))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sum(np.asarray(values, dtype=np.float64) * weights) / np.sum(weights)
    )


def _log_loss(
    labels: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> float:
    prediction = _probability(probability)
    truth = np.asarray(labels, dtype=np.float64)
    losses = -(truth * np.log(prediction) + (1.0 - truth) * np.log1p(-prediction))
    return _weighted_mean(losses, weights)


def _fit_logistic_parameters(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    features = np.asarray(matrix, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    row_weights = np.asarray(weights, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[0] != len(truth)
        or len(truth) != len(row_weights)
        or min(np.count_nonzero(truth == 0.0), np.count_nonzero(truth == 1.0)) < 2
        or not math.isfinite(l2)
        or l2 <= 0.0
    ):
        raise ValueError("historical ridge logistic inputs are invalid")
    total_weight = float(np.sum(row_weights))
    prevalence = float(np.sum(row_weights * truth) / total_weight)
    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    initial[0] = math.log(prevalence / (1.0 - prevalence))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficient = parameters[1:]
        logits = intercept + features @ coefficient
        loss = float(
            np.sum(row_weights * (np.logaddexp(0.0, logits) - truth * logits))
        ) / total_weight + 0.5 * l2 * float(np.dot(coefficient, coefficient)) / max(
            1, features.shape[1]
        )
        residual = row_weights * (expit(logits) - truth) / total_weight
        gradient = np.concatenate(
            (
                np.asarray([np.sum(residual)]),
                features.T @ residual + (l2 / max(1, features.shape[1])) * coefficient,
            )
        )
        return loss, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise RuntimeError(
            f"historical ridge logistic did not converge: {result.message}"
        )
    return float(result.x[0]), np.asarray(result.x[1:], dtype=np.float64)


def _fit_ridge_candidate(
    train: HistoricalModelPanel,
    tune: HistoricalModelPanel,
    *,
    family: str,
    feature_indices: np.ndarray,
) -> Mapping[str, object]:
    train_weights = _condition_weights(train.condition_ids)
    tune_weights = _condition_weights(tune.condition_ids)
    train_matrix = np.asarray(train.features[:, feature_indices], dtype=np.float64)
    tune_matrix = np.asarray(tune.features[:, feature_indices], dtype=np.float64)
    mean = np.average(train_matrix, axis=0, weights=train_weights)
    variance = np.average(
        np.square(train_matrix - mean),
        axis=0,
        weights=train_weights,
    )
    scale = np.sqrt(np.maximum(variance, 1e-12))
    train_standard = (train_matrix - mean) / scale
    tune_standard = (tune_matrix - mean) / scale
    best: dict[str, object] | None = None
    for l2 in RIDGE_L2_GRID:
        intercept, coefficient = _fit_logistic_parameters(
            train_standard,
            train.labels,
            train_weights,
            l2=l2,
        )
        prediction = expit(intercept + tune_standard @ coefficient)
        loss = _log_loss(tune.labels, prediction, tune_weights)
        candidate = {
            "family": family,
            "kind": "control" if family == "calendar_ridge_logistic" else "challenger",
            "candidate_id": f"{family}-l2-{format(l2, 'g')}",
            "feature_indices": feature_indices.tolist(),
            "model": {
                "type": "ridge_logistic",
                "l2": l2,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "intercept": intercept,
                "coefficient": coefficient.tolist(),
            },
            "raw_tune_log_loss": loss,
        }
        if best is None or (loss, str(candidate["candidate_id"])) < (
            float(best["raw_tune_log_loss"]),
            str(best["candidate_id"]),
        ):
            best = candidate
    if best is None:
        raise RuntimeError("historical ridge candidate selection failed")
    return best


def _fit_lightgbm_candidate(
    train: HistoricalModelPanel,
    tune: HistoricalModelPanel,
    *,
    compute_backend: str,
    progress: ProgressCallback | None,
) -> Mapping[str, object]:
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        MODEL_SEED,
        reproducible=True,
        pin_opencl_device=True,
    )
    train_weights = _condition_weights(train.condition_ids)
    tune_weights = _condition_weights(tune.condition_ids)
    train_set = lgb.Dataset(
        train.features,
        label=train.labels,
        weight=train_weights,
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )
    tune_set = lgb.Dataset(
        tune.features,
        label=tune.labels,
        weight=tune_weights,
        reference=train_set,
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )
    best: dict[str, object] | None = None
    for index, frozen in enumerate(_LIGHTGBM_GRID, start=1):
        if progress:
            progress(
                "historical_model_candidate_started",
                {
                    "family": "binance_shallow_lightgbm",
                    "candidate": frozen["candidate"],
                    "candidate_index": index,
                    "candidate_count": len(_LIGHTGBM_GRID),
                    "backend_kind": backend_kind,
                    "backend_device": backend_device,
                },
            )
        parameters: dict[str, object] = {
            **backend,
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": frozen["learning_rate"],
            "num_leaves": frozen["num_leaves"],
            "max_depth": frozen["max_depth"],
            "min_data_in_leaf": frozen["min_data_in_leaf"],
            "lambda_l1": 0.0,
            "lambda_l2": frozen["lambda_l2"],
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "max_bin": 63,
            "histogram_pool_size": 256,
            "feature_pre_filter": False,
        }
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=256,
            valid_sets=[tune_set],
            valid_names=["tune"],
            callbacks=[lgb.early_stopping(32, verbose=False), lgb.log_evaluation(0)],
        )
        best_iteration = int(booster.best_iteration or booster.current_iteration())
        tune_prediction = _probability(
            np.asarray(
                booster.predict(tune.features, num_iteration=best_iteration),
                dtype=np.float64,
            )
        )
        loss = _log_loss(tune.labels, tune_prediction, tune_weights)
        model_string = booster.model_to_string(num_iteration=best_iteration)
        reload_prediction = np.asarray(
            lgb.Booster(model_str=model_string).predict(tune.features),
            dtype=np.float64,
        )
        reload_difference = float(
            np.max(np.abs(tune_prediction - reload_prediction), initial=0.0)
        )
        if reload_difference > 1e-12:
            raise RuntimeError("historical LightGBM serialization identity failed")
        candidate = {
            "family": "binance_shallow_lightgbm",
            "kind": "challenger",
            "candidate_id": str(frozen["candidate"]),
            "feature_indices": list(range(len(FEATURE_NAMES))),
            "model": {
                "type": "lightgbm",
                "parameters": dict(frozen),
                "best_iteration": best_iteration,
                "model_string": model_string,
                "model_sha256": hashlib.sha256(
                    model_string.encode("utf-8")
                ).hexdigest(),
                "reload_max_absolute_difference": reload_difference,
                "lightgbm_version": str(lgb.__version__),
                "backend_requested": compute_backend,
                "backend_kind": backend_kind,
                "backend_device": backend_device,
            },
            "raw_tune_log_loss": loss,
        }
        if best is None or (loss, str(candidate["candidate_id"])) < (
            float(best["raw_tune_log_loss"]),
            str(best["candidate_id"]),
        ):
            best = candidate
        if progress:
            progress(
                "historical_model_candidate_completed",
                {
                    "family": "binance_shallow_lightgbm",
                    "candidate": frozen["candidate"],
                    "best_iteration": best_iteration,
                    "tune_log_loss": loss,
                },
            )
    if best is None:
        raise RuntimeError("historical LightGBM candidate selection failed")
    return best


def _raw_prediction(
    candidate: Mapping[str, object],
    features: np.ndarray,
) -> np.ndarray:
    model = candidate.get("model")
    indexes = np.asarray(candidate.get("feature_indices"), dtype=np.int64)
    if not isinstance(model, Mapping) or indexes.ndim != 1:
        raise ValueError("historical candidate artifact is malformed")
    model_type = str(model.get("type"))
    if model_type == "constant":
        return np.full(
            len(features),
            float(model["probability"]),
            dtype=np.float64,
        )
    if model_type == "ridge_logistic":
        matrix = np.asarray(features[:, indexes], dtype=np.float64)
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coefficient = np.asarray(model["coefficient"], dtype=np.float64)
        if (
            mean.shape != (len(indexes),)
            or scale.shape != mean.shape
            or coefficient.shape != mean.shape
            or np.any(scale <= 0.0)
        ):
            raise ValueError("historical ridge artifact dimensions differ")
        return expit(
            float(model["intercept"]) + ((matrix - mean) / scale) @ coefficient
        )
    if model_type == "lightgbm":
        booster = lgb.Booster(model_str=str(model["model_string"]))
        matrix = np.asarray(features[:, indexes], dtype=np.float32)
        return np.asarray(booster.predict(matrix), dtype=np.float64)
    raise ValueError("historical candidate model type is unsupported")


def _fit_platt_calibration(
    labels: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> Mapping[str, object]:
    raw = _probability(probability)
    logits = np.log(raw) - np.log1p(-raw)
    before = _log_loss(labels, raw, weights)
    total_weight = float(np.sum(weights))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        calibrated = expit(intercept + slope * logits)
        loss = _log_loss(labels, calibrated, weights)
        residual = weights * (calibrated - labels) / total_weight
        return loss, np.asarray(
            [np.sum(residual), np.sum(residual * logits)],
            dtype=np.float64,
        )

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((-8.0, 8.0), (-4.0, 4.0)),
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
    )
    retained = bool(
        result.success
        and np.all(np.isfinite(result.x))
        and before - float(result.fun) > 1e-9
    )
    return {
        "type": "platt_logistic",
        "retained": retained,
        "intercept": float(result.x[0]) if retained else 0.0,
        "slope": float(result.x[1]) if retained else 1.0,
        "tune_log_loss_before": before,
        "tune_log_loss_after": float(result.fun) if retained else before,
    }


def predict_historical_candidate(
    candidate: Mapping[str, object],
    features: np.ndarray,
) -> np.ndarray:
    raw = _probability(_raw_prediction(candidate, features))
    calibration = candidate.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("historical candidate calibration is missing")
    if calibration.get("retained") is True:
        logits = np.log(raw) - np.log1p(-raw)
        raw = expit(
            float(calibration["intercept"]) + float(calibration["slope"]) * logits
        )
    return _probability(raw)


def _verify_candidate_artifact(candidate: Mapping[str, object]) -> None:
    value = dict(candidate)
    claimed = str(value.pop("artifact_sha256", ""))
    if len(claimed) != 64 or _canonical_sha256(value) != claimed:
        raise ValueError("historical candidate artifact integrity failed")


def _calibration_diagnostics(
    labels: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    prediction = _probability(probability)
    logits = np.log(prediction) - np.log1p(-prediction)
    intercept, parameters = _fit_logistic_parameters(
        logits.reshape(-1, 1),
        labels,
        weights,
        l2=1e-8,
    )
    return intercept, float(parameters[0])


def _ordered_conditions(condition_ids: np.ndarray) -> tuple[object, ...]:
    return tuple(dict.fromkeys(np.asarray(condition_ids, dtype=object).tolist()))


def condition_balanced_binary_metrics(
    panel: HistoricalModelPanel,
    probability: np.ndarray,
) -> Mapping[str, float]:
    prediction = _probability(probability)
    weights = _condition_weights(panel.condition_ids)
    truth = panel.labels
    predicted_class = prediction >= 0.5
    positive = truth == 1.0
    negative = ~positive
    intercept, slope = _calibration_diagnostics(truth, prediction, weights)
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        selected = (
            (prediction >= lower) & (prediction <= upper)
            if index == 9
            else (prediction >= lower) & (prediction < upper)
        )
        if np.any(selected):
            bin_weight = float(np.sum(weights[selected]))
            ece += (
                bin_weight
                / float(np.sum(weights))
                * abs(
                    _weighted_mean(prediction[selected], weights[selected])
                    - _weighted_mean(truth[selected], weights[selected])
                )
            )
    return {
        "log_loss": _log_loss(truth, prediction, weights),
        "brier_score": _weighted_mean(np.square(prediction - truth), weights),
        "accuracy": _weighted_mean(predicted_class == positive, weights),
        "balanced_accuracy": 0.5
        * (
            _weighted_mean(predicted_class[positive], weights[positive])
            + _weighted_mean(~predicted_class[negative], weights[negative])
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
    }


def fit_historical_pretest_candidates(
    train: HistoricalModelPanel,
    tune: HistoricalModelPanel,
    *,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> tuple[Mapping[str, object], ...]:
    train.validate(expected_roles=("train",))
    tune.validate(expected_roles=("tune",))
    if train.dataset_sha256 != tune.dataset_sha256:
        raise ValueError("historical train and tune dataset identities differ")
    train_weights = _condition_weights(train.condition_ids)
    prevalence = _weighted_mean(train.labels, train_weights)
    finalists: list[Mapping[str, object]] = [
        {
            "family": "training_prevalence",
            "kind": "control",
            "candidate_id": "training-prevalence",
            "feature_indices": [],
            "model": {"type": "constant", "probability": prevalence},
            "raw_tune_log_loss": _log_loss(
                tune.labels,
                np.full(len(tune.labels), prevalence),
                _condition_weights(tune.condition_ids),
            ),
        }
    ]
    calendar_indexes = np.asarray(
        [FEATURE_NAMES.index(name) for name in CALENDAR_FEATURE_NAMES],
        dtype=np.int64,
    )
    finalists.append(
        _fit_ridge_candidate(
            train,
            tune,
            family="calendar_ridge_logistic",
            feature_indices=calendar_indexes,
        )
    )
    finalists.append(
        _fit_ridge_candidate(
            train,
            tune,
            family="binance_ridge_logistic",
            feature_indices=np.arange(len(FEATURE_NAMES), dtype=np.int64),
        )
    )
    finalists.append(
        _fit_lightgbm_candidate(
            train,
            tune,
            compute_backend=compute_backend,
            progress=progress,
        )
    )
    output: list[Mapping[str, object]] = []
    tune_weights = _condition_weights(tune.condition_ids)
    for candidate in finalists:
        value = dict(candidate)
        raw = _raw_prediction(value, tune.features)
        calibration = _fit_platt_calibration(tune.labels, raw, tune_weights)
        value["calibration"] = calibration
        calibrated = predict_historical_candidate(value, tune.features)
        value["tune_metrics"] = dict(
            condition_balanced_binary_metrics(tune, calibrated)
        )
        value["artifact_sha256"] = _canonical_sha256(value)
        output.append(value)
    return tuple(output)


def _best_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    kind: str,
) -> Mapping[str, object]:
    eligible = [candidate for candidate in candidates if candidate.get("kind") == kind]
    if not eligible:
        raise ValueError(f"historical model has no {kind} candidate")
    return min(
        eligible,
        key=lambda candidate: (
            float(candidate["tune_metrics"]["log_loss"]),  # type: ignore[index]
            str(candidate["candidate_id"]),
        ),
    )


def freeze_historical_pretest(
    store: HistoricalScreenStore,
    *,
    source_commit: str,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> tuple[Mapping[str, object], str]:
    if store.state != "development_targets_complete":
        raise ValueError("historical pretest fit requires development targets")
    commit = str(source_commit).strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("historical pretest source commit is invalid")
    combined = load_historical_model_panel(store, roles=("train", "tune"))
    train_mask = combined.roles == "train"
    tune_mask = combined.roles == "tune"

    def subset(mask: np.ndarray) -> HistoricalModelPanel:
        return HistoricalModelPanel(
            condition_ids=combined.condition_ids[mask],
            roles=combined.roles[mask],
            event_start_ms=combined.event_start_ms[mask],
            decision_time_ms=combined.decision_time_ms[mask],
            features=combined.features[mask],
            labels=combined.labels[mask],
            dataset_sha256=combined.dataset_sha256,
        )

    train = subset(train_mask)
    tune = subset(tune_mask)
    train.validate(expected_roles=("train",))
    tune.validate(expected_roles=("tune",))
    candidates = fit_historical_pretest_candidates(
        train,
        tune,
        compute_backend=compute_backend,
        progress=progress,
    )
    best_control = _best_candidate(candidates, kind="control")
    best_challenger = _best_candidate(candidates, kind="challenger")
    selected = min(
        (best_control, best_challenger),
        key=lambda candidate: (
            float(candidate["tune_metrics"]["log_loss"]),  # type: ignore[index]
            str(candidate["candidate_id"]),
        ),
    )
    module_path = Path(__file__)
    source_root = module_path.parent
    artifact: dict[str, object] = {
        "schema_version": PRETEST_SCHEMA_VERSION,
        "contract_sha256": store.contract.contract_sha256,
        "dataset_sha256": combined.dataset_sha256,
        "source_commit": commit,
        "implementation_sha256": {
            "model": _file_sha256(module_path),
            "dataset": _file_sha256(source_root / "polymarket_historical_dataset.py"),
            "screen": _file_sha256(source_root / "polymarket_historical_screen.py"),
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_names_sha256": _canonical_sha256(FEATURE_NAMES),
        "partition": {
            "train_conditions": len(np.unique(train.condition_ids)),
            "train_rows": len(train.labels),
            "tune_conditions": len(np.unique(tune.condition_ids)),
            "tune_rows": len(tune.labels),
            "train_up_conditions": int(
                np.count_nonzero(
                    np.asarray(
                        [
                            train.labels[
                                np.flatnonzero(train.condition_ids == value)[0]
                            ]
                            for value in _ordered_conditions(train.condition_ids)
                        ]
                    )
                    == 1.0
                )
            ),
            "tune_up_conditions": int(
                np.count_nonzero(
                    np.asarray(
                        [
                            tune.labels[np.flatnonzero(tune.condition_ids == value)[0]]
                            for value in _ordered_conditions(tune.condition_ids)
                        ]
                    )
                    == 1.0
                )
            ),
        },
        "candidate_contract": {
            "ridge_l2_grid": list(RIDGE_L2_GRID),
            "lightgbm_grid": list(_LIGHTGBM_GRID),
            "selection_metric": "condition_balanced_tune_log_loss",
            "calibration": "bounded_tune_only_platt_logistic_if_improved",
            "test_use": "one_use_no_refit",
        },
        "candidates": list(candidates),
        "best_control_id": str(best_control["candidate_id"]),
        "best_challenger_id": str(best_challenger["candidate_id"]),
        "selected_candidate_id": str(selected["candidate_id"]),
        "evaluation_policy": {
            "bootstrap_repetitions": (
                store.contract.test_gates.bootstrap_repetitions
            ),
            "bootstrap_block_conditions": BOOTSTRAP_BLOCK_CONDITIONS,
            "bootstrap_seed": MODEL_SEED,
            "probability_threshold": 0.5,
            "test_driven_changes_allowed": False,
            "execution_or_pnl_claim_allowed": False,
        },
    }
    artifact_sha = store.record_pretest_artifact(artifact)
    return artifact, artifact_sha


def _condition_loss_delta(
    panel: HistoricalModelPanel,
    control: np.ndarray,
    challenger: np.ndarray,
) -> np.ndarray:
    truth = panel.labels
    control_loss = -(
        truth * np.log(_probability(control))
        + (1.0 - truth) * np.log1p(-_probability(control))
    )
    challenger_loss = -(
        truth * np.log(_probability(challenger))
        + (1.0 - truth) * np.log1p(-_probability(challenger))
    )
    deltas = []
    for condition in _ordered_conditions(panel.condition_ids):
        selected = panel.condition_ids == condition
        deltas.append(
            float(np.mean(control_loss[selected] - challenger_loss[selected]))
        )
    return np.asarray(deltas, dtype=np.float64)


def _paired_block_bootstrap(
    delta: np.ndarray,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    minimum_conditions: int = 250,
) -> Mapping[str, float | int]:
    values = np.asarray(delta, dtype=np.float64)
    selected_repetitions = int(repetitions)
    required_conditions = int(minimum_conditions)
    if (
        values.ndim != 1
        or len(values) < required_conditions
        or not np.all(np.isfinite(values))
        or not 100 <= selected_repetitions <= 100_000
        or required_conditions < 2
    ):
        raise ValueError("historical paired bootstrap inputs are invalid")
    generator = np.random.default_rng(MODEL_SEED)
    sample_count = math.ceil(len(values) / BOOTSTRAP_BLOCK_CONDITIONS)
    estimates = np.empty(selected_repetitions, dtype=np.float64)
    offsets = np.arange(BOOTSTRAP_BLOCK_CONDITIONS, dtype=np.int64)
    for repetition in range(selected_repetitions):
        starts = generator.integers(0, len(values), size=sample_count)
        indexes = (starts[:, None] + offsets[None, :]) % len(values)
        estimates[repetition] = float(np.mean(values[indexes.ravel()[: len(values)]]))
    return {
        "repetitions": selected_repetitions,
        "block_conditions": BOOTSTRAP_BLOCK_CONDITIONS,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def evaluate_historical_test_once(
    store: HistoricalScreenStore,
) -> tuple[Mapping[str, object], str]:
    if store.state != "targets_complete":
        raise ValueError("historical test evaluation requires test targets")
    pretest, pretest_sha = store.pretest_artifact()
    panel = load_historical_model_panel(store, roles=("test",))
    candidates_value = pretest.get("candidates")
    if not isinstance(candidates_value, list) or len(candidates_value) != 4:
        raise ValueError("historical pretest candidates differ")
    candidates = tuple(
        dict(candidate)
        for candidate in candidates_value
        if isinstance(candidate, Mapping)
    )
    if len(candidates) != 4:
        raise ValueError("historical pretest candidate payload is malformed")
    metrics: dict[str, Mapping[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for candidate in candidates:
        _verify_candidate_artifact(candidate)
        candidate_id = str(candidate["candidate_id"])
        prediction = predict_historical_candidate(candidate, panel.features)
        predictions[candidate_id] = prediction
        metrics[candidate_id] = condition_balanced_binary_metrics(panel, prediction)
    control_id = str(pretest["best_control_id"])
    challenger_id = str(pretest["best_challenger_id"])
    control = metrics[control_id]
    challenger = metrics[challenger_id]
    bootstrap = _paired_block_bootstrap(
        _condition_loss_delta(
            panel,
            predictions[control_id],
            predictions[challenger_id],
        ),
        repetitions=store.contract.test_gates.bootstrap_repetitions,
        minimum_conditions=(
            store.contract.test_gates.minimum_terminal_conditions
        ),
    )
    test_gates = store.contract.test_gates
    unique_conditions = _ordered_conditions(panel.condition_ids)
    condition_labels = np.asarray(
        [
            panel.labels[np.flatnonzero(panel.condition_ids == condition)[0]]
            for condition in unique_conditions
        ]
    )
    gates = {
        "minimum_terminal_conditions": (
            len(unique_conditions) >= test_gates.minimum_terminal_conditions
        ),
        "minimum_outcomes_per_class": min(
            np.count_nonzero(condition_labels == 0.0),
            np.count_nonzero(condition_labels == 1.0),
        )
        >= test_gates.minimum_outcomes_per_class,
        "minimum_decision_rows": (
            len(panel.labels) >= test_gates.minimum_decision_rows
        ),
        "challenger_log_loss_skill_positive": (
            float(challenger["log_loss"]) < float(control["log_loss"])
        ),
        "challenger_brier_skill_positive": (
            float(challenger["brier_score"]) < float(control["brier_score"])
        ),
        "challenger_balanced_accuracy_not_lower": (
            float(challenger["balanced_accuracy"])
            >= float(control["balanced_accuracy"])
        ),
        "paired_log_loss_improvement_lower_positive": (
            float(bootstrap["lower_95"]) > 0.0
        ),
        "calibration_slope_in_range": (
            test_gates.calibration_slope_minimum
            <= float(challenger["calibration_slope"])
            <= test_gates.calibration_slope_maximum
        ),
        "expected_calibration_error_at_most_contract_maximum": (
            float(challenger["expected_calibration_error"])
            <= test_gates.expected_calibration_error_maximum
        ),
    }
    artifact: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "contract_sha256": store.contract.contract_sha256,
        "dataset_sha256": panel.dataset_sha256,
        "pretest_artifact_sha256": pretest_sha,
        "scope": {
            "venue": "polymarket",
            "asset": "BTC",
            "market_variant": "fiveminute",
            "predictive_screen_only": True,
            "execution_or_profitability_claim": False,
        },
        "test": {
            "conditions": len(unique_conditions),
            "decision_rows": len(panel.labels),
            "up_conditions": int(np.count_nonzero(condition_labels == 1.0)),
            "down_conditions": int(np.count_nonzero(condition_labels == 0.0)),
            "first_event_start_ms": int(np.min(panel.event_start_ms)),
            "last_event_start_ms": int(np.max(panel.event_start_ms)),
        },
        "best_control_id": control_id,
        "best_challenger_id": challenger_id,
        "candidate_metrics": metrics,
        "challenger_skill": {
            "log_loss": 1.0
            - float(challenger["log_loss"]) / float(control["log_loss"]),
            "brier": 1.0
            - float(challenger["brier_score"]) / float(control["brier_score"]),
        },
        "paired_condition_block_bootstrap": bootstrap,
        "gates": gates,
        "accepted_predictive_edge": all(gates.values()),
        "failure_action": (
            "retain_as_predictive_research_only"
            if all(gates.values())
            else "reject_candidate_without_trading_or_profitability_claim"
        ),
    }
    artifact_sha = store.record_evaluation_artifact(artifact)
    return artifact, artifact_sha


__all__ = [
    "BOOTSTRAP_BLOCK_CONDITIONS",
    "BOOTSTRAP_REPETITIONS",
    "EVALUATION_SCHEMA_VERSION",
    "MODEL_SEED",
    "PRETEST_SCHEMA_VERSION",
    "RIDGE_L2_GRID",
    "HistoricalModelPanel",
    "condition_balanced_binary_metrics",
    "evaluate_historical_test_once",
    "fit_historical_pretest_candidates",
    "freeze_historical_pretest",
    "load_historical_model_panel",
    "predict_historical_candidate",
]
