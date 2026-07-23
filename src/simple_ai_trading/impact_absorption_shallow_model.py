"""Frozen shallow model family and deterministic artifacts for Round 73."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from typing import Literal

import lightgbm as lgb
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import zstandard

from .impact_absorption_model_dataset import (
    ROUND73_OBSERVED_STATUS,
    ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
    ROUND73_PRE_ENTRY_ABORT_STATUS,
    ROUND73_PRIMARY_ENTRY_DELAY_MS,
    ROUND73_PRIMARY_HORIZON_MS,
    ROUND73_RIGHT_CENSORED_STATUS,
)
from .impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
    ROUND73_EVALUATION_CONTRACT_SHA256,
    ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
    ROUND73_MODEL_FEATURE_NAMES_BY_LAYER,
    ROUND73_MODEL_FEATURE_SHA256_BY_LAYER,
)
from .impact_absorption_model_slice import Round73SymbolModelSlice
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS
from .impact_absorption_target_store_v2 import _stream_hash
from .impact_absorption_target_store_v3 import (
    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
)
from .impact_absorption_targets import ROUND73_TARGET_MAX_STATE_LATENESS_NS
from .lightgbm_backend import (
    lightgbm_backend_parameters,
    selected_opencl_gpu_device,
)


ROUND73_SHALLOW_MODEL_SCHEMA_VERSION = "round-073-shallow-symbol-model-v1"
ROUND73_SHALLOW_PREPROCESSOR_SCHEMA_VERSION = "round-073-shallow-symbol-preprocessor-v1"
ROUND73_SHALLOW_PREDICTION_SCHEMA_VERSION = "round-073-shallow-predictions-v1"
ROUND73_SHALLOW_TRAINING_SCHEMA_VERSION = "round-073-shallow-training-run-v1"
ROUND73_SHALLOW_CANDIDATES = (
    "linear_l1_tape",
    "l1_tape",
    "l2_state",
    "impact_absorption",
)
ROUND73_PROBABILITY_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
ROUND73_MODEL_SEED = 20260722

_PREDICTION_MAGIC = b"SAT73P1\n"
_PREDICTION_COMPRESSION_LEVEL = 3
_PROBABILITY_EPSILON = 1e-6
_LINEAR_L2 = 1e-4
_LINEAR_HUBER_DELTA_BPS = 5.0
_LINEAR_MAXIMUM_ITERATIONS = 80
_MAXIMUM_BOOSTING_ITERATIONS = 256
_EARLY_STOPPING_ROUNDS = 30
_MINIMUM_DATA_IN_LEAF = 200
_MODEL_SELECTION_MINIMUM_RELATIVE_IMPROVEMENT = 0.002
_BOOTSTRAP_DRAWS = 10_000
_BOOTSTRAP_LOWER_QUANTILE = 0.05
_MINIMUM_COMPLETED_TUNING_TRADES = 25
_TUNING_MAXIMUM_POSITION_NS = (
    ROUND73_PRIMARY_ENTRY_DELAY_MS + ROUND73_PRIMARY_HORIZON_MS
) * 1_000_000 + 2 * ROUND73_TARGET_MAX_STATE_LATENESS_NS

CandidatePredictions = Mapping[str, tuple[np.ndarray, np.ndarray]]
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class Round73InsufficientModelSupport(ValueError):
    """A symbol is valid but cannot support the frozen model family."""


def _process_peak_working_set_bytes() -> int | None:
    if os.name == "nt":

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory.restype = ctypes.c_int
        process = get_current_process()
        if get_process_memory(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            return int(counters.peak_working_set_size)
        return None
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _feature_indexes(layer: str) -> np.ndarray:
    index_by_name = {
        name: index for index, name in enumerate(ROUND73_ACTION_ALIGNED_FEATURE_NAMES)
    }
    return np.asarray(
        [index_by_name[name] for name in ROUND73_MODEL_FEATURE_NAMES_BY_LAYER[layer]],
        dtype=np.int64,
    )


_FEATURE_INDEXES = {
    layer: _feature_indexes(layer) for layer in ROUND73_MODEL_FEATURE_NAMES_BY_LAYER
}


def _project(values: np.ndarray, layer: str) -> np.ndarray:
    output = np.ascontiguousarray(
        values[:, _FEATURE_INDEXES[layer]],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("Round 73 projected model features are nonfinite")
    return output


def _binary_losses(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(probability, dtype=np.float64)
    if (
        truth.ndim != 1
        or predicted.shape != truth.shape
        or len(truth) == 0
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(predicted))
        or np.any((truth != 0.0) & (truth != 1.0))
        or np.any((predicted < 0.0) | (predicted > 1.0))
    ):
        raise ValueError("Round 73 binary metric inputs are invalid")
    clipped = np.clip(predicted, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    log_loss = -float(
        np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped))
    )
    return log_loss, float(np.mean(np.square(predicted - truth)))


def _continuous_losses(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[float, float]:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if (
        truth.ndim != 1
        or predicted.shape != truth.shape
        or len(truth) == 0
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(predicted))
    ):
        raise ValueError("Round 73 continuous metric inputs are invalid")
    error = predicted - truth
    return float(np.mean(np.square(error))), float(np.mean(np.abs(error)))


def _fit_linear_preprocessor(
    training_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(training_features, dtype=np.float64)
    median = np.median(values, axis=0)
    lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
    iqr = upper - lower
    retained = np.flatnonzero(np.isfinite(iqr) & (iqr > 0.0)).astype(np.int64)
    if (
        not np.all(np.isfinite(median))
        or not np.all(np.isfinite(iqr))
        or len(retained) == 0
    ):
        raise ValueError("Round 73 linear preprocessor has no finite support")
    return median, iqr, retained


def _scale_linear(
    features: np.ndarray,
    *,
    median: np.ndarray,
    iqr: np.ndarray,
    retained: np.ndarray,
) -> np.ndarray:
    output = np.ascontiguousarray(
        (np.asarray(features[:, retained], dtype=np.float64) - median[retained])
        / iqr[retained],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("Round 73 scaled linear features are nonfinite")
    return output


def _fit_logistic(features: np.ndarray, target: np.ndarray) -> dict[str, object]:
    values = np.asarray(features, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if len(np.unique(truth)) != 2:
        raise ValueError("Round 73 logistic training target is single class")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:-1]
        intercept = parameters[-1]
        score = values @ coefficients + intercept
        probability = expit(score)
        loss = float(
            np.mean(np.logaddexp(0.0, score) - truth * score)
            + 0.5 * _LINEAR_L2 * np.dot(coefficients, coefficients)
        )
        residual = probability - truth
        gradient = np.empty_like(parameters)
        gradient[:-1] = values.T @ residual / len(truth) + _LINEAR_L2 * coefficients
        gradient[-1] = float(np.mean(residual))
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(values.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": _LINEAR_MAXIMUM_ITERATIONS, "ftol": 1e-12, "gtol": 1e-7},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise RuntimeError(f"Round 73 logistic fit failed: {result.message}")
    return {
        "kind": "logistic_regression",
        "coefficients": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "l2": _LINEAR_L2,
    }


def _huber_loss_and_gradient(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(residual)
    quadratic = absolute <= _LINEAR_HUBER_DELTA_BPS
    loss = np.where(
        quadratic,
        0.5 * np.square(residual),
        _LINEAR_HUBER_DELTA_BPS * (absolute - 0.5 * _LINEAR_HUBER_DELTA_BPS),
    )
    gradient = np.where(
        quadratic,
        residual,
        _LINEAR_HUBER_DELTA_BPS * np.sign(residual),
    )
    return loss, gradient


def _fit_huber(features: np.ndarray, target: np.ndarray) -> dict[str, object]:
    values = np.asarray(features, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:-1]
        intercept = parameters[-1]
        residual = values @ coefficients + intercept - truth
        row_loss, row_gradient = _huber_loss_and_gradient(residual)
        loss = float(
            np.mean(row_loss) + 0.5 * _LINEAR_L2 * np.dot(coefficients, coefficients)
        )
        gradient = np.empty_like(parameters)
        gradient[:-1] = values.T @ row_gradient / len(truth) + _LINEAR_L2 * coefficients
        gradient[-1] = float(np.mean(row_gradient))
        return loss, gradient

    initial = np.zeros(values.shape[1] + 1, dtype=np.float64)
    initial[-1] = float(np.median(truth))
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": _LINEAR_MAXIMUM_ITERATIONS, "ftol": 1e-12, "gtol": 1e-7},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise RuntimeError(f"Round 73 Huber fit failed: {result.message}")
    return {
        "kind": "huber_regression",
        "coefficients": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "l2": _LINEAR_L2,
        "delta_bps": _LINEAR_HUBER_DELTA_BPS,
    }


def _predict_linear(model: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
    intercept = float(model.get("intercept", float("nan")))
    values = np.asarray(features, dtype=np.float64)
    if (
        coefficients.shape != (values.shape[1],)
        or not np.all(np.isfinite(coefficients))
        or not math.isfinite(intercept)
    ):
        raise ValueError("Round 73 serialized linear model is invalid")
    score = values @ coefficients + intercept
    if str(model.get("kind")) == "logistic_regression":
        score = expit(score)
    output = np.ascontiguousarray(score, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("Round 73 linear predictions are nonfinite")
    return output


def _lightgbm_parameters(
    backend: Mapping[str, object],
    *,
    objective: str,
) -> dict[str, object]:
    output = {
        **backend,
        "objective": objective,
        "metric": "binary_logloss" if objective == "binary" else "l2",
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_data_in_leaf": _MINIMUM_DATA_IN_LEAF,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "histogram_pool_size": 512,
    }
    if objective == "huber":
        output["alpha"] = 0.9
    return output


def _fit_lightgbm_head(
    training_features: np.ndarray,
    training_target: np.ndarray,
    tuning_features: np.ndarray,
    tuning_target: np.ndarray,
    *,
    feature_names: Sequence[str],
    backend_parameters: Mapping[str, object],
    objective: Literal["binary", "huber"],
) -> tuple[str, int, np.ndarray, np.ndarray, float]:
    train = lgb.Dataset(
        training_features,
        label=training_target,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    tune = lgb.Dataset(
        tuning_features,
        label=tuning_target,
        reference=train,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    booster = lgb.train(
        _lightgbm_parameters(backend_parameters, objective=objective),
        train,
        num_boost_round=_MAXIMUM_BOOSTING_ITERATIONS,
        valid_sets=(tune,),
        valid_names=("tuning",),
        callbacks=(
            lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ),
    )
    iteration = int(booster.best_iteration or booster.current_iteration())
    if not 1 <= iteration <= _MAXIMUM_BOOSTING_ITERATIONS:
        raise RuntimeError("Round 73 LightGBM best iteration is invalid")
    train_prediction = np.asarray(
        booster.predict(training_features, num_iteration=iteration), dtype=np.float64
    )
    tune_prediction = np.asarray(
        booster.predict(tuning_features, num_iteration=iteration), dtype=np.float64
    )
    model_string = booster.model_to_string(num_iteration=iteration)
    reloaded = lgb.Booster(model_str=model_string)
    reload_train = np.asarray(reloaded.predict(training_features), dtype=np.float64)
    reload_tune = np.asarray(reloaded.predict(tuning_features), dtype=np.float64)
    maximum_difference = float(
        max(
            np.max(np.abs(train_prediction - reload_train), initial=0.0),
            np.max(np.abs(tune_prediction - reload_tune), initial=0.0),
        )
    )
    if (
        maximum_difference > 1e-12
        or not np.all(np.isfinite(train_prediction))
        or not np.all(np.isfinite(tune_prediction))
        or (
            objective == "binary"
            and np.any((tune_prediction < 0) | (tune_prediction > 1))
        )
    ):
        raise RuntimeError("Round 73 serialized LightGBM prediction identity failed")
    return (
        model_string,
        iteration,
        train_prediction,
        tune_prediction,
        maximum_difference,
    )


def _training_role_row_identity(
    dataset: Round73SymbolModelSlice,
    role: str,
) -> dict[str, object]:
    mask = dataset.role_mask(role) & dataset.model_label_mask
    raw = (
        np.ascontiguousarray(dataset.option_sha256_binary[mask])
        .view(np.uint8)
        .reshape(-1, 32)
    )
    values = [bytes(value).hex() for value in raw]
    return {"row_count": len(values), "rows_sha256": _stream_hash(values)}


def _selection_score(
    binary_target: np.ndarray,
    continuous_target: np.ndarray,
    binary_prediction: np.ndarray,
    continuous_prediction: np.ndarray,
    *,
    prevalence: float,
) -> tuple[float, Mapping[str, float]]:
    log_loss, brier = _binary_losses(binary_target, binary_prediction)
    mse, mae = _continuous_losses(continuous_target, continuous_prediction)
    baseline_log, baseline_brier = _binary_losses(
        binary_target, np.full(len(binary_target), prevalence)
    )
    baseline_mse, baseline_mae = _continuous_losses(
        continuous_target, np.zeros(len(continuous_target), dtype=np.float64)
    )
    denominators = (baseline_log, baseline_brier, baseline_mse, baseline_mae)
    if any(value <= 0.0 or not math.isfinite(value) for value in denominators):
        raise ValueError("Round 73 tuning controls have no positive loss support")
    ratios = (
        log_loss / baseline_log,
        brier / baseline_brier,
        mse / baseline_mse,
        mae / baseline_mae,
    )
    score = float(np.mean(ratios))
    return score, {
        "log_loss": log_loss,
        "brier_score": brier,
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "normalized_composite_loss": score,
    }


def _select_candidate(
    tuning_metrics: Mapping[str, Mapping[str, float]],
) -> str:
    selected = ROUND73_SHALLOW_CANDIDATES[0]
    selected_score = float(tuning_metrics[selected]["normalized_composite_loss"])
    for candidate in ROUND73_SHALLOW_CANDIDATES[1:]:
        score = float(tuning_metrics[candidate]["normalized_composite_loss"])
        relative = (selected_score - score) / selected_score
        if relative >= _MODEL_SELECTION_MINIMUM_RELATIVE_IMPROVEMENT:
            selected = candidate
            selected_score = score
    return selected


def _bootstrap_expectancy_lower(
    net_bps: np.ndarray,
    run_id_binary: np.ndarray,
    *,
    seed: int,
) -> float:
    values = np.asarray(net_bps, dtype=np.float64)
    runs = np.asarray(run_id_binary)
    unique_runs, inverse = np.unique(runs, return_inverse=True)
    if len(values) == 0 or len(unique_runs) < 2:
        return float("nan")
    sums = np.bincount(inverse, weights=values, minlength=len(unique_runs))
    counts = np.bincount(inverse, minlength=len(unique_runs))
    random = np.random.default_rng(seed)
    samples = np.empty(_BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, _BOOTSTRAP_DRAWS, 512):
        stop = min(_BOOTSTRAP_DRAWS, start + 512)
        indexes = random.integers(
            0,
            len(unique_runs),
            size=(stop - start, len(unique_runs)),
            endpoint=False,
        )
        sample_sums = np.sum(sums[indexes], axis=1)
        sample_counts = np.sum(counts[indexes], axis=1)
        samples[start:stop] = sample_sums / sample_counts
    return float(np.quantile(samples, _BOOTSTRAP_LOWER_QUANTILE))


@dataclass(frozen=True)
class _ThresholdResult:
    threshold: float
    completed_trades: int
    attempted_actions: int
    pre_entry_aborts: int
    unresolved_risk_count: int
    lower_expectancy_bps: float
    selected_row_indexes: np.ndarray


def _evaluate_tuning_threshold(
    dataset: Round73SymbolModelSlice,
    probability: np.ndarray,
    predicted_net_bps: np.ndarray,
    *,
    threshold: float,
) -> _ThresholdResult:
    tuning = dataset.role_mask("tuning")
    pair_rows = np.flatnonzero(tuning[::2])
    long_rows = pair_rows * 2
    short_rows = long_rows + 1
    long_ok = (probability[long_rows] >= threshold) & (
        predicted_net_bps[long_rows] > 0.0
    )
    short_ok = (probability[short_rows] >= threshold) & (
        predicted_net_bps[short_rows] > 0.0
    )
    selected = np.full(len(pair_rows), -1, dtype=np.int64)
    selected[long_ok & ~short_ok] = long_rows[long_ok & ~short_ok]
    selected[short_ok & ~long_ok] = short_rows[short_ok & ~long_ok]
    both = long_ok & short_ok
    selected[both & (predicted_net_bps[long_rows] > predicted_net_bps[short_rows])] = (
        long_rows[both & (predicted_net_bps[long_rows] > predicted_net_bps[short_rows])]
    )
    selected[both & (predicted_net_bps[short_rows] > predicted_net_bps[long_rows])] = (
        short_rows[
            both & (predicted_net_bps[short_rows] > predicted_net_bps[long_rows])
        ]
    )
    selected = selected[selected >= 0]
    accepted: list[int] = []
    active_until_wall_ns = -1
    attempted = 0
    pre_entry_aborts = 0
    unresolved = 0
    for row_index in selected:
        status = int(dataset.outcome_status[row_index])
        if status == ROUND73_RIGHT_CENSORED_STATUS:
            continue
        wall_ns = int(dataset.anchor_wall_ns[row_index])
        if wall_ns < active_until_wall_ns:
            continue
        attempted += 1
        if status == ROUND73_PRE_ENTRY_ABORT_STATUS:
            pre_entry_aborts += 1
            continue
        if status == ROUND73_POST_ENTRY_UNRESOLVED_STATUS:
            unresolved += 1
            active_until_wall_ns = np.iinfo(np.int64).max
            continue
        if status != ROUND73_OBSERVED_STATUS:
            raise ValueError("Round 73 tuning policy encountered an unknown status")
        accepted.append(int(row_index))
        active_until_wall_ns = wall_ns + _TUNING_MAXIMUM_POSITION_NS
    accepted_indexes = np.asarray(accepted, dtype=np.int64)
    lower = (
        _bootstrap_expectancy_lower(
            dataset.continuous_target_bps[accepted_indexes],
            dataset.run_id_binary[accepted_indexes],
            seed=ROUND73_MODEL_SEED + int(round(threshold * 100)),
        )
        if len(accepted_indexes) >= _MINIMUM_COMPLETED_TUNING_TRADES and unresolved == 0
        else float("nan")
    )
    return _ThresholdResult(
        threshold=threshold,
        completed_trades=len(accepted_indexes),
        attempted_actions=attempted,
        pre_entry_aborts=pre_entry_aborts,
        unresolved_risk_count=unresolved,
        lower_expectancy_bps=lower,
        selected_row_indexes=accepted_indexes,
    )


def _select_threshold(
    dataset: Round73SymbolModelSlice,
    probability: np.ndarray,
    predicted_net_bps: np.ndarray,
) -> tuple[float, bool, tuple[Mapping[str, object], ...]]:
    results = tuple(
        _evaluate_tuning_threshold(
            dataset,
            probability,
            predicted_net_bps,
            threshold=threshold,
        )
        for threshold in ROUND73_PROBABILITY_THRESHOLDS
    )
    admissible = [
        item
        for item in results
        if item.completed_trades >= _MINIMUM_COMPLETED_TUNING_TRADES
        and item.unresolved_risk_count == 0
        and math.isfinite(item.lower_expectancy_bps)
        and item.lower_expectancy_bps > 0.0
    ]
    selected = (
        max(
            admissible,
            key=lambda item: (
                item.lower_expectancy_bps,
                item.threshold,
                -item.completed_trades,
            ),
        )
        if admissible
        else results[-1]
    )
    admissible_thresholds = {item.threshold for item in admissible}
    report = tuple(
        {
            "threshold": item.threshold,
            "completed_trades": item.completed_trades,
            "attempted_actions": item.attempted_actions,
            "pre_entry_aborts": item.pre_entry_aborts,
            "unresolved_risk_count": item.unresolved_risk_count,
            "lower_95_percent_expectancy_bps": (
                item.lower_expectancy_bps
                if math.isfinite(item.lower_expectancy_bps)
                else None
            ),
            "positive_lower_expectancy_required": True,
            "admissible": item.threshold in admissible_thresholds,
        }
        for item in results
    )
    return selected.threshold, bool(admissible), report


def encode_round73_prediction_artifact(
    *,
    symbol: str,
    role: str,
    source_rows_sha256: str,
    row_indexes: np.ndarray,
    predictions: CandidatePredictions,
) -> bytes:
    if (
        symbol not in IMPACT_CAPTURE_SYMBOLS
        or role not in {"training", "tuning", "test"}
        or len(source_rows_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_rows_sha256)
        or set(predictions) != set(ROUND73_SHALLOW_CANDIDATES)
    ):
        raise ValueError("Round 73 prediction artifact identity is invalid")
    indexes = np.ascontiguousarray(row_indexes, dtype="<i8")
    ordered = tuple(ROUND73_SHALLOW_CANDIDATES)
    arrays: list[np.ndarray] = [indexes]
    for candidate in ordered:
        binary, continuous = predictions[candidate]
        arrays.extend(
            (
                np.ascontiguousarray(binary[row_indexes], dtype="<f8"),
                np.ascontiguousarray(continuous[row_indexes], dtype="<f8"),
            )
        )
    raw = b"".join(memoryview(value).cast("B") for value in arrays)
    header = {
        "schema_version": ROUND73_SHALLOW_PREDICTION_SCHEMA_VERSION,
        "symbol": symbol,
        "role": role,
        "source_rows_sha256": source_rows_sha256,
        "row_count": len(indexes),
        "candidate_order": list(ordered),
        "array_order": [
            "row_index",
            *[
                f"{candidate}:{head}"
                for candidate in ordered
                for head in ("positive_probability", "predicted_net_bps")
            ],
        ],
        "row_index_dtype": "<i8",
        "prediction_dtype": "<f8",
        "uncompressed_payload_bytes": len(raw),
        "uncompressed_payload_sha256": _sha256_bytes(raw),
        "compression": "zstd",
        "compression_level": _PREDICTION_COMPRESSION_LEVEL,
    }
    header_bytes = _canonical_bytes(header)
    compressed = zstandard.ZstdCompressor(
        level=_PREDICTION_COMPRESSION_LEVEL,
        threads=0,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)
    return (
        _PREDICTION_MAGIC
        + struct.pack("<Q", len(header_bytes))
        + header_bytes
        + compressed
    )


def decode_round73_prediction_artifact(payload: bytes) -> Mapping[str, object]:
    value = bytes(payload)
    prefix = len(_PREDICTION_MAGIC) + 8
    if len(value) <= prefix or not value.startswith(_PREDICTION_MAGIC):
        raise ValueError("Round 73 prediction artifact framing is invalid")
    header_size = struct.unpack("<Q", value[len(_PREDICTION_MAGIC) : prefix])[0]
    if header_size <= 0 or prefix + header_size >= len(value):
        raise ValueError("Round 73 prediction artifact header length is invalid")
    try:
        header = json.loads(value[prefix : prefix + header_size].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 73 prediction artifact header is invalid") from exc
    if (
        not isinstance(header, Mapping)
        or header.get("schema_version") != ROUND73_SHALLOW_PREDICTION_SCHEMA_VERSION
        or header.get("candidate_order") != list(ROUND73_SHALLOW_CANDIDATES)
        or header.get("row_index_dtype") != "<i8"
        or header.get("prediction_dtype") != "<f8"
        or header.get("compression") != "zstd"
    ):
        raise ValueError("Round 73 prediction artifact contract differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            value[prefix + header_size :],
            max_output_size=int(header["uncompressed_payload_bytes"]),
        )
    except (KeyError, TypeError, ValueError, zstandard.ZstdError) as exc:
        raise ValueError("Round 73 prediction artifact decompression failed") from exc
    rows = int(header.get("row_count", -1))
    expected_arrays = 1 + 2 * len(ROUND73_SHALLOW_CANDIDATES)
    if (
        rows < 0
        or len(raw) != rows * 8 * expected_arrays
        or _sha256_bytes(raw) != header.get("uncompressed_payload_sha256")
    ):
        raise ValueError("Round 73 prediction artifact payload differs")
    matrix = np.frombuffer(raw, dtype="<f8").reshape(expected_arrays, rows)
    indexes = np.frombuffer(raw, dtype="<i8", count=rows).copy()
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_index, candidate in enumerate(ROUND73_SHALLOW_CANDIDATES):
        binary = matrix[1 + 2 * candidate_index].copy()
        continuous = matrix[2 + 2 * candidate_index].copy()
        binary.setflags(write=False)
        continuous.setflags(write=False)
        predictions[candidate] = (binary, continuous)
    indexes.setflags(write=False)
    return {"header": dict(header), "row_indexes": indexes, "predictions": predictions}


@dataclass(frozen=True)
class Round73PreparedPretestArtifacts:
    model_manifest: Mapping[str, object]
    artifacts: Mapping[str, bytes]
    symbol_reports: tuple[Mapping[str, object], ...]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": ROUND73_SHALLOW_TRAINING_SCHEMA_VERSION,
            "symbol_reports": list(self.symbol_reports),
            "artifact_count": len(self.artifacts),
            "artifact_bytes": sum(len(value) for value in self.artifacts.values()),
            "artifact_manifest_sha256": _sha256_json(
                [
                    {
                        "name": name,
                        "sha256": _sha256_bytes(payload),
                        "bytes": len(payload),
                    }
                    for name, payload in sorted(self.artifacts.items())
                ]
            ),
            "test_target_read": False,
            "model_evaluated": False,
            "profitability_claim": False,
            "trading_authority": False,
        }


def _fit_symbol(
    dataset: Round73SymbolModelSlice,
    *,
    backend_parameters: Mapping[str, object],
    backend_kind: str,
    backend_device: str,
    progress_callback: ProgressCallback | None,
) -> tuple[Mapping[str, object], Mapping[str, bytes], Mapping[str, object]]:
    started = time.perf_counter()
    process_peak_before = _process_peak_working_set_bytes()
    dataset.validate()
    if dataset.role_scope != "development":
        raise ValueError("Round 73 shallow model may only fit development rows")
    training_mask = dataset.role_mask("training") & dataset.model_label_mask
    tuning_mask = dataset.role_mask("tuning") & dataset.model_label_mask
    if (
        np.count_nonzero(training_mask) < 2 * _MINIMUM_DATA_IN_LEAF
        or np.count_nonzero(tuning_mask) < _MINIMUM_COMPLETED_TUNING_TRADES
        or len(np.unique(dataset.binary_target[training_mask])) != 2
        or len(np.unique(dataset.binary_target[tuning_mask])) != 2
    ):
        raise Round73InsufficientModelSupport(
            f"insufficient training or tuning support for {dataset.symbol}"
        )
    training_prevalence = float(np.mean(dataset.binary_target[training_mask]))
    l1 = _project(dataset.feature_values, "l1_tape")
    median, iqr, retained = _fit_linear_preprocessor(l1[training_mask])
    linear_training = _scale_linear(
        l1[training_mask], median=median, iqr=iqr, retained=retained
    )
    linear_tuning = _scale_linear(
        l1[tuning_mask], median=median, iqr=iqr, retained=retained
    )
    linear_all = _scale_linear(l1, median=median, iqr=iqr, retained=retained)
    logistic = _fit_logistic(linear_training, dataset.binary_target[training_mask])
    huber = _fit_huber(linear_training, dataset.continuous_target_bps[training_mask])
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "linear_l1_tape": (
            _predict_linear(logistic, linear_all),
            _predict_linear(huber, linear_all),
        )
    }
    models: dict[str, object] = {
        "linear_l1_tape": {
            "family": "logistic_huber",
            "feature_layer": "l1_tape",
            "feature_names_sha256": ROUND73_MODEL_FEATURE_SHA256_BY_LAYER["l1_tape"],
            "retained_feature_indexes": retained.tolist(),
            "binary": logistic,
            "continuous": huber,
        }
    }
    if progress_callback is not None:
        progress_callback(
            "linear_control_completed",
            {"symbol": dataset.symbol, "feature_layer": "l1_tape"},
        )
    del linear_training, linear_tuning, linear_all

    maximum_reload_difference = 0.0
    for layer in ("l1_tape", "l2_state", "impact_absorption"):
        if progress_callback is not None:
            progress_callback(
                "lightgbm_layer_started",
                {"symbol": dataset.symbol, "feature_layer": layer},
            )
        features = l1 if layer == "l1_tape" else _project(dataset.feature_values, layer)
        names = ROUND73_MODEL_FEATURE_NAMES_BY_LAYER[layer]
        (
            binary_model,
            binary_iteration,
            _train_binary,
            _tune_binary,
            binary_difference,
        ) = _fit_lightgbm_head(
            features[training_mask],
            dataset.binary_target[training_mask],
            features[tuning_mask],
            dataset.binary_target[tuning_mask],
            feature_names=names,
            backend_parameters=backend_parameters,
            objective="binary",
        )
        (
            continuous_model,
            continuous_iteration,
            _train_net,
            _tune_net,
            continuous_difference,
        ) = _fit_lightgbm_head(
            features[training_mask],
            dataset.continuous_target_bps[training_mask],
            features[tuning_mask],
            dataset.continuous_target_bps[tuning_mask],
            feature_names=names,
            backend_parameters=backend_parameters,
            objective="huber",
        )
        binary_booster = lgb.Booster(model_str=binary_model)
        continuous_booster = lgb.Booster(model_str=continuous_model)
        binary_all = np.asarray(binary_booster.predict(features), dtype=np.float64)
        continuous_all = np.asarray(
            continuous_booster.predict(features), dtype=np.float64
        )
        if (
            not np.all(np.isfinite(binary_all))
            or np.any((binary_all < 0.0) | (binary_all > 1.0))
            or not np.all(np.isfinite(continuous_all))
        ):
            raise RuntimeError("Round 73 LightGBM full predictions are invalid")
        predictions[layer] = (binary_all, continuous_all)
        models[layer] = {
            "family": "lightgbm",
            "feature_layer": layer,
            "feature_names_sha256": ROUND73_MODEL_FEATURE_SHA256_BY_LAYER[layer],
            "binary_model_string": binary_model,
            "continuous_model_string": continuous_model,
            "binary_best_iteration": binary_iteration,
            "continuous_best_iteration": continuous_iteration,
        }
        maximum_reload_difference = max(
            maximum_reload_difference, binary_difference, continuous_difference
        )
        if progress_callback is not None:
            progress_callback(
                "lightgbm_layer_completed",
                {
                    "symbol": dataset.symbol,
                    "feature_layer": layer,
                    "binary_best_iteration": binary_iteration,
                    "continuous_best_iteration": continuous_iteration,
                },
            )
        if layer != "l1_tape":
            del features

    tuning_metrics: dict[str, Mapping[str, float]] = {}
    for candidate in ROUND73_SHALLOW_CANDIDATES:
        binary, continuous = predictions[candidate]
        tuning_metrics[candidate] = _selection_score(
            dataset.binary_target[tuning_mask],
            dataset.continuous_target_bps[tuning_mask],
            binary[tuning_mask],
            continuous[tuning_mask],
            prevalence=training_prevalence,
        )[1]
    selected_candidate = _select_candidate(tuning_metrics)
    selected_probability, selected_net = predictions[selected_candidate]
    threshold, action_enabled, threshold_report = _select_threshold(
        dataset, selected_probability, selected_net
    )
    selected_family = (
        "logistic_regression" if selected_candidate == "linear_l1_tape" else "lightgbm"
    )
    selected_layer = (
        "l1_tape" if selected_candidate == "linear_l1_tape" else selected_candidate
    )
    l1_names = ROUND73_MODEL_FEATURE_NAMES_BY_LAYER["l1_tape"]
    dropped = [
        name for index, name in enumerate(l1_names) if index not in set(retained)
    ]
    preprocessor = {
        "schema_version": ROUND73_SHALLOW_PREPROCESSOR_SCHEMA_VERSION,
        "staged_holdout_contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
        "feature_contract_sha256": ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
        "symbol": dataset.symbol,
        "l1_feature_names": list(l1_names),
        "l1_feature_names_sha256": ROUND73_MODEL_FEATURE_SHA256_BY_LAYER["l1_tape"],
        "median": median.tolist(),
        "iqr": iqr.tolist(),
        "retained_feature_indexes": retained.tolist(),
        "dropped_zero_iqr_columns": dropped,
    }
    process_peak_after = _process_peak_working_set_bytes()
    resource_observation = {
        "training_wall_seconds": time.perf_counter() - started,
        "process_peak_working_set_bytes_before_fit": process_peak_before,
        "process_peak_working_set_bytes_after_fit": process_peak_after,
        "process_peak_measurement_scope": "operating_system_process_lifetime",
        "device_peak_memory_bytes": None,
        "device_peak_memory_measurement": (
            "unavailable: LightGBM exposes no per-training device-allocation "
            "telemetry through its Python API"
        ),
        "model_size_measurement": (
            "exact serialized byte counts are recorded by the pretest artifact manifest"
        ),
    }
    model = {
        "schema_version": ROUND73_SHALLOW_MODEL_SCHEMA_VERSION,
        "staged_holdout_contract_sha256": ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
        "evaluation_contract_sha256": ROUND73_EVALUATION_CONTRACT_SHA256,
        "feature_contract_sha256": ROUND73_MODEL_FEATURE_CONTRACT_SHA256,
        "feature_names_sha256": ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
        "symbol": dataset.symbol,
        "candidate_order": list(ROUND73_SHALLOW_CANDIDATES),
        "models": models,
        "selected_candidate": selected_candidate,
        "selected_model_family": selected_family,
        "selected_feature_layer": selected_layer,
        "selected_probability_threshold": threshold,
        "action_enabled": action_enabled,
        "training_prevalence": training_prevalence,
        "tuning_metrics": tuning_metrics,
        "threshold_selection": list(threshold_report),
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "lightgbm_version": str(lgb.__version__),
        "maximum_reload_prediction_difference": maximum_reload_difference,
        "resource_observation": resource_observation,
        "test_target_read": False,
    }
    model_bytes = _canonical_bytes(model)
    preprocessor_bytes = _canonical_bytes(preprocessor)
    training_indexes = np.flatnonzero(dataset.role_mask("training"))
    tuning_indexes = np.flatnonzero(dataset.role_mask("tuning"))
    stem = dataset.symbol.lower()
    artifact_names = {
        "model": f"{stem}-round73-model-v1.json",
        "preprocessor": f"{stem}-round73-preprocessor-v1.json",
        "training_predictions": f"{stem}-round73-training-predictions-v1.bin.zst",
        "tuning_predictions": f"{stem}-round73-tuning-predictions-v1.bin.zst",
    }
    artifacts = {
        artifact_names["model"]: model_bytes,
        artifact_names["preprocessor"]: preprocessor_bytes,
        artifact_names["training_predictions"]: encode_round73_prediction_artifact(
            symbol=dataset.symbol,
            role="training",
            source_rows_sha256=dataset.source_rows_sha256,
            row_indexes=training_indexes,
            predictions=predictions,
        ),
        artifact_names["tuning_predictions"]: encode_round73_prediction_artifact(
            symbol=dataset.symbol,
            role="tuning",
            source_rows_sha256=dataset.source_rows_sha256,
            row_indexes=tuning_indexes,
            predictions=predictions,
        ),
    }
    selected_iterations = (
        1
        if selected_candidate == "linear_l1_tape"
        else max(
            int(models[selected_candidate]["binary_best_iteration"]),
            int(models[selected_candidate]["continuous_best_iteration"]),
        )
    )
    symbol_manifest = {
        "status": "enabled",
        "model_family": selected_family,
        "selected_feature_layer": selected_layer,
        "best_boosting_iteration": selected_iterations,
        "probability_threshold": threshold,
        "artifact_names": artifact_names,
    }
    report = {
        "symbol": dataset.symbol,
        "rows": dataset.rows,
        "training_label_rows": int(np.count_nonzero(training_mask)),
        "tuning_label_rows": int(np.count_nonzero(tuning_mask)),
        "selected_candidate": selected_candidate,
        "selected_probability_threshold": threshold,
        "action_enabled": action_enabled,
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "maximum_reload_prediction_difference": maximum_reload_difference,
        "resource_observation": resource_observation,
        "model_sha256": _sha256_bytes(model_bytes),
        "preprocessor_sha256": _sha256_bytes(preprocessor_bytes),
        "test_target_read": False,
    }
    return symbol_manifest, artifacts, report


def prepare_round73_pretest_artifacts(
    datasets: Iterable[Round73SymbolModelSlice],
    *,
    compute_backend: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> Round73PreparedPretestArtifacts:
    """Fit every symbol without accepting or reading any test-role dataset."""

    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        ROUND73_MODEL_SEED,
        reproducible=True,
        pin_opencl_device=True,
    )
    if backend_kind == "opencl" and backend_parameters.get("gpu_use_dp") is not True:
        raise RuntimeError("Round 73 OpenCL training requires FP64 accumulation")
    opencl_identity = None
    if backend_kind == "opencl":
        platform_id = backend_parameters.get("gpu_platform_id")
        device_id = backend_parameters.get("gpu_device_id")
        if not isinstance(platform_id, int) or not isinstance(device_id, int):
            raise RuntimeError("Round 73 OpenCL training device was not pinned")
        opencl_identity = selected_opencl_gpu_device(platform_id, device_id)
        backend_device = opencl_identity.display_name
    if progress_callback is not None:
        progress_callback(
            "backend_resolved",
            {
                "backend_kind": backend_kind,
                "backend_device": backend_device,
                "gpu_accelerated": backend_kind in {"opencl", "cuda"},
            },
        )
    artifacts: dict[str, bytes] = {}
    symbol_models: dict[str, object] = {}
    symbol_reports: list[Mapping[str, object]] = []
    row_identities: dict[str, dict[str, object]] = {
        "training": {},
        "tuning": {},
    }
    dropped: dict[str, list[str]] = {}
    seen: set[str] = set()
    for dataset in datasets:
        symbol = dataset.symbol
        if symbol not in IMPACT_CAPTURE_SYMBOLS or symbol in seen:
            raise ValueError(
                "Round 73 development model slices must cover every symbol once"
            )
        seen.add(symbol)
        dataset.validate()
        if progress_callback is not None:
            progress_callback(
                "symbol_fit_started",
                {"symbol": symbol, "rows": dataset.rows},
            )
        for role in ("training", "tuning"):
            row_identities[role][symbol] = _training_role_row_identity(dataset, role)
        try:
            symbol_manifest, symbol_artifacts, report = _fit_symbol(
                dataset,
                backend_parameters=backend_parameters,
                backend_kind=backend_kind,
                backend_device=backend_device,
                progress_callback=progress_callback,
            )
        except Round73InsufficientModelSupport as exc:
            reason = str(exc)
            symbol_models[symbol] = {"status": "disabled", "reason": reason}
            symbol_reports.append(
                {
                    "symbol": symbol,
                    "rows": dataset.rows,
                    "status": "disabled",
                    "reason": reason,
                    "test_target_read": False,
                }
            )
            dropped[symbol] = []
            if progress_callback is not None:
                progress_callback(
                    "symbol_disabled",
                    {"symbol": symbol, "reason": reason},
                )
            continue
        overlap = set(artifacts) & set(symbol_artifacts)
        if overlap:
            raise ValueError("Round 73 model artifact names overlap")
        artifacts.update(symbol_artifacts)
        symbol_models[symbol] = symbol_manifest
        symbol_reports.append(report)
        preprocessor_name = symbol_manifest["artifact_names"]["preprocessor"]
        preprocessor = json.loads(symbol_artifacts[preprocessor_name].decode("ascii"))
        dropped[symbol] = list(preprocessor["dropped_zero_iqr_columns"])
        if progress_callback is not None:
            progress_callback(
                "symbol_fit_completed",
                {
                    "symbol": symbol,
                    "selected_candidate": report["selected_candidate"],
                    "action_enabled": report["action_enabled"],
                },
            )
    if seen != set(IMPACT_CAPTURE_SYMBOLS):
        raise ValueError(
            "Round 73 development model slices must cover every symbol once"
        )
    if not artifacts:
        raise Round73InsufficientModelSupport(
            "no Round 73 symbol has enough support for the frozen model family"
        )
    compute_identity = {
        "resolved_backend": backend_kind,
        "device_name": backend_device,
        "platform_name": (
            opencl_identity.platform_name if opencl_identity else "LightGBM"
        ),
        "device_type": "gpu" if backend_kind in {"opencl", "cuda"} else "cpu",
        "gpu_accelerated": backend_kind in {"opencl", "cuda"},
        "library_versions": {
            "lightgbm": str(lgb.__version__),
            "numpy": str(np.__version__),
            "python": platform.python_version(),
            "scipy": str(scipy.__version__),
            "zstandard": str(zstandard.__version__),
        },
    }
    if opencl_identity is not None:
        compute_identity["opencl_device"] = opencl_identity.as_dict()
    model_manifest = {
        "feature_schema": {
            "feature_names": list(ROUND73_ACTION_ALIGNED_FEATURE_NAMES),
            "feature_names_sha256": ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
            "transforms": {
                "action_alignment": "round-073-action-aligned-features-v1",
                "linear_scaler": "training_median_iqr",
                "tree_scaler": "none",
            },
            "dropped_zero_iqr_columns": dropped,
        },
        "row_identities": row_identities,
        "compute_backend": compute_identity,
        "symbol_models": symbol_models,
        "action_policy": {
            "candidate_probability_thresholds": list(ROUND73_PROBABILITY_THRESHOLDS),
            "one_active_position_per_symbol": True,
            "pre_entry_revalidation": True,
            "exact_side_score_tie_policy": "no_trade",
            "profit_reinvestment": False,
            "leverage": 1.0,
        },
    }
    return Round73PreparedPretestArtifacts(
        model_manifest=model_manifest,
        artifacts=artifacts,
        symbol_reports=tuple(symbol_reports),
    )


def _strict_json_artifact(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(bytes(payload).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 73 {label} artifact is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 73 {label} artifact must be an object")
    return value


def predict_round73_frozen_symbol_model(
    dataset: Round73SymbolModelSlice,
    *,
    model_payload: bytes,
    preprocessor_payload: bytes,
) -> tuple[CandidatePredictions, Mapping[str, object]]:
    """Score only serialized pretest bytes; fitting and calibration are impossible."""

    dataset.validate()
    model = _strict_json_artifact(model_payload, "model")
    preprocessor = _strict_json_artifact(preprocessor_payload, "preprocessor")
    if (
        model.get("schema_version") != ROUND73_SHALLOW_MODEL_SCHEMA_VERSION
        or model.get("symbol") != dataset.symbol
        or model.get("candidate_order") != list(ROUND73_SHALLOW_CANDIDATES)
        or model.get("feature_names_sha256")
        != ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256
        or model.get("staged_holdout_contract_sha256")
        != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or model.get("evaluation_contract_sha256") != ROUND73_EVALUATION_CONTRACT_SHA256
        or model.get("feature_contract_sha256") != ROUND73_MODEL_FEATURE_CONTRACT_SHA256
        or model.get("test_target_read") is not False
        or preprocessor.get("schema_version")
        != ROUND73_SHALLOW_PREPROCESSOR_SCHEMA_VERSION
        or preprocessor.get("symbol") != dataset.symbol
        or preprocessor.get("staged_holdout_contract_sha256")
        != ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256
        or preprocessor.get("feature_contract_sha256")
        != ROUND73_MODEL_FEATURE_CONTRACT_SHA256
    ):
        raise ValueError("Round 73 frozen symbol artifact identity differs")
    models = model.get("models")
    if not isinstance(models, Mapping) or set(models) != set(
        ROUND73_SHALLOW_CANDIDATES
    ):
        raise ValueError("Round 73 frozen model candidates differ")
    l1 = _project(dataset.feature_values, "l1_tape")
    median = np.asarray(preprocessor.get("median"), dtype=np.float64)
    iqr = np.asarray(preprocessor.get("iqr"), dtype=np.float64)
    retained = np.asarray(preprocessor.get("retained_feature_indexes"), dtype=np.int64)
    if (
        median.shape != (l1.shape[1],)
        or iqr.shape != median.shape
        or retained.ndim != 1
        or len(retained) == 0
        or np.any(np.diff(retained) <= 0)
        or retained[0] < 0
        or retained[-1] >= l1.shape[1]
    ):
        raise ValueError("Round 73 frozen preprocessor values differ")
    linear_features = _scale_linear(l1, median=median, iqr=iqr, retained=retained)
    linear_model = models["linear_l1_tape"]
    if (
        not isinstance(linear_model, Mapping)
        or linear_model.get("family") != "logistic_huber"
        or linear_model.get("feature_layer") != "l1_tape"
        or linear_model.get("feature_names_sha256")
        != ROUND73_MODEL_FEATURE_SHA256_BY_LAYER["l1_tape"]
        or linear_model.get("retained_feature_indexes") != retained.tolist()
    ):
        raise ValueError("Round 73 frozen linear model is invalid")
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "linear_l1_tape": (
            _predict_linear(linear_model["binary"], linear_features),
            _predict_linear(linear_model["continuous"], linear_features),
        )
    }
    del linear_features
    for layer in ("l1_tape", "l2_state", "impact_absorption"):
        candidate = models[layer]
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("family") != "lightgbm"
            or candidate.get("feature_layer") != layer
            or candidate.get("feature_names_sha256")
            != ROUND73_MODEL_FEATURE_SHA256_BY_LAYER[layer]
        ):
            raise ValueError("Round 73 frozen LightGBM model identity differs")
        features = l1 if layer == "l1_tape" else _project(dataset.feature_values, layer)
        try:
            binary = lgb.Booster(model_str=str(candidate["binary_model_string"]))
            continuous = lgb.Booster(
                model_str=str(candidate["continuous_model_string"])
            )
            binary_prediction = np.asarray(binary.predict(features), dtype=np.float64)
            continuous_prediction = np.asarray(
                continuous.predict(features), dtype=np.float64
            )
        except (KeyError, lgb.basic.LightGBMError) as exc:
            raise ValueError(
                "Round 73 frozen LightGBM model could not be loaded"
            ) from exc
        if (
            binary_prediction.shape != (dataset.rows,)
            or continuous_prediction.shape != (dataset.rows,)
            or not np.all(np.isfinite(binary_prediction))
            or np.any((binary_prediction < 0.0) | (binary_prediction > 1.0))
            or not np.all(np.isfinite(continuous_prediction))
        ):
            raise ValueError("Round 73 frozen model predictions are invalid")
        predictions[layer] = (binary_prediction, continuous_prediction)
        if layer != "l1_tape":
            del features
    return predictions, model


__all__ = [
    "ROUND73_MODEL_SEED",
    "ROUND73_PROBABILITY_THRESHOLDS",
    "ROUND73_SHALLOW_CANDIDATES",
    "ROUND73_SHALLOW_MODEL_SCHEMA_VERSION",
    "ROUND73_SHALLOW_PREDICTION_SCHEMA_VERSION",
    "ROUND73_SHALLOW_PREPROCESSOR_SCHEMA_VERSION",
    "ROUND73_SHALLOW_TRAINING_SCHEMA_VERSION",
    "ProgressCallback",
    "Round73PreparedPretestArtifacts",
    "Round73InsufficientModelSupport",
    "decode_round73_prediction_artifact",
    "encode_round73_prediction_artifact",
    "predict_round73_frozen_symbol_model",
    "prepare_round73_pretest_artifacts",
]
