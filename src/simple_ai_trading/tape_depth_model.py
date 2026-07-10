"""Purged gross-return forecasting for the long-history tape/depth dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Callable, Mapping

import lightgbm as lgb
import numpy as np

from .microstructure_model import (
    _apply_platt_scaling,
    _auc,
    _backend_parameters,
    _fit_platt_scaling,
)
from .storage import write_json_atomic
from .tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_FEATURE_VERSION,
    TAPE_DEPTH_TARGET_MODE,
    TapeDepthForecastDataset,
)


TAPE_DEPTH_MODEL_SCHEMA_VERSION = "tape-depth-gross-forecast-v1"
_RISK_LEVELS = frozenset({"conservative", "regular", "aggressive"})


@dataclass(frozen=True)
class TapeDepthSplitEvidence:
    train_rows: int
    tuning_rows: int
    calibration_rows: int
    evaluation_rows: int
    train_end_ms: int
    tuning_start_ms: int
    tuning_end_ms: int
    calibration_start_ms: int
    calibration_end_ms: int
    evaluation_start_ms: int
    purge_ms: int
    purged_rows: int


@dataclass(frozen=True)
class TapeDepthForecastMetrics:
    rows: int
    direction_auc: float
    direction_brier: float
    prevalence_brier: float
    direction_accuracy: float
    majority_accuracy: float
    mean_absolute_error_bps: float
    zero_baseline_mae_bps: float
    root_mean_squared_error_bps: float
    zero_baseline_rmse_bps: float
    pearson_information_coefficient: float
    spearman_information_coefficient: float
    interval_80_coverage: float
    interval_crossing_rate: float
    top_decile_rows: int
    top_decile_signed_gross_bps: float
    top_decile_mean_signed_gross_bps: float
    top_decile_positive_rate: float


@dataclass(frozen=True)
class TapeDepthModelArtifact:
    schema_version: str
    model_family: str
    status: str
    rejection_reasons: tuple[str, ...]
    trading_authority: bool
    execution_claim: bool
    symbol: str
    risk_level: str
    feature_version: str
    feature_names: tuple[str, ...]
    target_mode: str
    horizon_seconds: int
    total_latency_ms: int
    decision_cadence_seconds: int
    maximum_depth_age_ms: int
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    split: TapeDepthSplitEvidence
    best_iterations: Mapping[str, int]
    probability_calibration: tuple[float, float]
    evaluation_metrics: TapeDepthForecastMetrics
    model_strings: Mapping[str, str]
    dataset_summary: Mapping[str, object]
    trained_at: str

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["feature_names"] = list(self.feature_names)
        return payload


def _risk_parameters(risk_level: str, train_rows: int) -> dict[str, object]:
    if risk_level == "conservative":
        leaves, depth, min_leaf, l2 = 31, 6, 256, 0.10
    elif risk_level == "regular":
        leaves, depth, min_leaf, l2 = 47, 7, 160, 0.05
    else:
        leaves, depth, min_leaf, l2 = 63, 8, 96, 0.025
    return {
        "learning_rate": 0.025,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": max(32, min(min_leaf, train_rows // 100)),
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": l2,
        "max_bin": 127,
    }


def _purged_segments(
    dataset: TapeDepthForecastDataset,
    *,
    minimum_segment_rows: int,
) -> tuple[dict[str, np.ndarray], TapeDepthSplitEvidence]:
    rows = dataset.rows
    minimum = int(minimum_segment_rows)
    if minimum < 128:
        raise ValueError("minimum_segment_rows must be at least 128")
    if rows < minimum * 4 + 16:
        raise ValueError("tape/depth dataset is too small for four purged segments")
    boundaries = (int(rows * 0.60), int(rows * 0.75), int(rows * 0.85))
    train_raw = np.arange(0, boundaries[0], dtype=np.int64)
    tuning_raw = np.arange(boundaries[0], boundaries[1], dtype=np.int64)
    calibration_raw = np.arange(boundaries[1], boundaries[2], dtype=np.int64)
    evaluation = np.arange(boundaries[2], rows, dtype=np.int64)
    times = np.asarray(dataset.decision_time_ms, dtype=np.int64)
    exits = np.asarray(dataset.target_exit_time_ms, dtype=np.int64)

    def purge_before(indexes: np.ndarray, next_start: int) -> np.ndarray:
        return indexes[exits[indexes] < times[next_start]]

    train = purge_before(train_raw, int(tuning_raw[0]))
    tuning = purge_before(tuning_raw, int(calibration_raw[0]))
    calibration = purge_before(calibration_raw, int(evaluation[0]))
    segments = {
        "train": train,
        "tuning": tuning,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    if any(len(indexes) < minimum for indexes in segments.values()):
        raise ValueError("purging left a tape/depth segment below its minimum")
    target_spans = exits - times
    if np.any(target_spans <= 0) or np.any(target_spans != target_spans[0]):
        raise ValueError("tape/depth target spans are inconsistent")
    purge_ms = int(target_spans[0])
    evidence = TapeDepthSplitEvidence(
        train_rows=len(train),
        tuning_rows=len(tuning),
        calibration_rows=len(calibration),
        evaluation_rows=len(evaluation),
        train_end_ms=int(times[train[-1]]),
        tuning_start_ms=int(times[tuning[0]]),
        tuning_end_ms=int(times[tuning[-1]]),
        calibration_start_ms=int(times[calibration[0]]),
        calibration_end_ms=int(times[calibration[-1]]),
        evaluation_start_ms=int(times[evaluation[0]]),
        purge_ms=purge_ms,
        purged_rows=(
            len(train_raw) - len(train)
            + len(tuning_raw) - len(tuning)
            + len(calibration_raw) - len(calibration)
        ),
    )
    return segments, evidence


def _weights(targets: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(targets, dtype=np.float64))
    scale = max(1.0, float(np.quantile(values, 0.90)))
    return (1.0 + np.clip(values / scale, 0.0, 3.0)).astype(np.float32)


def _train_with_early_stopping(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_tuning: np.ndarray,
    y_tuning: np.ndarray,
    parameters: Mapping[str, object],
    objective: str,
    metric: str,
    alpha: float | None = None,
    train_weights: np.ndarray | None = None,
    tuning_weights: np.ndarray | None = None,
) -> int:
    config = dict(parameters)
    config.update({"objective": objective, "metric": metric})
    if alpha is not None:
        config["alpha"] = float(alpha)
    train = lgb.Dataset(
        x_train,
        label=y_train,
        weight=_weights(y_train) if train_weights is None else train_weights,
        free_raw_data=False,
    )
    tuning = lgb.Dataset(
        x_tuning,
        label=y_tuning,
        weight=_weights(y_tuning) if tuning_weights is None else tuning_weights,
        reference=train,
        free_raw_data=False,
    )
    model = lgb.train(
        config,
        train,
        num_boost_round=1_200,
        valid_sets=[tuning],
        valid_names=["tuning"],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    return max(1, int(model.best_iteration or model.current_iteration()))


def _train_fixed(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    parameters: Mapping[str, object],
    objective: str,
    metric: str,
    iterations: int,
    alpha: float | None = None,
    sample_weights: np.ndarray | None = None,
) -> lgb.Booster:
    config = dict(parameters)
    config.update({"objective": objective, "metric": metric})
    if alpha is not None:
        config["alpha"] = float(alpha)
    return lgb.train(
        config,
        lgb.Dataset(
            features,
            label=targets,
            weight=_weights(targets) if sample_weights is None else sample_weights,
            free_raw_data=False,
        ),
        num_boost_round=int(iterations),
        callbacks=[lgb.log_evaluation(0)],
    )


def _rank(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    order = np.argsort(source, kind="stable")
    ranks = np.empty(len(source), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and source[order[end]] == source[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if len(first) < 2 or np.std(first) <= 0.0 or np.std(second) <= 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _evaluation_metrics(
    *,
    targets: np.ndarray,
    direction_probability: np.ndarray,
    mean_prediction: np.ndarray,
    lower_prediction: np.ndarray,
    upper_prediction: np.ndarray,
) -> TapeDepthForecastMetrics:
    actual = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(direction_probability, dtype=np.float64)
    predicted = np.asarray(mean_prediction, dtype=np.float64)
    lower_raw = np.asarray(lower_prediction, dtype=np.float64)
    upper_raw = np.asarray(upper_prediction, dtype=np.float64)
    labels = (actual > 0.0).astype(np.float64)
    prevalence = float(np.mean(labels))
    lower = np.minimum(lower_raw, upper_raw)
    upper = np.maximum(lower_raw, upper_raw)
    crossing = float(np.mean(lower_raw > upper_raw))
    threshold = float(np.quantile(np.abs(predicted), 0.90))
    selected = np.flatnonzero(np.abs(predicted) >= threshold)
    signed = np.sign(predicted[selected]) * actual[selected]
    errors = predicted - actual
    return TapeDepthForecastMetrics(
        rows=len(actual),
        direction_auc=_auc(labels.astype(np.int8), probabilities),
        direction_brier=float(np.mean((probabilities - labels) ** 2)),
        prevalence_brier=float(np.mean((prevalence - labels) ** 2)),
        direction_accuracy=float(np.mean((probabilities >= 0.5) == (labels > 0.5))),
        majority_accuracy=max(prevalence, 1.0 - prevalence),
        mean_absolute_error_bps=float(np.mean(np.abs(errors))),
        zero_baseline_mae_bps=float(np.mean(np.abs(actual))),
        root_mean_squared_error_bps=float(np.sqrt(np.mean(errors**2))),
        zero_baseline_rmse_bps=float(np.sqrt(np.mean(actual**2))),
        pearson_information_coefficient=_correlation(predicted, actual),
        spearman_information_coefficient=_correlation(_rank(predicted), _rank(actual)),
        interval_80_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
        interval_crossing_rate=crossing,
        top_decile_rows=len(selected),
        top_decile_signed_gross_bps=float(np.sum(signed)),
        top_decile_mean_signed_gross_bps=float(np.mean(signed)),
        top_decile_positive_rate=float(np.mean(signed > 0.0)),
    )


def train_tape_depth_forecaster(
    dataset: TapeDepthForecastDataset,
    *,
    risk_level: str = "conservative",
    compute_backend: str = "auto",
    seed: int = 20260710,
    minimum_segment_rows: int = 2_000,
    progress: Callable[[str, int, int], None] | None = None,
) -> TapeDepthModelArtifact:
    """Train a research-only forecaster that cannot authorize execution."""

    risk = str(risk_level).strip().lower()
    if risk not in _RISK_LEVELS:
        raise ValueError("risk_level must be conservative, regular, or aggressive")
    if (
        dataset.feature_version != TAPE_DEPTH_FEATURE_VERSION
        or dataset.feature_names != TAPE_DEPTH_FEATURE_NAMES
        or dataset.target_mode != TAPE_DEPTH_TARGET_MODE
    ):
        raise ValueError("tape/depth dataset contract is unsupported")
    if not bool(dataset.source_evidence.get("verified")):
        raise ValueError("tape/depth source evidence is not verified")
    segments, split = _purged_segments(
        dataset,
        minimum_segment_rows=minimum_segment_rows,
    )
    x = np.asarray(dataset.features, dtype=np.float32)
    target = np.asarray(dataset.gross_return_bps, dtype=np.float32)
    labels = (target > 0.0).astype(np.float32)
    backend, backend_kind, backend_device = _backend_parameters(compute_backend, seed)
    parameters = {**backend, **_risk_parameters(risk, len(segments["train"]))}
    train = segments["train"]
    tuning = segments["tuning"]
    calibration = segments["calibration"]
    evaluation = segments["evaluation"]
    specifications = {
        "direction": (labels, "binary", "binary_logloss", None),
        "mean": (target, "huber", "l1", 0.90),
        "lower": (target, "quantile", "quantile", 0.10),
        "upper": (target, "quantile", "quantile", 0.90),
    }
    best_iterations: dict[str, int] = {}
    models: dict[str, lgb.Booster] = {}
    final_fit = np.concatenate((train, tuning))
    economic_weights = _weights(target)
    total_steps = len(specifications) * 2 + 2
    completed = 0
    for name, (values, objective, metric, alpha) in specifications.items():
        if progress:
            progress(f"tune-{name}", completed, total_steps)
        rounds = _train_with_early_stopping(
            x_train=x[train],
            y_train=values[train],
            x_tuning=x[tuning],
            y_tuning=values[tuning],
            parameters=parameters,
            objective=objective,
            metric=metric,
            alpha=alpha,
            train_weights=economic_weights[train],
            tuning_weights=economic_weights[tuning],
        )
        best_iterations[name] = rounds
        completed += 1
        if progress:
            progress(f"refit-{name}", completed, total_steps)
        models[name] = _train_fixed(
            features=x[final_fit],
            targets=values[final_fit],
            parameters=parameters,
            objective=objective,
            metric=metric,
            iterations=rounds,
            alpha=alpha,
            sample_weights=economic_weights[final_fit],
        )
        completed += 1
    if progress:
        progress("calibrate-direction", completed, total_steps)
    raw_calibration = np.asarray(models["direction"].predict(x[calibration]), dtype=np.float64)
    probability_calibration = _fit_platt_scaling(
        raw_calibration,
        labels[calibration],
    )
    completed += 1
    if progress:
        progress("evaluate", completed, total_steps)
    probability = _apply_platt_scaling(
        np.asarray(models["direction"].predict(x[evaluation]), dtype=np.float64),
        probability_calibration,
    )
    mean_prediction = np.asarray(models["mean"].predict(x[evaluation]), dtype=np.float64)
    lower_prediction = np.asarray(models["lower"].predict(x[evaluation]), dtype=np.float64)
    upper_prediction = np.asarray(models["upper"].predict(x[evaluation]), dtype=np.float64)
    if not all(
        np.all(np.isfinite(values))
        for values in (probability, mean_prediction, lower_prediction, upper_prediction)
    ):
        raise ValueError("tape/depth forecaster emitted non-finite evaluation values")
    metrics = _evaluation_metrics(
        targets=target[evaluation],
        direction_probability=probability,
        mean_prediction=mean_prediction,
        lower_prediction=lower_prediction,
        upper_prediction=upper_prediction,
    )
    reasons: list[str] = []
    if metrics.direction_auc <= 0.51:
        reasons.append("direction_auc_not_above_0_51")
    if metrics.direction_brier >= metrics.prevalence_brier:
        reasons.append("direction_brier_not_better_than_prevalence")
    if metrics.mean_absolute_error_bps >= metrics.zero_baseline_mae_bps:
        reasons.append("mean_forecast_mae_not_better_than_zero")
    if metrics.spearman_information_coefficient <= 0.0:
        reasons.append("spearman_information_coefficient_not_positive")
    if metrics.top_decile_mean_signed_gross_bps <= 0.0:
        reasons.append("top_decile_signed_gross_return_not_positive")
    if not 0.50 <= metrics.interval_80_coverage <= 0.98:
        reasons.append("quantile_interval_coverage_outside_sanity_band")
    if metrics.interval_crossing_rate > 0.05:
        reasons.append("quantile_interval_crossing_rate_too_high")
    model_strings = {
        name: model.model_to_string(num_iteration=best_iterations[name])
        for name, model in models.items()
    }
    if progress:
        progress("complete", total_steps, total_steps)
    return TapeDepthModelArtifact(
        schema_version=TAPE_DEPTH_MODEL_SCHEMA_VERSION,
        model_family="lightgbm_direction_huber_quantile_ensemble",
        status="research_candidate" if not reasons else "rejected",
        rejection_reasons=tuple(reasons),
        trading_authority=False,
        execution_claim=False,
        symbol=dataset.symbol,
        risk_level=risk,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        target_mode=dataset.target_mode,
        horizon_seconds=dataset.horizon_seconds,
        total_latency_ms=dataset.total_latency_ms,
        decision_cadence_seconds=dataset.decision_cadence_seconds,
        maximum_depth_age_ms=dataset.maximum_depth_age_ms,
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        split=split,
        best_iterations=best_iterations,
        probability_calibration=probability_calibration,
        evaluation_metrics=metrics,
        model_strings=model_strings,
        dataset_summary=dataset.summary(),
        trained_at=datetime.now(tz=UTC).isoformat(),
    )


def save_tape_depth_model_artifact(
    artifact: TapeDepthModelArtifact,
    path: str | Path,
) -> None:
    if artifact.trading_authority or artifact.execution_claim:
        raise ValueError("gross tape/depth artifacts cannot carry execution authority")
    write_json_atomic(Path(path), artifact.asdict(), indent=2, sort_keys=True)


def load_tape_depth_model_artifact(path: str | Path) -> TapeDepthModelArtifact:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("tape/depth artifact is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("tape/depth artifact must be a JSON object")
    if payload.get("schema_version") != TAPE_DEPTH_MODEL_SCHEMA_VERSION:
        raise ValueError("tape/depth artifact schema is unsupported")
    if payload.get("feature_version") != TAPE_DEPTH_FEATURE_VERSION:
        raise ValueError("tape/depth feature version is unsupported")
    if tuple(payload.get("feature_names") or ()) != TAPE_DEPTH_FEATURE_NAMES:
        raise ValueError("tape/depth feature names drifted")
    if payload.get("target_mode") != TAPE_DEPTH_TARGET_MODE:
        raise ValueError("tape/depth target mode is unsupported")
    if payload.get("status") not in {"research_candidate", "rejected"}:
        raise ValueError("tape/depth artifact status is unsupported")
    if payload.get("trading_authority") is not False or payload.get("execution_claim") is not False:
        raise ValueError("tape/depth gross forecast cannot authorize trading")
    try:
        values = dict(payload)
        values["rejection_reasons"] = tuple(payload.get("rejection_reasons") or ())
        values["feature_names"] = tuple(payload.get("feature_names") or ())
        values["split"] = TapeDepthSplitEvidence(**dict(payload["split"]))
        values["evaluation_metrics"] = TapeDepthForecastMetrics(
            **dict(payload["evaluation_metrics"])
        )
        values["probability_calibration"] = tuple(
            payload["probability_calibration"]
        )
        return TapeDepthModelArtifact(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tape/depth artifact fields are incomplete") from exc


__all__ = [
    "TAPE_DEPTH_MODEL_SCHEMA_VERSION",
    "TapeDepthForecastMetrics",
    "TapeDepthModelArtifact",
    "TapeDepthSplitEvidence",
    "load_tape_depth_model_artifact",
    "save_tape_depth_model_artifact",
    "train_tape_depth_forecaster",
]
