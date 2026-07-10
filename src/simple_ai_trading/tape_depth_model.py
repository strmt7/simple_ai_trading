"""Purged gross-return forecasting for the long-history tape/depth dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import lightgbm as lgb
from lightgbm.basic import LightGBMError
import numpy as np

from .lightgbm_backend import lightgbm_backend_parameters
from .microstructure_model import (
    _apply_platt_scaling,
    _auc,
    _fit_platt_scaling,
)
from .storage import write_json_atomic
from .tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_FEATURE_VERSION,
    TAPE_DEPTH_TARGET_MODE,
    TapeDepthForecastDataset,
    tape_depth_dataset_fingerprint,
)


TAPE_DEPTH_MODEL_SCHEMA_VERSION = "tape-depth-gross-forecast-v5"
_RISK_LEVELS = frozenset({"conservative", "regular", "aggressive"})
_MODEL_PROFILES = frozenset({"regularized", "balanced", "expressive"})
_FEATURE_SETS = frozenset({"core", "tape_derived", "cross_asset", "full"})
_DERIVED_PREFIXES = (
    "vwap_deviation_bps_",
    "price_efficiency_",
    "trade_observation_rate_",
    "quote_volume_rate_acceleration_",
    "trade_rate_acceleration_",
    "flow_price_alignment_",
)
_DEPTH_PREFIXES = ("depth_", "log_depth_", "log_bid_depth_", "log_ask_depth_")
_CROSS_ASSET_PREFIXES = (
    "cross_asset_",
    "peer_",
    "relative_return_vs_",
    "btc_anchor_",
)


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
    calibration_threshold_rows: int
    calibration_threshold_signed_gross_bps: float
    calibration_threshold_mean_signed_gross_bps: float
    calibration_threshold_positive_rate: float


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
    model_profile: str
    feature_set: str
    feature_version: str
    feature_names: tuple[str, ...]
    model_feature_names: tuple[str, ...]
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
    training_weight_scale_bps: float
    probability_calibration: tuple[float, float]
    signal_threshold_bps: float
    evaluation_metrics: TapeDepthForecastMetrics
    model_strings: Mapping[str, str]
    dataset_fingerprint: str
    dataset_summary: Mapping[str, object]
    trained_at: str

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["feature_names"] = list(self.feature_names)
        payload["model_feature_names"] = list(self.model_feature_names)
        return payload


@dataclass(frozen=True)
class TapeDepthPredictionBatch:
    decision_time_ms: np.ndarray
    target_entry_time_ms: np.ndarray
    target_exit_time_ms: np.ndarray
    actual_gross_return_bps: np.ndarray
    direction_probability: np.ndarray
    mean_prediction_bps: np.ndarray
    lower_prediction_bps: np.ndarray
    upper_prediction_bps: np.ndarray
    signal_threshold_bps: float

    @property
    def rows(self) -> int:
        return int(len(self.decision_time_ms))

    def metrics(self) -> TapeDepthForecastMetrics:
        return _evaluation_metrics(
            targets=self.actual_gross_return_bps,
            direction_probability=self.direction_probability,
            mean_prediction=self.mean_prediction_bps,
            lower_prediction=self.lower_prediction_bps,
            upper_prediction=self.upper_prediction_bps,
            signal_threshold_bps=self.signal_threshold_bps,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        arrays = (
            ("decision_time_ms", self.decision_time_ms, "<i8"),
            ("target_entry_time_ms", self.target_entry_time_ms, "<i8"),
            ("target_exit_time_ms", self.target_exit_time_ms, "<i8"),
            ("actual_gross_return_bps", self.actual_gross_return_bps, "<f8"),
            ("direction_probability", self.direction_probability, "<f8"),
            ("mean_prediction_bps", self.mean_prediction_bps, "<f8"),
            ("lower_prediction_bps", self.lower_prediction_bps, "<f8"),
            ("upper_prediction_bps", self.upper_prediction_bps, "<f8"),
            (
                "signal_threshold_bps",
                np.asarray([self.signal_threshold_bps], dtype=np.float64),
                "<f8",
            ),
        )
        for name, values, dtype in arrays:
            canonical = np.ascontiguousarray(np.asarray(values, dtype=dtype))
            digest.update(name.encode("ascii") + b"\x00")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()


def _model_parameters(model_profile: str, train_rows: int) -> dict[str, object]:
    if model_profile == "regularized":
        leaves, depth, min_leaf, l2 = 31, 6, 256, 0.10
    elif model_profile == "balanced":
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


def _selected_feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "full":
        return TAPE_DEPTH_FEATURE_NAMES
    names = tuple(
        name
        for name in TAPE_DEPTH_FEATURE_NAMES
        if not name.startswith(_DEPTH_PREFIXES)
    )
    if feature_set == "cross_asset":
        return names
    names = tuple(
        name for name in names if not name.startswith(_CROSS_ASSET_PREFIXES)
    )
    if feature_set == "core":
        names = tuple(
            name for name in names if not name.startswith(_DERIVED_PREFIXES)
        )
    return names


def _purged_segments(
    dataset: TapeDepthForecastDataset,
    *,
    minimum_segment_rows: int,
    split_boundaries_ms: tuple[int, int, int] | None = None,
) -> tuple[dict[str, np.ndarray], TapeDepthSplitEvidence]:
    rows = dataset.rows
    minimum = int(minimum_segment_rows)
    if minimum < 128:
        raise ValueError("minimum_segment_rows must be at least 128")
    if rows < minimum * 4 + 16:
        raise ValueError("tape/depth dataset is too small for four purged segments")
    times = np.asarray(dataset.decision_time_ms, dtype=np.int64)
    if split_boundaries_ms is None:
        boundaries = (int(rows * 0.60), int(rows * 0.75), int(rows * 0.85))
    else:
        requested = tuple(int(value) for value in split_boundaries_ms)
        if not requested[0] < requested[1] < requested[2]:
            raise ValueError("tape/depth split boundaries must increase")
        boundaries = tuple(
            int(np.searchsorted(times, value, side="left")) for value in requested
        )
        if any(
            index <= 0
            or index >= rows
            or int(times[index]) != requested[position]
            for position, index in enumerate(boundaries)
        ):
            raise ValueError("tape/depth split boundaries are absent from the dataset")
    train_raw = np.arange(0, boundaries[0], dtype=np.int64)
    tuning_raw = np.arange(boundaries[0], boundaries[1], dtype=np.int64)
    calibration_raw = np.arange(boundaries[1], boundaries[2], dtype=np.int64)
    evaluation = np.arange(boundaries[2], rows, dtype=np.int64)
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


def _weight_scale(targets: np.ndarray) -> float:
    values = np.abs(np.asarray(targets, dtype=np.float64))
    if len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("tape/depth weight calibration targets are invalid")
    return max(1.0, float(np.quantile(values, 0.90)))


def _weights(targets: np.ndarray, *, scale_bps: float | None = None) -> np.ndarray:
    values = np.abs(np.asarray(targets, dtype=np.float64))
    scale = _weight_scale(values) if scale_bps is None else float(scale_bps)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("tape/depth weight scale must be positive and finite")
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
    signal_threshold_bps: float,
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
    threshold = float(signal_threshold_bps)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tape/depth signal threshold must be non-negative and finite")
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
        calibration_threshold_rows=len(selected),
        calibration_threshold_signed_gross_bps=(
            float(np.sum(signed)) if len(signed) else 0.0
        ),
        calibration_threshold_mean_signed_gross_bps=(
            float(np.mean(signed)) if len(signed) else 0.0
        ),
        calibration_threshold_positive_rate=(
            float(np.mean(signed > 0.0)) if len(signed) else 0.0
        ),
    )


def train_tape_depth_forecaster(
    dataset: TapeDepthForecastDataset,
    *,
    risk_level: str = "conservative",
    model_profile: str = "regularized",
    feature_set: str = "full",
    compute_backend: str = "auto",
    seed: int = 20260710,
    minimum_segment_rows: int = 2_000,
    split_boundaries_ms: tuple[int, int, int] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> TapeDepthModelArtifact:
    """Train a research-only forecaster that cannot authorize execution."""

    risk = str(risk_level).strip().lower()
    if risk not in _RISK_LEVELS:
        raise ValueError("risk_level must be conservative, regular, or aggressive")
    profile = str(model_profile).strip().lower()
    if profile not in _MODEL_PROFILES:
        raise ValueError("model_profile must be regularized, balanced, or expressive")
    selected_feature_set = str(feature_set).strip().lower()
    if selected_feature_set not in _FEATURE_SETS:
        raise ValueError("feature_set must be core, tape_derived, cross_asset, or full")
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
        split_boundaries_ms=split_boundaries_ms,
    )
    model_feature_names = _selected_feature_names(selected_feature_set)
    model_feature_indexes = [
        dataset.feature_names.index(name) for name in model_feature_names
    ]
    x = np.asarray(dataset.features[:, model_feature_indexes], dtype=np.float32)
    exact_target = np.asarray(dataset.gross_return_bps, dtype=np.float64)
    target = exact_target.astype(np.float32)
    labels = (target > 0.0).astype(np.float32)
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        seed,
    )
    parameters = {**backend, **_model_parameters(profile, len(segments["train"]))}
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
    training_weight_scale_bps = _weight_scale(exact_target[train])
    economic_weights = _weights(
        exact_target,
        scale_bps=training_weight_scale_bps,
    )
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
    model_strings = {
        name: model.model_to_string(num_iteration=best_iterations[name])
        for name, model in models.items()
    }
    try:
        canonical_models = {
            name: lgb.Booster(model_str=model_string)
            for name, model_string in model_strings.items()
        }
    except LightGBMError as exc:
        raise ValueError("serialized tape/depth model could not be reloaded") from exc
    if progress:
        progress("calibrate-direction", completed, total_steps)
    raw_calibration = np.asarray(
        canonical_models["direction"].predict(x[calibration]),
        dtype=np.float64,
    )
    probability_calibration = _fit_platt_scaling(
        raw_calibration,
        labels[calibration],
    )
    calibration_mean_prediction = np.asarray(
        canonical_models["mean"].predict(x[calibration]),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(calibration_mean_prediction)):
        raise ValueError("tape/depth forecaster emitted non-finite calibration values")
    signal_threshold_bps = float(
        np.quantile(np.abs(calibration_mean_prediction), 0.90)
    )
    completed += 1
    if progress:
        progress("evaluate", completed, total_steps)
    probability = _apply_platt_scaling(
        np.asarray(canonical_models["direction"].predict(x[evaluation]), dtype=np.float64),
        probability_calibration,
    )
    mean_prediction = np.asarray(
        canonical_models["mean"].predict(x[evaluation]), dtype=np.float64
    )
    lower_prediction = np.asarray(
        canonical_models["lower"].predict(x[evaluation]), dtype=np.float64
    )
    upper_prediction = np.asarray(
        canonical_models["upper"].predict(x[evaluation]), dtype=np.float64
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (probability, mean_prediction, lower_prediction, upper_prediction)
    ):
        raise ValueError("tape/depth forecaster emitted non-finite evaluation values")
    metrics = _evaluation_metrics(
        targets=exact_target[evaluation],
        direction_probability=probability,
        mean_prediction=mean_prediction,
        lower_prediction=lower_prediction,
        upper_prediction=upper_prediction,
        signal_threshold_bps=signal_threshold_bps,
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
    if metrics.calibration_threshold_mean_signed_gross_bps <= 0.0:
        reasons.append("calibration_threshold_signed_gross_return_not_positive")
    if not 0.50 <= metrics.interval_80_coverage <= 0.98:
        reasons.append("quantile_interval_coverage_outside_sanity_band")
    if metrics.interval_crossing_rate > 0.05:
        reasons.append("quantile_interval_crossing_rate_too_high")
    if progress:
        progress("complete", total_steps, total_steps)
    dataset_fingerprint = tape_depth_dataset_fingerprint(dataset)
    dataset_summary = dataset.summary()
    if dataset_summary.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("tape/depth dataset summary fingerprint drifted")
    return TapeDepthModelArtifact(
        schema_version=TAPE_DEPTH_MODEL_SCHEMA_VERSION,
        model_family="lightgbm_direction_huber_quantile_ensemble",
        status="research_candidate" if not reasons else "rejected",
        rejection_reasons=tuple(reasons),
        trading_authority=False,
        execution_claim=False,
        symbol=dataset.symbol,
        risk_level=risk,
        model_profile=profile,
        feature_set=selected_feature_set,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        model_feature_names=model_feature_names,
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
        training_weight_scale_bps=training_weight_scale_bps,
        probability_calibration=probability_calibration,
        signal_threshold_bps=signal_threshold_bps,
        evaluation_metrics=metrics,
        model_strings=model_strings,
        dataset_fingerprint=dataset_fingerprint,
        dataset_summary=dataset_summary,
        trained_at=datetime.now(tz=UTC).isoformat(),
    )


def _metrics_match(
    expected: TapeDepthForecastMetrics,
    actual: TapeDepthForecastMetrics,
) -> bool:
    expected_values = asdict(expected)
    actual_values = asdict(actual)
    if expected_values.keys() != actual_values.keys():
        return False
    for name, expected_value in expected_values.items():
        actual_value = actual_values[name]
        if isinstance(expected_value, int):
            if int(actual_value) != expected_value:
                return False
        elif not math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            return False
    return True


def score_tape_depth_evaluation(
    artifact: TapeDepthModelArtifact,
    dataset: TapeDepthForecastDataset,
) -> TapeDepthPredictionBatch:
    """Reproduce the artifact's untouched evaluation segment exactly."""

    if artifact.schema_version != TAPE_DEPTH_MODEL_SCHEMA_VERSION:
        raise ValueError("tape/depth artifact schema is unsupported")
    if artifact.trading_authority or artifact.execution_claim:
        raise ValueError("gross tape/depth artifacts cannot carry execution authority")
    if (
        artifact.symbol != dataset.symbol
        or artifact.feature_version != dataset.feature_version
        or artifact.feature_names != dataset.feature_names
        or artifact.target_mode != dataset.target_mode
        or artifact.horizon_seconds != dataset.horizon_seconds
        or artifact.total_latency_ms != dataset.total_latency_ms
        or artifact.decision_cadence_seconds != dataset.decision_cadence_seconds
        or artifact.maximum_depth_age_ms != dataset.maximum_depth_age_ms
    ):
        raise ValueError("tape/depth artifact and dataset contracts differ")
    if artifact.feature_set not in _FEATURE_SETS:
        raise ValueError("tape/depth artifact feature set is unsupported")
    expected_model_features = _selected_feature_names(artifact.feature_set)
    if artifact.model_feature_names != expected_model_features:
        raise ValueError("tape/depth model feature names drifted")
    fingerprint = tape_depth_dataset_fingerprint(dataset)
    if artifact.dataset_fingerprint != fingerprint:
        raise ValueError("tape/depth dataset fingerprint differs from the artifact")
    segments, split = _purged_segments(
        dataset,
        minimum_segment_rows=128,
        split_boundaries_ms=(
            artifact.split.tuning_start_ms,
            artifact.split.calibration_start_ms,
            artifact.split.evaluation_start_ms,
        ),
    )
    if split != artifact.split:
        raise ValueError("tape/depth evaluation split differs from the artifact")
    expected_weight_scale = _weight_scale(
        np.asarray(dataset.gross_return_bps[segments["train"]], dtype=np.float64)
    )
    if not math.isclose(
        artifact.training_weight_scale_bps,
        expected_weight_scale,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("tape/depth training weight scale differs from the artifact")
    required_models = {"direction", "mean", "lower", "upper"}
    if set(artifact.model_strings) != required_models:
        raise ValueError("tape/depth artifact model ensemble is incomplete")
    try:
        models = {
            name: lgb.Booster(model_str=str(artifact.model_strings[name]))
            for name in sorted(required_models)
        }
    except (LightGBMError, TypeError, ValueError) as exc:
        raise ValueError("tape/depth artifact model strings are invalid") from exc
    evaluation = segments["evaluation"]
    feature_indexes = [dataset.feature_names.index(name) for name in expected_model_features]
    calibration_features = np.asarray(
        dataset.features[segments["calibration"]][:, feature_indexes],
        dtype=np.float32,
    )
    replayed_threshold = float(
        np.quantile(
            np.abs(
                np.asarray(
                    models["mean"].predict(calibration_features),
                    dtype=np.float64,
                )
            ),
            0.90,
        )
    )
    if not math.isclose(
        artifact.signal_threshold_bps,
        replayed_threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("tape/depth calibration signal threshold differs")
    features = np.asarray(
        dataset.features[evaluation][:, feature_indexes],
        dtype=np.float32,
    )
    probability = _apply_platt_scaling(
        np.asarray(models["direction"].predict(features), dtype=np.float64),
        artifact.probability_calibration,
    )
    batch = TapeDepthPredictionBatch(
        decision_time_ms=np.asarray(dataset.decision_time_ms[evaluation], dtype=np.int64),
        target_entry_time_ms=np.asarray(
            dataset.target_entry_time_ms[evaluation], dtype=np.int64
        ),
        target_exit_time_ms=np.asarray(
            dataset.target_exit_time_ms[evaluation], dtype=np.int64
        ),
        actual_gross_return_bps=np.asarray(
            dataset.gross_return_bps[evaluation], dtype=np.float64
        ),
        direction_probability=probability,
        mean_prediction_bps=np.asarray(
            models["mean"].predict(features), dtype=np.float64
        ),
        lower_prediction_bps=np.asarray(
            models["lower"].predict(features), dtype=np.float64
        ),
        upper_prediction_bps=np.asarray(
            models["upper"].predict(features), dtype=np.float64
        ),
        signal_threshold_bps=artifact.signal_threshold_bps,
    )
    predicted_arrays = (
        batch.direction_probability,
        batch.mean_prediction_bps,
        batch.lower_prediction_bps,
        batch.upper_prediction_bps,
    )
    if not all(np.all(np.isfinite(values)) for values in predicted_arrays):
        raise ValueError("tape/depth replay emitted non-finite predictions")
    if not _metrics_match(artifact.evaluation_metrics, batch.metrics()):
        raise ValueError("tape/depth replay metrics differ from the artifact")
    return batch


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
    if payload.get("model_profile") not in _MODEL_PROFILES:
        raise ValueError("tape/depth model profile is unsupported")
    feature_set = str(payload.get("feature_set") or "")
    if feature_set not in _FEATURE_SETS:
        raise ValueError("tape/depth feature set is unsupported")
    model_feature_names = tuple(payload.get("model_feature_names") or ())
    if model_feature_names != _selected_feature_names(feature_set):
        raise ValueError("tape/depth model feature names drifted")
    if payload.get("trading_authority") is not False or payload.get("execution_claim") is not False:
        raise ValueError("tape/depth gross forecast cannot authorize trading")
    dataset_fingerprint = str(payload.get("dataset_fingerprint") or "").lower()
    if len(dataset_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_fingerprint
    ):
        raise ValueError("tape/depth dataset fingerprint is invalid")
    dataset_summary = payload.get("dataset_summary")
    if (
        not isinstance(dataset_summary, Mapping)
        or dataset_summary.get("dataset_fingerprint") != dataset_fingerprint
    ):
        raise ValueError("tape/depth dataset summary binding is invalid")
    try:
        training_weight_scale_bps = float(payload["training_weight_scale_bps"])
        signal_threshold_bps = float(payload["signal_threshold_bps"])
        probability_calibration = tuple(
            float(value) for value in payload["probability_calibration"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tape/depth calibration fields are incomplete") from exc
    if (
        not math.isfinite(training_weight_scale_bps)
        or training_weight_scale_bps <= 0.0
        or not math.isfinite(signal_threshold_bps)
        or signal_threshold_bps < 0.0
        or len(probability_calibration) != 2
        or not all(math.isfinite(value) for value in probability_calibration)
    ):
        raise ValueError("tape/depth calibration fields are invalid")
    try:
        values = dict(payload)
        values["rejection_reasons"] = tuple(payload.get("rejection_reasons") or ())
        values["feature_names"] = tuple(payload.get("feature_names") or ())
        values["model_feature_names"] = model_feature_names
        values["split"] = TapeDepthSplitEvidence(**dict(payload["split"]))
        values["evaluation_metrics"] = TapeDepthForecastMetrics(
            **dict(payload["evaluation_metrics"])
        )
        values["training_weight_scale_bps"] = training_weight_scale_bps
        values["probability_calibration"] = probability_calibration
        values["signal_threshold_bps"] = signal_threshold_bps
        return TapeDepthModelArtifact(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tape/depth artifact fields are incomplete") from exc


__all__ = [
    "TAPE_DEPTH_MODEL_SCHEMA_VERSION",
    "TapeDepthForecastMetrics",
    "TapeDepthModelArtifact",
    "TapeDepthPredictionBatch",
    "TapeDepthSplitEvidence",
    "load_tape_depth_model_artifact",
    "save_tape_depth_model_artifact",
    "score_tape_depth_evaluation",
    "train_tape_depth_forecaster",
]
