"""Fixed-model training and cost-aware replay for the Round 37 lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .cross_asset_cost_data import (
    CrossAssetDataset,
    HORIZONS_MINUTES,
    MINUTE_MS,
    SYMBOLS,
    role_by_name,
)
from .lightgbm_backend import lightgbm_backend_parameters


EXECUTION_CHARGE_BPS = 12.0
THRESHOLD_GRID_BPS = (12.0, 15.0, 18.0, 24.0, 30.0)
SEED = 3701


@dataclass(frozen=True)
class ModelArtifactEvidence:
    model_id: str
    family: str
    horizon_minutes: int
    symbol: str
    feature_count: int
    training_rows: int
    early_stop_rows: int
    best_iteration: int
    backend_kind: str
    backend_device: str
    path: str
    bytes: int
    sha256: str
    reload_max_abs_prediction_error: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionMetrics:
    rows: int
    mean_actual_bps: float
    mean_prediction_bps: float
    mae_bps: float
    rmse_bps: float
    pearson_information_coefficient: float
    spearman_information_coefficient: float
    direction_accuracy: float
    prediction_standard_deviation_bps: float
    actual_standard_deviation_bps: float
    nonfinite_predictions: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayMetrics:
    threshold_bps: float | None
    total_trades: int
    trades_by_symbol: Mapping[str, int]
    active_utc_days: int
    total_net_bps: float
    mean_net_bps: float
    median_net_bps: float
    positive_rate: float
    profit_factor: float | None
    median_monthly_net_bps: float
    negative_month_fraction: float
    day_block_bootstrap_mean_net_bps_lower_95: float | None
    day_block_bootstrap_mean_net_bps_median: float | None
    day_block_bootstrap_mean_net_bps_upper_95: float | None
    candidate_rows: int
    overlap_rejections: int
    nonfinite_outcomes: int

    def asdict(self) -> dict[str, object]:
        return {**asdict(self), "trades_by_symbol": dict(self.trades_by_symbol)}


@dataclass(frozen=True)
class ThresholdTrace:
    threshold_bps: float
    support_passed: bool
    replay: ReplayMetrics

    def asdict(self) -> dict[str, object]:
        return {
            "threshold_bps": self.threshold_bps,
            "support_passed": self.support_passed,
            "replay": self.replay.asdict(),
        }


@dataclass(frozen=True)
class CandidateResult:
    family: str
    horizon_minutes: int
    amplitude_slopes: Mapping[str, float]
    calibration_prediction_metrics: PredictionMetrics
    viability_prediction_metrics: PredictionMetrics
    threshold_trace: tuple[ThresholdTrace, ...]
    selected_threshold_bps: float | None
    calibration_replay: ReplayMetrics
    viability_replay: ReplayMetrics
    viability_gate_passed: bool
    viability_gate_reasons: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "amplitude_slopes": dict(self.amplitude_slopes),
            "calibration_prediction_metrics": self.calibration_prediction_metrics.asdict(),
            "viability_prediction_metrics": self.viability_prediction_metrics.asdict(),
            "threshold_trace": [item.asdict() for item in self.threshold_trace],
            "calibration_replay": self.calibration_replay.asdict(),
            "viability_replay": self.viability_replay.asdict(),
            "viability_gate_reasons": list(self.viability_gate_reasons),
        }


@dataclass(frozen=True)
class TrainedCandidates:
    predictions: Mapping[tuple[str, int], np.ndarray]
    model_artifacts: tuple[ModelArtifactEvidence, ...]
    backend_kind: str
    backend_device: str


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size != left.size:
        return 0.0
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def prediction_metrics(actual: np.ndarray, prediction: np.ndarray) -> PredictionMetrics:
    finite = np.isfinite(actual) & np.isfinite(prediction)
    nonfinite = int(np.count_nonzero(~np.isfinite(prediction)))
    actual_finite = actual[finite].astype(np.float64, copy=False)
    prediction_finite = prediction[finite].astype(np.float64, copy=False)
    if actual_finite.size == 0:
        return PredictionMetrics(
            rows=0,
            mean_actual_bps=0.0,
            mean_prediction_bps=0.0,
            mae_bps=0.0,
            rmse_bps=0.0,
            pearson_information_coefficient=0.0,
            spearman_information_coefficient=0.0,
            direction_accuracy=0.0,
            prediction_standard_deviation_bps=0.0,
            actual_standard_deviation_bps=0.0,
            nonfinite_predictions=nonfinite,
        )
    residual = prediction_finite - actual_finite
    nonzero = actual_finite != 0.0
    direction_accuracy = (
        float(np.mean(np.sign(prediction_finite[nonzero]) == np.sign(actual_finite[nonzero])))
        if np.any(nonzero)
        else 0.0
    )
    return PredictionMetrics(
        rows=int(actual_finite.size),
        mean_actual_bps=float(np.mean(actual_finite)),
        mean_prediction_bps=float(np.mean(prediction_finite)),
        mae_bps=float(np.mean(np.abs(residual))),
        rmse_bps=float(np.sqrt(np.mean(residual * residual))),
        pearson_information_coefficient=_correlation(prediction_finite, actual_finite),
        spearman_information_coefficient=_correlation(
            _rankdata(prediction_finite),
            _rankdata(actual_finite),
        ),
        direction_accuracy=direction_accuracy,
        prediction_standard_deviation_bps=float(np.std(prediction_finite)),
        actual_standard_deviation_bps=float(np.std(actual_finite)),
        nonfinite_predictions=nonfinite,
    )


def _lightgbm_parameters(
    compute_backend: str,
    seed: int,
) -> tuple[dict[str, object], str, str]:
    backend, kind, device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
    )
    backend.update(
        {
            "objective": "regression_l1",
            "metric": "l1",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": 1000,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "max_bin": 255,
            "feature_pre_filter": False,
        }
    )
    return backend, kind, device


def _fit_lightgbm(
    dataset: CrossAssetDataset,
    *,
    horizon: int,
    symbol_index: int | None,
    model_id: str,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, ModelArtifactEvidence, str, str]:
    role_masks = dataset.role_masks[horizon]
    training = role_masks["training"].copy()
    early_stop = role_masks["early_stop"].copy()
    symbol = "shared"
    family = "shared_cross_asset_lightgbm"
    if symbol_index is not None:
        training &= dataset.symbol_index == symbol_index
        early_stop &= dataset.symbol_index == symbol_index
        symbol = SYMBOLS[symbol_index]
        family = "per_symbol_lightgbm"
    parameters, backend_kind, backend_device = _lightgbm_parameters(
        compute_backend,
        seed,
    )
    train_rows = int(np.count_nonzero(training))
    early_stop_rows = int(np.count_nonzero(early_stop))
    if train_rows == 0 or early_stop_rows == 0:
        raise ValueError(f"{model_id} has empty training or early-stop role")
    if progress is not None:
        progress(
            "model_training",
            {
                "model_id": model_id,
                "status": "started",
                "training_rows": train_rows,
                "early_stop_rows": early_stop_rows,
                "backend_kind": backend_kind,
                "backend_device": backend_device,
            },
        )
    train_set = lgb.Dataset(
        dataset.features[training],
        label=dataset.gross_return_bps[horizon][training],
        feature_name=list(dataset.feature_names),
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        dataset.features[early_stop],
        label=dataset.gross_return_bps[horizon][early_stop],
        feature_name=list(dataset.feature_names),
        reference=train_set,
        free_raw_data=True,
    )
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=1000,
        valid_sets=[validation_set],
        valid_names=["early_stop"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    prediction = np.full(dataset.rows, np.nan, dtype=np.float32)
    prediction_mask = np.ones(dataset.rows, dtype=bool)
    if symbol_index is not None:
        prediction_mask = dataset.symbol_index == symbol_index
    predicted = booster.predict(
        dataset.features[prediction_mask],
        num_iteration=booster.best_iteration,
    )
    prediction[prediction_mask] = np.asarray(predicted, dtype=np.float32)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    reloaded = lgb.Booster(model_file=str(model_path))
    reload_probe_mask = np.flatnonzero(early_stop)[:4096]
    original_probe = booster.predict(
        dataset.features[reload_probe_mask],
        num_iteration=booster.best_iteration,
    )
    reload_probe = reloaded.predict(
        dataset.features[reload_probe_mask],
        num_iteration=reloaded.best_iteration,
    )
    reload_error = float(
        np.max(np.abs(np.asarray(original_probe) - np.asarray(reload_probe)))
    )
    if not math.isfinite(reload_error) or reload_error > 1e-12:
        raise RuntimeError(f"{model_id} artifact reload error is {reload_error}")
    artifact = ModelArtifactEvidence(
        model_id=model_id,
        family=family,
        horizon_minutes=horizon,
        symbol=symbol,
        feature_count=len(dataset.feature_names),
        training_rows=train_rows,
        early_stop_rows=early_stop_rows,
        best_iteration=int(booster.best_iteration),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(model_path),
        bytes=model_path.stat().st_size,
        sha256=_file_sha256(model_path),
        reload_max_abs_prediction_error=reload_error,
    )
    if progress is not None:
        progress(
            "model_training",
            {
                "model_id": model_id,
                "status": "complete",
                "best_iteration": artifact.best_iteration,
                "artifact_sha256": artifact.sha256,
            },
        )
    return prediction, artifact, backend_kind, backend_device


def _fit_ridge_prediction(
    dataset: CrossAssetDataset,
    *,
    horizon: int,
    ridge: float = 10.0,
) -> np.ndarray:
    training = dataset.role_masks[horizon]["training"]
    x_train = dataset.features[training].astype(np.float64)
    y_train = dataset.gross_return_bps[horizon][training].astype(np.float64)
    means = np.mean(x_train, axis=0)
    standard_deviations = np.std(x_train, axis=0)
    standard_deviations[standard_deviations < 1e-8] = 1.0
    normalized = (x_train - means) / standard_deviations
    gram = normalized.T @ normalized
    gram.flat[:: gram.shape[0] + 1] += float(ridge)
    weights = np.linalg.solve(gram, normalized.T @ y_train)
    intercept = float(np.mean(y_train))
    prediction = (
        (dataset.features.astype(np.float64) - means) / standard_deviations
    ) @ weights + intercept
    return prediction.astype(np.float32)


def train_fixed_candidates(
    dataset: CrossAssetDataset,
    *,
    model_dir: str | Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> TrainedCandidates:
    """Train every prespecified model without viability-role model selection."""

    root = Path(model_dir)
    predictions: dict[tuple[str, int], np.ndarray] = {}
    artifacts: list[ModelArtifactEvidence] = []
    backend_kinds: set[str] = set()
    backend_devices: set[str] = set()
    for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
        shared, artifact, kind, device = _fit_lightgbm(
            dataset,
            horizon=horizon,
            symbol_index=None,
            model_id=f"shared_h{horizon}",
            model_dir=root,
            compute_backend=compute_backend,
            seed=SEED + horizon_index * 10,
            progress=progress,
        )
        predictions[("shared_cross_asset_lightgbm", horizon)] = shared
        artifacts.append(artifact)
        backend_kinds.add(kind)
        backend_devices.add(device)
        combined = np.full(dataset.rows, np.nan, dtype=np.float32)
        for symbol_index, symbol in enumerate(SYMBOLS):
            current, artifact, kind, device = _fit_lightgbm(
                dataset,
                horizon=horizon,
                symbol_index=symbol_index,
                model_id=f"per_symbol_{symbol.lower()}_h{horizon}",
                model_dir=root,
                compute_backend=compute_backend,
                seed=SEED + horizon_index * 10 + symbol_index + 1,
                progress=progress,
            )
            mask = dataset.symbol_index == symbol_index
            combined[mask] = current[mask]
            artifacts.append(artifact)
            backend_kinds.add(kind)
            backend_devices.add(device)
        predictions[("per_symbol_lightgbm", horizon)] = combined
        if progress is not None:
            progress(
                "baseline_fit",
                {"family": "linear_ridge", "horizon_minutes": horizon},
            )
        predictions[("linear_ridge", horizon)] = _fit_ridge_prediction(
            dataset,
            horizon=horizon,
        )
        predictions[("persistence", horizon)] = dataset.persistence_prediction_bps[
            horizon
        ].copy()
        predictions[("zero_return", horizon)] = np.zeros(dataset.rows, dtype=np.float32)
    if len(backend_kinds) != 1 or len(backend_devices) != 1:
        raise RuntimeError(
            f"inconsistent LightGBM backends: {backend_kinds} {backend_devices}"
        )
    return TrainedCandidates(
        predictions=predictions,
        model_artifacts=tuple(artifacts),
        backend_kind=next(iter(backend_kinds)),
        backend_device=next(iter(backend_devices)),
    )


def _amplitude_calibrate(
    dataset: CrossAssetDataset,
    prediction: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, dict[str, float]]:
    calibrated = np.full(prediction.size, np.nan, dtype=np.float32)
    slopes: dict[str, float] = {}
    calibration = dataset.role_masks[horizon]["calibration"]
    actual = dataset.gross_return_bps[horizon]
    for symbol_index, symbol in enumerate(SYMBOLS):
        fit_mask = (
            calibration
            & (dataset.symbol_index == symbol_index)
            & np.isfinite(prediction)
            & np.isfinite(actual)
        )
        numerator = float(np.dot(prediction[fit_mask], actual[fit_mask]))
        denominator = float(np.dot(prediction[fit_mask], prediction[fit_mask]))
        slope = numerator / denominator if denominator > 1e-12 else 0.0
        slope = float(np.clip(slope, 0.0, 3.0))
        slopes[symbol] = slope
        symbol_mask = dataset.symbol_index == symbol_index
        calibrated[symbol_mask] = prediction[symbol_mask] * slope
    return calibrated, slopes


def _empty_replay(threshold_bps: float | None) -> ReplayMetrics:
    return ReplayMetrics(
        threshold_bps=threshold_bps,
        total_trades=0,
        trades_by_symbol={symbol: 0 for symbol in SYMBOLS},
        active_utc_days=0,
        total_net_bps=0.0,
        mean_net_bps=0.0,
        median_net_bps=0.0,
        positive_rate=0.0,
        profit_factor=0.0,
        median_monthly_net_bps=0.0,
        negative_month_fraction=1.0,
        day_block_bootstrap_mean_net_bps_lower_95=None,
        day_block_bootstrap_mean_net_bps_median=None,
        day_block_bootstrap_mean_net_bps_upper_95=None,
        candidate_rows=0,
        overlap_rejections=0,
        nonfinite_outcomes=0,
    )


def _stationary_bootstrap_mean_net(
    daily_net: np.ndarray,
    daily_trades: np.ndarray,
    *,
    samples: int,
    mean_block_length: int,
    seed: int,
) -> tuple[float, float, float] | None:
    days = daily_net.size
    if days == 0 or int(np.sum(daily_trades)) == 0:
        return None
    generator = np.random.default_rng(seed)
    results = np.empty(samples, dtype=np.float64)
    restart_probability = 1.0 / float(mean_block_length)
    for sample in range(samples):
        index = int(generator.integers(0, days))
        total_net = 0.0
        total_trades = 0
        for _ in range(days):
            total_net += float(daily_net[index])
            total_trades += int(daily_trades[index])
            if generator.random() < restart_probability:
                index = int(generator.integers(0, days))
            else:
                index = (index + 1) % days
        results[sample] = total_net / total_trades if total_trades else 0.0
    return (
        float(np.quantile(results, 0.025)),
        float(np.quantile(results, 0.5)),
        float(np.quantile(results, 0.975)),
    )


def replay_nonoverlapping(
    dataset: CrossAssetDataset,
    calibrated_prediction: np.ndarray,
    *,
    horizon: int,
    role: str,
    threshold_bps: float | None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 3711,
) -> ReplayMetrics:
    if threshold_bps is None:
        return _empty_replay(None)
    role_mask = dataset.role_masks[horizon][role]
    actual = dataset.gross_return_bps[horizon]
    candidates = (
        role_mask
        & np.isfinite(calibrated_prediction)
        & np.isfinite(actual)
        & (np.abs(calibrated_prediction) >= threshold_bps)
    )
    selected: list[int] = []
    overlap_rejections = 0
    trades_by_symbol: dict[str, int] = {}
    for symbol_index, symbol in enumerate(SYMBOLS):
        indices = np.flatnonzero(candidates & (dataset.symbol_index == symbol_index))
        indices = indices[np.argsort(dataset.decision_time_ms[indices], kind="stable")]
        next_available_ms = -1
        symbol_selected: list[int] = []
        for index in indices:
            entry_time = int(dataset.decision_time_ms[index]) + MINUTE_MS
            if entry_time < next_available_ms:
                overlap_rejections += 1
                continue
            symbol_selected.append(int(index))
            next_available_ms = entry_time + horizon * MINUTE_MS
        selected.extend(symbol_selected)
        trades_by_symbol[symbol] = len(symbol_selected)
    if not selected:
        empty = _empty_replay(threshold_bps)
        return ReplayMetrics(
            **{
                **empty.asdict(),
                "candidate_rows": int(np.count_nonzero(candidates)),
                "overlap_rejections": overlap_rejections,
            }
        )
    indices = np.asarray(selected, dtype=np.int64)
    indices = indices[np.argsort(dataset.decision_time_ms[indices], kind="stable")]
    direction = np.sign(calibrated_prediction[indices]).astype(np.float64)
    net = direction * actual[indices].astype(np.float64) - EXECUTION_CHARGE_BPS
    nonfinite = int(np.count_nonzero(~np.isfinite(net)))
    if nonfinite:
        raise ValueError(f"replay produced {nonfinite} non-finite outcomes")
    positive_sum = float(np.sum(net[net > 0.0]))
    negative_sum = float(np.sum(net[net < 0.0]))
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0.0 else None
    role_spec = role_by_name(role)
    first_day = datetime.fromisoformat(role_spec.start).replace(tzinfo=UTC)
    last_day = datetime.fromisoformat(role_spec.end).replace(tzinfo=UTC)
    day_count = (last_day - first_day).days + 1
    daily_net = np.zeros(day_count, dtype=np.float64)
    daily_trades = np.zeros(day_count, dtype=np.int64)
    month_totals: dict[str, float] = {}
    for index, outcome in zip(indices, net, strict=True):
        timestamp = datetime.fromtimestamp(
            int(dataset.decision_time_ms[index]) / 1000.0,
            UTC,
        )
        day_index = (timestamp.date() - first_day.date()).days
        daily_net[day_index] += outcome
        daily_trades[day_index] += 1
        month = f"{timestamp.year:04d}-{timestamp.month:02d}"
        month_totals[month] = month_totals.get(month, 0.0) + float(outcome)
    months = np.asarray(list(month_totals.values()), dtype=np.float64)
    bootstrap = _stationary_bootstrap_mean_net(
        daily_net,
        daily_trades,
        samples=bootstrap_samples,
        mean_block_length=5,
        seed=bootstrap_seed,
    )
    return ReplayMetrics(
        threshold_bps=float(threshold_bps),
        total_trades=int(indices.size),
        trades_by_symbol=trades_by_symbol,
        active_utc_days=int(np.count_nonzero(daily_trades)),
        total_net_bps=float(np.sum(net)),
        mean_net_bps=float(np.mean(net)),
        median_net_bps=float(np.median(net)),
        positive_rate=float(np.mean(net > 0.0)),
        profit_factor=profit_factor,
        median_monthly_net_bps=float(np.median(months)) if months.size else 0.0,
        negative_month_fraction=float(np.mean(months < 0.0)) if months.size else 1.0,
        day_block_bootstrap_mean_net_bps_lower_95=(bootstrap[0] if bootstrap else None),
        day_block_bootstrap_mean_net_bps_median=(bootstrap[1] if bootstrap else None),
        day_block_bootstrap_mean_net_bps_upper_95=(bootstrap[2] if bootstrap else None),
        candidate_rows=int(np.count_nonzero(candidates)),
        overlap_rejections=overlap_rejections,
        nonfinite_outcomes=nonfinite,
    )


def _select_threshold(
    traces: Sequence[ThresholdTrace],
) -> ThresholdTrace | None:
    eligible = [item for item in traces if item.support_passed]
    if not eligible:
        return None

    def key(item: ThresholdTrace) -> tuple[float, float, float, float]:
        lower = item.replay.day_block_bootstrap_mean_net_bps_lower_95
        return (
            float(lower) if lower is not None else float("-inf"),
            item.replay.median_monthly_net_bps,
            item.replay.mean_net_bps,
            item.threshold_bps,
        )

    return max(eligible, key=key)


def evaluate_candidates(
    dataset: CrossAssetDataset,
    trained: TrainedCandidates,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[CandidateResult, ...]:
    results: list[CandidateResult] = []
    for (family, horizon), prediction in trained.predictions.items():
        calibrated, slopes = _amplitude_calibrate(
            dataset,
            prediction,
            horizon=horizon,
        )
        calibration_mask = dataset.role_masks[horizon]["calibration"]
        viability_mask = dataset.role_masks[horizon]["viability"]
        actual = dataset.gross_return_bps[horizon]
        traces: list[ThresholdTrace] = []
        for threshold in THRESHOLD_GRID_BPS:
            replay = replay_nonoverlapping(
                dataset,
                calibrated,
                horizon=horizon,
                role="calibration",
                threshold_bps=threshold,
                bootstrap_samples=1000,
                bootstrap_seed=3711,
            )
            support = replay.total_trades >= 90 and all(
                replay.trades_by_symbol.get(symbol, 0) >= 15 for symbol in SYMBOLS
            )
            traces.append(
                ThresholdTrace(
                    threshold_bps=threshold,
                    support_passed=support,
                    replay=replay,
                )
            )
        selected = _select_threshold(traces)
        threshold = selected.threshold_bps if selected else None
        calibration_replay = (
            selected.replay if selected is not None else _empty_replay(None)
        )
        viability_replay = replay_nonoverlapping(
            dataset,
            calibrated,
            horizon=horizon,
            role="viability",
            threshold_bps=threshold,
            bootstrap_samples=2000,
            bootstrap_seed=3712,
        )
        reasons: list[str] = []
        if threshold is None:
            reasons.append("no_calibration_threshold_met_support")
        if viability_replay.total_trades < 180:
            reasons.append("viability_total_trades<180")
        for symbol in SYMBOLS:
            if viability_replay.trades_by_symbol.get(symbol, 0) < 30:
                reasons.append(f"{symbol}_viability_trades<30")
        if viability_replay.active_utc_days < 90:
            reasons.append("viability_active_utc_days<90")
        if viability_replay.mean_net_bps <= 0.0:
            reasons.append("viability_mean_net_bps<=0")
        if viability_replay.median_monthly_net_bps <= 0.0:
            reasons.append("viability_median_monthly_net_bps<=0")
        if (
            viability_replay.profit_factor is not None
            and viability_replay.profit_factor < 1.05
        ):
            reasons.append("viability_profit_factor<1.05")
        if viability_replay.negative_month_fraction > 0.45:
            reasons.append("viability_negative_month_fraction>0.45")
        lower = viability_replay.day_block_bootstrap_mean_net_bps_lower_95
        if lower is None or lower <= 0.0:
            reasons.append("viability_day_block_lower_95<=0")
        calibration_metrics = prediction_metrics(
            actual[calibration_mask],
            calibrated[calibration_mask],
        )
        viability_metrics = prediction_metrics(
            actual[viability_mask],
            calibrated[viability_mask],
        )
        if viability_metrics.nonfinite_predictions:
            reasons.append("viability_nonfinite_predictions")
        result = CandidateResult(
            family=family,
            horizon_minutes=horizon,
            amplitude_slopes=slopes,
            calibration_prediction_metrics=calibration_metrics,
            viability_prediction_metrics=viability_metrics,
            threshold_trace=tuple(traces),
            selected_threshold_bps=threshold,
            calibration_replay=calibration_replay,
            viability_replay=viability_replay,
            viability_gate_passed=not reasons,
            viability_gate_reasons=tuple(dict.fromkeys(reasons)),
        )
        results.append(result)
        if progress is not None:
            progress(
                "candidate_evaluation",
                {
                    "family": family,
                    "horizon_minutes": horizon,
                    "selected_threshold_bps": threshold,
                    "viability_trades": viability_replay.total_trades,
                    "viability_mean_net_bps": viability_replay.mean_net_bps,
                    "viability_gate_passed": result.viability_gate_passed,
                },
            )
    return tuple(
        sorted(results, key=lambda item: (item.family, item.horizon_minutes))
    )


def calibrated_predictions(
    dataset: CrossAssetDataset,
    trained: TrainedCandidates,
    *,
    family: str,
    horizon: int,
) -> tuple[np.ndarray, Mapping[str, float]]:
    prediction = trained.predictions[(family, horizon)]
    return _amplitude_calibrate(dataset, prediction, horizon=horizon)


__all__ = [
    "CandidateResult",
    "EXECUTION_CHARGE_BPS",
    "ModelArtifactEvidence",
    "PredictionMetrics",
    "ReplayMetrics",
    "THRESHOLD_GRID_BPS",
    "ThresholdTrace",
    "TrainedCandidates",
    "calibrated_predictions",
    "evaluate_candidates",
    "prediction_metrics",
    "replay_nonoverlapping",
    "train_fixed_candidates",
]
