"""GPU LightGBM direct-action and two-stage hurdle models for Round 38."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from .cross_asset_cost_data import HORIZONS_MINUTES, MINUTE_MS, SYMBOLS, role_by_name
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .lightgbm_backend import lightgbm_backend_parameters


ARCHITECTURES = (
    "shared_direct_multiclass",
    "per_symbol_direct_multiclass",
    "shared_two_stage_hurdle",
    "per_symbol_two_stage_hurdle",
)
FEATURE_SETS = ("price_flow_only", "price_flow_plus_premium_and_funding")
ACTION_PROBABILITY_GRID = (0.40, 0.45, 0.50, 0.55, 0.60)
DIRECTION_MARGIN_GRID = (0.05, 0.10, 0.15, 0.20)
SEED = 3801


@dataclass(frozen=True)
class ProbabilityModelArtifact:
    model_id: str
    candidate_id: str
    target_head: str
    architecture: str
    feature_set: str
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
    top_feature_gain: tuple[tuple[str, float], ...]

    def asdict(self) -> dict[str, object]:
        value = asdict(self)
        value["top_feature_gain"] = [
            {"feature": feature, "gain": gain}
            for feature, gain in self.top_feature_gain
        ]
        return value


@dataclass(frozen=True)
class ClassificationMetrics:
    rows: int
    class_counts: tuple[int, int, int]
    multiclass_log_loss: float
    multiclass_brier_score: float
    expected_calibration_error: float
    accuracy: float
    balanced_accuracy: float
    confusion_matrix: tuple[tuple[int, int, int], ...]
    mean_probabilities: tuple[float, float, float]
    maximum_probability_mean: float
    nonfinite_probabilities: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionReplayMetrics:
    maximum_action_probability: float | None
    direction_probability_margin: float | None
    total_trades: int
    trades_by_symbol: Mapping[str, int]
    maximum_single_symbol_fraction: float
    active_utc_days: int
    total_net_bps: float
    mean_net_bps: float
    median_net_bps: float
    positive_rate: float
    profit_factor: float | None
    median_monthly_net_bps: float
    negative_month_fraction: float
    maximum_peak_to_trough_drawdown_bps: float
    longest_loss_streak: int
    total_funding_cash_flow_bps: float
    mean_funding_cash_flow_bps: float
    day_block_bootstrap_mean_net_bps_lower_95: float | None
    day_block_bootstrap_mean_net_bps_median: float | None
    day_block_bootstrap_mean_net_bps_upper_95: float | None
    candidate_rows: int
    overlap_rejections: int
    nonfinite_outcomes: int

    def asdict(self) -> dict[str, object]:
        return {**asdict(self), "trades_by_symbol": dict(self.trades_by_symbol)}


@dataclass(frozen=True)
class ReplayOutcome:
    metrics: ActionReplayMetrics
    selected_indices: np.ndarray
    selected_direction: np.ndarray
    net_return_bps: np.ndarray


@dataclass(frozen=True)
class PassedCandidate:
    candidate_id: str
    architecture: str
    feature_set: str
    horizon_minutes: int
    maximum_action_probability: float
    direction_probability_margin: float
    probabilities: np.ndarray
    viability: ReplayOutcome


@dataclass(frozen=True)
class Round38ModelScreen:
    candidate_results: tuple[Mapping[str, object], ...]
    model_artifacts: tuple[ProbabilityModelArtifact, ...]
    passed_candidates: tuple[PassedCandidate, ...]
    backend_kind: str
    backend_device: str


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lightgbm_parameters(
    compute_backend: str,
    *,
    objective: str,
    seed: int,
) -> tuple[dict[str, object], str, str]:
    parameters, kind, device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
    )
    parameters.update(
        {
            "objective": objective,
            "metric": "multi_logloss" if objective == "multiclass" else "binary_logloss",
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
    if objective == "multiclass":
        parameters["num_class"] = 3
    return parameters, kind, device


def _fit_booster(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    architecture: str,
    feature_set: str,
    horizon: int,
    symbol_index: int | None,
    target_head: str,
    objective: str,
    labels: np.ndarray,
    eligible_label_mask: np.ndarray,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, ProbabilityModelArtifact, str, str]:
    features = dataset.feature_view(feature_set)
    training = dataset.role_masks[horizon]["training"] & eligible_label_mask
    early_stop = dataset.role_masks[horizon]["early_stop"] & eligible_label_mask
    prediction_mask = np.ones(dataset.rows, dtype=bool)
    symbol = "shared"
    if symbol_index is not None:
        symbol_mask = dataset.symbol_index == symbol_index
        training &= symbol_mask
        early_stop &= symbol_mask
        prediction_mask = symbol_mask
        symbol = SYMBOLS[symbol_index]
    training_rows = int(np.count_nonzero(training))
    early_stop_rows = int(np.count_nonzero(early_stop))
    if training_rows == 0 or early_stop_rows == 0:
        raise ValueError(f"{candidate_id}/{target_head}/{symbol} has an empty role")
    parameters, backend_kind, backend_device = _lightgbm_parameters(
        compute_backend,
        objective=objective,
        seed=seed,
    )
    model_id = f"{candidate_id}_{target_head}_{symbol.lower()}"
    if progress is not None:
        progress(
            "round38_model_training",
            {
                "model_id": model_id,
                "status": "started",
                "training_rows": training_rows,
                "early_stop_rows": early_stop_rows,
                "backend_kind": backend_kind,
                "backend_device": backend_device,
            },
        )
    feature_names = list(dataset.feature_names[: features.shape[1]])
    train_set = lgb.Dataset(
        features[training],
        label=labels[training],
        feature_name=feature_names,
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        features[early_stop],
        label=labels[early_stop],
        feature_name=feature_names,
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
    raw_prediction = booster.predict(
        features[prediction_mask], num_iteration=booster.best_iteration
    )
    if objective == "multiclass":
        prediction = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
        prediction[prediction_mask] = np.asarray(raw_prediction, dtype=np.float32)
    else:
        prediction = np.full(dataset.rows, np.nan, dtype=np.float32)
        prediction[prediction_mask] = np.asarray(raw_prediction, dtype=np.float32)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    reloaded = lgb.Booster(model_file=str(model_path))
    probe = np.flatnonzero(early_stop)[:4096]
    original_probe = np.asarray(
        booster.predict(features[probe], num_iteration=booster.best_iteration)
    )
    reload_probe = np.asarray(reloaded.predict(features[probe]))
    reload_error = float(np.max(np.abs(original_probe - reload_probe)))
    if not math.isfinite(reload_error) or reload_error > 1e-12:
        raise RuntimeError(f"{model_id} reload error is {reload_error}")
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:20]
    top_gain = tuple(
        (feature_names[int(index)], float(gain[int(index)])) for index in order
    )
    artifact = ProbabilityModelArtifact(
        model_id=model_id,
        candidate_id=candidate_id,
        target_head=target_head,
        architecture=architecture,
        feature_set=feature_set,
        horizon_minutes=horizon,
        symbol=symbol,
        feature_count=features.shape[1],
        training_rows=training_rows,
        early_stop_rows=early_stop_rows,
        best_iteration=int(booster.best_iteration),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(model_path),
        bytes=model_path.stat().st_size,
        sha256=_file_sha256(model_path),
        reload_max_abs_prediction_error=reload_error,
        top_feature_gain=top_gain,
    )
    if progress is not None:
        progress(
            "round38_model_training",
            {
                "model_id": model_id,
                "status": "complete",
                "best_iteration": artifact.best_iteration,
                "artifact_sha256": artifact.sha256,
            },
        )
    return prediction, artifact, backend_kind, backend_device


def _fit_direct_candidate(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    architecture: str,
    feature_set: str,
    horizon: int,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, list[ProbabilityModelArtifact], set[str], set[str]]:
    labels = dataset.target_class[horizon]
    eligible = np.ones(dataset.rows, dtype=bool)
    artifacts: list[ProbabilityModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    if architecture.startswith("shared"):
        probabilities, artifact, kind, device = _fit_booster(
            dataset,
            candidate_id=candidate_id,
            architecture=architecture,
            feature_set=feature_set,
            horizon=horizon,
            symbol_index=None,
            target_head="direct",
            objective="multiclass",
            labels=labels,
            eligible_label_mask=eligible,
            model_dir=model_dir,
            compute_backend=compute_backend,
            seed=seed,
            progress=progress,
        )
        artifacts.append(artifact)
        kinds.add(kind)
        devices.add(device)
        return probabilities, artifacts, kinds, devices
    probabilities = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
    for symbol_index, symbol in enumerate(SYMBOLS):
        current, artifact, kind, device = _fit_booster(
            dataset,
            candidate_id=candidate_id,
            architecture=architecture,
            feature_set=feature_set,
            horizon=horizon,
            symbol_index=symbol_index,
            target_head="direct",
            objective="multiclass",
            labels=labels,
            eligible_label_mask=eligible,
            model_dir=model_dir,
            compute_backend=compute_backend,
            seed=seed + symbol_index,
            progress=progress,
        )
        mask = dataset.symbol_index == symbol_index
        probabilities[mask] = current[mask]
        artifacts.append(artifact)
        kinds.add(kind)
        devices.add(device)
    return probabilities, artifacts, kinds, devices


def _fit_hurdle_candidate(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    architecture: str,
    feature_set: str,
    horizon: int,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, list[ProbabilityModelArtifact], set[str], set[str]]:
    target = dataset.target_class[horizon]
    opportunity = (target != 1).astype(np.int8)
    direction = (target == 2).astype(np.int8)
    all_rows = np.ones(dataset.rows, dtype=bool)
    opportunity_rows = target != 1
    p_opportunity = np.full(dataset.rows, np.nan, dtype=np.float32)
    p_long_given_opportunity = np.full(dataset.rows, np.nan, dtype=np.float32)
    artifacts: list[ProbabilityModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    symbol_indices: tuple[int | None, ...] = (
        (None,) if architecture.startswith("shared") else tuple(range(len(SYMBOLS)))
    )
    for position, symbol_index in enumerate(symbol_indices):
        current_opportunity, artifact, kind, device = _fit_booster(
            dataset,
            candidate_id=candidate_id,
            architecture=architecture,
            feature_set=feature_set,
            horizon=horizon,
            symbol_index=symbol_index,
            target_head="opportunity",
            objective="binary",
            labels=opportunity,
            eligible_label_mask=all_rows,
            model_dir=model_dir,
            compute_backend=compute_backend,
            seed=seed + position * 2,
            progress=progress,
        )
        current_direction, direction_artifact, direction_kind, direction_device = (
            _fit_booster(
                dataset,
                candidate_id=candidate_id,
                architecture=architecture,
                feature_set=feature_set,
                horizon=horizon,
                symbol_index=symbol_index,
                target_head="direction",
                objective="binary",
                labels=direction,
                eligible_label_mask=opportunity_rows,
                model_dir=model_dir,
                compute_backend=compute_backend,
                seed=seed + position * 2 + 1,
                progress=progress,
            )
        )
        mask = (
            np.ones(dataset.rows, dtype=bool)
            if symbol_index is None
            else dataset.symbol_index == symbol_index
        )
        p_opportunity[mask] = current_opportunity[mask]
        p_long_given_opportunity[mask] = current_direction[mask]
        artifacts.extend((artifact, direction_artifact))
        kinds.update((kind, direction_kind))
        devices.update((device, direction_device))
    p_long = p_opportunity * p_long_given_opportunity
    p_short = p_opportunity * (1.0 - p_long_given_opportunity)
    probabilities = np.column_stack((p_short, 1.0 - p_opportunity, p_long)).astype(
        np.float32
    )
    return probabilities, artifacts, kinds, devices


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-9, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    rows = np.arange(labels.size, dtype=np.int64)

    def loss(temperature: float) -> float:
        scaled = _temperature_scale(probabilities, temperature)
        return float(-np.mean(np.log(np.clip(scaled[rows, labels], 1e-12, 1.0))))

    before = loss(1.0)
    result = minimize_scalar(
        loss,
        bounds=(0.25, 4.0),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 100},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("probability temperature calibration failed")
    temperature = float(result.x)
    return temperature, before, loss(temperature)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> ClassificationMetrics:
    nonfinite = int(np.count_nonzero(~np.isfinite(probabilities)))
    if nonfinite or labels.size == 0 or probabilities.shape != (labels.size, 3):
        raise ValueError("classification metric inputs are invalid")
    row_sums = np.sum(probabilities, axis=1)
    if np.max(np.abs(row_sums - 1.0)) > 1e-5 or np.any(probabilities < 0.0):
        raise ValueError("classification probabilities are not normalized")
    rows = np.arange(labels.size, dtype=np.int64)
    prediction = np.argmax(probabilities, axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    recalls = np.divide(
        np.diag(confusion),
        np.sum(confusion, axis=1),
        out=np.zeros(3, dtype=np.float64),
        where=np.sum(confusion, axis=1) > 0,
    )
    target = np.eye(3, dtype=np.float64)[labels]
    confidence = np.max(probabilities, axis=1)
    correct = (prediction == labels).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            (confidence <= upper) if upper >= 1.0 else (confidence < upper)
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) - float(np.mean(correct[mask]))
            )
    return ClassificationMetrics(
        rows=int(labels.size),
        class_counts=tuple(int(np.count_nonzero(labels == item)) for item in range(3)),
        multiclass_log_loss=float(
            -np.mean(np.log(np.clip(probabilities[rows, labels], 1e-12, 1.0)))
        ),
        multiclass_brier_score=float(np.mean(np.sum((probabilities - target) ** 2, axis=1))),
        expected_calibration_error=ece,
        accuracy=float(np.mean(correct)),
        balanced_accuracy=float(np.mean(recalls)),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        mean_probabilities=tuple(float(value) for value in np.mean(probabilities, axis=0)),
        maximum_probability_mean=float(np.mean(confidence)),
        nonfinite_probabilities=nonfinite,
    )


def _stationary_bootstrap_mean_net(
    daily_net: np.ndarray,
    daily_trades: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float] | None:
    if daily_net.size == 0 or int(np.sum(daily_trades)) == 0 or samples <= 0:
        return None
    generator = np.random.default_rng(seed)
    results = np.empty(samples, dtype=np.float64)
    restart_probability = 0.2
    for sample in range(samples):
        index = int(generator.integers(0, daily_net.size))
        total_net = 0.0
        total_trades = 0
        for _ in range(daily_net.size):
            total_net += float(daily_net[index])
            total_trades += int(daily_trades[index])
            if generator.random() < restart_probability:
                index = int(generator.integers(0, daily_net.size))
            else:
                index = (index + 1) % daily_net.size
        results[sample] = total_net / total_trades if total_trades else 0.0
    return tuple(float(value) for value in np.quantile(results, (0.025, 0.5, 0.975)))


def _empty_replay(
    maximum_action_probability: float | None,
    direction_probability_margin: float | None,
) -> ReplayOutcome:
    return ReplayOutcome(
        metrics=ActionReplayMetrics(
            maximum_action_probability=maximum_action_probability,
            direction_probability_margin=direction_probability_margin,
            total_trades=0,
            trades_by_symbol={symbol: 0 for symbol in SYMBOLS},
            maximum_single_symbol_fraction=0.0,
            active_utc_days=0,
            total_net_bps=0.0,
            mean_net_bps=0.0,
            median_net_bps=0.0,
            positive_rate=0.0,
            profit_factor=0.0,
            median_monthly_net_bps=0.0,
            negative_month_fraction=1.0,
            maximum_peak_to_trough_drawdown_bps=0.0,
            longest_loss_streak=0,
            total_funding_cash_flow_bps=0.0,
            mean_funding_cash_flow_bps=0.0,
            day_block_bootstrap_mean_net_bps_lower_95=None,
            day_block_bootstrap_mean_net_bps_median=None,
            day_block_bootstrap_mean_net_bps_upper_95=None,
            candidate_rows=0,
            overlap_rejections=0,
            nonfinite_outcomes=0,
        ),
        selected_indices=np.empty(0, dtype=np.int64),
        selected_direction=np.empty(0, dtype=np.int8),
        net_return_bps=np.empty(0, dtype=np.float64),
    )


def replay_actions(
    dataset: DerivativesHurdleDataset,
    probabilities: np.ndarray,
    *,
    horizon: int,
    role: str,
    maximum_action_probability: float | None,
    direction_probability_margin: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> ReplayOutcome:
    if maximum_action_probability is None or direction_probability_margin is None:
        return _empty_replay(maximum_action_probability, direction_probability_margin)
    p_short = probabilities[:, 0]
    p_long = probabilities[:, 2]
    confidence = np.maximum(p_short, p_long)
    margin = np.abs(p_long - p_short)
    direction = np.where(p_long > p_short, 1, -1).astype(np.int8)
    candidates = (
        dataset.role_masks[horizon][role]
        & np.isfinite(confidence)
        & (confidence >= maximum_action_probability)
        & (margin >= direction_probability_margin)
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
        empty = _empty_replay(
            maximum_action_probability, direction_probability_margin
        )
        return ReplayOutcome(
            metrics=ActionReplayMetrics(
                **{
                    **empty.metrics.asdict(),
                    "candidate_rows": int(np.count_nonzero(candidates)),
                    "overlap_rejections": overlap_rejections,
                }
            ),
            selected_indices=empty.selected_indices,
            selected_direction=empty.selected_direction,
            net_return_bps=empty.net_return_bps,
        )
    indices = np.asarray(selected, dtype=np.int64)
    indices = indices[np.argsort(dataset.decision_time_ms[indices], kind="stable")]
    selected_direction = direction[indices]
    long_mask = selected_direction > 0
    net = np.where(
        long_mask,
        dataset.long_net_utility_bps[horizon][indices],
        dataset.short_net_utility_bps[horizon][indices],
    ).astype(np.float64)
    raw_funding = dataset.funding_cash_flow_bps[horizon][indices].astype(np.float64)
    funding_component = np.where(long_mask, -raw_funding, raw_funding)
    nonfinite = int(
        np.count_nonzero(~np.isfinite(net))
        + np.count_nonzero(~np.isfinite(funding_component))
    )
    if nonfinite:
        raise ValueError(f"action replay produced {nonfinite} nonfinite outcomes")
    role_spec = role_by_name(role)
    first_day = datetime.fromisoformat(role_spec.start).replace(tzinfo=UTC)
    last_day = datetime.fromisoformat(role_spec.end).replace(tzinfo=UTC)
    day_count = (last_day - first_day).days + 1
    daily_net = np.zeros(day_count, dtype=np.float64)
    daily_trades = np.zeros(day_count, dtype=np.int64)
    month_totals: dict[str, float] = {}
    for index, outcome in zip(indices, net, strict=True):
        timestamp = datetime.fromtimestamp(
            int(dataset.decision_time_ms[index]) / 1000.0, UTC
        )
        day_index = (timestamp.date() - first_day.date()).days
        daily_net[day_index] += outcome
        daily_trades[day_index] += 1
        month = f"{timestamp.year:04d}-{timestamp.month:02d}"
        month_totals[month] = month_totals.get(month, 0.0) + float(outcome)
    bootstrap = _stationary_bootstrap_mean_net(
        daily_net,
        daily_trades,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    positive_sum = float(np.sum(net[net > 0.0]))
    negative_sum = float(np.sum(net[net < 0.0]))
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0.0 else None
    equity = np.cumsum(net)
    drawdown = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:] - equity
    longest_loss_streak = 0
    current_streak = 0
    for outcome in net:
        current_streak = current_streak + 1 if outcome < 0.0 else 0
        longest_loss_streak = max(longest_loss_streak, current_streak)
    counts = np.asarray(list(trades_by_symbol.values()), dtype=np.float64)
    months = np.asarray(list(month_totals.values()), dtype=np.float64)
    metrics = ActionReplayMetrics(
        maximum_action_probability=float(maximum_action_probability),
        direction_probability_margin=float(direction_probability_margin),
        total_trades=int(indices.size),
        trades_by_symbol=trades_by_symbol,
        maximum_single_symbol_fraction=float(np.max(counts) / indices.size),
        active_utc_days=int(np.count_nonzero(daily_trades)),
        total_net_bps=float(np.sum(net)),
        mean_net_bps=float(np.mean(net)),
        median_net_bps=float(np.median(net)),
        positive_rate=float(np.mean(net > 0.0)),
        profit_factor=profit_factor,
        median_monthly_net_bps=float(np.median(months)) if months.size else 0.0,
        negative_month_fraction=float(np.mean(months < 0.0)) if months.size else 1.0,
        maximum_peak_to_trough_drawdown_bps=float(np.max(drawdown)),
        longest_loss_streak=longest_loss_streak,
        total_funding_cash_flow_bps=float(np.sum(funding_component)),
        mean_funding_cash_flow_bps=float(np.mean(funding_component)),
        day_block_bootstrap_mean_net_bps_lower_95=(bootstrap[0] if bootstrap else None),
        day_block_bootstrap_mean_net_bps_median=(bootstrap[1] if bootstrap else None),
        day_block_bootstrap_mean_net_bps_upper_95=(bootstrap[2] if bootstrap else None),
        candidate_rows=int(np.count_nonzero(candidates)),
        overlap_rejections=overlap_rejections,
        nonfinite_outcomes=nonfinite,
    )
    return ReplayOutcome(
        metrics=metrics,
        selected_indices=indices,
        selected_direction=selected_direction,
        net_return_bps=net,
    )


def _threshold_selection_key(item: Mapping[str, object]) -> tuple[float, ...]:
    replay = item["replay"]
    lower = replay["day_block_bootstrap_mean_net_bps_lower_95"]
    return (
        float(lower) if lower is not None else float("-inf"),
        float(replay["median_monthly_net_bps"]),
        float(replay["mean_net_bps"]),
        float(item["maximum_action_probability"]),
        float(item["direction_probability_margin"]),
    )


def _class_balance(
    dataset: DerivativesHurdleDataset,
    *,
    horizon: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    target = dataset.target_class[horizon]
    for role in ("training", "early_stop", "calibration", "viability"):
        mask = dataset.role_masks[horizon][role]
        counts = [int(np.count_nonzero(target[mask] == value)) for value in range(3)]
        output[role] = {"short": counts[0], "abstain": counts[1], "long": counts[2]}
    return output


def _evaluate_candidate(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    architecture: str,
    feature_set: str,
    horizon: int,
    probabilities: np.ndarray,
) -> tuple[dict[str, object], PassedCandidate | None]:
    early_mask = dataset.role_masks[horizon]["early_stop"]
    target = dataset.target_class[horizon]
    temperature, log_loss_before, log_loss_after = _fit_temperature(
        probabilities[early_mask], target[early_mask]
    )
    calibrated = _temperature_scale(probabilities, temperature).astype(np.float32)
    traces: list[dict[str, object]] = []
    for probability_threshold in ACTION_PROBABILITY_GRID:
        for margin_threshold in DIRECTION_MARGIN_GRID:
            initial = replay_actions(
                dataset,
                calibrated,
                horizon=horizon,
                role="calibration",
                maximum_action_probability=probability_threshold,
                direction_probability_margin=margin_threshold,
                bootstrap_samples=0,
                bootstrap_seed=3811,
            )
            metrics = initial.metrics
            support = (
                metrics.total_trades >= 90
                and metrics.active_utc_days >= 45
                and all(metrics.trades_by_symbol.get(symbol, 0) >= 15 for symbol in SYMBOLS)
            )
            if support:
                metrics = replay_actions(
                    dataset,
                    calibrated,
                    horizon=horizon,
                    role="calibration",
                    maximum_action_probability=probability_threshold,
                    direction_probability_margin=margin_threshold,
                    bootstrap_samples=1000,
                    bootstrap_seed=3811,
                ).metrics
            traces.append(
                {
                    "maximum_action_probability": probability_threshold,
                    "direction_probability_margin": margin_threshold,
                    "support_passed": support,
                    "replay": metrics.asdict(),
                }
            )
    eligible = [item for item in traces if item["support_passed"]]
    selected = max(eligible, key=_threshold_selection_key) if eligible else None
    if selected is None:
        calibration = _empty_replay(None, None).metrics
        viability = _empty_replay(None, None)
    else:
        calibration = ActionReplayMetrics(**selected["replay"])
        viability = replay_actions(
            dataset,
            calibrated,
            horizon=horizon,
            role="viability",
            maximum_action_probability=float(selected["maximum_action_probability"]),
            direction_probability_margin=float(selected["direction_probability_margin"]),
            bootstrap_samples=2000,
            bootstrap_seed=3812,
        )
    reasons: list[str] = []
    if selected is None:
        reasons.append("no_calibration_action_threshold_met_support")
    metrics = viability.metrics
    if metrics.total_trades < 180:
        reasons.append("viability_total_trades<180")
    for symbol in SYMBOLS:
        if metrics.trades_by_symbol.get(symbol, 0) < 30:
            reasons.append(f"{symbol}_viability_trades<30")
    if metrics.active_utc_days < 90:
        reasons.append("viability_active_utc_days<90")
    if metrics.mean_net_bps <= 0.0:
        reasons.append("viability_mean_net_bps<=0")
    if metrics.median_monthly_net_bps <= 0.0:
        reasons.append("viability_median_monthly_net_bps<=0")
    if metrics.profit_factor is None or metrics.profit_factor < 1.05:
        reasons.append("viability_profit_factor<1.05")
    if metrics.negative_month_fraction > 0.45:
        reasons.append("viability_negative_month_fraction>0.45")
    lower = metrics.day_block_bootstrap_mean_net_bps_lower_95
    if lower is None or lower <= 0.0:
        reasons.append("viability_day_block_lower_95<=0")
    if metrics.maximum_single_symbol_fraction > 0.50:
        reasons.append("viability_single_symbol_fraction>0.50")
    passed = not reasons
    result: dict[str, object] = {
        "candidate_id": candidate_id,
        "architecture": architecture,
        "feature_set": feature_set,
        "horizon_minutes": horizon,
        "class_balance": _class_balance(dataset, horizon=horizon),
        "temperature_calibration": {
            "temperature": temperature,
            "early_stop_log_loss_before": log_loss_before,
            "early_stop_log_loss_after": log_loss_after,
        },
        "early_stop_classification": classification_metrics(
            target[early_mask], calibrated[early_mask]
        ).asdict(),
        "calibration_classification": classification_metrics(
            target[dataset.role_masks[horizon]["calibration"]],
            calibrated[dataset.role_masks[horizon]["calibration"]],
        ).asdict(),
        "viability_classification": classification_metrics(
            target[dataset.role_masks[horizon]["viability"]],
            calibrated[dataset.role_masks[horizon]["viability"]],
        ).asdict(),
        "threshold_trace": traces,
        "selected_action_threshold": (
            {
                "maximum_action_probability": selected["maximum_action_probability"],
                "direction_probability_margin": selected[
                    "direction_probability_margin"
                ],
            }
            if selected is not None
            else None
        ),
        "calibration_replay": calibration.asdict(),
        "viability_replay": metrics.asdict(),
        "viability_gate_passed": passed,
        "viability_gate_reasons": reasons,
    }
    retained = (
        PassedCandidate(
            candidate_id=candidate_id,
            architecture=architecture,
            feature_set=feature_set,
            horizon_minutes=horizon,
            maximum_action_probability=float(
                selected["maximum_action_probability"]
            ),
            direction_probability_margin=float(
                selected["direction_probability_margin"]
            ),
            probabilities=calibrated,
            viability=viability,
        )
        if passed
        else None
    )
    return result, retained


def run_fixed_model_screen(
    dataset: DerivativesHurdleDataset,
    *,
    model_dir: str | Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> Round38ModelScreen:
    """Train and evaluate all 32 frozen candidates without viability selection."""

    root = Path(model_dir)
    candidate_results: list[Mapping[str, object]] = []
    artifacts: list[ProbabilityModelArtifact] = []
    passed: list[PassedCandidate] = []
    backend_kinds: set[str] = set()
    backend_devices: set[str] = set()
    candidate_index = 0
    for feature_set in FEATURE_SETS:
        for architecture in ARCHITECTURES:
            for horizon in HORIZONS_MINUTES:
                candidate_index += 1
                candidate_id = f"{feature_set}_{architecture}_h{horizon}"
                if progress is not None:
                    progress(
                        "round38_candidate",
                        {
                            "candidate_id": candidate_id,
                            "candidate_index": candidate_index,
                            "candidate_total": 32,
                            "status": "started",
                        },
                    )
                seed = SEED + candidate_index * 20
                if "direct_multiclass" in architecture:
                    probabilities, current_artifacts, kinds, devices = (
                        _fit_direct_candidate(
                            dataset,
                            candidate_id=candidate_id,
                            architecture=architecture,
                            feature_set=feature_set,
                            horizon=horizon,
                            model_dir=root,
                            compute_backend=compute_backend,
                            seed=seed,
                            progress=progress,
                        )
                    )
                else:
                    probabilities, current_artifacts, kinds, devices = (
                        _fit_hurdle_candidate(
                            dataset,
                            candidate_id=candidate_id,
                            architecture=architecture,
                            feature_set=feature_set,
                            horizon=horizon,
                            model_dir=root,
                            compute_backend=compute_backend,
                            seed=seed,
                            progress=progress,
                        )
                    )
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"{candidate_id} has nonfinite probabilities")
                result, retained = _evaluate_candidate(
                    dataset,
                    candidate_id=candidate_id,
                    architecture=architecture,
                    feature_set=feature_set,
                    horizon=horizon,
                    probabilities=probabilities,
                )
                candidate_results.append(result)
                artifacts.extend(current_artifacts)
                backend_kinds.update(kinds)
                backend_devices.update(devices)
                if retained is not None:
                    passed.append(retained)
                if progress is not None:
                    progress(
                        "round38_candidate",
                        {
                            "candidate_id": candidate_id,
                            "candidate_index": candidate_index,
                            "candidate_total": 32,
                            "status": "complete",
                            "viability_gate_passed": result[
                                "viability_gate_passed"
                            ],
                            "viability_trades": result["viability_replay"][
                                "total_trades"
                            ],
                            "viability_mean_net_bps": result["viability_replay"][
                                "mean_net_bps"
                            ],
                        },
                    )
                del probabilities
    if (
        len(candidate_results) != 32
        or len(artifacts) != 96
        or len(backend_kinds) != 1
        or len(backend_devices) != 1
    ):
        raise RuntimeError(
            "Round 38 fixed model screen is incomplete: "
            f"candidates={len(candidate_results)} artifacts={len(artifacts)} "
            f"backends={backend_kinds}/{backend_devices}"
        )
    return Round38ModelScreen(
        candidate_results=tuple(candidate_results),
        model_artifacts=tuple(artifacts),
        passed_candidates=tuple(passed),
        backend_kind=next(iter(backend_kinds)),
        backend_device=next(iter(backend_devices)),
    )


__all__ = [
    "ACTION_PROBABILITY_GRID",
    "ARCHITECTURES",
    "ActionReplayMetrics",
    "ClassificationMetrics",
    "DIRECTION_MARGIN_GRID",
    "FEATURE_SETS",
    "PassedCandidate",
    "ProbabilityModelArtifact",
    "ReplayOutcome",
    "Round38ModelScreen",
    "classification_metrics",
    "replay_actions",
    "run_fixed_model_screen",
]
