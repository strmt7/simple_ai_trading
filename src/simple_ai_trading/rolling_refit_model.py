"""Causal monthly LightGBM refits and utility-weighting ablation for Round 39."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping

import lightgbm as lgb
import numpy as np

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .derivatives_hurdle_model import (
    ACTION_PROBABILITY_GRID,
    DIRECTION_MARGIN_GRID,
    ActionReplayMetrics,
    ReplayOutcome,
    _fit_temperature,
    _lightgbm_parameters,
    _stationary_bootstrap_mean_net,
    _temperature_scale,
    _threshold_selection_key,
    classification_metrics,
)


SEED = 3901
EVALUATION_MONTHS = (
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
)
WEIGHTINGS = ("equal", "bounded_economic_utility")
MONTHLY_MINIMUM_TRADES = 30
MONTHLY_MINIMUM_TRADES_PER_SYMBOL = 5
MONTHLY_MINIMUM_ACTIVE_DAYS = 15


@dataclass(frozen=True)
class MonthlySchedule:
    evaluation_month: str
    training_start: str
    training_end: str
    early_stop_start: str
    early_stop_end: str
    calibration_start: str
    calibration_end: str
    evaluation_start: str
    evaluation_end: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RollingModelArtifact:
    model_id: str
    candidate_id: str
    evaluation_month: str
    target_head: str
    architecture: str
    weighting: str
    horizon_minutes: int
    symbol: str
    feature_count: int
    training_rows: int
    early_stop_rows: int
    training_weight_minimum: float
    training_weight_mean: float
    training_weight_maximum: float
    utility_weight_normalizer_bps: float | None
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
class RollingSupportCandidate:
    candidate_id: str
    architecture: str
    weighting: str
    horizon_minutes: int
    probabilities: np.ndarray
    monthly_calibration_masks: Mapping[str, np.ndarray]
    monthly_thresholds: Mapping[str, tuple[float, float] | None]
    evaluation: ReplayOutcome


@dataclass(frozen=True)
class Round39ModelScreen:
    schedules: tuple[Mapping[str, object], ...]
    candidate_results: tuple[Mapping[str, object], ...]
    model_artifacts: tuple[RollingModelArtifact, ...]
    support_candidates: tuple[RollingSupportCandidate, ...]
    passed_candidates: tuple[RollingSupportCandidate, ...]
    backend_kind: str
    backend_device: str


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _month_start(value: str) -> date:
    parsed = datetime.strptime(value, "%Y-%m").date()
    return parsed.replace(day=1)


def _inclusive_end(exclusive: date) -> str:
    return date.fromordinal(exclusive.toordinal() - 1).isoformat()


def frozen_monthly_schedules() -> tuple[MonthlySchedule, ...]:
    schedules: list[MonthlySchedule] = []
    for evaluation_month in EVALUATION_MONTHS:
        evaluation_start = _month_start(evaluation_month)
        evaluation_end = _add_months(evaluation_start, 1)
        calibration_start = _add_months(evaluation_start, -1)
        early_stop_start = _add_months(evaluation_start, -3)
        training_start = _add_months(evaluation_start, -27)
        schedules.append(
            MonthlySchedule(
                evaluation_month=evaluation_month,
                training_start=training_start.isoformat(),
                training_end=_inclusive_end(early_stop_start),
                early_stop_start=early_stop_start.isoformat(),
                early_stop_end=_inclusive_end(calibration_start),
                calibration_start=calibration_start.isoformat(),
                calibration_end=_inclusive_end(evaluation_start),
                evaluation_start=evaluation_start.isoformat(),
                evaluation_end=_inclusive_end(evaluation_end),
            )
        )
    return tuple(schedules)


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def _end_exclusive_ms(inclusive: str) -> int:
    value = datetime.fromisoformat(inclusive).replace(tzinfo=UTC)
    return int((value.timestamp() + 86_400) * 1000)


def _window_mask(
    dataset: DerivativesHurdleDataset,
    *,
    horizon_minutes: int,
    start: str,
    end: str,
) -> np.ndarray:
    start_ms = _date_ms(start)
    end_ms = _end_exclusive_ms(end)
    exit_time_ms = dataset.decision_time_ms + (horizon_minutes + 1) * MINUTE_MS
    return (
        (dataset.decision_time_ms >= start_ms)
        & (dataset.decision_time_ms < end_ms)
        & (exit_time_ms < end_ms)
    )


def schedule_masks(
    dataset: DerivativesHurdleDataset,
    schedule: MonthlySchedule,
    horizon_minutes: int,
) -> dict[str, np.ndarray]:
    masks = {
        "training": _window_mask(
            dataset,
            horizon_minutes=horizon_minutes,
            start=schedule.training_start,
            end=schedule.training_end,
        ),
        "early_stop": _window_mask(
            dataset,
            horizon_minutes=horizon_minutes,
            start=schedule.early_stop_start,
            end=schedule.early_stop_end,
        ),
        "calibration": _window_mask(
            dataset,
            horizon_minutes=horizon_minutes,
            start=schedule.calibration_start,
            end=schedule.calibration_end,
        ),
        "evaluation": _window_mask(
            dataset,
            horizon_minutes=horizon_minutes,
            start=schedule.evaluation_start,
            end=schedule.evaluation_end,
        ),
    }
    combined = np.zeros(dataset.rows, dtype=np.int8)
    for mask in masks.values():
        combined += mask.astype(np.int8)
    if np.any(combined > 1):
        raise ValueError(f"{schedule.evaluation_month} role masks overlap")
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError(f"{schedule.evaluation_month} contains an empty role")
    return masks


def _training_weights(
    dataset: DerivativesHurdleDataset,
    *,
    horizon_minutes: int,
    training_mask: np.ndarray,
    weighting: str,
    target_head: str,
) -> tuple[np.ndarray, float | None]:
    if weighting == "equal":
        return np.ones(int(np.count_nonzero(training_mask)), dtype=np.float32), None
    if weighting != "bounded_economic_utility":
        raise KeyError(weighting)
    long_utility = dataset.long_net_utility_bps[horizon_minutes].astype(np.float64)
    short_utility = dataset.short_net_utility_bps[horizon_minutes].astype(np.float64)
    if target_head == "direction":
        salience = np.abs(long_utility - short_utility)
    else:
        salience = np.maximum(np.maximum(long_utility, short_utility), 0.0)
    selected = salience[training_mask]
    positive = selected[selected > 0.0]
    if positive.size == 0:
        raise ValueError("economic utility weighting has no positive salience")
    normalizer = float(np.quantile(positive, 0.90))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("economic utility normalizer is invalid")
    weights = 1.0 + np.minimum(selected / normalizer, 2.0)
    if (
        not np.isfinite(weights).all()
        or float(np.min(weights)) < 1.0
        or float(np.max(weights)) > 3.0 + 1e-7
    ):
        raise ValueError("economic utility weights violate frozen bounds")
    return weights.astype(np.float32), normalizer


def _fit_booster(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    architecture: str,
    weighting: str,
    horizon_minutes: int,
    schedule: MonthlySchedule,
    masks: Mapping[str, np.ndarray],
    symbol_index: int | None,
    target_head: str,
    objective: str,
    labels: np.ndarray,
    eligible_label_mask: np.ndarray,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, RollingModelArtifact, str, str]:
    features = dataset.feature_view("price_flow_only")
    training = masks["training"] & eligible_label_mask
    early_stop = masks["early_stop"] & eligible_label_mask
    prediction_mask = masks["early_stop"] | masks["calibration"] | masks["evaluation"]
    symbol = "shared"
    if symbol_index is not None:
        symbol_mask = dataset.symbol_index == symbol_index
        training &= symbol_mask
        early_stop &= symbol_mask
        prediction_mask &= symbol_mask
        symbol = SYMBOLS[symbol_index]
    training_rows = int(np.count_nonzero(training))
    early_stop_rows = int(np.count_nonzero(early_stop))
    if training_rows == 0 or early_stop_rows == 0:
        raise ValueError(
            f"{candidate_id}/{schedule.evaluation_month}/{target_head}/{symbol} has an empty role"
        )
    weights, normalizer = _training_weights(
        dataset,
        horizon_minutes=horizon_minutes,
        training_mask=training,
        weighting=weighting,
        target_head=target_head,
    )
    parameters, backend_kind, backend_device = _lightgbm_parameters(
        compute_backend,
        objective=objective,
        seed=seed,
    )
    model_id = (
        f"{candidate_id}_{schedule.evaluation_month.replace('-', '')}_"
        f"{target_head}_{symbol.lower()}"
    )
    if progress is not None:
        progress(
            "round39_model_training",
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
        weight=weights,
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
    raw = booster.predict(
        features[prediction_mask], num_iteration=booster.best_iteration
    )
    if objective == "multiclass":
        prediction = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
        prediction[prediction_mask] = np.asarray(raw, dtype=np.float32)
    else:
        prediction = np.full(dataset.rows, np.nan, dtype=np.float32)
        prediction[prediction_mask] = np.asarray(raw, dtype=np.float32)
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
    artifact = RollingModelArtifact(
        model_id=model_id,
        candidate_id=candidate_id,
        evaluation_month=schedule.evaluation_month,
        target_head=target_head,
        architecture=architecture,
        weighting=weighting,
        horizon_minutes=horizon_minutes,
        symbol=symbol,
        feature_count=features.shape[1],
        training_rows=training_rows,
        early_stop_rows=early_stop_rows,
        training_weight_minimum=float(np.min(weights)),
        training_weight_mean=float(np.mean(weights)),
        training_weight_maximum=float(np.max(weights)),
        utility_weight_normalizer_bps=normalizer,
        best_iteration=int(booster.best_iteration),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(model_path),
        bytes=model_path.stat().st_size,
        sha256=_file_sha256(model_path),
        reload_max_abs_prediction_error=reload_error,
        top_feature_gain=tuple(
            (feature_names[int(index)], float(gain[int(index)])) for index in order
        ),
    )
    if progress is not None:
        progress(
            "round39_model_training",
            {
                "model_id": model_id,
                "status": "complete",
                "best_iteration": artifact.best_iteration,
                "artifact_sha256": artifact.sha256,
            },
        )
    return prediction, artifact, backend_kind, backend_device


def _fit_month_candidate(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    architecture: str,
    weighting: str,
    horizon_minutes: int,
    schedule: MonthlySchedule,
    masks: Mapping[str, np.ndarray],
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, list[RollingModelArtifact], set[str], set[str]]:
    target = dataset.target_class[horizon_minutes]
    artifacts: list[RollingModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    if architecture == "per_symbol_direct_multiclass_lightgbm":
        probabilities = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
        for symbol_index in range(len(SYMBOLS)):
            current, artifact, kind, device = _fit_booster(
                dataset,
                candidate_id=candidate_id,
                architecture=architecture,
                weighting=weighting,
                horizon_minutes=horizon_minutes,
                schedule=schedule,
                masks=masks,
                symbol_index=symbol_index,
                target_head="direct",
                objective="multiclass",
                labels=target,
                eligible_label_mask=np.ones(dataset.rows, dtype=bool),
                model_dir=model_dir,
                compute_backend=compute_backend,
                seed=seed + symbol_index,
                progress=progress,
            )
            symbol_mask = dataset.symbol_index == symbol_index
            probabilities[symbol_mask] = current[symbol_mask]
            artifacts.append(artifact)
            kinds.add(kind)
            devices.add(device)
        return probabilities, artifacts, kinds, devices
    if architecture != "shared_two_stage_hurdle_lightgbm":
        raise KeyError(architecture)
    opportunity = (target != 1).astype(np.int8)
    direction = (target == 2).astype(np.int8)
    p_opportunity, artifact, kind, device = _fit_booster(
        dataset,
        candidate_id=candidate_id,
        architecture=architecture,
        weighting=weighting,
        horizon_minutes=horizon_minutes,
        schedule=schedule,
        masks=masks,
        symbol_index=None,
        target_head="opportunity",
        objective="binary",
        labels=opportunity,
        eligible_label_mask=np.ones(dataset.rows, dtype=bool),
        model_dir=model_dir,
        compute_backend=compute_backend,
        seed=seed,
        progress=progress,
    )
    p_long_given_opportunity, direction_artifact, direction_kind, direction_device = (
        _fit_booster(
            dataset,
            candidate_id=candidate_id,
            architecture=architecture,
            weighting=weighting,
            horizon_minutes=horizon_minutes,
            schedule=schedule,
            masks=masks,
            symbol_index=None,
            target_head="direction",
            objective="binary",
            labels=direction,
            eligible_label_mask=target != 1,
            model_dir=model_dir,
            compute_backend=compute_backend,
            seed=seed + 1,
            progress=progress,
        )
    )
    p_long = p_opportunity * p_long_given_opportunity
    p_short = p_opportunity * (1.0 - p_long_given_opportunity)
    probabilities = np.column_stack(
        (p_short, 1.0 - p_opportunity, p_long)
    ).astype(np.float32)
    artifacts.extend((artifact, direction_artifact))
    kinds.update((kind, direction_kind))
    devices.update((device, direction_device))
    return probabilities, artifacts, kinds, devices


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


def _replay_candidate_mask(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_mask: np.ndarray,
    direction: np.ndarray,
    horizon_minutes: int,
    period_start: str,
    period_end: str,
    maximum_action_probability: float | None,
    direction_probability_margin: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> ReplayOutcome:
    selected: list[int] = []
    overlap_rejections = 0
    trades_by_symbol: dict[str, int] = {}
    for symbol_index, symbol in enumerate(SYMBOLS):
        indices = np.flatnonzero(
            candidate_mask & (dataset.symbol_index == symbol_index)
        )
        indices = indices[np.argsort(dataset.decision_time_ms[indices], kind="stable")]
        next_available_ms = -1
        symbol_selected: list[int] = []
        for index in indices:
            entry_time = int(dataset.decision_time_ms[index]) + MINUTE_MS
            if entry_time < next_available_ms:
                overlap_rejections += 1
                continue
            symbol_selected.append(int(index))
            next_available_ms = entry_time + horizon_minutes * MINUTE_MS
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
                    "candidate_rows": int(np.count_nonzero(candidate_mask)),
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
        dataset.long_net_utility_bps[horizon_minutes][indices],
        dataset.short_net_utility_bps[horizon_minutes][indices],
    ).astype(np.float64)
    raw_funding = dataset.funding_cash_flow_bps[horizon_minutes][indices].astype(
        np.float64
    )
    funding_component = np.where(long_mask, -raw_funding, raw_funding)
    nonfinite = int(
        np.count_nonzero(~np.isfinite(net))
        + np.count_nonzero(~np.isfinite(funding_component))
    )
    if nonfinite:
        raise ValueError(f"rolling replay produced {nonfinite} nonfinite outcomes")
    first_day = datetime.fromisoformat(period_start).replace(tzinfo=UTC)
    last_day = datetime.fromisoformat(period_end).replace(tzinfo=UTC)
    day_count = (last_day.date() - first_day.date()).days + 1
    daily_net = np.zeros(day_count, dtype=np.float64)
    daily_trades = np.zeros(day_count, dtype=np.int64)
    month_totals: dict[str, float] = {}
    for index, outcome in zip(indices, net, strict=True):
        timestamp = datetime.fromtimestamp(
            int(dataset.decision_time_ms[index]) / 1000.0, UTC
        )
        day_index = (timestamp.date() - first_day.date()).days
        if not 0 <= day_index < day_count:
            raise ValueError("rolling replay selected a row outside its period")
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
    return ReplayOutcome(
        metrics=ActionReplayMetrics(
            maximum_action_probability=maximum_action_probability,
            direction_probability_margin=direction_probability_margin,
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
            negative_month_fraction=float(np.mean(months < 0.0))
            if months.size
            else 1.0,
            maximum_peak_to_trough_drawdown_bps=float(np.max(drawdown)),
            longest_loss_streak=longest_loss_streak,
            total_funding_cash_flow_bps=float(np.sum(funding_component)),
            mean_funding_cash_flow_bps=float(np.mean(funding_component)),
            day_block_bootstrap_mean_net_bps_lower_95=(
                bootstrap[0] if bootstrap else None
            ),
            day_block_bootstrap_mean_net_bps_median=(
                bootstrap[1] if bootstrap else None
            ),
            day_block_bootstrap_mean_net_bps_upper_95=(
                bootstrap[2] if bootstrap else None
            ),
            candidate_rows=int(np.count_nonzero(candidate_mask)),
            overlap_rejections=overlap_rejections,
            nonfinite_outcomes=nonfinite,
        ),
        selected_indices=indices,
        selected_direction=selected_direction,
        net_return_bps=net,
    )


def replay_probabilities(
    dataset: DerivativesHurdleDataset,
    probabilities: np.ndarray,
    *,
    eligibility_mask: np.ndarray,
    horizon_minutes: int,
    period_start: str,
    period_end: str,
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
        eligibility_mask
        & np.isfinite(confidence)
        & (confidence >= maximum_action_probability)
        & (margin >= direction_probability_margin)
    )
    return _replay_candidate_mask(
        dataset,
        candidate_mask=candidates,
        direction=direction,
        horizon_minutes=horizon_minutes,
        period_start=period_start,
        period_end=period_end,
        maximum_action_probability=maximum_action_probability,
        direction_probability_margin=direction_probability_margin,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _apply_temperature(
    probabilities: np.ndarray,
    mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    calibrated = np.full_like(probabilities, np.nan, dtype=np.float32)
    calibrated[mask] = _temperature_scale(
        probabilities[mask], temperature
    ).astype(np.float32)
    return calibrated


def _evaluate_month(
    dataset: DerivativesHurdleDataset,
    *,
    candidate_id: str,
    schedule: MonthlySchedule,
    masks: Mapping[str, np.ndarray],
    horizon_minutes: int,
    probabilities: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, tuple[float, float] | None, ReplayOutcome]:
    prediction_mask = masks["early_stop"] | masks["calibration"] | masks["evaluation"]
    if not np.isfinite(probabilities[prediction_mask]).all():
        raise ValueError(f"{candidate_id}/{schedule.evaluation_month} has nonfinite probabilities")
    target = dataset.target_class[horizon_minutes]
    temperature, log_loss_before, log_loss_after = _fit_temperature(
        probabilities[masks["early_stop"]], target[masks["early_stop"]]
    )
    calibrated = _apply_temperature(probabilities, prediction_mask, temperature)
    traces: list[dict[str, object]] = []
    for probability_threshold in ACTION_PROBABILITY_GRID:
        for margin_threshold in DIRECTION_MARGIN_GRID:
            initial = replay_probabilities(
                dataset,
                calibrated,
                eligibility_mask=masks["calibration"],
                horizon_minutes=horizon_minutes,
                period_start=schedule.calibration_start,
                period_end=schedule.calibration_end,
                maximum_action_probability=probability_threshold,
                direction_probability_margin=margin_threshold,
                bootstrap_samples=0,
                bootstrap_seed=3911,
            )
            metrics = initial.metrics
            support = (
                metrics.total_trades >= MONTHLY_MINIMUM_TRADES
                and metrics.active_utc_days >= MONTHLY_MINIMUM_ACTIVE_DAYS
                and all(
                    metrics.trades_by_symbol.get(symbol, 0)
                    >= MONTHLY_MINIMUM_TRADES_PER_SYMBOL
                    for symbol in SYMBOLS
                )
            )
            if support:
                metrics = replay_probabilities(
                    dataset,
                    calibrated,
                    eligibility_mask=masks["calibration"],
                    horizon_minutes=horizon_minutes,
                    period_start=schedule.calibration_start,
                    period_end=schedule.calibration_end,
                    maximum_action_probability=probability_threshold,
                    direction_probability_margin=margin_threshold,
                    bootstrap_samples=500,
                    bootstrap_seed=3911,
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
    threshold = (
        (
            float(selected["maximum_action_probability"]),
            float(selected["direction_probability_margin"]),
        )
        if selected is not None
        else None
    )
    if threshold is None:
        calibration = _empty_replay(None, None).metrics
        evaluation = _empty_replay(None, None)
    else:
        calibration = ActionReplayMetrics(**selected["replay"])
        evaluation = replay_probabilities(
            dataset,
            calibrated,
            eligibility_mask=masks["evaluation"],
            horizon_minutes=horizon_minutes,
            period_start=schedule.evaluation_start,
            period_end=schedule.evaluation_end,
            maximum_action_probability=threshold[0],
            direction_probability_margin=threshold[1],
            bootstrap_samples=500,
            bootstrap_seed=3912,
        )
    result: dict[str, object] = {
        "evaluation_month": schedule.evaluation_month,
        "schedule": schedule.asdict(),
        "role_rows": {
            role: int(np.count_nonzero(mask)) for role, mask in masks.items()
        },
        "temperature_calibration": {
            "temperature": temperature,
            "early_stop_log_loss_before": log_loss_before,
            "early_stop_log_loss_after": log_loss_after,
        },
        "early_stop_classification": classification_metrics(
            target[masks["early_stop"]], calibrated[masks["early_stop"]]
        ).asdict(),
        "calibration_classification": classification_metrics(
            target[masks["calibration"]], calibrated[masks["calibration"]]
        ).asdict(),
        "evaluation_classification": classification_metrics(
            target[masks["evaluation"]], calibrated[masks["evaluation"]]
        ).asdict(),
        "threshold_trace": traces,
        "selected_action_threshold": (
            {
                "maximum_action_probability": threshold[0],
                "direction_probability_margin": threshold[1],
            }
            if threshold is not None
            else None
        ),
        "calibration_replay": calibration.asdict(),
        "evaluation_replay": evaluation.metrics.asdict(),
    }
    return result, calibrated, threshold, evaluation


def _maximum_single_month_positive_fraction(
    dataset: DerivativesHurdleDataset,
    outcome: ReplayOutcome,
) -> float:
    positive_by_month: dict[str, float] = {}
    for row, net in zip(
        outcome.selected_indices, outcome.net_return_bps, strict=True
    ):
        timestamp = datetime.fromtimestamp(
            int(dataset.decision_time_ms[int(row)]) / 1000.0, UTC
        )
        month = f"{timestamp.year:04d}-{timestamp.month:02d}"
        if net > 0.0:
            positive_by_month[month] = positive_by_month.get(month, 0.0) + float(net)
    total = sum(positive_by_month.values())
    return max(positive_by_month.values(), default=0.0) / total if total > 0.0 else 1.0


def _aggregate_gate_reasons(
    metrics: ActionReplayMetrics,
    *,
    selected_months: int,
    months_with_trades: int,
    maximum_single_month_positive_fraction: float,
) -> tuple[list[str], list[str]]:
    support: list[str] = []
    if selected_months < 4:
        support.append("selected_threshold_months<4")
    if metrics.total_trades < 270:
        support.append("aggregate_total_trades<270")
    for symbol in SYMBOLS:
        if metrics.trades_by_symbol.get(symbol, 0) < 45:
            support.append(f"{symbol}_aggregate_trades<45")
    if metrics.active_utc_days < 90:
        support.append("aggregate_active_utc_days<90")
    if months_with_trades < 4:
        support.append("aggregate_months_with_trades<4")
    if metrics.maximum_single_symbol_fraction > 0.50:
        support.append("aggregate_single_symbol_fraction>0.50")
    reasons = list(support)
    if metrics.mean_net_bps <= 0.0:
        reasons.append("aggregate_mean_net_bps<=0")
    if metrics.median_monthly_net_bps <= 0.0:
        reasons.append("aggregate_median_monthly_net_bps<=0")
    if metrics.profit_factor is None or metrics.profit_factor < 1.05:
        reasons.append("aggregate_profit_factor<1.05")
    if metrics.negative_month_fraction > 0.45:
        reasons.append("aggregate_negative_month_fraction>0.45")
    lower = metrics.day_block_bootstrap_mean_net_bps_lower_95
    if lower is None or lower <= 0.0:
        reasons.append("aggregate_day_block_lower_95<=0")
    if maximum_single_month_positive_fraction > 0.50:
        reasons.append("single_month_fraction_of_positive_net_bps>0.50")
    if metrics.nonfinite_outcomes:
        reasons.append("aggregate_nonfinite_outcomes>0")
    return support, reasons


def _candidate_specs() -> tuple[tuple[str, str, str, int], ...]:
    return (
        (
            "rolling_shared_hurdle_h120_equal",
            "shared_two_stage_hurdle_lightgbm",
            "equal",
            120,
        ),
        (
            "rolling_shared_hurdle_h120_utility",
            "shared_two_stage_hurdle_lightgbm",
            "bounded_economic_utility",
            120,
        ),
        (
            "rolling_per_symbol_direct_h30_equal",
            "per_symbol_direct_multiclass_lightgbm",
            "equal",
            30,
        ),
        (
            "rolling_per_symbol_direct_h30_utility",
            "per_symbol_direct_multiclass_lightgbm",
            "bounded_economic_utility",
            30,
        ),
    )


def run_rolling_refit_screen(
    dataset: DerivativesHurdleDataset,
    *,
    model_dir: str | Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> Round39ModelScreen:
    """Train the four frozen rolling candidates and evaluate six causal months."""

    root = Path(model_dir)
    schedules = frozen_monthly_schedules()
    artifacts: list[RollingModelArtifact] = []
    results: list[Mapping[str, object]] = []
    support_candidates: list[RollingSupportCandidate] = []
    passed_candidates: list[RollingSupportCandidate] = []
    backend_kinds: set[str] = set()
    backend_devices: set[str] = set()
    for candidate_index, (
        candidate_id,
        architecture,
        weighting,
        horizon_minutes,
    ) in enumerate(_candidate_specs(), start=1):
        if progress is not None:
            progress(
                "round39_candidate",
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "candidate_total": 4,
                    "status": "started",
                },
            )
        stitched_probabilities = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
        stitched_candidate_mask = np.zeros(dataset.rows, dtype=bool)
        stitched_direction = np.zeros(dataset.rows, dtype=np.int8)
        monthly_results: list[Mapping[str, object]] = []
        monthly_calibration_masks: dict[str, np.ndarray] = {}
        monthly_thresholds: dict[str, tuple[float, float] | None] = {}
        for month_index, schedule in enumerate(schedules, start=1):
            masks = schedule_masks(dataset, schedule, horizon_minutes)
            probabilities, current_artifacts, kinds, devices = _fit_month_candidate(
                dataset,
                candidate_id=candidate_id,
                architecture=architecture,
                weighting=weighting,
                horizon_minutes=horizon_minutes,
                schedule=schedule,
                masks=masks,
                model_dir=root,
                compute_backend=compute_backend,
                seed=SEED + candidate_index * 100 + month_index * 10,
                progress=progress,
            )
            month_result, calibrated, threshold, evaluation = _evaluate_month(
                dataset,
                candidate_id=candidate_id,
                schedule=schedule,
                masks=masks,
                horizon_minutes=horizon_minutes,
                probabilities=probabilities,
            )
            monthly_results.append(month_result)
            monthly_calibration_masks[schedule.evaluation_month] = masks[
                "calibration"
            ].copy()
            monthly_thresholds[schedule.evaluation_month] = threshold
            stitched_probabilities[masks["evaluation"]] = calibrated[
                masks["evaluation"]
            ]
            if threshold is not None:
                p_short = calibrated[:, 0]
                p_long = calibrated[:, 2]
                confidence = np.maximum(p_short, p_long)
                margin = np.abs(p_long - p_short)
                routed = (
                    masks["evaluation"]
                    & np.isfinite(confidence)
                    & (confidence >= threshold[0])
                    & (margin >= threshold[1])
                )
                stitched_candidate_mask |= routed
                stitched_direction[routed] = np.where(
                    p_long[routed] > p_short[routed], 1, -1
                ).astype(np.int8)
            artifacts.extend(current_artifacts)
            backend_kinds.update(kinds)
            backend_devices.update(devices)
            if progress is not None:
                progress(
                    "round39_month",
                    {
                        "candidate_id": candidate_id,
                        "evaluation_month": schedule.evaluation_month,
                        "status": "complete",
                        "threshold_selected": threshold is not None,
                        "evaluation_trades": evaluation.metrics.total_trades,
                        "evaluation_mean_net_bps": evaluation.metrics.mean_net_bps,
                    },
                )
            del probabilities, calibrated
        aggregate = _replay_candidate_mask(
            dataset,
            candidate_mask=stitched_candidate_mask,
            direction=stitched_direction,
            horizon_minutes=horizon_minutes,
            period_start=schedules[0].evaluation_start,
            period_end=schedules[-1].evaluation_end,
            maximum_action_probability=None,
            direction_probability_margin=None,
            bootstrap_samples=2000,
            bootstrap_seed=3913,
        )
        selected_months = sum(
            threshold is not None for threshold in monthly_thresholds.values()
        )
        months_with_trades = sum(
            int(month["evaluation_replay"]["total_trades"]) > 0
            for month in monthly_results
        )
        month_fraction = _maximum_single_month_positive_fraction(dataset, aggregate)
        support_reasons, gate_reasons = _aggregate_gate_reasons(
            aggregate.metrics,
            selected_months=selected_months,
            months_with_trades=months_with_trades,
            maximum_single_month_positive_fraction=month_fraction,
        )
        support_passed = not support_reasons
        passed = not gate_reasons
        result = {
            "candidate_id": candidate_id,
            "architecture": architecture,
            "weighting": weighting,
            "horizon_minutes": horizon_minutes,
            "monthly_results": monthly_results,
            "selected_threshold_months": selected_months,
            "months_with_trades": months_with_trades,
            "aggregate_replay": aggregate.metrics.asdict(),
            "maximum_single_month_fraction_of_positive_net_bps": month_fraction,
            "ai_entry_support_passed": support_passed,
            "ai_entry_support_reasons": support_reasons,
            "aggregate_ml_gate_passed": passed,
            "aggregate_ml_gate_reasons": gate_reasons,
        }
        results.append(result)
        retained = RollingSupportCandidate(
            candidate_id=candidate_id,
            architecture=architecture,
            weighting=weighting,
            horizon_minutes=horizon_minutes,
            probabilities=stitched_probabilities,
            monthly_calibration_masks=monthly_calibration_masks,
            monthly_thresholds=monthly_thresholds,
            evaluation=aggregate,
        )
        if support_passed:
            support_candidates.append(retained)
        if passed:
            passed_candidates.append(retained)
        if progress is not None:
            progress(
                "round39_candidate",
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "candidate_total": 4,
                    "status": "complete",
                    "ai_entry_support_passed": support_passed,
                    "aggregate_ml_gate_passed": passed,
                    "aggregate_trades": aggregate.metrics.total_trades,
                    "aggregate_mean_net_bps": aggregate.metrics.mean_net_bps,
                },
            )
    if (
        len(results) != 4
        or len(artifacts) != 60
        or len(backend_kinds) != 1
        or len(backend_devices) != 1
    ):
        raise RuntimeError(
            "Round 39 model/artifact/backend cardinality violates the frozen design"
        )
    schedule_evidence: list[Mapping[str, object]] = []
    for schedule in schedules:
        entry: dict[str, object] = schedule.asdict()
        entry["role_rows_by_horizon"] = {
            str(horizon): {
                role: int(np.count_nonzero(mask))
                for role, mask in schedule_masks(dataset, schedule, horizon).items()
            }
            for horizon in (30, 120)
        }
        schedule_evidence.append(entry)
    return Round39ModelScreen(
        schedules=tuple(schedule_evidence),
        candidate_results=tuple(results),
        model_artifacts=tuple(artifacts),
        support_candidates=tuple(support_candidates),
        passed_candidates=tuple(passed_candidates),
        backend_kind=next(iter(backend_kinds)),
        backend_device=next(iter(backend_devices)),
    )


__all__ = [
    "EVALUATION_MONTHS",
    "MonthlySchedule",
    "RollingModelArtifact",
    "RollingSupportCandidate",
    "Round39ModelScreen",
    "frozen_monthly_schedules",
    "replay_probabilities",
    "run_rolling_refit_screen",
    "schedule_masks",
]
