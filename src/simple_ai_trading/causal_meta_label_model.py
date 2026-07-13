"""Causal stacked profitability meta-labeling for the Round 40 model screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .derivatives_hurdle_model import (
    ActionReplayMetrics,
    ReplayOutcome,
    _fit_temperature,
    _lightgbm_parameters,
    _temperature_scale,
    classification_metrics,
)
from .rolling_refit_model import (
    _aggregate_gate_reasons,
    _maximum_single_month_positive_fraction,
    _replay_candidate_mask,
)


ROUND = 40
SEED = 4001
HORIZON_MINUTES = 30
EVALUATION_MONTHS = (
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
)
META_PROBABILITY_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PRIMARY_MARGIN_GRID = (0.00, 0.05, 0.10, 0.15)
MAXIMUM_ENTRIES_PER_SYMBOL_DAY = 8
MONTHLY_MINIMUM_TRADES = 30
MONTHLY_MINIMUM_TRADES_PER_SYMBOL = 5
MONTHLY_MINIMUM_ACTIVE_DAYS = 15
META_FEATURE_COUNT = 81


@dataclass(frozen=True)
class CausalMetaSchedule:
    evaluation_month: str
    base_training_start: str
    base_training_end: str
    base_early_stop_start: str
    base_early_stop_end: str
    meta_fit_start: str
    meta_fit_end: str
    meta_early_stop_start: str
    meta_early_stop_end: str
    threshold_calibration_start: str
    threshold_calibration_end: str
    evaluation_start: str
    evaluation_end: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CausalMetaModelArtifact:
    model_id: str
    model_role: str
    evaluation_month: str
    symbol: str
    feature_count: int
    training_rows: int
    early_stop_rows: int
    positive_training_rows: int | None
    positive_early_stop_rows: int | None
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
class CapacityReplay:
    outcome: ReplayOutcome
    threshold_candidate_rows: int
    overlap_rejections: int
    capacity_rejections: int
    maximum_entries_in_one_symbol_day: int

    def evidence(self) -> dict[str, object]:
        return {
            "threshold_candidate_rows": self.threshold_candidate_rows,
            "overlap_rejections": self.overlap_rejections,
            "capacity_rejections": self.capacity_rejections,
            "maximum_entries_in_one_symbol_day": (
                self.maximum_entries_in_one_symbol_day
            ),
            "replay": self.outcome.metrics.asdict(),
        }


@dataclass(frozen=True)
class CausalMetaCandidate:
    candidate_id: str
    horizon_minutes: int
    primary_probabilities: np.ndarray
    meta_probabilities: np.ndarray
    monthly_thresholds: Mapping[str, tuple[float, float] | None]
    evaluation: ReplayOutcome


@dataclass(frozen=True)
class Round40ModelScreen:
    schedules: tuple[Mapping[str, object], ...]
    candidate_result: Mapping[str, object]
    model_artifacts: tuple[CausalMetaModelArtifact, ...]
    candidate: CausalMetaCandidate
    passed_candidate: CausalMetaCandidate | None
    backend_kind: str
    backend_device: str


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _month_start(value: str) -> date:
    parsed = datetime.strptime(value, "%Y-%m").date()
    return parsed.replace(day=1)


def _add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _inclusive_end(exclusive: date) -> str:
    return date.fromordinal(exclusive.toordinal() - 1).isoformat()


def frozen_meta_schedules() -> tuple[CausalMetaSchedule, ...]:
    schedules: list[CausalMetaSchedule] = []
    for evaluation_month in EVALUATION_MONTHS:
        evaluation_start = _month_start(evaluation_month)
        threshold_start = _add_months(evaluation_start, -1)
        meta_start = _add_months(evaluation_start, -2)
        meta_split = meta_start + timedelta(days=20)
        early_stop_start = _add_months(evaluation_start, -3)
        training_start = _add_months(early_stop_start, -15)
        evaluation_end = _add_months(evaluation_start, 1)
        schedules.append(
            CausalMetaSchedule(
                evaluation_month=evaluation_month,
                base_training_start=training_start.isoformat(),
                base_training_end=_inclusive_end(early_stop_start),
                base_early_stop_start=early_stop_start.isoformat(),
                base_early_stop_end=_inclusive_end(meta_start),
                meta_fit_start=meta_start.isoformat(),
                meta_fit_end=_inclusive_end(meta_split),
                meta_early_stop_start=meta_split.isoformat(),
                meta_early_stop_end=_inclusive_end(threshold_start),
                threshold_calibration_start=threshold_start.isoformat(),
                threshold_calibration_end=_inclusive_end(evaluation_start),
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
    start: str,
    end: str,
) -> np.ndarray:
    start_ms = _date_ms(start)
    end_ms = _end_exclusive_ms(end)
    exit_time_ms = dataset.decision_time_ms + (HORIZON_MINUTES + 1) * MINUTE_MS
    return (
        (dataset.decision_time_ms >= start_ms)
        & (dataset.decision_time_ms < end_ms)
        & (exit_time_ms < end_ms)
    )


def meta_schedule_masks(
    dataset: DerivativesHurdleDataset,
    schedule: CausalMetaSchedule,
) -> dict[str, np.ndarray]:
    masks = {
        "base_training": _window_mask(
            dataset,
            start=schedule.base_training_start,
            end=schedule.base_training_end,
        ),
        "base_early_stop": _window_mask(
            dataset,
            start=schedule.base_early_stop_start,
            end=schedule.base_early_stop_end,
        ),
        "meta_fit": _window_mask(
            dataset,
            start=schedule.meta_fit_start,
            end=schedule.meta_fit_end,
        ),
        "meta_early_stop": _window_mask(
            dataset,
            start=schedule.meta_early_stop_start,
            end=schedule.meta_early_stop_end,
        ),
        "threshold_calibration": _window_mask(
            dataset,
            start=schedule.threshold_calibration_start,
            end=schedule.threshold_calibration_end,
        ),
        "evaluation": _window_mask(
            dataset,
            start=schedule.evaluation_start,
            end=schedule.evaluation_end,
        ),
    }
    combined = np.zeros(dataset.rows, dtype=np.int8)
    for mask in masks.values():
        combined += mask.astype(np.int8)
    if np.any(combined > 1):
        raise ValueError(f"{schedule.evaluation_month} Round 40 roles overlap")
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError(f"{schedule.evaluation_month} contains an empty Round 40 role")
    return masks


def _artifact(
    *,
    booster: lgb.Booster,
    reloaded: lgb.Booster,
    probe_features: np.ndarray,
    model_path: Path,
    model_id: str,
    model_role: str,
    evaluation_month: str,
    symbol: str,
    feature_names: Sequence[str],
    training_rows: int,
    early_stop_rows: int,
    positive_training_rows: int | None,
    positive_early_stop_rows: int | None,
    backend_kind: str,
    backend_device: str,
) -> CausalMetaModelArtifact:
    original = np.asarray(
        booster.predict(probe_features, num_iteration=booster.best_iteration)
    )
    restored = np.asarray(reloaded.predict(probe_features))
    reload_error = float(np.max(np.abs(original - restored)))
    if not math.isfinite(reload_error) or reload_error > 1e-12:
        raise RuntimeError(f"{model_id} reload error is {reload_error}")
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:20]
    return CausalMetaModelArtifact(
        model_id=model_id,
        model_role=model_role,
        evaluation_month=evaluation_month,
        symbol=symbol,
        feature_count=len(feature_names),
        training_rows=training_rows,
        early_stop_rows=early_stop_rows,
        positive_training_rows=positive_training_rows,
        positive_early_stop_rows=positive_early_stop_rows,
        best_iteration=int(booster.best_iteration),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(model_path),
        bytes=model_path.stat().st_size,
        sha256=_file_sha256(model_path),
        reload_max_abs_prediction_error=reload_error,
        top_feature_gain=tuple(
            (str(feature_names[int(index)]), float(gain[int(index)]))
            for index in order
        ),
    )


def _fit_primary_models(
    dataset: DerivativesHurdleDataset,
    *,
    schedule: CausalMetaSchedule,
    masks: Mapping[str, np.ndarray],
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, list[CausalMetaModelArtifact], set[str], set[str]]:
    features = dataset.feature_view("price_flow_only")
    feature_names = list(dataset.feature_names[: dataset.price_flow_feature_count])
    target = dataset.target_class[HORIZON_MINUTES]
    prediction_scope = (
        masks["base_early_stop"]
        | masks["meta_fit"]
        | masks["meta_early_stop"]
        | masks["threshold_calibration"]
        | masks["evaluation"]
    )
    probabilities = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
    artifacts: list[CausalMetaModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    for symbol_index, symbol in enumerate(SYMBOLS):
        symbol_mask = dataset.symbol_index == symbol_index
        training = masks["base_training"] & symbol_mask
        early_stop = masks["base_early_stop"] & symbol_mask
        prediction = prediction_scope & symbol_mask
        training_rows = int(np.count_nonzero(training))
        early_stop_rows = int(np.count_nonzero(early_stop))
        if training_rows == 0 or early_stop_rows == 0:
            raise ValueError(
                f"{schedule.evaluation_month}/{symbol} has an empty primary role"
            )
        parameters, kind, device = _lightgbm_parameters(
            compute_backend,
            objective="multiclass",
            seed=seed + symbol_index,
        )
        model_id = (
            f"round40_{schedule.evaluation_month.replace('-', '')}_primary_"
            f"{symbol.lower()}"
        )
        if progress is not None:
            progress(
                "round40_model_training",
                {
                    "model_id": model_id,
                    "model_role": "primary",
                    "status": "started",
                    "training_rows": training_rows,
                    "early_stop_rows": early_stop_rows,
                    "backend_kind": kind,
                    "backend_device": device,
                },
            )
        train_set = lgb.Dataset(
            features[training],
            label=target[training],
            feature_name=feature_names,
            free_raw_data=True,
        )
        validation_set = lgb.Dataset(
            features[early_stop],
            label=target[early_stop],
            feature_name=feature_names,
            reference=train_set,
            free_raw_data=True,
        )
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=1000,
            valid_sets=[validation_set],
            valid_names=["base_early_stop"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        probabilities[prediction] = np.asarray(
            booster.predict(
                features[prediction], num_iteration=booster.best_iteration
            ),
            dtype=np.float32,
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_id}.txt"
        booster.save_model(str(model_path), num_iteration=booster.best_iteration)
        reloaded = lgb.Booster(model_file=str(model_path))
        probe = np.flatnonzero(early_stop)[:4096]
        artifact = _artifact(
            booster=booster,
            reloaded=reloaded,
            probe_features=features[probe],
            model_path=model_path,
            model_id=model_id,
            model_role="primary",
            evaluation_month=schedule.evaluation_month,
            symbol=symbol,
            feature_names=feature_names,
            training_rows=training_rows,
            early_stop_rows=early_stop_rows,
            positive_training_rows=None,
            positive_early_stop_rows=None,
            backend_kind=kind,
            backend_device=device,
        )
        artifacts.append(artifact)
        kinds.add(kind)
        devices.add(device)
        if progress is not None:
            progress(
                "round40_model_training",
                {
                    "model_id": model_id,
                    "model_role": "primary",
                    "status": "complete",
                    "best_iteration": artifact.best_iteration,
                    "artifact_sha256": artifact.sha256,
                },
            )
    if not np.isfinite(probabilities[prediction_scope]).all():
        raise ValueError(
            f"{schedule.evaluation_month} primary probabilities are nonfinite"
        )
    return probabilities, artifacts, kinds, devices


def build_meta_features(
    dataset: DerivativesHurdleDataset,
    primary_probabilities: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Build the frozen 81-column causal meta-label matrix for selected rows."""

    indices = np.asarray(indices, dtype=np.int64)
    base = dataset.feature_view("price_flow_only")[indices]
    probabilities = np.asarray(primary_probabilities[indices], dtype=np.float32)
    if base.shape[1] != 71 or probabilities.shape != (indices.size, 3):
        raise ValueError("Round 40 meta feature inputs violate the frozen shape")
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    p_short = probabilities[:, 0]
    p_long = probabilities[:, 2]
    action_probability = np.maximum(p_short, p_long)
    margin = np.abs(p_long - p_short)
    entropy = -np.sum(clipped * np.log(clipped), axis=1).astype(np.float32)
    proposed_side = np.where(p_long > p_short, 1.0, -1.0).astype(np.float32)
    one_hot = np.eye(len(SYMBOLS), dtype=np.float32)[dataset.symbol_index[indices]]
    derived = np.column_stack(
        (
            probabilities,
            action_probability,
            margin,
            entropy,
            proposed_side,
            one_hot,
        )
    ).astype(np.float32)
    matrix = np.column_stack((base, derived)).astype(np.float32, copy=False)
    if matrix.shape != (indices.size, META_FEATURE_COUNT):
        raise RuntimeError("Round 40 meta feature cardinality is incorrect")
    if not np.isfinite(matrix).all():
        raise ValueError("Round 40 meta features contain nonfinite values")
    return matrix


def _meta_feature_names(dataset: DerivativesHurdleDataset) -> list[str]:
    names = list(dataset.feature_names[: dataset.price_flow_feature_count])
    names.extend(
        (
            "primary_p_short",
            "primary_p_abstain",
            "primary_p_long",
            "primary_action_probability",
            "primary_direction_margin",
            "primary_probability_entropy",
            "primary_proposed_side",
            "symbol_is_btcusdt",
            "symbol_is_ethusdt",
            "symbol_is_solusdt",
        )
    )
    if len(names) != META_FEATURE_COUNT:
        raise RuntimeError("Round 40 meta feature names violate the frozen count")
    return names


def _meta_labels(
    dataset: DerivativesHurdleDataset,
    primary_probabilities: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_short = primary_probabilities[indices, 0]
    p_long = primary_probabilities[indices, 2]
    direction = np.where(p_long > p_short, 1, -1).astype(np.int8)
    net = np.where(
        direction > 0,
        dataset.long_net_utility_bps[HORIZON_MINUTES][indices],
        dataset.short_net_utility_bps[HORIZON_MINUTES][indices],
    ).astype(np.float32)
    if not np.isfinite(net).all():
        raise ValueError("Round 40 meta labels contain nonfinite net utility")
    return (net > 0.0).astype(np.int8), direction


def _fit_meta_model(
    dataset: DerivativesHurdleDataset,
    *,
    schedule: CausalMetaSchedule,
    masks: Mapping[str, np.ndarray],
    primary_probabilities: np.ndarray,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
    round_number: int = ROUND,
) -> tuple[np.ndarray, CausalMetaModelArtifact, str, str, dict[str, object]]:
    fit_indices = np.flatnonzero(masks["meta_fit"])
    early_indices = np.flatnonzero(masks["meta_early_stop"])
    prediction_mask = (
        masks["meta_fit"]
        | masks["meta_early_stop"]
        | masks["threshold_calibration"]
        | masks["evaluation"]
    )
    prediction_indices = np.flatnonzero(prediction_mask)
    fit_features = build_meta_features(dataset, primary_probabilities, fit_indices)
    early_features = build_meta_features(
        dataset, primary_probabilities, early_indices
    )
    fit_labels, _ = _meta_labels(dataset, primary_probabilities, fit_indices)
    early_labels, _ = _meta_labels(dataset, primary_probabilities, early_indices)
    if np.unique(fit_labels).size != 2 or np.unique(early_labels).size != 2:
        raise ValueError(
            f"{schedule.evaluation_month} meta role lacks both target classes"
        )
    parameters, kind, device = _lightgbm_parameters(
        compute_backend,
        objective="binary",
        seed=seed,
    )
    parameters.update({"num_leaves": 31, "min_data_in_leaf": 200})
    model_id = (
        f"round{round_number}_{schedule.evaluation_month.replace('-', '')}_"
        "meta_shared"
    )
    if progress is not None:
        progress(
                f"round{round_number}_model_training",
            {
                "model_id": model_id,
                "model_role": "meta_label",
                "status": "started",
                "training_rows": int(fit_indices.size),
                "early_stop_rows": int(early_indices.size),
                "backend_kind": kind,
                "backend_device": device,
            },
        )
    feature_names = _meta_feature_names(dataset)
    train_set = lgb.Dataset(
        fit_features,
        label=fit_labels,
        feature_name=feature_names,
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        early_features,
        label=early_labels,
        feature_name=feature_names,
        reference=train_set,
        free_raw_data=True,
    )
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=1000,
        valid_sets=[validation_set],
        valid_names=["meta_early_stop"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    prediction_features = build_meta_features(
        dataset, primary_probabilities, prediction_indices
    )
    meta_probabilities = np.full(dataset.rows, np.nan, dtype=np.float32)
    meta_probabilities[prediction_indices] = np.asarray(
        booster.predict(
            prediction_features, num_iteration=booster.best_iteration
        ),
        dtype=np.float32,
    )
    if not np.isfinite(meta_probabilities[prediction_mask]).all():
        raise ValueError(
            f"{schedule.evaluation_month} meta probabilities are nonfinite"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    reloaded = lgb.Booster(model_file=str(model_path))
    probe_features = early_features[:4096]
    artifact = _artifact(
        booster=booster,
        reloaded=reloaded,
        probe_features=probe_features,
        model_path=model_path,
        model_id=model_id,
        model_role="meta_label",
        evaluation_month=schedule.evaluation_month,
        symbol="shared",
        feature_names=feature_names,
        training_rows=int(fit_indices.size),
        early_stop_rows=int(early_indices.size),
        positive_training_rows=int(np.count_nonzero(fit_labels)),
        positive_early_stop_rows=int(np.count_nonzero(early_labels)),
        backend_kind=kind,
        backend_device=device,
    )
    if progress is not None:
        progress(
            f"round{round_number}_model_training",
            {
                "model_id": model_id,
                "model_role": "meta_label",
                "status": "complete",
                "best_iteration": artifact.best_iteration,
                "artifact_sha256": artifact.sha256,
            },
        )
    diagnostics = {
        "fit": binary_classification_metrics(
            fit_labels, meta_probabilities[fit_indices]
        ),
        "early_stop": binary_classification_metrics(
            early_labels, meta_probabilities[early_indices]
        ),
    }
    return meta_probabilities, artifact, kind, device, diagnostics


def binary_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape or labels.size == 0:
        raise ValueError("binary metric arrays have incompatible shapes")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("binary metric probabilities are invalid")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(np.count_nonzero(negative))
    auc: float | None = None
    if n_positive and n_negative:
        ranks = rankdata(probabilities, method="average")
        auc = float(
            (np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative)
        )
    return {
        "rows": int(labels.size),
        "positive_rows": n_positive,
        "positive_fraction": float(np.mean(positive)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        ),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "roc_auc": auc,
    }


def replay_meta_actions(
    dataset: DerivativesHurdleDataset,
    primary_probabilities: np.ndarray,
    meta_probabilities: np.ndarray,
    *,
    eligibility_mask: np.ndarray,
    period_start: str,
    period_end: str,
    meta_probability_threshold: float | None,
    primary_margin_threshold: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> CapacityReplay:
    if meta_probability_threshold is None or primary_margin_threshold is None:
        empty_mask = np.zeros(dataset.rows, dtype=bool)
        direction = np.zeros(dataset.rows, dtype=np.int8)
        outcome = _replay_candidate_mask(
            dataset,
            candidate_mask=empty_mask,
            direction=direction,
            horizon_minutes=HORIZON_MINUTES,
            period_start=period_start,
            period_end=period_end,
            maximum_action_probability=None,
            direction_probability_margin=None,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        return CapacityReplay(outcome, 0, 0, 0, 0)
    p_short = primary_probabilities[:, 0]
    p_long = primary_probabilities[:, 2]
    margin = np.abs(p_long - p_short)
    direction = np.where(p_long > p_short, 1, -1).astype(np.int8)
    threshold_candidates = (
        eligibility_mask
        & np.isfinite(meta_probabilities)
        & np.isfinite(margin)
        & (meta_probabilities >= meta_probability_threshold)
        & (margin >= primary_margin_threshold)
    )
    selected_mask = np.zeros(dataset.rows, dtype=bool)
    overlap_rejections = 0
    capacity_rejections = 0
    maximum_entries = 0
    for symbol_index in range(len(SYMBOLS)):
        indices = np.flatnonzero(
            threshold_candidates & (dataset.symbol_index == symbol_index)
        )
        indices = indices[
            np.argsort(dataset.decision_time_ms[indices], kind="stable")
        ]
        next_available_ms = -1
        day_counts: dict[int, int] = {}
        for index in indices:
            entry_time_ms = int(dataset.decision_time_ms[index]) + MINUTE_MS
            if entry_time_ms < next_available_ms:
                overlap_rejections += 1
                continue
            utc_day = entry_time_ms // (86_400 * 1000)
            used = day_counts.get(utc_day, 0)
            if used >= MAXIMUM_ENTRIES_PER_SYMBOL_DAY:
                capacity_rejections += 1
                continue
            selected_mask[index] = True
            used += 1
            day_counts[utc_day] = used
            maximum_entries = max(maximum_entries, used)
            next_available_ms = entry_time_ms + HORIZON_MINUTES * MINUTE_MS
    outcome = _replay_candidate_mask(
        dataset,
        candidate_mask=selected_mask,
        direction=direction,
        horizon_minutes=HORIZON_MINUTES,
        period_start=period_start,
        period_end=period_end,
        maximum_action_probability=meta_probability_threshold,
        direction_probability_margin=primary_margin_threshold,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    metrics = ActionReplayMetrics(
        **{
            **outcome.metrics.asdict(),
            "candidate_rows": int(np.count_nonzero(threshold_candidates)),
            "overlap_rejections": overlap_rejections,
        }
    )
    return CapacityReplay(
        outcome=ReplayOutcome(
            metrics=metrics,
            selected_indices=outcome.selected_indices,
            selected_direction=outcome.selected_direction,
            net_return_bps=outcome.net_return_bps,
        ),
        threshold_candidate_rows=int(np.count_nonzero(threshold_candidates)),
        overlap_rejections=overlap_rejections,
        capacity_rejections=capacity_rejections,
        maximum_entries_in_one_symbol_day=maximum_entries,
    )


def _threshold_key(item: Mapping[str, object]) -> tuple[float, ...]:
    replay = item["capacity_replay"]["replay"]
    lower = replay["day_block_bootstrap_mean_net_bps_lower_95"]
    profit_factor = replay["profit_factor"]
    return (
        float(lower) if lower is not None else float("-inf"),
        float(replay["mean_net_bps"]),
        float(profit_factor) if profit_factor is not None else float("-inf"),
        -float(replay["total_trades"]),
        float(item["meta_probability_threshold"]),
        float(item["primary_margin_threshold"]),
    )


def _evaluate_month(
    dataset: DerivativesHurdleDataset,
    *,
    schedule: CausalMetaSchedule,
    masks: Mapping[str, np.ndarray],
    primary_probabilities: np.ndarray,
    meta_probabilities: np.ndarray,
    meta_diagnostics: Mapping[str, object],
) -> tuple[
    dict[str, object],
    tuple[float, float] | None,
    CapacityReplay,
]:
    traces: list[dict[str, object]] = []
    for meta_threshold in META_PROBABILITY_GRID:
        for margin_threshold in PRIMARY_MARGIN_GRID:
            initial = replay_meta_actions(
                dataset,
                primary_probabilities,
                meta_probabilities,
                eligibility_mask=masks["threshold_calibration"],
                period_start=schedule.threshold_calibration_start,
                period_end=schedule.threshold_calibration_end,
                meta_probability_threshold=meta_threshold,
                primary_margin_threshold=margin_threshold,
                bootstrap_samples=0,
                bootstrap_seed=4011,
            )
            metrics = initial.outcome.metrics
            support = (
                metrics.total_trades >= MONTHLY_MINIMUM_TRADES
                and metrics.active_utc_days >= MONTHLY_MINIMUM_ACTIVE_DAYS
                and metrics.maximum_single_symbol_fraction <= 0.50
                and all(
                    metrics.trades_by_symbol.get(symbol, 0)
                    >= MONTHLY_MINIMUM_TRADES_PER_SYMBOL
                    for symbol in SYMBOLS
                )
            )
            economic_candidate = (
                support
                and metrics.mean_net_bps > 0.0
                and metrics.profit_factor is not None
                and metrics.profit_factor >= 1.0
            )
            current = initial
            if economic_candidate:
                current = replay_meta_actions(
                    dataset,
                    primary_probabilities,
                    meta_probabilities,
                    eligibility_mask=masks["threshold_calibration"],
                    period_start=schedule.threshold_calibration_start,
                    period_end=schedule.threshold_calibration_end,
                    meta_probability_threshold=meta_threshold,
                    primary_margin_threshold=margin_threshold,
                    bootstrap_samples=500,
                    bootstrap_seed=4011,
                )
            lower = current.outcome.metrics.day_block_bootstrap_mean_net_bps_lower_95
            economic_gate = bool(
                economic_candidate and lower is not None and lower > 0.0
            )
            traces.append(
                {
                    "meta_probability_threshold": meta_threshold,
                    "primary_margin_threshold": margin_threshold,
                    "support_passed": support,
                    "economic_gate_passed": economic_gate,
                    "capacity_replay": current.evidence(),
                }
            )
    eligible = [item for item in traces if item["economic_gate_passed"]]
    selected = max(eligible, key=_threshold_key) if eligible else None
    threshold = (
        (
            float(selected["meta_probability_threshold"]),
            float(selected["primary_margin_threshold"]),
        )
        if selected is not None
        else None
    )
    evaluation = replay_meta_actions(
        dataset,
        primary_probabilities,
        meta_probabilities,
        eligibility_mask=masks["evaluation"],
        period_start=schedule.evaluation_start,
        period_end=schedule.evaluation_end,
        meta_probability_threshold=threshold[0] if threshold else None,
        primary_margin_threshold=threshold[1] if threshold else None,
        bootstrap_samples=500 if threshold else 0,
        bootstrap_seed=4012,
    )
    target = dataset.target_class[HORIZON_MINUTES]
    meta_threshold_indices = np.flatnonzero(masks["threshold_calibration"])
    meta_evaluation_indices = np.flatnonzero(masks["evaluation"])
    threshold_labels, _ = _meta_labels(
        dataset, primary_probabilities, meta_threshold_indices
    )
    evaluation_labels, _ = _meta_labels(
        dataset, primary_probabilities, meta_evaluation_indices
    )
    result = {
        "evaluation_month": schedule.evaluation_month,
        "schedule": schedule.asdict(),
        "role_rows": {
            role: int(np.count_nonzero(mask)) for role, mask in masks.items()
        },
        "primary_classification": {
            role: classification_metrics(
                target[mask], primary_probabilities[mask]
            ).asdict()
            for role, mask in (
                ("base_early_stop", masks["base_early_stop"]),
                ("threshold_calibration", masks["threshold_calibration"]),
                ("evaluation", masks["evaluation"]),
            )
        },
        "meta_classification": {
            **meta_diagnostics,
            "threshold_calibration": binary_classification_metrics(
                threshold_labels, meta_probabilities[meta_threshold_indices]
            ),
            "evaluation": binary_classification_metrics(
                evaluation_labels, meta_probabilities[meta_evaluation_indices]
            ),
        },
        "threshold_trace": traces,
        "selected_threshold": (
            {
                "meta_probability_threshold": threshold[0],
                "primary_margin_threshold": threshold[1],
            }
            if threshold
            else None
        ),
        "calibration_replay": (
            selected["capacity_replay"] if selected else None
        ),
        "evaluation_replay": evaluation.evidence(),
    }
    return result, threshold, evaluation


def run_causal_meta_label_screen(
    dataset: DerivativesHurdleDataset,
    *,
    model_dir: str | Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> Round40ModelScreen:
    """Run the frozen six-month causal stacked meta-label viability screen."""

    if dataset.price_flow_feature_count != 71 or dataset.rows != 1_098_105:
        raise ValueError("Round 40 dataset identity violates the frozen design")
    root = Path(model_dir)
    schedules = frozen_meta_schedules()
    artifacts: list[CausalMetaModelArtifact] = []
    monthly_results: list[Mapping[str, object]] = []
    monthly_thresholds: dict[str, tuple[float, float] | None] = {}
    stitched_primary = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
    stitched_meta = np.full(dataset.rows, np.nan, dtype=np.float32)
    stitched_selected = np.zeros(dataset.rows, dtype=bool)
    stitched_direction = np.zeros(dataset.rows, dtype=np.int8)
    backend_kinds: set[str] = set()
    backend_devices: set[str] = set()
    for month_index, schedule in enumerate(schedules, start=1):
        if progress is not None:
            progress(
                "round40_month",
                {
                    "evaluation_month": schedule.evaluation_month,
                    "month_index": month_index,
                    "month_total": len(schedules),
                    "status": "started",
                },
            )
        masks = meta_schedule_masks(dataset, schedule)
        primary, primary_artifacts, kinds, devices = _fit_primary_models(
            dataset,
            schedule=schedule,
            masks=masks,
            model_dir=root,
            compute_backend=compute_backend,
            seed=SEED + month_index * 10,
            progress=progress,
        )
        temperature, _, _ = _fit_temperature(
            primary[masks["base_early_stop"]],
            dataset.target_class[HORIZON_MINUTES][masks["base_early_stop"]],
        )
        primary_prediction_scope = (
            masks["base_early_stop"]
            | masks["meta_fit"]
            | masks["meta_early_stop"]
            | masks["threshold_calibration"]
            | masks["evaluation"]
        )
        calibrated_primary = np.full_like(primary, np.nan, dtype=np.float32)
        calibrated_primary[primary_prediction_scope] = _temperature_scale(
            primary[primary_prediction_scope], temperature
        ).astype(np.float32)
        meta, meta_artifact, kind, device, meta_diagnostics = _fit_meta_model(
            dataset,
            schedule=schedule,
            masks=masks,
            primary_probabilities=calibrated_primary,
            model_dir=root,
            compute_backend=compute_backend,
            seed=SEED + month_index * 10 + 5,
            progress=progress,
        )
        month_result, threshold, evaluation = _evaluate_month(
            dataset,
            schedule=schedule,
            masks=masks,
            primary_probabilities=calibrated_primary,
            meta_probabilities=meta,
            meta_diagnostics=meta_diagnostics,
        )
        monthly_results.append(month_result)
        monthly_thresholds[schedule.evaluation_month] = threshold
        stitched_primary[masks["evaluation"]] = calibrated_primary[
            masks["evaluation"]
        ]
        stitched_meta[masks["evaluation"]] = meta[masks["evaluation"]]
        if threshold is not None:
            stitched_selected[evaluation.outcome.selected_indices] = True
            stitched_direction[evaluation.outcome.selected_indices] = (
                evaluation.outcome.selected_direction
            )
        artifacts.extend(primary_artifacts)
        artifacts.append(meta_artifact)
        backend_kinds.update(kinds)
        backend_kinds.add(kind)
        backend_devices.update(devices)
        backend_devices.add(device)
        if progress is not None:
            progress(
                "round40_month",
                {
                    "evaluation_month": schedule.evaluation_month,
                    "month_index": month_index,
                    "month_total": len(schedules),
                    "status": "complete",
                    "threshold_selected": threshold is not None,
                    "evaluation_trades": evaluation.outcome.metrics.total_trades,
                    "evaluation_mean_net_bps": (
                        evaluation.outcome.metrics.mean_net_bps
                    ),
                },
            )
        del primary, calibrated_primary, meta
    aggregate = _replay_candidate_mask(
        dataset,
        candidate_mask=stitched_selected,
        direction=stitched_direction,
        horizon_minutes=HORIZON_MINUTES,
        period_start=schedules[0].evaluation_start,
        period_end=schedules[-1].evaluation_end,
        maximum_action_probability=None,
        direction_probability_margin=None,
        bootstrap_samples=2000,
        bootstrap_seed=4013,
    )
    selected_months = sum(value is not None for value in monthly_thresholds.values())
    months_with_trades = sum(
        int(result["evaluation_replay"]["replay"]["total_trades"]) > 0
        for result in monthly_results
    )
    month_fraction = _maximum_single_month_positive_fraction(dataset, aggregate)
    support_reasons, gate_reasons = _aggregate_gate_reasons(
        aggregate.metrics,
        selected_months=selected_months,
        months_with_trades=months_with_trades,
        maximum_single_month_positive_fraction=month_fraction,
    )
    passed = not gate_reasons
    candidate_id = "causal_per_symbol_direct_h30_shared_profitability_meta"
    candidate_result = {
        "candidate_id": candidate_id,
        "architecture": "per_symbol_direct_multiclass_plus_shared_binary_meta",
        "horizon_minutes": HORIZON_MINUTES,
        "monthly_results": monthly_results,
        "selected_threshold_months": selected_months,
        "months_with_trades": months_with_trades,
        "aggregate_replay": aggregate.metrics.asdict(),
        "maximum_single_month_fraction_of_positive_net_bps": month_fraction,
        "support_passed": not support_reasons,
        "support_reasons": support_reasons,
        "aggregate_gate_passed": passed,
        "aggregate_gate_reasons": gate_reasons,
    }
    candidate = CausalMetaCandidate(
        candidate_id=candidate_id,
        horizon_minutes=HORIZON_MINUTES,
        primary_probabilities=stitched_primary,
        meta_probabilities=stitched_meta,
        monthly_thresholds=monthly_thresholds,
        evaluation=aggregate,
    )
    if (
        len(artifacts) != 24
        or len(backend_kinds) != 1
        or len(backend_devices) != 1
    ):
        raise RuntimeError(
            "Round 40 model artifact or backend cardinality violates the design"
        )
    schedule_evidence: list[Mapping[str, object]] = []
    for schedule in schedules:
        masks = meta_schedule_masks(dataset, schedule)
        schedule_evidence.append(
            {
                **schedule.asdict(),
                "role_rows": {
                    role: int(np.count_nonzero(mask))
                    for role, mask in masks.items()
                },
            }
        )
    return Round40ModelScreen(
        schedules=tuple(schedule_evidence),
        candidate_result=candidate_result,
        model_artifacts=tuple(artifacts),
        candidate=candidate,
        passed_candidate=candidate if passed else None,
        backend_kind=next(iter(backend_kinds)),
        backend_device=next(iter(backend_devices)),
    )


__all__ = [
    "EVALUATION_MONTHS",
    "HORIZON_MINUTES",
    "MAXIMUM_ENTRIES_PER_SYMBOL_DAY",
    "META_FEATURE_COUNT",
    "META_PROBABILITY_GRID",
    "PRIMARY_MARGIN_GRID",
    "CapacityReplay",
    "CausalMetaCandidate",
    "CausalMetaModelArtifact",
    "CausalMetaSchedule",
    "Round40ModelScreen",
    "binary_classification_metrics",
    "build_meta_features",
    "frozen_meta_schedules",
    "meta_schedule_masks",
    "replay_meta_actions",
    "run_causal_meta_label_screen",
]
