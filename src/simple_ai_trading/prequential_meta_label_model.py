"""Longer-history prequential stacked meta-labeling for Round 41."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Mapping

import lightgbm as lgb
import numpy as np

from .causal_meta_label_model import (
    HORIZON_MINUTES,
    META_PROBABILITY_GRID,
    PRIMARY_MARGIN_GRID,
    CausalMetaCandidate,
    CausalMetaModelArtifact,
    ProgressCallback,
    _aggregate_gate_reasons,
    _artifact,
    _fit_meta_model,
    _maximum_single_month_positive_fraction,
    _replay_candidate_mask,
    _threshold_key,
    binary_classification_metrics,
    replay_meta_actions,
)
from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import DerivativesHurdleDataset
from .derivatives_hurdle_model import (
    _fit_temperature,
    _lightgbm_parameters,
    _temperature_scale,
    classification_metrics,
)


ROUND = 41
SEED = 4101
PRIMARY_TARGET_MONTHS = (
    "2023-11",
    "2023-12",
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06",
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
)
EVALUATION_MONTHS = (
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
)
MONTHLY_MINIMUM_TRADES = 30
MONTHLY_MINIMUM_TRADES_PER_SYMBOL = 5
MONTHLY_MINIMUM_ACTIVE_DAYS = 15


@dataclass(frozen=True)
class PrequentialPrimarySchedule:
    target_month: str
    training_start: str
    training_end: str
    early_stop_start: str
    early_stop_end: str
    prediction_start: str
    prediction_end: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PrequentialMetaSchedule:
    evaluation_month: str
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
class Round41ModelScreen:
    primary_schedules: tuple[Mapping[str, object], ...]
    meta_schedules: tuple[Mapping[str, object], ...]
    candidate_result: Mapping[str, object]
    model_artifacts: tuple[CausalMetaModelArtifact, ...]
    candidate: CausalMetaCandidate
    passed_candidate: CausalMetaCandidate | None
    backend_kind: str
    backend_device: str


def _month_start(value: str) -> date:
    parsed = datetime.strptime(value, "%Y-%m").date()
    return parsed.replace(day=1)


def _add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _inclusive_end(exclusive: date) -> str:
    return date.fromordinal(exclusive.toordinal() - 1).isoformat()


def frozen_primary_schedules() -> tuple[PrequentialPrimarySchedule, ...]:
    schedules: list[PrequentialPrimarySchedule] = []
    for target_month in PRIMARY_TARGET_MONTHS:
        prediction_start = _month_start(target_month)
        early_stop_start = _add_months(prediction_start, -1)
        training_start = _add_months(early_stop_start, -15)
        prediction_end = _add_months(prediction_start, 1)
        schedules.append(
            PrequentialPrimarySchedule(
                target_month=target_month,
                training_start=training_start.isoformat(),
                training_end=_inclusive_end(early_stop_start),
                early_stop_start=early_stop_start.isoformat(),
                early_stop_end=_inclusive_end(prediction_start),
                prediction_start=prediction_start.isoformat(),
                prediction_end=_inclusive_end(prediction_end),
            )
        )
    return tuple(schedules)


def frozen_meta_schedules() -> tuple[PrequentialMetaSchedule, ...]:
    schedules: list[PrequentialMetaSchedule] = []
    for evaluation_month in EVALUATION_MONTHS:
        evaluation_start = _month_start(evaluation_month)
        threshold_start = _add_months(evaluation_start, -1)
        early_stop_start = _add_months(evaluation_start, -2)
        fit_start = _add_months(evaluation_start, -8)
        evaluation_end = _add_months(evaluation_start, 1)
        schedules.append(
            PrequentialMetaSchedule(
                evaluation_month=evaluation_month,
                meta_fit_start=fit_start.isoformat(),
                meta_fit_end=_inclusive_end(early_stop_start),
                meta_early_stop_start=early_stop_start.isoformat(),
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


def primary_schedule_masks(
    dataset: DerivativesHurdleDataset,
    schedule: PrequentialPrimarySchedule,
) -> dict[str, np.ndarray]:
    masks = {
        "training": _window_mask(
            dataset, start=schedule.training_start, end=schedule.training_end
        ),
        "early_stop": _window_mask(
            dataset, start=schedule.early_stop_start, end=schedule.early_stop_end
        ),
        "prediction": _window_mask(
            dataset, start=schedule.prediction_start, end=schedule.prediction_end
        ),
    }
    combined = np.zeros(dataset.rows, dtype=np.int8)
    for mask in masks.values():
        combined += mask.astype(np.int8)
    if np.any(combined > 1):
        raise ValueError(f"{schedule.target_month} primary roles overlap")
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError(f"{schedule.target_month} has an empty primary role")
    return masks


def meta_schedule_masks(
    dataset: DerivativesHurdleDataset,
    schedule: PrequentialMetaSchedule,
) -> dict[str, np.ndarray]:
    masks = {
        "meta_fit": _window_mask(
            dataset, start=schedule.meta_fit_start, end=schedule.meta_fit_end
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
            dataset, start=schedule.evaluation_start, end=schedule.evaluation_end
        ),
    }
    combined = np.zeros(dataset.rows, dtype=np.int8)
    for mask in masks.values():
        combined += mask.astype(np.int8)
    if np.any(combined > 1):
        raise ValueError(f"{schedule.evaluation_month} meta roles overlap")
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError(f"{schedule.evaluation_month} has an empty meta role")
    return masks


def _fit_primary_month(
    dataset: DerivativesHurdleDataset,
    *,
    schedule: PrequentialPrimarySchedule,
    masks: Mapping[str, np.ndarray],
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[
    np.ndarray,
    list[CausalMetaModelArtifact],
    set[str],
    set[str],
    dict[str, object],
]:
    features = dataset.feature_view("price_flow_only")
    feature_names = list(dataset.feature_names[: dataset.price_flow_feature_count])
    target = dataset.target_class[HORIZON_MINUTES]
    probabilities = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
    artifacts: list[CausalMetaModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    for symbol_index, symbol in enumerate(SYMBOLS):
        symbol_mask = dataset.symbol_index == symbol_index
        training = masks["training"] & symbol_mask
        early_stop = masks["early_stop"] & symbol_mask
        prediction = masks["prediction"] & symbol_mask
        training_rows = int(np.count_nonzero(training))
        early_stop_rows = int(np.count_nonzero(early_stop))
        parameters, kind, device = _lightgbm_parameters(
            compute_backend,
            objective="multiclass",
            seed=seed + symbol_index,
        )
        model_id = (
            f"round41_{schedule.target_month.replace('-', '')}_primary_"
            f"{symbol.lower()}"
        )
        if progress is not None:
            progress(
                "round41_model_training",
                {
                    "model_id": model_id,
                    "model_role": "prequential_primary",
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
            valid_names=["early_stop"],
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
            model_role="prequential_primary",
            evaluation_month=schedule.target_month,
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
                "round41_model_training",
                {
                    "model_id": model_id,
                    "model_role": "prequential_primary",
                    "status": "complete",
                    "best_iteration": artifact.best_iteration,
                    "artifact_sha256": artifact.sha256,
                },
            )
    if not np.isfinite(probabilities[masks["prediction"]]).all():
        raise ValueError(f"{schedule.target_month} primary predictions are nonfinite")
    temperature, before, after = _fit_temperature(
        _predict_early_stop(
            dataset,
            schedule=schedule,
            model_artifacts=artifacts,
            features=features,
            mask=masks["early_stop"],
        ),
        target[masks["early_stop"]],
    )
    probabilities[masks["prediction"]] = _temperature_scale(
        probabilities[masks["prediction"]], temperature
    ).astype(np.float32)
    diagnostics = {
        "target_month": schedule.target_month,
        "temperature": temperature,
        "early_stop_log_loss_before": before,
        "early_stop_log_loss_after": after,
        "prediction_classification": classification_metrics(
            target[masks["prediction"]], probabilities[masks["prediction"]]
        ).asdict(),
    }
    return probabilities, artifacts, kinds, devices, diagnostics


def _predict_early_stop(
    dataset: DerivativesHurdleDataset,
    *,
    schedule: PrequentialPrimarySchedule,
    model_artifacts: list[CausalMetaModelArtifact],
    features: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    probabilities = np.full((int(np.count_nonzero(mask)), 3), np.nan, dtype=np.float32)
    masked_rows = np.flatnonzero(mask)
    row_position = np.full(dataset.rows, -1, dtype=np.int64)
    row_position[masked_rows] = np.arange(masked_rows.size)
    for symbol_index, symbol in enumerate(SYMBOLS):
        artifact = next(item for item in model_artifacts if item.symbol == symbol)
        booster = lgb.Booster(model_file=artifact.path)
        rows = np.flatnonzero(mask & (dataset.symbol_index == symbol_index))
        probabilities[row_position[rows]] = np.asarray(
            booster.predict(features[rows]), dtype=np.float32
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{schedule.target_month} early-stop probabilities are nonfinite"
        )
    return probabilities


def _meta_labels(
    dataset: DerivativesHurdleDataset,
    primary_probabilities: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    direction = np.where(
        primary_probabilities[indices, 2] > primary_probabilities[indices, 0],
        1,
        -1,
    )
    net = np.where(
        direction > 0,
        dataset.long_net_utility_bps[HORIZON_MINUTES][indices],
        dataset.short_net_utility_bps[HORIZON_MINUTES][indices],
    )
    return (net > 0.0).astype(np.int8)


def _evaluate_meta_month(
    dataset: DerivativesHurdleDataset,
    *,
    schedule: PrequentialMetaSchedule,
    masks: Mapping[str, np.ndarray],
    primary_probabilities: np.ndarray,
    meta_probabilities: np.ndarray,
    meta_diagnostics: Mapping[str, object],
) -> tuple[dict[str, object], tuple[float, float] | None, object]:
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
                bootstrap_seed=4111,
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
                    bootstrap_seed=4111,
                )
            lower = current.outcome.metrics.day_block_bootstrap_mean_net_bps_lower_95
            traces.append(
                {
                    "meta_probability_threshold": meta_threshold,
                    "primary_margin_threshold": margin_threshold,
                    "support_passed": support,
                    "economic_gate_passed": bool(
                        economic_candidate and lower is not None and lower > 0.0
                    ),
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
        if selected
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
        bootstrap_seed=4112,
    )
    classifications: dict[str, object] = dict(meta_diagnostics)
    for role in ("threshold_calibration", "evaluation"):
        indices = np.flatnonzero(masks[role])
        classifications[role] = binary_classification_metrics(
            _meta_labels(dataset, primary_probabilities, indices),
            meta_probabilities[indices],
        )
    result = {
        "evaluation_month": schedule.evaluation_month,
        "schedule": schedule.asdict(),
        "role_rows": {
            role: int(np.count_nonzero(mask)) for role, mask in masks.items()
        },
        "meta_classification": classifications,
        "threshold_trace": traces,
        "selected_threshold": (
            {
                "meta_probability_threshold": threshold[0],
                "primary_margin_threshold": threshold[1],
            }
            if threshold
            else None
        ),
        "calibration_replay": selected["capacity_replay"] if selected else None,
        "evaluation_replay": evaluation.evidence(),
    }
    return result, threshold, evaluation


def run_prequential_meta_label_screen(
    dataset: DerivativesHurdleDataset,
    *,
    model_dir: str | Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> Round41ModelScreen:
    """Run the frozen prequential primary panel and six meta refits."""

    if dataset.price_flow_feature_count != 71 or dataset.rows != 1_098_105:
        raise ValueError("Round 41 dataset identity violates the frozen design")
    root = Path(model_dir)
    primary_panel = np.full((dataset.rows, 3), np.nan, dtype=np.float32)
    primary_schedules = frozen_primary_schedules()
    primary_evidence: list[Mapping[str, object]] = []
    artifacts: list[CausalMetaModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    for index, schedule in enumerate(primary_schedules, start=1):
        masks = primary_schedule_masks(dataset, schedule)
        current, current_artifacts, current_kinds, current_devices, diagnostics = (
            _fit_primary_month(
                dataset,
                schedule=schedule,
                masks=masks,
                model_dir=root,
                compute_backend=compute_backend,
                seed=SEED + index * 10,
                progress=progress,
            )
        )
        primary_panel[masks["prediction"]] = current[masks["prediction"]]
        artifacts.extend(current_artifacts)
        kinds.update(current_kinds)
        devices.update(current_devices)
        primary_evidence.append(
            {
                **schedule.asdict(),
                "role_rows": {
                    role: int(np.count_nonzero(mask))
                    for role, mask in masks.items()
                },
                "diagnostics": diagnostics,
            }
        )
        if progress is not None:
            progress(
                "round41_primary_month",
                {
                    "status": "complete",
                    "target_month": schedule.target_month,
                    "month_index": index,
                    "month_total": len(primary_schedules),
                },
            )
        del current
    meta_schedules = frozen_meta_schedules()
    monthly_results: list[Mapping[str, object]] = []
    monthly_thresholds: dict[str, tuple[float, float] | None] = {}
    stitched_meta = np.full(dataset.rows, np.nan, dtype=np.float32)
    stitched_selected = np.zeros(dataset.rows, dtype=bool)
    stitched_direction = np.zeros(dataset.rows, dtype=np.int8)
    panel_available = np.isfinite(primary_panel).all(axis=1)
    for index, schedule in enumerate(meta_schedules, start=1):
        masks = {
            role: mask & panel_available
            for role, mask in meta_schedule_masks(dataset, schedule).items()
        }
        if any(not np.any(mask) for mask in masks.values()):
            raise ValueError(
                f"{schedule.evaluation_month} has an empty available panel role"
            )
        scope = (
            masks["meta_fit"]
            | masks["meta_early_stop"]
            | masks["threshold_calibration"]
            | masks["evaluation"]
        )
        if not np.isfinite(primary_panel[scope]).all():
            raise ValueError(
                f"{schedule.evaluation_month} prequential panel is incomplete"
            )
        meta, artifact, kind, device, diagnostics = _fit_meta_model(
            dataset,
            schedule=schedule,  # type: ignore[arg-type]
            masks=masks,
            primary_probabilities=primary_panel,
            model_dir=root,
            compute_backend=compute_backend,
            seed=SEED + 1000 + index * 10,
            progress=progress,
            round_number=ROUND,
        )
        month_result, threshold, evaluation = _evaluate_meta_month(
            dataset,
            schedule=schedule,
            masks=masks,
            primary_probabilities=primary_panel,
            meta_probabilities=meta,
            meta_diagnostics=diagnostics,
        )
        monthly_results.append(month_result)
        monthly_thresholds[schedule.evaluation_month] = threshold
        stitched_meta[masks["evaluation"]] = meta[masks["evaluation"]]
        if threshold:
            stitched_selected[evaluation.outcome.selected_indices] = True
            stitched_direction[evaluation.outcome.selected_indices] = (
                evaluation.outcome.selected_direction
            )
        artifacts.append(artifact)
        kinds.add(kind)
        devices.add(device)
        if progress is not None:
            progress(
                "round41_meta_month",
                {
                    "status": "complete",
                    "evaluation_month": schedule.evaluation_month,
                    "month_index": index,
                    "month_total": len(meta_schedules),
                    "threshold_selected": threshold is not None,
                    "evaluation_trades": evaluation.outcome.metrics.total_trades,
                    "evaluation_mean_net_bps": evaluation.outcome.metrics.mean_net_bps,
                },
            )
        del meta
    aggregate = _replay_candidate_mask(
        dataset,
        candidate_mask=stitched_selected,
        direction=stitched_direction,
        horizon_minutes=HORIZON_MINUTES,
        period_start=meta_schedules[0].evaluation_start,
        period_end=meta_schedules[-1].evaluation_end,
        maximum_action_probability=None,
        direction_probability_margin=None,
        bootstrap_samples=2000,
        bootstrap_seed=4113,
    )
    selected_months = sum(value is not None for value in monthly_thresholds.values())
    months_with_trades = sum(
        int(item["evaluation_replay"]["replay"]["total_trades"]) > 0
        for item in monthly_results
    )
    month_fraction = _maximum_single_month_positive_fraction(dataset, aggregate)
    support_reasons, gate_reasons = _aggregate_gate_reasons(
        aggregate.metrics,
        selected_months=selected_months,
        months_with_trades=months_with_trades,
        maximum_single_month_positive_fraction=month_fraction,
    )
    candidate_id = "prequential_h30_shared_profitability_meta_6m"
    result = {
        "candidate_id": candidate_id,
        "architecture": "prequential_per_symbol_direct_plus_shared_binary_meta",
        "horizon_minutes": HORIZON_MINUTES,
        "monthly_results": monthly_results,
        "selected_threshold_months": selected_months,
        "months_with_trades": months_with_trades,
        "aggregate_replay": aggregate.metrics.asdict(),
        "maximum_single_month_fraction_of_positive_net_bps": month_fraction,
        "support_passed": not support_reasons,
        "support_reasons": support_reasons,
        "aggregate_gate_passed": not gate_reasons,
        "aggregate_gate_reasons": gate_reasons,
    }
    candidate = CausalMetaCandidate(
        candidate_id=candidate_id,
        horizon_minutes=HORIZON_MINUTES,
        primary_probabilities=primary_panel,
        meta_probabilities=stitched_meta,
        monthly_thresholds=monthly_thresholds,
        evaluation=aggregate,
    )
    if len(artifacts) != 48 or len(kinds) != 1 or len(devices) != 1:
        raise RuntimeError("Round 41 artifact or backend cardinality is invalid")
    return Round41ModelScreen(
        primary_schedules=tuple(primary_evidence),
        meta_schedules=tuple(
            {
                **schedule.asdict(),
                "role_rows": {
                    role: int(np.count_nonzero(mask & panel_available))
                    for role, mask in meta_schedule_masks(
                        dataset, schedule
                    ).items()
                },
            }
            for schedule in meta_schedules
        ),
        candidate_result=result,
        model_artifacts=tuple(artifacts),
        candidate=candidate,
        passed_candidate=candidate if not gate_reasons else None,
        backend_kind=next(iter(kinds)),
        backend_device=next(iter(devices)),
    )


__all__ = [
    "EVALUATION_MONTHS",
    "PRIMARY_TARGET_MONTHS",
    "PrequentialMetaSchedule",
    "PrequentialPrimarySchedule",
    "Round41ModelScreen",
    "frozen_meta_schedules",
    "frozen_primary_schedules",
    "meta_schedule_masks",
    "primary_schedule_masks",
    "run_prequential_meta_label_screen",
]
