"""Causal daily refit and abstaining policy utilities for gross models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .microstructure_architecture import GrossActionScoreBatch
from .microstructure_features import (
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .microstructure_model import TradingMetrics, _simulate_non_overlapping_trace


WALK_FORWARD_SCHEMA_VERSION = "gross-daily-walk-forward-v1"
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class WalkForwardFitSpec:
    candidate_id: str
    training_window_days: int
    early_stop_days: int
    calibration_days: int
    recency_half_life_days: float | None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("walk-forward candidate_id cannot be empty")
        if self.training_window_days != 0 and self.training_window_days < 10:
            raise ValueError(
                "walk-forward training window must be zero or at least 10 days"
            )
        if not 1 <= self.early_stop_days <= 10:
            raise ValueError("walk-forward early-stop window is invalid")
        if not 1 <= self.calibration_days <= 10:
            raise ValueError("walk-forward calibration window is invalid")
        if self.recency_half_life_days is not None and (
            not math.isfinite(self.recency_half_life_days)
            or self.recency_half_life_days < 1.0
        ):
            raise ValueError("walk-forward recency half-life is invalid")


@dataclass(frozen=True)
class WalkForwardDayPlan:
    evaluation_day_id: int
    train_indexes: np.ndarray
    early_stop_indexes: np.ndarray
    calibration_indexes: np.ndarray
    evaluation_indexes: np.ndarray
    evidence: Mapping[str, object]


@dataclass(frozen=True)
class ActionTrace:
    metrics: TradingMetrics
    gross_bps: tuple[float, ...]
    net_bps: tuple[float, ...]
    sides: tuple[int, ...]
    timestamps_ms: tuple[int, ...]
    source_endpoint_indexes: tuple[int, ...]
    portfolio_claim: bool = False
    trading_authority: bool = False

    @property
    def mean_gross_bps(self) -> float:
        return float(np.mean(self.gross_bps)) if self.gross_bps else 0.0

    @property
    def total_gross_bps(self) -> float:
        return float(np.sum(self.gross_bps)) if self.gross_bps else 0.0

    def asdict(self) -> dict[str, object]:
        output = asdict(self)
        output["mean_gross_bps"] = self.mean_gross_bps
        output["total_gross_bps"] = self.total_gross_bps
        return output


@dataclass(frozen=True)
class ThresholdSelection:
    accepted: bool
    threshold: float | None
    quantile: float | None
    selected_trace: ActionTrace
    candidates: tuple[Mapping[str, object], ...]
    rejection_reasons: tuple[str, ...]
    trading_authority: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "threshold": self.threshold,
            "quantile": self.quantile,
            "selected_trace": self.selected_trace.asdict(),
            "candidates": [dict(value) for value in self.candidates],
            "rejection_reasons": list(self.rejection_reasons),
            "trading_authority": self.trading_authority,
        }


def _window_indexes(
    dataset: MicrostructureDataset,
    event_mask: np.ndarray,
    *,
    first_ms: int,
    end_exclusive_ms: int,
    next_role_start_ms: int | None,
) -> np.ndarray:
    indexes = np.flatnonzero(
        event_mask
        & (dataset.decision_time_ms >= int(first_ms))
        & (dataset.decision_time_ms < int(end_exclusive_ms))
    ).astype(np.int64)
    if next_role_start_ms is not None:
        indexes = indexes[
            (dataset.long_exit_time_ms[indexes] < int(next_role_start_ms))
            & (dataset.short_exit_time_ms[indexes] < int(next_role_start_ms))
        ]
    return indexes


def plan_walk_forward_day(
    dataset: MicrostructureDataset,
    event_mask: np.ndarray,
    *,
    evaluation_day_id: int,
    corpus_start_day_id: int,
    spec: WalkForwardFitSpec,
    minimum_rows: int = 256,
) -> WalkForwardDayPlan:
    """Build purged train/stop/calibration/evaluation roles for one day."""

    validate_microstructure_dataset(dataset)
    mask = np.asarray(event_mask, dtype=bool)
    if mask.shape != (dataset.rows,):
        raise ValueError("walk-forward event mask shape is invalid")
    evaluation_day = int(evaluation_day_id)
    corpus_day = int(corpus_start_day_id)
    minimum = int(minimum_rows)
    if evaluation_day <= corpus_day or minimum < 32:
        raise ValueError("walk-forward day planning inputs are invalid")
    evaluation_start = evaluation_day * _DAY_MS
    calibration_start = (evaluation_day - int(spec.calibration_days)) * _DAY_MS
    early_stop_start = (
        evaluation_day - int(spec.calibration_days) - int(spec.early_stop_days)
    ) * _DAY_MS
    if spec.training_window_days == 0:
        train_start = corpus_day * _DAY_MS
    else:
        train_start = (
            max(
                corpus_day,
                evaluation_day
                - int(spec.calibration_days)
                - int(spec.early_stop_days)
                - int(spec.training_window_days),
            )
            * _DAY_MS
        )
    train = _window_indexes(
        dataset,
        mask,
        first_ms=train_start,
        end_exclusive_ms=early_stop_start,
        next_role_start_ms=early_stop_start,
    )
    early_stop = _window_indexes(
        dataset,
        mask,
        first_ms=early_stop_start,
        end_exclusive_ms=calibration_start,
        next_role_start_ms=calibration_start,
    )
    calibration = _window_indexes(
        dataset,
        mask,
        first_ms=calibration_start,
        end_exclusive_ms=evaluation_start,
        next_role_start_ms=evaluation_start,
    )
    evaluation = _window_indexes(
        dataset,
        mask,
        first_ms=evaluation_start,
        end_exclusive_ms=evaluation_start + _DAY_MS,
        next_role_start_ms=evaluation_start + _DAY_MS,
    )
    roles = {
        "train": train,
        "early_stop": early_stop,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    insufficient = {
        name: len(indexes) for name, indexes in roles.items() if len(indexes) < minimum
    }
    if insufficient:
        raise ValueError(f"walk-forward role support is insufficient: {insufficient}")
    evidence = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "evaluation_day_id": evaluation_day,
        "train_start_day_id": train_start // _DAY_MS,
        "early_stop_start_day_id": early_stop_start // _DAY_MS,
        "calibration_start_day_id": calibration_start // _DAY_MS,
        "evaluation_start_day_id": evaluation_day,
        "training_rows": len(train),
        "early_stop_rows": len(early_stop),
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "train_last_exit_ms": int(
            max(
                np.max(dataset.long_exit_time_ms[train]),
                np.max(dataset.short_exit_time_ms[train]),
            )
        ),
        "early_stop_last_exit_ms": int(
            max(
                np.max(dataset.long_exit_time_ms[early_stop]),
                np.max(dataset.short_exit_time_ms[early_stop]),
            )
        ),
        "calibration_last_exit_ms": int(
            max(
                np.max(dataset.long_exit_time_ms[calibration]),
                np.max(dataset.short_exit_time_ms[calibration]),
            )
        ),
        "terminal_holdout_accessed": False,
    }
    if (
        int(evidence["train_last_exit_ms"]) >= early_stop_start
        or int(evidence["early_stop_last_exit_ms"]) >= calibration_start
        or int(evidence["calibration_last_exit_ms"]) >= evaluation_start
    ):
        raise ValueError("walk-forward purge contract failed")
    return WalkForwardDayPlan(
        evaluation_day_id=evaluation_day,
        train_indexes=train,
        early_stop_indexes=early_stop,
        calibration_indexes=calibration,
        evaluation_indexes=evaluation,
        evidence=evidence,
    )


def recency_weighted_uniqueness(
    dataset: MicrostructureDataset,
    indexes: np.ndarray,
    uniqueness: np.ndarray,
    *,
    half_life_days: float | None,
) -> np.ndarray:
    selected = np.asarray(indexes, dtype=np.int64)
    base = np.asarray(uniqueness, dtype=np.float64)
    if (
        selected.ndim != 1
        or base.shape != selected.shape
        or selected.size == 0
        or np.any(base <= 0.0)
        or not np.all(np.isfinite(base))
    ):
        raise ValueError("walk-forward uniqueness weights are invalid")
    if half_life_days is None:
        output = base
    else:
        half_life = float(half_life_days)
        if not math.isfinite(half_life) or half_life < 1.0:
            raise ValueError("walk-forward half-life is invalid")
        age_days = (
            int(np.max(dataset.decision_time_ms[selected]))
            - dataset.decision_time_ms[selected]
        ) / _DAY_MS
        output = base * np.exp2(-age_days / half_life)
    mean = float(np.mean(output))
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("walk-forward recency weights are non-finite")
    normalized = output / mean
    return normalized.astype(np.float32)


def _empty_trace() -> ActionTrace:
    metrics = TradingMetrics(
        trades=0,
        total_net_bps=0.0,
        mean_net_bps=0.0,
        median_net_bps=0.0,
        win_rate=0.0,
        profit_factor=None,
        max_drawdown_bps=0.0,
        worst_trade_bps=0.0,
        best_trade_bps=0.0,
        long_trades=0,
        short_trades=0,
        active_days=0,
        trades_per_active_day=0.0,
    )
    return ActionTrace(metrics, (), (), (), (), ())


def simulate_action_trace(
    dataset: MicrostructureDataset,
    actual_gross_bps: np.ndarray,
    score: GrossActionScoreBatch,
    *,
    strength_threshold: float,
) -> ActionTrace:
    """Simulate one position at a time with same-UTC-day fixed-horizon exits."""

    validate_microstructure_dataset(dataset)
    endpoints = np.asarray(score.endpoint_indexes, dtype=np.int64)
    gross = np.asarray(actual_gross_bps, dtype=np.float64)
    threshold = float(strength_threshold)
    if (
        gross.shape != (dataset.rows,)
        or endpoints.ndim != 1
        or endpoints.size == 0
        or endpoints[0] < 0
        or endpoints[-1] >= dataset.rows
        or np.any(np.diff(endpoints) <= 0)
        or score.side.shape != endpoints.shape
        or score.strength.shape != endpoints.shape
        or np.any(~np.isin(score.side, (-1, 0, 1)))
        or not np.all(np.isfinite(score.strength))
        or np.any(score.strength < 0.0)
        or np.any((score.side == 0) != (score.strength == 0.0))
        or not math.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError("walk-forward action trace contract is invalid")
    side = np.asarray(score.side, dtype=np.int8)
    strength = np.asarray(score.strength, dtype=np.float64)
    decision_day = dataset.decision_time_ms[endpoints] // _DAY_MS
    long_same_day = dataset.long_exit_time_ms[endpoints] // _DAY_MS == decision_day
    short_same_day = dataset.short_exit_time_ms[endpoints] // _DAY_MS == decision_day
    long_eligible = (
        (side == 1) & long_same_day & dataset.long_liquidity_eligible[endpoints]
    )
    short_eligible = (
        (side == -1) & short_same_day & dataset.short_liquidity_eligible[endpoints]
    )
    negative = np.full(len(endpoints), -1.0, dtype=np.float64)
    trace = _simulate_non_overlapping_trace(
        timestamps=dataset.decision_time_ms[endpoints],
        long_exit_times=dataset.long_exit_time_ms[endpoints],
        short_exit_times=dataset.short_exit_time_ms[endpoints],
        long_targets=dataset.long_net_bps[endpoints],
        short_targets=dataset.short_net_bps[endpoints],
        long_edge=np.where(side == 1, strength, negative),
        short_edge=np.where(side == -1, strength, negative),
        long_probability=np.ones(len(endpoints), dtype=np.float64),
        short_probability=np.ones(len(endpoints), dtype=np.float64),
        edge_threshold=threshold,
        probability_threshold=1.0,
        long_eligible=long_eligible,
        short_eligible=short_eligible,
    )
    timestamp_to_position = {
        int(timestamp): position
        for position, timestamp in enumerate(dataset.decision_time_ms[endpoints])
    }
    source_indexes: list[int] = []
    gross_values: list[float] = []
    for timestamp, trade_side in zip(trace.timestamps, trace.sides, strict=True):
        position = timestamp_to_position[int(timestamp)]
        source_index = int(endpoints[position])
        source_indexes.append(source_index)
        gross_values.append(float(trade_side) * float(gross[source_index]))
    return ActionTrace(
        metrics=trace.metrics,
        gross_bps=tuple(gross_values),
        net_bps=trace.pnls,
        sides=trace.sides,
        timestamps_ms=trace.timestamps,
        source_endpoint_indexes=tuple(source_indexes),
    )


def _positive_day_ratio(trace: ActionTrace) -> float:
    if not trace.timestamps_ms:
        return 0.0
    daily: dict[int, float] = {}
    for timestamp, value in zip(trace.timestamps_ms, trace.net_bps, strict=True):
        day = int(timestamp) // _DAY_MS
        daily[day] = daily.get(day, 0.0) + float(value)
    return float(np.mean(np.asarray(list(daily.values())) > 0.0))


def select_calibrated_threshold(
    dataset: MicrostructureDataset,
    actual_gross_bps: np.ndarray,
    score: GrossActionScoreBatch,
    *,
    quantiles: Sequence[float],
    minimum_trades: int,
    maximum_drawdown_bps: float,
    minimum_positive_day_ratio: float,
    drawdown_penalty: float,
) -> ThresholdSelection:
    """Select a prior-only threshold or return a mandatory abstention."""

    endpoints = np.asarray(score.endpoint_indexes, dtype=np.int64)
    values = np.asarray(score.strength, dtype=np.float64)
    side = np.asarray(score.side, dtype=np.int8)
    if (
        endpoints.ndim != 1
        or endpoints.size == 0
        or endpoints[0] < 0
        or endpoints[-1] >= dataset.rows
        or np.any(np.diff(endpoints) <= 0)
        or side.shape != endpoints.shape
        or values.shape != endpoints.shape
    ):
        raise ValueError("walk-forward threshold score contract is invalid")
    decision_day = dataset.decision_time_ms[endpoints] // _DAY_MS
    long_eligible = (
        (side == 1)
        & dataset.long_liquidity_eligible[endpoints]
        & (dataset.long_exit_time_ms[endpoints] // _DAY_MS == decision_day)
    )
    short_eligible = (
        (side == -1)
        & dataset.short_liquidity_eligible[endpoints]
        & (dataset.short_exit_time_ms[endpoints] // _DAY_MS == decision_day)
    )
    active = (values > 0.0) & (long_eligible | short_eligible)
    if not np.any(active):
        return ThresholdSelection(
            accepted=False,
            threshold=None,
            quantile=None,
            selected_trace=_empty_trace(),
            candidates=(),
            rejection_reasons=("calibration_has_no_active_scores",),
        )
    requested = tuple(float(value) for value in quantiles)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in requested)
        or int(minimum_trades) < 1
        or not math.isfinite(maximum_drawdown_bps)
        or maximum_drawdown_bps <= 0.0
        or not 0.0 <= minimum_positive_day_ratio <= 1.0
        or not math.isfinite(drawdown_penalty)
        or drawdown_penalty < 0.0
    ):
        raise ValueError("walk-forward threshold policy is invalid")
    candidates: list[dict[str, object]] = []
    accepted: list[tuple[tuple[float, float, int], float, float, ActionTrace]] = []
    for quantile in requested:
        threshold = float(np.quantile(values[active], quantile))
        trace = simulate_action_trace(
            dataset,
            actual_gross_bps,
            score,
            strength_threshold=threshold,
        )
        positive_day_ratio = _positive_day_ratio(trace)
        reasons: list[str] = []
        if trace.metrics.trades < int(minimum_trades):
            reasons.append("minimum_calibration_trades_not_met")
        if trace.metrics.total_net_bps <= 0.0:
            reasons.append("calibration_total_net_not_positive")
        if trace.metrics.max_drawdown_bps > float(maximum_drawdown_bps):
            reasons.append("calibration_drawdown_limit_exceeded")
        if positive_day_ratio < float(minimum_positive_day_ratio):
            reasons.append("calibration_positive_day_ratio_not_met")
        utility = float(
            trace.metrics.total_net_bps
            - float(drawdown_penalty) * trace.metrics.max_drawdown_bps
        )
        candidate = {
            "quantile": quantile,
            "threshold": threshold,
            "utility_bps": utility,
            "positive_day_ratio": positive_day_ratio,
            "metrics": asdict(trace.metrics),
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "trading_authority": False,
        }
        candidates.append(candidate)
        if not reasons:
            accepted.append(
                (
                    (utility, trace.metrics.mean_net_bps, trace.metrics.trades),
                    quantile,
                    threshold,
                    trace,
                )
            )
    if not accepted:
        return ThresholdSelection(
            accepted=False,
            threshold=None,
            quantile=None,
            selected_trace=_empty_trace(),
            candidates=tuple(candidates),
            rejection_reasons=("no_calibration_threshold_passed_risk_gates",),
        )
    _rank, quantile, threshold, trace = max(accepted, key=lambda value: value[0])
    return ThresholdSelection(
        accepted=True,
        threshold=threshold,
        quantile=quantile,
        selected_trace=trace,
        candidates=tuple(candidates),
        rejection_reasons=(),
    )


__all__ = [
    "WALK_FORWARD_SCHEMA_VERSION",
    "ActionTrace",
    "ThresholdSelection",
    "WalkForwardDayPlan",
    "WalkForwardFitSpec",
    "plan_walk_forward_day",
    "recency_weighted_uniqueness",
    "select_calibrated_threshold",
    "simulate_action_trace",
]
