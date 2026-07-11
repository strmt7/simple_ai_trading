"""Selective, research-only policy evaluation for adaptive barrier forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .microstructure_action_architecture import ActionValueEnsembleBatch
from .microstructure_barriers import (
    AdaptiveBarrierTargets,
    validate_adaptive_barrier_targets,
)
from .microstructure_features import (
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .microstructure_model import (
    TradingMetrics,
    _simulate_non_overlapping_trace,
)


ACTION_POLICY_SCHEMA_VERSION = "adaptive-action-policy-v1"
_DAY_MS = 86_400_000
_NO_TRADE_THRESHOLD = np.finfo(float).max
_PROFILES = {"conservative", "regular", "aggressive"}
_GATE_FIELDS = {
    "minimum_trades",
    "minimum_total_net_bps",
    "maximum_drawdown_bps",
    "minimum_positive_day_ratio",
    "minimum_worst_trade_bps",
    "minimum_profit_factor",
}


@dataclass(frozen=True)
class ActionPolicySpec:
    profile: str
    epistemic_penalty: float
    minimum_profitable_probability: float
    minimum_member_agreement: float
    maximum_epistemic_std_bps: float
    minimum_lower_bound_bps: float

    def __post_init__(self) -> None:
        values = (
            self.epistemic_penalty,
            self.minimum_profitable_probability,
            self.minimum_member_agreement,
            self.maximum_epistemic_std_bps,
            self.minimum_lower_bound_bps,
        )
        if self.profile not in _PROFILES:
            raise ValueError("adaptive action policy profile is unsupported")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("adaptive action policy values must be finite")
        if (
            not 0.0 <= self.epistemic_penalty <= 5.0
            or not 0.5 <= self.minimum_profitable_probability < 1.0
            or not 0.5 <= self.minimum_member_agreement <= 1.0
            or self.maximum_epistemic_std_bps <= 0.0
            or self.minimum_lower_bound_bps >= 0.0
        ):
            raise ValueError("adaptive action policy values are outside bounds")


@dataclass(frozen=True)
class ActionScoreBatch:
    endpoint_indexes: np.ndarray
    side: np.ndarray
    strength_bps: np.ndarray
    eligible: np.ndarray
    profile: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))


@dataclass(frozen=True)
class BarrierActionTrace:
    scenario: str
    metrics: TradingMetrics
    net_bps: tuple[float, ...]
    sides: tuple[int, ...]
    timestamps_ms: tuple[int, ...]
    exit_times_ms: tuple[int, ...]
    source_endpoint_indexes: tuple[int, ...]
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BarrierThresholdSelection:
    accepted: bool
    threshold_bps: float | None
    quantile: float | None
    base_trace: BarrierActionTrace
    stress_trace: BarrierActionTrace
    candidates: tuple[Mapping[str, object], ...]
    rejection_reasons: tuple[str, ...]
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "threshold_bps": self.threshold_bps,
            "quantile": self.quantile,
            "base_trace": self.base_trace.asdict(),
            "stress_trace": self.stress_trace.asdict(),
            "candidates": [dict(value) for value in self.candidates],
            "rejection_reasons": list(self.rejection_reasons),
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }


def _ensemble_arrays(ensemble: ActionValueEnsembleBatch) -> tuple[np.ndarray, ...]:
    endpoints = np.asarray(ensemble.endpoint_indexes, dtype=np.int64)
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            ensemble.long_mean_bps,
            ensemble.short_mean_bps,
            ensemble.long_epistemic_std_bps,
            ensemble.short_epistemic_std_bps,
            ensemble.long_profitable_probability,
            ensemble.short_profitable_probability,
            ensemble.long_lower_bps,
            ensemble.short_lower_bps,
            ensemble.long_upper_bps,
            ensemble.short_upper_bps,
            ensemble.long_positive_member_ratio,
            ensemble.short_positive_member_ratio,
        )
    )
    if (
        endpoints.ndim != 1
        or endpoints.size == 0
        or np.any(np.diff(endpoints) <= 0)
        or ensemble.member_count < 2
        or ensemble.trading_authority
        or ensemble.execution_claim
        or ensemble.profitability_claim
        or ensemble.portfolio_claim
        or ensemble.leverage_applied
        or any(values.shape != endpoints.shape for values in arrays)
        or any(not np.all(np.isfinite(values)) for values in arrays)
    ):
        raise ValueError("adaptive action ensemble contract is invalid")
    return endpoints, *arrays


def derive_action_scores(
    ensemble: ActionValueEnsembleBatch,
    spec: ActionPolicySpec,
) -> ActionScoreBatch:
    """Select a side only when mean, class, tail and ensemble gates agree."""

    (
        endpoints,
        long_mean,
        short_mean,
        long_std,
        short_std,
        long_probability,
        short_probability,
        long_lower,
        short_lower,
        _long_upper,
        _short_upper,
        long_agreement,
        short_agreement,
    ) = _ensemble_arrays(ensemble)
    long_strength = long_mean - spec.epistemic_penalty * long_std
    short_strength = short_mean - spec.epistemic_penalty * short_std
    long_eligible = (
        (long_strength > 0.0)
        & (long_probability >= spec.minimum_profitable_probability)
        & (long_agreement >= spec.minimum_member_agreement)
        & (long_std <= spec.maximum_epistemic_std_bps)
        & (long_lower >= spec.minimum_lower_bound_bps)
    )
    short_eligible = (
        (short_strength > 0.0)
        & (short_probability >= spec.minimum_profitable_probability)
        & (short_agreement >= spec.minimum_member_agreement)
        & (short_std <= spec.maximum_epistemic_std_bps)
        & (short_lower >= spec.minimum_lower_bound_bps)
    )
    choose_long = long_eligible & (~short_eligible | (long_strength >= short_strength))
    choose_short = short_eligible & ~choose_long
    side = np.zeros(len(endpoints), dtype=np.int8)
    side[choose_long] = 1
    side[choose_short] = -1
    strength = np.zeros(len(endpoints), dtype=np.float64)
    strength[choose_long] = long_strength[choose_long]
    strength[choose_short] = short_strength[choose_short]
    return ActionScoreBatch(
        endpoint_indexes=endpoints.copy(),
        side=side,
        strength_bps=strength,
        eligible=side != 0,
        profile=spec.profile,
    )


def _target_positions(
    targets: AdaptiveBarrierTargets,
    endpoints: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(targets.source_indexes, endpoints)
    if np.any(positions >= targets.rows):
        raise ValueError("adaptive action endpoint is absent from barrier targets")
    if not np.array_equal(targets.source_indexes[positions], endpoints) or not np.all(
        targets.valid[positions]
    ):
        raise ValueError("adaptive action endpoint is not a valid barrier target")
    return positions


def _scenario_arrays(
    targets: AdaptiveBarrierTargets,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scenario == "base":
        return (
            targets.base_long_net_bps,
            targets.base_short_net_bps,
            targets.base_long_exit_time_ms,
            targets.base_short_exit_time_ms,
        )
    if scenario == "stress":
        return (
            targets.stress_long_net_bps,
            targets.stress_short_net_bps,
            targets.stress_long_exit_time_ms,
            targets.stress_short_exit_time_ms,
        )
    raise ValueError("adaptive action target scenario is unsupported")


def simulate_barrier_action_trace(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    score: ActionScoreBatch,
    *,
    scenario: str,
    strength_threshold_bps: float,
) -> BarrierActionTrace:
    """Replay one position at a time against exact scenario-specific exits."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, targets)
    threshold = float(strength_threshold_bps)
    endpoints = np.asarray(score.endpoint_indexes, dtype=np.int64)
    side = np.asarray(score.side, dtype=np.int8)
    strength = np.asarray(score.strength_bps, dtype=np.float64)
    eligible = np.asarray(score.eligible, dtype=bool)
    if (
        endpoints.ndim != 1
        or endpoints.size == 0
        or endpoints[0] < 0
        or endpoints[-1] >= dataset.rows
        or np.any(np.diff(endpoints) <= 0)
        or side.shape != endpoints.shape
        or strength.shape != endpoints.shape
        or eligible.shape != endpoints.shape
        or np.any(~np.isin(side, (-1, 0, 1)))
        or not np.all(np.isfinite(strength))
        or np.any(strength < 0.0)
        or np.any(eligible != (side != 0))
        or np.any((side == 0) != (strength == 0.0))
        or score.profile not in _PROFILES
        or score.trading_authority
        or score.execution_claim
        or score.profitability_claim
        or score.portfolio_claim
        or score.leverage_applied
        or not math.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError("adaptive action score contract is invalid")
    positions = _target_positions(targets, endpoints)
    long_target, short_target, long_exit, short_exit = _scenario_arrays(
        targets, scenario
    )
    long_values = np.asarray(long_target[positions], dtype=np.float64)
    short_values = np.asarray(short_target[positions], dtype=np.float64)
    long_exits = np.asarray(long_exit[positions], dtype=np.int64)
    short_exits = np.asarray(short_exit[positions], dtype=np.int64)
    timestamps = dataset.decision_time_ms[endpoints]
    decision_days = timestamps // _DAY_MS
    long_eligible = (
        (side == 1)
        & eligible
        & dataset.long_liquidity_eligible[endpoints]
        & (long_exits // _DAY_MS == decision_days)
    )
    short_eligible = (
        (side == -1)
        & eligible
        & dataset.short_liquidity_eligible[endpoints]
        & (short_exits // _DAY_MS == decision_days)
    )
    negative = np.full(len(endpoints), -1.0, dtype=np.float64)
    trace = _simulate_non_overlapping_trace(
        timestamps=timestamps,
        long_exit_times=long_exits,
        short_exit_times=short_exits,
        long_targets=long_values,
        short_targets=short_values,
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
        int(timestamp): index for index, timestamp in enumerate(timestamps)
    }
    source_indexes: list[int] = []
    exit_times: list[int] = []
    for timestamp, executed_side in zip(trace.timestamps, trace.sides, strict=True):
        position = timestamp_to_position[int(timestamp)]
        source_indexes.append(int(endpoints[position]))
        exit_times.append(
            int(long_exits[position] if executed_side == 1 else short_exits[position])
        )
    return BarrierActionTrace(
        scenario=scenario,
        metrics=trace.metrics,
        net_bps=trace.pnls,
        sides=trace.sides,
        timestamps_ms=trace.timestamps,
        exit_times_ms=tuple(exit_times),
        source_endpoint_indexes=tuple(source_indexes),
    )


def _positive_day_ratio(
    trace: BarrierActionTrace,
    expected_days: Sequence[int],
) -> float:
    days = tuple(int(value) for value in expected_days)
    if not days or len(set(days)) != len(days):
        raise ValueError("adaptive action expected days are invalid")
    daily = {day: 0.0 for day in days}
    for timestamp, pnl in zip(trace.timestamps_ms, trace.net_bps, strict=True):
        day = int(timestamp) // _DAY_MS
        if day not in daily:
            raise ValueError("adaptive action trace lies outside expected days")
        daily[day] += float(pnl)
    return float(np.mean(np.asarray(tuple(daily.values())) > 0.0))


def barrier_trace_gate_reasons(
    trace: BarrierActionTrace,
    *,
    expected_days: Sequence[int],
    gates: Mapping[str, object],
) -> list[str]:
    if set(gates) != _GATE_FIELDS:
        raise ValueError("adaptive action risk controls are incomplete")
    values = tuple(float(gates[name]) for name in _GATE_FIELDS)
    if (
        not all(math.isfinite(value) for value in values)
        or int(gates["minimum_trades"]) < 1
        or float(gates["minimum_total_net_bps"]) < 0.0
        or float(gates["maximum_drawdown_bps"]) <= 0.0
        or not 0.0 <= float(gates["minimum_positive_day_ratio"]) <= 1.0
        or float(gates["minimum_worst_trade_bps"]) >= 0.0
        or float(gates["minimum_profit_factor"]) < 1.0
    ):
        raise ValueError("adaptive action risk controls are invalid")
    metrics = trace.metrics
    reasons: list[str] = []
    if metrics.trades < int(gates["minimum_trades"]):
        reasons.append("minimum_trades_not_met")
    if metrics.total_net_bps <= float(gates["minimum_total_net_bps"]):
        reasons.append("total_net_gate_failed")
    if metrics.max_drawdown_bps > float(gates["maximum_drawdown_bps"]):
        reasons.append("drawdown_gate_failed")
    if _positive_day_ratio(trace, expected_days) < float(
        gates["minimum_positive_day_ratio"]
    ):
        reasons.append("positive_day_ratio_gate_failed")
    if metrics.trades and metrics.worst_trade_bps < float(
        gates["minimum_worst_trade_bps"]
    ):
        reasons.append("worst_trade_gate_failed")
    if metrics.profit_factor is None or metrics.profit_factor < float(
        gates["minimum_profit_factor"]
    ):
        reasons.append("profit_factor_gate_failed")
    return reasons


def select_barrier_threshold(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    score: ActionScoreBatch,
    *,
    quantiles: Sequence[float],
    expected_days: Sequence[int],
    gates: Mapping[str, object],
    drawdown_penalty: float,
) -> BarrierThresholdSelection:
    """Choose one threshold on prior adverse-stress outcomes or abstain."""

    requested = tuple(float(value) for value in quantiles)
    penalty = float(drawdown_penalty)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in requested)
        or not math.isfinite(penalty)
        or penalty < 0.0
    ):
        raise ValueError("adaptive action threshold policy is invalid")
    empty_base = simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario="base",
        strength_threshold_bps=_NO_TRADE_THRESHOLD,
    )
    empty_stress = simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario="stress",
        strength_threshold_bps=_NO_TRADE_THRESHOLD,
    )
    barrier_trace_gate_reasons(
        empty_stress,
        expected_days=expected_days,
        gates=gates,
    )
    active = np.asarray(score.eligible, dtype=bool) & (
        np.asarray(score.strength_bps, dtype=np.float64) > 0.0
    )
    if not np.any(active):
        return BarrierThresholdSelection(
            accepted=False,
            threshold_bps=None,
            quantile=None,
            base_trace=empty_base,
            stress_trace=empty_stress,
            candidates=(),
            rejection_reasons=("calibration_has_no_eligible_scores",),
        )
    candidates: list[dict[str, object]] = []
    accepted: list[
        tuple[
            tuple[float, float, float, int],
            float,
            float,
            BarrierActionTrace,
            BarrierActionTrace,
        ]
    ] = []
    strengths = np.asarray(score.strength_bps, dtype=np.float64)
    for quantile in requested:
        threshold = float(np.quantile(strengths[active], quantile))
        base_trace = simulate_barrier_action_trace(
            dataset,
            targets,
            score,
            scenario="base",
            strength_threshold_bps=threshold,
        )
        stress_trace = simulate_barrier_action_trace(
            dataset,
            targets,
            score,
            scenario="stress",
            strength_threshold_bps=threshold,
        )
        reasons = barrier_trace_gate_reasons(
            stress_trace,
            expected_days=expected_days,
            gates=gates,
        )
        utility = float(
            stress_trace.metrics.total_net_bps
            - penalty * stress_trace.metrics.max_drawdown_bps
        )
        candidates.append(
            {
                "quantile": quantile,
                "threshold_bps": threshold,
                "utility_bps": utility,
                "base_metrics": asdict(base_trace.metrics),
                "stress_metrics": asdict(stress_trace.metrics),
                "stress_positive_day_ratio": _positive_day_ratio(
                    stress_trace, expected_days
                ),
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
        if not reasons:
            accepted.append(
                (
                    (
                        utility,
                        stress_trace.metrics.mean_net_bps,
                        base_trace.metrics.total_net_bps,
                        stress_trace.metrics.trades,
                    ),
                    quantile,
                    threshold,
                    base_trace,
                    stress_trace,
                )
            )
    if not accepted:
        return BarrierThresholdSelection(
            accepted=False,
            threshold_bps=None,
            quantile=None,
            base_trace=empty_base,
            stress_trace=empty_stress,
            candidates=tuple(candidates),
            rejection_reasons=("no_calibration_threshold_passed_stress_gates",),
        )
    _rank, quantile, threshold, base_trace, stress_trace = max(
        accepted, key=lambda value: value[0]
    )
    return BarrierThresholdSelection(
        accepted=True,
        threshold_bps=threshold,
        quantile=quantile,
        base_trace=base_trace,
        stress_trace=stress_trace,
        candidates=tuple(candidates),
        rejection_reasons=(),
    )


__all__ = [
    "ACTION_POLICY_SCHEMA_VERSION",
    "ActionPolicySpec",
    "ActionScoreBatch",
    "BarrierActionTrace",
    "BarrierThresholdSelection",
    "barrier_trace_gate_reasons",
    "derive_action_scores",
    "select_barrier_threshold",
    "simulate_barrier_action_trace",
]
