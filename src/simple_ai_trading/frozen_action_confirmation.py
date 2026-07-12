"""Nested, research-only confirmation of precommitted action thresholds."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping, Sequence

import numpy as np

from .microstructure_action_architecture import ActionValueEnsembleBatch
from .microstructure_action_policy import (
    ActionPolicySpec,
    barrier_trace_gate_reasons,
    derive_action_scores,
    simulate_barrier_action_trace,
)
from .microstructure_barriers import AdaptiveBarrierTargets
from .microstructure_features import MicrostructureDataset


FROZEN_CONFIRMATION_SCHEMA_VERSION = "frozen-action-confirmation-v1"
_PROFILE_FIELDS = (
    "profile",
    "epistemic_penalty",
    "minimum_profitable_probability",
    "minimum_member_agreement",
    "maximum_epistemic_std_bps",
    "minimum_lower_bound_bps",
)


def _policy_spec(profile: Mapping[str, object]) -> ActionPolicySpec:
    missing = [name for name in _PROFILE_FIELDS if name not in profile]
    if missing:
        raise ValueError(
            "frozen confirmation profile is missing fields: " + ",".join(missing)
        )
    return ActionPolicySpec(
        profile=str(profile["profile"]),
        epistemic_penalty=float(profile["epistemic_penalty"]),
        minimum_profitable_probability=float(
            profile["minimum_profitable_probability"]
        ),
        minimum_member_agreement=float(profile["minimum_member_agreement"]),
        maximum_epistemic_std_bps=float(profile["maximum_epistemic_std_bps"]),
        minimum_lower_bound_bps=float(profile["minimum_lower_bound_bps"]),
    )


def _frozen_candidates(
    candidates: Sequence[Mapping[str, object]],
) -> tuple[tuple[float, float], ...]:
    output: list[tuple[float, float]] = []
    seen_quantiles: set[float] = set()
    seen_thresholds: set[float] = set()
    for candidate in candidates:
        quantile = float(candidate.get("quantile", float("nan")))
        threshold = float(candidate.get("threshold_bps", float("nan")))
        if (
            not math.isfinite(quantile)
            or not 0.0 < quantile < 1.0
            or not math.isfinite(threshold)
            or threshold <= 0.0
            or quantile in seen_quantiles
            or threshold in seen_thresholds
        ):
            raise ValueError("frozen confirmation threshold candidate is invalid")
        seen_quantiles.add(quantile)
        seen_thresholds.add(threshold)
        output.append((quantile, threshold))
    if not output:
        raise ValueError("frozen confirmation requires threshold candidates")
    return tuple(output)


def evaluate_frozen_profile_candidates(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    prediction: ActionValueEnsembleBatch,
    *,
    profile: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    gates: Mapping[str, object],
    expected_days: Sequence[int],
    drawdown_penalty: float,
    stage: str,
) -> dict[str, object]:
    """Evaluate fixed thresholds and select only within the opened stage."""

    if not stage.strip():
        raise ValueError("frozen confirmation stage is empty")
    if not math.isfinite(float(drawdown_penalty)) or float(drawdown_penalty) < 0.0:
        raise ValueError("frozen confirmation drawdown penalty is invalid")
    days = tuple(int(value) for value in expected_days)
    if not days or tuple(sorted(set(days))) != days:
        raise ValueError("frozen confirmation expected days are invalid")
    spec = _policy_spec(profile)
    score = derive_action_scores(prediction, spec)
    rows: list[dict[str, object]] = []
    for quantile, threshold in _frozen_candidates(candidates):
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
            expected_days=days,
            gates=gates,
        )
        utility = float(
            stress_trace.metrics.total_net_bps
            - float(drawdown_penalty) * stress_trace.metrics.max_drawdown_bps
        )
        rows.append(
            {
                "quantile": quantile,
                "threshold_bps": threshold,
                "passed": not reasons,
                "gate_reasons": list(reasons),
                "drawdown_adjusted_utility_bps": utility,
                "base_trace": base_trace.asdict(),
                "stress_trace": stress_trace.asdict(),
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
    passing = [row for row in rows if bool(row["passed"])]
    selected = (
        max(
            passing,
            key=lambda row: (
                float(row["drawdown_adjusted_utility_bps"]),
                float(row["stress_trace"]["metrics"]["mean_net_bps"]),
                int(row["stress_trace"]["metrics"]["trades"]),
                -float(row["quantile"]),
            ),
        )
        if passing
        else None
    )
    return {
        "schema_version": FROZEN_CONFIRMATION_SCHEMA_VERSION,
        "stage": stage,
        "profile": spec.profile,
        "policy_spec": asdict(spec),
        "eligible_rows": int(np.sum(score.eligible)),
        "candidate_count": len(rows),
        "passing_candidate_count": len(passing),
        "passed": selected is not None,
        "selected_quantile": (
            float(selected["quantile"]) if selected is not None else None
        ),
        "selected_threshold_bps": (
            float(selected["threshold_bps"]) if selected is not None else None
        ),
        "candidates": rows,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def evaluate_frozen_profile_stage(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    prediction: ActionValueEnsembleBatch,
    *,
    profile: Mapping[str, object],
    quantile: float,
    threshold_bps: float,
    gates: Mapping[str, object],
    expected_days: Sequence[int],
    drawdown_penalty: float,
    stage: str,
) -> dict[str, object]:
    """Evaluate one threshold selected strictly in an earlier stage."""

    result = evaluate_frozen_profile_candidates(
        dataset,
        targets,
        prediction,
        profile=profile,
        candidates=(
            {"quantile": float(quantile), "threshold_bps": float(threshold_bps)},
        ),
        gates=gates,
        expected_days=expected_days,
        drawdown_penalty=drawdown_penalty,
        stage=stage,
    )
    result["threshold_source"] = "prior_stage_fixed"
    return result


__all__ = [
    "FROZEN_CONFIRMATION_SCHEMA_VERSION",
    "evaluate_frozen_profile_candidates",
    "evaluate_frozen_profile_stage",
]
