"""Fail-closed policy gates for factorized selective-action ensembles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .microstructure_action_policy import ActionPolicySpec, ActionScoreBatch
from .microstructure_selective_action_lightgbm import SelectiveActionEnsembleBatch


SELECTIVE_ACTION_POLICY_SCHEMA_VERSION = "selective-action-consensus-policy-v1"


@dataclass(frozen=True)
class SelectiveActionPolicySpec:
    """Frozen risk controls plus a conditional-direction confidence floor."""

    action_policy: ActionPolicySpec
    minimum_conditional_direction_confidence: float

    def __post_init__(self) -> None:
        confidence = float(self.minimum_conditional_direction_confidence)
        if not math.isfinite(confidence) or not 0.5 < confidence < 1.0:
            raise ValueError("conditional-direction confidence is outside bounds")

    @property
    def profile(self) -> str:
        return self.action_policy.profile


def _validated_arrays(
    ensemble: SelectiveActionEnsembleBatch,
) -> tuple[np.ndarray, ...]:
    action = ensemble.action_values
    endpoints = np.asarray(action.endpoint_indexes, dtype=np.int64)
    vectors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            action.long_mean_bps,
            action.short_mean_bps,
            action.long_epistemic_std_bps,
            action.short_epistemic_std_bps,
            action.long_lower_bps,
            action.short_lower_bps,
            action.long_positive_member_ratio,
            action.short_positive_member_ratio,
            ensemble.opportunity_probability_mean,
            ensemble.conditional_long_probability_mean,
            ensemble.side_consensus_member_ratio,
        )
    )
    opportunity_members = np.asarray(
        ensemble.opportunity_member_probabilities,
        dtype=np.float64,
    )
    direction_members = np.asarray(
        ensemble.conditional_long_member_probabilities,
        dtype=np.float64,
    )
    rows = len(endpoints)
    if (
        rows <= 0
        or endpoints.ndim != 1
        or np.any(np.diff(endpoints) <= 0)
        or ensemble.member_count < 2
        or action.member_count != ensemble.member_count
        or any(value.shape != (rows,) for value in vectors)
        or opportunity_members.shape != (ensemble.member_count, rows)
        or direction_members.shape != (ensemble.member_count, rows)
        or any(not np.all(np.isfinite(value)) for value in vectors)
        or not np.all(np.isfinite(opportunity_members))
        or not np.all(np.isfinite(direction_members))
        or np.any(opportunity_members < 0.0)
        or np.any(opportunity_members > 1.0)
        or np.any(direction_members < 0.0)
        or np.any(direction_members > 1.0)
        or ensemble.trading_authority
        or ensemble.execution_claim
        or ensemble.profitability_claim
        or ensemble.portfolio_claim
        or ensemble.leverage_applied
        or action.trading_authority
        or action.execution_claim
        or action.profitability_claim
        or action.portfolio_claim
        or action.leverage_applied
    ):
        raise ValueError("selective-action policy ensemble contract is invalid")
    return endpoints, *vectors, opportunity_members, direction_members


def derive_selective_action_scores(
    ensemble: SelectiveActionEnsembleBatch,
    spec: SelectiveActionPolicySpec,
) -> ActionScoreBatch:
    """Select long or short only when every precommitted gate agrees."""

    (
        endpoints,
        long_mean,
        short_mean,
        long_std,
        short_std,
        long_lower,
        short_lower,
        long_positive_ratio,
        short_positive_ratio,
        opportunity_probability,
        conditional_long_probability,
        side_consensus_ratio,
        opportunity_members,
        direction_members,
    ) = _validated_arrays(ensemble)
    base = spec.action_policy
    direction_floor = float(spec.minimum_conditional_direction_confidence)
    opportunity_floor = float(base.minimum_profitable_probability)
    agreement_floor = float(base.minimum_member_agreement)

    long_strength = long_mean - float(base.epistemic_penalty) * long_std
    short_strength = short_mean - float(base.epistemic_penalty) * short_std
    opportunity_agreement = np.mean(
        opportunity_members >= opportunity_floor,
        axis=0,
    )
    long_direction_agreement = np.mean(
        direction_members >= direction_floor,
        axis=0,
    )
    short_direction_agreement = np.mean(
        direction_members <= 1.0 - direction_floor,
        axis=0,
    )
    common = (
        (opportunity_probability >= opportunity_floor)
        & (opportunity_agreement >= agreement_floor)
        & (side_consensus_ratio >= agreement_floor)
    )
    long_eligible = (
        common
        & (long_strength > 0.0)
        & (long_std <= float(base.maximum_epistemic_std_bps))
        & (long_lower >= float(base.minimum_lower_bound_bps))
        & (long_positive_ratio >= agreement_floor)
        & (conditional_long_probability >= direction_floor)
        & (long_direction_agreement >= agreement_floor)
    )
    short_eligible = (
        common
        & (short_strength > 0.0)
        & (short_std <= float(base.maximum_epistemic_std_bps))
        & (short_lower >= float(base.minimum_lower_bound_bps))
        & (short_positive_ratio >= agreement_floor)
        & (conditional_long_probability <= 1.0 - direction_floor)
        & (short_direction_agreement >= agreement_floor)
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
        profile=base.profile,
    )


__all__ = [
    "SELECTIVE_ACTION_POLICY_SCHEMA_VERSION",
    "SelectiveActionPolicySpec",
    "derive_selective_action_scores",
]
