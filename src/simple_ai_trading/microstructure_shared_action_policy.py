"""Consensus gating for shared action-conditional value ensembles."""

from __future__ import annotations

import numpy as np

from .microstructure_action_policy import (
    ActionPolicySpec,
    ActionScoreBatch,
    derive_action_scores,
)
from .microstructure_shared_action_lightgbm import SharedActionEnsembleBatch


SHARED_ACTION_POLICY_SCHEMA_VERSION = "shared-action-consensus-policy-v1"


def derive_shared_action_scores(
    ensemble: SharedActionEnsembleBatch,
    spec: ActionPolicySpec,
) -> ActionScoreBatch:
    """Require action-value and antisymmetric advantage heads to agree."""

    base = derive_action_scores(ensemble.action_values, spec)
    side = np.asarray(base.side, dtype=np.int8)
    mean = np.asarray(ensemble.signed_advantage_mean_bps, dtype=np.float64)
    std = np.asarray(
        ensemble.signed_advantage_epistemic_std_bps,
        dtype=np.float64,
    )
    long_agreement = np.asarray(
        ensemble.advantage_long_member_ratio,
        dtype=np.float64,
    )
    short_agreement = np.asarray(
        ensemble.advantage_short_member_ratio,
        dtype=np.float64,
    )
    consensus = np.asarray(
        ensemble.side_consensus_member_ratio,
        dtype=np.float64,
    )
    if any(
        values.shape != side.shape
        for values in (mean, std, long_agreement, short_agreement, consensus)
    ):
        raise ValueError("shared-action policy ensemble shape drifted")
    advantage_strength = np.abs(mean) - spec.epistemic_penalty * std
    advantage_side = np.sign(mean).astype(np.int8)
    agreement = np.where(side == 1, long_agreement, short_agreement)
    accepted = (
        np.asarray(base.eligible, dtype=bool)
        & (side == advantage_side)
        & (advantage_strength > 0.0)
        & (std <= spec.maximum_epistemic_std_bps)
        & (agreement >= spec.minimum_member_agreement)
        & (consensus >= spec.minimum_member_agreement)
    )
    filtered_side = np.where(accepted, side, 0).astype(np.int8)
    filtered_strength = np.where(
        accepted,
        np.asarray(base.strength_bps, dtype=np.float64),
        0.0,
    )
    return ActionScoreBatch(
        endpoint_indexes=np.asarray(base.endpoint_indexes, dtype=np.int64).copy(),
        side=filtered_side,
        strength_bps=filtered_strength,
        eligible=accepted,
        profile=base.profile,
    )


__all__ = [
    "SHARED_ACTION_POLICY_SCHEMA_VERSION",
    "derive_shared_action_scores",
]
