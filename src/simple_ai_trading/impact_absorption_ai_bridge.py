"""Causal bridge from Round 74 model output to constrained AI review."""

from __future__ import annotations

import math

import torch

from .impact_absorption_ai_protocol import (
    ROUND74_AI_REVIEW_HORIZONS_SECONDS,
    Round74AIReviewRequest,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)


ROUND74_AI_BRIDGE_SCHEMA_VERSION = "round-074-ai-bridge-v1"
_ASSET_FEATURE_INDICES = tuple(
    ROUND74_EVENT_FEATURE_NAMES.index(name)
    for name in (
        "symbol_is_btcusdt",
        "symbol_is_ethusdt",
        "symbol_is_solusdt",
    )
)


def _finite_probability(value: torch.Tensor, label: str) -> float:
    selected = float(torch.sigmoid(value.detach()).item())
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"Round 74 AI bridge {label} differs")
    return selected


def _finite_tuple(
    value: torch.Tensor,
    *,
    expected_length: int,
    label: str,
) -> tuple[float, ...]:
    detached = value.detach()
    if detached.ndim != 1 or detached.shape[0] != expected_length:
        raise ValueError(f"Round 74 AI bridge {label} shape differs")
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"Round 74 AI bridge {label} is nonfinite")
    return tuple(float(item) for item in detached.cpu().tolist())


def _validate_feature_values(
    feature_values: torch.Tensor,
    *,
    batch_size: int,
    row_index: int,
    asset_slot: int,
) -> torch.Tensor:
    expected_shape = (
        batch_size,
        ROUND74_EVENT_SEQUENCE_LENGTH,
        len(ROUND74_EVENT_FEATURE_NAMES),
    )
    if (
        feature_values.shape != expected_shape
        or not feature_values.is_floating_point()
        or not bool(torch.isfinite(feature_values).all())
        or isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or not 0 <= row_index < batch_size
        or isinstance(asset_slot, bool)
        or not isinstance(asset_slot, int)
        or not 0 <= asset_slot < len(_ASSET_FEATURE_INDICES)
    ):
        raise ValueError("Round 74 AI bridge feature context differs")
    selected = feature_values[row_index]
    final_identity = selected[-1, list(_ASSET_FEATURE_INDICES)].detach()
    expected_identity = torch.zeros_like(final_identity)
    expected_identity[asset_slot] = 1.0
    if not bool(torch.equal(final_identity, expected_identity)):
        raise ValueError("Round 74 AI bridge final asset identity differs")
    return selected


def build_round74_ai_review_request(
    *,
    model_output: Round74EventModelOutput,
    scaled_feature_values: torch.Tensor,
    row_index: int,
    asset_slot: int,
    side: str,
    horizon_seconds: int,
    pretest_policy_sha256: str,
    sample_sha256: str,
    deterministic_risk_state_sha256: str,
    requested_wall_ns: int,
    expires_wall_ns: int,
    proposed_risk_size_bps: int,
) -> Round74AIReviewRequest:
    """Build one target-free, anonymized review request on the model device."""

    if scaled_feature_values.ndim != 3:
        raise ValueError("Round 74 AI bridge feature dimensions differ")
    batch_size = int(scaled_feature_values.shape[0])
    model_output.validate(batch_size)
    selected = _validate_feature_values(
        scaled_feature_values,
        batch_size=batch_size,
        row_index=row_index,
        asset_slot=asset_slot,
    )
    if side not in ROUND74_EVENT_PAYOFF_SIDES:
        raise ValueError("Round 74 AI bridge side differs")
    if horizon_seconds not in ROUND74_AI_REVIEW_HORIZONS_SECONDS:
        raise ValueError("Round 74 AI bridge horizon differs")
    horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(horizon_seconds)
    side_index = ROUND74_EVENT_PAYOFF_SIDES.index(side)
    feature_count = len(ROUND74_EVENT_FEATURE_NAMES)
    feature_last = _finite_tuple(
        selected[-1],
        expected_length=feature_count,
        label="last features",
    )
    feature_mean = _finite_tuple(
        selected.mean(dim=0),
        expected_length=feature_count,
        label="mean features",
    )
    feature_standard_deviation = _finite_tuple(
        selected.std(dim=0, unbiased=False),
        expected_length=feature_count,
        label="feature standard deviations",
    )
    quantile_count = int(model_output.payoff_quantiles_bps.shape[-1])
    payoff_quantiles = _finite_tuple(
        model_output.payoff_quantiles_bps[
            row_index,
            horizon_index,
            side_index,
        ],
        expected_length=quantile_count,
        label="payoff quantiles",
    )
    adverse_excursion_quantiles = _finite_tuple(
        model_output.maximum_adverse_excursion_quantiles_bps[
            row_index,
            horizon_index,
            side_index,
        ],
        expected_length=quantile_count,
        label="adverse-excursion quantiles",
    )
    request = Round74AIReviewRequest(
        pretest_policy_sha256=pretest_policy_sha256,
        sample_sha256=sample_sha256,
        deterministic_risk_state_sha256=(deterministic_risk_state_sha256),
        asset_slot=asset_slot,
        side=side,
        horizon_seconds=horizon_seconds,
        requested_wall_ns=requested_wall_ns,
        expires_wall_ns=expires_wall_ns,
        proposed_risk_size_bps=proposed_risk_size_bps,
        feature_last=feature_last,
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        payoff_quantiles_bps=payoff_quantiles,
        maximum_adverse_excursion_quantiles_bps=(adverse_excursion_quantiles),
        positive_payoff_probability=_finite_probability(
            model_output.positive_payoff_logits[
                row_index,
                horizon_index,
                side_index,
            ],
            "positive-payoff probability",
        ),
        adverse_selection_probability=_finite_probability(
            model_output.adverse_selection_logits[
                row_index,
                horizon_index,
                side_index,
            ],
            "adverse-selection probability",
        ),
        regime_unpredictability_probability=_finite_probability(
            model_output.regime_unpredictability_logits[
                row_index,
                horizon_index,
            ],
            "regime-unpredictability probability",
        ),
    )
    request.validate()
    return request


__all__ = [
    "ROUND74_AI_BRIDGE_SCHEMA_VERSION",
    "build_round74_ai_review_request",
]
