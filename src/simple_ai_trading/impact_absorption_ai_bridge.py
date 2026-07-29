"""Causal bridge from Round 74 model output to constrained AI review."""

from __future__ import annotations

import math

import torch

from .impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    apply_round74_probability_calibration,
    apply_round74_risk_quantile_calibration,
)
from .impact_absorption_ai_protocol import (
    ROUND74_AI_REVIEW_HORIZONS_SECONDS,
    ROUND74_AI_TEMPORAL_BLOCK_COUNT,
    ROUND74_AI_TEMPORAL_BLOCK_EVENTS,
    ROUND74_AI_TEMPORAL_FEATURE_NAMES,
    Round74AIReviewRequest,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)


ROUND74_AI_BRIDGE_SCHEMA_VERSION = "round-074-ai-bridge-v7"
ROUND74_AI_RECENT_BLOCK_EVENTS = 16
_ASSET_FEATURE_INDICES = tuple(
    ROUND74_EVENT_FEATURE_NAMES.index(name)
    for name in (
        "symbol_is_btcusdt",
        "symbol_is_ethusdt",
        "symbol_is_solusdt",
    )
)
_TEMPORAL_FEATURE_INDICES = tuple(
    ROUND74_EVENT_FEATURE_NAMES.index(name)
    for name in ROUND74_AI_TEMPORAL_FEATURE_NAMES
)


def _finite_probability(
    value: torch.Tensor,
    label: str,
) -> float:
    selected = float(value.detach().item())
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
    risk_profile: str,
    probability_calibration: Round74ProbabilityCalibration,
    requested_wall_ns: int,
    expires_wall_ns: int,
    proposed_risk_size_bps: int,
) -> Round74AIReviewRequest:
    """Build one target-free, anonymized review request on the model device."""

    if scaled_feature_values.ndim != 3:
        raise ValueError("Round 74 AI bridge feature dimensions differ")
    batch_size = int(scaled_feature_values.shape[0])
    model_output.validate(batch_size)
    probability_calibration.validate()
    if probability_calibration.pretest_policy_sha256 != pretest_policy_sha256:
        raise ValueError("Round 74 AI bridge calibration policy differs")
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
    recent = selected[-ROUND74_AI_RECENT_BLOCK_EVENTS:]
    prior = selected[
        -(2 * ROUND74_AI_RECENT_BLOCK_EVENTS) : -ROUND74_AI_RECENT_BLOCK_EVENTS
    ]
    feature_recent_change = _finite_tuple(
        recent.mean(dim=0) - prior.mean(dim=0),
        expected_length=feature_count,
        label="recent feature changes",
    )
    temporal_event_count = (
        ROUND74_AI_TEMPORAL_BLOCK_COUNT * ROUND74_AI_TEMPORAL_BLOCK_EVENTS
    )
    temporal = selected[-temporal_event_count:, list(_TEMPORAL_FEATURE_INDICES)]
    temporal = temporal.reshape(
        ROUND74_AI_TEMPORAL_BLOCK_COUNT,
        ROUND74_AI_TEMPORAL_BLOCK_EVENTS,
        len(_TEMPORAL_FEATURE_INDICES),
    ).mean(dim=1)
    feature_recent_block_means = tuple(
        _finite_tuple(
            temporal[index],
            expected_length=len(_TEMPORAL_FEATURE_INDICES),
            label="recent block features",
        )
        for index in range(ROUND74_AI_TEMPORAL_BLOCK_COUNT)
    )
    positive, adverse, unpredictable = apply_round74_probability_calibration(
        probability_calibration,
        positive_payoff_logits=model_output.positive_payoff_logits,
        adverse_selection_logits=model_output.adverse_selection_logits,
        regime_unpredictability_logits=(model_output.regime_unpredictability_logits),
    )
    calibrated_payoff = model_output.payoff_quantiles_bps
    calibrated_mae = model_output.maximum_adverse_excursion_quantiles_bps
    if probability_calibration.risk_quantiles is not None:
        calibrated_payoff, calibrated_mae = apply_round74_risk_quantile_calibration(
            probability_calibration.risk_quantiles,
            payoff_quantiles_bps=calibrated_payoff,
            maximum_adverse_excursion_quantiles_bps=calibrated_mae,
        )
    quantile_count = int(model_output.payoff_quantiles_bps.shape[-1])
    payoff_quantiles = _finite_tuple(
        calibrated_payoff[
            row_index,
            horizon_index,
            side_index,
        ],
        expected_length=quantile_count,
        label="payoff quantiles",
    )
    adverse_excursion_quantiles = _finite_tuple(
        calibrated_mae[
            row_index,
            horizon_index,
            side_index,
        ],
        expected_length=quantile_count,
        label="adverse-excursion quantiles",
    )
    positive_payoff_probability = _finite_probability(
        positive[
            row_index,
            horizon_index,
            side_index,
        ],
        "positive-payoff probability",
    )
    opposing_positive_payoff_probability = _finite_probability(
        positive[
            row_index,
            horizon_index,
            1 - side_index,
        ],
        "opposing positive-payoff probability",
    )
    neither_positive_payoff_probability = _finite_probability(
        1.0 - positive[row_index, horizon_index].sum(),
        "neither positive-payoff probability",
    )
    diagnostics = model_output.epistemic_diagnostics
    if diagnostics is None:
        peer_count = 0
        epistemic_values = (0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        diagnostics.validate(batch_size)
        peer_count = diagnostics.peer_count
        epistemic_values = (
            float(
                diagnostics.payoff_quantile_standard_deviation_bps[
                    row_index,
                    horizon_index,
                    side_index,
                ]
                .square()
                .mean()
                .sqrt()
                .detach()
                .item()
            ),
            float(
                diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps[
                    row_index,
                    horizon_index,
                    side_index,
                ]
                .square()
                .mean()
                .sqrt()
                .detach()
                .item()
            ),
            float(
                diagnostics.positive_payoff_probability_standard_deviation[
                    row_index,
                    horizon_index,
                    side_index,
                ]
                .detach()
                .item()
            ),
            float(
                diagnostics.adverse_selection_probability_standard_deviation[
                    row_index,
                    horizon_index,
                    side_index,
                ]
                .detach()
                .item()
            ),
            float(
                diagnostics.regime_unpredictability_probability_standard_deviation[
                    row_index,
                    horizon_index,
                ]
                .detach()
                .item()
            ),
        )
    request = Round74AIReviewRequest(
        pretest_policy_sha256=pretest_policy_sha256,
        probability_calibration_sha256=(probability_calibration.calibration_sha256),
        sample_sha256=sample_sha256,
        deterministic_risk_state_sha256=(deterministic_risk_state_sha256),
        risk_profile=risk_profile,
        asset_slot=asset_slot,
        side=side,
        horizon_seconds=horizon_seconds,
        requested_wall_ns=requested_wall_ns,
        expires_wall_ns=expires_wall_ns,
        proposed_risk_size_bps=proposed_risk_size_bps,
        feature_last=feature_last,
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        feature_recent_change=feature_recent_change,
        feature_recent_block_means=feature_recent_block_means,
        payoff_quantiles_bps=payoff_quantiles,
        maximum_adverse_excursion_quantiles_bps=(adverse_excursion_quantiles),
        positive_payoff_probability=positive_payoff_probability,
        opposing_positive_payoff_probability=(opposing_positive_payoff_probability),
        neither_positive_payoff_probability=neither_positive_payoff_probability,
        adverse_selection_probability=_finite_probability(
            adverse[
                row_index,
                horizon_index,
                side_index,
            ],
            "adverse-selection probability",
        ),
        regime_unpredictability_probability=_finite_probability(
            unpredictable[
                row_index,
                horizon_index,
            ],
            "regime-unpredictability probability",
        ),
        epistemic_peer_count=peer_count,
        payoff_quantile_peer_standard_deviation_rms_bps=epistemic_values[0],
        maximum_adverse_excursion_quantile_peer_standard_deviation_rms_bps=(
            epistemic_values[1]
        ),
        positive_payoff_probability_peer_standard_deviation=epistemic_values[2],
        adverse_selection_probability_peer_standard_deviation=epistemic_values[3],
        regime_unpredictability_probability_peer_standard_deviation=(
            epistemic_values[4]
        ),
    )
    request.validate()
    return request


__all__ = [
    "ROUND74_AI_BRIDGE_SCHEMA_VERSION",
    "ROUND74_AI_RECENT_BLOCK_EVENTS",
    "build_round74_ai_review_request",
]
