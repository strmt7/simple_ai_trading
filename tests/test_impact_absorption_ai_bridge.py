from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from simple_ai_trading.impact_absorption_ai_bridge import (
    build_round74_ai_review_request,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    Round74TemperatureFit,
)
from simple_ai_trading.impact_absorption_event_model import (
    Round74EventModelOutput,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)


WALL_NS = 1_800_000_000_000_000_000


def _features() -> torch.Tensor:
    values = torch.zeros(
        (
            2,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        ),
        dtype=torch.float32,
    )
    btc = ROUND74_EVENT_FEATURE_NAMES.index("symbol_is_btcusdt")
    eth = ROUND74_EVENT_FEATURE_NAMES.index("symbol_is_ethusdt")
    values[0, :, btc] = 1.0
    values[1, :, eth] = 1.0
    values[0, :, 10] = torch.linspace(
        -1.0,
        1.0,
        ROUND74_EVENT_SEQUENCE_LENGTH,
    )
    return values


def _output() -> Round74EventModelOutput:
    batch = 2
    horizons = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    sides = len(ROUND74_EVENT_PAYOFF_SIDES)
    quantiles = len(ROUND74_EVENT_PAYOFF_QUANTILES)
    payoff = torch.tensor(
        [-5.0, -1.0, 2.0, 4.0, 7.0],
        dtype=torch.float32,
    ).expand(batch, horizons, sides, quantiles)
    adverse = torch.tensor(
        [1.0, 2.0, 3.0, 5.0, 8.0],
        dtype=torch.float32,
    ).expand(batch, horizons, sides, quantiles)
    return Round74EventModelOutput(
        payoff_quantiles_bps=payoff,
        maximum_adverse_excursion_quantiles_bps=adverse,
        positive_payoff_logits=torch.zeros(
            (batch, horizons, sides),
            dtype=torch.float32,
        ),
        adverse_selection_logits=torch.full(
            (batch, horizons, sides),
            -1.0,
            dtype=torch.float32,
        ),
        regime_unpredictability_logits=torch.full(
            (batch, horizons),
            -2.0,
            dtype=torch.float32,
        ),
    )


def _fit(temperature: float) -> Round74TemperatureFit:
    return Round74TemperatureFit(
        temperature=temperature,
        eligible_observations=2,
        positive_observations=1,
        calibration_runs=6,
        minimum_run_observations=1,
        maximum_run_observations=1,
        uncalibrated_run_balanced_nll=0.5,
        calibrated_run_balanced_nll=0.5,
        uncalibrated_nll=0.5,
        calibrated_nll=0.5,
        uncalibrated_brier=0.2,
        calibrated_brier=0.2,
        uncalibrated_ece=0.1,
        calibrated_ece=0.1,
    )


def _calibration() -> Round74ProbabilityCalibration:
    calibration_run_ids = tuple(f"{index + 1:032x}" for index in range(6))
    return Round74ProbabilityCalibration(
        pretest_policy_sha256="1" * 64,
        tuning_subpartition_sha256="4" * 64,
        calibration_source_sha256="5" * 64,
        calibration_data_sha256="6" * 64,
        calibration_run_ids=calibration_run_ids,
        calibration_row_run_ids_sha256="7" * 64,
        positive_payoff=_fit(2.0),
        adverse_selection=_fit(3.0),
        regime_unpredictability=_fit(4.0),
        backend_kind="cpu",
        backend_device="test",
    )


def _build(**changes: object):
    values: dict[str, object] = {
        "model_output": _output(),
        "scaled_feature_values": _features(),
        "row_index": 0,
        "asset_slot": 0,
        "side": "long",
        "horizon_seconds": 30,
        "pretest_policy_sha256": "1" * 64,
        "sample_sha256": "2" * 64,
        "deterministic_risk_state_sha256": "3" * 64,
        "probability_calibration": _calibration(),
        "requested_wall_ns": WALL_NS,
        "expires_wall_ns": WALL_NS + 10_000_000_000,
        "proposed_risk_size_bps": 2_500,
    }
    values.update(changes)
    return build_round74_ai_review_request(**values)


def test_bridge_builds_target_free_request_from_causal_prediction() -> None:
    request = _build()

    assert request.asset_slot == 0
    assert request.side == "long"
    assert request.horizon_seconds == 30
    assert request.payoff_quantiles_bps == (-5.0, -1.0, 2.0, 4.0, 7.0)
    assert request.maximum_adverse_excursion_quantiles_bps == (
        1.0,
        2.0,
        3.0,
        5.0,
        8.0,
    )
    assert request.positive_payoff_probability == 0.5
    assert request.adverse_selection_probability == pytest.approx(0.4174298)
    assert request.regime_unpredictability_probability == pytest.approx(0.37754067)
    assert request.probability_calibration_sha256 == (_calibration().calibration_sha256)
    assert (
        request.feature_last[ROUND74_EVENT_FEATURE_NAMES.index("symbol_is_btcusdt")]
        == 1.0
    )
    assert request.feature_mean[10] == pytest.approx(0.0, abs=1e-7)
    assert request.feature_standard_deviation[10] > 0.0
    assert request.feature_recent_change[10] > 0.0
    assert (
        request.feature_recent_change[
            ROUND74_EVENT_FEATURE_NAMES.index("symbol_is_btcusdt")
        ]
        == 0.0
    )


def test_bridge_does_not_require_realized_target_arrays() -> None:
    request = _build()
    payload = request.as_dict()

    assert "net_payoff_bps" not in payload
    assert "realized" not in str(payload)
    assert "target_context_sha256" not in payload
    assert payload["future_outcome_exposed_to_ai"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"row_index": 2},
        {"asset_slot": 3},
        {"side": "flat"},
        {"horizon_seconds": 5},
    ],
)
def test_bridge_rejects_invalid_selection(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Round 74 AI bridge"):
        _build(**changes)


def test_bridge_rejects_asset_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="asset identity differs"):
        _build(asset_slot=1)


def test_bridge_rejects_calibration_from_another_policy() -> None:
    calibration = replace(
        _calibration(),
        pretest_policy_sha256="9" * 64,
    )
    with pytest.raises(ValueError, match="calibration policy differs"):
        _build(probability_calibration=calibration)


def test_bridge_rejects_nonfinite_features_or_predictions() -> None:
    features = _features()
    features[0, 2, 10] = float("nan")
    with pytest.raises(ValueError, match="feature context differs"):
        _build(scaled_feature_values=features)

    output = _output()
    logits = output.positive_payoff_logits.clone()
    logits[0, 2, 0] = float("inf")
    with pytest.raises(ValueError, match="model output contains nonfinite"):
        _build(
            model_output=replace(
                output,
                positive_payoff_logits=logits,
            )
        )


def test_bridge_detaches_model_tensors_before_serialization() -> None:
    output = _output()
    logits = output.positive_payoff_logits.clone().requires_grad_(True)
    request = _build(
        model_output=replace(
            output,
            positive_payoff_logits=logits,
        )
    )

    assert isinstance(request.positive_payoff_probability, float)
