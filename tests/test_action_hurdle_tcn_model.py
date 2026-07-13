from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from simple_ai_trading.action_hurdle_tcn_model import (
    CANDIDATES,
    FEATURE_COUNT,
    HORIZONS_MINUTES,
    RECEPTIVE_FIELD_STEPS,
    SIDES,
    SUPERVISED_STEPS,
    WINDOW_STEPS,
    DirectActionMeanTCN,
    HurdleActionValueTCN,
    ProbabilityCalibration,
    apply_probability_calibration,
    binary_logit_loss,
    cpu_action_hurdle_preflight,
    fit_action_target_scaler,
    fit_probability_calibration,
    fit_severity_multiplier,
    gamma_mean_score,
    hurdle_expected_net_bps,
    pairwise_action_rank_loss,
    side_net_targets,
)
from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.minute_logistic_mixture_tcn_model import MinuteTemporalDataset


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-049-cost-aware-action-hurdle-tcn-design.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _dataset() -> MinuteTemporalDataset:
    timestamps = 16
    signed = np.zeros((timestamps, len(SYMBOLS), 4), dtype=np.float32)
    signed[:, :, 0] = np.resize(
        np.asarray([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]), timestamps
    )[:, None]
    signed[:, :, 1] = np.resize(
        np.asarray([-40.0, -20.0, 0.0, 20.0, 40.0]), timestamps
    )[:, None]
    training = np.zeros(timestamps, dtype=bool)
    training[:10] = True
    return MinuteTemporalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(FEATURE_COUNT)),
        timestamps_ms=np.arange(timestamps, dtype=np.int64) * 300_000,
        features=np.zeros((timestamps, len(SYMBOLS), FEATURE_COUNT), dtype=np.float32),
        signed_target_bps=signed,
        role_masks={
            "training": training,
            "early_stop": np.arange(timestamps) == 10,
            "calibration": np.arange(timestamps) == 11,
            "viability": np.arange(timestamps) >= 12,
        },
        feature_stream_sha256="a" * 64,
        target_stream_sha256="b" * 64,
        dataset_sha256="c" * 64,
        source_evidence={},
    )


def test_round49_design_identity_and_model_constants_match() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    canonical = dict(design)
    claimed = canonical.pop("design_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert design["round"] == 49
    assert design["status"] == "frozen"
    assert [item["id"] for item in design["model_contract"]["candidates"]] == list(
        CANDIDATES
    )
    assert design["data_contract"]["primary_executable_horizon_minutes"] == 15
    assert design["data_contract"]["auxiliary_nonexecuting_horizon_minutes"] == 30
    assert design["model_contract"]["seeds"] == [4901, 4902, 4903]
    assert RECEPTIVE_FIELD_STEPS == 361
    assert WINDOW_STEPS == 576
    assert SUPERVISED_STEPS == 216


def test_round49_models_have_expected_shapes_and_positive_severities() -> None:
    generator = np.random.default_rng(49)
    values = torch.from_numpy(
        generator.normal(size=(2, FEATURE_COUNT, 48)).astype(np.float32)
    )

    logits, direct, auxiliary = DirectActionMeanTCN()(values)
    assert logits.shape == (2, len(HORIZONS_MINUTES), len(SIDES), 48)
    assert direct.shape == (2, len(SIDES), 48)
    assert auxiliary.shape == (2, 48)

    logits, gain, loss, auxiliary = HurdleActionValueTCN()(values)
    assert logits.shape == (2, len(HORIZONS_MINUTES), len(SIDES), 48)
    assert gain.shape == (2, len(SIDES), 48)
    assert loss.shape == (2, len(SIDES), 48)
    assert auxiliary.shape == (2, 48)
    assert torch.all(gain > 0.0)
    assert torch.all(loss > 0.0)


def test_gamma_score_elicits_the_conditional_arithmetic_mean() -> None:
    target = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]])
    condition = torch.ones_like(target, dtype=torch.bool)
    arithmetic_mean = torch.full_like(target, 4.0)
    too_low = torch.full_like(target, 2.0)
    too_high = torch.full_like(target, 7.0)

    optimum = gamma_mean_score(arithmetic_mean, target, condition)
    assert optimum < gamma_mean_score(too_low, target, condition)
    assert optimum < gamma_mean_score(too_high, target, condition)


def test_directml_safe_binary_logit_loss_matches_pytorch() -> None:
    logits = torch.tensor([-100.0, -2.0, 0.0, 2.0, 100.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])

    torch.testing.assert_close(
        binary_logit_loss(logits, labels),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels),
    )


def test_hurdle_expected_value_uses_the_exact_net_identity() -> None:
    probability = torch.tensor([[[0.25], [0.75]]])
    logits = torch.logit(probability).unsqueeze(1)
    gain_multiplier = torch.ones((1, 2, 1))
    loss_multiplier = torch.ones((1, 2, 1))
    gain_baseline = torch.tensor([[20.0, 30.0]], dtype=torch.float64)
    loss_baseline = torch.tensor([[10.0, 40.0]], dtype=torch.float64)

    actual = hurdle_expected_net_bps(
        logits,
        gain_multiplier,
        loss_multiplier,
        gain_baseline,
        loss_baseline,
    )
    expected = (
        probability * gain_baseline[..., None]
        - (1.0 - probability) * loss_baseline[..., None]
    )
    torch.testing.assert_close(actual, expected.to(dtype=torch.float32))
    assert actual.dtype == torch.float32


def test_pairwise_rank_loss_prefers_correct_temporal_order() -> None:
    target = torch.arange(40, dtype=torch.float32).reshape(1, 1, -1)
    correct = target.clone()
    reversed_prediction = torch.flip(target, dims=(-1,))

    assert pairwise_action_rank_loss(
        correct, target, offset=4
    ) < pairwise_action_rank_loss(reversed_prediction, target, offset=4)


def test_probability_calibration_is_bounded_and_nonworsening() -> None:
    logits = np.linspace(-2.0, 2.0, 1000, dtype=np.float64)
    labels = (logits + np.sin(np.arange(logits.size)) * 0.5 > 0.5).astype(np.float64)
    calibration = fit_probability_calibration(logits, labels)

    assert 0.25 <= calibration.slope <= 4.0
    assert -4.0 <= calibration.intercept <= 4.0
    assert calibration.binary_log_loss_after <= calibration.binary_log_loss_before


def test_probability_calibration_stays_strictly_inside_float32_unit_interval() -> None:
    calibration = ProbabilityCalibration(4.0, 0.0, 0.0, 0.0)
    probabilities = apply_probability_calibration(
        np.asarray([-1e6, 1e6], dtype=np.float32), calibration
    )

    assert probabilities.dtype == np.float32
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities < 1.0)


def test_severity_calibration_recovers_multiplicative_bias() -> None:
    prediction = np.full(100, 10.0)
    target = np.full(100, 15.0)
    condition = np.ones(100, dtype=bool)
    multiplier, before, after = fit_severity_multiplier(prediction, target, condition)

    assert multiplier == 1.5
    assert after < before


def test_side_targets_and_training_scaler_preserve_financial_identity() -> None:
    dataset = _dataset()
    targets = side_net_targets(dataset)
    scaler = fit_action_target_scaler(dataset, targets)

    np.testing.assert_allclose(
        targets[..., 0],
        -dataset.signed_target_bps[..., : len(HORIZONS_MINUTES)] - 12.0,
    )
    np.testing.assert_allclose(
        targets[..., 1],
        dataset.signed_target_bps[..., : len(HORIZONS_MINUTES)] - 12.0,
    )
    assert np.all(scaler.direct_scale_bps > 0.0)
    assert np.all(scaler.gain_mean_bps > 0.0)
    assert np.all(scaler.loss_mean_bps > 0.0)


def test_cpu_preflight_updates_both_candidates_without_nonfinite_values() -> None:
    _, report = cpu_action_hurdle_preflight()

    assert report["backend_kind"] == "cpu"
    assert report["cpu_fallback_warnings"] == 0
    assert [item["candidate_id"] for item in report["candidates"]] == list(CANDIDATES)
    for candidate in report["candidates"]:
        assert candidate["objective"] == candidate["objective"]
        assert (
            candidate["extreme_severity_objective"]
            == candidate["extreme_severity_objective"]
        )
        assert min(candidate["parameter_changes"].values()) > 0.0
