from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_TUNING_CALIBRATION_RUNS,
    ROUND74_TUNING_MODEL_SELECTION_RUNS,
    ROUND74_TUNING_POLICY_SELECTION_RUNS,
    Round74ProbabilityCalibration,
    Round74TuningSubpartition,
    apply_round74_probability_calibration,
    build_round74_tuning_subpartition,
    fit_round74_probability_calibration,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
)


def _partition(tuning_runs: int = 24) -> Round74EventRunPartition:
    roles = ("training",) + ("tuning",) * tuning_runs + ("test",)
    entries: list[Round74EventRunPartitionEntry] = []
    base = 1_800_000_000_000_000_000
    spacing = 2_000_000_000_000
    duration = 1_200_000_000_000
    for index, role in enumerate(roles):
        start = base + index * spacing
        entries.append(
            Round74EventRunPartitionEntry(
                run_id=f"{index + 1:032x}",
                role=role,
                capture_report_sha256=f"{index + 1:064x}",
                capture_start_wall_ns=start,
                capture_end_wall_ns=start + duration,
                eligible_anchor_start_wall_ns=start + 320_000_000_000,
                eligible_anchor_end_wall_ns=start + 800_000_000_000,
            )
        )
    return Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="f" * 64,
    )


def _calibration() -> Round74ProbabilityCalibration:
    action_logits = torch.tensor(
        [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0],
        dtype=torch.float32,
    ).reshape(1, 3, 2)
    action_labels = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        dtype=torch.float32,
    ).reshape(1, 3, 2)
    action_mask = torch.ones_like(action_logits)
    regime_logits = torch.tensor(
        [-3.0, -1.0, 1.0, 3.0],
        dtype=torch.float32,
    ).reshape(1, 4)
    regime_labels = torch.tensor(
        [0.0, 1.0, 0.0, 1.0],
        dtype=torch.float32,
    ).reshape(1, 4)
    return fit_round74_probability_calibration(
        positive_payoff_logits=action_logits,
        positive_payoff_labels=action_labels,
        adverse_selection_logits=-action_logits,
        adverse_selection_labels=1.0 - action_labels,
        action_eligibility=action_mask,
        regime_unpredictability_logits=regime_logits,
        regime_unpredictability_labels=regime_labels,
        regime_eligibility=torch.ones_like(regime_logits),
        pretest_policy_sha256="1" * 64,
        tuning_subpartition_sha256="2" * 64,
        calibration_source_sha256="3" * 64,
        backend_kind="cpu",
        backend_device="test",
    )


def test_tuning_subpartition_is_chronological_disjoint_and_bound() -> None:
    split = build_round74_tuning_subpartition(_partition())
    restored = Round74TuningSubpartition.from_dict(split.as_dict())

    assert restored == split
    assert len(split.model_selection_run_ids) == (ROUND74_TUNING_MODEL_SELECTION_RUNS)
    assert len(split.calibration_run_ids) == (ROUND74_TUNING_CALIBRATION_RUNS)
    assert len(split.policy_selection_run_ids) == (ROUND74_TUNING_POLICY_SELECTION_RUNS)
    assert split.model_selection_run_ids[-1] < split.calibration_run_ids[0]
    assert split.calibration_run_ids[-1] < split.policy_selection_run_ids[0]
    assert len(split.subpartition_sha256) == 64
    assert split.as_dict()["sealed_test_run_accessed"] is False


def test_tuning_subpartition_rejects_missing_or_reused_runs() -> None:
    with pytest.raises(ValueError, match="tuning run count differs"):
        build_round74_tuning_subpartition(_partition(tuning_runs=23))

    split = build_round74_tuning_subpartition(_partition())
    with pytest.raises(ValueError, match="subpartition differs"):
        replace(
            split,
            calibration_run_ids=(
                split.model_selection_run_ids[0],
                *split.calibration_run_ids[1:],
            ),
        ).validate()


def test_temperature_calibration_is_deterministic_and_hash_bound() -> None:
    first = _calibration()
    second = _calibration()

    assert first == second
    assert first.calibration_sha256 == second.calibration_sha256
    assert first.positive_payoff.calibrated_nll <= (
        first.positive_payoff.uncalibrated_nll + 1e-7
    )
    assert first.adverse_selection.calibrated_nll <= (
        first.adverse_selection.uncalibrated_nll + 1e-7
    )
    assert first.regime_unpredictability.calibrated_nll <= (
        first.regime_unpredictability.uncalibrated_nll + 1e-7
    )
    payload = first.as_dict()
    assert payload["sealed_test_accessed"] is False
    assert payload["calibration_implies_financial_edge"] is False
    assert payload["candidate_temperature_count"] == 257


def test_temperature_application_uses_frozen_head_specific_values() -> None:
    calibration = _calibration()
    logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    positive, adverse, regime = apply_round74_probability_calibration(
        calibration,
        positive_payoff_logits=logits,
        adverse_selection_logits=logits,
        regime_unpredictability_logits=logits,
    )

    assert torch.equal(
        positive,
        torch.sigmoid(logits / calibration.positive_payoff.temperature),
    )
    assert torch.equal(
        adverse,
        torch.sigmoid(logits / calibration.adverse_selection.temperature),
    )
    assert torch.equal(
        regime,
        torch.sigmoid(logits / calibration.regime_unpredictability.temperature),
    )


def test_temperature_calibration_rejects_missing_class_support() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    labels = torch.zeros_like(logits)
    mask = torch.ones_like(logits)

    with pytest.raises(ValueError, match="class support differs"):
        fit_round74_probability_calibration(
            positive_payoff_logits=logits,
            positive_payoff_labels=labels,
            adverse_selection_logits=logits,
            adverse_selection_labels=labels,
            action_eligibility=mask,
            regime_unpredictability_logits=logits,
            regime_unpredictability_labels=labels,
            regime_eligibility=mask,
            pretest_policy_sha256="1" * 64,
            tuning_subpartition_sha256="2" * 64,
            calibration_source_sha256="3" * 64,
            backend_kind="cpu",
            backend_device="test",
        )


def test_temperature_calibration_rejects_invalid_mask_or_nonfinite_data() -> None:
    logits = torch.tensor(
        [[-1.0, 1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[0.0, 1.0]],
        dtype=torch.float32,
    )
    invalid_mask = torch.tensor(
        [[1.0, 0.5]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="calibration panel differs"):
        fit_round74_probability_calibration(
            positive_payoff_logits=logits,
            positive_payoff_labels=labels,
            adverse_selection_logits=logits,
            adverse_selection_labels=labels,
            action_eligibility=invalid_mask,
            regime_unpredictability_logits=logits,
            regime_unpredictability_labels=labels,
            regime_eligibility=invalid_mask,
            pretest_policy_sha256="1" * 64,
            tuning_subpartition_sha256="2" * 64,
            calibration_source_sha256="3" * 64,
            backend_kind="cpu",
            backend_device="test",
        )
