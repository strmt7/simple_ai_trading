from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_RISK_QUANTILE_CALIBRATION_PRIOR_SCHEMA_VERSION,
    ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
    ROUND74_TUNING_CALIBRATION_RUNS,
    ROUND74_TUNING_MODEL_SELECTION_RUNS,
    ROUND74_TUNING_POLICY_SELECTION_RUNS,
    Round74ProbabilityCalibration,
    Round74RiskQuantileCalibration,
    Round74TuningSubpartition,
    apply_round74_probability_calibration,
    apply_round74_risk_quantile_calibration,
    build_round74_tuning_subpartition,
    fit_round74_probability_calibration,
    fit_round74_risk_quantile_calibration,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)
from simple_ai_trading.round74_segmented_model_operator import (
    Round74SegmentedTuningSubpartition,
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


def _tuning_subpartition() -> Round74TuningSubpartition:
    return build_round74_tuning_subpartition(_partition())


def _segmented_tuning_subpartition() -> Round74SegmentedTuningSubpartition:
    run_ids = tuple(f"{index:032x}" for index in range(100, 191))
    result = Round74SegmentedTuningSubpartition(
        parent_partition_sha256="1" * 64,
        cohort_plan_sha256="2" * 64,
        model_selection_run_ids=run_ids[:45],
        calibration_run_ids=run_ids[45:68],
        policy_selection_run_ids=run_ids[68:],
        model_selection_slot_ordinals=tuple(range(514, 559)),
        calibration_slot_ordinals=tuple(range(566, 589)),
        policy_selection_slot_ordinals=tuple(range(592, 615)),
        model_selection_eligible_anchor_ns=(900_000_000_000,) * 45,
        calibration_eligible_anchor_ns=(900_000_000_000,) * 23,
        policy_selection_eligible_anchor_ns=(900_000_000_000,) * 23,
    )
    result.validate()
    return result


def _calibration() -> Round74ProbabilityCalibration:
    action_logits = (
        torch.tensor(
            [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0],
            dtype=torch.float32,
        )
        .reshape(1, 3, 2)
        .repeat(6, 1, 1)
    )
    action_labels = (
        torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            dtype=torch.float32,
        )
        .reshape(1, 3, 2)
        .repeat(6, 1, 1)
    )
    action_mask = torch.ones_like(action_logits)
    regime_logits = (
        torch.tensor(
            [-3.0, -1.0, 1.0, 3.0],
            dtype=torch.float32,
        )
        .reshape(1, 4)
        .repeat(6, 1)
    )
    regime_labels = (
        torch.tensor(
            [0.0, 1.0, 0.0, 1.0],
            dtype=torch.float32,
        )
        .reshape(1, 4)
        .repeat(6, 1)
    )
    subpartition = _tuning_subpartition()
    run_ids = subpartition.calibration_run_ids
    return fit_round74_probability_calibration(
        positive_payoff_logits=action_logits,
        positive_payoff_labels=action_labels,
        adverse_selection_logits=-action_logits,
        adverse_selection_labels=1.0 - action_labels,
        action_eligibility=action_mask,
        regime_unpredictability_logits=regime_logits,
        regime_unpredictability_labels=regime_labels,
        regime_eligibility=torch.ones_like(regime_logits),
        row_run_ids=run_ids,
        tuning_subpartition=subpartition,
        pretest_policy_sha256="1" * 64,
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
    assert first.positive_payoff.calibrated_run_balanced_nll <= (
        first.positive_payoff.uncalibrated_run_balanced_nll + 1e-7
    )
    assert first.adverse_selection.calibrated_run_balanced_nll <= (
        first.adverse_selection.uncalibrated_run_balanced_nll + 1e-7
    )
    assert first.regime_unpredictability.calibrated_run_balanced_nll <= (
        first.regime_unpredictability.uncalibrated_run_balanced_nll + 1e-7
    )
    assert first.positive_payoff.calibration_runs == 6
    assert first.calibration_run_ids == _tuning_subpartition().calibration_run_ids
    payload = first.as_dict()
    assert payload["sealed_test_accessed"] is False
    assert payload["calibration_implies_financial_edge"] is False
    assert payload["candidate_temperature_count"] == 257


def test_temperature_selection_is_invariant_to_busy_run_duplication() -> None:
    subpartition = _tuning_subpartition()
    expected = subpartition.calibration_run_ids
    base_logits = torch.tensor(
        [
            [-4.0, 4.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    base_labels = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    def fit(logits: torch.Tensor, labels: torch.Tensor, runs: tuple[str, ...]):
        mask = torch.ones_like(logits)
        return fit_round74_probability_calibration(
            positive_payoff_logits=logits,
            positive_payoff_labels=labels,
            adverse_selection_logits=-logits,
            adverse_selection_labels=1.0 - labels,
            action_eligibility=mask,
            regime_unpredictability_logits=logits,
            regime_unpredictability_labels=labels,
            regime_eligibility=mask,
            row_run_ids=runs,
            tuning_subpartition=subpartition,
            pretest_policy_sha256="1" * 64,
            calibration_source_sha256="3" * 64,
            backend_kind="cpu",
            backend_device="test",
        )

    baseline = fit(base_logits, base_labels, expected)
    repeated = 100
    busy_logits = torch.cat(
        (base_logits[:1].repeat(repeated, 1), base_logits[1:]),
    )
    busy_labels = torch.cat(
        (base_labels[:1].repeat(repeated, 1), base_labels[1:]),
    )
    busy_runs = (expected[0],) * repeated + expected[1:]
    duplicated = fit(busy_logits, busy_labels, busy_runs)

    assert duplicated.positive_payoff.temperature == (
        baseline.positive_payoff.temperature
    )
    assert duplicated.positive_payoff.maximum_run_observations == repeated * 2
    assert duplicated.positive_payoff.minimum_run_observations == 2
    assert duplicated.calibration_data_sha256 != baseline.calibration_data_sha256


def test_eligible_target_temperature_tracks_duration_normalized_rows() -> None:
    subpartition = _tuning_subpartition()
    expected = subpartition.calibration_run_ids
    base_logits = torch.tensor(
        [
            [-4.0, 4.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    base_labels = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    def fit(logits: torch.Tensor, labels: torch.Tensor, runs: tuple[str, ...]):
        mask = torch.ones_like(logits)
        return fit_round74_probability_calibration(
            positive_payoff_logits=logits,
            positive_payoff_labels=labels,
            adverse_selection_logits=-logits,
            adverse_selection_labels=1.0 - labels,
            action_eligibility=mask,
            regime_unpredictability_logits=logits,
            regime_unpredictability_labels=labels,
            regime_eligibility=mask,
            row_run_ids=runs,
            tuning_subpartition=subpartition,
            pretest_policy_sha256="1" * 64,
            calibration_source_sha256="3" * 64,
            backend_kind="cpu",
            backend_device="test",
            optimization_population="eligible_target",
        )

    baseline = fit(base_logits, base_labels, expected)
    repeated = 100
    duplicated = fit(
        torch.cat((base_logits[:1].repeat(repeated, 1), base_logits[1:])),
        torch.cat((base_labels[:1].repeat(repeated, 1), base_labels[1:])),
        (expected[0],) * repeated + expected[1:],
    )

    assert baseline.optimization_population == "eligible_target"
    assert duplicated.positive_payoff.temperature != (
        baseline.positive_payoff.temperature
    )
    assert duplicated.positive_payoff.calibrated_nll <= (
        duplicated.positive_payoff.uncalibrated_nll + 1e-7
    )
    assert Round74ProbabilityCalibration.from_dict(duplicated.as_dict()) == duplicated


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


def _risk_calibration_panel(
    subpartition: (
        Round74TuningSubpartition | Round74SegmentedTuningSubpartition | None
    ) = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
    tuple[str, ...],
]:
    selected = subpartition or _tuning_subpartition()
    runs = selected.calibration_run_ids
    row_runs = tuple(
        run_id for run_id in runs for _symbol in ROUND74_EVENT_SYMBOLS for _ in range(2)
    )
    row_symbols = tuple(
        symbol for _run_id in runs for symbol in ROUND74_EVENT_SYMBOLS for _ in range(2)
    )
    horizons = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    sides = len(ROUND74_EVENT_PAYOFF_SIDES)
    quantiles = len(ROUND74_EVENT_PAYOFF_QUANTILES)
    rows = len(row_runs)
    payoff = torch.tensor(
        (-2.0, -1.0, 1.0, 2.0, 3.0),
        dtype=torch.float32,
    ).reshape(1, 1, 1, quantiles)
    payoff = payoff.repeat(rows, horizons, sides, 1)
    mae = torch.tensor(
        (0.1, 0.2, 0.3, 0.4, 0.5),
        dtype=torch.float32,
    ).reshape(1, 1, 1, quantiles)
    mae = mae.repeat(rows, horizons, sides, 1)
    payoff_targets = (
        torch.tensor(
            (-3.0, 1.0),
            dtype=torch.float32,
        )
        .reshape(2, 1, 1)
        .repeat(len(runs) * len(ROUND74_EVENT_SYMBOLS), horizons, sides)
    )
    mae_targets = (
        torch.tensor(
            (1.5, 0.2),
            dtype=torch.float32,
        )
        .reshape(2, 1, 1)
        .repeat(len(runs) * len(ROUND74_EVENT_SYMBOLS), horizons, sides)
    )
    eligibility = torch.ones((rows, horizons, sides), dtype=torch.float32)
    return (
        payoff,
        payoff_targets,
        mae,
        mae_targets,
        eligibility,
        row_runs,
        row_symbols,
    )


def test_segmented_calibration_uses_every_frozen_calibration_segment() -> None:
    subpartition = _segmented_tuning_subpartition()
    (
        payoff,
        targets,
        mae,
        mae_targets,
        eligibility,
        row_runs,
        row_symbols,
    ) = _risk_calibration_panel(subpartition)
    rows = len(row_runs)
    action_logits = torch.tensor(
        (-2.0, 2.0),
        dtype=torch.float32,
    ).repeat(rows, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS), 1)
    action_labels = torch.tensor(
        (0.0, 1.0),
        dtype=torch.float32,
    ).repeat(rows, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS), 1)
    regime_logits = torch.tensor(
        (-2.0, 2.0, -2.0, 2.0),
        dtype=torch.float32,
    ).repeat(rows, 1)
    regime_labels = torch.tensor(
        (0.0, 1.0, 0.0, 1.0),
        dtype=torch.float32,
    ).repeat(rows, 1)

    calibration = fit_round74_probability_calibration(
        positive_payoff_logits=action_logits,
        positive_payoff_labels=action_labels,
        adverse_selection_logits=-action_logits,
        adverse_selection_labels=1.0 - action_labels,
        action_eligibility=eligibility,
        regime_unpredictability_logits=regime_logits,
        regime_unpredictability_labels=regime_labels,
        regime_eligibility=torch.ones_like(regime_logits),
        payoff_quantiles_bps=payoff,
        net_payoff_bps=targets,
        maximum_adverse_excursion_quantiles_bps=mae,
        maximum_adverse_excursion_bps=mae_targets,
        row_symbols=row_symbols,
        row_run_ids=row_runs,
        tuning_subpartition=subpartition,
        pretest_policy_sha256="1" * 64,
        calibration_source_sha256="3" * 64,
        backend_kind="cpu",
        backend_device="test",
        optimization_population="eligible_target",
    )

    assert calibration.calibration_run_ids == subpartition.calibration_run_ids
    assert calibration.positive_payoff.calibration_runs == 23
    assert calibration.adverse_selection.calibration_runs == 23
    assert calibration.regime_unpredictability.calibration_runs == 23
    assert calibration.risk_quantiles is not None
    assert calibration.risk_quantiles.calibration_runs == 23
    assert Round74ProbabilityCalibration.from_dict(calibration.as_dict()) == calibration


def test_risk_quantile_calibration_widens_only_deployed_tails() -> None:
    (
        payoff,
        targets,
        mae,
        mae_targets,
        eligibility,
        row_runs,
        row_symbols,
    ) = _risk_calibration_panel()
    calibration = fit_round74_risk_quantile_calibration(
        payoff_quantiles_bps=payoff,
        net_payoff_bps=targets,
        maximum_adverse_excursion_quantiles_bps=mae,
        maximum_adverse_excursion_bps=mae_targets,
        action_eligibility=eligibility,
        row_run_ids=row_runs,
        row_symbols=row_symbols,
        expected_run_ids=_tuning_subpartition().calibration_run_ids,
        optimization_population="capture_run",
    )
    adjusted_payoff, adjusted_mae = apply_round74_risk_quantile_calibration(
        calibration,
        payoff_quantiles_bps=payoff,
        maximum_adverse_excursion_quantiles_bps=mae,
    )

    assert calibration.payoff_lower_offsets_bps[0][0] == pytest.approx((1.0, 2.0))
    assert calibration.mae_upper_offsets_bps[0][0] == pytest.approx(1.0)
    assert calibration.payoff_lower_empirical_coverage_before[0][0] == (
        pytest.approx((0.5, 0.5))
    )
    assert calibration.payoff_lower_empirical_coverage_after[0][0] == (
        pytest.approx((1.0, 1.0))
    )
    assert calibration.mae_upper_empirical_coverage_before[0][0] == (pytest.approx(0.5))
    assert calibration.mae_upper_empirical_coverage_after[0][0] == (pytest.approx(1.0))
    assert torch.equal(adjusted_payoff[..., 2:], payoff[..., 2:])
    assert torch.all(adjusted_payoff[..., 0] == -3.0)
    assert torch.all(adjusted_payoff[..., 1] == -3.0)
    assert torch.equal(adjusted_mae[..., :-1], mae[..., :-1])
    assert torch.allclose(adjusted_mae[..., -1], torch.full_like(mae[..., -1], 1.5))
    assert (
        Round74RiskQuantileCalibration.from_dict(calibration.as_dict()) == calibration
    )
    prior = replace(
        calibration,
        schema_version=ROUND74_RISK_QUANTILE_CALIBRATION_PRIOR_SCHEMA_VERSION,
    )
    assert Round74RiskQuantileCalibration.from_dict(prior.as_dict()) == prior


def test_risk_quantile_calibration_uses_worst_run_symbol_group() -> None:
    (
        payoff,
        targets,
        mae,
        mae_targets,
        eligibility,
        row_runs,
        row_symbols,
    ) = _risk_calibration_panel()
    sol_mask = torch.tensor(
        tuple(symbol == "SOLUSDT" for symbol in row_symbols),
        dtype=torch.bool,
    )
    targets[sol_mask] -= 4.0
    mae_targets[sol_mask] += 4.0

    calibration = fit_round74_risk_quantile_calibration(
        payoff_quantiles_bps=payoff,
        net_payoff_bps=targets,
        maximum_adverse_excursion_quantiles_bps=mae,
        maximum_adverse_excursion_bps=mae_targets,
        action_eligibility=eligibility,
        row_run_ids=row_runs,
        row_symbols=row_symbols,
        expected_run_ids=_tuning_subpartition().calibration_run_ids,
        optimization_population="capture_run",
    )

    assert calibration.payoff_lower_offsets_bps[0][0] == pytest.approx((5.0, 6.0))
    assert calibration.mae_upper_offsets_bps[0][0] == pytest.approx(5.0)
    assert calibration.payoff_lower_empirical_coverage_after[0][0] == (
        pytest.approx((1.0, 1.0))
    )
    assert calibration.mae_upper_empirical_coverage_after[0][0] == pytest.approx(1.0)


def test_risk_quantile_calibration_rejects_undercovered_fitted_tail() -> None:
    (
        payoff,
        targets,
        mae,
        mae_targets,
        eligibility,
        row_runs,
        row_symbols,
    ) = _risk_calibration_panel()
    calibration = fit_round74_risk_quantile_calibration(
        payoff_quantiles_bps=payoff,
        net_payoff_bps=targets,
        maximum_adverse_excursion_quantiles_bps=mae,
        maximum_adverse_excursion_bps=mae_targets,
        action_eligibility=eligibility,
        row_run_ids=row_runs,
        row_symbols=row_symbols,
        expected_run_ids=_tuning_subpartition().calibration_run_ids,
        optimization_population="capture_run",
    )
    undercovered = replace(
        calibration,
        payoff_lower_empirical_coverage_after=tuple(
            tuple(
                (0.89, pair[1]) if horizon == 0 and side == 0 else pair
                for side, pair in enumerate(row)
            )
            for horizon, row in enumerate(
                calibration.payoff_lower_empirical_coverage_after
            )
        ),
    )

    with pytest.raises(ValueError, match="bounds differ"):
        undercovered.validate()


def test_composite_calibration_hash_binds_risk_targets_and_offsets() -> None:
    (
        payoff,
        targets,
        mae,
        mae_targets,
        eligibility,
        row_runs,
        row_symbols,
    ) = _risk_calibration_panel()
    action_logits = torch.tensor(
        (-2.0, 2.0),
        dtype=torch.float32,
    ).repeat(len(row_runs), len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS), 1)
    action_labels = torch.tensor(
        (0.0, 1.0),
        dtype=torch.float32,
    ).repeat(len(row_runs), len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS), 1)
    action_labels[-4:] = 1.0 - action_labels[-4:]
    regime_logits = torch.full(
        (len(row_runs), len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)),
        -2.0,
        dtype=torch.float32,
    )
    regime_labels = torch.zeros_like(regime_logits)
    regime_labels[-4:] = 1.0
    subpartition = _tuning_subpartition()
    calibration = fit_round74_probability_calibration(
        positive_payoff_logits=action_logits,
        positive_payoff_labels=action_labels,
        adverse_selection_logits=-action_logits,
        adverse_selection_labels=1.0 - action_labels,
        action_eligibility=eligibility,
        regime_unpredictability_logits=regime_logits,
        regime_unpredictability_labels=regime_labels,
        regime_eligibility=torch.ones_like(regime_logits),
        payoff_quantiles_bps=payoff,
        net_payoff_bps=targets,
        maximum_adverse_excursion_quantiles_bps=mae,
        maximum_adverse_excursion_bps=mae_targets,
        row_symbols=row_symbols,
        row_run_ids=row_runs,
        tuning_subpartition=subpartition,
        pretest_policy_sha256="1" * 64,
        calibration_source_sha256="3" * 64,
        backend_kind="cpu",
        backend_device="test",
        optimization_population="eligible_target",
    )

    assert calibration.schema_version == ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
    assert calibration.risk_quantiles is not None
    assert calibration.risk_quantiles.optimization_population == "eligible_target"
    assert Round74ProbabilityCalibration.from_dict(calibration.as_dict()) == calibration


def test_temperature_calibration_rejects_missing_class_support() -> None:
    logits = torch.zeros((6, 2), dtype=torch.float32)
    labels = torch.zeros_like(logits)
    mask = torch.ones_like(logits)
    subpartition = _tuning_subpartition()
    run_ids = subpartition.calibration_run_ids

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
            row_run_ids=run_ids,
            tuning_subpartition=subpartition,
            pretest_policy_sha256="1" * 64,
            calibration_source_sha256="3" * 64,
            backend_kind="cpu",
            backend_device="test",
        )


def test_temperature_calibration_rejects_invalid_mask_or_nonfinite_data() -> None:
    logits = torch.tensor(
        [[-1.0, 1.0]] * 6,
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[0.0, 1.0]] * 6,
        dtype=torch.float32,
    )
    invalid_mask = torch.tensor(
        [[1.0, 0.5]] * 6,
        dtype=torch.float32,
    )
    subpartition = _tuning_subpartition()
    run_ids = subpartition.calibration_run_ids

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
            row_run_ids=run_ids,
            tuning_subpartition=subpartition,
            pretest_policy_sha256="1" * 64,
            calibration_source_sha256="3" * 64,
            backend_kind="cpu",
            backend_device="test",
        )
