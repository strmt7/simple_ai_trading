from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from simple_ai_trading import (
    impact_absorption_ai_execution_replay as replay_subject,
    impact_absorption_event_action_policy as action_policy_subject,
    impact_absorption_event_sealed_evaluation as sealed_subject,
    round74_ai_qualification_operator as qualification_subject,
)
from simple_ai_trading.impact_absorption_ai_protocol import (
    Round74AIReviewDecision,
)
from simple_ai_trading.impact_absorption_ai_execution_replay import (
    Round74AIQualificationStoreExecutionReplayProvider,
    Round74AIExecutionReplayInstruction,
    Round74SealedStoreExecutionReplayProvider,
    build_round74_ai_execution_replay_instructions,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (
    Round74PreparedSealedAIReviewProvider,
    prepare_round74_target_free_ai_reviews,
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    Round74AIRuntimeOutcome,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
    Round74AIPretestQualificationPanel,
    Round74AIQualificationPopulation,
    build_round74_ai_pretest_qualification,
    evaluate_round74_ai_overlay_development,
)
from simple_ai_trading.impact_absorption_event_action_policy import (
    Round74ActionPolicySelection,
    Round74ActionThresholdEvaluation,
    Round74ActionTrace,
    build_round74_action_inference_context,
    derive_round74_action_candidates,
    round74_action_profile,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
    Round74NoInformationQuantileBaseline,
    Round74ProbabilityCalibration,
    Round74RiskQuantileCalibration,
    Round74TemperatureFit,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_model import (
    Round74EventModelOutput,
)
from simple_ai_trading.impact_absorption_event_sealed_ledger import (
    Round74SealedEvaluationClaim,
    Round74SealedEvaluationLedger,
    Round74SealedLedgerError,
    Round74SealedReuseError,
    build_round74_sealed_dataset_identity,
)
from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    Round74TargetFreeCandidateInference,
    evaluate_round74_sealed_once,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
    ROUND74_EVENT_SYMBOLS,
)
from simple_ai_trading.impact_absorption_store import ImpactAbsorptionStore
from simple_ai_trading.impact_absorption_target_assembly import (
    Round74SourceTargetAssembly,
)
from simple_ai_trading.round74_segmented_model_operator import (
    Round74SegmentedTestPopulation,
    build_round74_segmented_sealed_dataset_identity,
)
from simple_ai_trading.round74_event_model_operator import (
    Round74PreparedTuningRoles,
)


TEST_RUNS = tuple(f"{index:032x}" for index in range(100, 124))
POLICY_RUNS = tuple(f"{index:032x}" for index in range(200, 206))


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _test_batch(
    *,
    role: str = "test",
    runs: tuple[str, ...] = TEST_RUNS,
) -> Round74EventTrainingBatch:
    rows = len(runs)
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    wall = np.arange(rows, dtype=np.int64) * 4_000_000_000_000
    wall += 1_800_000_000_000_000_000
    features = np.zeros(
        (
            rows,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    symbols = tuple(
        ROUND74_EVENT_SYMBOLS[index % len(ROUND74_EVENT_SYMBOLS)]
        for index in range(rows)
    )
    for index, symbol in enumerate(symbols):
        features[
            index,
            :,
            ROUND74_EVENT_FEATURE_NAMES.index(f"symbol_is_{symbol.lower()}"),
        ] = 1.0
    entry = np.full(action_shape, 10, dtype=np.int64)
    exit_value = np.full(action_shape, 20, dtype=np.int64)
    result = Round74EventTrainingBatch(
        role=role,
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=runs,
        symbol=symbols,
        decision_monotonic_ns=_readonly(np.full(rows, 1_000_000_000, dtype=np.int64)),
        decision_wall_ns=_readonly(wall),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.arange(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{300 + index:064x}" for index in range(rows)),
        feature_window_sha256=tuple(f"{400 + index:064x}" for index in range(rows)),
        target_context_sha256=tuple("3" * 64 for _ in range(rows)),
        test_access_sha256=tuple(
            "4" * 64 if role == "test" else "" for _ in range(rows)
        ),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(entry),
        actual_exit_monotonic_ns=_readonly(exit_value),
        net_payoff_bps=_readonly(np.ones(action_shape, dtype=np.float32)),
        maximum_adverse_excursion_bps=_readonly(
            np.ones(action_shape, dtype=np.float32)
        ),
        adverse_selection=_readonly(np.zeros(action_shape, dtype=np.float32)),
        regime_unpredictability=_readonly(np.zeros(regime_shape, dtype=np.float32)),
        action_eligibility=_readonly(np.ones(action_shape, dtype=np.float32)),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    result.validate()
    return result


def _selection(
    *,
    policy_runs: tuple[str, ...] = POLICY_RUNS,
    optimization_population: str = "capture_run",
) -> Round74ActionPolicySelection:
    rows = len(policy_runs)
    symbols = tuple(
        ROUND74_EVENT_SYMBOLS[index % len(ROUND74_EVENT_SYMBOLS)]
        for index in range(rows)
    )
    payoff_pattern = (2.0, -1.0, 2.0, -1.0, 2.0, 2.0)
    net_payoff_bps = tuple(
        payoff_pattern[index % len(payoff_pattern)] / 3.0 for index in range(rows)
    )
    maximum_adverse_excursion_bps = (1.0 / 3.0,) * rows
    entry_monotonic_ns = (10,) * rows
    exit_monotonic_ns = (20,) * rows
    metrics = action_policy_subject._trace_metrics(
        run_ids=policy_runs,
        symbols=symbols,
        net_payoff_bps=net_payoff_bps,
        maximum_adverse_excursion_bps=maximum_adverse_excursion_bps,
        adverse_selection=(0,) * rows,
        entry_monotonic_ns=entry_monotonic_ns,
        exit_monotonic_ns=exit_monotonic_ns,
        expected_run_ids=policy_runs,
    )
    trace = Round74ActionTrace(
        threshold_score=1.0,
        expected_run_ids=policy_runs,
        row_index=tuple(range(rows)),
        run_id=policy_runs,
        symbol=symbols,
        feature_row_sha256=tuple(f"{400 + index:064x}" for index in range(rows)),
        horizon_seconds=(30,) * rows,
        side=(1,) * rows,
        entry_monotonic_ns=entry_monotonic_ns,
        exit_monotonic_ns=exit_monotonic_ns,
        net_payoff_bps=net_payoff_bps,
        maximum_adverse_excursion_bps=maximum_adverse_excursion_bps,
        adverse_selection=(0,) * rows,
        skipped_target_ineligible=0,
        skipped_same_symbol_overlap=0,
        metrics=metrics,
    )
    quantiles = round74_action_profile("aggressive").threshold_quantiles
    evaluations = tuple(
        Round74ActionThresholdEvaluation(
            quantile=quantile,
            threshold_score=1.0,
            objective_bps=6.0,
            accepted=True,
            rejection_reasons=(),
            trace=trace,
            objective_semantics=(
                "total_net_bps_minus_worst_drawdown_and_total_mae_penalties"
                if optimization_population == "eligible_target"
                else (
                    "mean_run_net_bps_minus_worst_drawdown_and_mean_run_mae_penalties"
                )
            ),
        )
        for quantile in quantiles
    )
    result = Round74ActionPolicySelection(
        profile="aggressive",
        pretest_policy_sha256="5" * 64,
        probability_calibration_sha256="6" * 64,
        tuning_subpartition_sha256="7" * 64,
        target_batch_sha256=("8" * 64,),
        candidate_sha256=("9" * 64,),
        accepted=True,
        selected_quantile=quantiles[0],
        selected_threshold_score=1.0,
        evaluations=evaluations,
        rejection_reasons=(),
        optimization_population=optimization_population,
    )
    result.validate()
    return result


def _calibration() -> Round74ProbabilityCalibration:
    horizons = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    sides = len(ROUND74_EVENT_PAYOFF_SIDES)
    quantiles = len(ROUND74_EVENT_PAYOFF_QUANTILES)
    fit = Round74TemperatureFit(
        temperature=1.0,
        eligible_observations=10,
        positive_observations=5,
        calibration_runs=6,
        minimum_run_observations=1,
        maximum_run_observations=2,
        uncalibrated_run_balanced_nll=0.5,
        calibrated_run_balanced_nll=0.5,
        uncalibrated_nll=0.5,
        calibrated_nll=0.5,
        uncalibrated_brier=0.2,
        calibrated_brier=0.2,
        uncalibrated_ece=0.1,
        calibrated_ece=0.1,
    )
    zero_lower = tuple(
        tuple((0.0, 0.0) for _side in range(sides)) for _horizon in range(horizons)
    )
    full_lower_coverage = tuple(
        tuple((1.0, 1.0) for _side in range(sides)) for _horizon in range(horizons)
    )
    zero_matrix = tuple(
        tuple(0.0 for _side in range(sides)) for _horizon in range(horizons)
    )
    full_coverage = tuple(
        tuple(1.0 for _side in range(sides)) for _horizon in range(horizons)
    )
    observation_matrix = tuple(
        tuple(6 for _side in range(sides)) for _horizon in range(horizons)
    )
    risk_quantiles = Round74RiskQuantileCalibration(
        payoff_lower_offsets_bps=zero_lower,
        mae_upper_offsets_bps=zero_matrix,
        eligible_observations=observation_matrix,
        payoff_lower_empirical_coverage_before=full_lower_coverage,
        payoff_lower_empirical_coverage_after=full_lower_coverage,
        mae_upper_empirical_coverage_before=full_coverage,
        mae_upper_empirical_coverage_after=full_coverage,
        calibration_runs=6,
        optimization_population="capture_run",
    )
    payoff_baseline = tuple(
        tuple(
            tuple(
                tuple(0.0 for _quantile in range(quantiles)) for _side in range(sides)
            )
            for _horizon in range(horizons)
        )
        for _symbol in ROUND74_EVENT_SYMBOLS
    )
    mae_baseline = tuple(
        tuple(
            tuple(
                tuple(4.0 for _quantile in range(quantiles)) for _side in range(sides)
            )
            for _horizon in range(horizons)
        )
        for _symbol in ROUND74_EVENT_SYMBOLS
    )
    baseline_observations = tuple(
        tuple(tuple(6 for _side in range(sides)) for _horizon in range(horizons))
        for _symbol in ROUND74_EVENT_SYMBOLS
    )
    quantile_baseline = Round74NoInformationQuantileBaseline(
        payoff_quantiles_bps=payoff_baseline,
        maximum_adverse_excursion_quantiles_bps=mae_baseline,
        eligible_observations=baseline_observations,
        calibration_runs=6,
        optimization_population="capture_run",
    )
    result = Round74ProbabilityCalibration(
        pretest_policy_sha256="5" * 64,
        tuning_subpartition_sha256="7" * 64,
        calibration_source_sha256="a" * 64,
        calibration_data_sha256="b" * 64,
        calibration_run_ids=tuple(f"{index + 1:032x}" for index in range(6)),
        calibration_row_run_ids_sha256="c" * 64,
        positive_payoff=fit,
        adverse_selection=fit,
        regime_unpredictability=fit,
        backend_kind="cpu",
        backend_device="test",
        risk_quantiles=risk_quantiles,
        quantile_baseline=quantile_baseline,
        schema_version=ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
    )
    result.validate()
    return result


def _model_output(rows: int) -> Round74EventModelOutput:
    horizons = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    sides = len(ROUND74_EVENT_PAYOFF_SIDES)
    quantiles = len(ROUND74_EVENT_PAYOFF_QUANTILES)
    payoff = torch.full(
        (rows, horizons, sides, quantiles),
        -5.0,
        dtype=torch.float32,
    )
    mae = torch.ones_like(payoff)
    positive = torch.full((rows, horizons, sides), -2.0, dtype=torch.float32)
    adverse = torch.full((rows, horizons, sides), 2.0, dtype=torch.float32)
    regime = torch.full((rows, horizons), -2.0, dtype=torch.float32)
    horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(30)
    payoff[:, horizon_index, 0, :] = torch.tensor((2.0, 4.0, 8.0, 10.0, 12.0))
    selected_mae = torch.tensor((0.2, 0.4, 0.8, 1.2, 2.0))
    mae[:, :horizon_index, 0, :] = torch.minimum(
        mae[:, :horizon_index, 0, :],
        selected_mae,
    )
    mae[:, horizon_index:, 0, :] = torch.maximum(
        mae[:, horizon_index:, 0, :],
        selected_mae,
    )
    positive[:, horizon_index, 0] = 2.0
    adverse[:, horizon_index, 0] = -2.0
    result = Round74EventModelOutput(
        payoff_quantiles_bps=payoff,
        maximum_adverse_excursion_quantiles_bps=mae,
        positive_payoff_logits=positive,
        adverse_selection_logits=adverse,
        regime_unpredictability_logits=regime,
    )
    result.validate(rows)
    return result


class _Model(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        return _model_output(int(values.shape[0]))


def _reviews(
    batch: Round74EventTrainingBatch,
    calibration: Round74ProbabilityCalibration,
    selection: Round74ActionPolicySelection,
    *,
    manifest: str,
) -> tuple[Round74AIPairedReviewEvidence, ...]:
    candidates = derive_round74_action_candidates(
        _model_output(batch.rows),
        build_round74_action_inference_context(batch),
        calibration,
        pretest_policy_sha256=selection.pretest_policy_sha256,
        profile=selection.profile,
    )
    decision = Round74AIReviewDecision(
        verdict="allow_unchanged",
        size_multiplier_bps=10_000,
        confidence_bps=8_000,
        reason_codes=("none",),
    )
    result = tuple(
        Round74AIPairedReviewEvidence(
            row_index=index,
            feature_row_sha256=candidates.feature_row_sha256[index],
            run_id=candidates.run_id[index],
            symbol=candidates.symbol[index],
            side=int(candidates.side[index]),
            horizon_seconds=int(candidates.horizon_seconds[index]),
            pretest_policy_sha256=selection.pretest_policy_sha256,
            probability_calibration_sha256=(selection.probability_calibration_sha256),
            request_sha256=f"{1_000 + index:064x}",
            runtime_outcome_sha256=f"{2_000 + index:064x}",
            model_manifest_sha256=manifest,
            runtime_status="accepted",
            runtime_elapsed_ns=1_000,
            queue_delay_ns=0,
            effective_review_latency_ns=1_000,
            action_validity_latency_ns=ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
            action_latency_eligible=True,
            size_multiplier_bps=10_000,
            decision=decision,
        )
        for index in range(batch.rows)
        if candidates.eligible[index]
        and candidates.quality_score[index]
        >= float(selection.selected_threshold_score or 0.0)
    )
    for value in result:
        value.validate()
    return result


def _execution_replays(
    reviews: tuple[Round74AIPairedReviewEvidence, ...],
    *,
    partition_sha256: str,
) -> tuple[Round74AIExecutionReplayEvidence, ...]:
    result: list[Round74AIExecutionReplayEvidence] = []
    for index, review in enumerate(reviews):
        requested_multiplier = (
            review.decision.size_multiplier_bps
            if review.runtime_status == "accepted" and review.decision is not None
            else 0
        )
        executed = requested_multiplier > 0 and review.action_latency_eligible
        status = (
            "executed"
            if executed
            else (
                "historical_review_expired" if requested_multiplier > 0 else "ai_veto"
            )
        )
        evidence = Round74AIExecutionReplayEvidence(
            row_index=review.row_index,
            feature_row_sha256=review.feature_row_sha256,
            run_id=review.run_id,
            symbol=review.symbol,
            side=review.side,
            horizon_seconds=review.horizon_seconds,
            source_review_sha256=review.review_sha256,
            partition_sha256=partition_sha256,
            source_capture_report_sha256=f"{3_000 + index:064x}",
            target_spec_sha256="f" * 64,
            status=status,
            requested_size_multiplier_bps=requested_multiplier,
            applied_size_multiplier_bps=(requested_multiplier if executed else 0),
            exact_l2_replay_performed=executed,
            target_outcome_sha256=(f"{4_000 + index:064x}" if executed else None),
            target_context_sha256=(f"{5_000 + index:064x}" if executed else None),
            target_ineligible_reason="",
            requested_entry_monotonic_ns=10 if executed else None,
            actual_entry_monotonic_ns=11 if executed else None,
            actual_exit_monotonic_ns=20 if executed else None,
            reference_quote_notional=1_000.0 if executed else None,
            actual_entry_quote_notional=1_000.0 if executed else None,
            actual_deployed_capital_bps=10_000.0 if executed else 0.0,
            position_net_payoff_bps=1.0 if executed else 0.0,
            position_maximum_adverse_excursion_bps=(1.0 if executed else 0.0),
            capital_scaled_net_payoff_bps=1.0 if executed else 0.0,
            capital_scaled_maximum_adverse_excursion_bps=(1.0 if executed else 0.0),
            adverse_selection=False,
        )
        evidence.validate()
        result.append(evidence)
    return tuple(result)


def _ai_pretest_qualification(
    selection: Round74ActionPolicySelection,
    *,
    manifests: tuple[str, str] = ("a" * 64, "b" * 64),
) -> Round74AIPretestQualificationPanel:
    selected = [
        evaluation
        for evaluation in selection.evaluations
        if evaluation.accepted
        and evaluation.quantile == selection.selected_quantile
        and evaluation.threshold_score == selection.selected_threshold_score
    ]
    assert len(selected) == 1
    policy_trace = selected[0].trace
    qualification_run_ids = tuple(
        f"{90_000 + index:032x}" for index in range(len(policy_trace.expected_run_ids))
    )
    trace = replace(
        policy_trace,
        expected_run_ids=qualification_run_ids,
        run_id=qualification_run_ids,
        feature_row_sha256=tuple(
            f"{91_000 + index:064x}" for index in range(len(qualification_run_ids))
        ),
    )
    trace.validate()
    qualification_population = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256=selection.tuning_subpartition_sha256,
        prior_run_ids=policy_trace.expected_run_ids,
        prior_slot_ordinals=tuple(range(100, 100 + len(policy_trace.expected_run_ids))),
        run_ids=qualification_run_ids,
        slot_ordinals=tuple(range(200, 200 + len(qualification_run_ids))),
        eligible_anchor_ns=(900_000_000_000,) * len(qualification_run_ids),
    )
    qualification_population.validate()
    reports = []
    for model_index, manifest in enumerate(manifests):
        reviews = []
        executions = []
        for index, baseline_net_bps in enumerate(trace.net_payoff_bps):
            retained = baseline_net_bps >= 0.0
            multiplier = 10_000 if retained else 0
            decision = Round74AIReviewDecision(
                verdict="allow_unchanged" if retained else "veto",
                size_multiplier_bps=multiplier,
                confidence_bps=8_000,
                reason_codes=("none",) if retained else ("forecast_uncertainty",),
            )
            review = Round74AIPairedReviewEvidence(
                row_index=trace.row_index[index],
                feature_row_sha256=trace.feature_row_sha256[index],
                run_id=trace.run_id[index],
                symbol=trace.symbol[index],
                side=trace.side[index],
                horizon_seconds=trace.horizon_seconds[index],
                pretest_policy_sha256=selection.pretest_policy_sha256,
                probability_calibration_sha256=(
                    selection.probability_calibration_sha256
                ),
                request_sha256=f"{10_000 + model_index * 1_000 + index:064x}",
                runtime_outcome_sha256=(f"{20_000 + model_index * 1_000 + index:064x}"),
                model_manifest_sha256=manifest,
                runtime_status="accepted",
                runtime_elapsed_ns=1_000,
                queue_delay_ns=0,
                effective_review_latency_ns=1_000,
                action_validity_latency_ns=(ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS),
                action_latency_eligible=True,
                size_multiplier_bps=multiplier,
                decision=decision,
            )
            review.validate()
            reviews.append(review)
            position_net_bps = (
                baseline_net_bps / trace.position_capital_fraction if retained else 0.0
            )
            position_mae_bps = (
                trace.maximum_adverse_excursion_bps[index]
                / trace.position_capital_fraction
                if retained
                else 0.0
            )
            execution = Round74AIExecutionReplayEvidence(
                row_index=review.row_index,
                feature_row_sha256=review.feature_row_sha256,
                run_id=review.run_id,
                symbol=review.symbol,
                side=review.side,
                horizon_seconds=review.horizon_seconds,
                source_review_sha256=review.review_sha256,
                partition_sha256="e" * 64,
                source_capture_report_sha256=(
                    f"{30_000 + model_index * 1_000 + index:064x}"
                ),
                target_spec_sha256="f" * 64,
                status="executed" if retained else "ai_veto",
                requested_size_multiplier_bps=multiplier,
                applied_size_multiplier_bps=multiplier,
                exact_l2_replay_performed=retained,
                target_outcome_sha256=(
                    f"{40_000 + model_index * 1_000 + index:064x}" if retained else None
                ),
                target_context_sha256=(
                    f"{50_000 + model_index * 1_000 + index:064x}" if retained else None
                ),
                target_ineligible_reason="",
                requested_entry_monotonic_ns=(
                    trace.entry_monotonic_ns[index] if retained else None
                ),
                actual_entry_monotonic_ns=(
                    trace.entry_monotonic_ns[index] if retained else None
                ),
                actual_exit_monotonic_ns=(
                    trace.exit_monotonic_ns[index] if retained else None
                ),
                reference_quote_notional=1_000.0 if retained else None,
                actual_entry_quote_notional=1_000.0 if retained else None,
                actual_deployed_capital_bps=10_000.0 if retained else 0.0,
                position_net_payoff_bps=position_net_bps,
                position_maximum_adverse_excursion_bps=position_mae_bps,
                capital_scaled_net_payoff_bps=position_net_bps,
                capital_scaled_maximum_adverse_excursion_bps=position_mae_bps,
                adverse_selection=bool(trace.adverse_selection[index]),
            )
            execution.validate()
            executions.append(execution)
        report = evaluate_round74_ai_overlay_development(
            selection,
            tuple(reviews),
            tuple(executions),
            qualification_population=qualification_population,
            qualification_trace=trace,
            qualification_candidate_sha256=("d" * 64,),
        )
        assert report.development_gate_passed
        reports.append(report)
    qualification = build_round74_ai_pretest_qualification(tuple(reports))
    assert qualification.qualification_passed
    return qualification


def _candidate_inference(
    batch: Round74EventTrainingBatch,
    calibration: Round74ProbabilityCalibration,
    selection: Round74ActionPolicySelection,
) -> Round74TargetFreeCandidateInference:
    context = build_round74_action_inference_context(batch)
    output = _model_output(batch.rows)
    candidates = derive_round74_action_candidates(
        output,
        context,
        calibration,
        pretest_policy_sha256=selection.pretest_policy_sha256,
        profile=selection.profile,
    )
    result = Round74TargetFreeCandidateInference(
        contexts=(context,),
        model_outputs=(output,),
        candidates=(candidates,),
        pretest_policy_sha256=selection.pretest_policy_sha256,
        pretest_model_sha256="d" * 64,
        probability_calibration_sha256=calibration.calibration_sha256,
        action_selection_sha256=selection.selection_sha256,
        profile=selection.profile,
        inference_backend_kind="cpu",
        inference_backend_device="cpu",
        inference_backend_vendor="test",
        inference_warning_count=0,
    )
    result.validate()
    return result


def _blocked_review(
    _config: object,
    manifest: object,
    request: object,
    *,
    deterministic_risk_gate_passed: bool,
    observed_wall_ns: int,
) -> Round74AIRuntimeOutcome:
    result = Round74AIRuntimeOutcome(
        status="blocked_capability",
        request_sha256=request.request_sha256,
        manifest_sha256=manifest.manifest_sha256,
        deterministic_risk_gate_passed=deterministic_risk_gate_passed,
        observed_wall_ns=observed_wall_ns,
        proposed_risk_size_bps=request.proposed_risk_size_bps,
        approved_risk_size_bps=0,
        capability={"ok": False},
        resolved_model_digest=None,
        resolved_model_metadata_sha256=None,
        worker_result=None,
        elapsed_ns=5_000_000_000_000,
        failure_class="AICapabilityGate",
        message="blocked by test capability gate",
    )
    result.validate()
    return result


def _provider_partition() -> Round74EventRunPartition:
    second_ns = 1_000_000_000
    first_wall_ns = 1_700_000_000_000_000_000
    role_run_ids = (
        ("training", f"{1:032x}"),
        ("tuning", f"{2:032x}"),
        *(("test", run_id) for run_id in TEST_RUNS),
    )
    entries = tuple(
        Round74EventRunPartitionEntry(
            run_id=run_id,
            role=role,
            capture_report_sha256=f"{index + 1:064x}",
            capture_start_wall_ns=first_wall_ns + index * 2_000 * second_ns,
            capture_end_wall_ns=(
                first_wall_ns + index * 2_000 * second_ns + 1_000 * second_ns
            ),
            eligible_anchor_start_wall_ns=(
                first_wall_ns + index * 2_000 * second_ns + 310_500_000_000
            ),
            eligible_anchor_end_wall_ns=(
                first_wall_ns + index * 2_000 * second_ns + 600 * second_ns
            ),
        )
        for index, (role, run_id) in enumerate(role_run_ids)
    )
    result = Round74EventRunPartition(
        entries=entries,
        cohort_plan_sha256="d" * 64,
    )
    result.validate()
    return result


def _replay_instruction(
    *,
    row_index: int,
    run_id: str,
    manifest: str,
    partition_sha256: str,
    action_selection_sha256: str,
) -> Round74AIExecutionReplayInstruction:
    result = Round74AIExecutionReplayInstruction(
        row_index=row_index,
        run_id=run_id,
        symbol="BTCUSDT",
        anchor_index=row_index,
        decision_monotonic_ns=row_index,
        decision_wall_ns=1_800_000_000_000_000_000 + row_index,
        endpoint_frame_index=row_index,
        endpoint_message_index=0,
        sample_sha256=f"{1_000 + row_index:064x}",
        feature_window_sha256=f"{2_000 + row_index:064x}",
        feature_row_sha256=f"{3_000 + row_index:064x}",
        side=1,
        horizon_seconds=30,
        source_review_sha256=f"{4_000 + row_index:064x}",
        model_manifest_sha256=manifest,
        runtime_status="blocked_capability",
        effective_review_latency_ns=0,
        action_latency_eligible=False,
        requested_size_multiplier_bps=0,
        pre_replay_status="runtime_veto",
        partition_sha256=partition_sha256,
        action_selection_sha256=action_selection_sha256,
    )
    result.validate()
    return result


def test_reservation_consumes_test_access_before_evaluation(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    selection = _selection()
    qualification = _ai_pretest_qualification(selection)
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=selection,
        ai_pretest_qualification=qualification,
    )

    assert claim.status == "reserved"
    assert claim.rows == 24
    assert claim.test_run_ids == TEST_RUNS
    assert claim.ai_manifest_sha256 == ("a" * 64, "b" * 64)
    assert ledger.claim_matches(claim, required_status="reserved")
    assert len(claim.claim_sha256) == 64
    assert Round74SealedEvaluationClaim.from_mapping(claim.as_dict()) == claim
    assert not Path(f"{ledger.path}-wal").exists()
    with pytest.raises(Round74SealedReuseError, match="already reserved"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=selection,
            ai_pretest_qualification=qualification,
        )


def test_segmented_reservation_and_bootstrap_keep_every_test_segment(
    tmp_path: Path,
) -> None:
    test_runs = tuple(f"{1_000 + index:032x}" for index in range(90))
    policy_runs = tuple(f"{2_000 + index:032x}" for index in range(23))
    batch = _test_batch(runs=test_runs)
    selection = _selection(
        policy_runs=policy_runs,
        optimization_population="eligible_target",
    )
    test_population = Round74SegmentedTestPopulation(
        parent_partition_sha256=batch.partition_sha256,
        cohort_plan_sha256="e" * 64,
        test_run_ids=test_runs,
        test_slot_ordinals=tuple(range(617, 707)),
        test_eligible_anchor_ns=(900_000_000_000,) * len(test_runs),
    )
    identity = build_round74_segmented_sealed_dataset_identity(
        (batch,),
        test_population=test_population,
    )
    ledger = Round74SealedEvaluationLedger(tmp_path / "segmented.sqlite3")

    claim = ledger.reserve_identity(
        test_identity=identity,
        action_selection=selection,
        ai_pretest_qualification=_ai_pretest_qualification(selection),
    )
    bootstrap = sealed_subject._run_bootstrap(
        test_runs,
        np.ones(len(test_runs), dtype=np.float64),
        expected_run_ids=test_runs,
        seed=sealed_subject.ROUND74_SEALED_BOOTSTRAP_SEED,
        optimization_population="eligible_target",
    )

    assert claim.optimization_population == "eligible_target"
    assert claim.test_population_sha256 == test_population.population_sha256
    assert claim.test_run_ids == test_runs
    assert Round74SealedEvaluationClaim.from_mapping(claim.as_dict()) == claim
    assert bootstrap.blocks == 90
    assert bootstrap.optimization_population == "eligible_target"
    assert bootstrap.point_mean_run_net_bps == pytest.approx(1.0)
    assert bootstrap.mean_block_length_runs == 10
    assert bootstrap.resampling_method == "circular_stationary_bootstrap"
    assert bootstrap.as_dict()["chronological_dependence_preserved"] is True
    assert bootstrap.as_dict()["iid_capture_run_resampling_permitted"] is False
    with pytest.raises(ValueError, match="dataset identity differs"):
        build_round74_sealed_dataset_identity(
            (batch,),
            optimization_population="capture_run",
        )
    with pytest.raises(ValueError, match="test population differs"):
        build_round74_segmented_sealed_dataset_identity(
            (_test_batch(runs=test_runs[:-1]),),
            test_population=test_population,
        )


def test_v1_ledger_migrates_only_to_legacy_capture_population(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE round74_governance_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO round74_governance_metadata (key, value)
            VALUES ('schema_version', 'round-074-sealed-ledger-v1');
            INSERT INTO round74_governance_metadata (key, value)
            VALUES ('ledger_id', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
            CREATE TABLE round74_sealed_claims (
                reservation_id TEXT PRIMARY KEY,
                ledger_id TEXT NOT NULL,
                test_access_sha256 TEXT NOT NULL UNIQUE,
                dataset_sha256 TEXT NOT NULL UNIQUE,
                partition_sha256 TEXT NOT NULL,
                scaler_sha256 TEXT NOT NULL,
                pretest_policy_sha256 TEXT NOT NULL,
                probability_calibration_sha256 TEXT NOT NULL,
                action_selection_sha256 TEXT NOT NULL,
                ai_manifest_sha256_json TEXT NOT NULL,
                profile TEXT NOT NULL,
                test_run_ids_json TEXT NOT NULL,
                batch_sha256_json TEXT NOT NULL,
                rows INTEGER NOT NULL,
                first_wall_ns INTEGER NOT NULL,
                last_wall_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_outcome TEXT NOT NULL DEFAULT '',
                result_sha256 TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                reserved_at_ns INTEGER NOT NULL,
                completed_at_ns INTEGER
            );
            """
        )
        connection.execute(
            """
            INSERT INTO round74_sealed_claims (
                reservation_id, ledger_id, test_access_sha256,
                dataset_sha256, partition_sha256, scaler_sha256,
                pretest_policy_sha256, probability_calibration_sha256,
                action_selection_sha256, ai_manifest_sha256_json,
                profile, test_run_ids_json, batch_sha256_json, rows,
                first_wall_ns, last_wall_ns, status, reserved_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1" * 64,
                "a" * 64,
                "2" * 64,
                "3" * 64,
                "4" * 64,
                "5" * 64,
                "6" * 64,
                "7" * 64,
                "8" * 64,
                json.dumps(["9" * 64]),
                "aggressive",
                json.dumps(list(TEST_RUNS)),
                json.dumps(["b" * 64]),
                len(TEST_RUNS),
                1,
                2,
                "reserved",
                1,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    ledger = Round74SealedEvaluationLedger(path)
    migrated = ledger._connect()
    try:
        columns = {
            str(row[1])
            for row in migrated.execute(
                "PRAGMA table_info(round74_sealed_claims)"
            ).fetchall()
        }
        schema = migrated.execute(
            """
            SELECT value FROM round74_governance_metadata
            WHERE key = 'schema_version'
            """
        ).fetchone()
        migrated_population = migrated.execute(
            """
            SELECT optimization_population, test_population_sha256
            FROM round74_sealed_claims
            WHERE reservation_id = ?
            """,
            ["1" * 64],
        ).fetchone()
    finally:
        migrated.close()

    assert "optimization_population" in columns
    assert "test_population_sha256" in columns
    assert "ai_pretest_qualification_sha256" in columns
    assert "ai_pretest_qualification_required" in columns
    assert schema is not None
    assert schema[0] == "round-074-sealed-ledger-v3"
    assert migrated_population is not None
    assert migrated_population[0] == "capture_run"
    assert len(migrated_population[1]) == 64
    migrated_claim = ledger.claim("1" * 64)
    assert migrated_claim.ai_pretest_qualification_sha256 == ""
    assert not migrated_claim.ai_pretest_qualification_required


def test_completed_or_failed_reservation_cannot_be_reset_or_finalized_twice(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    selection = _selection()
    qualification = _ai_pretest_qualification(selection)
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=selection,
        ai_pretest_qualification=qualification,
    )
    completed = ledger.finalize(
        claim.reservation_id,
        result_outcome="candidate_failed_predeclared_gates",
        result_sha256="c" * 64,
    )

    assert completed.status == "complete"
    assert ledger.claim_matches(completed, required_status="complete")
    with pytest.raises(Round74SealedLedgerError, match="already finalized"):
        ledger.finalize(
            claim.reservation_id,
            result_outcome="candidate_passed_predeclared_gates",
            result_sha256="d" * 64,
        )
    with pytest.raises(Round74SealedReuseError):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=selection,
            ai_pretest_qualification=qualification,
        )


def test_evaluation_error_is_durable_and_still_consumes_test(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    selection = _selection()
    qualification = _ai_pretest_qualification(selection)
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=selection,
        ai_pretest_qualification=qualification,
    )
    failed = ledger.finalize(
        claim.reservation_id,
        result_outcome="evaluation_error",
        result_sha256="e" * 64,
        error="worker interrupted after reservation",
    )

    assert failed.status == "failed"
    assert failed.error == "worker interrupted after reservation"
    assert ledger.claim_matches(failed, required_status="failed")
    with pytest.raises(Round74SealedReuseError):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=selection,
            ai_pretest_qualification=qualification,
        )


def test_sealed_evaluator_finalizes_failure_after_reservation(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    batch = _test_batch()
    identity = build_round74_sealed_dataset_identity((batch,))
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    qualification = _ai_pretest_qualification(selection)
    loader_status: list[str] = []

    def load_after_reservation(
        *,
        claim: Round74SealedEvaluationClaim,
    ) -> tuple[Round74EventTrainingBatch, ...]:
        assert ledger.claim_matches(claim, required_status="reserved")
        loader_status.append(claim.status)
        return (batch,)

    with pytest.raises(ValueError, match="pretest policy could not be read"):
        evaluate_round74_sealed_once(
            identity,
            test_batch_loader=load_after_reservation,
            action_selection=selection,
            probability_calibration=calibration,
            pretest_policy_path=tmp_path / "missing-policy.json",
            ai_pretest_qualification=qualification,
            ai_review_provider=lambda **_kwargs: {
                "a" * 64: (),
                "b" * 64: (),
            },
            ai_execution_replay_provider=lambda **_kwargs: {
                "a" * 64: (),
                "b" * 64: (),
            },
            ledger=ledger,
            compute_backend="cpu",
        )

    assert loader_status == ["reserved"]
    with pytest.raises(Round74SealedReuseError, match="status=failed"):
        ledger.reserve(
            test_batches=(batch,),
            action_selection=selection,
            ai_pretest_qualification=qualification,
        )


def test_sealed_evaluator_scores_bound_model_and_finalizes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    batch = _test_batch()
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    manifests = ("c" * 64, "e" * 64)
    policy = {
        "policy_sha256": selection.pretest_policy_sha256,
        "model_artifact": {"sha256": "d" * 64},
        "development_data": {
            "partition_sha256": batch.partition_sha256,
            "scaler_sha256": batch.scaler_sha256,
            "window_representation": batch.window_representation,
        },
    }
    monkeypatch.setattr(
        sealed_subject,
        "load_round74_pretest_policy",
        lambda _path: (_Model(), policy),
    )

    reviews_by_manifest = {
        manifest: list(
            _reviews(
                batch,
                calibration,
                selection,
                manifest=manifest,
            )
        )
        for manifest in manifests
    }
    reviews = reviews_by_manifest[manifests[0]]
    reviews[0] = replace(
        reviews[0],
        runtime_elapsed_ns=ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS + 1,
        effective_review_latency_ns=ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS + 1,
        action_latency_eligible=False,
    )
    reviews[0].validate()
    execution_replays_by_manifest = {
        manifest: _execution_replays(
            tuple(model_reviews),
            partition_sha256=batch.partition_sha256,
        )
        for manifest, model_reviews in reviews_by_manifest.items()
    }
    provider_order: list[str] = []

    def load_test_batches(
        *,
        claim: Round74SealedEvaluationClaim,
    ) -> tuple[Round74EventTrainingBatch, ...]:
        assert ledger.claim_matches(claim, required_status="reserved")
        provider_order.append("load")
        return (batch,)

    def review_provider(
        *,
        claim: Round74SealedEvaluationClaim,
        **_kwargs: object,
    ) -> Mapping[str, tuple[Round74AIPairedReviewEvidence, ...]]:
        assert ledger.claim_matches(claim, required_status="reserved")
        provider_order.append("review")
        return {
            manifest: tuple(model_reviews)
            for manifest, model_reviews in reviews_by_manifest.items()
        }

    def replay_provider(
        *,
        claim: Round74SealedEvaluationClaim,
        instructions_by_manifest: Mapping[
            str,
            tuple[Round74AIExecutionReplayInstruction, ...],
        ],
    ) -> Mapping[str, tuple[Round74AIExecutionReplayEvidence, ...]]:
        assert ledger.claim_matches(claim, required_status="reserved")
        assert tuple(instructions_by_manifest) == manifests
        assert all(
            instruction.action_selection_sha256 == claim.action_selection_sha256
            for instructions in instructions_by_manifest.values()
            for instruction in instructions
        )
        provider_order.append("replay")
        return execution_replays_by_manifest

    outcome = evaluate_round74_sealed_once(
        build_round74_sealed_dataset_identity((batch,)),
        test_batch_loader=load_test_batches,
        action_selection=selection,
        probability_calibration=calibration,
        pretest_policy_path=tmp_path / "policy.json",
        ai_pretest_qualification=_ai_pretest_qualification(
            selection,
            manifests=manifests,
        ),
        ai_review_provider=review_provider,
        ai_execution_replay_provider=replay_provider,
        ledger=ledger,
        compute_backend="cpu",
    )

    assert provider_order == ["load", "review", "replay"]
    assert outcome.finalized_claim.status == "complete"
    assert outcome.report.result_outcome == "candidate_failed_predeclared_gates"
    assert outcome.report.qualified_configuration == ()
    assert outcome.report.baseline_metrics.executed_trades == 24
    assert outcome.report.baseline_metrics.financial_gate_passed
    assert not outcome.report.predictive_gate.gate_passed
    assert (
        "positive_payoff:non_single_class_evidence_missing"
        in outcome.report.predictive_gate.gate_reasons
    )
    assert len(outcome.report.ai_overlays) == 2
    with pytest.raises(ValueError, match="sealed evaluation report differs"):
        replace(
            outcome.report,
            ai_overlays=(
                outcome.report.ai_overlays[0],
                outcome.report.ai_overlays[0],
            ),
        ).validate()
    assert outcome.report.predictive_diagnostics.eligible_action_targets == (
        24
        * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
        * len(ROUND74_EVENT_PAYOFF_SIDES)
    )
    overlay = outcome.report.ai_overlays[0]
    assert not overlay.uplift_gate_passed
    assert overlay.runtime_success_rate == 1.0
    assert overlay.action_latency_eligible_reviews == 23
    assert overlay.retained_trades == 23
    assert overlay.exact_replay_required_reviews == 23
    assert overlay.exact_replay_completed_reviews == 23
    assert len(overlay.paired_runs) == 24
    assert tuple(value.run_id for value in overlay.paired_runs) == TEST_RUNS
    assert sum(value.paired_observations for value in overlay.paired_runs) == 24
    assert tuple(
        (value.symbol, value.horizon_seconds)
        for value in overlay.paired_symbol_horizons
    ) == tuple((symbol, 30) for symbol in ROUND74_EVENT_SYMBOLS)
    assert (
        sum(value.paired_observations for value in overlay.paired_symbol_horizons) == 24
    )
    assert (
        sum(value.paired_observations for value in overlay.paired_run_symbol_horizons)
        == 24
    )
    assert overlay.strategy_metrics.total_net_bps == pytest.approx(23.0 / 3.0)
    assert (
        "positive_paired_delta_familywise_confidence_lower_bound_not_met"
        in overlay.gate_reasons
    )
    assert overlay.strategy_metrics.selected_action_target_ineligible == 0
    assert outcome.report.inference_backend_kind == "cpu"
    assert outcome.report.profitability_claim is False
    assert ledger.claim_matches(outcome.finalized_claim, required_status="complete")
    replay_selection = replace(
        selection,
        evaluations=tuple(
            replace(value, trace=outcome.report.baseline_trace)
            for value in selection.evaluations
        ),
    )
    replay_selection.validate()
    instructions = build_round74_ai_execution_replay_instructions(
        replay_selection,
        contexts=(build_round74_action_inference_context(batch),),
        reviews=tuple(reviews),
    )
    assert instructions[0].pre_replay_status == "historical_review_expired"
    assert instructions[0].requested_size_multiplier_bps == 10_000
    assert not instructions[0].action_latency_eligible


def test_predictive_gate_requires_familywise_skill_for_every_forecast_task() -> None:
    batch = _test_batch(
        runs=tuple(run_id for run_id in TEST_RUNS for _symbol in ROUND74_EVENT_SYMBOLS)
    )
    rows = batch.rows
    horizon_count = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    side_count = len(ROUND74_EVENT_PAYOFF_SIDES)
    quantile_count = len(ROUND74_EVENT_PAYOFF_QUANTILES)
    occurrence = np.arange(rows, dtype=np.int64) // len(ROUND74_EVENT_SYMBOLS)
    positive_truth = occurrence % 2 == 1
    regime_truth = (occurrence // 2) % 2 == 1
    adverse_truth = np.logical_xor(positive_truth, regime_truth)
    payoff_sign = np.where(positive_truth, 1.0, -1.0).astype(np.float32)
    action_shape = (rows, horizon_count, side_count)
    regime_shape = (rows, horizon_count)
    batch = replace(
        batch,
        net_payoff_bps=_readonly(
            np.broadcast_to(payoff_sign[:, None, None], action_shape).copy()
        ),
        adverse_selection=_readonly(
            np.broadcast_to(
                adverse_truth.astype(np.float32)[:, None, None],
                action_shape,
            ).copy()
        ),
        regime_unpredictability=_readonly(
            np.broadcast_to(
                regime_truth.astype(np.float32)[:, None],
                regime_shape,
            ).copy()
        ),
    )
    batch.validate()
    payoff_offsets = np.asarray(
        (-0.50, -0.25, 0.0, 0.25, 0.50),
        dtype=np.float32,
    )
    payoff = np.broadcast_to(
        payoff_sign[:, None, None, None] + payoff_offsets,
        (rows, horizon_count, side_count, quantile_count),
    ).copy()
    mae_quantiles = np.asarray(
        (0.20, 0.40, 0.60, 0.80, 1.00),
        dtype=np.float32,
    )
    mae = np.broadcast_to(
        mae_quantiles,
        (rows, horizon_count, side_count, quantile_count),
    ).copy()
    positive_logits = np.broadcast_to(
        np.where(positive_truth, 4.0, -4.0)[:, None, None],
        action_shape,
    ).copy()
    adverse_logits = np.broadcast_to(
        np.where(adverse_truth, 4.0, -4.0)[:, None, None],
        action_shape,
    ).copy()
    regime_logits = np.broadcast_to(
        np.where(regime_truth, 4.0, -4.0)[:, None],
        regime_shape,
    ).copy()
    output = Round74EventModelOutput(
        payoff_quantiles_bps=torch.from_numpy(payoff),
        maximum_adverse_excursion_quantiles_bps=torch.from_numpy(mae),
        positive_payoff_logits=torch.from_numpy(positive_logits),
        adverse_selection_logits=torch.from_numpy(adverse_logits),
        regime_unpredictability_logits=torch.from_numpy(regime_logits),
    )
    output.validate(rows)
    accumulator = sealed_subject._PredictiveAccumulator()
    accumulator.update(batch, output, _calibration())
    diagnostics, gate = accumulator.result(expected_run_ids=TEST_RUNS)

    assert diagnostics.eligible_action_targets == rows * horizon_count * side_count
    assert gate.gate_passed
    assert gate.gate_reasons == ()
    assert tuple(value.task for value in gate.task_skills) == (
        "positive_payoff",
        "adverse_selection",
        "regime_unpredictability",
        "net_payoff_quantiles",
        "maximum_adverse_excursion_quantiles",
    )
    assert all(value.scope_symbol == "all" for value in gate.task_skills)
    assert tuple(
        (value.scope_symbol, value.task) for value in gate.symbol_task_skills
    ) == tuple(
        (symbol, task)
        for symbol in ROUND74_EVENT_SYMBOLS
        for task in sealed_subject.ROUND74_SEALED_PREDICTIVE_TASKS
    )
    assert all(value.gate_passed for value in gate.task_skills)
    assert all(value.gate_passed for value in gate.symbol_task_skills)
    assert all(
        value.covered_capture_runs == len(TEST_RUNS) for value in gate.task_skills
    )
    assert all(
        value.covered_capture_runs == len(TEST_RUNS)
        for value in gate.symbol_task_skills
    )
    binary_skills = gate.task_skills[:3]
    quantile_skills = gate.task_skills[3:]
    assert all(value.brier_skill_score > 0.0 for value in binary_skills)
    assert all(
        value.familywise_lower_mean_run_brier_improvement > 0.0
        for value in binary_skills
    )
    assert all(value.pinball_skill_score > 0.0 for value in quantile_skills)
    assert all(
        value.familywise_lower_mean_run_pinball_improvement_bps > 0.0
        for value in quantile_skills
    )

    sol_rows = torch.tensor(
        tuple(symbol == "SOLUSDT" for symbol in batch.symbol),
        dtype=torch.bool,
    )
    sol_unskilled_payoff = output.payoff_quantiles_bps.clone()
    sol_unskilled_payoff[sol_rows] = 0.0
    masked = sealed_subject._PredictiveAccumulator()
    masked.update(
        batch,
        replace(output, payoff_quantiles_bps=sol_unskilled_payoff),
        _calibration(),
    )
    _masked_diagnostics, masked_gate = masked.result(expected_run_ids=TEST_RUNS)
    pooled_payoff = next(
        value
        for value in masked_gate.task_skills
        if value.task == "net_payoff_quantiles"
    )
    sol_payoff = next(
        value
        for value in masked_gate.symbol_task_skills
        if value.scope_symbol == "SOLUSDT" and value.task == "net_payoff_quantiles"
    )
    assert pooled_payoff.gate_passed
    assert not sol_payoff.gate_passed
    assert not masked_gate.gate_passed
    assert (
        "SOLUSDT:net_payoff_quantiles:positive_pinball_skill_not_met"
        in masked_gate.gate_reasons
    )

    unskilled_output = replace(
        output,
        positive_payoff_logits=torch.zeros_like(output.positive_payoff_logits),
        adverse_selection_logits=torch.zeros_like(output.adverse_selection_logits),
        regime_unpredictability_logits=torch.zeros_like(
            output.regime_unpredictability_logits
        ),
    )
    unskilled = sealed_subject._PredictiveAccumulator()
    unskilled.update(batch, unskilled_output, _calibration())
    _diagnostics, unskilled_gate = unskilled.result(expected_run_ids=TEST_RUNS)
    assert not unskilled_gate.gate_passed
    assert "positive_payoff:positive_brier_skill_not_met" in unskilled_gate.gate_reasons


def test_sealed_ai_overlay_rejects_aggregate_gain_that_harms_subgroups() -> None:
    batch = _test_batch()
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    candidates = derive_round74_action_candidates(
        _model_output(batch.rows),
        build_round74_action_inference_context(batch),
        calibration,
        pretest_policy_sha256=selection.pretest_policy_sha256,
        profile=selection.profile,
    )
    trace = sealed_subject._simulate_round74_action_trace_batches(
        (batch,),
        (candidates,),
        threshold_score=float(selection.selected_threshold_score or 0.0),
        expected_run_ids=TEST_RUNS,
        required_role="test",
        expected_run_count=24,
    )
    manifest = "c" * 64
    reviews = _reviews(
        batch,
        calibration,
        selection,
        manifest=manifest,
    )
    executions = []
    for execution in _execution_replays(
        reviews,
        partition_sha256=batch.partition_sha256,
    ):
        position_net_bps = 0.3 if execution.symbol == "SOLUSDT" else 3.0
        updated = replace(
            execution,
            position_net_payoff_bps=position_net_bps,
            capital_scaled_net_payoff_bps=position_net_bps,
        )
        updated.validate()
        executions.append(updated)

    overlay = sealed_subject._ai_overlay(
        trace,
        reviews,
        tuple(executions),
        manifest=manifest,
        expected_partition_sha256=batch.partition_sha256,
        profile=selection.profile,
        seed=sealed_subject.ROUND74_SEALED_BOOTSTRAP_SEED,
    )

    assert overlay.strategy_metrics.financial_gate_passed
    assert overlay.strategy_metrics.total_net_bps > trace.metrics.total_net_bps
    assert (
        overlay.paired_delta_bootstrap.two_ai_model_bonferroni_lower_mean_run_net_bps
        > 0.0
    )
    assert not overlay.uplift_gate_passed
    assert "paired_run_noninferiority_not_met" in overlay.gate_reasons
    assert "paired_symbol_horizon_noninferiority_not_met" in overlay.gate_reasons
    assert "paired_run_symbol_horizon_noninferiority_not_met" in overlay.gate_reasons
    assert sum(value.delta_net_bps < 0.0 for value in overlay.paired_runs) == 8
    sol = next(
        value for value in overlay.paired_symbol_horizons if value.symbol == "SOLUSDT"
    )
    assert sol.delta_net_bps < 0.0
    assert all(
        value.delta_net_bps > 0.0
        for value in overlay.paired_symbol_horizons
        if value.symbol != "SOLUSDT"
    )
    corrupted = replace(
        overlay.paired_runs[0],
        delta_net_bps=overlay.paired_runs[0].delta_net_bps + 1.0,
    )
    with pytest.raises(ValueError, match="paired run delta differs"):
        replace(
            overlay,
            paired_runs=(corrupted, *overlay.paired_runs[1:]),
        ).validate()


def test_sealed_ai_overlay_rejects_harm_hidden_by_both_aggregate_panels() -> None:
    expanded_runs = tuple(
        run_id for run_id in TEST_RUNS for _symbol in ROUND74_EVENT_SYMBOLS
    )
    batch = _test_batch(runs=expanded_runs)
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    candidates = derive_round74_action_candidates(
        _model_output(batch.rows),
        build_round74_action_inference_context(batch),
        calibration,
        pretest_policy_sha256=selection.pretest_policy_sha256,
        profile=selection.profile,
    )
    trace = sealed_subject._simulate_round74_action_trace_batches(
        (batch,),
        (candidates,),
        threshold_score=float(selection.selected_threshold_score or 0.0),
        expected_run_ids=TEST_RUNS,
        required_role="test",
        expected_run_count=24,
    )
    manifest = "c" * 64
    reviews = _reviews(
        batch,
        calibration,
        selection,
        manifest=manifest,
    )
    executions = []
    for index, execution in enumerate(
        _execution_replays(
            reviews,
            partition_sha256=batch.partition_sha256,
        )
    ):
        position_net_bps = 0.3 if index == 0 else 3.0
        updated = replace(
            execution,
            source_capture_report_sha256="d" * 64,
            position_net_payoff_bps=position_net_bps,
            capital_scaled_net_payoff_bps=position_net_bps,
        )
        updated.validate()
        executions.append(updated)

    overlay = sealed_subject._ai_overlay(
        trace,
        reviews,
        tuple(executions),
        manifest=manifest,
        expected_partition_sha256=batch.partition_sha256,
        profile=selection.profile,
        seed=sealed_subject.ROUND74_SEALED_BOOTSTRAP_SEED,
    )

    assert overlay.strategy_metrics.financial_gate_passed
    assert overlay.strategy_metrics.total_net_bps > trace.metrics.total_net_bps
    assert all(value.delta_net_bps >= 0.0 for value in overlay.paired_runs)
    assert all(value.delta_net_bps >= 0.0 for value in overlay.paired_symbol_horizons)
    harmed = [
        value
        for value in overlay.paired_run_symbol_horizons
        if value.delta_net_bps < 0.0
    ]
    assert len(harmed) == 1
    assert (harmed[0].run_id, harmed[0].symbol, harmed[0].horizon_seconds) == (
        TEST_RUNS[0],
        "BTCUSDT",
        30,
    )
    assert not overlay.uplift_gate_passed
    assert "paired_run_noninferiority_not_met" not in overlay.gate_reasons
    assert "paired_symbol_horizon_noninferiority_not_met" not in overlay.gate_reasons
    assert "paired_run_symbol_horizon_noninferiority_not_met" in overlay.gate_reasons
    corrupted = replace(
        overlay.paired_run_symbol_horizons[0],
        paired_observations=2,
    )
    with pytest.raises(ValueError, match="sealed AI overlay differs"):
        replace(
            overlay,
            paired_run_symbol_horizons=(
                corrupted,
                *overlay.paired_run_symbol_horizons[1:],
            ),
        ).validate()
    with pytest.raises(ValueError, match="sealed AI baseline pairing differs"):
        replace(
            overlay,
            uplift_gate_passed=True,
            gate_reasons=(),
        ).validate_against_baseline(
            trace,
            profile=selection.profile,
        )


def test_sealed_evaluator_rejects_incomplete_ai_family_before_reservation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.sqlite3"
    ledger = Round74SealedEvaluationLedger(path)
    batch = _test_batch()
    selection = _selection()
    qualification = _ai_pretest_qualification(selection)
    incomplete = replace(
        qualification,
        development_reports=(qualification.development_reports[0],),
    )

    with pytest.raises(ValueError, match="pretest qualification differs"):
        evaluate_round74_sealed_once(
            build_round74_sealed_dataset_identity((batch,)),
            test_batch_loader=lambda **_kwargs: (batch,),
            action_selection=selection,
            probability_calibration=_calibration(),
            pretest_policy_path=tmp_path / "policy.json",
            ai_pretest_qualification=incomplete,
            ai_review_provider=lambda **_kwargs: {},
            ai_execution_replay_provider=lambda **_kwargs: {},
            ledger=ledger,
            compute_backend="cpu",
        )

    assert not path.exists()


def test_sealed_financial_gate_rejects_future_censored_selected_action() -> None:
    batch = _test_batch()
    horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(30)
    eligibility = np.array(batch.action_eligibility, copy=True)
    entry = np.array(batch.actual_entry_monotonic_ns, copy=True)
    exit_value = np.array(batch.actual_exit_monotonic_ns, copy=True)
    payoff = np.array(batch.net_payoff_bps, copy=True)
    mae = np.array(batch.maximum_adverse_excursion_bps, copy=True)
    adverse = np.array(batch.adverse_selection, copy=True)
    eligibility[0, horizon_index, 0] = 0.0
    entry[0, horizon_index, 0] = -1
    exit_value[0, horizon_index, 0] = -1
    payoff[0, horizon_index, 0] = 0.0
    mae[0, horizon_index, 0] = 0.0
    adverse[0, horizon_index, 0] = 0.0
    batch = replace(
        batch,
        action_eligibility=_readonly(eligibility),
        actual_entry_monotonic_ns=_readonly(entry),
        actual_exit_monotonic_ns=_readonly(exit_value),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(mae),
        adverse_selection=_readonly(adverse),
    )
    batch.validate()
    selection = _selection()
    calibration = _calibration()
    candidates = derive_round74_action_candidates(
        _model_output(batch.rows),
        build_round74_action_inference_context(batch),
        calibration,
        pretest_policy_sha256=selection.pretest_policy_sha256,
        profile=selection.profile,
    )
    trace = sealed_subject._simulate_round74_action_trace_batches(
        (batch,),
        (candidates,),
        threshold_score=float(selection.selected_threshold_score or 0.0),
        expected_run_ids=TEST_RUNS,
        required_role="test",
        expected_run_count=24,
    )
    metrics = sealed_subject._baseline_strategy_metrics(
        trace,
        profile=selection.profile,
        seed=1,
    )

    assert trace.skipped_target_ineligible == 1
    assert metrics.selected_action_target_ineligible == 1
    assert not metrics.financial_gate_passed
    assert "selected_action_target_coverage_incomplete" in metrics.gate_reasons


def test_sealed_bootstrap_controls_three_configuration_familywise_error() -> None:
    values = np.asarray((-13.0, 0.0, *(2.0 for _ in range(22))))
    evidence = sealed_subject._run_bootstrap(
        TEST_RUNS,
        values,
        expected_run_ids=TEST_RUNS,
        seed=sealed_subject.ROUND74_SEALED_BOOTSTRAP_SEED,
    )

    assert evidence.one_sided_95_lower_mean_run_net_bps > 0.0
    assert evidence.two_ai_model_bonferroni_lower_mean_run_net_bps <= 0.0
    assert (
        evidence.three_configuration_bonferroni_lower_mean_run_net_bps
        <= evidence.two_ai_model_bonferroni_lower_mean_run_net_bps
    )
    assert evidence.as_dict()["qualification_configuration_count"] == 3
    assert evidence.as_dict()["paired_ai_model_count"] == 2


def test_sealed_stationary_bootstrap_preserves_chronological_dependence() -> None:
    clustered = np.asarray((4.0,) * 12 + (-2.0,) * 12)
    interleaved = np.asarray((4.0, -2.0) * 12)
    clustered_evidence = sealed_subject._run_bootstrap(
        TEST_RUNS,
        clustered,
        expected_run_ids=TEST_RUNS,
        seed=sealed_subject.ROUND74_SEALED_BOOTSTRAP_SEED,
    )
    interleaved_evidence = sealed_subject._run_bootstrap(
        TEST_RUNS,
        interleaved,
        expected_run_ids=TEST_RUNS,
        seed=sealed_subject.ROUND74_SEALED_BOOTSTRAP_SEED,
    )

    assert clustered_evidence.point_mean_run_net_bps == pytest.approx(1.0)
    assert interleaved_evidence.point_mean_run_net_bps == pytest.approx(1.0)
    assert clustered_evidence.mean_block_length_runs == 5
    assert clustered_evidence.restart_probability == pytest.approx(0.2)
    assert (
        clustered_evidence.one_sided_95_lower_mean_run_net_bps
        < interleaved_evidence.one_sided_95_lower_mean_run_net_bps
    )
    assert clustered_evidence.as_dict()["circular_wraparound"] is True


def test_target_free_two_model_review_panel_preserves_blocked_observations() -> None:
    batch = _test_batch()
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    inference = _candidate_inference(batch, calibration, selection)
    progress: list[Mapping[str, object]] = []
    prepared_models: list[str] = []
    finalized_models: list[str] = []
    model_invocations: list[str] = []

    def counted_blocked_review(
        config: object,
        manifest: object,
        request: object,
        *,
        deterministic_risk_gate_passed: bool,
        observed_wall_ns: int,
    ) -> Round74AIRuntimeOutcome:
        model_invocations.append(request.sample_sha256)
        return _blocked_review(
            config,
            manifest,
            request,
            deterministic_risk_gate_passed=deterministic_risk_gate_passed,
            observed_wall_ns=observed_wall_ns,
        )

    panel = prepare_round74_target_free_ai_reviews(
        inference,
        action_selection=selection,
        probability_calibration=calibration,
        review_runner=counted_blocked_review,
        model_batch_preparer=lambda binding: prepared_models.append(binding.model_name),
        model_batch_finalizer=lambda binding: finalized_models.append(
            binding.model_name
        ),
        progress_callback=progress.append,
        wall_time_ns=lambda: 1_900_000_000_000_000_000,
    )

    default_models = round74_default_ai_review_model_panel()
    assert len(default_models) == 2
    assert tuple(value.manifest.model_artifact_sha256 for value in default_models) == (
        "083c6422a2dd90d62ec33638ab84271edddd2cf1fa6a9841898ea18a35e27b87",
        "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
    )
    assert len(panel.rows) == 24
    assert panel.reviews[0][0].queue_delay_ns == 0
    assert panel.reviews[0][1].queue_delay_ns == 1_000_000_000_000
    assert (
        panel.reviews[0][1].queue_delay_ns
        <= panel.reviews[0][1].effective_review_latency_ns
        < panel.reviews[0][1].queue_delay_ns + 1_000_000_000
    )
    assert len(panel.reviews) == 2
    assert panel.model_batch_unload_enforced is True
    assert prepared_models == ["fino1:8b", "qwen3:8b"]
    assert finalized_models == ["fino1:8b", "qwen3:8b"]
    assert all(len(value) == 24 for value in panel.reviews)
    assert all(
        review.runtime_status in {"blocked_capability", "blocked_expired"}
        and review.action_latency_eligible is False
        and review.size_multiplier_bps == 0
        and review.decision is None
        for reviews in panel.reviews
        for review in reviews
    )
    assert any(
        review.queue_expired_before_inference
        for reviews in panel.reviews
        for review in reviews
    )
    assert 0 < len(model_invocations) < 48
    assert len(progress) == 48
    assert progress[-1]["completed_reviews"] == 48
    assert any(value["queue_expired_before_inference"] for value in progress)
    assert set(panel.reviews_by_manifest()) == {
        binding.manifest.manifest_sha256 for binding in default_models
    }
    assert panel.target_fields_accessed is False
    assert panel.trading_authority is False
    assert len(panel.panel_sha256) == 64


def test_target_free_review_panel_finalizes_after_preload_failure() -> None:
    batch = _test_batch()
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    inference = _candidate_inference(batch, calibration, selection)
    finalized_models: list[str] = []

    with pytest.raises(RuntimeError, match="preload failed"):
        prepare_round74_target_free_ai_reviews(
            inference,
            action_selection=selection,
            probability_calibration=calibration,
            review_runner=_blocked_review,
            model_batch_preparer=lambda _binding: (_ for _ in ()).throw(
                RuntimeError("preload failed")
            ),
            model_batch_finalizer=lambda binding: finalized_models.append(
                binding.model_name
            ),
            wall_time_ns=lambda: 1_900_000_000_000_000_000,
        )

    assert finalized_models == ["fino1:8b"]


def test_prepared_review_provider_binds_reserved_claim_and_model_panel(
    tmp_path: Path,
) -> None:
    batch = _test_batch()
    calibration = _calibration()
    selection = replace(
        _selection(),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    inference = _candidate_inference(batch, calibration, selection)
    bindings = round74_default_ai_review_model_panel()
    manifests = tuple(value.manifest.manifest_sha256 for value in bindings)
    qualification = _ai_pretest_qualification(selection, manifests=manifests)
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(batch,),
        action_selection=selection,
        ai_pretest_qualification=qualification,
    )
    progress: list[Mapping[str, object]] = []
    provider = Round74PreparedSealedAIReviewProvider(
        probability_calibration=calibration,
        model_bindings=bindings,
        ai_pretest_qualification=qualification,
        review_runner=_blocked_review,
        progress_callback=progress.append,
        wall_time_ns=lambda: 1_900_000_000_000_000_000,
    )

    reviews = provider(
        claim=claim,
        manifests=manifests,
        inference=inference,
        action_selection=selection,
    )

    assert tuple(reviews) == manifests
    assert all(len(value) == batch.rows for value in reviews.values())
    assert all(
        review.runtime_status in {"blocked_capability", "blocked_expired"}
        and review.action_latency_eligible is False
        and review.size_multiplier_bps == 0
        for model_reviews in reviews.values()
        for review in model_reviews
    )
    assert any(
        review.queue_expired_before_inference
        for model_reviews in reviews.values()
        for review in model_reviews
    )
    assert len(progress) == len(manifests) * batch.rows
    with pytest.raises(ValueError, match="review identity differs"):
        provider(
            claim=claim,
            manifests=tuple(reversed(manifests)),
            inference=inference,
            action_selection=selection,
        )


def test_store_replay_provider_reconciles_each_run_and_restores_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _provider_partition()
    batch = replace(_test_batch(), partition_sha256=partition.partition_sha256)
    batch.validate()
    selection = _selection()
    manifests = ("a" * 64, "b" * 64)
    manifest = manifests[0]
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(batch,),
        action_selection=selection,
        ai_pretest_qualification=_ai_pretest_qualification(
            selection,
            manifests=manifests,
        ),
    )
    store = ImpactAbsorptionStore(
        tmp_path / "capture.duckdb",
        read_only=True,
    )
    assembly = object.__new__(Round74SourceTargetAssembly)
    provider = Round74SealedStoreExecutionReplayProvider(
        store=store,
        partition=partition,
        assembly_by_run_id={run_id: assembly for run_id in claim.test_run_ids},
    )
    instructions_by_manifest = {
        selected_manifest: (
            _replay_instruction(
                row_index=0,
                run_id=claim.test_run_ids[0],
                manifest=selected_manifest,
                partition_sha256=claim.partition_sha256,
                action_selection_sha256=claim.action_selection_sha256,
            ),
            _replay_instruction(
                row_index=1,
                run_id=claim.test_run_ids[1],
                manifest=selected_manifest,
                partition_sha256=claim.partition_sha256,
                action_selection_sha256=claim.action_selection_sha256,
            ),
            _replay_instruction(
                row_index=2,
                run_id=claim.test_run_ids[0],
                manifest=selected_manifest,
                partition_sha256=claim.partition_sha256,
                action_selection_sha256=claim.action_selection_sha256,
            ),
        )
        for selected_manifest in manifests
    }
    replayed_runs: list[str] = []

    def replay_run(
        selected_store: object,
        *,
        partition: Round74EventRunPartition,
        run_id: str,
        assembly: Round74SourceTargetAssembly,
        instructions: tuple[Round74AIExecutionReplayInstruction, ...],
    ) -> tuple[Round74AIExecutionReplayEvidence, ...]:
        assert selected_store is store
        assert partition is provider.partition
        assert assembly is provider.assembly_by_run_id[run_id]
        replayed_runs.append(run_id)
        return tuple(
            replay_subject._non_replay_evidence(
                row,
                capture_report_sha256=partition.entry(run_id).capture_report_sha256,
                target_spec_sha256="f" * 64,
            )
            for row in instructions
        )

    monkeypatch.setattr(
        replay_subject,
        "replay_round74_ai_execution_store_run",
        replay_run,
    )
    replayed = provider(
        claim=claim,
        instructions_by_manifest=instructions_by_manifest,
    )

    assert replayed_runs == list(claim.test_run_ids[:2]) * len(manifests)
    assert tuple(value.row_index for value in replayed[manifest]) == (0, 1, 2)
    assert all(
        value.status == "runtime_veto"
        for model_replays in replayed.values()
        for value in model_replays
    )
    with pytest.raises(ValueError, match="provider identity differs"):
        provider(
            claim=claim,
            instructions_by_manifest={manifest: instructions_by_manifest[manifest]},
        )


def test_development_data_is_rejected_before_ledger_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.sqlite3"
    ledger = Round74SealedEvaluationLedger(path)
    selection = _selection()

    with pytest.raises(ValueError, match="rejects development data"):
        ledger.reserve(
            test_batches=(_test_batch(role="tuning"),),
            action_selection=selection,
            ai_pretest_qualification=_ai_pretest_qualification(selection),
        )

    assert not path.exists()


def test_ai_qualification_cannot_reuse_sealed_or_other_tuning_population(
    tmp_path: Path,
) -> None:
    policy_runs = tuple(f"{30_000 + index:032x}" for index in range(24))
    selection = _selection(policy_runs=policy_runs)
    qualification = _ai_pretest_qualification(selection)
    population = qualification.development_reports[0].qualification_population
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")

    with pytest.raises(ValueError, match="reused test runs"):
        ledger.reserve(
            test_batches=(_test_batch(runs=population.run_ids),),
            action_selection=selection,
            ai_pretest_qualification=qualification,
        )

    wrong_population = replace(
        population,
        parent_tuning_subpartition_sha256="f" * 64,
    )
    wrong_population.validate()
    wrong_reports = tuple(
        replace(report, qualification_population=wrong_population)
        for report in qualification.development_reports
    )
    for report in wrong_reports:
        report.validate()
    wrong_qualification = build_round74_ai_pretest_qualification(wrong_reports)
    with pytest.raises(ValueError, match="qualification identity differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=selection,
            ai_pretest_qualification=wrong_qualification,
        )

    assert not ledger.path.exists()


def test_tampered_claim_or_duplicate_manifest_panel_is_rejected(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    selection = _selection()
    qualification = _ai_pretest_qualification(selection)
    duplicate_report = replace(
        qualification.development_reports[1],
        model_manifest_sha256=qualification.model_manifest_sha256[0],
    )
    duplicate_report.validate()
    duplicate = replace(
        qualification,
        development_reports=(
            qualification.development_reports[0],
            duplicate_report,
        ),
    )
    oversized = replace(
        qualification,
        development_reports=(
            *qualification.development_reports,
            replace(
                qualification.development_reports[1],
                model_manifest_sha256="c" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="pretest qualification differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=selection,
            ai_pretest_qualification=duplicate,
        )
    with pytest.raises(ValueError, match="pretest qualification differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=selection,
            ai_pretest_qualification=oversized,
        )
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=selection,
        ai_pretest_qualification=qualification,
    )
    tampered = replace(claim, dataset_sha256="f" * 64)
    tampered.validate()

    assert not ledger.claim_matches(tampered, required_status="reserved")
    tampered_payload = claim.as_dict()
    tampered_payload["rows"] = 25
    with pytest.raises(ValueError, match="digest differs"):
        Round74SealedEvaluationClaim.from_mapping(tampered_payload)


def test_ai_qualification_operator_uses_disjoint_tuning_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_batches = tuple(
        _test_batch(
            role="tuning",
            runs=(run_id, run_id) if index == 0 else (run_id,),
        )
        for index, run_id in enumerate(TEST_RUNS)
    )
    batches = tuple(
        replace(
            batch,
            sample_sha256=tuple(
                f"{7_000 + index * 10 + local_index:064x}"
                for local_index in range(batch.rows)
            ),
            feature_window_sha256=tuple(
                f"{8_000 + index * 10 + local_index:064x}"
                for local_index in range(batch.rows)
            ),
            target_context_sha256=tuple(
                f"{9_000 + index * 10 + local_index:064x}"
                for local_index in range(batch.rows)
            ),
            decision_wall_ns=_readonly(
                np.arange(batch.rows, dtype=np.int64) * 4_000_000_000_000
                + 1_800_000_000_000_000_000
                + index * 10_000_000_000_000
            ),
        )
        for index, batch in enumerate(raw_batches)
    )
    for batch in batches:
        batch.validate()
    base_calibration = _calibration()
    assert base_calibration.risk_quantiles is not None
    assert base_calibration.quantile_baseline is not None
    calibration = replace(
        base_calibration,
        risk_quantiles=replace(
            base_calibration.risk_quantiles,
            optimization_population="eligible_target",
        ),
        quantile_baseline=replace(
            base_calibration.quantile_baseline,
            optimization_population="eligible_target",
        ),
        optimization_population="eligible_target",
    )
    calibration.validate()
    selection = replace(
        _selection(optimization_population="eligible_target"),
        probability_calibration_sha256=calibration.calibration_sha256,
    )
    selection.validate()
    population = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256=selection.tuning_subpartition_sha256,
        prior_run_ids=POLICY_RUNS,
        prior_slot_ordinals=tuple(range(594, 600)),
        run_ids=TEST_RUNS,
        slot_ordinals=tuple(range(600, 624)),
        eligible_anchor_ns=(900_000_000_000,) * len(TEST_RUNS),
    )
    population.validate()
    contexts = tuple(build_round74_action_inference_context(batch) for batch in batches)
    outputs = tuple(_model_output(batch.rows) for batch in batches)
    candidates = tuple(
        derive_round74_action_candidates(
            output,
            context,
            calibration,
            pretest_policy_sha256=selection.pretest_policy_sha256,
            profile=selection.profile,
        )
        for output, context in zip(outputs, contexts, strict=True)
    )
    inference = Round74TargetFreeCandidateInference(
        contexts=contexts,
        model_outputs=outputs,
        candidates=candidates,
        pretest_policy_sha256=selection.pretest_policy_sha256,
        pretest_model_sha256="d" * 64,
        probability_calibration_sha256=calibration.calibration_sha256,
        action_selection_sha256=selection.selection_sha256,
        profile=selection.profile,
        inference_backend_kind="cpu",
        inference_backend_device="cpu",
        inference_backend_vendor="test",
        inference_warning_count=0,
        optimization_population="eligible_target",
        data_scope="ai_qualification_tuning",
        expected_run_ids=TEST_RUNS,
    )
    inference.validate()
    monkeypatch.setattr(
        qualification_subject,
        "infer_round74_target_free_candidates",
        lambda *_args, **_kwargs: inference,
    )

    def replay_provider(
        *,
        qualification_population: Round74AIQualificationPopulation,
        action_selection: Round74ActionPolicySelection,
        instructions_by_manifest: Mapping[
            str,
            tuple[Round74AIExecutionReplayInstruction, ...],
        ],
    ) -> Mapping[str, tuple[Round74AIExecutionReplayEvidence, ...]]:
        assert (
            qualification_population.population_sha256 == population.population_sha256
        )
        assert action_selection.selection_sha256 == selection.selection_sha256
        return {
            manifest: tuple(
                replay_subject._non_replay_evidence(
                    instruction,
                    capture_report_sha256=f"{index + 6_000:064x}",
                    target_spec_sha256="e" * 64,
                )
                for index, instruction in enumerate(instructions)
            )
            for manifest, instructions in instructions_by_manifest.items()
        }

    output_path = tmp_path / "ai-qualification.json"
    result = qualification_subject.run_round74_ai_pretest_qualification(
        batches,
        qualification_population=population,
        action_selection=selection,
        probability_calibration=calibration,
        pretest_policy_path=tmp_path / "not-read-by-injected-inference.json",
        execution_replay_provider=replay_provider,
        qualification_output_path=output_path,
        review_runner=_blocked_review,
        model_batch_preparer=lambda _binding: None,
        model_batch_finalizer=lambda _binding: None,
    )

    assert output_path.is_file()
    assert result.inference.data_scope == "ai_qualification_tuning"
    assert result.baseline_trace.expected_run_ids == TEST_RUNS
    assert result.baseline_trace.metrics.trades > len(TEST_RUNS)
    assert result.baseline_trace.run_id[:2] == (TEST_RUNS[0], TEST_RUNS[0])
    assert len(result.development_reports) == 2
    assert not result.qualification.qualification_passed
    assert all(
        instruction.action_selection_sha256 == selection.selection_sha256
        for _manifest, instructions in result.instructions_by_manifest
        for instruction in instructions
    )


def test_ai_qualification_store_provider_is_tuning_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _provider_partition()
    selection = _selection()
    run_id = f"{2:032x}"
    population = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256=selection.tuning_subpartition_sha256,
        prior_run_ids=POLICY_RUNS,
        prior_slot_ordinals=tuple(range(len(POLICY_RUNS))),
        run_ids=(run_id,),
        slot_ordinals=(len(POLICY_RUNS),),
        eligible_anchor_ns=(900_000_000_000,),
    )
    population.validate()
    store = ImpactAbsorptionStore(tmp_path / "qualification.duckdb", read_only=True)
    assembly = object.__new__(Round74SourceTargetAssembly)
    provider = Round74AIQualificationStoreExecutionReplayProvider(
        store=store,
        partition=partition,
        qualification_population=population,
        assembly_by_run_id={run_id: assembly},
    )
    manifests = ("a" * 64, "b" * 64)
    instructions = {
        manifest: (
            _replay_instruction(
                row_index=0,
                run_id=run_id,
                manifest=manifest,
                partition_sha256=partition.partition_sha256,
                action_selection_sha256=selection.selection_sha256,
            ),
        )
        for manifest in manifests
    }
    observed_runs: list[str] = []

    def replay_run(
        selected_store: object,
        *,
        partition: Round74EventRunPartition,
        run_id: str,
        assembly: Round74SourceTargetAssembly,
        instructions: tuple[Round74AIExecutionReplayInstruction, ...],
    ) -> tuple[Round74AIExecutionReplayEvidence, ...]:
        assert selected_store is store
        assert partition is provider.partition
        assert assembly is provider.assembly_by_run_id[run_id]
        observed_runs.append(run_id)
        return tuple(
            replay_subject._non_replay_evidence(
                row,
                capture_report_sha256=partition.entry(run_id).capture_report_sha256,
                target_spec_sha256="f" * 64,
            )
            for row in instructions
        )

    monkeypatch.setattr(
        replay_subject,
        "replay_round74_ai_execution_store_run",
        replay_run,
    )
    replayed = provider(
        qualification_population=population,
        action_selection=selection,
        instructions_by_manifest=instructions,
    )

    assert observed_runs == [run_id, run_id]
    assert tuple(replayed) == manifests
    assert all(rows[0].status == "runtime_veto" for rows in replayed.values())


def test_prepared_ai_qualification_wrapper_forwards_only_fourth_subrole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ai_run_ids = TEST_RUNS[:2]
    batches = tuple(_test_batch(role="tuning", runs=(run_id,)) for run_id in ai_run_ids)
    subpartition = SimpleNamespace(
        parent_partition_sha256=batches[0].partition_sha256,
        model_selection_run_ids=(),
        calibration_run_ids=(),
        policy_selection_run_ids=(),
        ai_qualification_run_ids=ai_run_ids,
        subpartition_sha256="7" * 64,
        validate=lambda: None,
    )
    roles = Round74PreparedTuningRoles(
        subpartition=subpartition,
        model_selection_batches=(),
        calibration_batches=(),
        policy_selection_batches=(),
        ai_qualification_batches=batches,
    )
    roles.validate()
    population = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256="7" * 64,
        prior_run_ids=POLICY_RUNS,
        prior_slot_ordinals=tuple(range(len(POLICY_RUNS))),
        run_ids=ai_run_ids,
        slot_ordinals=(600, 601),
        eligible_anchor_ns=(900_000_000_000, 900_000_000_000),
    )
    sentinel = object()
    observed: dict[str, object] = {}

    def run(batches: object, **kwargs: object) -> object:
        observed["batches"] = batches
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        qualification_subject,
        "run_round74_ai_pretest_qualification",
        run,
    )
    result = qualification_subject.run_round74_prepared_ai_pretest_qualification(
        roles,
        qualification_population=population,
        action_selection=_selection(),
        probability_calibration=_calibration(),
        pretest_policy_path="policy.json",
        execution_replay_provider=lambda **_kwargs: {},
        qualification_output_path="qualification.json",
    )

    assert result is sentinel
    assert observed["batches"] == batches
    assert observed["qualification_population"] is population
