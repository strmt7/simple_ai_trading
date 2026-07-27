from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_ai_trading import (
    impact_absorption_event_sealed_evaluation as sealed_subject,
)
from simple_ai_trading.impact_absorption_ai_protocol import (
    Round74AIReviewDecision,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (
    prepare_round74_target_free_ai_reviews,
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    Round74AIRuntimeOutcome,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    Round74AIPairedReviewEvidence,
)
from simple_ai_trading.impact_absorption_event_action_policy import (
    Round74ActionPolicySelection,
    Round74ActionThresholdEvaluation,
    Round74ActionTrace,
    Round74ActionTraceMetrics,
    build_round74_action_inference_context,
    derive_round74_action_candidates,
    round74_action_profile,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    Round74TemperatureFit,
)
from simple_ai_trading.impact_absorption_event_dataset import (
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


TEST_RUNS = tuple(f"{index:032x}" for index in range(100, 124))
POLICY_RUNS = tuple(f"{index:032x}" for index in range(200, 206))


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _test_batch(*, role: str = "test") -> Round74EventTrainingBatch:
    rows = 24
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
        run_id=TEST_RUNS,
        symbol=symbols,
        decision_monotonic_ns=_readonly(np.full(rows, 1_000_000_000, dtype=np.int64)),
        decision_wall_ns=_readonly(wall),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.arange(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{300 + index:064x}" for index in range(rows)),
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


def _selection() -> Round74ActionPolicySelection:
    metrics = Round74ActionTraceMetrics(
        trades=6,
        active_runs=6,
        distinct_symbols=3,
        total_net_bps=6.0,
        mean_run_net_bps=1.0,
        mean_net_bps=1.0,
        median_net_bps=1.0,
        win_rate=1.0,
        profit_factor=None,
        maximum_drawdown_bps=0.0,
        gross_profit_bps=6.0,
        gross_loss_bps=0.0,
        worst_trade_bps=1.0,
        mean_maximum_adverse_excursion_bps=1.0,
        mean_run_maximum_adverse_excursion_bps=1.0,
        adverse_selection_rate=0.0,
        profitable_run_ratio=1.0,
        maximum_symbol_trade_share=1.0 / 3.0,
    )
    trace = Round74ActionTrace(
        threshold_score=1.0,
        expected_run_ids=POLICY_RUNS,
        row_index=tuple(range(6)),
        run_id=POLICY_RUNS,
        symbol=ROUND74_EVENT_SYMBOLS * 2,
        feature_row_sha256=tuple(f"{400 + index:064x}" for index in range(6)),
        horizon_seconds=(30,) * 6,
        side=(1,) * 6,
        entry_monotonic_ns=(10,) * 6,
        exit_monotonic_ns=(20,) * 6,
        net_payoff_bps=(1.0,) * 6,
        maximum_adverse_excursion_bps=(1.0,) * 6,
        adverse_selection=(0,) * 6,
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
    )
    result.validate()
    return result


def _calibration() -> Round74ProbabilityCalibration:
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
    mae[:, horizon_index, 0, :] = torch.tensor((0.2, 0.4, 0.8, 1.2, 2.0))
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
            same_entry_latency_budget_ns=1_000_000,
            same_entry_latency_eligible=True,
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


def test_reservation_consumes_test_access_before_evaluation(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64, "b" * 64),
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
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64, "b" * 64),
        )


def test_completed_or_failed_reservation_cannot_be_reset_or_finalized_twice(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64,),
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
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
        )


def test_evaluation_error_is_durable_and_still_consumes_test(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64,),
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
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
        )


def test_sealed_evaluator_finalizes_failure_after_reservation(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    batch = _test_batch()

    with pytest.raises(ValueError, match="pretest policy could not be read"):
        evaluate_round74_sealed_once(
            (batch,),
            action_selection=_selection(),
            probability_calibration=_calibration(),
            pretest_policy_path=tmp_path / "missing-policy.json",
            ai_reviews_by_manifest={"a" * 64: ()},
            ledger=ledger,
            compute_backend="cpu",
        )

    with pytest.raises(Round74SealedReuseError, match="status=failed"):
        ledger.reserve(
            test_batches=(batch,),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
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
    manifest = "c" * 64
    policy = {
        "policy_sha256": selection.pretest_policy_sha256,
        "model_artifact": {"sha256": "d" * 64},
        "development_data": {
            "partition_sha256": batch.partition_sha256,
            "scaler_sha256": batch.scaler_sha256,
        },
    }
    monkeypatch.setattr(
        sealed_subject,
        "load_round74_pretest_policy",
        lambda _path: (_Model(), policy),
    )

    reviews = list(
        _reviews(
            batch,
            calibration,
            selection,
            manifest=manifest,
        )
    )
    reviews[0] = replace(
        reviews[0],
        runtime_elapsed_ns=1_000_001,
        effective_review_latency_ns=1_000_001,
        same_entry_latency_eligible=False,
        size_multiplier_bps=0,
    )
    reviews[0].validate()
    outcome = evaluate_round74_sealed_once(
        (batch,),
        action_selection=selection,
        probability_calibration=calibration,
        pretest_policy_path=tmp_path / "policy.json",
        ai_reviews_by_manifest={manifest: tuple(reviews)},
        ledger=ledger,
        compute_backend="cpu",
    )

    assert outcome.finalized_claim.status == "complete"
    assert outcome.report.result_outcome == "candidate_passed_predeclared_gates"
    assert outcome.report.qualified_configuration == ("ml_baseline",)
    assert outcome.report.baseline_metrics.executed_trades == 24
    assert outcome.report.baseline_metrics.financial_gate_passed
    assert outcome.report.predictive_diagnostics.eligible_action_targets == (
        24
        * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
        * len(ROUND74_EVENT_PAYOFF_SIDES)
    )
    overlay = outcome.report.ai_overlays[0]
    assert not overlay.uplift_gate_passed
    assert overlay.runtime_success_rate == 1.0
    assert overlay.same_entry_latency_eligible_reviews == 23
    assert overlay.retained_trades == 23
    assert "same_entry_latency_eligibility_rate_not_met" in overlay.gate_reasons
    assert overlay.strategy_metrics.selected_action_target_ineligible == 0
    assert outcome.report.inference_backend_kind == "cpu"
    assert outcome.report.profitability_claim is False
    assert ledger.claim_matches(outcome.finalized_claim, required_status="complete")


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
    metrics = sealed_subject._strategy_metrics(
        trace,
        np.ones(trace.metrics.trades, dtype=np.float64),
        profile=selection.profile,
        seed=1,
    )

    assert trace.skipped_target_ineligible == 1
    assert metrics.selected_action_target_ineligible == 1
    assert not metrics.financial_gate_passed
    assert "selected_action_target_coverage_incomplete" in metrics.gate_reasons


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

    def blocked_review(
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

    panel = prepare_round74_target_free_ai_reviews(
        inference,
        action_selection=selection,
        probability_calibration=calibration,
        same_entry_latency_budget_ns=1_000_000,
        review_runner=blocked_review,
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
    assert panel.same_entry_latency_budget_ns == 1_000_000
    assert panel.reviews[0][0].queue_delay_ns == 0
    assert panel.reviews[0][1].queue_delay_ns == 1_000_000_000_000
    assert (
        panel.reviews[0][1].effective_review_latency_ns
        == 6_000_000_000_000
    )
    assert len(panel.reviews) == 2
    assert all(len(value) == 24 for value in panel.reviews)
    assert all(
        review.runtime_status == "blocked_capability"
        and review.same_entry_latency_eligible is False
        and review.size_multiplier_bps == 0
        for reviews in panel.reviews
        for review in reviews
    )
    assert len(progress) == 48
    assert progress[-1]["completed_reviews"] == 48
    assert set(panel.reviews_by_manifest()) == {
        binding.manifest.manifest_sha256 for binding in default_models
    }
    assert panel.target_fields_accessed is False
    assert panel.trading_authority is False
    assert len(panel.panel_sha256) == 64


def test_development_data_is_rejected_before_ledger_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.sqlite3"
    ledger = Round74SealedEvaluationLedger(path)

    with pytest.raises(ValueError, match="rejects development data"):
        ledger.reserve(
            test_batches=(_test_batch(role="tuning"),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
        )

    assert not path.exists()


def test_tampered_claim_or_duplicate_manifest_panel_is_rejected(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    with pytest.raises(ValueError, match="manifest panel differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64, "a" * 64),
        )
    with pytest.raises(ValueError, match="manifest panel differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64, "b" * 64, "c" * 64),
        )
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64,),
    )
    tampered = replace(claim, dataset_sha256="f" * 64)
    tampered.validate()

    assert not ledger.claim_matches(tampered, required_status="reserved")
    tampered_payload = claim.as_dict()
    tampered_payload["rows"] = 25
    with pytest.raises(ValueError, match="digest differs"):
        Round74SealedEvaluationClaim.from_mapping(tampered_payload)
