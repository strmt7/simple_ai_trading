from __future__ import annotations

from dataclasses import replace

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.impact_absorption_ai_protocol import (
    Round74AIModelManifest,
    Round74AIReviewDecision,
    Round74AIReviewRequest,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    Round74AIRuntimeOutcome,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
    evaluate_round74_ai_overlay_development,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    Round74AIWorkerResult,
)
from simple_ai_trading.impact_absorption_event_action_policy import (
    Round74ActionPolicySelection,
    Round74ActionThresholdEvaluation,
    Round74ActionTrace,
    Round74ActionTraceMetrics,
    round74_action_profile,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
)


RUNS = tuple(f"{index:032x}" for index in range(1, 7))
FEATURES = tuple(f"{index + 100:064x}" for index in range(6))
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT") * 2
PAYOFFS = (2.0, -1.0, 2.0, -1.0, 2.0, 2.0)
WALL_NS = 1_800_000_000_000_000_000
SAME_ENTRY_LATENCY_BUDGET_NS = 1_000_000


def _trace() -> Round74ActionTrace:
    metrics = Round74ActionTraceMetrics(
        trades=6,
        active_runs=6,
        distinct_symbols=3,
        total_net_bps=6.0,
        mean_run_net_bps=1.0,
        mean_net_bps=1.0,
        median_net_bps=2.0,
        win_rate=4.0 / 6.0,
        profit_factor=4.0,
        maximum_drawdown_bps=1.0,
        gross_profit_bps=8.0,
        gross_loss_bps=2.0,
        worst_trade_bps=-1.0,
        mean_maximum_adverse_excursion_bps=1.0,
        mean_run_maximum_adverse_excursion_bps=1.0,
        adverse_selection_rate=0.0,
        profitable_run_ratio=4.0 / 6.0,
        maximum_symbol_trade_share=1.0 / 3.0,
    )
    result = Round74ActionTrace(
        threshold_score=1.0,
        expected_run_ids=RUNS,
        row_index=tuple(range(6)),
        run_id=RUNS,
        symbol=SYMBOLS,
        feature_row_sha256=FEATURES,
        horizon_seconds=(30,) * 6,
        side=(1,) * 6,
        entry_monotonic_ns=(10,) * 6,
        exit_monotonic_ns=(20,) * 6,
        net_payoff_bps=PAYOFFS,
        maximum_adverse_excursion_bps=(1.0,) * 6,
        adverse_selection=(0,) * 6,
        skipped_target_ineligible=0,
        skipped_same_symbol_overlap=0,
        metrics=metrics,
    )
    result.validate()
    return result


def _selection() -> Round74ActionPolicySelection:
    trace = _trace()
    quantiles = round74_action_profile("aggressive").threshold_quantiles
    evaluations = tuple(
        Round74ActionThresholdEvaluation(
            quantile=quantile,
            threshold_score=1.0,
            objective_bps=4.0,
            accepted=True,
            rejection_reasons=(),
            trace=trace,
        )
        for quantile in quantiles
    )
    result = Round74ActionPolicySelection(
        profile="aggressive",
        pretest_policy_sha256="1" * 64,
        probability_calibration_sha256="2" * 64,
        tuning_subpartition_sha256="3" * 64,
        target_batch_sha256=("4" * 64,),
        candidate_sha256=("5" * 64,),
        accepted=True,
        selected_quantile=quantiles[0],
        selected_threshold_score=1.0,
        evaluations=evaluations,
        rejection_reasons=(),
    )
    result.validate()
    return result


def _review(
    index: int,
    multiplier: int,
    *,
    runtime_elapsed_ns: int = 1_000,
) -> Round74AIPairedReviewEvidence:
    if multiplier == 10_000:
        decision = Round74AIReviewDecision(
            verdict="allow_unchanged",
            size_multiplier_bps=10_000,
            confidence_bps=8_000,
            reason_codes=("none",),
        )
    elif multiplier == 0:
        decision = Round74AIReviewDecision(
            verdict="veto",
            size_multiplier_bps=0,
            confidence_bps=8_000,
            reason_codes=("forecast_uncertainty",),
        )
    else:
        decision = Round74AIReviewDecision(
            verdict="reduce",
            size_multiplier_bps=multiplier,
            confidence_bps=8_000,
            reason_codes=("forecast_uncertainty",),
        )
    latency_eligible = runtime_elapsed_ns <= SAME_ENTRY_LATENCY_BUDGET_NS
    result = Round74AIPairedReviewEvidence(
        row_index=index,
        feature_row_sha256=FEATURES[index],
        run_id=RUNS[index],
        symbol=SYMBOLS[index],
        side=1,
        horizon_seconds=30,
        pretest_policy_sha256="1" * 64,
        probability_calibration_sha256="2" * 64,
        request_sha256=f"{index + 200:064x}",
        runtime_outcome_sha256=f"{index + 300:064x}",
        model_manifest_sha256="6" * 64,
        runtime_status="accepted",
        runtime_elapsed_ns=runtime_elapsed_ns,
        queue_delay_ns=0,
        effective_review_latency_ns=runtime_elapsed_ns,
        same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
        same_entry_latency_eligible=latency_eligible,
        size_multiplier_bps=multiplier if latency_eligible else 0,
        decision=decision,
    )
    result.validate()
    return result


def _runtime_request() -> Round74AIReviewRequest:
    count = len(ROUND74_EVENT_FEATURE_NAMES)
    return Round74AIReviewRequest(
        pretest_policy_sha256="1" * 64,
        probability_calibration_sha256="2" * 64,
        sample_sha256=FEATURES[0],
        deterministic_risk_state_sha256="7" * 64,
        asset_slot=0,
        side="long",
        horizon_seconds=30,
        requested_wall_ns=WALL_NS,
        expires_wall_ns=WALL_NS + 20_000_000_000,
        proposed_risk_size_bps=2_500,
        feature_last=tuple(0.0 for _ in range(count)),
        feature_mean=tuple(0.0 for _ in range(count)),
        feature_standard_deviation=tuple(0.0 for _ in range(count)),
        feature_recent_change=tuple(0.0 for _ in range(count)),
        payoff_quantiles_bps=(-2.0, 0.0, 2.0, 4.0, 6.0),
        maximum_adverse_excursion_quantiles_bps=(
            0.2,
            0.4,
            0.8,
            1.2,
            2.0,
        ),
        positive_payoff_probability=0.7,
        adverse_selection_probability=0.2,
        regime_unpredictability_probability=0.2,
    )


def _runtime_manifest() -> Round74AIModelManifest:
    return Round74AIModelManifest(
        model_id="TheFinAI/Fino1-8B",
        model_revision="a" * 40,
        model_artifact_sha256="d" * 64,
        model_artifact_kind="ollama_manifest",
        parameter_count=8_000_000_000,
        quantization="q6_k",
        runtime_backend="llama.cpp-vulkan",
        runtime_version="0.32.4",
        license_id="llama3.1",
        model_card_url="https://huggingface.co/TheFinAI/Fino1-8B",
        minimum_vram_bytes=8 * 1024**3,
        finance_specialized=True,
    )


def _runtime_outcome(*, status: str = "accepted") -> Round74AIRuntimeOutcome:
    request = _runtime_request()
    manifest = _runtime_manifest()
    worker: Round74AIWorkerResult | None = None
    approved = 0
    failure: str | None = "AICapabilityGate"
    if status == "accepted":
        decision = Round74AIReviewDecision(
            verdict="reduce",
            size_multiplier_bps=5_000,
            confidence_bps=8_000,
            reason_codes=("forecast_uncertainty",),
        )
        worker = Round74AIWorkerResult(
            envelope_sha256="8" * 64,
            manifest_sha256=manifest.manifest_sha256,
            request_sha256=request.request_sha256,
            model_name="fino1:8b",
            model_digest="d" * 64,
            model_metadata_sha256="e" * 64,
            system_prompt_sha256="9" * 64,
            user_prompt_sha256="a" * 64,
            raw_response_sha256="b" * 64,
            decision=decision,
            residency=OllamaResidencyReport(
                requested_model="fino1:8b",
                status="gpu_resident",
                loaded_model="fino1:8b",
                digest="d" * 64,
                size_bytes=1_000,
                size_vram_bytes=1_000,
                vram_to_model_ratio=1.0,
            ),
            prompt_eval_count=100,
            eval_count=20,
            total_duration_ns=1_000,
            load_duration_ns=100,
            prompt_eval_duration_ns=300,
            eval_duration_ns=600,
        )
        approved = 1_250
        failure = None
    result = Round74AIRuntimeOutcome(
        status=status,
        request_sha256=request.request_sha256,
        manifest_sha256=manifest.manifest_sha256,
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS + 1_000_000_000,
        proposed_risk_size_bps=request.proposed_risk_size_bps,
        approved_risk_size_bps=approved,
        capability={},
        resolved_model_digest="d" * 64 if worker is not None else None,
        resolved_model_metadata_sha256=("e" * 64 if worker is not None else None),
        worker_result=worker.as_dict() if worker is not None else None,
        elapsed_ns=1_000,
        failure_class=failure,
        message="ok" if worker is not None else "blocked",
    )
    result.validate()
    return result


def _execution(
    review: Round74AIPairedReviewEvidence,
    index: int,
) -> Round74AIExecutionReplayEvidence:
    decision = review.decision
    requested_multiplier = (
        decision.size_multiplier_bps if decision is not None else 0
    )
    executed = (
        review.runtime_status == "accepted" and requested_multiplier > 0
    )
    if review.runtime_status != "accepted":
        status = "runtime_veto"
    elif requested_multiplier == 0:
        status = "ai_veto"
    else:
        status = "executed"
    scale = requested_multiplier / 10_000.0 if executed else 0.0
    result = Round74AIExecutionReplayEvidence(
        row_index=review.row_index,
        feature_row_sha256=review.feature_row_sha256,
        run_id=review.run_id,
        symbol=review.symbol,
        side=review.side,
        horizon_seconds=review.horizon_seconds,
        source_review_sha256=review.review_sha256,
        partition_sha256="7" * 64,
        source_capture_report_sha256="8" * 64,
        target_spec_sha256="9" * 64,
        status=status,
        requested_size_multiplier_bps=requested_multiplier,
        applied_size_multiplier_bps=(
            requested_multiplier if executed else 0
        ),
        exact_l2_replay_performed=executed,
        target_outcome_sha256=f"{index + 400:064x}" if executed else None,
        target_context_sha256=f"{index + 500:064x}" if executed else None,
        target_ineligible_reason="",
        requested_entry_monotonic_ns=11 if executed else None,
        actual_entry_monotonic_ns=12 if executed else None,
        actual_exit_monotonic_ns=20 if executed else None,
        capital_scaled_net_payoff_bps=PAYOFFS[index] * scale,
        capital_scaled_maximum_adverse_excursion_bps=1.0 * scale,
        adverse_selection=False,
    )
    result.validate()
    return result


def _executions(
    reviews: tuple[Round74AIPairedReviewEvidence, ...],
) -> tuple[Round74AIExecutionReplayEvidence, ...]:
    return tuple(_execution(review, index) for index, review in enumerate(reviews))


def test_ai_overlay_can_only_improve_by_vetoing_preexisting_losses() -> None:
    reviews = tuple(
        _review(index, 0 if payoff < 0.0 else 10_000)
        for index, payoff in enumerate(PAYOFFS)
    )

    report = evaluate_round74_ai_overlay_development(
        _selection(),
        reviews,
        _executions(reviews),
    )

    assert report.development_gate_passed
    assert report.ai_metrics.total_net_bps == 8.0
    assert report.ai_metrics.maximum_drawdown_bps == 0.0
    assert report.ai_metrics.retained_trades == 4
    assert report.ai_metrics.distinct_retained_symbols == 3
    assert report.sealed_test_accessed is False
    assert report.ai_model_selection_permitted is False
    assert report.promotion_authority is False
    assert report.profitability_claim is False
    assert len(report.report_sha256) == 64


def test_all_veto_overlay_fails_closed_without_dropping_pairs() -> None:
    reviews = tuple(_review(index, 0) for index in range(6))
    report = evaluate_round74_ai_overlay_development(
        _selection(),
        reviews,
        _executions(reviews),
    )

    assert not report.development_gate_passed
    assert report.ai_metrics.vetoed_trades == 6
    assert report.ai_metrics.retained_trades == 0
    assert report.ai_metrics.baseline_trades == 6
    assert len(report.paired_runs) == 6
    assert "retained_trade_ratio_not_met" in report.gate_reasons
    assert "positive_paired_after_cost_uplift_not_met" in report.gate_reasons


def test_blocked_runtime_review_is_a_paired_zero_exposure_veto() -> None:
    reviews = [_review(index, 10_000) for index in range(6)]
    reviews[0] = replace(
        reviews[0],
        runtime_status="blocked_capability",
        same_entry_latency_eligible=False,
        size_multiplier_bps=0,
        decision=None,
    )
    reviews[0].validate()

    report = evaluate_round74_ai_overlay_development(
        _selection(),
        tuple(reviews),
        _executions(tuple(reviews)),
    )

    assert not report.development_gate_passed
    assert report.ai_metrics.baseline_trades == 6
    assert report.ai_metrics.runtime_accepted_reviews == 5
    assert report.ai_metrics.vetoed_trades == 1
    assert "runtime_success_rate_not_met" in report.gate_reasons


def test_review_coverage_and_action_identity_must_match_exactly() -> None:
    reviews = tuple(_review(index, 10_000) for index in range(6))

    try:
        evaluate_round74_ai_overlay_development(
            _selection(),
            reviews[:-1],
            _executions(reviews),
        )
    except ValueError as exc:
        assert "coverage differs" in str(exc)
    else:
        raise AssertionError("missing AI review was accepted")

    changed = list(reviews)
    changed[2] = replace(changed[2], side=-1)
    changed[2].validate()
    try:
        evaluate_round74_ai_overlay_development(
            _selection(),
            tuple(changed),
            _executions(tuple(changed)),
        )
    except ValueError as exc:
        assert "action identity differs" in str(exc)
    else:
        raise AssertionError("mismatched AI action was accepted")


def test_runtime_evidence_reduction_is_recomputed_and_bound() -> None:
    request = _runtime_request()
    outcome = _runtime_outcome()

    evidence = Round74AIPairedReviewEvidence.from_runtime(
        row_index=0,
        feature_row_sha256=FEATURES[0],
        run_id=RUNS[0],
        symbol="BTCUSDT",
        side=1,
        horizon_seconds=30,
        request=request,
        outcome=outcome,
        same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
        queue_delay_ns=0,
    )

    assert evidence.runtime_status == "accepted"
    assert evidence.runtime_elapsed_ns == 1_000
    assert evidence.same_entry_latency_eligible
    assert evidence.size_multiplier_bps == 5_000
    assert evidence.decision is not None
    changed = replace(outcome, approved_risk_size_bps=1_000)
    changed.validate()
    try:
        Round74AIPairedReviewEvidence.from_runtime(
            row_index=0,
            feature_row_sha256=FEATURES[0],
            run_id=RUNS[0],
            symbol="BTCUSDT",
            side=1,
            horizon_seconds=30,
            request=request,
            outcome=changed,
            same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
            queue_delay_ns=0,
        )
    except ValueError as exc:
        assert "approved risk size differs" in str(exc)
    else:
        raise AssertionError("mismatched approved risk was accepted")


def test_blocked_runtime_evidence_reduces_to_zero_without_decision() -> None:
    evidence = Round74AIPairedReviewEvidence.from_runtime(
        row_index=0,
        feature_row_sha256=FEATURES[0],
        run_id=RUNS[0],
        symbol="BTCUSDT",
        side=1,
        horizon_seconds=30,
        request=_runtime_request(),
        outcome=_runtime_outcome(status="blocked_capability"),
        same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
        queue_delay_ns=0,
    )

    assert evidence.runtime_status == "blocked_capability"
    assert evidence.same_entry_latency_eligible is False
    assert evidence.size_multiplier_bps == 0
    assert evidence.decision is None


def test_late_accepted_review_is_audited_but_cannot_inherit_ml_fill() -> None:
    request = _runtime_request()
    late_outcome = replace(
        _runtime_outcome(),
        elapsed_ns=SAME_ENTRY_LATENCY_BUDGET_NS + 1,
    )
    late_outcome.validate()

    evidence = Round74AIPairedReviewEvidence.from_runtime(
        row_index=0,
        feature_row_sha256=FEATURES[0],
        run_id=RUNS[0],
        symbol="BTCUSDT",
        side=1,
        horizon_seconds=30,
        request=request,
        outcome=late_outcome,
        same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
        queue_delay_ns=0,
    )

    assert evidence.runtime_status == "accepted"
    assert evidence.decision is not None
    assert evidence.decision.size_multiplier_bps == 5_000
    assert evidence.same_entry_latency_eligible is False
    assert evidence.size_multiplier_bps == 0
    assert evidence.as_dict()["latency_adjusted_replay_performed"] is False
    on_time_evidence = Round74AIPairedReviewEvidence.from_runtime(
        row_index=0,
        feature_row_sha256=FEATURES[0],
        run_id=RUNS[0],
        symbol="BTCUSDT",
        side=1,
        horizon_seconds=30,
        request=request,
        outcome=_runtime_outcome(),
        same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
        queue_delay_ns=0,
    )
    assert evidence.review_sha256 != on_time_evidence.review_sha256

    reviews = [
        _review(index, 0 if payoff < 0.0 else 10_000)
        for index, payoff in enumerate(PAYOFFS)
    ]
    reviews[0] = _review(
        0,
        10_000,
        runtime_elapsed_ns=SAME_ENTRY_LATENCY_BUDGET_NS + 1,
    )
    report = evaluate_round74_ai_overlay_development(
        _selection(),
        tuple(reviews),
        _executions(tuple(reviews)),
    )

    assert report.development_gate_passed
    assert report.ai_metrics.runtime_success_rate == 1.0
    assert report.ai_metrics.same_entry_latency_eligible_reviews == 5
    assert report.ai_metrics.exact_replay_required_reviews == 4
    assert report.ai_metrics.exact_replay_completed_reviews == 4
    assert report.as_dict()["latency_adjusted_replay_performed"] is True
    assert "same_entry_latency_eligibility_rate_not_met" not in report.gate_reasons


def test_exact_execution_evidence_is_causal_and_fail_closed() -> None:
    executed = _execution(_review(0, 10_000), 0)
    ineligible = replace(
        executed,
        status="target_ineligible",
        applied_size_multiplier_bps=0,
        target_ineligible_reason="entry_book_missing",
        actual_entry_monotonic_ns=None,
        actual_exit_monotonic_ns=None,
        capital_scaled_net_payoff_bps=0.0,
        capital_scaled_maximum_adverse_excursion_bps=0.0,
        adverse_selection=False,
    )
    ineligible.validate()

    for changed in (
        replace(executed, actual_entry_monotonic_ns=10),
        replace(
            executed,
            status="ai_veto",
            applied_size_multiplier_bps=0,
            exact_l2_replay_performed=False,
            target_outcome_sha256=None,
            target_context_sha256=None,
            requested_entry_monotonic_ns=None,
            actual_entry_monotonic_ns=None,
            actual_exit_monotonic_ns=None,
            capital_scaled_net_payoff_bps=0.0,
            capital_scaled_maximum_adverse_excursion_bps=0.0,
        ),
    ):
        try:
            changed.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid AI execution evidence was accepted")


def test_queue_wait_is_included_in_same_entry_latency() -> None:
    evidence = Round74AIPairedReviewEvidence.from_runtime(
        row_index=0,
        feature_row_sha256=FEATURES[0],
        run_id=RUNS[0],
        symbol="BTCUSDT",
        side=1,
        horizon_seconds=30,
        request=_runtime_request(),
        outcome=_runtime_outcome(),
        same_entry_latency_budget_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
        queue_delay_ns=SAME_ENTRY_LATENCY_BUDGET_NS,
    )

    assert evidence.runtime_elapsed_ns == 1_000
    assert evidence.queue_delay_ns == SAME_ENTRY_LATENCY_BUDGET_NS
    assert evidence.effective_review_latency_ns == 1_001_000
    assert evidence.same_entry_latency_eligible is False
    assert evidence.decision is not None
    assert evidence.size_multiplier_bps == 0
