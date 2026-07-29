from __future__ import annotations

from dataclasses import replace
import json

import pytest

import simple_ai_trading.impact_absorption_event_action_policy as action_policy_subject
from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.impact_absorption_ai_protocol import (
    Round74AIModelManifest,
    Round74AIReviewDecision,
    Round74AIReviewRequest,
    ROUND74_AI_TEMPORAL_BLOCK_COUNT,
    ROUND74_AI_TEMPORAL_FEATURE_NAMES,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    Round74AIRuntimeOutcome,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
    Round74AIQualificationPopulation,
    Round74AIUpliftDevelopmentReport,
    build_round74_ai_pretest_qualification,
    evaluate_round74_ai_overlay_development,
    load_round74_ai_pretest_qualification,
    write_round74_ai_pretest_qualification,
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
POLICY_RUNS = tuple(f"{index:032x}" for index in range(11, 17))
POLICY_FEATURES = tuple(f"{index + 700:064x}" for index in range(6))
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT") * 2
PAYOFFS = (2.0, -1.0, 2.0, -1.0, 2.0, 2.0)
WALL_NS = 1_800_000_000_000_000_000
ACTION_VALIDITY_LATENCY_NS = ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS


def _trace() -> Round74ActionTrace:
    portfolio_payoffs = tuple(value / 3.0 for value in PAYOFFS)
    metrics = Round74ActionTraceMetrics(
        trades=6,
        active_runs=6,
        distinct_symbols=3,
        total_net_bps=2.0,
        mean_run_net_bps=1.0 / 3.0,
        mean_net_bps=1.0 / 3.0,
        median_net_bps=2.0 / 3.0,
        win_rate=4.0 / 6.0,
        profit_factor=4.0,
        maximum_drawdown_bps=2.0 / 3.0,
        realized_maximum_drawdown_bps=1.0 / 3.0,
        maximum_concurrent_adverse_excursion_bps=1.0 / 3.0,
        gross_profit_bps=8.0 / 3.0,
        gross_loss_bps=2.0 / 3.0,
        worst_trade_bps=-1.0 / 3.0,
        mean_maximum_adverse_excursion_bps=1.0 / 3.0,
        mean_run_maximum_adverse_excursion_bps=1.0 / 3.0,
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
        net_payoff_bps=portfolio_payoffs,
        maximum_adverse_excursion_bps=(1.0 / 3.0,) * 6,
        adverse_selection=(0,) * 6,
        skipped_target_ineligible=0,
        skipped_same_symbol_overlap=0,
        metrics=metrics,
    )
    result.validate()
    return result


def _selection() -> Round74ActionPolicySelection:
    trace = replace(
        _trace(),
        expected_run_ids=POLICY_RUNS,
        run_id=POLICY_RUNS,
        feature_row_sha256=POLICY_FEATURES,
    )
    trace.validate()
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


def _qualification_population() -> Round74AIQualificationPopulation:
    result = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256="3" * 64,
        prior_run_ids=POLICY_RUNS,
        prior_slot_ordinals=tuple(range(594, 600)),
        run_ids=RUNS,
        slot_ordinals=tuple(range(600, 606)),
        eligible_anchor_ns=(900_000_000_000,) * len(RUNS),
    )
    result.validate()
    return result


def _evaluate(
    action_selection: Round74ActionPolicySelection,
    reviews: tuple[Round74AIPairedReviewEvidence, ...],
    executions: tuple[Round74AIExecutionReplayEvidence, ...],
) -> Round74AIUpliftDevelopmentReport:
    return evaluate_round74_ai_overlay_development(
        action_selection,
        reviews,
        executions,
        qualification_population=_qualification_population(),
        qualification_trace=_trace(),
        qualification_candidate_sha256=("a" * 64,),
    )


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
    latency_eligible = runtime_elapsed_ns <= ACTION_VALIDITY_LATENCY_NS
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
        action_validity_latency_ns=ACTION_VALIDITY_LATENCY_NS,
        action_latency_eligible=latency_eligible,
        size_multiplier_bps=multiplier,
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
        risk_profile="aggressive",
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
        feature_recent_block_means=tuple(
            tuple(0.0 for _ in ROUND74_AI_TEMPORAL_FEATURE_NAMES)
            for _ in range(ROUND74_AI_TEMPORAL_BLOCK_COUNT)
        ),
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
    requested_multiplier = decision.size_multiplier_bps if decision is not None else 0
    executed = (
        review.runtime_status == "accepted"
        and review.action_latency_eligible
        and requested_multiplier > 0
    )
    if review.runtime_status != "accepted":
        status = "runtime_veto"
    elif requested_multiplier == 0:
        status = "ai_veto"
    elif not review.action_latency_eligible:
        status = "historical_review_expired"
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
        applied_size_multiplier_bps=(requested_multiplier if executed else 0),
        exact_l2_replay_performed=executed,
        target_outcome_sha256=f"{index + 400:064x}" if executed else None,
        target_context_sha256=f"{index + 500:064x}" if executed else None,
        target_ineligible_reason="",
        requested_entry_monotonic_ns=11 if executed else None,
        actual_entry_monotonic_ns=12 if executed else None,
        actual_exit_monotonic_ns=20 if executed else None,
        reference_quote_notional=1_000.0 if executed else None,
        actual_entry_quote_notional=(1_000.0 * scale if executed else None),
        actual_deployed_capital_bps=(requested_multiplier if executed else 0.0),
        position_net_payoff_bps=PAYOFFS[index] if executed else 0.0,
        position_maximum_adverse_excursion_bps=(1.0 if executed else 0.0),
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

    report = _evaluate(
        _selection(),
        reviews,
        _executions(reviews),
    )

    assert report.development_gate_passed
    assert report.ai_metrics.total_net_bps == 8.0 / 3.0
    assert report.ai_metrics.realized_maximum_drawdown_bps == 0.0
    assert report.ai_metrics.maximum_concurrent_adverse_excursion_bps == (1.0 / 3.0)
    assert report.ai_metrics.maximum_drawdown_bps == 1.0 / 3.0
    assert report.ai_metrics.retained_trades == 4
    assert report.ai_metrics.distinct_retained_symbols == 3
    assert len(report.paired_symbol_horizons) == 3
    assert len(report.paired_run_symbol_horizons) == 6
    assert all(
        float(value["delta_net_bps"]) >= 0.0 for value in report.paired_symbol_horizons
    )
    assert all(
        float(value["delta_net_bps"]) >= 0.0
        for value in report.paired_run_symbol_horizons
    )
    assert report.sealed_test_accessed is False
    assert report.ai_model_selection_permitted is False
    assert report.promotion_authority is False
    assert report.profitability_claim is False
    assert len(report.report_sha256) == 64


def test_ai_pretest_qualification_binds_two_passing_development_reports(
    tmp_path,
) -> None:
    reviews = tuple(
        _review(index, 0 if payoff < 0.0 else 10_000)
        for index, payoff in enumerate(PAYOFFS)
    )
    first = _evaluate(
        _selection(),
        reviews,
        _executions(reviews),
    )
    second = replace(first, model_manifest_sha256="7" * 64)
    second.validate()

    qualification = build_round74_ai_pretest_qualification((second, first))

    assert qualification.qualification_passed
    assert qualification.model_manifest_sha256 == ("6" * 64, "7" * 64)
    assert qualification.gate_reasons == ()
    assert not qualification.sealed_test_accessed
    assert not qualification.model_selection_performed
    output = tmp_path / "ai-pretest-qualification.json"
    write_round74_ai_pretest_qualification(qualification, output)
    restored = load_round74_ai_pretest_qualification(output)
    assert restored.as_dict() == qualification.as_dict()
    write_round74_ai_pretest_qualification(qualification, output)
    different = build_round74_ai_pretest_qualification(
        (
            first,
            replace(second, model_manifest_sha256="8" * 64),
        )
    )
    with pytest.raises(
        FileExistsError,
        match="immutable AI pretest qualification differs",
    ):
        write_round74_ai_pretest_qualification(different, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["development_reports"][0]["ai_metrics"]["total_net_bps"] += 1.0
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_round74_ai_pretest_qualification(output)

    for mutate in (
        lambda value: value.__setitem__("qualification_passed", 1),
        lambda value: value.__setitem__("sealed_test_accessed", 0),
        lambda value: value["development_reports"][0].__setitem__(
            "trading_authority",
            0,
        ),
    ):
        payload = qualification.as_dict()
        mutate(payload)
        output.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_round74_ai_pretest_qualification(output)


def test_ai_pretest_qualification_rejects_a_failed_model() -> None:
    passing_reviews = tuple(
        _review(index, 0 if payoff < 0.0 else 10_000)
        for index, payoff in enumerate(PAYOFFS)
    )
    passing = _evaluate(
        _selection(),
        passing_reviews,
        _executions(passing_reviews),
    )
    failed_reviews = tuple(_review(index, 0) for index in range(6))
    failed = replace(
        _evaluate(
            _selection(),
            failed_reviews,
            _executions(failed_reviews),
        ),
        model_manifest_sha256="7" * 64,
    )
    failed.validate()

    qualification = build_round74_ai_pretest_qualification((passing, failed))

    assert not qualification.qualification_passed
    assert qualification.gate_reasons
    assert all(
        reason.startswith(f"model:{'7' * 64}:") for reason in qualification.gate_reasons
    )


def test_ai_development_rejects_policy_selection_run_reuse() -> None:
    reviews = tuple(_review(index, 10_000) for index in range(6))
    overlapping_population = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256="3" * 64,
        prior_run_ids=RUNS,
        prior_slot_ordinals=tuple(range(594, 600)),
        run_ids=POLICY_RUNS,
        slot_ordinals=tuple(range(600, 606)),
        eligible_anchor_ns=(900_000_000_000,) * len(POLICY_RUNS),
    )
    overlapping_population.validate()
    overlapping_trace = replace(
        _trace(),
        expected_run_ids=POLICY_RUNS,
        run_id=POLICY_RUNS,
        feature_row_sha256=POLICY_FEATURES,
    )
    overlapping_trace.validate()

    with pytest.raises(ValueError, match="qualification population identity differs"):
        evaluate_round74_ai_overlay_development(
            _selection(),
            reviews,
            _executions(reviews),
            qualification_population=overlapping_population,
            qualification_trace=overlapping_trace,
            qualification_candidate_sha256=("a" * 64,),
        )


def test_aggregate_ai_uplift_cannot_hide_run_or_asset_harm() -> None:
    reviews = tuple(
        _review(
            index,
            (5_000 if index == 2 else 0 if payoff < 0.0 else 10_000),
        )
        for index, payoff in enumerate(PAYOFFS)
    )

    report = _evaluate(
        _selection(),
        reviews,
        _executions(reviews),
    )

    assert report.ai_metrics.total_net_bps > report.baseline_trace.metrics.total_net_bps
    assert not report.development_gate_passed
    assert "paired_run_noninferiority_not_met" in report.gate_reasons
    assert "paired_symbol_horizon_noninferiority_not_met" in report.gate_reasons
    assert "paired_run_symbol_horizon_noninferiority_not_met" in report.gate_reasons
    sol = next(
        value for value in report.paired_symbol_horizons if value["symbol"] == "SOLUSDT"
    )
    assert float(sol["delta_net_bps"]) < 0.0
    corrupted = tuple(dict(value) for value in report.paired_symbol_horizons)
    corrupted[0]["delta_net_bps"] = float(corrupted[0]["delta_net_bps"]) + 1.0
    try:
        replace(report, paired_symbol_horizons=corrupted).validate()
    except ValueError as exc:
        assert "report differs" in str(exc)
    else:
        raise AssertionError("corrupted AI subgroup evidence was accepted")


def test_run_and_symbol_aggregates_cannot_hide_one_harmed_cell() -> None:
    run_ids = (RUNS[0], RUNS[0], RUNS[1], RUNS[1], RUNS[2], RUNS[2])
    expected_run_ids = RUNS[:3]
    symbols = ("BTCUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT", "ETHUSDT", "SOLUSDT")
    position_payoffs = (3.0, -3.0, -3.0, 3.0, 3.0, 3.0)
    baseline_payoffs = tuple(value / 3.0 for value in position_payoffs)
    entry = (10,) * len(run_ids)
    exit_value = (20,) * len(run_ids)
    trace = Round74ActionTrace(
        threshold_score=1.0,
        expected_run_ids=expected_run_ids,
        row_index=tuple(range(len(run_ids))),
        run_id=run_ids,
        symbol=symbols,
        feature_row_sha256=FEATURES,
        horizon_seconds=(30,) * len(run_ids),
        side=(1,) * len(run_ids),
        entry_monotonic_ns=entry,
        exit_monotonic_ns=exit_value,
        net_payoff_bps=baseline_payoffs,
        maximum_adverse_excursion_bps=(1.0 / 3.0,) * len(run_ids),
        adverse_selection=(0,) * len(run_ids),
        skipped_target_ineligible=0,
        skipped_same_symbol_overlap=0,
        metrics=action_policy_subject._trace_metrics(
            run_ids=run_ids,
            symbols=symbols,
            net_payoff_bps=baseline_payoffs,
            maximum_adverse_excursion_bps=(1.0 / 3.0,) * len(run_ids),
            adverse_selection=(0,) * len(run_ids),
            entry_monotonic_ns=entry,
            exit_monotonic_ns=exit_value,
            expected_run_ids=expected_run_ids,
        ),
    )
    trace.validate()
    population = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256="3" * 64,
        prior_run_ids=POLICY_RUNS,
        prior_slot_ordinals=tuple(range(594, 600)),
        run_ids=expected_run_ids,
        slot_ordinals=tuple(range(600, 603)),
        eligible_anchor_ns=(900_000_000_000,) * len(expected_run_ids),
    )
    population.validate()
    multipliers = (5_000, 0, 0, 10_000, 10_000, 10_000)
    reviews = []
    executions = []
    for index, (run_id, symbol, payoff, multiplier) in enumerate(
        zip(run_ids, symbols, position_payoffs, multipliers, strict=True)
    ):
        review = replace(
            _review(index, multiplier),
            run_id=run_id,
            symbol=symbol,
        )
        review.validate()
        reviews.append(review)
        execution = _execution(review, index)
        if execution.status == "executed":
            scale = multiplier / 10_000.0
            execution = replace(
                execution,
                position_net_payoff_bps=payoff,
                capital_scaled_net_payoff_bps=payoff * scale,
                position_maximum_adverse_excursion_bps=1.0,
                capital_scaled_maximum_adverse_excursion_bps=scale,
            )
        execution.validate()
        executions.append(execution)

    report = evaluate_round74_ai_overlay_development(
        _selection(),
        tuple(reviews),
        tuple(executions),
        qualification_population=population,
        qualification_trace=trace,
        qualification_candidate_sha256=("a" * 64,),
    )

    assert report.ai_metrics.total_net_bps > trace.metrics.total_net_bps
    assert all(float(value["delta_net_bps"]) >= 0.0 for value in report.paired_runs)
    assert all(
        float(value["delta_net_bps"]) >= 0.0 for value in report.paired_symbol_horizons
    )
    harmed = [
        value
        for value in report.paired_run_symbol_horizons
        if float(value["delta_net_bps"]) < 0.0
    ]
    assert len(harmed) == 1
    assert (harmed[0]["run_id"], harmed[0]["symbol"]) == (
        expected_run_ids[0],
        "BTCUSDT",
    )
    assert not report.development_gate_passed
    assert report.gate_reasons == ("paired_run_symbol_horizon_noninferiority_not_met",)
    corrupted = tuple(dict(value) for value in report.paired_run_symbol_horizons)
    corrupted[0]["paired_observations"] = 2
    with pytest.raises(ValueError, match="report differs"):
        replace(report, paired_run_symbol_horizons=corrupted).validate()
    with pytest.raises(ValueError, match="report differs"):
        replace(
            report,
            development_gate_passed=True,
            gate_reasons=(),
        ).validate()


def test_cell_profit_cannot_hide_worse_capital_scaled_adverse_excursion() -> None:
    reviews = tuple(
        _review(index, 0 if payoff < 0.0 else 10_000)
        for index, payoff in enumerate(PAYOFFS)
    )
    executions = list(_executions(reviews))
    executions[0] = replace(
        executions[0],
        position_maximum_adverse_excursion_bps=1.5,
        capital_scaled_maximum_adverse_excursion_bps=1.5,
    )
    executions[0].validate()

    report = _evaluate(
        _selection(),
        reviews,
        tuple(executions),
    )

    assert report.ai_metrics.total_net_bps > report.baseline_trace.metrics.total_net_bps
    assert (
        report.ai_metrics.maximum_drawdown_bps
        <= report.baseline_trace.metrics.maximum_drawdown_bps
    )
    assert all(
        float(value["delta_net_bps"]) >= 0.0
        for value in report.paired_run_symbol_horizons
    )
    harmed = [
        value
        for value in report.paired_run_symbol_horizons
        if float(value["delta_aggregate_adverse_excursion_bps"]) > 0.0
    ]
    assert len(harmed) == 1
    assert (
        harmed[0]["run_id"],
        harmed[0]["symbol"],
        harmed[0]["horizon_seconds"],
    ) == (RUNS[0], "BTCUSDT", 30)
    assert report.gate_reasons == (
        "paired_run_symbol_horizon_adverse_excursion_noninferiority_not_met",
    )
    corrupted = tuple(dict(value) for value in report.paired_run_symbol_horizons)
    corrupted[0]["ai_aggregate_adverse_excursion_bps"] = 0.0
    with pytest.raises(ValueError, match="report differs"):
        replace(report, paired_run_symbol_horizons=corrupted).validate()


def test_all_veto_overlay_fails_closed_without_dropping_pairs() -> None:
    reviews = tuple(_review(index, 0) for index in range(6))
    report = _evaluate(
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
        action_latency_eligible=False,
        size_multiplier_bps=0,
        decision=None,
    )
    reviews[0].validate()

    report = _evaluate(
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
        _evaluate(
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
        _evaluate(
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
        queue_delay_ns=0,
    )

    assert evidence.runtime_status == "accepted"
    assert evidence.runtime_elapsed_ns == 1_000
    assert evidence.action_latency_eligible
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
        queue_delay_ns=0,
    )

    assert evidence.runtime_status == "blocked_capability"
    assert evidence.action_latency_eligible is False
    assert evidence.size_multiplier_bps == 0
    assert evidence.decision is None


def test_expired_accepted_review_is_audited_and_cannot_reach_replay() -> None:
    request = _runtime_request()
    late_outcome = replace(
        _runtime_outcome(),
        elapsed_ns=ACTION_VALIDITY_LATENCY_NS + 1,
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
        queue_delay_ns=0,
    )

    assert evidence.runtime_status == "accepted"
    assert evidence.decision is not None
    assert evidence.decision.size_multiplier_bps == 5_000
    assert evidence.action_latency_eligible is False
    assert evidence.size_multiplier_bps == 5_000
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
        runtime_elapsed_ns=ACTION_VALIDITY_LATENCY_NS + 1,
    )
    report = _evaluate(
        _selection(),
        tuple(reviews),
        _executions(tuple(reviews)),
    )

    assert not report.development_gate_passed
    assert report.ai_metrics.runtime_success_rate == 1.0
    assert report.ai_metrics.action_latency_eligible_reviews == 5
    assert report.ai_metrics.exact_replay_required_reviews == 3
    assert report.ai_metrics.exact_replay_completed_reviews == 3
    assert report.as_dict()["latency_adjusted_replay_performed"] is True
    assert "paired_run_noninferiority_not_met" in report.gate_reasons


def test_exact_execution_evidence_is_causal_and_fail_closed() -> None:
    executed = _execution(_review(0, 10_000), 0)
    ineligible = replace(
        executed,
        status="target_ineligible",
        applied_size_multiplier_bps=0,
        target_ineligible_reason="entry_book_missing",
        actual_entry_monotonic_ns=None,
        actual_exit_monotonic_ns=None,
        reference_quote_notional=None,
        actual_entry_quote_notional=None,
        actual_deployed_capital_bps=0.0,
        position_net_payoff_bps=0.0,
        position_maximum_adverse_excursion_bps=0.0,
        capital_scaled_net_payoff_bps=0.0,
        capital_scaled_maximum_adverse_excursion_bps=0.0,
        adverse_selection=False,
    )
    ineligible.validate()

    for changed in (
        replace(executed, actual_entry_monotonic_ns=10),
        replace(
            executed,
            actual_deployed_capital_bps=(executed.actual_deployed_capital_bps + 1.0),
        ),
        replace(
            executed,
            capital_scaled_net_payoff_bps=(
                executed.capital_scaled_net_payoff_bps + 1.0
            ),
        ),
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
            reference_quote_notional=None,
            actual_entry_quote_notional=None,
            actual_deployed_capital_bps=0.0,
            position_net_payoff_bps=0.0,
            position_maximum_adverse_excursion_bps=0.0,
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


def test_queue_wait_is_included_in_action_latency() -> None:
    try:
        Round74AIPairedReviewEvidence.from_runtime(
            row_index=0,
            feature_row_sha256=FEATURES[0],
            run_id=RUNS[0],
            symbol="BTCUSDT",
            side=1,
            horizon_seconds=30,
            request=_runtime_request(),
            outcome=_runtime_outcome(),
            queue_delay_ns=ACTION_VALIDITY_LATENCY_NS,
        )
    except ValueError as exc:
        assert "expired AI queue request differs" in str(exc)
    else:
        raise AssertionError("expired queued AI inference was accepted")

    evidence = Round74AIPairedReviewEvidence.from_runtime(
        row_index=0,
        feature_row_sha256=FEATURES[0],
        run_id=RUNS[0],
        symbol="BTCUSDT",
        side=1,
        horizon_seconds=30,
        request=_runtime_request(),
        outcome=_runtime_outcome(status="blocked_expired"),
        queue_delay_ns=ACTION_VALIDITY_LATENCY_NS,
    )

    assert evidence.runtime_elapsed_ns == 1_000
    assert evidence.queue_delay_ns == ACTION_VALIDITY_LATENCY_NS
    assert evidence.effective_review_latency_ns == ACTION_VALIDITY_LATENCY_NS + 1_000
    assert evidence.action_latency_eligible is False
    assert evidence.queue_expired_before_inference is True
    assert evidence.runtime_status == "blocked_expired"
    assert evidence.decision is None
    assert evidence.size_multiplier_bps == 0
