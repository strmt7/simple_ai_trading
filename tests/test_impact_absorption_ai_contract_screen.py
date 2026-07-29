from __future__ import annotations

from dataclasses import replace
import json

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.impact_absorption_ai_contract_screen import (
    ROUND74_AI_CONTRACT_CASE_IDS,
    evaluate_round74_ai_contract_outcome,
    evaluate_round74_ai_mirror_consistency,
    round74_ai_contract_cases,
)
from simple_ai_trading.impact_absorption_ai_protocol import (
    Round74AIReviewDecision,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (
    round74_ai_review_challenger_model_panel,
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    Round74AIRuntimeOutcome,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    Round74AIWorkerResult,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_BINARY_FEATURE_COUNT,
)


def _outcome(
    case,
    decision: Round74AIReviewDecision,
) -> Round74AIRuntimeOutcome:
    request = case.build_request(1_800_000_000_000_000_000)
    worker = Round74AIWorkerResult(
        envelope_sha256="1" * 64,
        manifest_sha256="2" * 64,
        request_sha256=request.request_sha256,
        model_name="test:8b",
        model_digest="3" * 64,
        model_metadata_sha256="4" * 64,
        system_prompt_sha256="5" * 64,
        user_prompt_sha256="6" * 64,
        raw_response_sha256="7" * 64,
        decision=decision,
        residency=OllamaResidencyReport(
            requested_model="test:8b",
            status="gpu_resident",
            loaded_model="test:8b",
            digest="3" * 64,
            size_bytes=1,
            size_vram_bytes=1,
            vram_to_model_ratio=1.0,
        ),
        prompt_eval_count=1,
        eval_count=1,
        total_duration_ns=1,
        load_duration_ns=0,
        prompt_eval_duration_ns=0,
        eval_duration_ns=1,
    )
    return Round74AIRuntimeOutcome(
        status="accepted",
        request_sha256=request.request_sha256,
        manifest_sha256="2" * 64,
        deterministic_risk_gate_passed=True,
        observed_wall_ns=request.requested_wall_ns,
        proposed_risk_size_bps=request.proposed_risk_size_bps,
        approved_risk_size_bps=(
            request.proposed_risk_size_bps * decision.size_multiplier_bps // 10_000
        ),
        capability={},
        resolved_model_digest="3" * 64,
        resolved_model_metadata_sha256="4" * 64,
        worker_result=worker.as_dict(),
        elapsed_ns=1,
        failure_class=None,
        message="accepted",
    )


def test_round74_ai_contract_cases_are_frozen_anonymized_and_non_market() -> None:
    cases = round74_ai_contract_cases()

    assert tuple(case.case_id for case in cases) == ROUND74_AI_CONTRACT_CASE_IDS
    assert len({case.case_sha256 for case in cases}) == len(cases) == 10
    for case in cases:
        request = case.build_request(1_800_000_000_000_000_000)
        payload = request.prompt_payload()
        encoded = json.dumps(payload)
        assert payload["binary_feature_count"] == ROUND74_EVENT_BINARY_FEATURE_COUNT
        assert "BTCUSDT" not in encoded
        assert "ETHUSDT" not in encoded
        assert "SOLUSDT" not in encoded
        assert str(request.requested_wall_ns) not in encoded
        assert case.as_dict()["synthetic_non_market_contract_packet"] is True
        assert case.as_dict()["financial_edge_tested"] is False
        with pytest.raises(TypeError):
            case.request_template["risk_profile"] = "aggressive"


def test_round74_fin_r1_is_pinned_as_screen_only_challenger() -> None:
    default_panel = round74_default_ai_review_model_panel()
    challengers = round74_ai_review_challenger_model_panel()

    assert tuple(value.model_name for value in default_panel) == (
        "fino1:8b",
        "qwen3:8b",
    )
    assert tuple(value.model_name for value in challengers) == ("fin-r1:8b",)
    challenger = challengers[0]
    assert challenger.manifest.model_id == "SUFE-AIFLM-Lab/Fin-R1"
    assert challenger.manifest.model_revision == (
        "026768c4a015b591b54b240743edeac1de0970fa"
    )
    assert challenger.manifest.model_artifact_sha256 == (
        "7a02f6045046a36f53f1541e6fe0ceaff202c2ca48a47c1292fc82e055a4a377"
    )
    assert challenger.manifest.parameter_count == 7_620_000_000
    assert challenger.manifest.quantization == "q6_k"
    assert {value.manifest.manifest_sha256 for value in default_panel}.isdisjoint(
        value.manifest.manifest_sha256 for value in challengers
    )


def test_round74_ai_contract_mirrors_directional_packets() -> None:
    cases = round74_ai_contract_cases()
    long = cases[0].build_request(1_800_000_000_000_000_000)
    short = cases[1].build_request(1_800_000_000_000_000_000)

    assert long.side == "long"
    assert short.side == "short"
    assert long.payoff_quantiles_bps == short.payoff_quantiles_bps
    assert (
        long.maximum_adverse_excursion_quantiles_bps
        == short.maximum_adverse_excursion_quantiles_bps
    )
    assert long.positive_payoff_probability == short.positive_payoff_probability
    assert (
        long.opposing_positive_payoff_probability
        == short.opposing_positive_payoff_probability
    )
    assert (
        long.neither_positive_payoff_probability
        == short.neither_positive_payoff_probability
    )

    ambiguous_long = cases[4].build_request(1_800_000_000_000_000_000)
    ambiguous_short = cases[5].build_request(1_800_000_000_000_000_000)
    for request in (ambiguous_long, ambiguous_short):
        assert request.positive_payoff_probability == 0.58
        assert request.opposing_positive_payoff_probability == 0.39
        assert request.neither_positive_payoff_probability == pytest.approx(0.03)
        prompt = request.prompt_payload()["positive_payoff_outcome_probabilities"]
        assert prompt["proposed_minus_opposing_margin"] == 0.19


def test_round74_ai_contract_scores_retention_and_risk_semantics() -> None:
    cases = round74_ai_contract_cases()
    retained = evaluate_round74_ai_contract_outcome(
        cases[0],
        _outcome(
            cases[0],
            Round74AIReviewDecision(
                verdict="allow_unchanged",
                size_multiplier_bps=10_000,
                confidence_bps=8_000,
                reason_codes=("none",),
            ),
        ),
    )
    constrained = evaluate_round74_ai_contract_outcome(
        cases[2],
        _outcome(
            cases[2],
            Round74AIReviewDecision(
                verdict="veto",
                size_multiplier_bps=0,
                confidence_bps=8_000,
                reason_codes=("regime_unpredictability",),
            ),
        ),
    )

    assert retained["runtime_accepted"] is True
    assert retained["semantic_passed"] is True
    assert constrained["runtime_accepted"] is True
    assert constrained["semantic_passed"] is True
    wrong_reason = replace(
        _outcome(
            cases[2],
            Round74AIReviewDecision(
                verdict="veto",
                size_multiplier_bps=0,
                confidence_bps=8_000,
                reason_codes=("stale_state",),
            ),
        ),
        elapsed_ns=2,
    )
    assert (
        evaluate_round74_ai_contract_outcome(cases[2], wrong_reason)["semantic_passed"]
        is False
    )


def test_round74_ai_contract_scores_mirrored_side_consistency() -> None:
    cases = round74_ai_contract_cases()
    decision = Round74AIReviewDecision(
        verdict="reduce",
        size_multiplier_bps=7_500,
        confidence_bps=8_000,
        reason_codes=("forecast_uncertainty",),
    )
    results = [
        evaluate_round74_ai_contract_outcome(case, _outcome(case, decision))
        for case in cases[:2]
    ]

    checks = evaluate_round74_ai_mirror_consistency(results)

    assert checks[0]["pair_id"] == "benign"
    assert checks[0]["passed"] is True
    assert checks[0]["size_multiplier_difference_bps"] == 0
    assert checks[1]["pair_id"] == "unpredictable"
    assert checks[1]["complete"] is False
    assert checks[1]["passed"] is False
    assert checks[2]["pair_id"] == "directional_ambiguity"
    assert checks[2]["complete"] is False
    assert checks[2]["passed"] is False


def test_round74_ai_contract_scores_ambiguity_mirror_consistency() -> None:
    cases = round74_ai_contract_cases()
    decision = Round74AIReviewDecision(
        verdict="reduce",
        size_multiplier_bps=5_000,
        confidence_bps=8_000,
        reason_codes=("model_inconsistency",),
    )
    results = [
        evaluate_round74_ai_contract_outcome(case, _outcome(case, decision))
        for case in cases[4:6]
    ]

    checks = evaluate_round74_ai_mirror_consistency(results)

    assert checks[2]["pair_id"] == "directional_ambiguity"
    assert checks[2]["complete"] is True
    assert checks[2]["runtime_accepted"] is True
    assert checks[2]["passed"] is True
