"""Actual-shape, target-free prompt envelope for Round 28 AI candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
import math

from .ai_runtime import ollama_residency_from_mapping
from .polymarket_round28_ai_cases import (
    Round28AICase,
    Round28AICasePanel,
    round28_ai_case_prompt,
)
from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS,
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    validate_round28_ai_contract,
)
from .polymarket_round28_ai_host import (
    Round28AIHostCandidate,
    validate_round28_ai_host_report,
)
from .polymarket_round28_ai_inference import (
    Round28AIInferenceReport,
    run_round28_ai_inference,
    validate_round28_ai_inference_report,
)
from .polymarket_round28_book_ticker import POLYMARKET_ROUND28_FEATURE_NAMES


POLYMARKET_ROUND28_AI_PROMPT_ENVELOPE_SCHEMA_VERSION = (
    "polymarket-round28-ai-prompt-envelope-report-v1"
)
_CONTEXT_TOKENS = 8_192
_MAXIMUM_OUTPUT_TOKENS = 96
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_NON_AUTHORITY_FLAGS = (
    "target_accessed",
    "outcome_accessed",
    "future_books_accessed",
    "pnl_accessed",
    "credentials_used",
    "orders_submitted",
    "trading_authority",
    "edge_claim",
    "profitability_claim",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 AI prompt envelope {name} SHA-256 differs")
    return selected


def build_round28_ai_prompt_envelope_panel() -> Round28AICasePanel:
    """Build one deterministic maximum-width case without market observations."""

    causal_features = tuple(
        (
            name,
            (
                -1.2345678901234567e200
                if index % 2
                else 1.2345678901234567e200
            ),
        )
        for index, name in enumerate(POLYMARKET_ROUND28_FEATURE_NAMES)
    )
    synthetic_evidence = _canonical_sha256(
        {
            "schema_version": "polymarket-round28-ai-synthetic-envelope-source-v1",
            "feature_count": len(causal_features),
            "market_data_used": False,
            "target_used": False,
            "outcome_used": False,
        }
    )
    provisional_case = Round28AICase(
        partition_role="selection",
        condition_id="0x" + "1" * 64,
        event_start_ms=1_700_000_000_000,
        market_end_ms=1_700_000_300_000,
        decision_time_ms=1_700_000_150_000,
        proposed_side="Up",
        token_id="synthetic-envelope-no-market-data",
        predicted_probability=0.55,
        market_prior_probability_up=0.5,
        quantity="5",
        limit_price="0.50",
        decision_tick_size="0.01",
        decision_average_price="0.50",
        decision_fee_quote="0.01",
        expected_edge_per_contract="0.02",
        segment_id="synthetic-envelope",
        connection_id="synthetic-envelope",
        decision_book_event_id="synthetic-envelope",
        decision_source_payload_sha256=synthetic_evidence,
        feature_row_sha256=synthetic_evidence,
        feature_source_chain_sha256=synthetic_evidence,
        selection_claim_sha256="2" * 64,
        model_name="synthetic-envelope",
        model_feature_view="round28_bbo_augmented",
        model_sha256="3" * 64,
        causal_features=causal_features,
        source_evidence_sha256=synthetic_evidence,
        case_sha256="",
    )
    case = replace(
        provisional_case,
        case_sha256=_canonical_sha256(provisional_case.identity_payload()),
    ).validated()
    condition_ids_sha256 = _canonical_sha256([case.condition_id])
    provisional_panel = Round28AICasePanel(
        partition_role="selection",
        source_run_id="synthetic-prompt-envelope",
        model_name=case.model_name,
        model_sha256=case.model_sha256,
        selection_claim_sha256=case.selection_claim_sha256,
        source_audit_sha256=synthetic_evidence,
        economic_config={"synthetic_prompt_envelope": True},
        evaluated_condition_count=1,
        evaluated_condition_ids_sha256=condition_ids_sha256,
        baseline_candidate_population_sha256=_canonical_sha256(
            [case.case_sha256]
        ),
        selection_reason_counts={},
        cases=(case,),
        panel_sha256="",
    )
    return replace(
        provisional_panel,
        panel_sha256=_canonical_sha256(provisional_panel.identity_payload()),
    ).validated()


def _report_body(
    *,
    host_report: Mapping[str, object],
    candidate: Round28AIHostCandidate,
    panel: Round28AICasePanel,
    inference: Round28AIInferenceReport,
) -> dict[str, object]:
    response = inference.responses[0]
    prompt = round28_ai_case_prompt(panel.cases[0])
    prompt_tokens = response.prompt_tokens
    residency = ollama_residency_from_mapping(inference.residency)
    checks = {
        "actual_278_feature_prompt_shape": len(panel.cases[0].causal_features)
        == 278,
        "single_synthetic_case_not_economic_evidence": (
            inference.candidate_eligible_for_matched_evaluation is False
        ),
        "structured_response_valid": response.valid,
        "within_case_timeout": (
            response.valid
            and response.wall_latency_ms
            <= math.ceil(POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS * 1_000)
        ),
        "prompt_and_maximum_output_fit_context": (
            type(prompt_tokens) is int
            and prompt_tokens + _MAXIMUM_OUTPUT_TOKENS <= _CONTEXT_TOKENS
        ),
        "exclusive_full_gpu_residency": residency.fully_gpu_resident,
        "provider_unloaded_after_probe": (
            inference.unload_observed and inference.unload_failure is None
        ),
        "no_inference_failure": inference.status_counts == {"valid": 1},
    }
    return {
        "schema_version": POLYMARKET_ROUND28_AI_PROMPT_ENVELOPE_SCHEMA_VERSION,
        "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
        "candidate": asdict(candidate),
        "host_qualification_report_sha256": host_report["report_sha256"],
        "synthetic_case_panel_sha256": panel.panel_sha256,
        "synthetic_prompt_sha256": hashlib.sha256(prompt.encode("ascii")).hexdigest(),
        "synthetic_prompt_bytes": len(prompt.encode("ascii")),
        "inference_report": inference.asdict(),
        "checks": checks,
        "passed": all(checks.values()),
        "knowledge": {
            "market_data_used": False,
            "official_outcomes_accessed": False,
            "performance_metrics_computed": False,
            "stage1_feature_rows_accessed": False,
        },
        **{field: False for field in _NON_AUTHORITY_FLAGS},
    }


def evaluate_round28_ai_prompt_envelope(
    *,
    contract: Mapping[str, object],
    host_qualification_report: Mapping[str, object],
) -> dict[str, object]:
    """Run one actual-shape synthetic prompt under the production deadline."""

    selected_contract = validate_round28_ai_contract(contract)
    host_report, candidate = validate_round28_ai_host_report(
        host_qualification_report,
        contract=selected_contract,
    )
    panel = build_round28_ai_prompt_envelope_panel()
    inference = run_round28_ai_inference(
        panel=panel,
        candidate=candidate,
        contract=selected_contract,
        host_qualification_report=host_report,
    )
    body = _report_body(
        host_report=host_report,
        candidate=candidate,
        panel=panel,
        inference=inference,
    )
    body["report_sha256"] = _canonical_sha256(body)
    return validate_round28_ai_prompt_envelope_report(
        body,
        contract=selected_contract,
        host_qualification_report=host_report,
    )


def validate_round28_ai_prompt_envelope_report(
    value: Mapping[str, object],
    *,
    contract: Mapping[str, object],
    host_qualification_report: Mapping[str, object],
) -> dict[str, object]:
    """Recursively validate one prompt-envelope receipt and host binding."""

    selected_contract = validate_round28_ai_contract(contract)
    host_report, candidate = validate_round28_ai_host_report(
        host_qualification_report,
        contract=selected_contract,
    )
    report = dict(value)
    claimed = _sha256(report.pop("report_sha256", ""), name="report")
    raw_inference = report.get("inference_report")
    if not isinstance(raw_inference, Mapping):
        raise ValueError("Round 28 AI prompt envelope inference differs")
    panel = build_round28_ai_prompt_envelope_panel()
    inference = validate_round28_ai_inference_report(
        raw_inference,
        contract=selected_contract,
        host_qualification_report=host_report,
        panel=panel,
    )
    expected = _report_body(
        host_report=host_report,
        candidate=candidate,
        panel=panel,
        inference=inference,
    )
    if (
        report != expected or claimed != _canonical_sha256(report)
    ):
        raise ValueError("Round 28 AI prompt envelope report differs")
    return {**report, "report_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND28_AI_PROMPT_ENVELOPE_SCHEMA_VERSION",
    "build_round28_ai_prompt_envelope_panel",
    "evaluate_round28_ai_prompt_envelope",
    "validate_round28_ai_prompt_envelope_report",
]
