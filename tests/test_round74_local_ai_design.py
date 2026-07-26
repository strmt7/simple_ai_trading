from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption_ai_protocol import (
    ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_HORIZONS_SECONDS,
    ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-review-design-v1.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def test_round74_local_ai_design_is_source_bound_and_fail_closed() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    source = artifact["source_binding"]
    assert source["sha256"] == hashlib.sha256(
        (REPOSITORY / source["path"]).read_bytes()
    ).hexdigest()
    assert source["model_manifest_schema_version"] == (
        ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION
    )
    assert source["review_request_schema_version"] == (
        ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
    )
    assert source["review_decision_schema_version"] == (
        ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION
    )
    architecture = artifact["architecture"]
    assert architecture["supported_review_horizons_seconds"] == list(
        ROUND74_AI_REVIEW_HORIZONS_SECONDS
    )
    assert architecture["event_by_event_execution_loop_member"] is False
    assert architecture["one_and_five_second_ml_paths_wait_for_ai"] is False
    assert architecture["absolute_date_exposed_to_ai"] is False
    assert architecture["real_symbol_exposed_to_ai"] is False
    assert architecture["future_outcome_exposed_to_ai"] is False
    for key in (
        "ai_may_create_trade_side",
        "ai_may_increase_risk",
        "ai_may_set_leverage",
        "ai_may_submit_cancel_or_close_orders",
    ):
        assert architecture[key] is False
    assert (
        architecture[
            "deterministic_data_execution_and_risk_gates_remain_authoritative"
        ]
        is True
    )


def test_round74_local_ai_candidates_are_pinned_but_unpromoted() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    candidates = artifact["candidate_preselection"]
    finance = candidates["finance_candidate"]
    control = candidates["general_control"]

    assert finance["model_id"] == "TheFinAI/Fino1-8B"
    assert len(finance["repository_revision"]) == 40
    assert finance["license_id"] == "llama3.1"
    assert finance["trading_edge_established"] is False
    assert finance["pinned_quantized_artifact_ready"] is False
    assert control["model_id"] == "Qwen/Qwen3-8B"
    assert len(control["repository_revision"]) == 40
    assert control["license_id"] == "Apache-2.0"
    assert (
        control["pinned_onnx_conversion"][
            "int4_external_data_observed_bytes"
        ]
        == 6_076_825_600
    )
    assert control["trading_edge_established"] is False
    assert candidates["finance_brand_or_parameter_count_implies_edge"] is False
    assert (
        candidates["promotion_requires_our_paired_sealed_market_evidence"]
        is True
    )

    status = artifact["status"]
    assert status["protocol_implemented"] is True
    for key in (
        "candidate_weights_downloaded",
        "candidate_weight_hash_verified",
        "actual_multibillion_parameter_inference_completed",
        "host_latency_preflight_completed",
        "representative_market_ai_evaluation_completed",
        "ai_uplift_established",
        "financial_edge_established",
        "profitability_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    ):
        assert status[key] is False


def test_round74_local_ai_evaluation_cannot_win_by_all_veto() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    evaluation = artifact["paired_evaluation_contract"]

    assert evaluation["same_events_targets_costs_and_timing"] is True
    assert evaluation["historical_symbol_and_absolute_date_masked"] is True
    assert evaluation["ai_can_only_preserve_reduce_veto_or_abstain"] is True
    assert evaluation["sealed_test_used_once_after_ai_policy_freeze"] is True
    assert evaluation["accuracy_ignored"] is False
    assert evaluation["profitability_required_by_data_filter_or_test_assertion"] is False
    assert (
        "enough remaining executable trades to avoid trivial all-veto behavior"
        in evaluation["promotion_requires"]
    )
