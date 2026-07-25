from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)


RESEARCH = Path("docs/model-research/action-value")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_round73_v9_campaign_invalidation_is_hash_bound_and_pre_model() -> None:
    artifact = json.loads(
        (RESEARCH / "round-073-v9-corpus-invalidation-2026-07-25.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    assert artifact["campaign"]["qualified_and_indexed_hours_before_invalidation"] == 18
    assert artifact["campaign"]["completed_but_resource_rejected_hours"] == 2
    assert artifact["campaign"]["target_rows_opened"] is False
    assert len(artifact["accepted_completed_runs"]) == 18
    assert len(artifact["completed_resource_rejections"]) == 2
    assert all(
        row["audit_passed"] is True and row["capture_error"] == ""
        for row in artifact["completed_resource_rejections"]
    )
    decision = artifact["decision"]
    assert decision["all_18_indexed_v9_hours_eligible_for_modeling"] is False
    assert decision["resume_v9_rotation_permitted"] is False
    assert not any(artifact["authority"].values())


def test_round74_design_and_v10_contract_are_hash_bound_and_nonselective() -> None:
    design = json.loads(
        (RESEARCH / "round-074-capture-recovery-design-v1.json").read_text(
            encoding="utf-8"
        )
    )
    claimed_design = design.pop("design_sha256")
    contract = json.loads(
        (RESEARCH / "round-074-capture-contract-v10.json").read_text(encoding="utf-8")
    )
    claimed_contract = contract.pop("capture_contract_sha256")

    assert claimed_design == _canonical_sha256(design) == ROUND74_CAPTURE_DESIGN_SHA256
    assert (
        claimed_contract
        == _canonical_sha256(contract)
        == IMPACT_CAPTURE_V10_CONTRACT_SHA256
    )
    storage = contract["storage_schema_v10"]
    assert storage["run_schema"] == IMPACT_CAPTURE_V10_SCHEMA_VERSION
    assert storage["report_schema"] == IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
    resource = contract["host_resource_safety_v10"]
    assert resource["message_count_used_in_resource_verdict"] is False
    assert resource["bytes_per_message_retained_as_telemetry_only"] is True
    assert (
        resource["resource_failure_campaign_policy"]
        == "halt and require review; never skip the market interval and continue"
    )
    heartbeat = contract["websocket_heartbeat_v10"]
    assert heartbeat["automatic_pong_handling_required"] is True
    assert heartbeat["client_originated_keepalive_ping_interval"] is None
    calendar = contract["market_and_calendar_scope"]
    assert calendar["crypto_formal_daily_close"] is False
    assert calendar["listed_product_close_creates_crypto_close"] is False
    assert (
        calendar["listed_product_calendar_may_grant_crypto_execution_authority"]
        is False
    )
    assert contract["authorization"]["round_074_model_training_or_evaluation"] is False
