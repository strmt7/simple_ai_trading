from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"
CONTRACT = RESEARCH / "round-025-v2-condition-salvage-contract-v1.json"
AUDIT = RESEARCH / "round-025-v2-transport-failure-forensic-audit-2026-08-14.json"
FILE_SHA256 = "8ed8b829158bf63f9b0b1efbe60d0fcf8e6148dc5dca51f6922c2f575354bfbc"
CONTRACT_SHA256 = "f46c9c629427ab5e2ce5582bdec9be7f6e67bc8f69831fc27ebde1b1f13eafcb"


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


def test_round25_salvage_contract_is_target_blind_and_source_bound() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    canonical = dict(value)
    claimed = canonical.pop("contract_sha256")
    assert claimed == CONTRACT_SHA256
    assert claimed == _canonical_sha256(canonical)
    audit = json.loads(AUDIT.read_text(encoding="ascii"))
    assert value["parents"][
        "transport_failure_forensic_audit_artifact_sha256"
    ] == audit["artifact_sha256"]
    assert value["status"] == (
        "frozen_target_and_model_score_blind_before_forensic_materialization"
    )
    assert value["truth_state"]["targets_or_resolutions_accessed"] is False
    assert value["truth_state"]["feature_store_created"] is False


def test_round25_salvage_contract_fails_closed_and_grants_no_authority() -> None:
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    condition = value["condition_admission"]
    causal = value["causal_feature_policy"]
    claims = value["claims_and_authority"]
    assert condition["gap_intersection_behavior"] == "reject_entire_condition"
    assert condition["market_snapshot_must_precede_event_start"] is True
    assert condition["partial_condition_rows_allowed"] is False
    assert causal["exact_e18_twap_required"] is True
    assert causal["point_price_substitution_allowed"] is False
    assert causal["binance_price_substitution_allowed"] is False
    assert causal["decision_cadence_ms"] == 250
    assert claims == {
        "ai_uplift_claim_allowed": False,
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "predictive_edge_claim_allowed": False,
        "profitability_claim_allowed": False,
        "purpose": "short_horizon_viability_diagnostic_only",
    }
