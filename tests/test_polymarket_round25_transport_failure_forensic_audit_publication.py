from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-v2-transport-failure-forensic-audit-2026-08-14.json"
)
FILE_SHA256 = "91268c2964f49dcdeaff84426ab396815fab842e0c070808858030c495318578"
ARTIFACT_SHA256 = "8ee546844fada87ab4a542f6620bc5e83654b635b6a72145a338c02431c41276"


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


def _strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return [value] if isinstance(value, str) else []


def test_round25_transport_failure_audit_is_self_hashed_and_source_bound() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(ARTIFACT.read_text(encoding="ascii"))
    canonical = dict(value)
    claimed = canonical.pop("artifact_sha256")
    assert claimed == ARTIFACT_SHA256
    assert claimed == _canonical_sha256(canonical)
    assert value["status"] == (
        "forensic_audit_passed_capture_remains_globally_failed"
    )
    source = value["source_bindings"]
    assert source["run_id"] == "f96a24bdaa2d4f5f8cdad3f06193a0ce"
    assert source["recorder_report_sha256"] == (
        "45a613588f15ef45f57c51931d1b01e19f7cb7b0d6b08e7be0b0dd5b8d49631d"
    )
    assert source["evidence_manifest_sha256"] == (
        "9d45a88606239646302af3d0c8b3c1ffdbf824ea3836b73418bdaaad35f045d9"
    )
    assert not any(Path(text).is_absolute() for text in _strings(value))


def test_round25_transport_failure_audit_does_not_overclaim() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="ascii"))
    audit = value["deep_read_only_audit"]
    assert audit["raw_messages_verified"] == 13_816_367
    assert audit["decoded_events_verified"] == 13_813_224
    assert audit["out_of_window_message_count"] == 0
    assert audit["integrity_errors"] == []
    assert audit["database_mutated"] is False
    assert value["terminal_failure"]["global_run_status_changed"] is False
    assert value["forensic_replay_boundary"] == {
        "condition_features_materialized": False,
        "condition_population_defined": False,
        "gap_ledger_required": True,
        "hash_checked_receipts_only": True,
        "model_scores_consulted": False,
        "non_retryable_failure_accepted": False,
        "qualified_transport_failure_only": True,
        "requires_separate_target_blind_salvage_contract": True,
        "source_database_path_published": False,
        "source_database_read_only": True,
        "targets_or_resolutions_accessed": False,
    }
    assert all(value["claims_and_authority"][key] is False for key in (
        "predictive_accuracy_established",
        "financial_edge_established",
        "after_cost_profitability_established",
        "ai_uplift_established",
        "model_data_eligibility_granted",
        "paper_trading_authority",
        "live_trading_authority",
        "orders_submitted",
        "credentials_used",
    ))
