from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "docs/model-research/action-value/"
    "round-074-terminal-campaign-outcome-2026-08-10.json"
)
PLAN = (
    ROOT / "docs/model-research/action-value/"
    "round-074-segmented-event-cohort-plan-v3.json"
)
FILE_SHA256 = "744d0231c9c3d309a914b622caa49bc6fa3170aad0f43e0fffd42fb1315b9bdf"
ARTIFACT_SHA256 = "67cba6caf728d0ae2d0271e290d1dd7fe9ab4a7fa35f861683ebf16f080a2612"


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


def test_round74_terminal_campaign_outcome_is_hash_bound_and_fail_closed() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    claimed = value.pop("artifact_sha256")
    assert claimed == ARTIFACT_SHA256
    assert claimed == _canonical_sha256(value)
    assert value["schema_version"] == "round-074-terminal-campaign-outcome-v1"
    assert value["decision"] == {
        "model_data_eligible": False,
        "predictive_edge_established": False,
        "profitability_established": False,
        "reason": "training_eligible_anchor_quota_failed",
        "representative_training_performed": False,
        "sealed_target_manifests_read": False,
        "sealed_test_access_reserved_or_consumed": False,
        "status": "campaign_cannot_qualify_model",
    }
    assert value["scope"] == {
        "credentials_used": False,
        "live_trading_authority": False,
        "orders_submitted": False,
        "paper_trading_authority": False,
        "source_database_access": "not_opened",
    }


def test_round74_terminal_coverage_reconciles_without_local_paths() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    coverage = value["terminal_coverage"]
    assert coverage["total_slots"] == 720
    assert coverage["slot_evidence_kind_counts"] == {
        "late_adjudication": 7,
        "recovery": 239,
        "result": 474,
    }
    assert coverage["outcome_status_counts"] == {
        "admitted": 459,
        "missed": 239,
        "transport_excluded": 22,
    }
    quotas = coverage["role_quotas"]
    assert sum(row["planned_slot_count"] for row in quotas.values()) == 720
    assert sum(row["admitted_count"] for row in quotas.values()) == 459
    assert quotas["training"]["quota_passed"] is False
    assert quotas["tuning"]["quota_passed"] is True
    assert quotas["test"]["quota_passed"] is True
    assert quotas["training"]["deficit_eligible_anchor_ns"] == 162_322_767_182_900
    sources = value["source_bindings"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == sources["plan_file_sha256"]
    assert sources["recovery_build_location"] == "local_data_artifact_not_committed"
    assert Path(sources["recovery_build_path"]).name == sources["recovery_build_path"]
    assert not any(Path(text).is_absolute() for text in _strings(value))
