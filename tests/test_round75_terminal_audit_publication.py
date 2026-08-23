from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/model-research/action-value"
AUDIT = EVIDENCE / "round-075-terminal-campaign-audit-2026-08-23.json"
RECOVERY = EVIDENCE / "round-075-wal-copy-recovery-2026-08-23.json"
AMENDMENT = EVIDENCE / "round-075-post-campaign-amendment-v1.json"
FROZEN = EVIDENCE / "round-075-frozen-v4-source"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("artifact_sha256")
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert claimed == hashlib.sha256(canonical).hexdigest()
    value["artifact_sha256"] = claimed
    return value


def test_round75_terminal_audit_rejects_incomplete_capture() -> None:
    report = _bound(AUDIT)
    coverage = report["coverage"]
    roles = coverage["role_counts"]
    assert report["schema_version"] == "round-075-terminal-audit-v1"
    assert report["status"] == "rejected_incomplete_campaign"
    assert coverage["total_predeclared_slots"] == 720
    assert coverage["terminal_result_count"] == 35
    assert coverage["result_status_counts"] == {
        "admitted": 33,
        "transport_excluded": 2,
    }
    assert coverage["missed_receipt_count"] == 684
    assert coverage["incomplete_slot_ordinals"] == [67]
    assert coverage["uncovered_slot_ordinals"] == []
    assert coverage["all_slots_have_terminal_dispositions"] is False
    assert roles["training"]["admitted"] == 33
    assert roles["training"]["observed_raw_eligible_anchor_ns"] == 28_903_469_878_300
    assert roles["training"]["required_eligible_anchor_ns"] == 394_740_000_000_000
    assert roles["tuning"]["admitted"] == 0
    assert roles["test"]["admitted"] == 0
    assert all(role["eligible_anchor_quota_passed"] is False for role in roles.values())
    assert report["storage"]["wal_bytes"] == 9_431_058
    assert report["storage"]["all_campaign_wals_absent"] is False
    assert report["storage"]["source_databases_opened"] is False
    assert report["supervisor"]["classification"] == "campaign_terminal"
    assert report["supervisor"]["inspection_passed"] is True
    assert report["supervisor"]["automatic_start_permitted"] is False
    assert all(value is False for value in report["gates"].values())
    assert all(value is False for value in report["scope"].values())


def test_round75_terminal_sources_and_recovery_are_hash_bound() -> None:
    report = _bound(AUDIT)
    recovery = _bound(RECOVERY)
    sources = report["sources"]
    assert sources["contract_file_sha256"] == _sha256(
        EVIDENCE / "round-075-continuous-capture-contract-v4.json"
    )
    assert sources["activation_file_sha256"] == _sha256(
        EVIDENCE / "round-075-v4-host-activation-receipt-2026-08-10.json"
    )
    assert sources["plan_file_sha256"] == _sha256(
        EVIDENCE / "round-075-prospective-event-cohort-plan-v1.json"
    )
    for source_path, expected in sources["frozen_implementation_file_sha256"].items():
        assert _sha256(FROZEN / f"{source_path}.txt") == expected
    wal = report["storage"]["wal_files"]
    shard = next(
        row
        for row in report["storage"]["database_files"]
        if row["name"].endswith("shard-002.duckdb")
    )
    assert len(wal) == 1
    assert recovery["source"]["database_sha256"] == shard["sha256"]
    assert recovery["source"]["wal_sha256"] == wal[0]["sha256"]
    assert recovery["row_count_delta_after_wal_recovery"] == {
        "main.impact_capture_frame_v10": 196,
        "main.impact_capture_lane_state": 3,
        "main.impact_capture_report": 0,
        "main.impact_capture_run": 0,
        "main.impact_capture_segment": 0,
        "main.impact_rest_event_v10": 62,
    }
    assert recovery["admissibility"]["wal_payload_admitted"] is False
    assert recovery["method"]["original_database_opened"] is False
    assert recovery["method"]["original_wal_replayed"] is False
    source = (ROOT / "src/simple_ai_trading/round75_terminal_audit.py").read_text(
        encoding="utf-8"
    )
    assert "import duckdb" not in source


def test_round75_post_campaign_amendment_preserves_frozen_source() -> None:
    amendment = _bound(AMENDMENT)
    audit = _bound(AUDIT)
    recovery = _bound(RECOVERY)
    assert amendment["schema_version"] == "round-075-post-campaign-amendment-v1"
    assert (
        amendment["terminal_evidence"]["audit_artifact_sha256"]
        == audit["artifact_sha256"]
    )
    assert amendment["terminal_evidence"]["audit_file_sha256"] == _sha256(AUDIT)
    assert (
        amendment["terminal_evidence"]["recovery_artifact_sha256"]
        == recovery["artifact_sha256"]
    )
    assert amendment["terminal_evidence"]["recovery_file_sha256"] == _sha256(RECOVERY)
    for source_path, record in amendment["frozen_v4_source_mirror"].items():
        assert _sha256(ROOT / record["mirror_path"]) == record["sha256"]
        assert (
            audit["sources"]["frozen_implementation_file_sha256"][source_path]
            == (record["sha256"])
        )
    for source_path, expected in amendment["active_implementation"].items():
        assert _sha256(ROOT / source_path) == expected
    assert amendment["post_campaign_changes"] == {
        "campaign_terminal_is_a_non_restartable_supervisor_state": True,
        "campaign_terminal_process_presence_fails_closed_without_automatic_termination": True,
        "expired_campaign_missing_service_is_not_restarted": True,
        "original_database_or_wal_modification_permitted": False,
        "supervisor_schema_version": "round-075-capture-supervisor-v3",
        "terminal_audit_opens_source_databases": False,
    }
    assert all(value is False for value in amendment["scope"].values())
