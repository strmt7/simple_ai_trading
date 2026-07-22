from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-live-schema-probe-2026-07-22.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_round73_live_schema_probe_reconciles_without_model_claims() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    report = evidence["persisted_report"]

    assert evidence["report_sha256"] == _canonical_sha256(report)
    assert report["run_id"] == evidence["run_id"]
    assert report["design_sha256"] == evidence["design_sha256"]
    assert report["capture_contract_sha256"] == evidence["capture_contract_sha256"]
    assert report["writer_message_count"] == sum(report["event_counts"].values())
    assert report["writer_frame_count"] == evidence["storage_audit"]["frame_count"]
    assert report["audit_passed"] is True
    assert report["qualification_passed"] is False

    segments = {row["symbol"]: row for row in evidence["segments"]}
    assert set(segments) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    for symbol, counts in report["symbol_event_counts"].items():
        segment = segments[symbol]
        assert (
            segment["accepted_depth_updates"]
            + segment["stale_pre_snapshot_depth_updates"]
            == counts["depthUpdate"]
        )
        assert segment["accepted_depth_updates"] == counts["synchronizedDepthUpdate"]
        assert segment["sequence_gap_count"] == 0
        assert segment["crossed_book_count"] == 0
        assert segment["invalid_event_count"] == 0

    semantics = evidence["market_time_semantics"]
    assert "continuous market" in semantics["binance_spot_and_perpetual"]
    assert "actual venue calendars" in semantics["listed_etf_etp_and_futures"]
    assert semantics["listed_product_session_creates_crypto_close"] is False

    assert "forceOrder" not in report["event_counts"]
    assert evidence["observation_limits"]["force_order_messages_observed"] is False
    assert (
        "unknown"
        in evidence["observation_limits"]["liquidation_activity_interpretation"]
    )
    assert all(
        value is False for value in evidence["claims"].values() if value is not True
    )
    assert evidence["claims"]["schema_and_storage_probe"] is True
    supervisor = evidence["supervisor"]
    assert evidence["supervisor_sha256"] == _canonical_sha256(supervisor)
    assert supervisor["selected_run_id"] == evidence["run_id"]
    assert supervisor["attempt_count"] == 1
    assert supervisor["reconnect_count"] == 0
    assert supervisor["attempt_evidence_combined"] is False
    assert supervisor["reconnect_branch_live_fault_injected"] is False
