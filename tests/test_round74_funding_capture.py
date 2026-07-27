from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, urlsplit

from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_funding_schedule_evidence_claims,
)
from tools._round74_public_evidence_capture import canonical_sha256
from tools.capture_round74_funding_evidence import (
    ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION,
    ROUND74_FUNDING_LIMIT,
    ROUND74_FUNDING_URL,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-funding-evidence-quiet-run-2026-07-27.json"
)


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def test_round74_funding_capture_binds_complete_empty_responses() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("artifact_sha256")
    assert claimed == canonical_sha256(artifact)
    assert artifact["schema_version"] == ROUND74_FUNDING_CAPTURE_SCHEMA_VERSION
    assert artifact["database_open_mode"] == "read_only"

    capture = artifact["capture_binding"]
    assert capture["run_id"] == "892d1a3b74634d14b7db4d143f810079"
    assert capture["fresh_full_run_audit_passed"] is True
    assert capture["fresh_full_run_audit_errors"] == []
    assert capture["frame_count"] == 872
    assert capture["message_count"] == 881_173
    capture_path = REPOSITORY / capture["capture_artifact_path"]
    assert capture["capture_artifact_file_sha256"] == hashlib.sha256(
        capture_path.read_bytes()
    ).hexdigest()

    clock = artifact["clock_binding"]
    assert clock["probe_count"] == 61
    assert clock["interpolation_permitted"] is False
    assert clock["last_exchange_time_ms"] > clock["first_exchange_time_ms"]
    assert len(clock["probe_panel_sha256"]) == 64

    requests = artifact["requests"]
    assert [request["symbol"] for request in requests] == list(
        ROUND74_EVENT_TARGET_SYMBOLS
    )
    for request in requests:
        split = urlsplit(request["url"])
        query = parse_qs(split.query)
        assert f"{split.scheme}://{split.netloc}{split.path}" == (
            ROUND74_FUNDING_URL
        )
        assert query["symbol"] == [request["symbol"]]
        assert query["startTime"] == [
            str(clock["first_exchange_time_ms"])
        ]
        assert query["endTime"] == [str(clock["last_exchange_time_ms"])]
        assert query["limit"] == [str(ROUND74_FUNDING_LIMIT)]
        assert request["response_body_bytes"] == 2
        assert request["response_row_count"] == 0
        assert request["retry_count"] == 0
        assert request["credential_material_sent"] is False
        assert request["raw_payload_persisted"] is False

    assert artifact["funding_row_count_by_symbol"] == {
        symbol: 0 for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    boundaries = {
        symbol: tuple(tuple(interval) for interval in intervals)
        for symbol, intervals in (
            artifact["funding_boundary_intervals_monotonic_ns"].items()
        )
    }
    coverage = {
        symbol: tuple(interval)
        for symbol, interval in (
            artifact["funding_schedule_coverage_monotonic_ns"].items()
        )
    }
    assert boundaries == {
        symbol: () for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    evidence = Round74EventTargetEvidence.from_dict(
        artifact["target_evidence"]
    )
    assert evidence.record_count == 3
    assert evidence.binds(
        round74_funding_schedule_evidence_claims(
            funding_boundary_intervals_monotonic_ns=boundaries,
            funding_schedule_coverage_monotonic_ns=coverage,
        )
    )

    execution_commit = artifact["execution_git_commit"]
    source = artifact["source_binding"]
    for label in ("parser", "capture_tool", "transport_helper"):
        assert source[f"{label}_sha256"] == _git_blob_sha256(
            execution_commit,
            source[f"{label}_path"],
        )
    assert artifact["scope"]["capture_run_is_design_consumed"] is True
    assert (
        artifact["scope"]["capture_run_may_be_used_for_financial_evaluation"]
        is False
    )
    assert all(value is False for value in artifact["authority"].values())
