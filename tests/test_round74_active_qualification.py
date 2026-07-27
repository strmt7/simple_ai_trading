from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.round74_active_qualification import (
    ROUND74_ACTIVE_PREFLIGHT_SHA256,
    _evaluate_operator_result,
    classify_round74_activity,
    inspect_round74_active_readiness,
    load_round74_active_preflight,
    reserve_round74_active_attempt,
    round74_active_window_status,
)


REPOSITORY = Path(__file__).resolve().parents[1]
OPERATOR_CONTRACT = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-active-qualification-operator-v3.json"
)
HOST_SCHEDULE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-active-qualification-host-schedule-v2-2026-07-27.json"
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


def test_round74_active_operator_contract_binds_executable_bytes() -> None:
    contract = json.loads(OPERATOR_CONTRACT.read_text(encoding="utf-8"))
    claimed = contract.pop("artifact_sha256")
    source = contract["source_binding"]

    assert claimed == _canonical_sha256(contract)
    for path_key, hash_key in (
        ("operator_path", "operator_sha256"),
        ("wrapper_path", "wrapper_sha256"),
        ("adjudicator_path", "adjudicator_sha256"),
        ("adjudicator_wrapper_path", "adjudicator_wrapper_sha256"),
    ):
        assert (
            hashlib.sha256((REPOSITORY / source[path_key]).read_bytes()).hexdigest()
            == source[hash_key]
        )
    assert contract["execution_contract"]["automatic_retry_permitted"] is False
    assert contract["authority"]["order_submission"] is False


def test_round74_host_schedule_is_truthful_and_does_not_claim_execution() -> None:
    evidence = json.loads(HOST_SCHEDULE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["preflight_artifact_sha256"] == (ROUND74_ACTIVE_PREFLIGHT_SHA256)
    scheduler = evidence["host_scheduler"]
    assert scheduler["next_run_time_utc"] == "2026-07-27T12:55:00Z"
    assert scheduler["start_when_available"] is False
    assert scheduler["last_task_result_hex"] == "0x41303"
    assert evidence["post_install_readiness"]["capture_executable_passed"] is True
    limitations = evidence["limitations"]
    assert limitations["future_execution_proven_now"] is False
    assert limitations["active_regime_qualification_proven_now"] is False
    assert limitations["automatic_retry_permitted"] is False


def test_round74_active_preflight_binds_corrected_cap_and_exact_command() -> None:
    preflight = load_round74_active_preflight(REPOSITORY)

    assert preflight.artifact_sha256 == ROUND74_ACTIVE_PREFLIGHT_SHA256
    assert preflight.baseline_database_bytes == 9_433_526_272
    assert preflight.baseline_wal_bytes == 0
    assert preflight.database_growth_limit_bytes == 536_870_912
    assert preflight.configured_database_size_cap_bytes == 10_507_268_096
    assert preflight.arguments[-5:] == (
        "--database-size-cap-bytes",
        "10507268096",
        "--progress-interval-seconds",
        "30",
        "--json",
    )


def test_round74_active_readiness_before_window_does_not_reserve() -> None:
    preflight = load_round74_active_preflight(REPOSITORY)
    database_bytes = (
        preflight.database.stat().st_size if preflight.database.is_file() else 0
    )
    if (
        not preflight.executable.is_file()
        or database_bytes != preflight.baseline_database_bytes
    ):
        pytest.skip("Round 74 host-bound capture fixture is unavailable")

    readiness = inspect_round74_active_readiness(
        REPOSITORY,
        now_utc=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        process_provider=list,
    )

    assert readiness["window_status"] == "before_window"
    reservation_exists = Path(str(readiness["reservation_path"])).exists()
    assert readiness["ready_for_window"] is (not reservation_exists)
    assert readiness["can_start_now"] is False
    assert readiness["checks"]["capture_executable_passed"] is True
    assert readiness["checks"]["no_existing_reservation_passed"] is (
        not reservation_exists
    )


def test_round74_active_window_and_activity_boundaries_are_frozen() -> None:
    preflight = load_round74_active_preflight(REPOSITORY)

    assert (
        round74_active_window_status(
            preflight,
            datetime(2026, 7, 27, 12, 55, tzinfo=timezone.utc),
        )
        == "open"
    )
    assert (
        round74_active_window_status(
            preflight,
            datetime(2026, 7, 27, 12, 57, tzinfo=timezone.utc),
        )
        == "open"
    )
    assert (
        round74_active_window_status(
            preflight,
            datetime(2026, 7, 27, 12, 57, 0, 1, tzinfo=timezone.utc),
        )
        == "missed"
    )
    assert (
        classify_round74_activity(
            message_count=735_851,
            elapsed_seconds=1_000.0,
            active_minimum=735.8503256619431,
            quiet_maximum=510.26877174932963,
        )[0]
        == "active"
    )
    assert (
        classify_round74_activity(
            message_count=510_268,
            elapsed_seconds=1_000.0,
            active_minimum=735.8503256619431,
            quiet_maximum=510.26877174932963,
        )[0]
        == "quiet"
    )


def test_round74_attempt_reservation_is_permanent_and_exclusive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt-reservation.json"
    payload = {
        "preflight_sha256": ROUND74_ACTIVE_PREFLIGHT_SHA256,
        "automatic_retry_permitted": False,
    }

    reserve_round74_active_attempt(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        reserve_round74_active_attempt(path, payload)


def test_round74_active_verdict_requires_matching_fresh_audit() -> None:
    preflight = load_round74_active_preflight(REPOSITORY)
    attempt = {
        "schema_version": "round-074-capture-report-v10",
        "run_id": "a" * 32,
        "status": "completed",
        "capture_gate_passed": True,
        "qualification_passed": True,
        "data_qualification_passed": True,
        "resource_safety_passed": True,
        "resource_safety_errors": [],
        "audit_passed": True,
        "audit_errors": [],
        "capture_contract_sha256": (
            "5e245b0f398bb89ca579efcde6acef258fef4efa4204334b9657b43aa9e39cb0"
        ),
        "writer_frame_count": 42,
        "writer_message_count": 2_700_000,
        "writer_compressed_payload_bytes": 9_000_000,
        "elapsed_seconds": 3_600.0,
        "process_io_delta_write_bytes": 1_000_000,
    }
    report_sha256 = hashlib.sha256(
        json.dumps(
            attempt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    supervisor = {
        "schema_version": "round-074-capture-supervisor-report-v1",
        "capture_schema_version": "round-074-prospective-evidence-v10",
        "capture_contract_sha256": (
            "5e245b0f398bb89ca579efcde6acef258fef4efa4204334b9657b43aa9e39cb0"
        ),
        "design_sha256": (
            "b00e20499a0025c05cb27cc352d9444ce722493b5bdb592d628224343e81e136"
        ),
        "attempt_count": 1,
        "reconnect_count": 0,
        "reconnect_delays_seconds": [],
        "attempt_evidence_combined": False,
        "attempts": [attempt],
    }
    audits = [
        {
            "run_id": "a" * 32,
            "return_code": 0,
            "audit": {
                "passed": True,
                "errors": [],
                "run_id": "a" * 32,
                "run_status": "completed",
                "capture_contract_sha256": (
                    "5e245b0f398bb89ca579efcde6acef258fef4efa4204334b9657b43aa9e39cb0"
                ),
                "stored_report_schema_version": "round-074-capture-report-v10",
                "stored_report_sha256": report_sha256,
                "frame_count": 42,
                "message_count": 2_700_000,
                "compressed_payload_bytes": 9_000_000,
                "last_frame_sha256": "b" * 64,
            },
        }
    ]

    result = _evaluate_operator_result(
        preflight,
        capture_return_code=0,
        supervisor=supervisor,
        supervisor_parse_error="",
        audits=audits,
        watchdog_breaches=[],
    )

    assert result["outcome"] == "active_qualified"
    assert result["active_qualified"] is True
    assert result["errors"] == []

    audits[0]["audit"]["frame_count"] = 41
    rejected = _evaluate_operator_result(
        preflight,
        capture_return_code=0,
        supervisor=supervisor,
        supervisor_parse_error="",
        audits=audits,
        watchdog_breaches=[],
    )
    assert rejected["active_qualified"] is False
    assert rejected["errors"] == ["fresh_audit_or_report_identity_failed"]


def test_round74_retained_result_adjudicates_without_mutation() -> None:
    from simple_ai_trading.round74_active_adjudication import (
        ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH,
        build_round74_active_adjudication,
    )

    source_path = REPOSITORY / ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()

    artifact = build_round74_active_adjudication(
        REPOSITORY,
        adjudicated_at_utc=datetime(2026, 7, 27, 14, 5, tzinfo=timezone.utc),
    )

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
    assert artifact["defect"]["fresh_audit_stored_report_sha256_matches"] is True
    assert artifact["adjudication"]["capture_retried"] is False
    assert (
        artifact["adjudication"]["corrected_verdict"]["outcome"] == "active_qualified"
    )
    claimed = artifact.pop("artifact_sha256")
    assert claimed == _canonical_sha256(artifact)
