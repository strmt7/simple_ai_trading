from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.impact_absorption import (
    ROUND74_CAPTURE_DESIGN_SHA256,
)
from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS,
    Round74SegmentedCohortPlan,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)
import simple_ai_trading.round74_segmented_development_inputs as subject
from simple_ai_trading.round74_segmented_cohort_operator import (
    adjudicate_round74_segmented_supervisor,
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


def _plan() -> Round74SegmentedCohortPlan:
    return Round74SegmentedCohortPlan(
        scheduled_start_wall_ns=2_000_000_000_000_000_000,
        implementation_git_commit="c" * 40,
        prerequisite_artifact_sha256="a" * 64,
        prerequisite_window_start_wall_ns=1_999_990_000_000_000_000,
        prerequisite_window_end_wall_ns=1_999_995_000_000_000_000,
    )


def _terminal_wall_ns(plan: Round74SegmentedCohortPlan) -> int:
    return (
        plan.slot(plan.total_slots - 1).scheduled_end_wall_ns
        + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
    )


def _startup_supervisor() -> dict[str, object]:
    error = "startup:TimeoutError:opening handshake timed out"
    return {
        "schema_version": "round-074-capture-supervisor-report-v1",
        "design_sha256": ROUND74_CAPTURE_DESIGN_SHA256,
        "capture_schema_version": IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        "capture_contract_sha256": IMPACT_CAPTURE_V10_CONTRACT_SHA256,
        "status": "failed",
        "qualification_passed": False,
        "selected_run_id": "",
        "attempt_count": 1,
        "reconnect_count": 0,
        "reconnect_delays_seconds": [],
        "attempts": [],
        "startup_errors": [error],
        "terminal_error": error,
        "attempt_evidence_combined": False,
    }


def _write_terminal_slot(
    root: Path,
    plan: Round74SegmentedCohortPlan,
    *,
    slot_ordinal: int,
) -> tuple[Path, str]:
    slot = plan.slot(slot_ordinal)
    directory = root / f"slot-{slot_ordinal:03d}"
    directory.mkdir()
    supervisor = _startup_supervisor()
    stdout = json.dumps(supervisor, indent=2, sort_keys=True).encode("utf-8")
    stderr = b"capture failed before opening transport\n"
    (directory / "capture.stdout.json").write_bytes(stdout)
    (directory / "capture.stderr.log").write_bytes(stderr)
    adjudication = adjudicate_round74_segmented_supervisor(
        plan,
        slot_ordinal=slot_ordinal,
        supervisor_payload=supervisor,
        epoch_audit=None,
    )
    reservation: dict[str, object] = {
        "schema_version": "round-074-segmented-slot-reservation-v1",
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": slot.ordinal,
        "role": slot.role,
        "scheduled_start_wall_ns": slot.scheduled_start_wall_ns,
        "reserved_wall_ns": slot.scheduled_start_wall_ns + 1,
        "command_sha256": "b" * 64,
        "automatic_retry_permitted": False,
        "credentials_used": False,
        "orders_submitted": False,
    }
    reservation["reservation_sha256"] = _canonical_sha256(reservation)
    (directory / "reservation.json").write_text(
        json.dumps(reservation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result: dict[str, object] = {
        "schema_version": "round-074-segmented-campaign-slot-result-v1",
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": slot.ordinal,
        "role": slot.role,
        "reservation_sha256": reservation["reservation_sha256"],
        "capture_return_code": 2,
        "capture_stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "capture_stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "monitor_sample_count": 1,
        "maximum_observed_slot_growth_bytes": 0,
        "watchdog_breaches": [],
        "adjudication": adjudication.as_dict(),
        "automatic_retry_permitted": False,
        "credentials_used": False,
        "orders_submitted": False,
        "profitability_or_edge_claim": False,
        "trading_authority": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    state = {
        "schema_version": "round-074-segmented-slot-state-v1",
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": slot.ordinal,
        "phase": "terminal",
        "completed_at_utc": "2033-05-18T04:33:20Z",
        "result_sha256": result["result_sha256"],
    }
    (directory / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return directory, str(result["result_sha256"])


def test_segmented_terminal_slot_result_reloads_all_bound_evidence(
    tmp_path: Path,
) -> None:
    plan = _plan()
    directory, result_sha256 = _write_terminal_slot(
        tmp_path,
        plan,
        slot_ordinal=0,
    )

    outcome, observed_sha256 = subject._load_campaign_slot_result(
        plan,
        slot_ordinal=0,
        slot_directory=directory,
    )

    assert observed_sha256 == result_sha256
    assert outcome.status == "transport_excluded"
    assert outcome.reason_code == "startup_transport"
    (directory / "capture.stderr.log").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="terminal slot binding differs"):
        subject._load_campaign_slot_result(
            plan,
            slot_ordinal=0,
            slot_directory=directory,
        )


def test_segmented_recovery_is_terminal_only_and_hashes_every_surviving_file(
    tmp_path: Path,
) -> None:
    plan = _plan()
    directory = tmp_path / "slot-017"
    directory.mkdir()
    (directory / "reservation.json").write_bytes(b"reserved")
    (directory / "state.json").write_bytes(b"incomplete")
    terminal = _terminal_wall_ns(plan)

    with pytest.raises(ValueError, match="recovery outcome differs"):
        subject.build_round74_segmented_recovery_outcome(
            plan,
            slot_ordinal=17,
            observed_wall_ns=terminal - 1,
            slot_directory=directory,
        )
    recovery = subject.build_round74_segmented_recovery_outcome(
        plan,
        slot_ordinal=17,
        observed_wall_ns=terminal,
        slot_directory=directory,
    )
    restored = subject.Round74SegmentedRecoveryOutcome.from_dict(
        plan,
        recovery.as_dict(),
    )

    restored.verify_slot_directory(directory)
    assert restored.recovery_sha256 == recovery.recovery_sha256
    assert restored.outcome.status == "missed"
    assert restored.outcome.binding is None
    (directory / "state.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="slot evidence differs"):
        restored.verify_slot_directory(directory)
    (directory / "result.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="recovery rejects a result"):
        subject.build_round74_segmented_recovery_outcome(
            plan,
            slot_ordinal=17,
            observed_wall_ns=terminal,
            slot_directory=directory,
        )
