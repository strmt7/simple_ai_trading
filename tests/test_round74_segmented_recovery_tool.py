from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.build_round74_segmented_recovery_outcomes as subject
from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS,
    load_round74_segmented_cohort_plan,
)


_REPOSITORY = Path(__file__).resolve().parents[1]
_PLAN = (
    _REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-segmented-event-cohort-plan-v3.json"
)


def test_recovery_tool_refuses_to_classify_slots_before_campaign_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    state = repository / "state"
    state.mkdir()
    output = repository / "recovery"
    plan = load_round74_segmented_cohort_plan(_PLAN.read_text(encoding="utf-8"))
    campaign_terminal_wall_ns = (
        plan.slot(plan.total_slots - 1).scheduled_end_wall_ns
        + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
    )
    monkeypatch.setattr(subject.time, "time_ns", lambda: campaign_terminal_wall_ns - 1)

    result = subject.main(
        [
            "--repository",
            str(repository),
            "--plan",
            str(_PLAN),
            "--state-root",
            str(state),
            "--output",
            str(output),
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert "recovery outcome differs" in payload["error"]
    assert payload["admitted_data_created"] is False
    assert payload["database_opened"] is False
    assert payload["trading_authority"] is False


def test_resultless_captured_run_cannot_be_downgraded_without_database(
    tmp_path: Path,
) -> None:
    slot = tmp_path / "slot-017"
    slot.mkdir()
    for name in ("reservation.json", "state.json"):
        (slot / name).write_text("{}", encoding="utf-8")
    (slot / "capture.stderr.log").write_text("", encoding="utf-8")
    run_id = "a" * 32
    (slot / "capture.stdout.json").write_text(
        json.dumps({"attempts": [{"run_id": run_id}]}),
        encoding="utf-8",
    )

    supervisor = subject._resultless_supervisor(slot)

    assert supervisor == {"attempts": [{"run_id": run_id}]}
    with pytest.raises(ValueError, match="require --database shards"):
        subject._route_captured_runs((), (run_id,))


def test_resultless_supervisor_accepts_prior_recovery_evidence(
    tmp_path: Path,
) -> None:
    slot = tmp_path / "slot-014"
    slot.mkdir()
    for name in ("reservation.json", "state.json"):
        (slot / name).write_text("{}", encoding="utf-8")
    (slot / "capture.stderr.log").write_text("", encoding="utf-8")
    (slot / "recovery-audit-error.log").write_text("audit failed", encoding="utf-8")
    run_id = "b" * 32
    (slot / "capture.stdout.json").write_text(
        json.dumps({"attempts": [{"run_id": run_id}]}),
        encoding="utf-8",
    )

    assert subject._resultless_supervisor(slot) == {"attempts": [{"run_id": run_id}]}


@pytest.mark.parametrize(
    ("run_status", "run_error", "report", "expected"),
    [
        (
            "completed",
            "",
            {
                "status": "completed",
                "failure_class": "none",
                "error": "",
                "capture_gate_passed": True,
                "data_qualification_passed": True,
                "resource_safety_passed": True,
                "storage_efficiency_passed": True,
                "audit_passed": True,
                "audit_errors": [],
                "resource_safety_errors": [],
                "payload_cap_reached": False,
                "database_size_cap_reached": False,
            },
            None,
        ),
        ("running", "", None, "stored_report_missing"),
        (
            "failed",
            "writer fault",
            {
                "status": "failed",
                "failure_class": "writer",
                "error": "writer fault",
                "capture_gate_passed": False,
                "data_qualification_passed": False,
            },
            "unsupported_terminal_class",
        ),
    ],
)
def test_terminal_audit_preflight_is_conservative(
    run_status: str,
    run_error: str,
    report: dict[str, object] | None,
    expected: str | None,
) -> None:
    assert (
        subject._terminal_audit_preflight_reason(run_status, run_error, report)
        == expected
    )


def test_recovery_tool_resumes_one_immutable_partial_panel_timestamp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    state = repository / "state"
    state.mkdir()
    output = repository / "recovery"
    plan = load_round74_segmented_cohort_plan(_PLAN.read_text(encoding="utf-8"))
    terminal = (
        plan.slot(plan.total_slots - 1).scheduled_end_wall_ns
        + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
    )
    arguments = [
        "--repository",
        str(repository),
        "--plan",
        str(_PLAN),
        "--state-root",
        str(state),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(subject.time, "time_ns", lambda: terminal)
    assert subject.main(arguments) == 0
    capsys.readouterr()
    first = json.loads((output / "000.json").read_text(encoding="utf-8"))
    (output / "017.json").unlink()

    rebuilt_ordinals: list[int] = []
    original_builder = subject.build_round74_segmented_recovery_outcome

    def tracked_builder(*args: object, **kwargs: object) -> object:
        rebuilt_ordinals.append(int(kwargs["slot_ordinal"]))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        subject, "build_round74_segmented_recovery_outcome", tracked_builder
    )
    monkeypatch.setattr(subject.time, "time_ns", lambda: terminal + 1)
    assert subject.main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    restored = json.loads((output / "017.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == (
        subject.ROUND74_SEGMENTED_RECOVERY_BUILD_SCHEMA_VERSION
    )
    assert payload["recovery_count"] == plan.total_slots
    assert rebuilt_ordinals == [17]
    assert restored["observed_wall_ns"] == first["observed_wall_ns"] == terminal


def test_partial_recovery_panel_rejects_identity_and_timestamp_mixing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recovery"
    output.mkdir()
    plan_sha256 = "a" * 64
    (output / "000.json").write_text(
        json.dumps({"plan_sha256": "b" * 64, "observed_wall_ns": 100}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="partial recovery identity differs"):
        subject._recovery_observed_wall_ns(
            output,
            plan_sha256=plan_sha256,
            total_slots=720,
            fallback_wall_ns=200,
        )

    (output / "000.json").write_text(
        json.dumps({"plan_sha256": plan_sha256, "observed_wall_ns": 100}),
        encoding="utf-8",
    )
    (output / "001.json").write_text(
        json.dumps({"plan_sha256": plan_sha256, "observed_wall_ns": 101}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observation time differs"):
        subject._recovery_observed_wall_ns(
            output,
            plan_sha256=plan_sha256,
            total_slots=720,
            fallback_wall_ns=200,
        )
