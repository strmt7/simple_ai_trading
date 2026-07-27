from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.round74_active_qualification import (
    ROUND74_ACTIVE_PREFLIGHT_SHA256,
)
from simple_ai_trading.round74_event_cohort_operator import (
    ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS,
    ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES,
    ROUND74_EVENT_COHORT_PLAN_SHA256,
    ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES,
    _load_contiguous_bindings,
    load_round74_cohort_operator_plan,
    select_round74_cohort_slot,
    validate_round74_active_prerequisite,
)
from simple_ai_trading.storage import write_json_atomic


REPOSITORY = Path(__file__).resolve().parents[1]
OPERATOR_CONTRACT = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-operator-v4.json"
)
HOST_SCHEDULE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-host-schedule-v4-2026-07-27.json"
)
V1_SUPERSESSION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-v1-supersession-2026-07-27.json"
)
V2_SUPERSESSION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-v2-supersession-2026-07-27.json"
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


def test_round74_cohort_operator_contract_binds_executable_bytes() -> None:
    contract = json.loads(OPERATOR_CONTRACT.read_text(encoding="utf-8"))
    claimed = contract.pop("artifact_sha256")
    source = contract["source_binding"]

    assert claimed == _canonical_sha256(contract)
    for path_key, hash_key in (
        ("operator_path", "operator_sha256"),
        ("wrapper_path", "wrapper_sha256"),
    ):
        assert hashlib.sha256(
            (REPOSITORY / source[path_key]).read_bytes()
        ).hexdigest() == source[hash_key]
    execution = contract["slot_execution_contract"]
    assert execution["automatic_retry_permitted"] is False
    assert execution["partition_written_only_after_all_168_bindings"] is True
    partition = contract["partition_contract"]
    assert partition["maximum_target_span_ns"] == 310_500_000_000
    assert partition["minimum_purge_ns"] == 310_500_000_000
    assert partition["minimum_embargo_ns"] == 310_500_000_000


def test_round74_cohort_host_schedule_is_exact_and_pre_execution_only() -> None:
    evidence = json.loads(HOST_SCHEDULE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == ROUND74_EVENT_COHORT_PLAN_SHA256
    scheduler = evidence["host_scheduler"]
    assert scheduler["task_name"] == (
        "SimpleAITrading-Round74-EventCohort-v4"
    )
    assert scheduler["superseded_task_present"] is False
    assert scheduler["next_run_time_utc"] == "2026-07-27T14:15:00Z"
    assert scheduler["repetition_interval"] == "PT1H5M"
    assert scheduler["last_trigger_utc"] == "2026-08-04T03:10:00Z"
    assert scheduler["start_when_available"] is False
    assert scheduler["multiple_instances"] == "IgnoreNew"
    limitations = evidence["limitations"]
    assert limitations["future_execution_proven_now"] is False
    assert limitations["cohort_slot_admitted_now"] is False
    assert limitations["profitability_or_edge_claim"] is False


def test_round74_cohort_v1_was_superseded_before_slot_zero() -> None:
    evidence = json.loads(V1_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["correction_basis"][
        "selected_from_market_or_model_outcome"
    ] is False
    assert evidence["correction_basis"][
        "schedule_or_role_counts_changed"
    ] is False
    assert evidence["pre_supersession_state"]["slot_zero_started"] is False
    assert evidence["pre_supersession_state"][
        "cohort_market_data_collected"
    ] is False
    sequence = evidence["replacement_sequence"]
    assert sequence[
        "replacement_task_verified_ready_before_superseded_task_removal"
    ] is True
    assert sequence["superseded_task_removed"] is True


def test_round74_cohort_v2_state_lateness_was_corrected_before_slot_zero() -> None:
    evidence = json.loads(V2_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    basis = evidence["correction_basis"]
    assert basis["maximum_entry_state_lateness_ns"] == 250_000_000
    assert basis["maximum_exit_state_lateness_ns"] == 250_000_000
    assert basis["selected_from_market_or_model_outcome"] is False
    assert basis["schedule_or_role_counts_changed"] is False
    assert evidence["pre_supersession_state"]["slot_zero_started"] is False
    assert evidence["pre_supersession_state"][
        "cohort_market_data_collected"
    ] is False
    sequence = evidence["replacement_sequence"]
    assert sequence[
        "replacement_task_verified_ready_before_superseded_task_removal"
    ] is True
    assert sequence["superseded_task_removed"] is True


def test_round74_cohort_operator_binds_corrected_plan_and_resources() -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)

    assert plan.plan_sha256 == ROUND74_EVENT_COHORT_PLAN_SHA256
    assert ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES == 24 * 1024**3
    assert ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES == 4 * 1024**3
    assert ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS == 120
    assert plan.slot(1).scheduled_start_wall_ns - plan.slot(
        0
    ).scheduled_start_wall_ns == 3_900_000_000_000


def test_round74_cohort_slot_selection_never_shifts_window() -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)
    start = plan.scheduled_start_wall_ns

    assert select_round74_cohort_slot(
        plan,
        now_wall_ns=start - 1,
    ).status == "before_campaign"
    at_start = select_round74_cohort_slot(plan, now_wall_ns=start)
    assert (at_start.status, at_start.slot_ordinal) == ("open", 0)
    assert select_round74_cohort_slot(
        plan,
        now_wall_ns=start + 30_000_000_000,
    ).status == "open"
    assert select_round74_cohort_slot(
        plan,
        now_wall_ns=start + 30_000_000_001,
    ).status == "between_slots"
    second = select_round74_cohort_slot(
        plan,
        now_wall_ns=start + 3_900_000_000_000,
    )
    assert (second.status, second.slot_ordinal) == ("open", 1)


def _active_result() -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "round-074-active-qualification-operator-result-v1",
        "preflight_sha256": ROUND74_ACTIVE_PREFLIGHT_SHA256,
        "capture_return_code": 0,
        "watchdog_breaches": [],
        "supervisor_report": {
            "status": "completed",
            "qualification_passed": True,
            "reconnect_count": 0,
        },
        "fresh_process_audits": [
            {
                "return_code": 0,
                "audit": {"passed": True},
            }
        ],
        "verdict": {
            "outcome": "active_qualified",
            "active_qualified": True,
            "capture_data_passed": True,
            "activity_label": "active",
            "errors": [],
        },
        "automatic_retry_permitted": False,
        "orders_submitted": False,
        "credentials_used": False,
    }
    from simple_ai_trading.round74_event_cohort_operator import _canonical_sha256

    result["result_sha256"] = _canonical_sha256(result)
    return result


def test_round74_cohort_prerequisite_rejects_non_active_result(
    tmp_path: Path,
) -> None:
    target = (
        tmp_path
        / "data"
        / "round74-v10-active-qualification"
        / ROUND74_ACTIVE_PREFLIGHT_SHA256
        / "result.json"
    )
    passed = _active_result()
    write_json_atomic(target, passed, sort_keys=True)

    assert validate_round74_active_prerequisite(tmp_path) == passed

    failed = deepcopy(passed)
    failed["verdict"]["activity_label"] = "middle"
    failed.pop("result_sha256")
    from simple_ai_trading.round74_event_cohort_operator import _canonical_sha256

    failed["result_sha256"] = _canonical_sha256(failed)
    write_json_atomic(target, failed, sort_keys=True)
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_round74_active_prerequisite(tmp_path)


def test_round74_cohort_requires_every_prior_binding(tmp_path: Path) -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)

    with pytest.raises(ValueError, match="prior slot 0 is missing"):
        _load_contiguous_bindings(tmp_path, plan, before_ordinal=1)
