from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_STATE_HALF_LIVES_SECONDS,
)
from simple_ai_trading.round74_active_qualification import (
    ROUND74_ACTIVE_PREFLIGHT_RELATIVE_PATH,
    ROUND74_ACTIVE_PREFLIGHT_SHA256,
)
from simple_ai_trading.round74_active_adjudication import (
    ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH,
    ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH,
)
from simple_ai_trading.round74_event_cohort_operator import (
    ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS,
    ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES,
    ROUND74_EVENT_COHORT_MAXIMUM_STARTUP_LAUNCHES,
    ROUND74_EVENT_COHORT_PLAN_SHA256,
    ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES,
    ROUND74_EVENT_COHORT_STARTUP_PREREQUISITE_RELATIVE_PATH,
    ROUND74_EVENT_COHORT_STARTUP_PREREQUISITE_SHA256,
    ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS,
    _Round74CaptureLaunch,
    _load_contiguous_bindings,
    _pre_admission_startup_failure,
    _raise_for_capture_process_failure,
    _startup_relaunch_fits_slot,
    _validate_slot_audit,
    load_round74_cohort_operator_plan,
    run_round74_cohort_current_slot,
    select_round74_cohort_slot,
    validate_round74_active_prerequisite,
    validate_round74_startup_prerequisite,
)
from simple_ai_trading.storage import write_json_atomic


REPOSITORY = Path(__file__).resolve().parents[1]
OPERATOR_CONTRACT = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-operator-v12.json"
)
HOST_SCHEDULE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-host-schedule-v5-2026-07-27.json"
)
HISTORICAL_HOST_SCHEDULE_V6 = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-host-schedule-v6-2026-07-27.json"
)
CURRENT_HOST_SCHEDULE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-host-schedule-v7-2026-07-27.json"
)
CURRENT_HOST_SCHEDULE_V8 = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-host-schedule-v8-2026-07-27.json"
)
V5_SLOT_ZERO_FAILURE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-plan-v5-slot-000-failure-2026-07-27.json"
)
V5_R1_SLOT_ZERO_STARTUP_FAILURE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-plan-v5-r1-slot-000-startup-failure-2026-07-27.json"
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
V3_SUPERSESSION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-v3-supersession-2026-07-27.json"
)
V4_OPERATOR_SUPERSESSION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-operator-v4-supersession-2026-07-27.json"
)
V5_OPERATOR_SUPERSESSION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-operator-v5-supersession-2026-07-27.json"
)
V4_PLAN_R1_SUPERSESSION = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-plan-v4-r1-supersession-2026-07-27.json"
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


def _normalized_source_sha256(path: Path) -> str:
    normalized = (
        path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _startup_failure_supervisor(
    error: str = (
        "startup:RuntimeError:public_source:ConnectionClosedError:"
        "no close frame received or sent"
    ),
) -> dict[str, object]:
    return {
        "attempt_count": 1,
        "attempt_evidence_combined": False,
        "attempts": [],
        "capture_contract_sha256": (
            "5e245b0f398bb89ca579efcde6acef258fef4efa4204334b9657b43aa9e39cb0"
        ),
        "capture_schema_version": "round-074-prospective-evidence-v10",
        "design_sha256": (
            "b00e20499a0025c05cb27cc352d9444ce722493b5bdb592d628224343e81e136"
        ),
        "qualification_passed": False,
        "reconnect_count": 0,
        "reconnect_delays_seconds": [],
        "schema_version": "round-074-capture-supervisor-report-v1",
        "selected_run_id": "",
        "startup_errors": [error],
        "status": "failed",
        "terminal_error": error,
    }


def test_round74_cohort_operator_contract_binds_executable_bytes() -> None:
    contract = json.loads(OPERATOR_CONTRACT.read_text(encoding="utf-8"))
    claimed = contract.pop("artifact_sha256")
    source = contract["source_binding"]

    assert claimed == _canonical_sha256(contract)
    assert claimed == (
        "cb487f2817d090072b31841a0f438ff449a73b13ac0f1b805071e35d154ff145"
    )
    assert source["hash_mode"] == "utf8_lf_normalized_sha256"
    for path_key, hash_key in (
        ("operator_path", "operator_sha256"),
        ("cohort_path", "cohort_sha256"),
        ("wrapper_path", "wrapper_sha256"),
    ):
        assert (
            _normalized_source_sha256(REPOSITORY / source[path_key]) == source[hash_key]
        )
    execution = contract["slot_execution_contract"]
    assert execution["automatic_retry_permitted"] is False
    assert execution["maximum_reconnects"] == 0
    assert execution["heartbeat_state_persisted_during_capture"] is True
    assert execution["partition_written_only_after_all_168_bindings"] is True
    startup = contract["pre_admission_startup_contract"]
    assert startup["maximum_launches"] == 2
    assert startup["maximum_relaunches"] == 1
    assert startup["database_size_and_mtime_must_be_unchanged"] is True
    assert startup["wal_size_and_mtime_must_be_unchanged"] is True
    assert startup["failed_launch_evidence_combined_with_admitted_capture"] is False
    partition = contract["partition_contract"]
    assert partition["target_schema_version"] == "round-074-executable-event-target-v10"
    assert partition["dataset_schema_version"] == "round-074-event-dataset-v9"
    assert partition["event_sequence_schema_version"] == (
        "round-074-causal-event-sequence-v4"
    )
    assert partition["event_scaler_schema_version"] == (
        "round-074-event-feature-scaler-v4"
    )
    assert partition["feature_count"] == 66
    assert partition["feature_names_sha256"] == (
        "a753db09a2d6a089af16ad40eb2d0d4781921d635a0822b9fd0523bcddb28f4a"
    )
    assert partition["state_half_lives_seconds"] == list(
        ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
    )
    assert partition["common_reference_capital_accounting"] is True
    assert partition["actual_walked_entry_quote_notional_used"] is True
    assert partition["requested_size_fraction_used_as_realized_notional"] is False
    assert partition["maximum_target_span_ns"] == 310_500_000_000
    assert partition["minimum_purge_ns"] == 310_500_000_000
    assert partition["minimum_embargo_ns"] == 310_500_000_000


def test_round74_historical_host_schedule_remains_hash_bound() -> None:
    evidence = json.loads(HOST_SCHEDULE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == (
        "57eadcc86d2d672299aa2e3df81606e76deab76bf77fceefbe0c24c90d02dca2"
    )
    assert evidence["operator_contract_artifact_sha256"] == (
        "56f9696180d133ee4671fdaad9e3e362c6d7666f0e9eda6c8f64865d091fd285"
    )
    scheduler = evidence["host_scheduler"]
    assert scheduler["task_name"] == ("SimpleAITrading-Round74-EventCohort-v5")
    assert scheduler["superseded_task_present"] is True
    assert scheduler["superseded_task_state"] == "Disabled"
    assert scheduler["next_run_time_utc"] == "2026-07-27T16:00:00Z"
    assert scheduler["repetition_interval"] == "PT1H5M"
    assert scheduler["last_trigger_utc"] == "2026-08-04T04:55:00Z"
    assert scheduler["start_when_available"] is False
    assert scheduler["multiple_instances"] == "IgnoreNew"
    limitations = evidence["limitations"]
    assert limitations["active_prerequisite_proven_now"] is True
    assert limitations["future_execution_proven_now"] is False
    assert limitations["cohort_slot_admitted_now"] is False
    assert limitations["profitability_or_edge_claim"] is False


def test_round74_v6_host_schedule_remains_hash_bound() -> None:
    evidence = json.loads(HISTORICAL_HOST_SCHEDULE_V6.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == (
        "4373c432bcabb10071a0e60a90bf7ac99299139f223eb2a5afff920e6b78deb4"
    )
    assert evidence["operator_contract_artifact_sha256"] == (
        "0ef07ff29ae84064ded8956925959e79a55742751bef4890b2a3fd7a33073abf"
    )
    scheduler = evidence["host_scheduler"]
    assert scheduler["task_name"] == "SimpleAITrading-Round74-EventCohort-v6"
    assert scheduler["state"] == "Ready"
    assert scheduler["next_run_time_utc"] == "2026-07-27T20:00:00Z"
    assert scheduler["last_trigger_utc"] == "2026-08-05T12:45:00Z"
    assert scheduler["repetition_interval"] == "PT1H15M"
    assert scheduler["stop_at_duration_end"] is False
    assert scheduler["multiple_instances"] == "IgnoreNew"
    assert scheduler["start_when_available"] is False
    assert scheduler["task_execution_time_limit"] == "PT1H14M"
    assert evidence["superseded_host_scheduler"]["state"] == "Disabled"
    readiness = evidence["future_slot_zero_readiness"]
    assert readiness["ready_for_current_slot"] is True
    assert readiness["wal_absent"] is True
    assert readiness["no_active_capture_process"] is True
    limitations = evidence["limitations"]
    assert limitations["future_execution_proven_now"] is False
    assert limitations["cohort_slot_admitted_now"] is False
    assert limitations["profitability_or_edge_claim"] is False


def test_round74_v7_host_schedule_is_exact_and_pre_execution_only() -> None:
    evidence = json.loads(CURRENT_HOST_SCHEDULE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == (
        "a7e3afb41992599137b845e4e0d4ecee3b9c6cebadc9347617f551dcc04ec223"
    )
    assert evidence["operator_contract_artifact_sha256"] == (
        "c4c8b9bafdfec6cbfb95d96edf789e73d109e8d4117108ac69df407fd24b4962"
    )
    scheduler = evidence["host_scheduler"]
    assert scheduler["task_name"] == "SimpleAITrading-Round74-EventCohort-v7"
    assert scheduler["state"] == "Ready"
    assert scheduler["enabled"] is True
    assert scheduler["next_run_time_utc"] == "2026-07-27T23:00:00Z"
    assert scheduler["last_trigger_utc"] == "2026-08-05T15:45:00Z"
    assert scheduler["repetition_interval"] == "PT1H15M"
    assert scheduler["stop_at_duration_end"] is False
    assert scheduler["multiple_instances"] == "IgnoreNew"
    assert scheduler["start_when_available"] is False
    assert scheduler["task_execution_time_limit"] == "PT1H14M"
    assert evidence["superseded_host_scheduler"]["state"] == "Disabled"
    readiness = evidence["future_slot_zero_readiness"]
    assert readiness["ready_for_current_slot"] is True
    assert readiness["wal_absent"] is True
    assert readiness["no_active_capture_process"] is True
    limitations = evidence["limitations"]
    assert limitations["future_execution_proven_now"] is False
    assert limitations["cohort_slot_admitted_now"] is False
    assert limitations["failed_v5_capture_admitted"] is False
    assert limitations["profitability_or_edge_claim"] is False


def test_round74_v8_host_schedule_is_exact_and_pre_execution_only() -> None:
    evidence = json.loads(CURRENT_HOST_SCHEDULE_V8.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert claimed == (
        "c6e7e2fd5627af3b0d6c2008262c5b5583c2a77673ac8b2061899ca525102309"
    )
    assert evidence["cohort_plan_sha256"] == ROUND74_EVENT_COHORT_PLAN_SHA256
    assert evidence["operator_contract_artifact_sha256"] == (
        "cb487f2817d090072b31841a0f438ff449a73b13ac0f1b805071e35d154ff145"
    )
    scheduler = evidence["host_scheduler"]
    assert scheduler["task_name"] == "SimpleAITrading-Round74-EventCohort-v8"
    assert scheduler["state"] == "Ready"
    assert scheduler["enabled"] is True
    assert scheduler["next_run_time_utc"] == "2026-07-28T00:45:00Z"
    assert scheduler["last_trigger_utc"] == "2026-08-05T17:30:00Z"
    assert scheduler["repetition_interval"] == "PT1H15M"
    assert scheduler["stop_at_duration_end"] is False
    assert scheduler["multiple_instances"] == "IgnoreNew"
    assert scheduler["start_when_available"] is False
    assert scheduler["task_execution_time_limit"] == "PT1H14M"
    assert evidence["superseded_host_scheduler"]["state"] == "Disabled"
    readiness = evidence["future_slot_zero_readiness"]
    assert readiness["ready_for_current_slot"] is True
    assert readiness["startup_prerequisite_passed"] is True
    assert readiness["wal_absent"] is True
    assert readiness["no_active_capture_process"] is True
    limitations = evidence["limitations"]
    assert limitations["future_execution_proven_now"] is False
    assert limitations["cohort_slot_admitted_now"] is False
    assert limitations["failed_v5_r1_capture_admitted"] is False
    assert limitations["profitability_or_edge_claim"] is False


def test_round74_v5_slot_zero_failure_is_hash_bound_and_never_reused() -> None:
    evidence = json.loads(V5_SLOT_ZERO_FAILURE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    root_cause = evidence["root_cause"]
    assert root_cause["classification"] == "operator_identity_validation_defect"
    assert root_cause["raw_capture_failed"] is False
    assert root_cause["fresh_audit_failed"] is False
    assert root_cause["report_schema_contains_last_frame_sha256"] is False
    audit = evidence["capture_and_fresh_audit"]
    assert audit["report_hash_matches"] is True
    assert audit["run_id_matches"] is True
    assert audit["frame_count_matches"] is True
    assert audit["message_count_matches"] is True
    assert audit["compressed_payload_bytes_match"] is True
    adjudication = evidence["adjudication"]
    assert adjudication["outcome"] == "failed"
    assert adjudication["cohort_data_admitted"] is False
    assert adjudication["failed_capture_reused_as_cohort_data"] is False


def test_round74_v5_r1_startup_failure_is_hash_bound_and_never_reused() -> None:
    evidence = json.loads(V5_R1_SLOT_ZERO_STARTUP_FAILURE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert claimed == (
        "05d41b12b65604dd4451e8ecc12e05f51a634086e08cb4c409a7bb2e014bac3d"
    )
    supervisor = evidence["capture_supervisor"]
    assert supervisor["selected_run_id"] == ""
    assert supervisor["attempts"] == []
    assert supervisor["reconnect_count"] == 0
    assert evidence["resource_reconciliation"]["database_and_wal_growth_bytes"] == 0
    assert evidence["host_scheduler"]["state_after_failure_review"] == "Disabled"
    adjudication = evidence["adjudication"]
    assert adjudication["campaign_status"] == "permanently_failed"
    assert adjudication["cohort_data_admitted"] is False
    assert adjudication["campaign_reactivation_permitted"] is False


def test_round74_cohort_v1_was_superseded_before_slot_zero() -> None:
    evidence = json.loads(V1_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert (
        evidence["correction_basis"]["selected_from_market_or_model_outcome"] is False
    )
    assert evidence["correction_basis"]["schedule_or_role_counts_changed"] is False
    assert evidence["pre_supersession_state"]["slot_zero_started"] is False
    assert evidence["pre_supersession_state"]["cohort_market_data_collected"] is False
    sequence = evidence["replacement_sequence"]
    assert (
        sequence["replacement_task_verified_ready_before_superseded_task_removal"]
        is True
    )
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
    assert evidence["pre_supersession_state"]["cohort_market_data_collected"] is False
    sequence = evidence["replacement_sequence"]
    assert (
        sequence["replacement_task_verified_ready_before_superseded_task_removal"]
        is True
    )
    assert sequence["superseded_task_removed"] is True


def test_round74_cohort_v3_dataset_binding_was_corrected_before_slot_zero() -> None:
    evidence = json.loads(V3_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    basis = evidence["correction_basis"]
    assert basis["dataset_schema_before"] == "round-074-event-dataset-v5"
    assert basis["dataset_schema_after"] == "round-074-event-dataset-v7"
    assert basis["target_schema_after"] == ("round-074-executable-event-target-v10")
    assert (
        basis[
            "requested_size_fraction_substituted_for_realized_notional_after_correction"
        ]
        is False
    )
    assert basis["selected_from_market_or_model_outcome"] is False
    assert basis["schedule_or_role_counts_changed"] is False
    state = evidence["pre_supersession_state"]
    assert state["slot_zero_started"] is False
    assert state["cohort_market_data_collected"] is False
    sequence = evidence["replacement_sequence"]
    assert (
        sequence["replacement_task_verified_ready_before_superseded_task_removal"]
        is True
    )
    assert sequence["superseded_task_removed"] is True


def test_round74_cohort_operator_v4_was_superseded_before_slot_zero() -> None:
    evidence = json.loads(V4_OPERATOR_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == (
        "acf3e4feb8a918b03ab8d85c9ce730022aed1581181301ed513bd4ab4399dfcb"
    )
    assert evidence["superseded_operator"]["dataset_schema_version"] == (
        "round-074-event-dataset-v7"
    )
    assert evidence["replacement_operator"]["dataset_schema_version"] == (
        "round-074-event-dataset-v8"
    )
    basis = evidence["correction_basis"]
    assert basis["raw_capture_plan_changed"] is False
    assert basis["task_schedule_changed"] is False
    assert basis["roles_or_slot_times_changed"] is False
    assert basis["selected_from_market_model_or_target_outcome"] is False
    assert basis["slot_zero_started"] is False
    assert basis["cohort_market_data_collected"] is False
    assert evidence["authority"]["model_training_or_evaluation_performed"] is False
    assert evidence["authority"]["profitability_or_edge_claim"] is False


def test_round74_cohort_plan_r1_was_superseded_before_slot_zero() -> None:
    evidence = json.loads(V4_PLAN_R1_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["superseded"]["plan_sha256"] == (
        "acf3e4feb8a918b03ab8d85c9ce730022aed1581181301ed513bd4ab4399dfcb"
    )
    assert evidence["replacement"]["plan_sha256"] == (
        "57eadcc86d2d672299aa2e3df81606e76deab76bf77fceefbe0c24c90d02dca2"
    )
    basis = evidence["correction_basis"]
    assert basis["raw_active_result_retained"] is True
    assert basis["capture_retried"] is False
    assert basis["selected_from_market_model_or_target_outcome"] is False
    assert evidence["pre_supersession_state"]["slot_zero_started"] is False
    assert evidence["pre_supersession_state"]["campaign_halt_written"] is False
    assert evidence["authority"]["model_training_or_evaluation"] is False
    assert evidence["authority"]["profitability_or_edge_claim"] is False


def test_round74_cohort_operator_v5_was_superseded_before_slot_zero() -> None:
    evidence = json.loads(V5_OPERATOR_SUPERSESSION.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == (
        "acf3e4feb8a918b03ab8d85c9ce730022aed1581181301ed513bd4ab4399dfcb"
    )
    assert evidence["superseded_operator"]["dataset_schema_version"] == (
        "round-074-event-dataset-v8"
    )
    assert evidence["replacement_operator"]["dataset_schema_version"] == (
        "round-074-event-dataset-v9"
    )
    basis = evidence["correction_basis"]
    assert basis["raw_capture_plan_changed"] is False
    assert basis["task_schedule_changed"] is False
    assert basis["selected_from_market_model_or_target_outcome"] is False
    assert basis["slot_zero_started"] is False
    assert basis["cohort_market_data_collected"] is False


def test_round74_cohort_operator_binds_corrected_plan_and_resources() -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)

    assert (
        plan.plan_sha256
        == ROUND74_EVENT_COHORT_PLAN_SHA256
        == ("213a564026654905d62d2e74fd1c1944ff9ffd6d44af32557ccc20628ce59a04")
    )
    assert ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES == 24 * 1024**3
    assert ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES == 4 * 1024**3
    assert ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS == 300
    assert ROUND74_EVENT_COHORT_MAXIMUM_STARTUP_LAUNCHES == 2
    assert ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS == 0.25
    assert (
        plan.slot(1).scheduled_start_wall_ns - plan.slot(0).scheduled_start_wall_ns
        == 4_500_000_000_000
    )


def test_round74_cohort_slot_selection_never_shifts_window() -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)
    start = plan.scheduled_start_wall_ns

    assert (
        select_round74_cohort_slot(
            plan,
            now_wall_ns=start - 1,
        ).status
        == "before_campaign"
    )
    at_start = select_round74_cohort_slot(plan, now_wall_ns=start)
    assert (at_start.status, at_start.slot_ordinal) == ("open", 0)
    assert (
        select_round74_cohort_slot(
            plan,
            now_wall_ns=start + 30_000_000_000,
        ).status
        == "open"
    )
    assert (
        select_round74_cohort_slot(
            plan,
            now_wall_ns=start + 30_000_000_001,
        ).status
        == "between_slots"
    )
    second = select_round74_cohort_slot(
        plan,
        now_wall_ns=start + 4_500_000_000_000,
    )
    assert (second.status, second.slot_ordinal) == ("open", 1)


def test_round74_cohort_prerequisite_requires_exact_adjudicated_result(
    tmp_path: Path,
) -> None:
    for relative in (
        ROUND74_ACTIVE_PREFLIGHT_RELATIVE_PATH,
        ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH,
        ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / relative, destination)
    target = (
        tmp_path
        / "data"
        / "round74-v10-active-qualification"
        / ROUND74_ACTIVE_PREFLIGHT_SHA256
        / "result.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPOSITORY / ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH, target)

    validated = validate_round74_active_prerequisite(tmp_path)
    assert validated["source_result"]["result_sha256"] == (
        "baa31064a82fa6bb742f5de288661b7a8b19e3d88bddc6accbb8adaee84d20ab"
    )
    assert validated["adjudication"]["artifact_sha256"] == (
        "fef501a34da6b36bb004b1731b9751a1cdb52ce649fdc4fa6579640c7af962a7"
    )

    failed = deepcopy(validated["source_result"])
    failed["verdict"]["activity_label"] = "middle"
    failed.pop("result_sha256")
    from simple_ai_trading.round74_event_cohort_operator import _canonical_sha256

    failed["result_sha256"] = _canonical_sha256(failed)
    write_json_atomic(target, failed, sort_keys=True)
    with pytest.raises(ValueError, match="file identity differs"):
        validate_round74_active_prerequisite(tmp_path)


def test_round74_cohort_requires_exact_startup_prerequisite(
    tmp_path: Path,
) -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)
    payload = validate_round74_startup_prerequisite(REPOSITORY, plan)

    assert payload["artifact_sha256"] == (
        ROUND74_EVENT_COHORT_STARTUP_PREREQUISITE_SHA256
    )
    assert payload["live_capture"]["message_count"] == 202_725
    assert payload["fresh_process_audit"]["passed"] is True
    destination = tmp_path / ROUND74_EVENT_COHORT_STARTUP_PREREQUISITE_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    tampered = deepcopy(payload)
    tampered["live_capture"]["message_count"] = 202_724
    write_json_atomic(destination, tampered, sort_keys=True)

    with pytest.raises(ValueError, match="startup prerequisite differs"):
        validate_round74_startup_prerequisite(tmp_path, plan)


def test_round74_cohort_requires_every_prior_binding(tmp_path: Path) -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)

    with pytest.raises(ValueError, match="prior slot 0 is missing"):
        _load_contiguous_bindings(tmp_path, plan, before_ordinal=1)


def test_round74_cohort_process_failure_precedes_supervisor_json_parse() -> None:
    _raise_for_capture_process_failure(
        return_code=0,
        breaches=[],
        stop_method="",
        stdout_lines=['{"status":"completed"}'],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"return_code=-15 breaches=\['slot_capture_deadline_exceeded'\] "
            r"stop=terminate stdout=empty"
        ),
    ):
        _raise_for_capture_process_failure(
            return_code=-15,
            breaches=["slot_capture_deadline_exceeded"],
            stop_method="terminate",
            stdout_lines=[],
        )


def test_round74_cohort_relaunches_only_exact_zero_evidence_transport_startup(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "capture.stdout.json"
    stderr_path = tmp_path / "capture.stderr.log"
    supervisor = _startup_failure_supervisor()
    supervisor_text = json.dumps(supervisor)
    stdout_path.write_text(supervisor_text, encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    launch = _Round74CaptureLaunch(
        launch_ordinal=1,
        return_code=2,
        breaches=(),
        stop_method="",
        stdout_text=supervisor_text,
        stderr_text="",
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        monitor_samples=(),
        database_bytes_after=9_433_526_272,
        wal_bytes_after=0,
        database_mtime_ns_after=17,
        wal_mtime_ns_after=0,
    )

    accepted = _pre_admission_startup_failure(
        launch,
        baseline_database=9_433_526_272,
        baseline_wal=0,
        baseline_database_mtime_ns=17,
        baseline_wal_mtime_ns=0,
        active_capture_processes=[],
    )
    assert accepted == supervisor

    assert (
        _pre_admission_startup_failure(
            replace(launch, database_bytes_after=9_433_526_273),
            baseline_database=9_433_526_272,
            baseline_wal=0,
            baseline_database_mtime_ns=17,
            baseline_wal_mtime_ns=0,
            active_capture_processes=[],
        )
        is None
    )
    assert (
        _pre_admission_startup_failure(
            replace(launch, wal_bytes_after=1),
            baseline_database=9_433_526_272,
            baseline_wal=0,
            baseline_database_mtime_ns=17,
            baseline_wal_mtime_ns=0,
            active_capture_processes=[],
        )
        is None
    )
    assert (
        _pre_admission_startup_failure(
            replace(launch, database_mtime_ns_after=18),
            baseline_database=9_433_526_272,
            baseline_wal=0,
            baseline_database_mtime_ns=17,
            baseline_wal_mtime_ns=0,
            active_capture_processes=[],
        )
        is None
    )
    assert (
        _pre_admission_startup_failure(
            launch,
            baseline_database=9_433_526_272,
            baseline_wal=0,
            baseline_database_mtime_ns=17,
            baseline_wal_mtime_ns=0,
            active_capture_processes=[
                {"process_id": 7, "command_line": "impact-capture"}
            ],
        )
        is None
    )
    non_transport = _startup_failure_supervisor(
        "startup:ValueError:capture contract differs"
    )
    non_transport_text = json.dumps(non_transport)
    stdout_path.write_text(non_transport_text, encoding="utf-8")
    assert (
        _pre_admission_startup_failure(
            replace(launch, stdout_text=non_transport_text),
            baseline_database=9_433_526_272,
            baseline_wal=0,
            baseline_database_mtime_ns=17,
            baseline_wal_mtime_ns=0,
            active_capture_processes=[],
        )
        is None
    )
    run_created = _startup_failure_supervisor()
    run_created["attempts"] = [{"run_id": "1" * 32}]
    run_created["selected_run_id"] = "1" * 32
    run_created_text = json.dumps(run_created)
    stdout_path.write_text(run_created_text, encoding="utf-8")
    assert (
        _pre_admission_startup_failure(
            replace(launch, stdout_text=run_created_text),
            baseline_database=9_433_526_272,
            baseline_wal=0,
            baseline_database_mtime_ns=17,
            baseline_wal_mtime_ns=0,
            active_capture_processes=[],
        )
        is None
    )


def test_round74_cohort_startup_relaunch_must_fit_original_window() -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)
    window_end = plan.slot(0).start_window_end_wall_ns

    assert _startup_relaunch_fits_slot(
        plan,
        ordinal=0,
        now_wall_ns=(
            window_end
            - 1_000_000_000
            - int(ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS * 1e9)
        ),
    )
    assert not _startup_relaunch_fits_slot(
        plan,
        ordinal=0,
        now_wall_ns=(
            window_end
            - 1_000_000_000
            - int(ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS * 1e9)
            + 1
        ),
    )


def test_round74_cohort_fresh_audit_accepts_actual_v10_report_identity() -> None:
    report_sha256 = "a" * 64
    run_id = "1" * 32
    capture_contract_sha256 = "b" * 64
    binding = SimpleNamespace(
        run_id=run_id,
        report_sha256=report_sha256,
        frame_count=864,
        message_count=1_427_754,
        compressed_payload_bytes=50_642_043,
    )
    supervisor = {
        "attempts": [
            {
                "run_id": run_id,
                "capture_contract_sha256": capture_contract_sha256,
                "writer_frame_count": 864,
                "writer_message_count": 1_427_754,
                "writer_compressed_payload_bytes": 50_642_043,
            }
        ]
    }
    audit = {
        "_operator_evidence": {"return_code": 0},
        "passed": True,
        "errors": [],
        "run_id": run_id,
        "stored_report_sha256": report_sha256,
        "capture_contract_sha256": capture_contract_sha256,
        "last_frame_sha256": "c" * 64,
        "frame_count": 864,
        "message_count": 1_427_754,
        "compressed_payload_bytes": 50_642_043,
    }

    _validate_slot_audit(audit, binding, supervisor)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_contract_sha256", "d" * 64),
        ("frame_count", 863),
        ("message_count", 1_427_753),
        ("compressed_payload_bytes", 50_642_042),
    ],
)
def test_round74_cohort_fresh_audit_rejects_mismatched_v10_identity(
    field: str,
    value: object,
) -> None:
    report_sha256 = "a" * 64
    run_id = "1" * 32
    capture_contract_sha256 = "b" * 64
    binding = SimpleNamespace(
        run_id=run_id,
        report_sha256=report_sha256,
        frame_count=864,
        message_count=1_427_754,
        compressed_payload_bytes=50_642_043,
    )
    supervisor = {
        "attempts": [
            {
                "run_id": run_id,
                "capture_contract_sha256": capture_contract_sha256,
                "writer_frame_count": 864,
                "writer_message_count": 1_427_754,
                "writer_compressed_payload_bytes": 50_642_043,
            }
        ]
    }
    audit = {
        "_operator_evidence": {"return_code": 0},
        "passed": True,
        "errors": [],
        "run_id": run_id,
        "stored_report_sha256": report_sha256,
        "capture_contract_sha256": capture_contract_sha256,
        "last_frame_sha256": "c" * 64,
        "frame_count": 864,
        "message_count": 1_427_754,
        "compressed_payload_bytes": 50_642_043,
    }
    audit[field] = value

    with pytest.raises(ValueError, match="fresh audit identity differs"):
        _validate_slot_audit(audit, binding, supervisor)  # type: ignore[arg-type]


def test_round74_cohort_persists_live_progress_before_child_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simple_ai_trading import round74_event_cohort_operator as operator

    class ControlledProcess:
        pid = 41184

        def __init__(self) -> None:
            self.stdout = StringIO("")
            self.stderr = StringIO("")
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 1

        def wait(self, *, timeout: int) -> int:
            assert timeout == 60
            return 1

    plan = load_round74_cohort_operator_plan(REPOSITORY)
    controlled_process = ControlledProcess()
    monkeypatch.setattr(
        operator.subprocess, "Popen", lambda *_args, **_kwargs: controlled_process
    )
    monkeypatch.setattr(operator, "_active_capture_processes", lambda: [])
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        operator.time,
        "time_ns",
        lambda: plan.slot(0).scheduled_start_wall_ns,
    )

    with pytest.raises(ValueError, match=r"return_code=1.*stdout=empty"):
        operator._run_slot_process(tmp_path, plan, ordinal=0)

    state = json.loads(
        (operator._slot_root(tmp_path, 0) / "state.json").read_text(encoding="utf-8")
    )
    assert state["phase"] == "running"
    assert state["process_id"] == controlled_process.pid
    assert state["monitor_sample_count"] == 1
    assert state["last_progress"]["elapsed_seconds"] >= 0.0
    assert state["last_progress"]["database_and_wal_growth_bytes"] == 0
    assert state["last_progress_at_utc"].endswith("Z")


def test_round74_cohort_persists_one_excluded_startup_before_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simple_ai_trading import round74_event_cohort_operator as operator

    class ImmediateProcess:
        def __init__(self, pid: int, return_code: int, stdout: str) -> None:
            self.pid = pid
            self.return_code = return_code
            self.stdout = StringIO(stdout)
            self.stderr = StringIO("")

        def poll(self) -> int:
            return self.return_code

        def wait(self, *, timeout: int) -> int:
            assert timeout == 60
            return self.return_code

    plan = load_round74_cohort_operator_plan(REPOSITORY)
    processes = iter(
        (
            ImmediateProcess(41184, 2, json.dumps(_startup_failure_supervisor())),
            ImmediateProcess(41185, 1, "{}"),
        )
    )
    spawn_count = 0

    def spawn(*_args: object, **_kwargs: object) -> ImmediateProcess:
        nonlocal spawn_count
        spawn_count += 1
        return next(processes)

    monkeypatch.setattr(operator.subprocess, "Popen", spawn)
    monkeypatch.setattr(operator, "_active_capture_processes", lambda: [])
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        operator.time,
        "time_ns",
        lambda: plan.slot(0).scheduled_start_wall_ns,
    )

    with pytest.raises(ValueError, match=r"return_code=1.*stdout=present"):
        operator._run_slot_process(tmp_path, plan, ordinal=0)

    assert spawn_count == 2
    first = json.loads(
        (operator._slot_root(tmp_path, 0) / "startup-launch-001.json").read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        (operator._slot_root(tmp_path, 0) / "startup-launch-002.json").read_text(
            encoding="utf-8"
        )
    )
    state = json.loads(
        (operator._slot_root(tmp_path, 0) / "state.json").read_text(encoding="utf-8")
    )
    assert first["disposition"] == "excluded_transient_pre_admission_startup"
    assert first["selected_run_id"] == ""
    assert first["supervisor_attempts"] == []
    assert first["database_or_wal_changed"] is False
    assert first["pre_admission_startup_relaunch_permitted"] is True
    assert second["disposition"] == "terminal_capture_failure"
    assert second["pre_admission_startup_relaunch_permitted"] is False
    assert state["current_startup_launch_ordinal"] == 2
    assert len(state["pre_admission_startup_launches"]) == 2


def test_round74_cohort_failure_terminalizes_reserved_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simple_ai_trading import round74_event_cohort_operator as operator

    plan = load_round74_cohort_operator_plan(REPOSITORY)
    monkeypatch.setattr(
        operator,
        "load_round74_cohort_operator_plan",
        lambda _repository: plan,
    )
    monkeypatch.setattr(operator.time, "time_ns", lambda: plan.scheduled_start_wall_ns)
    monkeypatch.setattr(
        operator,
        "validate_round74_active_prerequisite",
        lambda _repository: {},
    )
    monkeypatch.setattr(
        operator,
        "_load_contiguous_bindings",
        lambda _repository, _plan, *, before_ordinal: [],
    )
    monkeypatch.setattr(
        operator,
        "inspect_round74_cohort_readiness",
        lambda _repository, *, now_wall_ns: {"ready_for_current_slot": True},
    )

    def fail_reserved_slot(
        repository: Path,
        _plan: object,
        *,
        ordinal: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        slot_root = operator._slot_root(repository, ordinal)
        reservation = {
            "schema_version": operator.ROUND74_EVENT_COHORT_OPERATOR_STATE_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "slot_ordinal": ordinal,
            "role": plan.slot(ordinal).role,
            "reserved_at_utc": "2026-07-27T16:00:00Z",
            "automatic_retry_permitted": False,
        }
        operator._durable_json_replace(
            slot_root / "attempt-reservation.json",
            reservation,
        )
        operator._durable_json_replace(
            slot_root / "state.json",
            {**reservation, "phase": "running", "process_id": 41184},
        )
        raise ValueError(
            "Round 74 cohort capture process failed: "
            "return_code=-15 breaches=['slot_capture_deadline_exceeded'] "
            "stop=terminate stdout=empty"
        )

    monkeypatch.setattr(operator, "_run_slot_process", fail_reserved_slot)

    with pytest.raises(ValueError, match="slot_capture_deadline_exceeded"):
        run_round74_cohort_current_slot(tmp_path)

    slot_root = operator._slot_root(tmp_path, 0)
    result = json.loads((slot_root / "result.json").read_text(encoding="utf-8"))
    state = json.loads((slot_root / "state.json").read_text(encoding="utf-8"))
    halt = json.loads(
        (operator._campaign_root(tmp_path) / "halt.json").read_text(encoding="utf-8")
    )

    assert result["outcome"] == "failed"
    assert result["automatic_retry_permitted"] is False
    assert state["phase"] == "terminal"
    assert state["outcome"] == "failed"
    assert state["process_id"] == 41184
    assert state["result_sha256"] == result["result_sha256"]
    assert state["automatic_retry_permitted"] is False
    assert halt["slot_ordinal"] == 0
    assert halt["automatic_retry_permitted"] is False
