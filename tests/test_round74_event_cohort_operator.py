from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from simple_ai_trading.impact_absorption_event_dataset import (
    ROUND74_EVENT_DATASET_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
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
    / "round-074-event-cohort-operator-v7.json"
)
HOST_SCHEDULE = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-cohort-host-schedule-v5-2026-07-27.json"
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


def test_round74_cohort_operator_contract_binds_executable_bytes() -> None:
    contract = json.loads(OPERATOR_CONTRACT.read_text(encoding="utf-8"))
    claimed = contract.pop("artifact_sha256")
    source = contract["source_binding"]

    assert claimed == _canonical_sha256(contract)
    for path_key, hash_key in (
        ("operator_path", "operator_sha256"),
        ("wrapper_path", "wrapper_sha256"),
    ):
        assert (
            hashlib.sha256((REPOSITORY / source[path_key]).read_bytes()).hexdigest()
            == source[hash_key]
        )
    execution = contract["slot_execution_contract"]
    assert execution["automatic_retry_permitted"] is False
    assert execution["partition_written_only_after_all_168_bindings"] is True
    partition = contract["partition_contract"]
    assert partition["target_schema_version"] == "round-074-executable-event-target-v10"
    assert partition["dataset_schema_version"] == (ROUND74_EVENT_DATASET_SCHEMA_VERSION)
    assert partition["event_sequence_schema_version"] == (
        ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION
    )
    assert partition["event_scaler_schema_version"] == (
        ROUND74_EVENT_SCALER_SCHEMA_VERSION
    )
    assert partition["feature_count"] == len(ROUND74_EVENT_FEATURE_NAMES) == 66
    assert partition["feature_names_sha256"] == ROUND74_EVENT_FEATURE_NAMES_SHA256
    assert partition["state_half_lives_seconds"] == list(
        ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
    )
    assert partition["common_reference_capital_accounting"] is True
    assert partition["actual_walked_entry_quote_notional_used"] is True
    assert partition["requested_size_fraction_used_as_realized_notional"] is False
    assert partition["maximum_target_span_ns"] == 310_500_000_000
    assert partition["minimum_purge_ns"] == 310_500_000_000
    assert partition["minimum_embargo_ns"] == 310_500_000_000


def test_round74_cohort_host_schedule_is_exact_and_pre_execution_only() -> None:
    evidence = json.loads(HOST_SCHEDULE.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["cohort_plan_sha256"] == ROUND74_EVENT_COHORT_PLAN_SHA256
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
    assert evidence["replacement"]["plan_sha256"] == (ROUND74_EVENT_COHORT_PLAN_SHA256)
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
        ROUND74_EVENT_DATASET_SCHEMA_VERSION
    )
    basis = evidence["correction_basis"]
    assert basis["raw_capture_plan_changed"] is False
    assert basis["task_schedule_changed"] is False
    assert basis["selected_from_market_model_or_target_outcome"] is False
    assert basis["slot_zero_started"] is False
    assert basis["cohort_market_data_collected"] is False


def test_round74_cohort_operator_binds_corrected_plan_and_resources() -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)

    assert plan.plan_sha256 == ROUND74_EVENT_COHORT_PLAN_SHA256
    assert ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES == 24 * 1024**3
    assert ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES == 4 * 1024**3
    assert ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS == 120
    assert (
        plan.slot(1).scheduled_start_wall_ns - plan.slot(0).scheduled_start_wall_ns
        == 3_900_000_000_000
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
        now_wall_ns=start + 3_900_000_000_000,
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


def test_round74_cohort_requires_every_prior_binding(tmp_path: Path) -> None:
    plan = load_round74_cohort_operator_plan(REPOSITORY)

    with pytest.raises(ValueError, match="prior slot 0 is missing"):
        _load_contiguous_bindings(tmp_path, plan, before_ordinal=1)
