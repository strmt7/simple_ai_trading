from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from simple_ai_trading.impact_absorption_event_cohort import (
    load_round74_event_cohort_plan,
)


REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH = REPOSITORY / "docs" / "model-research" / "action-value"
HISTORICAL_PLAN_PATH = RESEARCH / "round-074-event-cohort-plan-v4-r2.json"
PLAN_PATH = RESEARCH / "round-074-event-cohort-plan-v5.json"
PREFLIGHT_PATH = (
    RESEARCH / "round-074-v10-active-regime-qualification-preflight-2026-07-27.json"
)
ADJUDICATION_PATH = (
    RESEARCH / "round-074-v10-active-regime-qualification-adjudication-2026-07-27.json"
)
CADENCE_CORRECTION_PATH = (
    RESEARCH / "round-074-event-cohort-cadence-correction-2026-07-27.json"
)
SLOT_ZERO_FAILURE_PATH = (
    RESEARCH / "round-074-event-cohort-plan-v4-r2-slot-000-failure-2026-07-27.json"
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


def _wall_ns(value: str) -> int:
    return int(
        datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000_000
    )


def _git_blob(revision: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_round74_cohort_plan_is_hash_bound_and_predeclared() -> None:
    plan = load_round74_event_cohort_plan(PLAN_PATH.read_text(encoding="utf-8"))
    payload = plan.as_dict()

    assert payload["plan_sha256"] == (
        "4373c432bcabb10071a0e60a90bf7ac99299139f223eb2a5afff920e6b78deb4"
    )
    assert plan.total_slots == 168
    assert (plan.training_slots, plan.tuning_slots, plan.test_slots) == (
        120,
        24,
        24,
    )
    assert (
        datetime.fromtimestamp(
            plan.scheduled_start_wall_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat()
        == "2026-07-27T20:00:00+00:00"
    )
    assert (
        datetime.fromtimestamp(
            plan.slot(167).scheduled_end_wall_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat()
        == "2026-08-05T13:45:00+00:00"
    )
    assert plan.slot(119).role == "training"
    assert plan.slot(120).role == "tuning"
    assert plan.slot(143).role == "tuning"
    assert plan.slot(144).role == "test"
    assert (
        payload["capture_contract"]["failed_or_missed_slot_replacement_permitted"]
        is False
    )
    assert (
        payload["partition_policy"][
            "test_labels_or_outcomes_visible_before_pretest_seal"
        ]
        is False
    )
    assert payload["partition_policy"]["minimum_purge_ns"] == 310_500_000_000
    assert payload["partition_policy"]["minimum_embargo_ns"] == 310_500_000_000
    assert payload["partition_policy"]["maximum_target_span_ns"] == 310_500_000_000
    assert set(payload["scope"].values()) >= {False}
    assert payload["scope"]["financial_edge_tested_by_plan"] is False
    assert payload["scope"]["profitability_claim"] is False
    assert payload["scope"]["trading_authority"] is False
    assert payload["schedule_formula"] == {
        "ordinal_origin": 0,
        "slot_period_ns": 4_500_000_000_000,
        "capture_duration_ns": 3_600_000_000_000,
        "start_tolerance_ns": 30_000_000_000,
        "maximum_end_overhead_ns": 300_000_000_000,
        "fresh_audit_timeout_ns": 300_000_000_000,
    }


def test_round74_cohort_plan_binds_prerequisite_and_implementation() -> None:
    plan = load_round74_event_cohort_plan(PLAN_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    preflight_claimed = preflight.pop("artifact_sha256")
    adjudication = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    adjudication_claimed = adjudication.pop("artifact_sha256")

    assert preflight_claimed == _canonical_sha256(preflight)
    assert adjudication_claimed == _canonical_sha256(adjudication)
    assert plan.prerequisite_artifact_sha256 == adjudication_claimed
    window = preflight["fixed_execution_window"]
    assert plan.prerequisite_window_start_wall_ns == _wall_ns(
        window["earliest_start_utc"]
    )
    assert plan.prerequisite_window_end_wall_ns == (
        _wall_ns(window["latest_start_utc"])
        + int(window["duration_seconds"]) * 1_000_000_000
    )
    assert plan.prerequisite_window_end_wall_ns < (plan.scheduled_start_wall_ns)
    # The prospective plan binds the committed implementation that introduced
    # v5 schedule policy before the plan existed.
    path = "src/simple_ai_trading/impact_absorption_event_cohort.py"
    assert _git_blob(plan.implementation_git_commit, path) == _git_blob("HEAD", path)


def test_round74_historical_plan_remains_loadable_and_immutable() -> None:
    plan = load_round74_event_cohort_plan(
        HISTORICAL_PLAN_PATH.read_text(encoding="utf-8")
    )

    assert plan.plan_sha256 == (
        "57eadcc86d2d672299aa2e3df81606e76deab76bf77fceefbe0c24c90d02dca2"
    )
    assert plan.slot_period_ns == 3_900_000_000_000
    assert plan.maximum_end_overhead_ns == 120_000_000_000
    path = "src/simple_ai_trading/impact_absorption_event_cohort.py"
    assert _git_blob(plan.implementation_git_commit, path) == (
        "a5b5870a483335cca3bc0187bc75d2ea496cf2c9"
    )


def test_round74_slot_zero_failure_is_hash_bound_and_excluded() -> None:
    evidence = json.loads(SLOT_ZERO_FAILURE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["failed_campaign"]["automatic_retry_permitted"] is False
    assert (
        evidence["failed_campaign"]["failed_or_missed_slot_replacement_permitted"]
        is False
    )
    database = evidence["read_only_database_reconciliation"]
    assert database["message_count"] == 4_498_908
    assert database["frame_count"] == 837
    assert database["stored_capture_report_count"] == 0
    assert evidence["adjudication"]["cohort_data_admitted"] is False
    assert evidence["adjudication"]["profitability_or_edge_claim"] is False
    correction = evidence["prospective_correction"]
    assert correction["historical_slot_retried_or_replaced"] is False
    assert correction["new_slot_period_seconds"] == 4_500


def test_round74_cohort_cadence_correction_prevents_audit_overlap() -> None:
    evidence = json.loads(CADENCE_CORRECTION_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert (
        evidence["pre_observation_basis"][
            "correction_selected_from_market_or_model_outcome"
        ]
        is False
    )
    defect = evidence["defect"]
    assert defect["nominal_gap_seconds"] < (
        defect["maximum_start_tolerance_seconds"]
        + defect["maximum_end_overhead_seconds"]
    )
    correction = evidence["correction"]
    assert correction["nominal_gap_seconds"] == (
        correction["maximum_start_and_end_allowance_seconds"]
        + correction["maximum_inline_fresh_audit_seconds"]
        + correction["minimum_process_transition_reserve_seconds"]
    )
    assert correction["overlap_permitted"] is False
    assert correction["role_counts_changed"] is False
