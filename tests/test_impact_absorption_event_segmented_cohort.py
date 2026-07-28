from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from simple_ai_trading.impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from simple_ai_trading.impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS,
    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
)
from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS,
    ROUND74_SEGMENTED_COHORT_MISSED_REASON,
    ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS,
    ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
    ROUND74_SEGMENTED_COHORT_TOTAL_SLOTS,
    Round74SegmentedCohortCoverage,
    Round74SegmentedCohortPlan,
    Round74SegmentedCohortRunBinding,
    Round74SegmentedCohortSlotOutcome,
    bind_round74_segmented_probe_supervisor,
    build_round74_segmented_event_run_partition,
    iter_round74_v10_segment_event_observations,
    load_round74_segmented_cohort_binding,
    load_round74_segmented_cohort_coverage,
    load_round74_segmented_cohort_outcome,
    load_round74_segmented_cohort_plan,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
    ImpactAbsorptionStore,
)


_SECOND_NS = 1_000_000_000


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _plan() -> Round74SegmentedCohortPlan:
    return Round74SegmentedCohortPlan(
        scheduled_start_wall_ns=2_000_000_000_000_000_000,
        implementation_git_commit="c" * 40,
        prerequisite_artifact_sha256="a" * 64,
        prerequisite_window_start_wall_ns=1_999_900_000_000_000_000,
        prerequisite_window_end_wall_ns=1_999_904_000_000_000_000,
    )


def _supervisor(
    plan: Round74SegmentedCohortPlan,
    ordinal: int,
) -> tuple[dict[str, object], dict[str, object]]:
    slot = plan.slot(ordinal)
    run_id = f"{ordinal + 1:032x}"
    event_counts = {
        "aggTrade": 30,
        "bookTicker": 100,
        "depthUpdate": 60,
        "markPriceUpdate": 30,
        "forceOrder": 1,
        "serverTime": 10,
        "exchangeInfo": 1,
        "depthSnapshot": 3,
        "openInterest": 15,
    }
    report = {
        "schema_version": IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
        "design_sha256": ROUND74_CAPTURE_DESIGN_SHA256,
        "capture_contract_sha256": IMPACT_CAPTURE_V10_CONTRACT_SHA256,
        "run_id": run_id,
        "mode": "probe",
        "status": "completed",
        "qualification_passed": False,
        "capture_gate_passed": True,
        "data_qualification_passed": True,
        "resource_safety_passed": True,
        "storage_efficiency_passed": True,
        "audit_passed": True,
        "audit_errors": [],
        "resource_safety_errors": [],
        "error": "",
        "failure_class": "none",
        "payload_cap_reached": False,
        "database_size_cap_reached": False,
        "started_wall_ns": slot.scheduled_start_wall_ns,
        "ended_wall_ns": slot.scheduled_end_wall_ns + 5 * _SECOND_NS,
        "elapsed_seconds": 1_205.0,
        "event_counts": event_counts,
        "symbol_event_counts": {
            "BTCUSDT": {
                "aggTrade": 10,
                "bookTicker": 34,
                "depthSnapshot": 1,
                "depthUpdate": 20,
                "forceOrder": 1,
                "markPriceUpdate": 10,
                "openInterest": 5,
                "synchronizedDepthUpdate": 19,
            },
            "ETHUSDT": {
                "aggTrade": 10,
                "bookTicker": 33,
                "depthSnapshot": 1,
                "depthUpdate": 20,
                "forceOrder": 0,
                "markPriceUpdate": 10,
                "openInterest": 5,
                "synchronizedDepthUpdate": 19,
            },
            "SOLUSDT": {
                "aggTrade": 10,
                "bookTicker": 33,
                "depthSnapshot": 1,
                "depthUpdate": 20,
                "forceOrder": 0,
                "markPriceUpdate": 10,
                "openInterest": 5,
                "synchronizedDepthUpdate": 19,
            },
        },
        "writer_message_count": sum(event_counts.values()),
        "writer_frame_count": 12,
        "writer_compressed_payload_bytes": 4_096,
    }
    supervisor = {
        "schema_version": "round-074-capture-supervisor-report-v1",
        "design_sha256": ROUND74_CAPTURE_DESIGN_SHA256,
        "capture_schema_version": IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        "capture_contract_sha256": IMPACT_CAPTURE_V10_CONTRACT_SHA256,
        "status": "completed",
        "qualification_passed": False,
        "selected_run_id": run_id,
        "attempt_count": 1,
        "reconnect_count": 0,
        "reconnect_delays_seconds": [],
        "attempts": [report],
        "startup_errors": [],
        "terminal_error": "",
        "attempt_evidence_combined": False,
    }
    audit = {
        "schema_version": "round-074-capture-audit-v1",
        "passed": True,
        "errors": [],
        "run_id": run_id,
        "run_status": "completed",
        "stored_report_schema_version": IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
        "stored_report_sha256": _canonical_sha256(report),
        "capture_contract_sha256": IMPACT_CAPTURE_V10_CONTRACT_SHA256,
        "message_count": report["writer_message_count"],
        "frame_count": report["writer_frame_count"],
        "compressed_payload_bytes": report["writer_compressed_payload_bytes"],
        "last_frame_sha256": "d" * 64,
    }
    return supervisor, audit


def _binding(
    plan: Round74SegmentedCohortPlan,
    ordinal: int,
) -> Round74SegmentedCohortRunBinding:
    supervisor, audit = _supervisor(plan, ordinal)
    return bind_round74_segmented_probe_supervisor(
        plan,
        slot_ordinal=ordinal,
        supervisor_payload=supervisor,
        fresh_audit_payload=audit,
    )


def _outcomes(
    plan: Round74SegmentedCohortPlan,
    *,
    one_below_training_quorum: bool = False,
) -> tuple[Round74SegmentedCohortSlotOutcome, ...]:
    admitted_remaining = dict(ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS)
    if one_below_training_quorum:
        admitted_remaining["training"] -= 1
    selected: list[Round74SegmentedCohortSlotOutcome] = []
    for ordinal in range(plan.total_slots):
        role = plan.role_for_ordinal(ordinal)
        if admitted_remaining[role]:
            binding = _binding(plan, ordinal)
            selected.append(
                Round74SegmentedCohortSlotOutcome(
                    plan_sha256=plan.plan_sha256,
                    slot_ordinal=ordinal,
                    role=role,
                    status="admitted",
                    reason_code="admitted",
                    evidence_sha256=_canonical_sha256(
                        {"ordinal": ordinal, "status": "admitted"}
                    ),
                    binding=binding,
                )
            )
            admitted_remaining[role] -= 1
            continue
        status = "transport_excluded" if ordinal % 2 else "missed"
        selected.append(
            Round74SegmentedCohortSlotOutcome(
                plan_sha256=plan.plan_sha256,
                slot_ordinal=ordinal,
                role=role,
                status=status,
                reason_code=(
                    "in_run_transport"
                    if status == "transport_excluded"
                    else ROUND74_SEGMENTED_COHORT_MISSED_REASON
                ),
                evidence_sha256=_canonical_sha256(
                    {"ordinal": ordinal, "status": status}
                ),
            )
        )
    return tuple(selected)


def test_segmented_plan_is_exact_hash_bound_and_round_trips() -> None:
    plan = _plan()
    plan.validate()
    payload = plan.as_dict()

    assert plan.total_slots == 540
    assert ROUND74_SEGMENTED_COHORT_TOTAL_SLOTS == 540
    assert ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS == 1_200 * _SECOND_NS
    assert ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS == 1_500 * _SECOND_NS
    assert payload["role_counts"] == {"training": 386, "tuning": 77, "test": 77}
    assert payload["role_quorums"] == {"training": 360, "tuning": 72, "test": 72}
    assert payload["capture_contract"]["underlying_mode"] == "probe"
    assert payload["capture_contract"]["maximum_reconnects"] == 0
    assert payload["missingness_policy"]["all_admitted_units_included"] is True
    assert load_round74_segmented_cohort_plan(json.dumps(payload)).as_dict() == payload

    duplicate = json.dumps(payload).replace(
        "{",
        '{"schema_version":"duplicate",',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_round74_segmented_cohort_plan(duplicate)

    tampered = deepcopy(payload)
    tampered["role_quorums"]["training"] = 359
    with pytest.raises(ValueError, match="digest differs"):
        load_round74_segmented_cohort_plan(json.dumps(tampered))


def test_segmented_probe_binding_requires_fresh_exact_audit() -> None:
    plan = _plan()
    supervisor, audit = _supervisor(plan, 0)
    binding = bind_round74_segmented_probe_supervisor(
        plan,
        slot_ordinal=0,
        supervisor_payload=supervisor,
        fresh_audit_payload=audit,
    )

    assert binding.run_id == "0" * 31 + "1"
    assert binding.fresh_audit_sha256 == _canonical_sha256(audit)
    assert (
        load_round74_segmented_cohort_binding(json.dumps(binding.as_dict())).as_dict()
        == binding.as_dict()
    )

    mismatched = deepcopy(audit)
    mismatched["message_count"] = int(mismatched["message_count"]) - 1
    with pytest.raises(ValueError, match="fresh audit differs"):
        bind_round74_segmented_probe_supervisor(
            plan,
            slot_ordinal=0,
            supervisor_payload=supervisor,
            fresh_audit_payload=mismatched,
        )

    qualification = deepcopy(supervisor)
    qualification["qualification_passed"] = True
    with pytest.raises(ValueError, match="supervisor is not admissible"):
        bind_round74_segmented_probe_supervisor(
            plan,
            slot_ordinal=0,
            supervisor_payload=qualification,
            fresh_audit_payload=audit,
        )


def test_segmented_outcome_never_attaches_failed_prefix() -> None:
    plan = _plan()
    excluded = Round74SegmentedCohortSlotOutcome(
        plan_sha256=plan.plan_sha256,
        slot_ordinal=0,
        role="training",
        status="transport_excluded",
        reason_code="in_run_transport",
        evidence_sha256="e" * 64,
    )

    assert (
        load_round74_segmented_cohort_outcome(json.dumps(excluded.as_dict())).as_dict()
        == excluded.as_dict()
    )
    with pytest.raises(ValueError, match="contains a binding"):
        Round74SegmentedCohortSlotOutcome(
            **{
                **excluded.__dict__,
                "binding": _binding(plan, 0),
            }
        ).validate()


def test_segmented_coverage_uses_every_admitted_unit_and_is_gap_isolated() -> None:
    plan = _plan()
    outcomes = _outcomes(plan)
    coverage = Round74SegmentedCohortCoverage.build(plan, outcomes)
    payload = coverage.as_dict()

    assert len(coverage.partition.entries) == 360 + 72 + 72
    assert payload["role_counts"]["training"]["admitted"] == 360
    assert payload["role_counts"]["tuning"]["admitted"] == 72
    assert payload["role_counts"]["test"]["admitted"] == 72
    assert payload["all_admitted_units_included"] is True
    assert payload["transport_excluded_or_missed_units_included"] is False
    assert payload["cross_unit_feature_or_target_permitted"] is False
    assert len(payload["outcome_sha256"]) == plan.total_slots

    entries = coverage.partition.entries
    training_last = next(
        entry for entry in reversed(entries) if entry.role == "training"
    )
    tuning_first = next(entry for entry in entries if entry.role == "tuning")
    assert (
        training_last.eligible_anchor_end_wall_ns
        <= training_last.capture_end_wall_ns - ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
    )
    assert (
        tuning_first.eligible_anchor_start_wall_ns
        >= tuning_first.capture_start_wall_ns
        + ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS
    )
    assert tuple(entry.run_id for entry in entries) == tuple(
        outcome.binding.run_id for outcome in outcomes if outcome.binding is not None
    )
    assert (
        load_round74_segmented_cohort_coverage(
            json.dumps(payload),
            plan=plan,
            outcomes=outcomes,
        ).as_dict()
        == payload
    )


def test_segmented_coverage_fails_when_any_role_misses_quorum() -> None:
    plan = _plan()
    outcomes = _outcomes(plan, one_below_training_quorum=True)

    with pytest.raises(ValueError, match="role quorum failed"):
        build_round74_segmented_event_run_partition(plan, outcomes)


def test_segmented_replay_requires_read_only_store() -> None:
    plan = _plan()
    store = ImpactAbsorptionStore(":memory:")
    try:
        with pytest.raises(ValueError, match="read-only store"):
            list(
                iter_round74_v10_segment_event_observations(
                    store,
                    binding=_binding(plan, 0),
                )
            )
    finally:
        store.close()
