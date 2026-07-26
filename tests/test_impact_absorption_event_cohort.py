from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from simple_ai_trading.impact_absorption import (
    ROUND74_CAPTURE_DESIGN_SHA256,
)
from simple_ai_trading.impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS,
    ROUND74_EVENT_COHORT_SLOT_PERIOD_NS,
    ROUND74_EVENT_COHORT_START_TOLERANCE_NS,
    Round74EventCohortPlan,
    Round74EventCohortRunBinding,
    bind_round74_event_cohort_supervisor,
    build_round74_event_run_partition,
    load_round74_event_cohort_binding,
    load_round74_event_cohort_plan,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS,
    ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
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


def _plan() -> Round74EventCohortPlan:
    return Round74EventCohortPlan(
        scheduled_start_wall_ns=2_000_000_000_000_000_000,
        training_slots=2,
        tuning_slots=1,
        test_slots=1,
        prerequisite_artifact_sha256="a" * 64,
        prerequisite_window_start_wall_ns=1_999_900_000_000_000_000,
        prerequisite_window_end_wall_ns=1_999_904_000_000_000_000,
    )


def _supervisor(
    plan: Round74EventCohortPlan,
    ordinal: int,
) -> dict[str, object]:
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
        "mode": "qualification",
        "status": "completed",
        "qualification_passed": True,
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
        "elapsed_seconds": 3_605.0,
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
    return {
        "schema_version": "round-074-capture-supervisor-report-v1",
        "design_sha256": ROUND74_CAPTURE_DESIGN_SHA256,
        "capture_schema_version": IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        "capture_contract_sha256": IMPACT_CAPTURE_V10_CONTRACT_SHA256,
        "status": "completed",
        "qualification_passed": True,
        "selected_run_id": run_id,
        "attempt_count": 1,
        "reconnect_count": 0,
        "reconnect_delays_seconds": [],
        "attempts": [report],
        "startup_errors": [],
        "terminal_error": "",
        "attempt_evidence_combined": False,
    }


def _bindings(
    plan: Round74EventCohortPlan,
) -> tuple[Round74EventCohortRunBinding, ...]:
    return tuple(
        bind_round74_event_cohort_supervisor(
            plan,
            slot_ordinal=ordinal,
            supervisor_payload=_supervisor(plan, ordinal),
        )
        for ordinal in range(plan.total_slots)
    )


def test_plan_is_compact_deterministic_and_strictly_loadable() -> None:
    plan = _plan()
    payload = plan.as_dict()
    loaded = load_round74_event_cohort_plan(json.dumps(payload))

    assert loaded == plan
    assert loaded.plan_sha256 == payload["plan_sha256"]
    assert loaded.total_slots == 4
    assert [loaded.slot(index).role for index in range(4)] == [
        "training",
        "training",
        "tuning",
        "test",
    ]
    assert (
        loaded.slot(3).scheduled_start_wall_ns
        == loaded.scheduled_start_wall_ns
        + 3 * ROUND74_EVENT_COHORT_SLOT_PERIOD_NS
    )
    assert (
        loaded.slot(0).scheduled_end_wall_ns
        - loaded.slot(0).scheduled_start_wall_ns
        == ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS
    )
    assert (
        loaded.slot(0).start_window_end_wall_ns
        - loaded.slot(0).scheduled_start_wall_ns
        == ROUND74_EVENT_COHORT_START_TOLERANCE_NS
    )


def test_plan_loader_rejects_duplicate_keys_and_policy_tampering() -> None:
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_round74_event_cohort_plan('{"schema_version":"x","schema_version":"y"}')

    payload = _plan().as_dict()
    payload["capture_contract"]["maximum_reconnects"] = 1
    unsigned = dict(payload)
    unsigned.pop("plan_sha256")
    payload["plan_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="static policy differs"):
        load_round74_event_cohort_plan(json.dumps(payload))


def test_qualified_supervisor_binds_to_exact_predeclared_slot() -> None:
    plan = _plan()
    supervisor = _supervisor(plan, 0)
    binding = bind_round74_event_cohort_supervisor(
        plan,
        slot_ordinal=0,
        supervisor_payload=supervisor,
    )
    loaded = load_round74_event_cohort_binding(
        json.dumps(binding.as_dict())
    )

    assert loaded == binding
    assert binding.plan_sha256 == plan.plan_sha256
    assert binding.run_id == supervisor["selected_run_id"]
    assert binding.report_sha256 == _canonical_sha256(
        supervisor["attempts"][0]
    )
    assert binding.supervisor_sha256 == _canonical_sha256(supervisor)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("reconnect_count",), 1, "supervisor is not admissible"),
        (("attempt_count",), 2, "supervisor is not admissible"),
        (
            ("attempts", 0, "resource_safety_passed"),
            False,
            "capture report is not admissible",
        ),
        (
            ("attempts", 0, "started_wall_ns"),
            2_000_000_031_000_000_000,
            "capture report is not admissible",
        ),
        (
            ("attempts", 0, "symbol_event_counts", "SOLUSDT"),
            {},
            "event coverage is not admissible",
        ),
        (
            ("attempts", 0, "writer_message_count"),
            251,
            "capture counts differ",
        ),
        (
            ("attempts", 0, "symbol_event_counts", "BTCUSDT", "bookTicker"),
            33,
            "symbol totals differ",
        ),
    ],
)
def test_admission_fails_closed_on_capture_defects(
    path: tuple[object, ...],
    value: object,
    message: str,
) -> None:
    plan = _plan()
    supervisor = deepcopy(_supervisor(plan, 0))
    target: object = supervisor
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        bind_round74_event_cohort_supervisor(
            plan,
            slot_ordinal=0,
            supervisor_payload=supervisor,
        )


def test_binding_loader_rejects_noncanonical_field_types() -> None:
    binding = _bindings(_plan())[0]
    payload = binding.as_dict()
    payload["message_count"] = str(payload["message_count"])
    unsigned = dict(payload)
    unsigned.pop("binding_sha256")
    payload["binding_sha256"] = _canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="static policy differs"):
        load_round74_event_cohort_binding(json.dumps(payload))


def test_complete_admitted_cohort_builds_leak_resistant_partition() -> None:
    plan = _plan()
    bindings = _bindings(plan)
    partition = build_round74_event_run_partition(plan, bindings)

    assert [entry.role for entry in partition.entries] == [
        "training",
        "training",
        "tuning",
        "test",
    ]
    assert len(partition.partition_sha256) == 64
    assert (
        partition.entries[2].eligible_anchor_start_wall_ns
        == bindings[2].capture_start_wall_ns
        + ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS
    )
    assert (
        partition.entries[1].eligible_anchor_end_wall_ns
        == bindings[1].capture_end_wall_ns
        - ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
    )
    partition.validate()


def test_partition_requires_every_unique_frozen_slot() -> None:
    plan = _plan()
    bindings = _bindings(plan)

    with pytest.raises(ValueError, match="missing or extra"):
        build_round74_event_run_partition(plan, bindings[:-1])
    with pytest.raises(ValueError, match="identity is duplicated"):
        build_round74_event_run_partition(
            plan,
            (*bindings[:-1], bindings[0]),
        )
    reassigned = Round74EventCohortRunBinding(
        **{
            **bindings[2].__dict__,
            "role": "test",
        }
    )
    with pytest.raises(ValueError, match="binding role differs"):
        build_round74_event_run_partition(
            plan,
            (*bindings[:2], reassigned, bindings[3]),
        )
