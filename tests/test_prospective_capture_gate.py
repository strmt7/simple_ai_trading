from __future__ import annotations

from dataclasses import replace

import pytest

from simple_ai_trading.prospective_capture_gate import (
    CampaignPlan,
    PlannedCaptureSlot,
    SlotEvidence,
    evaluate_campaign_source_gate,
)


def _slot(
    slot_id: str,
    *,
    role: str,
    start: int,
    storage: str,
    capacity: int = 10,
) -> PlannedCaptureSlot:
    return PlannedCaptureSlot(
        slot_id=slot_id,
        role=role,
        scheduled_start_wall_ns=start,
        scheduled_end_wall_ns=start + 100,
        terminal_grace_ns=10,
        storage_namespace=storage,
        planned_capacity=capacity,
    )


def _plan(*slots: PlannedCaptureSlot, venue: str = "binance") -> CampaignPlan:
    return CampaignPlan(
        campaign_id=f"{venue}-source-recovery-v1",
        venue=venue,
        slots=slots,
        minimum_capacity_by_role={
            role: sum(slot.planned_capacity for slot in slots if slot.role == role)
            for role in {slot.role for slot in slots}
        },
    )


def _passed(slot: PlannedCaptureSlot) -> SlotEvidence:
    return SlotEvidence(
        slot_id=slot.slot_id,
        storage_namespace=slot.storage_namespace,
        status="passed",
        terminal=True,
        source_gate_passed=True,
        admitted_capacity=slot.planned_capacity,
    )


def _failed_with_quarantined_wal(slot: PlannedCaptureSlot) -> SlotEvidence:
    return SlotEvidence(
        slot_id=slot.slot_id,
        storage_namespace=slot.storage_namespace,
        status="failed",
        terminal=True,
        source_gate_passed=False,
        admitted_capacity=0,
        wal_present=True,
        storage_quarantined=True,
    )


def test_quarantined_binance_slot_does_not_poison_later_unique_slot() -> None:
    failed = _slot("train-001", role="training", start=0, storage="train-001.db")
    current = _slot("train-002", role="training", start=200, storage="train-002.db")
    plan = replace(
        _plan(failed, current),
        minimum_capacity_by_role={"training": 10},
    )

    report = evaluate_campaign_source_gate(
        plan,
        (_failed_with_quarantined_wal(failed),),
        observed_wall_ns=250,
    )

    assert report.status == "ready_for_next_slot"
    assert report.next_slot_id == "train-002"
    assert report.next_slot_start_permitted is True
    assert report.recoverable is True
    assert report.blockers == ()
    assert report.model_or_target_access_permitted is False
    assert report.edge_claim_permitted is False


def test_shared_storage_namespace_is_rejected_before_evidence() -> None:
    first = _slot("a", role="primary", start=0, storage="shared.db")
    second = _slot("b", role="primary", start=200, storage="shared.db")

    with pytest.raises(ValueError, match="storage namespaces must be unique"):
        evaluate_campaign_source_gate(
            replace(
                _plan(first, second, venue="polymarket"),
                minimum_capacity_by_role={"primary": 10},
            ),
            (),
            observed_wall_ns=0,
        )


def test_unquarantined_failure_blocks_later_polymarket_window() -> None:
    failed = _slot("stage-a", role="primary", start=0, storage="stage-a.db")
    current = _slot("stage-b", role="primary", start=200, storage="stage-b.db")
    plan = replace(
        _plan(failed, current, venue="polymarket"),
        minimum_capacity_by_role={"primary": 10},
    )
    evidence = replace(
        _failed_with_quarantined_wal(failed),
        storage_quarantined=False,
    )

    report = evaluate_campaign_source_gate(
        plan,
        (evidence,),
        observed_wall_ns=250,
    )

    assert report.status == "blocked_integrity"
    assert report.next_slot_start_permitted is False
    assert report.recoverable is True
    assert report.blockers == ("stage-a: failed storage is not quarantined",)


def test_missing_elapsed_slot_and_stale_running_slot_fail_closed() -> None:
    missing = _slot("missing", role="training", start=0, storage="missing.db")
    stale = _slot("stale", role="training", start=200, storage="stale.db")
    future = _slot("future", role="training", start=400, storage="future.db")
    plan = replace(
        _plan(missing, stale, future),
        minimum_capacity_by_role={"training": 10},
    )
    running = SlotEvidence(
        slot_id="stale",
        storage_namespace="stale.db",
        status="running",
        terminal=False,
        source_gate_passed=False,
        admitted_capacity=0,
    )

    report = evaluate_campaign_source_gate(
        plan,
        (running,),
        observed_wall_ns=350,
    )

    assert report.status == "blocked_integrity"
    assert report.blockers == (
        "missing: elapsed slot lacks a terminal disposition",
        "stale: running evidence exceeded its terminal grace",
    )


def test_source_population_can_pass_without_opening_targets_or_claiming_edge() -> None:
    training = _slot("train", role="training", start=0, storage="train.db")
    tuning = _slot("tune", role="tuning", start=200, storage="tune.db")
    test = _slot("test", role="test", start=400, storage="test.db")
    plan = _plan(training, tuning, test)

    report = evaluate_campaign_source_gate(
        plan,
        (_passed(training), _passed(tuning), _passed(test)),
        observed_wall_ns=600,
    )

    assert report.status == "source_population_ready"
    assert report.source_population_ready is True
    assert report.capacity_by_role == {"test": 10, "training": 10, "tuning": 10}
    assert report.next_slot_id is None
    assert report.next_slot_start_permitted is False
    assert report.model_or_target_access_permitted is False
    assert report.edge_claim_permitted is False
    assert report.asdict()["status"] == "source_population_ready"


def test_failed_role_quota_becomes_unrecoverable() -> None:
    only = _slot("only-test", role="test", start=0, storage="only-test.db")

    report = evaluate_campaign_source_gate(
        _plan(only),
        (_failed_with_quarantined_wal(only),),
        observed_wall_ns=200,
    )

    assert report.status == "blocked_unrecoverable"
    assert report.recoverable is False
    assert report.unrecoverable_roles == ("test",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"wal_present": True}, "passed slot retains a WAL"),
        ({"target_accessed": True}, "target or outcome access occurred"),
        ({"outcome_accessed": True}, "target or outcome access occurred"),
        ({"storage_namespace": "other.db"}, "storage namespace differs"),
    ],
)
def test_tampered_or_leaky_passed_evidence_blocks(
    mutation: dict[str, object],
    message: str,
) -> None:
    slot = _slot("primary", role="primary", start=0, storage="primary.db")
    evidence = replace(_passed(slot), **mutation)

    report = evaluate_campaign_source_gate(
        _plan(slot, venue="polymarket"),
        (evidence,),
        observed_wall_ns=200,
    )

    assert report.status == "blocked_integrity"
    assert report.blockers == (f"primary: {message}",)


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (
            SlotEvidence("unknown", "unknown.db", "missed", True, False, 0),
            "unknown slot evidence",
        ),
        (
            SlotEvidence("primary", "primary.db", "mystery", True, False, 0),
            "slot evidence status differs",
        ),
        (
            SlotEvidence("primary", "primary.db", "passed", False, True, 10),
            "terminal flag differs",
        ),
        (
            SlotEvidence("primary", "primary.db", "failed", True, True, 0),
            "source gate differs",
        ),
        (
            SlotEvidence("primary", "primary.db", "missed", True, False, 1),
            "admitted capacity differs",
        ),
    ],
)
def test_malformed_slot_evidence_is_rejected(
    evidence: SlotEvidence,
    message: str,
) -> None:
    slot = _slot("primary", role="primary", start=0, storage="primary.db")

    with pytest.raises(ValueError, match=message):
        evaluate_campaign_source_gate(
            _plan(slot, venue="polymarket"),
            (evidence,),
            observed_wall_ns=0,
        )


def test_duplicate_evidence_and_future_evidence_are_rejected() -> None:
    slot = _slot("primary", role="primary", start=100, storage="primary.db")
    passed = _passed(slot)

    with pytest.raises(ValueError, match="duplicate slot evidence"):
        evaluate_campaign_source_gate(
            _plan(slot, venue="polymarket"),
            (passed, passed),
            observed_wall_ns=200,
        )

    report = evaluate_campaign_source_gate(
        _plan(slot, venue="polymarket"),
        (passed,),
        observed_wall_ns=0,
    )
    assert report.status == "blocked_integrity"
    assert report.blockers == ("primary: evidence predates its fixed slot",)


def test_campaign_waits_without_permission_before_the_next_fixed_slot() -> None:
    slot = _slot("future", role="training", start=100, storage="future.db")

    report = evaluate_campaign_source_gate(
        _plan(slot),
        (),
        observed_wall_ns=0,
    )

    assert report.status == "waiting_for_fixed_slot"
    assert report.next_slot_start_permitted is False


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (CampaignPlan("", "binance", (), {}), "campaign identity differs"),
        (CampaignPlan("x", "other", (), {}), "campaign venue differs"),
        (CampaignPlan("x", "binance", (), {}), "campaign slots are empty"),
        (
            CampaignPlan(
                "x",
                "binance",
                (_slot("x", role="training", start=0, storage="x.db"),),
                {"other": 1},
            ),
            "role quota keys differ",
        ),
    ],
)
def test_invalid_plan_is_rejected(plan: CampaignPlan, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_campaign_source_gate(plan, (), observed_wall_ns=0)


@pytest.mark.parametrize(
    ("slot", "message"),
    [
        (
            _slot("", role="training", start=0, storage="x.db"),
            "capture slot identity differs",
        ),
        (
            replace(
                _slot("x", role="training", start=0, storage="x.db"),
                terminal_grace_ns=True,
            ),
            "capture slot numeric fields differ",
        ),
        (
            replace(
                _slot("x", role="training", start=0, storage="x.db"),
                planned_capacity=0,
            ),
            "capture slot bounds differ",
        ),
    ],
)
def test_invalid_slot_is_rejected(slot: PlannedCaptureSlot, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_campaign_source_gate(
            CampaignPlan("x", "binance", (slot,), {"training": 1}),
            (),
            observed_wall_ns=0,
        )


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (
            CampaignPlan(
                "x",
                "binance",
                (
                    _slot("same", role="training", start=0, storage="a.db"),
                    _slot("same", role="training", start=200, storage="b.db"),
                ),
                {"training": 1},
            ),
            "capture slot identities must be unique",
        ),
        (
            CampaignPlan(
                "x",
                "binance",
                (
                    _slot("later", role="training", start=200, storage="later.db"),
                    _slot("earlier", role="training", start=0, storage="earlier.db"),
                ),
                {"training": 1},
            ),
            "capture slots are not chronological",
        ),
        (
            CampaignPlan(
                "x",
                "binance",
                (
                    _slot("first", role="training", start=0, storage="first.db"),
                    _slot("overlap", role="training", start=50, storage="overlap.db"),
                ),
                {"training": 1},
            ),
            "capture slots overlap",
        ),
        (
            CampaignPlan(
                "x",
                "binance",
                (_slot("x", role="training", start=0, storage="x.db"),),
                {"training": 11},
            ),
            "campaign role quota differs",
        ),
    ],
)
def test_structurally_invalid_plan_is_rejected(
    plan: CampaignPlan,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_campaign_source_gate(plan, (), observed_wall_ns=0)


def test_invalid_evidence_flags_and_observation_time_are_rejected() -> None:
    slot = _slot("primary", role="primary", start=0, storage="primary.db")
    invalid_flag = replace(_passed(slot), wal_present=1)

    with pytest.raises(ValueError, match="slot evidence flags differ"):
        evaluate_campaign_source_gate(
            _plan(slot, venue="polymarket"),
            (invalid_flag,),
            observed_wall_ns=0,
        )
    with pytest.raises(ValueError, match="campaign observation time differs"):
        evaluate_campaign_source_gate(
            _plan(slot, venue="polymarket"),
            (),
            observed_wall_ns=-1,
        )
