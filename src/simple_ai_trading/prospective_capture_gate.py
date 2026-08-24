"""Venue-isolated source-continuity gates for prospective market research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


_VENUES = frozenset({"binance", "polymarket"})
_TERMINAL_STATUSES = frozenset({"passed", "failed", "missed"})
_SLOT_STATUSES = _TERMINAL_STATUSES | {"running"}


@dataclass(frozen=True, slots=True)
class PlannedCaptureSlot:
    """One fixed, target-blind capture window with an isolated storage namespace."""

    slot_id: str
    role: str
    scheduled_start_wall_ns: int
    scheduled_end_wall_ns: int
    terminal_grace_ns: int
    storage_namespace: str
    planned_capacity: int

    def validate(self) -> None:
        integer_values = (
            self.scheduled_start_wall_ns,
            self.scheduled_end_wall_ns,
            self.terminal_grace_ns,
            self.planned_capacity,
        )
        if not self.slot_id or not self.role or not self.storage_namespace:
            raise ValueError("capture slot identity differs")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise ValueError("capture slot numeric fields differ")
        if (
            self.scheduled_start_wall_ns < 0
            or self.scheduled_end_wall_ns <= self.scheduled_start_wall_ns
            or self.terminal_grace_ns < 0
            or self.planned_capacity <= 0
        ):
            raise ValueError("capture slot bounds differ")


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """A fixed prospective plan; quota values use venue-specific capacity units."""

    campaign_id: str
    venue: str
    slots: tuple[PlannedCaptureSlot, ...]
    minimum_capacity_by_role: Mapping[str, int]

    def validate(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign identity differs")
        if self.venue not in _VENUES:
            raise ValueError("campaign venue differs")
        if not self.slots:
            raise ValueError("campaign slots are empty")
        for slot in self.slots:
            slot.validate()
        slot_ids = tuple(slot.slot_id for slot in self.slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("capture slot identities must be unique")
        namespaces = tuple(slot.storage_namespace for slot in self.slots)
        if len(set(namespaces)) != len(namespaces):
            raise ValueError("capture storage namespaces must be unique")
        if tuple(self.slots) != tuple(
            sorted(self.slots, key=lambda slot: slot.scheduled_start_wall_ns)
        ):
            raise ValueError("capture slots are not chronological")
        if any(
            current.scheduled_start_wall_ns < previous.scheduled_end_wall_ns
            for previous, current in zip(self.slots, self.slots[1:], strict=False)
        ):
            raise ValueError("capture slots overlap")
        roles = {slot.role for slot in self.slots}
        if set(self.minimum_capacity_by_role) != roles:
            raise ValueError("campaign role quota keys differ")
        for role, minimum in self.minimum_capacity_by_role.items():
            planned = sum(
                slot.planned_capacity for slot in self.slots if slot.role == role
            )
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum <= 0
                or minimum > planned
            ):
                raise ValueError("campaign role quota differs")


@dataclass(frozen=True, slots=True)
class SlotEvidence:
    """Target-free terminal or in-progress evidence for one planned window."""

    slot_id: str
    storage_namespace: str
    status: str
    terminal: bool
    source_gate_passed: bool
    admitted_capacity: int
    wal_present: bool = False
    storage_quarantined: bool = False
    target_accessed: bool = False
    outcome_accessed: bool = False

    def validate(self, slot: PlannedCaptureSlot) -> None:
        if self.status not in _SLOT_STATUSES:
            raise ValueError("slot evidence status differs")
        expected_terminal = self.status in _TERMINAL_STATUSES
        if self.terminal is not expected_terminal:
            raise ValueError("slot evidence terminal flag differs")
        if self.source_gate_passed is not (self.status == "passed"):
            raise ValueError("slot evidence source gate differs")
        capacity_valid = (
            0 < self.admitted_capacity <= slot.planned_capacity
            if self.status == "passed"
            else self.admitted_capacity == 0
        )
        if (
            isinstance(self.admitted_capacity, bool)
            or not isinstance(self.admitted_capacity, int)
            or not capacity_valid
        ):
            raise ValueError("slot evidence admitted capacity differs")
        boolean_values = (
            self.wal_present,
            self.storage_quarantined,
            self.target_accessed,
            self.outcome_accessed,
        )
        if any(not isinstance(value, bool) for value in boolean_values):
            raise ValueError("slot evidence flags differ")


@dataclass(frozen=True, slots=True)
class CampaignSourceGateReport:
    campaign_id: str
    venue: str
    status: str
    blockers: tuple[str, ...]
    capacity_by_role: Mapping[str, int]
    recoverable: bool
    unrecoverable_roles: tuple[str, ...]
    source_population_ready: bool
    next_slot_id: str | None
    next_slot_start_permitted: bool
    model_or_target_access_permitted: bool = False
    edge_claim_permitted: bool = False

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _evidence_by_slot(
    plan: CampaignPlan,
    evidence: Sequence[SlotEvidence],
) -> dict[str, SlotEvidence]:
    slots = {slot.slot_id: slot for slot in plan.slots}
    indexed: dict[str, SlotEvidence] = {}
    for item in evidence:
        if item.slot_id not in slots:
            raise ValueError("unknown slot evidence")
        if item.slot_id in indexed:
            raise ValueError("duplicate slot evidence")
        item.validate(slots[item.slot_id])
        indexed[item.slot_id] = item
    return indexed


def _slot_blockers(
    slot: PlannedCaptureSlot,
    evidence: SlotEvidence | None,
    *,
    observed_wall_ns: int,
) -> tuple[str, ...]:
    terminal_deadline = slot.scheduled_end_wall_ns + slot.terminal_grace_ns
    if evidence is None:
        if observed_wall_ns > terminal_deadline:
            return (f"{slot.slot_id}: elapsed slot lacks a terminal disposition",)
        return ()
    reasons: list[str] = []
    if observed_wall_ns < slot.scheduled_start_wall_ns:
        reasons.append("evidence predates its fixed slot")
    if evidence.storage_namespace != slot.storage_namespace:
        reasons.append("storage namespace differs")
    if evidence.target_accessed or evidence.outcome_accessed:
        reasons.append("target or outcome access occurred")
    if evidence.status == "passed" and evidence.wal_present:
        reasons.append("passed slot retains a WAL")
    if evidence.status == "failed" and not evidence.storage_quarantined:
        reasons.append("failed storage is not quarantined")
    if evidence.status == "running" and observed_wall_ns > terminal_deadline:
        reasons.append("running evidence exceeded its terminal grace")
    return tuple(f"{slot.slot_id}: {reason}" for reason in reasons)


def evaluate_campaign_source_gate(
    plan: CampaignPlan,
    evidence: Sequence[SlotEvidence],
    *,
    observed_wall_ns: int,
) -> CampaignSourceGateReport:
    """Evaluate continuity without opening targets, outcomes, databases, or WALs."""

    plan.validate()
    if (
        isinstance(observed_wall_ns, bool)
        or not isinstance(observed_wall_ns, int)
        or observed_wall_ns < 0
    ):
        raise ValueError("campaign observation time differs")
    indexed = _evidence_by_slot(plan, evidence)
    blockers = tuple(
        blocker
        for slot in plan.slots
        for blocker in _slot_blockers(
            slot,
            indexed.get(slot.slot_id),
            observed_wall_ns=observed_wall_ns,
        )
    )

    capacity = {
        role: sum(
            item.admitted_capacity
            for slot in plan.slots
            if slot.role == role
            for item in (indexed.get(slot.slot_id),)
            if item is not None and item.status == "passed"
        )
        for role in sorted(plan.minimum_capacity_by_role)
    }
    possible_remaining = {
        role: sum(
            slot.planned_capacity
            for slot in plan.slots
            if slot.role == role
            and (
                (item := indexed.get(slot.slot_id)) is None or item.status == "running"
            )
            and observed_wall_ns <= slot.scheduled_end_wall_ns + slot.terminal_grace_ns
        )
        for role in sorted(plan.minimum_capacity_by_role)
    }
    unrecoverable = tuple(
        role
        for role in sorted(plan.minimum_capacity_by_role)
        if capacity[role] + possible_remaining[role]
        < plan.minimum_capacity_by_role[role]
    )
    all_terminal = len(indexed) == len(plan.slots) and all(
        item.terminal for item in indexed.values()
    )
    quotas_passed = all(
        capacity[role] >= minimum
        for role, minimum in plan.minimum_capacity_by_role.items()
    )
    source_ready = not blockers and all_terminal and quotas_passed
    next_slot = next(
        (
            slot
            for slot in plan.slots
            if slot.slot_id not in indexed
            and slot.scheduled_start_wall_ns
            <= observed_wall_ns
            <= slot.scheduled_end_wall_ns + slot.terminal_grace_ns
        ),
        None,
    )

    if blockers:
        status = "blocked_integrity"
    elif unrecoverable:
        status = "blocked_unrecoverable"
    elif source_ready:
        status = "source_population_ready"
    elif next_slot is not None:
        status = "ready_for_next_slot"
    else:
        status = "waiting_for_fixed_slot"

    return CampaignSourceGateReport(
        campaign_id=plan.campaign_id,
        venue=plan.venue,
        status=status,
        blockers=blockers,
        capacity_by_role=capacity,
        recoverable=not unrecoverable,
        unrecoverable_roles=unrecoverable,
        source_population_ready=source_ready,
        next_slot_id=next_slot.slot_id if next_slot is not None else None,
        next_slot_start_permitted=status == "ready_for_next_slot",
    )


__all__ = [
    "CampaignPlan",
    "CampaignSourceGateReport",
    "PlannedCaptureSlot",
    "SlotEvidence",
    "evaluate_campaign_source_gate",
]
