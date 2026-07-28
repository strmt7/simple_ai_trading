"""Deterministic admission for prospective Round 74 connection epochs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

from .impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortPlan,
    Round74SegmentedCohortSlotOutcome,
    Round74SegmentedTransportEpochAudit,
    audit_round74_v10_transport_epoch,
    bind_round74_segmented_probe_supervisor,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
    ImpactAbsorptionStore,
)


ROUND74_SEGMENTED_SLOT_ADJUDICATION_SCHEMA_VERSION = (
    "round-074-segmented-slot-adjudication-v1"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _strict_json_mapping(raw_text: str, label: str) -> Mapping[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Round 74 {label} has duplicate JSON keys")
            output[key] = value
        return output

    try:
        parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Round 74 {label} is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Round 74 {label} root differs")
    return parsed


def _validate_supervisor_envelope(supervisor: Mapping[str, object]) -> None:
    if (
        supervisor.get("schema_version")
        != "round-074-capture-supervisor-report-v1"
        or supervisor.get("design_sha256") != ROUND74_CAPTURE_DESIGN_SHA256
        or supervisor.get("capture_schema_version")
        != IMPACT_CAPTURE_V10_SCHEMA_VERSION
        or supervisor.get("capture_contract_sha256")
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or supervisor.get("status") not in {"completed", "failed"}
        or supervisor.get("qualification_passed") is not False
        or supervisor.get("attempt_count") != 1
        or supervisor.get("reconnect_count") != 0
        or supervisor.get("reconnect_delays_seconds") != []
        or supervisor.get("attempt_evidence_combined") is not False
        or not isinstance(supervisor.get("startup_errors"), list)
        or not isinstance(supervisor.get("attempts"), list)
    ):
        raise ValueError("Round 74 segmented supervisor envelope differs")


def _validate_startup_transport(supervisor: Mapping[str, object]) -> None:
    _validate_supervisor_envelope(supervisor)
    startup_errors = supervisor["startup_errors"]
    terminal_error = supervisor.get("terminal_error")
    if (
        supervisor.get("status") != "failed"
        or supervisor.get("selected_run_id") != ""
        or supervisor.get("attempts") != []
        or len(startup_errors) != 1
        or not isinstance(startup_errors[0], str)
        or not startup_errors[0]
        or terminal_error != startup_errors[0]
    ):
        raise ValueError("Round 74 segmented startup transport differs")


@dataclass(frozen=True)
class Round74SegmentedSlotAdjudication:
    """Hash-bound proof that one supervisor result was classified mechanically."""

    plan_sha256: str
    slot_ordinal: int
    supervisor_json: str
    outcome: Round74SegmentedCohortSlotOutcome
    epoch_audit: Round74SegmentedTransportEpochAudit | None
    schema_version: str = ROUND74_SEGMENTED_SLOT_ADJUDICATION_SCHEMA_VERSION

    def validate(self, plan: Round74SegmentedCohortPlan) -> None:
        plan.validate()
        slot = plan.slot(self.slot_ordinal)
        supervisor = _strict_json_mapping(
            self.supervisor_json,
            "segmented supervisor",
        )
        if (
            self.schema_version
            != ROUND74_SEGMENTED_SLOT_ADJUDICATION_SCHEMA_VERSION
            or self.plan_sha256 != plan.plan_sha256
            or self.outcome.plan_sha256 != plan.plan_sha256
            or self.outcome.slot_ordinal != slot.ordinal
            or self.outcome.role != slot.role
            or self.supervisor_json != _canonical_json(supervisor)
        ):
            raise ValueError("Round 74 segmented slot adjudication differs")
        self.outcome.validate()
        supervisor_sha256 = _canonical_sha256(supervisor)
        if self.epoch_audit is None:
            _validate_startup_transport(supervisor)
            if (
                self.outcome.status != "transport_excluded"
                or self.outcome.reason_code != "startup_transport"
                or self.outcome.binding is not None
                or self.outcome.evidence_sha256 != supervisor_sha256
            ):
                raise ValueError(
                    "Round 74 segmented startup adjudication differs"
                )
            return
        self.epoch_audit.validate()
        if self.epoch_audit.admission_supported:
            binding = bind_round74_segmented_probe_supervisor(
                plan,
                slot_ordinal=slot.ordinal,
                supervisor_payload=supervisor,
                fresh_epoch_audit_payload=self.epoch_audit.as_dict(),
            )
            if (
                self.outcome.status != "admitted"
                or self.outcome.reason_code != "admitted"
                or self.outcome.binding is None
                or self.outcome.binding.as_dict() != binding.as_dict()
                or self.outcome.evidence_sha256 != binding.binding_sha256
            ):
                raise ValueError(
                    "Round 74 segmented admitted adjudication differs"
                )
            return
        try:
            bind_round74_segmented_probe_supervisor(
                plan,
                slot_ordinal=slot.ordinal,
                supervisor_payload=supervisor,
                fresh_epoch_audit_payload=self.epoch_audit.as_dict(),
            )
        except ValueError as exc:
            if str(exc) != "Round 74 segmented cohort epoch audit differs":
                raise ValueError(
                    "Round 74 segmented excluded supervisor differs"
                ) from exc
        else:
            raise ValueError(
                "Round 74 segmented qualifying epoch was incorrectly excluded"
            )
        if (
            self.epoch_audit.terminal_status != "transport_ended"
            or self.outcome.status != "transport_excluded"
            or self.outcome.reason_code != "in_run_transport"
            or self.outcome.binding is not None
            or self.outcome.evidence_sha256
            != self.epoch_audit.epoch_audit_sha256
        ):
            raise ValueError("Round 74 segmented exclusion adjudication differs")

    @property
    def adjudication_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "slot_ordinal": self.slot_ordinal,
            "supervisor": dict(
                _strict_json_mapping(
                    self.supervisor_json,
                    "segmented supervisor",
                )
            ),
            "outcome": self.outcome.as_dict(),
            "epoch_audit": (
                None if self.epoch_audit is None else self.epoch_audit.as_dict()
            ),
        }
        if include_sha256:
            payload["adjudication_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        plan: Round74SegmentedCohortPlan,
        value: Mapping[str, object],
    ) -> Round74SegmentedSlotAdjudication:
        payload = dict(value)
        claimed = str(payload.pop("adjudication_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented adjudication digest differs")
        supervisor = payload.get("supervisor")
        outcome = payload.get("outcome")
        epoch = payload.get("epoch_audit")
        if (
            not isinstance(supervisor, Mapping)
            or not isinstance(outcome, Mapping)
            or epoch is not None
            and not isinstance(epoch, Mapping)
        ):
            raise ValueError("Round 74 segmented adjudication payload differs")
        selected = cls(
            plan_sha256=str(payload["plan_sha256"]),
            slot_ordinal=int(payload["slot_ordinal"]),
            supervisor_json=_canonical_json(supervisor),
            outcome=Round74SegmentedCohortSlotOutcome.from_dict(outcome),
            epoch_audit=(
                None
                if epoch is None
                else Round74SegmentedTransportEpochAudit.from_dict(epoch)
            ),
            schema_version=str(payload["schema_version"]),
        )
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented adjudication policy differs")
        selected.validate(plan)
        return selected


def adjudicate_round74_segmented_supervisor(
    plan: Round74SegmentedCohortPlan,
    *,
    slot_ordinal: int,
    supervisor_payload: Mapping[str, object],
    epoch_audit: Round74SegmentedTransportEpochAudit | None,
) -> Round74SegmentedSlotAdjudication:
    """Classify one result without market-, label-, or model-dependent choice."""

    plan.validate()
    slot = plan.slot(slot_ordinal)
    supervisor = dict(supervisor_payload)
    _validate_supervisor_envelope(supervisor)
    supervisor_json = _canonical_json(supervisor)
    supervisor_sha256 = _canonical_sha256(supervisor)
    attempts = supervisor["attempts"]
    if not attempts:
        if epoch_audit is not None:
            raise ValueError("Round 74 startup result cannot have an epoch audit")
        _validate_startup_transport(supervisor)
        outcome = Round74SegmentedCohortSlotOutcome(
            plan_sha256=plan.plan_sha256,
            slot_ordinal=slot.ordinal,
            role=slot.role,
            status="transport_excluded",
            reason_code="startup_transport",
            evidence_sha256=supervisor_sha256,
        )
    else:
        if epoch_audit is None:
            raise ValueError("Round 74 in-run result requires an epoch audit")
        epoch_audit.validate()
        if epoch_audit.admission_supported:
            binding = bind_round74_segmented_probe_supervisor(
                plan,
                slot_ordinal=slot.ordinal,
                supervisor_payload=supervisor,
                fresh_epoch_audit_payload=epoch_audit.as_dict(),
            )
            outcome = Round74SegmentedCohortSlotOutcome(
                plan_sha256=plan.plan_sha256,
                slot_ordinal=slot.ordinal,
                role=slot.role,
                status="admitted",
                reason_code="admitted",
                evidence_sha256=binding.binding_sha256,
                binding=binding,
            )
        else:
            try:
                bind_round74_segmented_probe_supervisor(
                    plan,
                    slot_ordinal=slot.ordinal,
                    supervisor_payload=supervisor,
                    fresh_epoch_audit_payload=epoch_audit.as_dict(),
                )
            except ValueError as exc:
                if str(exc) != "Round 74 segmented cohort epoch audit differs":
                    raise
            else:
                raise ValueError(
                    "Round 74 qualifying epoch cannot be transport-excluded"
                )
            outcome = Round74SegmentedCohortSlotOutcome(
                plan_sha256=plan.plan_sha256,
                slot_ordinal=slot.ordinal,
                role=slot.role,
                status="transport_excluded",
                reason_code="in_run_transport",
                evidence_sha256=epoch_audit.epoch_audit_sha256,
            )
    selected = Round74SegmentedSlotAdjudication(
        plan_sha256=plan.plan_sha256,
        slot_ordinal=slot.ordinal,
        supervisor_json=supervisor_json,
        outcome=outcome,
        epoch_audit=epoch_audit,
    )
    selected.validate(plan)
    return selected


def audit_and_adjudicate_round74_segmented_supervisor(
    plan: Round74SegmentedCohortPlan,
    *,
    slot_ordinal: int,
    supervisor_payload: Mapping[str, object],
    store: ImpactAbsorptionStore | None,
) -> Round74SegmentedSlotAdjudication:
    """Run the fresh epoch audit whenever the supervisor created a run."""

    attempts = supervisor_payload.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Round 74 segmented supervisor attempts differ")
    if not attempts:
        epoch_audit = None
    else:
        if len(attempts) != 1 or not isinstance(attempts[0], Mapping):
            raise ValueError("Round 74 segmented supervisor attempts differ")
        if store is None:
            raise ValueError("Round 74 in-run adjudication requires a store")
        epoch_audit = audit_round74_v10_transport_epoch(
            store,
            run_id=str(attempts[0].get("run_id", "")),
        )
    return adjudicate_round74_segmented_supervisor(
        plan,
        slot_ordinal=slot_ordinal,
        supervisor_payload=supervisor_payload,
        epoch_audit=epoch_audit,
    )


def load_round74_segmented_slot_adjudication(
    raw_text: str,
    *,
    plan: Round74SegmentedCohortPlan,
) -> Round74SegmentedSlotAdjudication:
    return Round74SegmentedSlotAdjudication.from_dict(
        plan,
        _strict_json_mapping(raw_text, "segmented adjudication"),
    )


__all__ = [
    "ROUND74_SEGMENTED_SLOT_ADJUDICATION_SCHEMA_VERSION",
    "Round74SegmentedSlotAdjudication",
    "adjudicate_round74_segmented_supervisor",
    "audit_and_adjudicate_round74_segmented_supervisor",
    "load_round74_segmented_slot_adjudication",
]
