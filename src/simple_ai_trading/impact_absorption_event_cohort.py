"""Predeclared capture slots and fail-closed Round 74 cohort admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS,
    ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS,
    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
    ROUND74_EVENT_PARTITION_ROLES,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)


ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION = "round-074-event-cohort-plan-v1"
ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION = (
    "round-074-event-cohort-binding-v1"
)
ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS = 3_600_000_000_000
ROUND74_EVENT_COHORT_SLOT_PERIOD_NS = 3_660_000_000_000
ROUND74_EVENT_COHORT_START_TOLERANCE_NS = 30_000_000_000
ROUND74_EVENT_COHORT_END_OVERHEAD_NS = 120_000_000_000
ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS = {
    "training": 120,
    "tuning": 24,
    "test": 24,
}
ROUND74_EVENT_COHORT_MAXIMUM_SLOTS = 720

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GLOBAL_EVENT_TYPES = frozenset(
    {
        "aggTrade",
        "bookTicker",
        "depthSnapshot",
        "depthUpdate",
        "exchangeInfo",
        "forceOrder",
        "markPriceUpdate",
        "openInterest",
        "serverTime",
    }
)
_SYMBOL_EVENT_TYPES = frozenset(
    {
        "aggTrade",
        "bookTicker",
        "depthSnapshot",
        "depthUpdate",
        "forceOrder",
        "markPriceUpdate",
        "openInterest",
        "synchronizedDepthUpdate",
    }
)
_SYMBOL_GLOBAL_EVENT_TYPES = _SYMBOL_EVENT_TYPES - {
    "synchronizedDepthUpdate"
}


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


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 cohort {label} digest is invalid")
    return selected


def _strict_json_object(raw_text: str, label: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(
                    f"Round 74 cohort {label} has duplicate JSON keys"
                )
            output[key] = value
        return output

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, dict):
        raise ValueError(f"Round 74 cohort {label} root differs")
    return parsed


@dataclass(frozen=True)
class Round74EventCohortSlot:
    """One non-replaceable scheduled one-hour capture."""

    ordinal: int
    role: str
    scheduled_start_wall_ns: int
    scheduled_end_wall_ns: int
    start_window_end_wall_ns: int

    def validate(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or int(self.scheduled_start_wall_ns) <= 0
            or int(self.scheduled_end_wall_ns)
            - int(self.scheduled_start_wall_ns)
            != ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS
            or int(self.start_window_end_wall_ns)
            - int(self.scheduled_start_wall_ns)
            != ROUND74_EVENT_COHORT_START_TOLERANCE_NS
        ):
            raise ValueError("Round 74 cohort slot identity differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "ordinal": self.ordinal,
            "role": self.role,
            "scheduled_start_wall_ns": self.scheduled_start_wall_ns,
            "scheduled_end_wall_ns": self.scheduled_end_wall_ns,
            "start_window_end_wall_ns": self.start_window_end_wall_ns,
        }


@dataclass(frozen=True)
class Round74EventCohortPlan:
    """Compact deterministic schedule frozen before any cohort event exists."""

    scheduled_start_wall_ns: int
    training_slots: int = 120
    tuning_slots: int = 24
    test_slots: int = 24
    prerequisite_artifact_sha256: str = ""
    prerequisite_window_start_wall_ns: int = 0
    prerequisite_window_end_wall_ns: int = 0
    schema_version: str = ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION

    @property
    def total_slots(self) -> int:
        return self.training_slots + self.tuning_slots + self.test_slots

    def validate(self) -> None:
        counts = (self.training_slots, self.tuning_slots, self.test_slots)
        if (
            self.schema_version != ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION
            or isinstance(self.scheduled_start_wall_ns, bool)
            or not isinstance(self.scheduled_start_wall_ns, int)
            or self.scheduled_start_wall_ns <= 0
            or self.scheduled_start_wall_ns % 1_000_000_000 != 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in counts
            )
            or self.total_slots > ROUND74_EVENT_COHORT_MAXIMUM_SLOTS
        ):
            raise ValueError("Round 74 cohort plan identity differs")
        _require_sha256(self.prerequisite_artifact_sha256, "prerequisite")
        if not (
            0
            < int(self.prerequisite_window_start_wall_ns)
            < int(self.prerequisite_window_end_wall_ns)
            < int(self.scheduled_start_wall_ns)
        ):
            raise ValueError("Round 74 cohort prerequisite window differs")
        if self.slot(self.total_slots - 1).scheduled_end_wall_ns >= 2**64:
            raise ValueError("Round 74 cohort schedule exceeds timestamp range")

    def role_for_ordinal(self, ordinal: int) -> str:
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < self.total_slots
        ):
            raise ValueError("Round 74 cohort slot ordinal differs")
        if ordinal < self.training_slots:
            return "training"
        if ordinal < self.training_slots + self.tuning_slots:
            return "tuning"
        return "test"

    def slot(self, ordinal: int) -> Round74EventCohortSlot:
        role = self.role_for_ordinal(ordinal)
        start = (
            int(self.scheduled_start_wall_ns)
            + int(ordinal) * ROUND74_EVENT_COHORT_SLOT_PERIOD_NS
        )
        slot = Round74EventCohortSlot(
            ordinal=int(ordinal),
            role=role,
            scheduled_start_wall_ns=start,
            scheduled_end_wall_ns=(
                start + ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS
            ),
            start_window_end_wall_ns=(
                start + ROUND74_EVENT_COHORT_START_TOLERANCE_NS
            ),
        )
        slot.validate()
        return slot

    @property
    def plan_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "scheduled_start_wall_ns": self.scheduled_start_wall_ns,
            "role_counts": {
                "training": self.training_slots,
                "tuning": self.tuning_slots,
                "test": self.test_slots,
            },
            "total_slots": self.total_slots,
            "schedule_formula": {
                "ordinal_origin": 0,
                "slot_period_ns": ROUND74_EVENT_COHORT_SLOT_PERIOD_NS,
                "capture_duration_ns": (
                    ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS
                ),
                "start_tolerance_ns": (
                    ROUND74_EVENT_COHORT_START_TOLERANCE_NS
                ),
                "maximum_end_overhead_ns": (
                    ROUND74_EVENT_COHORT_END_OVERHEAD_NS
                ),
            },
            "capture_contract": {
                "provider": "Binance USD-M public production market data",
                "symbols": list(IMPACT_CAPTURE_SYMBOLS),
                "capture_schema_version": (
                    IMPACT_CAPTURE_V10_SCHEMA_VERSION
                ),
                "capture_report_schema_version": (
                    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
                ),
                "capture_design_sha256": ROUND74_CAPTURE_DESIGN_SHA256,
                "capture_contract_sha256": (
                    IMPACT_CAPTURE_V10_CONTRACT_SHA256
                ),
                "mode": "qualification",
                "maximum_reconnects": 0,
                "failed_or_missed_slot_replacement_permitted": False,
                "credentials_used": False,
                "orders_submitted": False,
            },
            "prerequisite": {
                "artifact_sha256": self.prerequisite_artifact_sha256,
                "window_start_wall_ns": (
                    self.prerequisite_window_start_wall_ns
                ),
                "window_end_wall_ns": self.prerequisite_window_end_wall_ns,
                "must_pass_before_first_slot": True,
                "prerequisite_capture_is_model_cohort_data": False,
            },
            "partition_policy": {
                "split_unit": "whole capture run",
                "role_order": list(ROUND74_EVENT_PARTITION_ROLES),
                "random_row_split_permitted": False,
                "minimum_purge_ns": (
                    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
                ),
                "minimum_embargo_ns": (
                    ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS
                ),
                "maximum_target_span_ns": (
                    ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
                ),
                "test_labels_or_outcomes_visible_before_pretest_seal": False,
            },
            "scope": {
                "purpose": "initial live microstructure viability cohort",
                "long_horizon_edge_claim_permitted": False,
                "years_of_historical_coverage_claim": False,
                "financial_edge_tested_by_plan": False,
                "profitability_claim": False,
                "trading_authority": False,
            },
        }
        if include_sha256:
            payload["plan_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EventCohortPlan:
        payload = dict(value)
        claimed = str(payload.pop("plan_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 cohort plan digest differs")
        roles = payload.get("role_counts")
        prerequisite = payload.get("prerequisite")
        if not isinstance(roles, Mapping) or not isinstance(
            prerequisite,
            Mapping,
        ):
            raise ValueError("Round 74 cohort plan sections differ")
        try:
            selected = cls(
                scheduled_start_wall_ns=int(payload["scheduled_start_wall_ns"]),
                training_slots=int(roles["training"]),
                tuning_slots=int(roles["tuning"]),
                test_slots=int(roles["test"]),
                prerequisite_artifact_sha256=str(
                    prerequisite["artifact_sha256"]
                ),
                prerequisite_window_start_wall_ns=int(
                    prerequisite["window_start_wall_ns"]
                ),
                prerequisite_window_end_wall_ns=int(
                    prerequisite["window_end_wall_ns"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 cohort plan payload differs") from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 cohort plan static policy differs")
        selected.validate()
        return selected


@dataclass(frozen=True)
class Round74EventCohortRunBinding:
    """One admitted report tied to exactly one predeclared slot."""

    plan_sha256: str
    slot_ordinal: int
    role: str
    run_id: str
    report_sha256: str
    supervisor_sha256: str
    capture_start_wall_ns: int
    capture_end_wall_ns: int
    message_count: int
    frame_count: int
    compressed_payload_bytes: int
    schema_version: str = ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION

    def validate(self) -> None:
        counts = (
            self.message_count,
            self.frame_count,
            self.compressed_payload_bytes,
        )
        if (
            self.schema_version
            != ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or isinstance(self.slot_ordinal, bool)
            or not isinstance(self.slot_ordinal, int)
            or self.slot_ordinal < 0
            or int(self.capture_start_wall_ns) <= 0
            or int(self.capture_end_wall_ns)
            <= int(self.capture_start_wall_ns)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in counts
            )
        ):
            raise ValueError("Round 74 cohort run binding differs")
        _require_sha256(self.plan_sha256, "plan")
        _require_sha256(self.report_sha256, "report")
        _require_sha256(self.supervisor_sha256, "supervisor")

    @property
    def binding_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload = {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "slot_ordinal": self.slot_ordinal,
            "role": self.role,
            "run_id": self.run_id,
            "report_sha256": self.report_sha256,
            "supervisor_sha256": self.supervisor_sha256,
            "capture_start_wall_ns": self.capture_start_wall_ns,
            "capture_end_wall_ns": self.capture_end_wall_ns,
            "message_count": self.message_count,
            "frame_count": self.frame_count,
            "compressed_payload_bytes": self.compressed_payload_bytes,
        }
        if include_sha256:
            payload["binding_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EventCohortRunBinding:
        payload = dict(value)
        claimed = str(payload.pop("binding_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 cohort binding digest differs")
        try:
            selected = cls(
                plan_sha256=str(payload["plan_sha256"]),
                slot_ordinal=int(payload["slot_ordinal"]),
                role=str(payload["role"]),
                run_id=str(payload["run_id"]),
                report_sha256=str(payload["report_sha256"]),
                supervisor_sha256=str(payload["supervisor_sha256"]),
                capture_start_wall_ns=int(
                    payload["capture_start_wall_ns"]
                ),
                capture_end_wall_ns=int(payload["capture_end_wall_ns"]),
                message_count=int(payload["message_count"]),
                frame_count=int(payload["frame_count"]),
                compressed_payload_bytes=int(
                    payload["compressed_payload_bytes"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 cohort binding payload differs"
            ) from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 cohort binding static policy differs")
        selected.validate()
        return selected


def bind_round74_event_cohort_supervisor(
    plan: Round74EventCohortPlan,
    *,
    slot_ordinal: int,
    supervisor_payload: Mapping[str, object],
) -> Round74EventCohortRunBinding:
    """Admit one reconnect-free qualified supervisor into its frozen slot."""

    plan.validate()
    slot = plan.slot(slot_ordinal)
    supervisor = dict(supervisor_payload)
    supervisor_sha256 = _canonical_sha256(supervisor)
    attempts = supervisor.get("attempts")
    if (
        supervisor.get("schema_version")
        != "round-074-capture-supervisor-report-v1"
        or supervisor.get("design_sha256")
        != ROUND74_CAPTURE_DESIGN_SHA256
        or supervisor.get("capture_schema_version")
        != IMPACT_CAPTURE_V10_SCHEMA_VERSION
        or supervisor.get("capture_contract_sha256")
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or supervisor.get("status") != "completed"
        or supervisor.get("qualification_passed") is not True
        or supervisor.get("attempt_count") != 1
        or supervisor.get("reconnect_count") != 0
        or supervisor.get("reconnect_delays_seconds") != []
        or supervisor.get("startup_errors") != []
        or supervisor.get("terminal_error") != ""
        or supervisor.get("attempt_evidence_combined") is not False
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or not isinstance(attempts[0], Mapping)
    ):
        raise ValueError("Round 74 cohort supervisor is not admissible")
    report = dict(attempts[0])
    report_sha256 = _canonical_sha256(report)
    run_id = str(report.get("run_id", ""))
    start = report.get("started_wall_ns")
    end = report.get("ended_wall_ns")
    if (
        supervisor.get("selected_run_id") != run_id
        or _RUN_ID.fullmatch(run_id) is None
        or report.get("schema_version")
        != IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
        or report.get("capture_contract_sha256")
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or report.get("design_sha256") != ROUND74_CAPTURE_DESIGN_SHA256
        or report.get("mode") != "qualification"
        or report.get("status") != "completed"
        or report.get("qualification_passed") is not True
        or report.get("capture_gate_passed") is not True
        or report.get("data_qualification_passed") is not True
        or report.get("resource_safety_passed") is not True
        or report.get("storage_efficiency_passed") is not True
        or report.get("audit_passed") is not True
        or report.get("audit_errors") != []
        or report.get("resource_safety_errors") != []
        or report.get("error") != ""
        or report.get("failure_class") != "none"
        or report.get("payload_cap_reached") is not False
        or report.get("database_size_cap_reached") is not False
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not (
            slot.scheduled_start_wall_ns
            <= start
            <= slot.start_window_end_wall_ns
        )
        or end < slot.scheduled_end_wall_ns
        or end
        > slot.scheduled_end_wall_ns
        + ROUND74_EVENT_COHORT_START_TOLERANCE_NS
        + ROUND74_EVENT_COHORT_END_OVERHEAD_NS
    ):
        raise ValueError("Round 74 cohort capture report is not admissible")
    elapsed = report.get("elapsed_seconds")
    event_counts = report.get("event_counts")
    symbol_counts = report.get("symbol_event_counts")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 3_600.0
        or float(elapsed)
        > (
            ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS
            + ROUND74_EVENT_COHORT_START_TOLERANCE_NS
            + ROUND74_EVENT_COHORT_END_OVERHEAD_NS
        )
        / 1_000_000_000
        or not isinstance(event_counts, Mapping)
        or set(event_counts) != _GLOBAL_EVENT_TYPES
        or any(
            isinstance(event_counts.get(name), bool)
            or not isinstance(event_counts.get(name), int)
            or int(event_counts[name]) <= 0
            for name in ("aggTrade", "bookTicker", "depthUpdate")
        )
        or not isinstance(symbol_counts, Mapping)
        or set(symbol_counts) != set(IMPACT_CAPTURE_SYMBOLS)
        or any(
            not isinstance(symbol_counts[symbol], Mapping)
            or set(symbol_counts[symbol]) != _SYMBOL_EVENT_TYPES
            or any(
                isinstance(symbol_counts[symbol].get(name), bool)
                or not isinstance(symbol_counts[symbol].get(name), int)
                or int(symbol_counts[symbol][name]) <= 0
                for name in (
                    "aggTrade",
                    "bookTicker",
                    "depthUpdate",
                    "synchronizedDepthUpdate",
                )
            )
            for symbol in IMPACT_CAPTURE_SYMBOLS
        )
    ):
        raise ValueError("Round 74 cohort event coverage is not admissible")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in event_counts.values()
    ):
        raise ValueError("Round 74 cohort event counts differ")
    if any(
        any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in symbol_counts[symbol].values()
        )
        or int(symbol_counts[symbol]["synchronizedDepthUpdate"])
        > int(symbol_counts[symbol]["depthUpdate"])
        for symbol in IMPACT_CAPTURE_SYMBOLS
    ) or any(
        sum(
            int(symbol_counts[symbol][event_type])
            for symbol in IMPACT_CAPTURE_SYMBOLS
        )
        != int(event_counts[event_type])
        for event_type in _SYMBOL_GLOBAL_EVENT_TYPES
    ):
        raise ValueError("Round 74 cohort symbol totals differ")
    integer_fields = {
        "writer_message_count": report.get("writer_message_count"),
        "writer_frame_count": report.get("writer_frame_count"),
        "writer_compressed_payload_bytes": report.get(
            "writer_compressed_payload_bytes"
        ),
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in integer_fields.values()
    ) or int(integer_fields["writer_message_count"]) != sum(
        int(value) for value in event_counts.values()
    ):
        raise ValueError("Round 74 cohort capture counts differ")
    binding = Round74EventCohortRunBinding(
        plan_sha256=plan.plan_sha256,
        slot_ordinal=slot.ordinal,
        role=slot.role,
        run_id=run_id,
        report_sha256=report_sha256,
        supervisor_sha256=supervisor_sha256,
        capture_start_wall_ns=start,
        capture_end_wall_ns=end,
        message_count=int(integer_fields["writer_message_count"]),
        frame_count=int(integer_fields["writer_frame_count"]),
        compressed_payload_bytes=int(
            integer_fields["writer_compressed_payload_bytes"]
        ),
    )
    binding.validate()
    return binding


def build_round74_event_run_partition(
    plan: Round74EventCohortPlan,
    bindings: Sequence[Round74EventCohortRunBinding],
) -> Round74EventRunPartition:
    """Build a partition only when every predeclared slot was admitted once."""

    plan.validate()
    selected = tuple(bindings)
    if len(selected) != plan.total_slots:
        raise ValueError("Round 74 cohort has missing or extra slots")
    by_ordinal: dict[int, Round74EventCohortRunBinding] = {}
    run_ids: set[str] = set()
    report_hashes: set[str] = set()
    for binding in selected:
        binding.validate()
        if (
            binding.plan_sha256 != plan.plan_sha256
            or binding.slot_ordinal in by_ordinal
            or binding.run_id in run_ids
            or binding.report_sha256 in report_hashes
        ):
            raise ValueError("Round 74 cohort binding identity is duplicated")
        slot = plan.slot(binding.slot_ordinal)
        if binding.role != slot.role:
            raise ValueError("Round 74 cohort binding role differs")
        by_ordinal[binding.slot_ordinal] = binding
        run_ids.add(binding.run_id)
        report_hashes.add(binding.report_sha256)
    if set(by_ordinal) != set(range(plan.total_slots)):
        raise ValueError("Round 74 cohort slot coverage differs")
    entries: list[Round74EventRunPartitionEntry] = []
    prior_role: str | None = None
    for ordinal in range(plan.total_slots):
        binding = by_ordinal[ordinal]
        role_changed = prior_role is not None and binding.role != prior_role
        anchor_start = binding.capture_start_wall_ns + (
            ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS
            if role_changed
            else 0
        )
        anchor_end = (
            binding.capture_end_wall_ns
            - ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
        )
        if role_changed:
            previous = entries[-1]
            entries[-1] = Round74EventRunPartitionEntry(
                run_id=previous.run_id,
                role=previous.role,
                capture_report_sha256=previous.capture_report_sha256,
                capture_start_wall_ns=previous.capture_start_wall_ns,
                capture_end_wall_ns=previous.capture_end_wall_ns,
                eligible_anchor_start_wall_ns=(
                    previous.eligible_anchor_start_wall_ns
                ),
                eligible_anchor_end_wall_ns=min(
                    previous.eligible_anchor_end_wall_ns,
                    previous.capture_end_wall_ns
                    - ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
                ),
            )
        entries.append(
            Round74EventRunPartitionEntry(
                run_id=binding.run_id,
                role=binding.role,
                capture_report_sha256=binding.report_sha256,
                capture_start_wall_ns=binding.capture_start_wall_ns,
                capture_end_wall_ns=binding.capture_end_wall_ns,
                eligible_anchor_start_wall_ns=anchor_start,
                eligible_anchor_end_wall_ns=anchor_end,
            )
        )
        prior_role = binding.role
    partition = Round74EventRunPartition(entries=tuple(entries))
    partition.validate()
    return partition


def load_round74_event_cohort_plan(raw_text: str) -> Round74EventCohortPlan:
    return Round74EventCohortPlan.from_dict(
        _strict_json_object(raw_text, "plan")
    )


def load_round74_event_cohort_binding(
    raw_text: str,
) -> Round74EventCohortRunBinding:
    return Round74EventCohortRunBinding.from_dict(
        _strict_json_object(raw_text, "binding")
    )


__all__ = [
    "ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION",
    "ROUND74_EVENT_COHORT_CAPTURE_DURATION_NS",
    "ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS",
    "ROUND74_EVENT_COHORT_END_OVERHEAD_NS",
    "ROUND74_EVENT_COHORT_MAXIMUM_SLOTS",
    "ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION",
    "ROUND74_EVENT_COHORT_SLOT_PERIOD_NS",
    "ROUND74_EVENT_COHORT_START_TOLERANCE_NS",
    "Round74EventCohortPlan",
    "Round74EventCohortRunBinding",
    "Round74EventCohortSlot",
    "bind_round74_event_cohort_supervisor",
    "build_round74_event_run_partition",
    "load_round74_event_cohort_binding",
    "load_round74_event_cohort_plan",
]
