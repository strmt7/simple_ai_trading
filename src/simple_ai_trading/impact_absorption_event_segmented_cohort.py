"""Gap-isolated prospective transport units for the Round 74 event model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
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
from .impact_absorption_event_sequence import (
    Round74MultiSymbolEventReplay,
    Round74ReplayObservation,
    _strict_json_object,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
    ImpactAbsorptionStore,
    iter_impact_capture_v10_records,
    load_impact_capture_v10_preflight,
)


ROUND74_SEGMENTED_COHORT_PLAN_SCHEMA_VERSION = (
    "round-074-segmented-event-cohort-plan-v1"
)
ROUND74_SEGMENTED_COHORT_BINDING_SCHEMA_VERSION = (
    "round-074-segmented-event-cohort-binding-v1"
)
ROUND74_SEGMENTED_COHORT_OUTCOME_SCHEMA_VERSION = (
    "round-074-segmented-event-cohort-outcome-v1"
)
ROUND74_SEGMENTED_COHORT_COVERAGE_SCHEMA_VERSION = (
    "round-074-segmented-event-cohort-coverage-v1"
)
ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS = 1_200_000_000_000
ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS = 1_500_000_000_000
ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS = 30_000_000_000
ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS = 120_000_000_000
ROUND74_SEGMENTED_COHORT_FRESH_AUDIT_TIMEOUT_NS = 120_000_000_000
ROUND74_SEGMENTED_COHORT_ROLE_COUNTS = {
    "training": 386,
    "tuning": 77,
    "test": 77,
}
ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS = {
    "training": 360,
    "tuning": 72,
    "test": 72,
}
ROUND74_SEGMENTED_COHORT_TOTAL_SLOTS = sum(
    ROUND74_SEGMENTED_COHORT_ROLE_COUNTS.values()
)
ROUND74_SEGMENTED_COHORT_OUTCOME_STATUSES = (
    "admitted",
    "transport_excluded",
    "missed",
)
ROUND74_SEGMENTED_COHORT_EXCLUSION_REASONS = (
    "startup_transport",
    "in_run_transport",
)
ROUND74_SEGMENTED_COHORT_MISSED_REASON = "host_slot_missed"

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
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
_SYMBOL_GLOBAL_EVENT_TYPES = _SYMBOL_EVENT_TYPES - {"synchronizedDepthUpdate"}


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
                raise ValueError(
                    f"Round 74 segmented cohort {label} has duplicate JSON keys"
                )
            output[key] = value
        return output

    try:
        parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Round 74 segmented cohort {label} is not valid JSON"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Round 74 segmented cohort {label} root differs")
    return parsed


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 segmented cohort {label} digest is invalid")
    return selected


def _require_positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Round 74 segmented cohort {label} differs")
    return int(value)


@dataclass(frozen=True)
class Round74SegmentedCohortSlot:
    """One predeclared, independently auditable transport unit."""

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
            or int(self.scheduled_end_wall_ns) - int(self.scheduled_start_wall_ns)
            != ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS
            or int(self.start_window_end_wall_ns) - int(self.scheduled_start_wall_ns)
            != ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS
        ):
            raise ValueError("Round 74 segmented cohort slot identity differs")

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
class Round74SegmentedCohortPlan:
    """Frozen schedule with transport-only reserves and role quorums."""

    scheduled_start_wall_ns: int
    implementation_git_commit: str
    prerequisite_artifact_sha256: str
    prerequisite_window_start_wall_ns: int
    prerequisite_window_end_wall_ns: int
    schema_version: str = ROUND74_SEGMENTED_COHORT_PLAN_SCHEMA_VERSION

    @property
    def total_slots(self) -> int:
        return ROUND74_SEGMENTED_COHORT_TOTAL_SLOTS

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_SEGMENTED_COHORT_PLAN_SCHEMA_VERSION
            or isinstance(self.scheduled_start_wall_ns, bool)
            or not isinstance(self.scheduled_start_wall_ns, int)
            or self.scheduled_start_wall_ns <= 0
            or self.scheduled_start_wall_ns % 1_000_000_000 != 0
            or _GIT_COMMIT.fullmatch(self.implementation_git_commit) is None
        ):
            raise ValueError("Round 74 segmented cohort plan identity differs")
        _require_sha256(self.prerequisite_artifact_sha256, "prerequisite")
        if not (
            0
            < int(self.prerequisite_window_start_wall_ns)
            < int(self.prerequisite_window_end_wall_ns)
            < int(self.scheduled_start_wall_ns)
        ):
            raise ValueError("Round 74 segmented cohort prerequisite window differs")
        if (
            ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS
            <= ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS
            + ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS
            + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
        ):
            raise ValueError("Round 74 segmented cohort cadence has no safety margin")
        if self.slot(self.total_slots - 1).scheduled_end_wall_ns >= 2**64:
            raise ValueError(
                "Round 74 segmented cohort schedule exceeds timestamp range"
            )

    def role_for_ordinal(self, ordinal: int) -> str:
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < self.total_slots
        ):
            raise ValueError("Round 74 segmented cohort slot ordinal differs")
        training_end = ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
        tuning_end = training_end + ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["tuning"]
        if ordinal < training_end:
            return "training"
        if ordinal < tuning_end:
            return "tuning"
        return "test"

    def slot(self, ordinal: int) -> Round74SegmentedCohortSlot:
        role = self.role_for_ordinal(ordinal)
        start = (
            int(self.scheduled_start_wall_ns)
            + int(ordinal) * ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS
        )
        selected = Round74SegmentedCohortSlot(
            ordinal=int(ordinal),
            role=role,
            scheduled_start_wall_ns=start,
            scheduled_end_wall_ns=(
                start + ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS
            ),
            start_window_end_wall_ns=(
                start + ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS
            ),
        )
        selected.validate()
        return selected

    @property
    def plan_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "implementation_git_commit": self.implementation_git_commit,
            "scheduled_start_wall_ns": self.scheduled_start_wall_ns,
            "role_counts": dict(ROUND74_SEGMENTED_COHORT_ROLE_COUNTS),
            "role_quorums": dict(ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS),
            "total_slots": self.total_slots,
            "schedule_formula": {
                "ordinal_origin": 0,
                "slot_period_ns": ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
                "capture_duration_ns": ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS,
                "start_tolerance_ns": ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS,
                "maximum_end_overhead_ns": (ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS),
                "fresh_audit_timeout_ns": (
                    ROUND74_SEGMENTED_COHORT_FRESH_AUDIT_TIMEOUT_NS
                ),
            },
            "capture_contract": {
                "provider": "Binance USD-M public production market data",
                "symbols": list(IMPACT_CAPTURE_SYMBOLS),
                "capture_schema_version": IMPACT_CAPTURE_V10_SCHEMA_VERSION,
                "capture_report_schema_version": (
                    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
                ),
                "capture_design_sha256": ROUND74_CAPTURE_DESIGN_SHA256,
                "capture_contract_sha256": IMPACT_CAPTURE_V10_CONTRACT_SHA256,
                "underlying_mode": "probe",
                "segment_admission_mode": "prospective_transport_unit",
                "maximum_reconnects": 0,
                "in_unit_retry_permitted": False,
                "failed_prefix_salvage_permitted": False,
                "credentials_used": False,
                "orders_submitted": False,
            },
            "missingness_policy": {
                "all_predeclared_slots_require_terminal_outcomes": True,
                "all_admitted_units_included": True,
                "market_or_model_dependent_replacement_permitted": False,
                "transport_excluded_units_are_model_data": False,
                "missed_units_are_model_data": False,
                "role_quorums_must_pass": True,
            },
            "prerequisite": {
                "artifact_sha256": self.prerequisite_artifact_sha256,
                "window_start_wall_ns": self.prerequisite_window_start_wall_ns,
                "window_end_wall_ns": self.prerequisite_window_end_wall_ns,
                "must_pass_before_first_slot": True,
                "prerequisite_capture_is_model_cohort_data": False,
            },
            "partition_policy": {
                "split_unit": "whole_admitted_transport_unit",
                "role_order": list(ROUND74_EVENT_PARTITION_ROLES),
                "random_row_split_permitted": False,
                "cross_unit_feature_or_target_permitted": False,
                "minimum_purge_ns": ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
                "minimum_embargo_ns": ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS,
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
    ) -> Round74SegmentedCohortPlan:
        payload = dict(value)
        claimed = str(payload.pop("plan_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented cohort plan digest differs")
        prerequisite = payload.get("prerequisite")
        if not isinstance(prerequisite, Mapping):
            raise ValueError("Round 74 segmented cohort prerequisite differs")
        try:
            selected = cls(
                scheduled_start_wall_ns=int(payload["scheduled_start_wall_ns"]),
                implementation_git_commit=str(payload["implementation_git_commit"]),
                prerequisite_artifact_sha256=str(prerequisite["artifact_sha256"]),
                prerequisite_window_start_wall_ns=int(
                    prerequisite["window_start_wall_ns"]
                ),
                prerequisite_window_end_wall_ns=int(prerequisite["window_end_wall_ns"]),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 segmented cohort plan payload differs") from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented cohort static policy differs")
        selected.validate()
        return selected


@dataclass(frozen=True)
class Round74SegmentedCohortRunBinding:
    """One completed probe bound to a fresh independent exact-wire audit."""

    plan_sha256: str
    slot_ordinal: int
    role: str
    run_id: str
    report_sha256: str
    supervisor_sha256: str
    fresh_audit_sha256: str
    capture_start_wall_ns: int
    capture_end_wall_ns: int
    message_count: int
    frame_count: int
    compressed_payload_bytes: int
    schema_version: str = ROUND74_SEGMENTED_COHORT_BINDING_SCHEMA_VERSION

    def validate(self) -> None:
        counts = (
            self.message_count,
            self.frame_count,
            self.compressed_payload_bytes,
        )
        if (
            self.schema_version != ROUND74_SEGMENTED_COHORT_BINDING_SCHEMA_VERSION
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or isinstance(self.slot_ordinal, bool)
            or not isinstance(self.slot_ordinal, int)
            or self.slot_ordinal < 0
            or int(self.capture_start_wall_ns) <= 0
            or int(self.capture_end_wall_ns) <= int(self.capture_start_wall_ns)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in counts
            )
        ):
            raise ValueError("Round 74 segmented cohort run binding differs")
        _require_sha256(self.plan_sha256, "plan")
        _require_sha256(self.report_sha256, "report")
        _require_sha256(self.supervisor_sha256, "supervisor")
        _require_sha256(self.fresh_audit_sha256, "fresh audit")

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
            "fresh_audit_sha256": self.fresh_audit_sha256,
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
    ) -> Round74SegmentedCohortRunBinding:
        payload = dict(value)
        claimed = str(payload.pop("binding_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented cohort binding digest differs")
        try:
            selected = cls(
                plan_sha256=str(payload["plan_sha256"]),
                slot_ordinal=int(payload["slot_ordinal"]),
                role=str(payload["role"]),
                run_id=str(payload["run_id"]),
                report_sha256=str(payload["report_sha256"]),
                supervisor_sha256=str(payload["supervisor_sha256"]),
                fresh_audit_sha256=str(payload["fresh_audit_sha256"]),
                capture_start_wall_ns=int(payload["capture_start_wall_ns"]),
                capture_end_wall_ns=int(payload["capture_end_wall_ns"]),
                message_count=int(payload["message_count"]),
                frame_count=int(payload["frame_count"]),
                compressed_payload_bytes=int(payload["compressed_payload_bytes"]),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 segmented cohort binding payload differs"
            ) from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented cohort binding policy differs")
        selected.validate()
        return selected


def bind_round74_segmented_probe_supervisor(
    plan: Round74SegmentedCohortPlan,
    *,
    slot_ordinal: int,
    supervisor_payload: Mapping[str, object],
    fresh_audit_payload: Mapping[str, object],
) -> Round74SegmentedCohortRunBinding:
    """Admit one completed zero-reconnect probe after a fresh exact audit."""

    plan.validate()
    slot = plan.slot(slot_ordinal)
    supervisor = dict(supervisor_payload)
    attempts = supervisor.get("attempts")
    if (
        supervisor.get("schema_version") != "round-074-capture-supervisor-report-v1"
        or supervisor.get("design_sha256") != ROUND74_CAPTURE_DESIGN_SHA256
        or supervisor.get("capture_schema_version") != IMPACT_CAPTURE_V10_SCHEMA_VERSION
        or supervisor.get("capture_contract_sha256")
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or supervisor.get("status") != "completed"
        or supervisor.get("qualification_passed") is not False
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
        raise ValueError("Round 74 segmented cohort supervisor is not admissible")
    report = dict(attempts[0])
    report_sha256 = _canonical_sha256(report)
    run_id = str(report.get("run_id", ""))
    start = report.get("started_wall_ns")
    end = report.get("ended_wall_ns")
    if (
        supervisor.get("selected_run_id") != run_id
        or _RUN_ID.fullmatch(run_id) is None
        or report.get("schema_version") != IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
        or report.get("capture_contract_sha256") != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or report.get("design_sha256") != ROUND74_CAPTURE_DESIGN_SHA256
        or report.get("mode") != "probe"
        or report.get("status") != "completed"
        or report.get("qualification_passed") is not False
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
        or not (slot.scheduled_start_wall_ns <= start <= slot.start_window_end_wall_ns)
        or end < start + ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS
        or end
        > slot.scheduled_end_wall_ns
        + ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS
        + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
    ):
        raise ValueError("Round 74 segmented cohort report is not admissible")
    elapsed = report.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS / 1_000_000_000
        or float(elapsed)
        > (
            ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS
            + ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS
            + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
        )
        / 1_000_000_000
    ):
        raise ValueError("Round 74 segmented cohort elapsed time differs")
    _validate_event_coverage(report)
    counts = {
        "message_count": _require_positive_integer(
            report.get("writer_message_count"), "message count"
        ),
        "frame_count": _require_positive_integer(
            report.get("writer_frame_count"), "frame count"
        ),
        "compressed_payload_bytes": _require_positive_integer(
            report.get("writer_compressed_payload_bytes"),
            "compressed payload bytes",
        ),
    }
    if counts["message_count"] != sum(
        int(value) for value in dict(report["event_counts"]).values()
    ):
        raise ValueError("Round 74 segmented cohort capture counts differ")
    fresh_audit = dict(fresh_audit_payload)
    if (
        fresh_audit.get("schema_version") != "round-074-capture-audit-v1"
        or fresh_audit.get("passed") is not True
        or fresh_audit.get("errors") != []
        or fresh_audit.get("run_id") != run_id
        or fresh_audit.get("run_status") != "completed"
        or fresh_audit.get("stored_report_schema_version")
        != IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
        or fresh_audit.get("stored_report_sha256") != report_sha256
        or fresh_audit.get("capture_contract_sha256")
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or fresh_audit.get("message_count") != counts["message_count"]
        or fresh_audit.get("frame_count") != counts["frame_count"]
        or fresh_audit.get("compressed_payload_bytes")
        != counts["compressed_payload_bytes"]
        or _SHA256.fullmatch(str(fresh_audit.get("last_frame_sha256", ""))) is None
    ):
        raise ValueError("Round 74 segmented cohort fresh audit differs")
    binding = Round74SegmentedCohortRunBinding(
        plan_sha256=plan.plan_sha256,
        slot_ordinal=slot.ordinal,
        role=slot.role,
        run_id=run_id,
        report_sha256=report_sha256,
        supervisor_sha256=_canonical_sha256(supervisor),
        fresh_audit_sha256=_canonical_sha256(fresh_audit),
        capture_start_wall_ns=start,
        capture_end_wall_ns=end,
        message_count=counts["message_count"],
        frame_count=counts["frame_count"],
        compressed_payload_bytes=counts["compressed_payload_bytes"],
    )
    binding.validate()
    return binding


def _validate_event_coverage(report: Mapping[str, object]) -> None:
    event_counts = report.get("event_counts")
    symbol_counts = report.get("symbol_event_counts")
    if (
        not isinstance(event_counts, Mapping)
        or set(event_counts) != _GLOBAL_EVENT_TYPES
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in event_counts.values()
        )
        or any(
            isinstance(event_counts.get(name), bool)
            or not isinstance(event_counts.get(name), int)
            or int(event_counts[name]) <= 0
            for name in ("aggTrade", "bookTicker", "depthUpdate")
        )
        or not isinstance(symbol_counts, Mapping)
        or set(symbol_counts) != set(IMPACT_CAPTURE_SYMBOLS)
    ):
        raise ValueError("Round 74 segmented cohort event coverage differs")
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        counts = symbol_counts[symbol]
        if (
            not isinstance(counts, Mapping)
            or set(counts) != _SYMBOL_EVENT_TYPES
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts.values()
            )
            or any(
                int(counts[name]) <= 0
                for name in (
                    "aggTrade",
                    "bookTicker",
                    "depthUpdate",
                    "synchronizedDepthUpdate",
                )
            )
            or int(counts["synchronizedDepthUpdate"]) > int(counts["depthUpdate"])
        ):
            raise ValueError("Round 74 segmented cohort symbol coverage differs")
    if any(
        sum(int(symbol_counts[symbol][name]) for symbol in IMPACT_CAPTURE_SYMBOLS)
        != int(event_counts[name])
        for name in _SYMBOL_GLOBAL_EVENT_TYPES
    ):
        raise ValueError("Round 74 segmented cohort symbol totals differ")


@dataclass(frozen=True)
class Round74SegmentedCohortSlotOutcome:
    """Terminal disposition for every predeclared transport slot."""

    plan_sha256: str
    slot_ordinal: int
    role: str
    status: str
    reason_code: str
    evidence_sha256: str
    binding: Round74SegmentedCohortRunBinding | None = None
    schema_version: str = ROUND74_SEGMENTED_COHORT_OUTCOME_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_SEGMENTED_COHORT_OUTCOME_SCHEMA_VERSION
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or self.status not in ROUND74_SEGMENTED_COHORT_OUTCOME_STATUSES
            or isinstance(self.slot_ordinal, bool)
            or not isinstance(self.slot_ordinal, int)
            or self.slot_ordinal < 0
        ):
            raise ValueError("Round 74 segmented cohort outcome identity differs")
        _require_sha256(self.plan_sha256, "outcome plan")
        _require_sha256(self.evidence_sha256, "outcome evidence")
        if self.status == "admitted":
            if (
                self.reason_code != "admitted"
                or self.binding is None
                or self.binding.plan_sha256 != self.plan_sha256
                or self.binding.slot_ordinal != self.slot_ordinal
                or self.binding.role != self.role
            ):
                raise ValueError("Round 74 admitted segmented outcome differs")
            self.binding.validate()
        elif self.binding is not None:
            raise ValueError("Round 74 unavailable outcome contains a binding")
        elif self.status == "transport_excluded":
            if self.reason_code not in ROUND74_SEGMENTED_COHORT_EXCLUSION_REASONS:
                raise ValueError("Round 74 transport exclusion reason differs")
        elif self.reason_code != ROUND74_SEGMENTED_COHORT_MISSED_REASON:
            raise ValueError("Round 74 missed-slot reason differs")

    @property
    def outcome_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "slot_ordinal": self.slot_ordinal,
            "role": self.role,
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence_sha256": self.evidence_sha256,
            "binding": None if self.binding is None else self.binding.as_dict(),
        }
        if include_sha256:
            payload["outcome_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74SegmentedCohortSlotOutcome:
        payload = dict(value)
        claimed = str(payload.pop("outcome_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented cohort outcome digest differs")
        raw_binding = payload.get("binding")
        if raw_binding is not None and not isinstance(raw_binding, Mapping):
            raise ValueError("Round 74 segmented cohort outcome binding differs")
        try:
            selected = cls(
                plan_sha256=str(payload["plan_sha256"]),
                slot_ordinal=int(payload["slot_ordinal"]),
                role=str(payload["role"]),
                status=str(payload["status"]),
                reason_code=str(payload["reason_code"]),
                evidence_sha256=str(payload["evidence_sha256"]),
                binding=(
                    None
                    if raw_binding is None
                    else Round74SegmentedCohortRunBinding.from_dict(raw_binding)
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 segmented cohort outcome payload differs"
            ) from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented cohort outcome policy differs")
        selected.validate()
        return selected


def _validate_complete_outcomes(
    plan: Round74SegmentedCohortPlan,
    outcomes: Sequence[Round74SegmentedCohortSlotOutcome],
) -> tuple[Round74SegmentedCohortSlotOutcome, ...]:
    plan.validate()
    selected = tuple(outcomes)
    if len(selected) != plan.total_slots:
        raise ValueError("Round 74 segmented cohort outcome panel is incomplete")
    ordered = tuple(sorted(selected, key=lambda value: value.slot_ordinal))
    if tuple(value.slot_ordinal for value in ordered) != tuple(range(plan.total_slots)):
        raise ValueError("Round 74 segmented cohort slot coverage differs")
    admitted_runs: set[str] = set()
    admitted_reports: set[str] = set()
    for outcome in ordered:
        outcome.validate()
        slot = plan.slot(outcome.slot_ordinal)
        if outcome.plan_sha256 != plan.plan_sha256 or outcome.role != slot.role:
            raise ValueError("Round 74 segmented cohort outcome role differs")
        if outcome.binding is not None:
            if (
                outcome.binding.run_id in admitted_runs
                or outcome.binding.report_sha256 in admitted_reports
            ):
                raise ValueError(
                    "Round 74 segmented cohort admitted identity is duplicated"
                )
            admitted_runs.add(outcome.binding.run_id)
            admitted_reports.add(outcome.binding.report_sha256)
    return ordered


def build_round74_segmented_event_run_partition(
    plan: Round74SegmentedCohortPlan,
    outcomes: Sequence[Round74SegmentedCohortSlotOutcome],
) -> Round74EventRunPartition:
    """Build a chronological partition from every admitted transport unit."""

    ordered = _validate_complete_outcomes(plan, outcomes)
    admitted_by_role = {
        role: sum(
            outcome.status == "admitted" and outcome.role == role for outcome in ordered
        )
        for role in ROUND74_EVENT_PARTITION_ROLES
    }
    if any(
        admitted_by_role[role] < ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS[role]
        for role in ROUND74_EVENT_PARTITION_ROLES
    ):
        raise ValueError("Round 74 segmented cohort role quorum failed")
    entries: list[Round74EventRunPartitionEntry] = []
    prior_role: str | None = None
    for outcome in ordered:
        binding = outcome.binding
        if binding is None:
            continue
        role_changed = prior_role is not None and binding.role != prior_role
        anchor_start = binding.capture_start_wall_ns + (
            ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS if role_changed else 0
        )
        anchor_end = (
            binding.capture_end_wall_ns - ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
        )
        if role_changed:
            previous = entries[-1]
            entries[-1] = Round74EventRunPartitionEntry(
                run_id=previous.run_id,
                role=previous.role,
                capture_report_sha256=previous.capture_report_sha256,
                capture_start_wall_ns=previous.capture_start_wall_ns,
                capture_end_wall_ns=previous.capture_end_wall_ns,
                eligible_anchor_start_wall_ns=previous.eligible_anchor_start_wall_ns,
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
    partition = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256=plan.plan_sha256,
    )
    partition.validate()
    return partition


@dataclass(frozen=True)
class Round74SegmentedCohortCoverage:
    """Hash-bound missingness ledger and admitted-unit partition."""

    plan_sha256: str
    outcomes: tuple[Round74SegmentedCohortSlotOutcome, ...]
    partition: Round74EventRunPartition
    schema_version: str = ROUND74_SEGMENTED_COHORT_COVERAGE_SCHEMA_VERSION

    def validate(self, plan: Round74SegmentedCohortPlan) -> None:
        ordered = _validate_complete_outcomes(plan, self.outcomes)
        expected = build_round74_segmented_event_run_partition(plan, ordered)
        if (
            self.schema_version != ROUND74_SEGMENTED_COHORT_COVERAGE_SCHEMA_VERSION
            or self.plan_sha256 != plan.plan_sha256
            or self.partition.as_dict() != expected.as_dict()
        ):
            raise ValueError("Round 74 segmented cohort coverage differs")

    @property
    def coverage_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        counts = {
            role: {
                status: sum(
                    outcome.role == role and outcome.status == status
                    for outcome in self.outcomes
                )
                for status in ROUND74_SEGMENTED_COHORT_OUTCOME_STATUSES
            }
            for role in ROUND74_EVENT_PARTITION_ROLES
        }
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "role_counts": {
                role: {
                    "planned": ROUND74_SEGMENTED_COHORT_ROLE_COUNTS[role],
                    "required_admitted": (ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS[role]),
                    **counts[role],
                }
                for role in ROUND74_EVENT_PARTITION_ROLES
            },
            "outcome_sha256": [
                outcome.outcome_sha256
                for outcome in sorted(
                    self.outcomes,
                    key=lambda value: value.slot_ordinal,
                )
            ],
            "partition_sha256": self.partition.partition_sha256,
            "all_admitted_units_included": True,
            "transport_excluded_or_missed_units_included": False,
            "cross_unit_feature_or_target_permitted": False,
            "profitability_or_edge_claim": False,
            "trading_authority": False,
        }
        if include_sha256:
            payload["coverage_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def build(
        cls,
        plan: Round74SegmentedCohortPlan,
        outcomes: Sequence[Round74SegmentedCohortSlotOutcome],
    ) -> Round74SegmentedCohortCoverage:
        ordered = _validate_complete_outcomes(plan, outcomes)
        selected = cls(
            plan_sha256=plan.plan_sha256,
            outcomes=ordered,
            partition=build_round74_segmented_event_run_partition(plan, ordered),
        )
        selected.validate(plan)
        return selected

    @classmethod
    def from_dict(
        cls,
        plan: Round74SegmentedCohortPlan,
        outcomes: Sequence[Round74SegmentedCohortSlotOutcome],
        value: Mapping[str, object],
    ) -> Round74SegmentedCohortCoverage:
        payload = dict(value)
        claimed = str(payload.pop("coverage_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented cohort coverage digest differs")
        selected = cls.build(plan, outcomes)
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented cohort coverage policy differs")
        return selected


def iter_round74_v10_segment_event_observations(
    store: ImpactAbsorptionStore,
    *,
    binding: Round74SegmentedCohortRunBinding,
) -> Iterator[Round74ReplayObservation]:
    """Replay one admitted probe without permitting state across unit boundaries."""

    binding.validate()
    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 segmented replay requires an ImpactAbsorptionStore")
    if not store.read_only:
        raise ValueError("Round 74 segmented replay requires a read-only store")
    audit = store.audit_run(binding.run_id)
    audit_payload = audit.as_dict()
    if (
        not audit.passed
        or audit_payload.get("run_status") != "completed"
        or _canonical_sha256(audit_payload) != binding.fresh_audit_sha256
    ):
        raise ValueError("Round 74 segmented replay fresh audit differs")
    connection = store.connect()
    run = connection.execute(
        """
        SELECT status, schema_version, capture_contract_sha256
        FROM impact_capture_run WHERE run_id = ?
        """,
        [binding.run_id],
    ).fetchone()
    if run != (
        "completed",
        IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    ):
        raise ValueError("Round 74 segmented replay run identity differs")
    report_row = connection.execute(
        """
        SELECT schema_version, capture_contract_sha256, report_json, report_sha256
        FROM impact_capture_report WHERE run_id = ?
        """,
        [binding.run_id],
    ).fetchone()
    if report_row is None:
        raise ValueError("Round 74 segmented replay report is missing")
    report_text = str(report_row[2])
    report = _strict_json_object(report_text)
    if (
        str(report_row[0]) != IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
        or str(report_row[1]) != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or str(report_row[3]) != binding.report_sha256
        or hashlib.sha256(report_text.encode("ascii")).hexdigest()
        != binding.report_sha256
        or report.get("mode") != "probe"
        or report.get("status") != "completed"
        or report.get("qualification_passed") is not False
        or not all(
            report.get(field) is True
            for field in (
                "capture_gate_passed",
                "data_qualification_passed",
                "resource_safety_passed",
                "storage_efficiency_passed",
            )
        )
    ):
        raise ValueError("Round 74 segmented replay report identity differs")
    preflight = load_impact_capture_v10_preflight(
        connection,
        run_id=binding.run_id,
    )
    segment_rows = connection.execute(
        """
        SELECT symbol, status, tick_size
        FROM impact_capture_segment WHERE run_id = ? ORDER BY symbol
        """,
        [binding.run_id],
    ).fetchall()
    if tuple(str(row[0]) for row in segment_rows) != IMPACT_CAPTURE_SYMBOLS or any(
        str(row[1]) != "valid" for row in segment_rows
    ):
        raise ValueError("Round 74 segmented replay symbol state differs")
    tick_sizes = {str(row[0]): float(row[2]) for row in segment_rows}
    snapshots = {
        symbol: _strict_json_object(record.raw_text)
        for symbol, record in preflight.snapshot_records
    }
    replay = Round74MultiSymbolEventReplay(
        tick_sizes=tick_sizes,
        depth_snapshots=snapshots,
        feature_ready_wall_ns=preflight.ready_wall_ns,
    )
    for frame_index, message_index, record in iter_impact_capture_v10_records(
        connection,
        run_id=binding.run_id,
    ):
        observation = replay.consume_observation(
            frame_index=frame_index,
            message_index=message_index,
            record=record,
        )
        if observation is not None:
            yield observation


def load_round74_segmented_cohort_plan(
    raw_text: str,
) -> Round74SegmentedCohortPlan:
    parsed = _strict_json_mapping(raw_text, "plan")
    return Round74SegmentedCohortPlan.from_dict(parsed)


def load_round74_segmented_cohort_binding(
    raw_text: str,
) -> Round74SegmentedCohortRunBinding:
    parsed = _strict_json_mapping(raw_text, "binding")
    return Round74SegmentedCohortRunBinding.from_dict(parsed)


def load_round74_segmented_cohort_outcome(
    raw_text: str,
) -> Round74SegmentedCohortSlotOutcome:
    parsed = _strict_json_mapping(raw_text, "outcome")
    return Round74SegmentedCohortSlotOutcome.from_dict(parsed)


def load_round74_segmented_cohort_coverage(
    raw_text: str,
    *,
    plan: Round74SegmentedCohortPlan,
    outcomes: Sequence[Round74SegmentedCohortSlotOutcome],
) -> Round74SegmentedCohortCoverage:
    parsed = _strict_json_mapping(raw_text, "coverage")
    return Round74SegmentedCohortCoverage.from_dict(plan, outcomes, parsed)


__all__ = [
    "ROUND74_SEGMENTED_COHORT_BINDING_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS",
    "ROUND74_SEGMENTED_COHORT_COVERAGE_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS",
    "ROUND74_SEGMENTED_COHORT_EXCLUSION_REASONS",
    "ROUND74_SEGMENTED_COHORT_FRESH_AUDIT_TIMEOUT_NS",
    "ROUND74_SEGMENTED_COHORT_MISSED_REASON",
    "ROUND74_SEGMENTED_COHORT_OUTCOME_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_COHORT_OUTCOME_STATUSES",
    "ROUND74_SEGMENTED_COHORT_PLAN_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_COHORT_ROLE_COUNTS",
    "ROUND74_SEGMENTED_COHORT_ROLE_QUORUMS",
    "ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS",
    "ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS",
    "ROUND74_SEGMENTED_COHORT_TOTAL_SLOTS",
    "Round74SegmentedCohortCoverage",
    "Round74SegmentedCohortPlan",
    "Round74SegmentedCohortRunBinding",
    "Round74SegmentedCohortSlot",
    "Round74SegmentedCohortSlotOutcome",
    "bind_round74_segmented_probe_supervisor",
    "build_round74_segmented_event_run_partition",
    "iter_round74_v10_segment_event_observations",
    "load_round74_segmented_cohort_binding",
    "load_round74_segmented_cohort_coverage",
    "load_round74_segmented_cohort_outcome",
    "load_round74_segmented_cohort_plan",
]
