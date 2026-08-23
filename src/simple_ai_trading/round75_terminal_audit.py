"""Metadata-only terminal audit for the frozen Round 75 capture campaign."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO

from .impact_absorption_event_segmented_cohort import (
    ROUND74_EVENT_PARTITION_ROLES,
    ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS,
    ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
    Round74SegmentedCohortPlan,
    Round74SegmentedCohortSlotOutcome,
    Round74SegmentedTransportEpochAudit,
    load_round74_segmented_cohort_plan,
)
from .round75_continuous_capture import (
    ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION,
    ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION,
    ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION,
    ROUND75_SHARD_PREFIX,
    _canonical_sha256,
)


ROUND75_TERMINAL_AUDIT_SCHEMA_VERSION = "round-075-terminal-audit-v1"
_RESULT_SCHEMA_VERSION = "round-074-segmented-campaign-slot-result-v1"
_RESERVATION_SCHEMA_VERSION = "round-074-segmented-slot-reservation-v1"
_ACTIVATION_SCHEMA_VERSION = "round-075-host-activation-receipt-v2"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _within(root: Path, path: Path) -> bool:
    selected = path.resolve()
    resolved_root = root.resolve()
    return selected == resolved_root or resolved_root in selected.parents


def _load_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Round 75 {label} path differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Round 75 {label} root differs")
    return value


def _validate_bound_object(
    value: Mapping[str, object],
    *,
    digest_field: str,
    label: str,
) -> str:
    canonical = dict(value)
    claimed = canonical.pop(digest_field, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(canonical):
        raise ValueError(f"Round 75 {label} digest differs")
    return claimed


def _stream_sha256(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a regular file without loading it into memory."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("Round 75 audit file path differs")
    with path.open("rb") as handle:
        return _stream_sha256(handle)


def _file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "name": path.name,
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
        "filesystem_ctime_utc": datetime.fromtimestamp(
            stat.st_ctime,
            tz=timezone.utc,
        ).isoformat(),
        "last_write_at_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def _lease_available(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        return False
    with path.open("rb") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            return False
    return True


@dataclass(frozen=True)
class Round75TerminalAuditConfig:
    """Immutable evidence paths; campaign databases are hash-only inputs."""

    evidence_repository: Path
    capture_repository: Path
    contract_path: Path
    activation_path: Path
    plan_path: Path
    data_root: Path
    state_root: Path
    service_state_path: Path
    lease_path: Path

    def validate(self) -> None:
        evidence = self.evidence_repository.resolve()
        capture = self.capture_repository.resolve()
        if not evidence.is_dir() or not capture.is_dir():
            raise ValueError("Round 75 terminal audit repository differs")
        evidence_paths = (self.contract_path, self.activation_path, self.plan_path)
        capture_paths = (
            self.data_root,
            self.state_root,
            self.service_state_path,
            self.lease_path,
        )
        if (
            any(path.is_symlink() for path in (*evidence_paths, *capture_paths))
            or any(not _within(evidence, path) for path in evidence_paths)
            or any(not _within(capture, path) for path in capture_paths)
            or not self.data_root.is_dir()
            or not self.state_root.is_dir()
        ):
            raise ValueError("Round 75 terminal audit path boundary differs")


def _load_sources(
    config: Round75TerminalAuditConfig,
) -> tuple[Round74SegmentedCohortPlan, dict[str, object]]:
    contract = _load_object(config.contract_path, "contract")
    contract_sha256 = _validate_bound_object(
        contract,
        digest_field="artifact_sha256",
        label="contract",
    )
    activation = _load_object(config.activation_path, "activation")
    activation_sha256 = _validate_bound_object(
        activation,
        digest_field="artifact_sha256",
        label="activation",
    )
    plan = load_round74_segmented_cohort_plan(
        config.plan_path.read_text(encoding="utf-8")
    )
    activated_plan = contract.get("activated_plan")
    activation_contract = activation.get("contract")
    activation_plan = activation.get("prospective_plan")
    implementation = contract.get("implementation")
    if (
        contract.get("schema_version")
        != ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION
        or activation.get("schema_version") != _ACTIVATION_SCHEMA_VERSION
        or not isinstance(activated_plan, Mapping)
        or not isinstance(activation_contract, Mapping)
        or not isinstance(activation_plan, Mapping)
        or not isinstance(implementation, Mapping)
        or activated_plan.get("plan_sha256") != plan.plan_sha256
        or activated_plan.get("plan_file_sha256") != file_sha256(config.plan_path)
        or activation_contract.get("artifact_sha256") != contract_sha256
        or activation_contract.get("file_sha256") != file_sha256(config.contract_path)
        or activation_plan.get("plan_sha256") != plan.plan_sha256
        or activation_plan.get("file_sha256") != file_sha256(config.plan_path)
    ):
        raise ValueError("Round 75 contract, activation, or plan binding differs")
    source_hashes: dict[str, str] = {}
    for key, value in implementation.items():
        if not str(key).endswith("_path"):
            continue
        relative = str(value)
        digest_key = f"{str(key)[:-5]}_file_sha256"
        expected = implementation.get(digest_key)
        source = config.capture_repository / relative
        if not _within(config.capture_repository, source):
            raise ValueError("Round 75 frozen implementation escapes repository")
        actual = file_sha256(source)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError(f"Round 75 frozen implementation differs: {relative}")
        source_hashes[relative] = actual
    return plan, {
        "contract_artifact_sha256": contract_sha256,
        "contract_file_sha256": file_sha256(config.contract_path),
        "activation_artifact_sha256": activation_sha256,
        "activation_file_sha256": file_sha256(config.activation_path),
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": file_sha256(config.plan_path),
        "frozen_implementation_file_sha256": dict(sorted(source_hashes.items())),
    }


def _validate_reservation(
    path: Path,
    *,
    plan: Round74SegmentedCohortPlan,
) -> tuple[dict[str, object], str]:
    value = _load_object(path, "slot reservation")
    digest = _validate_bound_object(
        value,
        digest_field="reservation_sha256",
        label="slot reservation",
    )
    ordinal = value.get("slot_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise ValueError("Round 75 slot reservation ordinal differs")
    slot = plan.slot(ordinal)
    if (
        value.get("schema_version") != _RESERVATION_SCHEMA_VERSION
        or value.get("plan_sha256") != plan.plan_sha256
        or value.get("role") != slot.role
        or value.get("scheduled_start_wall_ns") != slot.scheduled_start_wall_ns
        or value.get("automatic_retry_permitted") is not False
        or value.get("credentials_used") is not False
        or value.get("orders_submitted") is not False
    ):
        raise ValueError("Round 75 slot reservation differs")
    return value, digest


def _validate_result(
    slot_root: Path,
    *,
    plan: Round74SegmentedCohortPlan,
) -> tuple[Round74SegmentedCohortSlotOutcome, dict[str, object]]:
    result = _load_object(slot_root / "result.json", "slot result")
    result_sha256 = _validate_bound_object(
        result,
        digest_field="result_sha256",
        label="slot result",
    )
    reservation, reservation_sha256 = _validate_reservation(
        slot_root / "reservation.json",
        plan=plan,
    )
    ordinal = reservation["slot_ordinal"]
    slot = plan.slot(int(ordinal))
    adjudication = result.get("adjudication")
    if not isinstance(adjudication, Mapping):
        raise ValueError("Round 75 slot adjudication differs")
    _validate_bound_object(
        adjudication,
        digest_field="adjudication_sha256",
        label="slot adjudication",
    )
    epoch = adjudication.get("epoch_audit")
    outcome_payload = adjudication.get("outcome")
    supervisor = adjudication.get("supervisor")
    if (
        not isinstance(epoch, Mapping)
        or not isinstance(outcome_payload, Mapping)
        or not isinstance(supervisor, Mapping)
    ):
        raise ValueError("Round 75 slot outcome differs")
    epoch_audit = Round74SegmentedTransportEpochAudit.from_dict(epoch)
    outcome = Round74SegmentedCohortSlotOutcome.from_dict(outcome_payload)
    if (
        result.get("schema_version") != _RESULT_SCHEMA_VERSION
        or result.get("plan_sha256") != plan.plan_sha256
        or result.get("slot_ordinal") != ordinal
        or result.get("role") != slot.role
        or result.get("reservation_sha256") != reservation_sha256
        or result.get("capture_stdout_sha256")
        != file_sha256(slot_root / "capture.stdout.json")
        or result.get("capture_stderr_sha256")
        != file_sha256(slot_root / "capture.stderr.log")
        or result.get("watchdog_breaches") != []
        or result.get("automatic_retry_permitted") is not False
        or result.get("credentials_used") is not False
        or result.get("orders_submitted") is not False
        or result.get("profitability_or_edge_claim") is not False
        or result.get("trading_authority") is not False
        or adjudication.get("plan_sha256") != plan.plan_sha256
        or adjudication.get("slot_ordinal") != ordinal
        or outcome.plan_sha256 != plan.plan_sha256
        or outcome.slot_ordinal != ordinal
        or outcome.role != slot.role
        or outcome.evidence_sha256
        not in {
            epoch_audit.epoch_audit_sha256,
            outcome.binding.binding_sha256 if outcome.binding else "",
        }
        or (
            outcome.binding is not None
            and outcome.binding.supervisor_sha256 != _canonical_sha256(dict(supervisor))
        )
    ):
        raise ValueError("Round 75 terminal slot result differs")
    state = _load_object(slot_root / "state.json", "terminal slot state")
    if (
        state.get("schema_version") != "round-074-segmented-slot-state-v1"
        or state.get("plan_sha256") != plan.plan_sha256
        or state.get("slot_ordinal") != ordinal
        or state.get("phase") != "terminal"
        or state.get("result_sha256") != result_sha256
    ):
        raise ValueError("Round 75 terminal slot state differs")
    return outcome, {
        "slot_ordinal": ordinal,
        "role": slot.role,
        "status": outcome.status,
        "reason_code": outcome.reason_code,
        "result_sha256": result_sha256,
    }


def _validate_missed(
    path: Path,
    *,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
) -> dict[str, object]:
    value = _load_object(path, "missed-slot receipt")
    missed_sha256 = _validate_bound_object(
        value,
        digest_field="missed_sha256",
        label="missed-slot receipt",
    )
    ordinal = value.get("slot_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise ValueError("Round 75 missed-slot ordinal differs")
    slot = plan.slot(ordinal)
    if (
        value.get("schema_version") != ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION
        or value.get("plan_sha256") != plan.plan_sha256
        or value.get("contract_sha256") != contract_sha256
        or value.get("role") != slot.role
        or value.get("scheduled_start_wall_ns") != slot.scheduled_start_wall_ns
        or value.get("start_window_end_wall_ns") != slot.start_window_end_wall_ns
        or value.get("automatic_retry_permitted") is not False
        or value.get("replacement_or_time_shift_permitted") is not False
        or value.get("model_data_admitted") is not False
        or value.get("credentials_used") is not False
        or value.get("orders_submitted") is not False
        or value.get("trading_authority") is not False
    ):
        raise ValueError("Round 75 missed-slot receipt differs")
    return {
        "slot_ordinal": ordinal,
        "role": slot.role,
        "reason_code": str(value.get("reason_code")),
        "missed_sha256": missed_sha256,
    }


def _slot_ordinal(path: Path, prefix: str) -> int:
    suffix = path.stem.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError("Round 75 slot path name differs")
    return int(suffix)


def _validate_incomplete_slot(
    slot_root: Path,
    *,
    plan: Round74SegmentedCohortPlan,
) -> dict[str, object]:
    reservation, reservation_sha256 = _validate_reservation(
        slot_root / "reservation.json",
        plan=plan,
    )
    ordinal = int(reservation["slot_ordinal"])
    state_path = slot_root / "state.json"
    state = _load_object(state_path, "incomplete slot state")
    if (
        state.get("schema_version") != "round-074-segmented-slot-state-v1"
        or state.get("plan_sha256") != plan.plan_sha256
        or state.get("slot_ordinal") != ordinal
        or state.get("phase") != "running"
        or isinstance(state.get("process_id"), bool)
        or not isinstance(state.get("process_id"), int)
        or int(state["process_id"]) <= 0
        or isinstance(state.get("monitor_sample_count"), bool)
        or not isinstance(state.get("monitor_sample_count"), int)
        or int(state["monitor_sample_count"]) <= 0
    ):
        raise ValueError("Round 75 incomplete slot state differs")
    logs: dict[str, object] = {}
    for name in ("capture.stdout.json", "capture.stderr.log"):
        path = slot_root / name
        logs[name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return {
        "slot_ordinal": ordinal,
        "role": reservation["role"],
        "reservation_sha256": reservation_sha256,
        "state_file_sha256": file_sha256(state_path),
        "state_phase": state["phase"],
        "monitor_sample_count": state["monitor_sample_count"],
        "logs": logs,
    }


def _audit_slots(
    config: Round75TerminalAuditConfig,
    *,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
) -> dict[str, object]:
    result_records: list[dict[str, object]] = []
    missed_records: list[dict[str, object]] = []
    outcomes: list[Round74SegmentedCohortSlotOutcome] = []
    incomplete_records: list[dict[str, object]] = []
    uncovered: list[int] = []
    raw_slot_roots = tuple(config.state_root.glob("slot-*"))
    raw_missed_paths = tuple((config.state_root / "missed-slots").glob("*.json"))
    if any(path.is_symlink() for path in (*raw_slot_roots, *raw_missed_paths)):
        raise ValueError("Round 75 terminal state contains a symbolic link")
    slot_roots = {
        _slot_ordinal(path, "slot-"): path for path in raw_slot_roots if path.is_dir()
    }
    missed_paths = {
        _slot_ordinal(path, ""): path for path in raw_missed_paths if path.is_file()
    }
    if any(ordinal >= plan.total_slots for ordinal in (*slot_roots, *missed_paths)):
        raise ValueError("Round 75 terminal slot ordinal exceeds plan")
    for ordinal in range(plan.total_slots):
        slot_root = slot_roots.get(ordinal)
        missed_path = missed_paths.get(ordinal)
        result_exists = slot_root is not None and (slot_root / "result.json").is_file()
        if result_exists and missed_path is not None:
            raise ValueError("Round 75 terminal slot has conflicting dispositions")
        if result_exists and slot_root is not None:
            outcome, record = _validate_result(slot_root, plan=plan)
            outcomes.append(outcome)
            result_records.append(record)
        elif missed_path is not None:
            missed_records.append(
                _validate_missed(
                    missed_path,
                    plan=plan,
                    contract_sha256=contract_sha256,
                )
            )
        elif slot_root is not None and (slot_root / "reservation.json").is_file():
            incomplete_record = _validate_incomplete_slot(slot_root, plan=plan)
            if incomplete_record["slot_ordinal"] != ordinal:
                raise ValueError("Round 75 incomplete slot identity differs")
            incomplete_records.append(incomplete_record)
        else:
            uncovered.append(ordinal)
    admitted = [outcome for outcome in outcomes if outcome.status == "admitted"]
    eligible_by_role = {
        role: sum(
            outcome.binding.eligible_anchor_duration_ns
            for outcome in admitted
            if outcome.role == role and outcome.binding is not None
        )
        for role in ROUND74_EVENT_PARTITION_ROLES
    }
    role_counts = {
        role: {
            "result": sum(record["role"] == role for record in result_records),
            "admitted": sum(outcome.role == role for outcome in admitted),
            "missed": sum(record["role"] == role for record in missed_records),
            "required_eligible_anchor_ns": (
                ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS[role]
            ),
            "observed_raw_eligible_anchor_ns": eligible_by_role[role],
            "eligible_anchor_quota_passed": eligible_by_role[role]
            >= ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS[role],
        }
        for role in ROUND74_EVENT_PARTITION_ROLES
    }
    statuses = Counter(str(record["status"]) for record in result_records)
    reasons = Counter(str(record["reason_code"]) for record in result_records)
    complete = (
        len(result_records) + len(missed_records) == plan.total_slots
        and not incomplete_records
        and not uncovered
    )
    return {
        "total_predeclared_slots": plan.total_slots,
        "terminal_result_count": len(result_records),
        "missed_receipt_count": len(missed_records),
        "incomplete_slot_ordinals": [
            record["slot_ordinal"] for record in incomplete_records
        ],
        "incomplete_slot_records": incomplete_records,
        "uncovered_slot_ordinals": uncovered,
        "all_slots_have_terminal_dispositions": complete,
        "result_status_counts": dict(sorted(statuses.items())),
        "result_reason_counts": dict(sorted(reasons.items())),
        "role_counts": role_counts,
        "result_manifest_sha256": _canonical_sha256(
            [
                {
                    "slot_ordinal": record["slot_ordinal"],
                    "result_sha256": record["result_sha256"],
                }
                for record in result_records
            ]
        ),
        "missed_manifest_sha256": _canonical_sha256(
            [
                {
                    "slot_ordinal": record["slot_ordinal"],
                    "missed_sha256": record["missed_sha256"],
                }
                for record in missed_records
            ]
        ),
    }


def _audit_storage(config: Round75TerminalAuditConfig) -> dict[str, object]:
    databases = sorted(config.data_root.glob(f"{ROUND75_SHARD_PREFIX}*.duckdb"))
    wals = sorted(config.data_root.glob(f"{ROUND75_SHARD_PREFIX}*.duckdb.wal"))
    if any(path.is_symlink() for path in (*databases, *wals)):
        raise ValueError("Round 75 terminal storage contains a symbolic link")
    database_records = [_file_record(path) for path in databases]
    wal_records = [_file_record(path) for path in wals]
    return {
        "database_files": database_records,
        "database_file_count": len(database_records),
        "database_bytes": sum(int(row["bytes"]) for row in database_records),
        "wal_files": wal_records,
        "wal_file_count": len(wal_records),
        "wal_bytes": sum(int(row["bytes"]) for row in wal_records),
        "all_campaign_wals_absent": not wal_records,
        "source_databases_opened": False,
    }


def _audit_service_state(
    config: Round75TerminalAuditConfig,
    *,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
) -> dict[str, object]:
    state = _load_object(config.service_state_path, "service state")
    state_sha256 = _validate_bound_object(
        state,
        digest_field="state_sha256",
        label="service state",
    )
    boundary = (
        plan.scheduled_start_wall_ns
        + plan.total_slots * ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS
    )
    passed = (
        state.get("schema_version") == ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION
        and state.get("plan_sha256") == plan.plan_sha256
        and state.get("contract_sha256") == contract_sha256
        and state.get("phase") == "campaign_terminal"
        and state.get("slot_ordinal") is None
        and isinstance(state.get("heartbeat_wall_ns"), int)
        and not isinstance(state.get("heartbeat_wall_ns"), bool)
        and int(state["heartbeat_wall_ns"]) >= boundary
        and state.get("credentials_used") is False
        and state.get("orders_submitted") is False
        and state.get("trading_authority") is False
    )
    return {
        "state_sha256": state_sha256,
        "phase": state.get("phase"),
        "heartbeat_wall_ns": state.get("heartbeat_wall_ns"),
        "campaign_boundary_wall_ns": boundary,
        "campaign_terminal_state_passed": passed,
        "lease_exclusively_available": _lease_available(config.lease_path),
    }


def _audit_supervisor_inspection(
    inspection: Mapping[str, object],
) -> dict[str, object]:
    canonical = dict(inspection)
    claimed = canonical.pop("inspection_sha256", None)
    digest_passed = isinstance(claimed, str) and claimed == _canonical_sha256(canonical)
    no_processes = all(
        inspection.get(field) == []
        for field in (
            "service_process_ids",
            "service_launcher_process_ids",
            "physical_service_process_ids",
            "capture_child_process_ids",
            "capture_child_launcher_process_ids",
            "physical_capture_child_process_ids",
        )
    )
    passed = (
        digest_passed
        and inspection.get("classification") == "campaign_terminal"
        and inspection.get("campaign_ended") is True
        and inspection.get("automatic_start_permitted") is False
        and inspection.get("automatic_repair_permitted") is False
        and inspection.get("state_error") == ""
        and no_processes
    )
    return {
        "inspection_sha256": claimed,
        "classification": inspection.get("classification"),
        "no_owned_processes_observed": no_processes,
        "automatic_start_permitted": inspection.get("automatic_start_permitted"),
        "inspection_passed": passed,
    }


def audit_round75_terminal_capture(
    config: Round75TerminalAuditConfig,
    *,
    supervisor_inspection: Mapping[str, object],
    observed_at_utc: str,
) -> dict[str, object]:
    """Audit hashes and terminal metadata without opening source databases."""

    config.validate()
    plan, sources = _load_sources(config)
    contract_sha256 = str(sources["contract_artifact_sha256"])
    coverage = _audit_slots(
        config,
        plan=plan,
        contract_sha256=contract_sha256,
    )
    storage = _audit_storage(config)
    service = _audit_service_state(
        config,
        plan=plan,
        contract_sha256=contract_sha256,
    )
    supervisor = _audit_supervisor_inspection(supervisor_inspection)
    quotas_passed = all(
        bool(row["eligible_anchor_quota_passed"])
        for row in coverage["role_counts"].values()
    )
    source_continuity_passed = (
        bool(coverage["all_slots_have_terminal_dispositions"])
        and bool(storage["all_campaign_wals_absent"])
        and bool(service["campaign_terminal_state_passed"])
        and bool(service["lease_exclusively_available"])
        and bool(supervisor["inspection_passed"])
    )
    report: dict[str, object] = {
        "schema_version": ROUND75_TERMINAL_AUDIT_SCHEMA_VERSION,
        "observed_at_utc": observed_at_utc,
        "status": "rejected_incomplete_campaign",
        "sources": sources,
        "service": service,
        "supervisor": supervisor,
        "coverage": coverage,
        "storage": storage,
        "gates": {
            "source_continuity_passed": source_continuity_passed,
            "role_eligible_anchor_quotas_passed": quotas_passed,
            "representative_training_completed": False,
            "model_data_access_permitted": False,
            "training_permitted": False,
            "tuning_permitted": False,
            "sealed_test_access_permitted": False,
            "financial_edge_established": False,
            "profitability_established": False,
            "ai_uplift_established": False,
            "trading_authority": False,
        },
        "scope": {
            "source_databases_opened": False,
            "source_wals_replayed": False,
            "target_data_accessed": False,
            "financial_outcomes_accessed": False,
            "model_training_performed": False,
            "credentials_used": False,
            "orders_submitted": False,
        },
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "ROUND75_TERMINAL_AUDIT_SCHEMA_VERSION",
    "Round75TerminalAuditConfig",
    "audit_round75_terminal_capture",
    "file_sha256",
]
