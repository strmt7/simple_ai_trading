"""Fail-closed inputs for the complete Round 74 segmented campaign."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS,
    ROUND74_SEGMENTED_COHORT_MISSED_REASON,
    Round74SegmentedCohortCoverage,
    Round74SegmentedCohortPlan,
    Round74SegmentedCohortRunBinding,
    Round74SegmentedCohortSlotOutcome,
    load_round74_segmented_cohort_plan,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .round74_segmented_campaign_runner import (
    ROUND74_SEGMENTED_CAMPAIGN_RUNNER_RESULT_SCHEMA_VERSION,
)
from .round74_segmented_cohort_operator import (
    Round74SegmentedSlotAdjudication,
)
from .round74_target_assembly_manifest import (
    load_and_audit_round74_target_assembly_manifest,
)
from .storage import write_json_atomic


ROUND74_SEGMENTED_DEVELOPMENT_INPUTS_SCHEMA_VERSION = (
    "round-074-segmented-development-inputs-v1"
)
ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION = (
    "round-074-segmented-recovery-outcome-v1"
)
ROUND74_SEGMENTED_DEVELOPMENT_EXECUTION_ENVIRONMENT = (
    "binance_usdm_mainnet"
)
ROUND74_SEGMENTED_INPUT_MAXIMUM_JSON_BYTES = 16 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLOT_DIRECTORY = re.compile(r"slot-(\d{3})")
_SLOT_FILES = frozenset(
    {
        "reservation.json",
        "state.json",
        "capture.stdout.json",
        "capture.stderr.log",
        "result.json",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "slot_ordinal",
        "role",
        "reservation_sha256",
        "capture_return_code",
        "capture_stdout_sha256",
        "capture_stderr_sha256",
        "monitor_sample_count",
        "maximum_observed_slot_growth_bytes",
        "watchdog_breaches",
        "adjudication",
        "automatic_retry_permitted",
        "credentials_used",
        "orders_submitted",
        "profitability_or_edge_claim",
        "trading_authority",
        "result_sha256",
    }
)
_RESERVATION_KEYS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "slot_ordinal",
        "role",
        "scheduled_start_wall_ns",
        "reserved_wall_ns",
        "command_sha256",
        "automatic_retry_permitted",
        "credentials_used",
        "orders_submitted",
        "reservation_sha256",
    }
)
_TERMINAL_STATE_KEYS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "slot_ordinal",
        "phase",
        "completed_at_utc",
        "result_sha256",
    }
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


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 segmented {label} digest differs")
    return selected


def _read_small_bytes(path: Path, label: str) -> bytes:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > ROUND74_SEGMENTED_INPUT_MAXIMUM_JSON_BYTES
    ):
        raise ValueError(f"Round 74 segmented {label} file differs")
    return path.read_bytes()


def _strict_json_mapping(raw: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(
                    f"Round 74 segmented {label} has duplicate JSON keys"
                )
            output[key] = value
        return output

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 74 segmented {label} is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Round 74 segmented {label} root differs")
    return parsed


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    return _strict_json_mapping(_read_small_bytes(path, label), label)


def _sha256_file(path: Path, label: str) -> str:
    return hashlib.sha256(_read_small_bytes(path, label)).hexdigest()


def _campaign_terminal_wall_ns(plan: Round74SegmentedCohortPlan) -> int:
    return (
        plan.slot(plan.total_slots - 1).scheduled_end_wall_ns
        + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
    )


def _validate_slot_directory(slot_directory: Path) -> tuple[Path, ...]:
    if slot_directory.is_symlink() or not slot_directory.is_dir():
        raise ValueError("Round 74 segmented slot directory differs")
    entries = tuple(slot_directory.iterdir())
    if any(
        entry.name not in _SLOT_FILES
        or entry.is_symlink()
        or not entry.is_file()
        for entry in entries
    ):
        raise ValueError("Round 74 segmented slot file panel differs")
    return tuple(sorted(entries, key=lambda path: path.name))


def _load_campaign_slot_result(
    plan: Round74SegmentedCohortPlan,
    *,
    slot_ordinal: int,
    slot_directory: Path,
) -> tuple[Round74SegmentedCohortSlotOutcome, str]:
    slot = plan.slot(slot_ordinal)
    entries = _validate_slot_directory(slot_directory)
    if {path.name for path in entries} != _SLOT_FILES:
        raise ValueError("Round 74 segmented terminal slot file panel differs")
    reservation_path = slot_directory / "reservation.json"
    state_path = slot_directory / "state.json"
    stdout_path = slot_directory / "capture.stdout.json"
    stderr_path = slot_directory / "capture.stderr.log"
    result_path = slot_directory / "result.json"

    reservation = dict(
        _read_json_mapping(reservation_path, "slot reservation")
    )
    if set(reservation) != _RESERVATION_KEYS:
        raise ValueError("Round 74 segmented reservation contract differs")
    claimed_reservation = _require_sha256(
        reservation.pop("reservation_sha256"),
        "reservation",
    )
    if claimed_reservation != _canonical_sha256(reservation):
        raise ValueError("Round 74 segmented reservation identity differs")
    reservation["reservation_sha256"] = claimed_reservation

    result = dict(_read_json_mapping(result_path, "slot result"))
    if set(result) != _RESULT_KEYS:
        raise ValueError("Round 74 segmented result contract differs")
    claimed_result = _require_sha256(
        result.pop("result_sha256"),
        "result",
    )
    if claimed_result != _canonical_sha256(result):
        raise ValueError("Round 74 segmented result identity differs")
    result["result_sha256"] = claimed_result

    state = dict(_read_json_mapping(state_path, "slot state"))
    stdout = dict(_read_json_mapping(stdout_path, "capture supervisor"))
    adjudication_payload = result.get("adjudication")
    if not isinstance(adjudication_payload, Mapping):
        raise ValueError("Round 74 segmented result adjudication differs")
    adjudication = Round74SegmentedSlotAdjudication.from_dict(
        plan,
        adjudication_payload,
    )
    if (
        set(state) != _TERMINAL_STATE_KEYS
        or reservation["schema_version"]
        != "round-074-segmented-slot-reservation-v1"
        or reservation["plan_sha256"] != plan.plan_sha256
        or reservation["slot_ordinal"] != slot.ordinal
        or reservation["role"] != slot.role
        or reservation["scheduled_start_wall_ns"]
        != slot.scheduled_start_wall_ns
        or isinstance(reservation["reserved_wall_ns"], bool)
        or not isinstance(reservation["reserved_wall_ns"], int)
        or not (
            slot.scheduled_start_wall_ns
            <= reservation["reserved_wall_ns"]
            <= slot.start_window_end_wall_ns
        )
        or _SHA256.fullmatch(str(reservation["command_sha256"])) is None
        or reservation["automatic_retry_permitted"] is not False
        or reservation["credentials_used"] is not False
        or reservation["orders_submitted"] is not False
        or result["schema_version"]
        != ROUND74_SEGMENTED_CAMPAIGN_RUNNER_RESULT_SCHEMA_VERSION
        or result["plan_sha256"] != plan.plan_sha256
        or result["slot_ordinal"] != slot.ordinal
        or result["role"] != slot.role
        or result["reservation_sha256"] != claimed_reservation
        or isinstance(result["capture_return_code"], bool)
        or not isinstance(result["capture_return_code"], int)
        or result["capture_return_code"] < 0
        or isinstance(result["monitor_sample_count"], bool)
        or not isinstance(result["monitor_sample_count"], int)
        or result["monitor_sample_count"] < 0
        or isinstance(result["maximum_observed_slot_growth_bytes"], bool)
        or not isinstance(result["maximum_observed_slot_growth_bytes"], int)
        or result["maximum_observed_slot_growth_bytes"] < 0
        or result["automatic_retry_permitted"] is not False
        or result["credentials_used"] is not False
        or result["orders_submitted"] is not False
        or result["profitability_or_edge_claim"] is not False
        or result["trading_authority"] is not False
        or result["watchdog_breaches"] != []
        or result["capture_stdout_sha256"]
        != _sha256_file(stdout_path, "capture stdout")
        or result["capture_stderr_sha256"]
        != _sha256_file(stderr_path, "capture stderr")
        or state["schema_version"] != "round-074-segmented-slot-state-v1"
        or state["plan_sha256"] != plan.plan_sha256
        or state["slot_ordinal"] != slot.ordinal
        or state["phase"] != "terminal"
        or not isinstance(state["completed_at_utc"], str)
        or not state["completed_at_utc"]
        or state["result_sha256"] != claimed_result
        or adjudication.slot_ordinal != slot.ordinal
        or adjudication.outcome.role != slot.role
        or adjudication.as_dict() != adjudication_payload
        or stdout != dict(adjudication_payload["supervisor"])
    ):
        raise ValueError("Round 74 segmented terminal slot binding differs")
    return adjudication.outcome, claimed_result


@dataclass(frozen=True)
class Round74SegmentedRecoveryOutcome:
    """Immutable proof that a result-less scheduled slot is not model data."""

    plan_sha256: str
    slot_ordinal: int
    role: str
    observed_wall_ns: int
    slot_file_sha256: tuple[tuple[str, str], ...]
    outcome: Round74SegmentedCohortSlotOutcome
    schema_version: str = ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "plan_sha256": self.plan_sha256,
            "slot_ordinal": self.slot_ordinal,
            "role": self.role,
            "observed_wall_ns": self.observed_wall_ns,
            "result_absent": True,
            "slot_file_sha256": dict(self.slot_file_sha256),
        }

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha256(self._evidence_payload())

    def validate(self, plan: Round74SegmentedCohortPlan) -> None:
        plan.validate()
        slot = plan.slot(self.slot_ordinal)
        names = tuple(name for name, _digest in self.slot_file_sha256)
        if (
            self.schema_version
            != ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION
            or self.plan_sha256 != plan.plan_sha256
            or self.role != slot.role
            or isinstance(self.observed_wall_ns, bool)
            or not isinstance(self.observed_wall_ns, int)
            or self.observed_wall_ns < _campaign_terminal_wall_ns(plan)
            or names != tuple(sorted(names))
            or len(names) != len(set(names))
            or any(name not in _SLOT_FILES - {"result.json"} for name in names)
            or any(
                _SHA256.fullmatch(digest) is None
                for _name, digest in self.slot_file_sha256
            )
            or self.outcome.plan_sha256 != plan.plan_sha256
            or self.outcome.slot_ordinal != slot.ordinal
            or self.outcome.role != slot.role
            or self.outcome.status != "missed"
            or self.outcome.reason_code != ROUND74_SEGMENTED_COHORT_MISSED_REASON
            or self.outcome.binding is not None
            or self.outcome.evidence_sha256 != self.evidence_sha256
        ):
            raise ValueError("Round 74 segmented recovery outcome differs")
        self.outcome.validate()

    @property
    def recovery_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            **self._evidence_payload(),
            "evidence_sha256": self.evidence_sha256,
            "outcome": self.outcome.as_dict(),
        }
        if include_sha256:
            payload["recovery_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        plan: Round74SegmentedCohortPlan,
        value: Mapping[str, object],
    ) -> Round74SegmentedRecoveryOutcome:
        payload = dict(value)
        claimed = _require_sha256(
            payload.pop("recovery_sha256", ""),
            "recovery",
        )
        raw_files = payload.get("slot_file_sha256")
        raw_outcome = payload.get("outcome")
        if (
            claimed != _canonical_sha256(payload)
            or not isinstance(raw_files, Mapping)
            or not isinstance(raw_outcome, Mapping)
            or payload.get("result_absent") is not True
        ):
            raise ValueError("Round 74 segmented recovery payload differs")
        selected = cls(
            plan_sha256=str(payload["plan_sha256"]),
            slot_ordinal=int(payload["slot_ordinal"]),
            role=str(payload["role"]),
            observed_wall_ns=int(payload["observed_wall_ns"]),
            slot_file_sha256=tuple(
                sorted(
                    (str(name), str(digest))
                    for name, digest in raw_files.items()
                )
            ),
            outcome=Round74SegmentedCohortSlotOutcome.from_dict(raw_outcome),
            schema_version=str(payload["schema_version"]),
        )
        if (
            payload.get("evidence_sha256") != selected.evidence_sha256
            or selected.as_dict(include_sha256=False) != payload
        ):
            raise ValueError("Round 74 segmented recovery identity differs")
        selected.validate(plan)
        return selected

    def verify_slot_directory(self, slot_directory: Path | None) -> None:
        if slot_directory is None:
            observed: tuple[tuple[str, str], ...] = ()
        else:
            entries = _validate_slot_directory(slot_directory)
            if any(path.name == "result.json" for path in entries):
                raise ValueError("Round 74 segmented recovery result appeared")
            observed = tuple(
                (
                    path.name,
                    hashlib.sha256(
                        _read_small_bytes(path, "recovery slot evidence")
                    ).hexdigest(),
                )
                for path in entries
            )
        if observed != self.slot_file_sha256:
            raise ValueError("Round 74 segmented recovery slot evidence differs")


def build_round74_segmented_recovery_outcome(
    plan: Round74SegmentedCohortPlan,
    *,
    slot_ordinal: int,
    observed_wall_ns: int,
    slot_directory: str | Path | None,
) -> Round74SegmentedRecoveryOutcome:
    """Build a missed outcome only after the entire schedule is terminal."""

    plan.validate()
    slot = plan.slot(slot_ordinal)
    directory = None if slot_directory is None else Path(slot_directory)
    if directory is None:
        files: tuple[tuple[str, str], ...] = ()
    else:
        entries = _validate_slot_directory(directory)
        if any(path.name == "result.json" for path in entries):
            raise ValueError("Round 74 segmented recovery rejects a result")
        files = tuple(
            (
                path.name,
                hashlib.sha256(
                    _read_small_bytes(path, "recovery slot evidence")
                ).hexdigest(),
            )
            for path in entries
        )
    evidence_payload = {
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": slot.ordinal,
        "role": slot.role,
        "observed_wall_ns": observed_wall_ns,
        "result_absent": True,
        "slot_file_sha256": dict(files),
    }
    outcome = Round74SegmentedCohortSlotOutcome(
        plan_sha256=plan.plan_sha256,
        slot_ordinal=slot.ordinal,
        role=slot.role,
        status="missed",
        reason_code=ROUND74_SEGMENTED_COHORT_MISSED_REASON,
        evidence_sha256=_canonical_sha256(evidence_payload),
    )
    selected = Round74SegmentedRecoveryOutcome(
        plan_sha256=plan.plan_sha256,
        slot_ordinal=slot.ordinal,
        role=slot.role,
        observed_wall_ns=observed_wall_ns,
        slot_file_sha256=files,
        outcome=outcome,
    )
    selected.validate(plan)
    return selected


def write_round74_segmented_recovery_outcome(
    recovery: Round74SegmentedRecoveryOutcome,
    *,
    plan: Round74SegmentedCohortPlan,
    path: str | Path,
) -> Path:
    """Publish one recovery immutably and verify a strict reload."""

    recovery.validate(plan)
    selected = Path(path)
    if selected.is_symlink() or selected.parent.is_symlink():
        raise ValueError("Round 74 segmented recovery path differs")
    if selected.exists():
        restored = Round74SegmentedRecoveryOutcome.from_dict(
            plan,
            _read_json_mapping(selected, "recovery outcome"),
        )
        if restored.recovery_sha256 != recovery.recovery_sha256:
            raise FileExistsError(
                "Round 74 immutable segmented recovery differs"
            )
        return selected
    write_json_atomic(selected, recovery.as_dict(), indent=2, sort_keys=True)
    restored = Round74SegmentedRecoveryOutcome.from_dict(
        plan,
        _read_json_mapping(selected, "recovery outcome"),
    )
    if restored.recovery_sha256 != recovery.recovery_sha256:
        raise RuntimeError("Round 74 segmented recovery reload differs")
    return selected


@dataclass(frozen=True)
class Round74SegmentedDevelopmentInputs:
    """Complete transport ledger and development-only target assemblies."""

    plan: Round74SegmentedCohortPlan
    coverage: Round74SegmentedCohortCoverage
    terminal_observed_wall_ns: int
    slot_evidence: tuple[tuple[int, str, str], ...]
    target_assemblies: tuple[tuple[str, Round74SourceTargetAssembly], ...]
    target_manifest_sha256_by_run_id: tuple[tuple[str, str], ...]
    schema_version: str = ROUND74_SEGMENTED_DEVELOPMENT_INPUTS_SCHEMA_VERSION

    def validate(self) -> None:
        self.plan.validate()
        self.coverage.validate(self.plan)
        assemblies = dict(self.target_assemblies)
        manifests = dict(self.target_manifest_sha256_by_run_id)
        development_entries = tuple(
            entry for entry in self.coverage.partition.entries if entry.role != "test"
        )
        development_run_ids = tuple(entry.run_id for entry in development_entries)
        evidence_ordinals = tuple(
            ordinal for ordinal, _kind, _digest in self.slot_evidence
        )
        outcome_by_ordinal = {
            outcome.slot_ordinal: outcome for outcome in self.coverage.outcomes
        }
        if (
            self.schema_version
            != ROUND74_SEGMENTED_DEVELOPMENT_INPUTS_SCHEMA_VERSION
            or self.coverage.plan_sha256 != self.plan.plan_sha256
            or isinstance(self.terminal_observed_wall_ns, bool)
            or not isinstance(self.terminal_observed_wall_ns, int)
            or self.terminal_observed_wall_ns
            < _campaign_terminal_wall_ns(self.plan)
            or evidence_ordinals != tuple(range(self.plan.total_slots))
            or any(
                kind not in {"result", "recovery"}
                or _SHA256.fullmatch(digest) is None
                or (
                    kind == "recovery"
                    and outcome_by_ordinal[ordinal].status != "missed"
                )
                or (
                    kind == "result"
                    and outcome_by_ordinal[ordinal].status == "missed"
                )
                for ordinal, kind, digest in self.slot_evidence
            )
            or tuple(assemblies) != development_run_ids
            or len(assemblies) != len(self.target_assemblies)
            or tuple(manifests) != development_run_ids
            or len(manifests) != len(self.target_manifest_sha256_by_run_id)
            or any(_SHA256.fullmatch(digest) is None for digest in manifests.values())
        ):
            raise ValueError("Round 74 segmented development input identity differs")
        for assembly in assemblies.values():
            if (
                not isinstance(assembly, Round74SourceTargetAssembly)
                or assembly.spec.execution_environment
                != ROUND74_SEGMENTED_DEVELOPMENT_EXECUTION_ENVIRONMENT
            ):
                raise ValueError(
                    "Round 74 segmented development target environment differs"
                )

    @property
    def inputs_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        role_counts = {
            role: sum(entry.role == role for entry in self.coverage.partition.entries)
            for role in ("training", "tuning", "test")
        }
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan.plan_sha256,
            "coverage_sha256": self.coverage.coverage_sha256,
            "partition_sha256": self.coverage.partition.partition_sha256,
            "terminal_observed_wall_ns": self.terminal_observed_wall_ns,
            "slot_evidence": [
                {
                    "slot_ordinal": ordinal,
                    "kind": kind,
                    "sha256": digest,
                }
                for ordinal, kind, digest in self.slot_evidence
            ],
            "admitted_role_counts": role_counts,
            "development_target_assembly_sha256": {
                run_id: assembly.assembly_sha256
                for run_id, assembly in self.target_assemblies
            },
            "development_target_manifest_sha256": dict(
                self.target_manifest_sha256_by_run_id
            ),
            "execution_environment": (
                ROUND74_SEGMENTED_DEVELOPMENT_EXECUTION_ENVIRONMENT
            ),
            "sealed_test_target_assemblies_read": False,
            "source_database_access": "not_opened",
            "model_training_started": False,
        }

    def target_assembly_by_run_id(
        self,
    ) -> dict[str, Round74SourceTargetAssembly]:
        self.validate()
        return dict(self.target_assemblies)

    def development_bindings_by_run_id(
        self,
    ) -> dict[str, Round74SegmentedCohortRunBinding]:
        self.validate()
        bindings = {
            outcome.binding.run_id: outcome.binding
            for outcome in self.coverage.outcomes
            if outcome.binding is not None and outcome.binding.role != "test"
        }
        expected = tuple(
            entry.run_id
            for entry in self.coverage.partition.entries
            if entry.role != "test"
        )
        if set(bindings) != set(expected):
            raise ValueError(
                "Round 74 segmented development binding panel differs"
            )
        return {run_id: bindings[run_id] for run_id in expected}


def _load_complete_campaign_outcomes(
    plan: Round74SegmentedCohortPlan,
    *,
    state_root: Path,
    recovery_directory: Path,
) -> tuple[
    tuple[Round74SegmentedCohortSlotOutcome, ...],
    tuple[tuple[int, str, str], ...],
]:
    expected_names = {
        f"slot-{ordinal:03d}" for ordinal in range(plan.total_slots)
    }
    entries = tuple(state_root.iterdir())
    observed_directories: dict[int, Path] = {}
    for entry in entries:
        match = _SLOT_DIRECTORY.fullmatch(entry.name)
        if (
            match is None
            or entry.name not in expected_names
            or entry.is_symlink()
            or not entry.is_dir()
        ):
            raise ValueError("Round 74 segmented campaign state panel differs")
        observed_directories[int(match.group(1))] = entry

    missing_result_ordinals = tuple(
        ordinal
        for ordinal in range(plan.total_slots)
        if ordinal not in observed_directories
        or not (observed_directories[ordinal] / "result.json").is_file()
    )
    recovery_entries = tuple(recovery_directory.iterdir())
    expected_recovery_names = {
        f"{ordinal:03d}.json" for ordinal in missing_result_ordinals
    }
    if (
        {entry.name for entry in recovery_entries} != expected_recovery_names
        or any(entry.is_symlink() or not entry.is_file() for entry in recovery_entries)
    ):
        raise ValueError("Round 74 segmented recovery file panel differs")

    outcomes: list[Round74SegmentedCohortSlotOutcome] = []
    evidence: list[tuple[int, str, str]] = []
    missing = set(missing_result_ordinals)
    for ordinal in range(plan.total_slots):
        slot_directory = observed_directories.get(ordinal)
        if ordinal not in missing:
            if slot_directory is None:
                raise ValueError("Round 74 segmented result slot disappeared")
            outcome, digest = _load_campaign_slot_result(
                plan,
                slot_ordinal=ordinal,
                slot_directory=slot_directory,
            )
            kind = "result"
        else:
            recovery_path = recovery_directory / f"{ordinal:03d}.json"
            recovery = Round74SegmentedRecoveryOutcome.from_dict(
                plan,
                _read_json_mapping(recovery_path, "recovery outcome"),
            )
            if recovery.slot_ordinal != ordinal:
                raise ValueError("Round 74 segmented recovery ordinal differs")
            recovery.verify_slot_directory(slot_directory)
            outcome = recovery.outcome
            digest = recovery.recovery_sha256
            kind = "recovery"
        outcomes.append(outcome)
        evidence.append((ordinal, kind, digest))
    return tuple(outcomes), tuple(evidence)


def load_round74_segmented_development_inputs(
    *,
    plan_path: str | Path,
    state_root: str | Path,
    recovery_outcome_directory: str | Path,
    target_assembly_directory: str | Path,
    source_artifact_root: str | Path,
    terminal_observed_wall_ns: int,
) -> Round74SegmentedDevelopmentInputs:
    """Load the complete campaign without opening test target manifests."""

    selected_plan_path = Path(plan_path)
    selected_state_root = Path(state_root)
    selected_recovery_directory = Path(recovery_outcome_directory)
    selected_assembly_directory = Path(target_assembly_directory)
    selected_source_root = Path(source_artifact_root)
    paths = (
        selected_plan_path,
        selected_state_root,
        selected_recovery_directory,
        selected_assembly_directory,
        selected_source_root,
    )
    if (
        any(path.is_symlink() for path in paths)
        or not selected_plan_path.is_file()
        or not selected_state_root.is_dir()
        or not selected_recovery_directory.is_dir()
        or not selected_assembly_directory.is_dir()
        or not selected_source_root.is_dir()
    ):
        raise ValueError("Round 74 segmented development path panel differs")
    plan = load_round74_segmented_cohort_plan(
        _read_small_bytes(selected_plan_path, "cohort plan").decode("utf-8")
    )
    if (
        isinstance(terminal_observed_wall_ns, bool)
        or not isinstance(terminal_observed_wall_ns, int)
        or terminal_observed_wall_ns < _campaign_terminal_wall_ns(plan)
    ):
        raise ValueError("Round 74 segmented campaign is not terminal")
    outcomes, slot_evidence = _load_complete_campaign_outcomes(
        plan,
        state_root=selected_state_root,
        recovery_directory=selected_recovery_directory,
    )
    coverage = Round74SegmentedCohortCoverage.build(plan, outcomes)
    binding_by_run_id = {
        outcome.binding.run_id: outcome.binding
        for outcome in outcomes
        if outcome.binding is not None
    }
    development_run_ids = tuple(
        entry.run_id
        for entry in coverage.partition.entries
        if entry.role != "test"
    )
    expected_assembly_names = {f"{run_id}.json" for run_id in development_run_ids}
    assembly_entries = tuple(selected_assembly_directory.iterdir())
    if (
        {entry.name for entry in assembly_entries} != expected_assembly_names
        or any(entry.is_symlink() or not entry.is_file() for entry in assembly_entries)
    ):
        raise ValueError(
            "Round 74 segmented development target manifest panel differs"
        )
    loaded = tuple(
        (
            run_id,
            load_and_audit_round74_target_assembly_manifest(
                manifest_path=selected_assembly_directory / f"{run_id}.json",
                source_artifact_root=selected_source_root,
            ),
        )
        for run_id in development_run_ids
    )
    if any(
        manifest.run_id != run_id
        or manifest.cohort_binding_sha256
        != binding_by_run_id[run_id].binding_sha256
        for run_id, manifest in loaded
    ):
        raise ValueError(
            "Round 74 segmented development target manifest binding differs"
        )
    selected = Round74SegmentedDevelopmentInputs(
        plan=plan,
        coverage=coverage,
        terminal_observed_wall_ns=terminal_observed_wall_ns,
        slot_evidence=slot_evidence,
        target_assemblies=tuple(
            (run_id, manifest.assembly) for run_id, manifest in loaded
        ),
        target_manifest_sha256_by_run_id=tuple(
            (run_id, manifest.manifest_sha256) for run_id, manifest in loaded
        ),
    )
    selected.validate()
    return selected


__all__ = [
    "ROUND74_SEGMENTED_DEVELOPMENT_EXECUTION_ENVIRONMENT",
    "ROUND74_SEGMENTED_DEVELOPMENT_INPUTS_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_INPUT_MAXIMUM_JSON_BYTES",
    "ROUND74_SEGMENTED_RECOVERY_OUTCOME_SCHEMA_VERSION",
    "Round74SegmentedDevelopmentInputs",
    "Round74SegmentedRecoveryOutcome",
    "build_round74_segmented_recovery_outcome",
    "load_round74_segmented_development_inputs",
    "write_round74_segmented_recovery_outcome",
]
