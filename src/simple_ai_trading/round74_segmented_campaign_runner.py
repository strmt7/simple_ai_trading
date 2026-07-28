"""Isolated one-slot runner for a frozen segmented Round 74 campaign."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import threading
import time

from .impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS,
    ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
    ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS,
    Round74SegmentedCohortPlan,
    Round74SegmentedTransportEpochAudit,
    load_round74_segmented_cohort_plan,
)
from .impact_absorption_store import ImpactAbsorptionStore
from .round74_active_qualification import (
    _durable_json_replace,
    _sha256_file,
    _stop_owned_process,
    _stream_reader,
)
from .round74_segmented_cohort_operator import (
    _canonical_sha256,
    _strict_json_mapping,
    audit_and_adjudicate_round74_segmented_supervisor,
)


ROUND74_SEGMENTED_CAMPAIGN_RUNNER_RESULT_SCHEMA_VERSION = (
    "round-074-segmented-campaign-slot-result-v1"
)
ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES = 64 * 1024 * 1024 * 1024
ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES = 512 * 1024 * 1024
ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES = 100 * 1024 * 1024 * 1024
ROUND74_SEGMENTED_CAMPAIGN_MONITOR_INTERVAL_SECONDS = 5.0
ROUND74_SEGMENTED_CAMPAIGN_HEARTBEAT_INTERVAL_SECONDS = 30.0

_FROZEN_SOURCE_PATHS = (
    "src/simple_ai_trading/impact_absorption_capture.py",
    "src/simple_ai_trading/impact_absorption_event_dataset.py",
    "src/simple_ai_trading/impact_absorption_event_segmented_cohort.py",
    "src/simple_ai_trading/impact_absorption_event_sequence.py",
    "src/simple_ai_trading/impact_absorption_store.py",
    "src/simple_ai_trading/round74_segmented_campaign_runner.py",
    "src/simple_ai_trading/round74_segmented_capture.py",
    "src/simple_ai_trading/round74_segmented_cohort_operator.py",
    "tools/run_round74_segmented_campaign_slot.py",
    "tools/run_round74_segmented_capture.py",
)


@dataclass(frozen=True)
class Round74SegmentedCampaignSlotSelection:
    """Current relation to the immutable prospective schedule."""

    status: str
    slot_ordinal: int | None
    offset_in_slot_period_ns: int | None


@dataclass(frozen=True)
class Round74SegmentedCampaignRunnerConfig:
    """Bounded host resources for one immutable campaign."""

    repository: Path
    plan_path: Path
    prerequisite_path: Path
    database_path: Path
    state_root: Path
    database_cap_bytes: int = ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES
    slot_growth_cap_bytes: int = ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES
    minimum_free_bytes: int = ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 2

    def validate(self) -> None:
        root = self.repository.resolve()
        paths = (
            self.plan_path.resolve(),
            self.prerequisite_path.resolve(),
            self.database_path.resolve(),
            self.state_root.resolve(),
        )
        if (
            not self.plan_path.is_file()
            or not self.prerequisite_path.is_file()
            or any(
                path != root and root not in path.parents
                for path in paths
            )
            or isinstance(self.database_cap_bytes, bool)
            or not isinstance(self.database_cap_bytes, int)
            or self.database_cap_bytes
            != ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES
            or isinstance(self.slot_growth_cap_bytes, bool)
            or not isinstance(self.slot_growth_cap_bytes, int)
            or self.slot_growth_cap_bytes
            != ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES
            or isinstance(self.minimum_free_bytes, bool)
            or not isinstance(self.minimum_free_bytes, int)
            or self.minimum_free_bytes
            != ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES
            or self.duckdb_memory_limit != "2GB"
            or isinstance(self.duckdb_threads, bool)
            or not isinstance(self.duckdb_threads, int)
            or self.duckdb_threads != 2
        ):
            raise ValueError("Round 74 segmented campaign runner config differs")


def select_round74_segmented_campaign_slot(
    plan: Round74SegmentedCohortPlan,
    *,
    now_wall_ns: int,
) -> Round74SegmentedCampaignSlotSelection:
    """Select only an on-time slot; never shift or replace the schedule."""

    plan.validate()
    if isinstance(now_wall_ns, bool) or not isinstance(now_wall_ns, int):
        raise ValueError("Round 74 segmented campaign wall time differs")
    if now_wall_ns < plan.scheduled_start_wall_ns:
        return Round74SegmentedCampaignSlotSelection(
            "before_campaign",
            None,
            None,
        )
    offset = now_wall_ns - plan.scheduled_start_wall_ns
    ordinal = offset // ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS
    if ordinal >= plan.total_slots:
        return Round74SegmentedCampaignSlotSelection(
            "after_campaign",
            None,
            None,
        )
    within = offset % ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS
    if within <= ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS:
        return Round74SegmentedCampaignSlotSelection(
            "open",
            int(ordinal),
            int(within),
        )
    return Round74SegmentedCampaignSlotSelection(
        "between_slots",
        int(ordinal),
        int(within),
    )


def _validate_prerequisite(
    config: Round74SegmentedCampaignRunnerConfig,
    plan: Round74SegmentedCohortPlan,
) -> None:
    payload = dict(
        _strict_json_mapping(
            config.prerequisite_path.read_text(encoding="utf-8"),
            "segmented campaign prerequisite",
        )
    )
    claimed = str(payload.pop("artifact_sha256", ""))
    source = payload.get("source")
    capture = payload.get("capture")
    epoch = payload.get("fresh_epoch_audit")
    verdict = payload.get("verdict")
    if (
        claimed != _canonical_sha256(payload)
        or claimed != plan.prerequisite_artifact_sha256
        or payload.get("schema_version")
        != "round-074-segmented-prerequisite-success-v1"
        or not isinstance(source, Mapping)
        or source.get("credentials_used") is not False
        or source.get("orders_submitted") is not False
        or not isinstance(capture, Mapping)
        or capture.get("started_wall_ns")
        != plan.prerequisite_window_start_wall_ns
        or capture.get("ended_wall_ns")
        != plan.prerequisite_window_end_wall_ns
        or capture.get("supervisor_status") != "completed"
        or capture.get("reconnect_count") != 0
        or capture.get("audit_passed") is not True
        or not isinstance(epoch, Mapping)
        or not isinstance(verdict, Mapping)
        or verdict.get("prerequisite_passed") is not True
        or verdict.get("cohort_plan_frozen") is not False
        or verdict.get("cohort_campaign_open") is not False
        or verdict.get("cohort_data_admitted") is not False
        or verdict.get("model_training_or_evaluation") is not False
        or verdict.get("profitability_or_edge_claim") is not False
        or verdict.get("trading_authority") is not False
    ):
        raise ValueError("Round 74 segmented campaign prerequisite differs")
    audited = Round74SegmentedTransportEpochAudit.from_dict(epoch)
    if (
        not audited.admission_supported
        or audited.run_id != capture.get("run_id")
        or audited.report_sha256 != capture.get("report_sha256")
        or audited.message_count != capture.get("message_count")
        or audited.frame_count != capture.get("frame_count")
        or audited.compressed_payload_bytes
        != capture.get("compressed_payload_bytes")
    ):
        raise ValueError("Round 74 segmented prerequisite epoch differs")


def _load_plan(config: Round74SegmentedCampaignRunnerConfig) -> Round74SegmentedCohortPlan:
    config.validate()
    plan = load_round74_segmented_cohort_plan(
        config.plan_path.read_text(encoding="utf-8")
    )
    _validate_prerequisite(config, plan)
    return plan


def _source_matches(
    repository: Path,
    implementation_commit: str,
) -> tuple[bool, tuple[str, ...]]:
    mismatches: list[str] = []
    for relative in _FROZEN_SOURCE_PATHS:
        completed = subprocess.run(  # nosec B603
            [
                "git",
                "diff",
                "--quiet",
                implementation_commit,
                "--",
                relative,
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0:
            mismatches.append(relative)
    return not mismatches, tuple(mismatches)


def _active_segmented_capture_processes() -> list[dict[str, object]]:
    if os.name == "nt":
        command = (
            "$items=Get-CimInstance Win32_Process | "
            "Where-Object {$_.Name -match '^python' -and "
            "$_.CommandLine -match "
            "'run_round74_segmented_capture.py'} | "
            "Select-Object ProcessId,CommandLine;"
            "@($items)|ConvertTo-Json -Compress"
        )
        completed = subprocess.run(  # nosec B603
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = json.loads(completed.stdout or "[]")
        rows = raw if isinstance(raw, list) else [raw]
        return [
            {
                "process_id": int(row.get("ProcessId", 0)),
                "command_line": str(row.get("CommandLine", "")),
            }
            for row in rows
            if isinstance(row, Mapping)
            and int(row.get("ProcessId", 0)) != os.getpid()
        ]
    output: list[dict[str, object]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("segmented campaign process inventory is unavailable")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        decoded = command_line.decode("utf-8", errors="replace")
        if "run_round74_segmented_capture.py" in decoded:
            output.append(
                {"process_id": int(entry.name), "command_line": decoded}
            )
    return output


def inspect_round74_segmented_campaign_readiness(
    config: Round74SegmentedCampaignRunnerConfig,
    *,
    now_wall_ns: int | None = None,
) -> dict[str, object]:
    """Return non-mutating evidence for the current schedule position."""

    plan = _load_plan(config)
    observed_ns = time.time_ns() if now_wall_ns is None else int(now_wall_ns)
    selection = select_round74_segmented_campaign_slot(
        plan,
        now_wall_ns=observed_ns,
    )
    source_passed, source_mismatches = _source_matches(
        config.repository,
        plan.implementation_git_commit,
    )
    database_bytes = (
        config.database_path.stat().st_size
        if config.database_path.exists()
        else 0
    )
    wal = Path(f"{config.database_path}.wal")
    wal_bytes = wal.stat().st_size if wal.exists() else 0
    free_bytes = shutil.disk_usage(config.database_path.parent).free
    processes = _active_segmented_capture_processes()
    slot_root = (
        None
        if selection.slot_ordinal is None
        else config.state_root / f"slot-{selection.slot_ordinal:03d}"
    )
    reservation = None if slot_root is None else slot_root / "reservation.json"
    checks = {
        "frozen_source_passed": source_passed,
        "database_cap_passed": (
            database_bytes + wal_bytes < config.database_cap_bytes
        ),
        "minimum_free_space_passed": free_bytes >= config.minimum_free_bytes,
        "wal_absent_before_slot_passed": wal_bytes == 0,
        "no_active_segmented_capture_passed": not processes,
        "slot_not_reserved_passed": (
            reservation is not None and not reservation.exists()
        ),
    }
    return {
        "schema_version": "round-074-segmented-campaign-readiness-v1",
        "observed_wall_ns": observed_ns,
        "plan_sha256": plan.plan_sha256,
        "selection": {
            "status": selection.status,
            "slot_ordinal": selection.slot_ordinal,
            "offset_in_slot_period_ns": selection.offset_in_slot_period_ns,
        },
        "can_start_now": selection.status == "open" and all(checks.values()),
        "checks": checks,
        "source_mismatches": list(source_mismatches),
        "database_bytes": database_bytes,
        "wal_bytes": wal_bytes,
        "free_bytes": free_bytes,
        "active_segmented_capture_processes": processes,
        "reservation_path": "" if reservation is None else str(reservation),
    }


def _reserve_slot(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def run_round74_segmented_campaign_current_slot(
    config: Round74SegmentedCampaignRunnerConfig,
    *,
    now_wall_ns: int | None = None,
) -> dict[str, object]:
    """Reserve, capture, audit, and adjudicate exactly one open slot."""

    readiness = inspect_round74_segmented_campaign_readiness(
        config,
        now_wall_ns=now_wall_ns,
    )
    if readiness["can_start_now"] is not True:
        raise ValueError("Round 74 segmented campaign is not ready for an open slot")
    plan = _load_plan(config)
    selection = readiness["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("Round 74 segmented campaign selection differs")
    ordinal = int(selection["slot_ordinal"])
    slot = plan.slot(ordinal)
    slot_root = config.state_root / f"slot-{ordinal:03d}"
    reservation_path = slot_root / "reservation.json"
    result_path = slot_root / "result.json"
    stdout_path = slot_root / "capture.stdout.json"
    stderr_path = slot_root / "capture.stderr.log"
    command = [
        sys.executable,
        str(config.repository / "tools" / "run_round74_segmented_capture.py"),
        "--database",
        str(config.database_path),
        "--database-size-cap-bytes",
        str(config.database_cap_bytes),
        "--memory-limit",
        config.duckdb_memory_limit,
        "--database-threads",
        str(config.duckdb_threads),
        "--progress-interval-seconds",
        "30",
        "--json",
    ]
    reservation = {
        "schema_version": "round-074-segmented-slot-reservation-v1",
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": ordinal,
        "role": slot.role,
        "scheduled_start_wall_ns": slot.scheduled_start_wall_ns,
        "reserved_wall_ns": time.time_ns(),
        "command_sha256": _canonical_sha256(command),
        "automatic_retry_permitted": False,
        "credentials_used": False,
        "orders_submitted": False,
    }
    reservation["reservation_sha256"] = _canonical_sha256(reservation)
    _reserve_slot(reservation_path, reservation)
    baseline_database = _path_bytes(config.database_path)
    baseline_wal = _path_bytes(Path(f"{config.database_path}.wal"))
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    monitor_samples: list[dict[str, object]] = []
    breaches: list[str] = []
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with (
        stdout_path.open("x", encoding="utf-8", newline="\n", buffering=1) as stdout,
        stderr_path.open("x", encoding="utf-8", newline="\n", buffering=1) as stderr,
    ):
        process = subprocess.Popen(  # nosec B603
            command,
            cwd=config.repository,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        if process.stdout is None or process.stderr is None:
            _stop_owned_process(process)
            raise RuntimeError("Round 74 segmented capture pipes are missing")
        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, stdout, stdout_lines),
            kwargs={"mirror": False},
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, stderr, stderr_lines),
            kwargs={"mirror": True},
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        started = time.monotonic()
        next_heartbeat = started
        hard_deadline_ns = (
            slot.scheduled_end_wall_ns
            + ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS
            + ROUND74_SEGMENTED_COHORT_END_OVERHEAD_NS
        )
        stop_method = ""
        while process.poll() is None:
            now = time.monotonic()
            database_bytes = _path_bytes(config.database_path)
            wal_bytes = _path_bytes(Path(f"{config.database_path}.wal"))
            growth = max(
                0,
                database_bytes
                + wal_bytes
                - baseline_database
                - baseline_wal,
            )
            free_bytes = shutil.disk_usage(config.database_path.parent).free
            if now >= next_heartbeat:
                sample = {
                    "elapsed_seconds": round(now - started, 3),
                    "database_bytes": database_bytes,
                    "wal_bytes": wal_bytes,
                    "slot_growth_bytes": growth,
                    "free_bytes": free_bytes,
                }
                monitor_samples.append(sample)
                _durable_json_replace(
                    slot_root / "state.json",
                    {
                        "schema_version": "round-074-segmented-slot-state-v1",
                        "plan_sha256": plan.plan_sha256,
                        "slot_ordinal": ordinal,
                        "phase": "running",
                        "process_id": process.pid,
                        "last_progress": sample,
                        "monitor_sample_count": len(monitor_samples),
                    },
                )
                print(
                    "round74-segmented-campaign-progress: "
                    f"slot={ordinal} elapsed={now - started:.1f}s "
                    f"database_bytes={database_bytes} wal_bytes={wal_bytes} "
                    f"growth_bytes={growth} free_bytes={free_bytes}",
                    file=sys.stderr,
                    flush=True,
                )
                next_heartbeat += (
                    ROUND74_SEGMENTED_CAMPAIGN_HEARTBEAT_INTERVAL_SECONDS
                )
            if growth > config.slot_growth_cap_bytes:
                breaches.append("slot_growth_cap_exceeded")
            if database_bytes + wal_bytes > config.database_cap_bytes:
                breaches.append("database_cap_exceeded")
            if free_bytes < config.minimum_free_bytes:
                breaches.append("minimum_free_space_breached")
            if time.time_ns() > hard_deadline_ns:
                breaches.append("slot_deadline_exceeded")
            if breaches:
                stop_method = _stop_owned_process(process)
                break
            time.sleep(ROUND74_SEGMENTED_CAMPAIGN_MONITOR_INTERVAL_SECONDS)
        return_code = process.wait(timeout=60)
        stdout_thread.join(timeout=30)
        stderr_thread.join(timeout=30)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise RuntimeError("Round 74 segmented output drain did not terminate")
    if breaches:
        raise ValueError(
            "Round 74 segmented campaign resource breach: "
            f"{breaches}; stop={stop_method or 'not_requested'}"
        )
    supervisor = _strict_json_mapping(
        "".join(stdout_lines),
        "segmented campaign supervisor",
    )
    attempts = supervisor.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Round 74 segmented campaign attempts differ")
    if attempts:
        with ImpactAbsorptionStore(
            config.database_path,
            read_only=True,
            memory_limit=config.duckdb_memory_limit,
            threads=config.duckdb_threads,
        ) as store:
            adjudication = audit_and_adjudicate_round74_segmented_supervisor(
                plan,
                slot_ordinal=ordinal,
                supervisor_payload=supervisor,
                store=store,
            )
    else:
        adjudication = audit_and_adjudicate_round74_segmented_supervisor(
            plan,
            slot_ordinal=ordinal,
            supervisor_payload=supervisor,
            store=None,
        )
    result: dict[str, object] = {
        "schema_version": ROUND74_SEGMENTED_CAMPAIGN_RUNNER_RESULT_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": ordinal,
        "role": slot.role,
        "reservation_sha256": reservation["reservation_sha256"],
        "capture_return_code": return_code,
        "capture_stdout_sha256": _sha256_file(stdout_path),
        "capture_stderr_sha256": _sha256_file(stderr_path),
        "monitor_sample_count": len(monitor_samples),
        "maximum_observed_slot_growth_bytes": max(
            (int(row["slot_growth_bytes"]) for row in monitor_samples),
            default=0,
        ),
        "watchdog_breaches": [],
        "adjudication": adjudication.as_dict(),
        "automatic_retry_permitted": False,
        "credentials_used": False,
        "orders_submitted": False,
        "profitability_or_edge_claim": False,
        "trading_authority": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    if result_path.exists():
        raise FileExistsError("Round 74 segmented slot result already exists")
    _durable_json_replace(result_path, result)
    _durable_json_replace(
        slot_root / "state.json",
        {
            "schema_version": "round-074-segmented-slot-state-v1",
            "plan_sha256": plan.plan_sha256,
            "slot_ordinal": ordinal,
            "phase": "terminal",
            "result_sha256": result["result_sha256"],
            "completed_at_utc": (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ),
        },
    )
    return result


__all__ = [
    "ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES",
    "ROUND74_SEGMENTED_CAMPAIGN_HEARTBEAT_INTERVAL_SECONDS",
    "ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES",
    "ROUND74_SEGMENTED_CAMPAIGN_MONITOR_INTERVAL_SECONDS",
    "ROUND74_SEGMENTED_CAMPAIGN_RUNNER_RESULT_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES",
    "Round74SegmentedCampaignRunnerConfig",
    "Round74SegmentedCampaignSlotSelection",
    "inspect_round74_segmented_campaign_readiness",
    "run_round74_segmented_campaign_current_slot",
    "select_round74_segmented_campaign_slot",
]
