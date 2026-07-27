"""Fail-closed one-slot operator for the predeclared Round 74 cohort."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import threading
import time

from .impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from .impact_absorption_capture import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)
from .impact_absorption_event_cohort import (
    Round74EventCohortPlan,
    Round74EventCohortRunBinding,
    bind_round74_event_cohort_supervisor,
    build_round74_event_run_partition,
    load_round74_event_cohort_binding,
    load_round74_event_cohort_plan,
)
from .round74_active_qualification import (
    ROUND74_ACTIVE_PREFLIGHT_SHA256,
    ROUND74_ACTIVE_RESULT_SCHEMA_VERSION,
    _active_capture_processes,
    _canonical_sha256,
    _database_and_wal_bytes,
    _durable_json_replace,
    _sha256_file,
    _stop_owned_process,
    _stream_reader,
    _strict_json_object,
    reserve_round74_active_attempt,
)
from .round74_active_adjudication import (
    ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH,
    ROUND74_ACTIVE_ADJUDICATION_SCHEMA_VERSION,
    ROUND74_ACTIVE_RESULT_FILE_SHA256,
    ROUND74_ACTIVE_RESULT_SHA256,
    build_round74_active_adjudication,
)


ROUND74_EVENT_COHORT_PLAN_RELATIVE_PATH = Path(
    "docs/model-research/action-value/round-074-event-cohort-plan-v5-r1.json"
)
ROUND74_EVENT_COHORT_PLAN_SHA256 = (
    "a7e3afb41992599137b845e4e0d4ecee3b9c6cebadc9347617f551dcc04ec223"
)
ROUND74_ACTIVE_ADJUDICATION_SHA256 = (
    "fef501a34da6b36bb004b1731b9751a1cdb52ce649fdc4fa6579640c7af962a7"
)
ROUND74_EVENT_COHORT_OPERATOR_STATE_SCHEMA_VERSION = (
    "round-074-event-cohort-slot-state-v4"
)
ROUND74_EVENT_COHORT_OPERATOR_RESULT_SCHEMA_VERSION = (
    "round-074-event-cohort-slot-result-v4"
)
ROUND74_EVENT_COHORT_STARTUP_LAUNCH_SCHEMA_VERSION = (
    "round-074-event-cohort-startup-launch-v1"
)
ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES = 24 * 1024 * 1024 * 1024
ROUND74_EVENT_COHORT_FREE_SPACE_MINIMUM_BYTES = 100 * 1024 * 1024 * 1024
ROUND74_EVENT_COHORT_PER_SLOT_GROWTH_LIMIT_BYTES = 512 * 1024 * 1024
ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS = 300
ROUND74_EVENT_COHORT_MAXIMUM_STARTUP_LAUNCHES = 2
ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS = 0.25
ROUND74_EVENT_COHORT_RELAUNCH_START_HEADROOM_NS = 1_000_000_000

_DATABASE_RELATIVE_PATH = Path("data/microstructure.duckdb")
_CAPTURE_ARGUMENTS = (
    "-m",
    "simple_ai_trading",
    "impact-capture",
    "--database",
    str(_DATABASE_RELATIVE_PATH).replace("\\", "/"),
    "--mode",
    "qualification",
    "--schema-version",
    "v10",
    "--duration-seconds",
    "3600",
    "--maximum-reconnects",
    "0",
    "--database-size-cap-bytes",
    str(ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES),
    "--progress-interval-seconds",
    "30",
    "--json",
)
_MONITOR_INTERVAL_SECONDS = 5.0
_HEARTBEAT_INTERVAL_SECONDS = 30.0
_TRANSIENT_STARTUP_ERROR_MARKERS = (
    "ConnectionClosedError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "InvalidHandshake",
    "InvalidStatus",
    "OSError",
    "TimeoutError",
)


@dataclass(frozen=True)
class Round74CohortSlotSelection:
    """Current wall-clock relation to the immutable slot schedule."""

    status: str
    slot_ordinal: int | None
    offset_in_slot_period_ns: int | None


@dataclass(frozen=True)
class _Round74CaptureLaunch:
    launch_ordinal: int
    return_code: int
    breaches: tuple[str, ...]
    stop_method: str
    stdout_text: str
    stderr_text: str
    stdout_path: Path
    stderr_path: Path
    monitor_samples: tuple[dict[str, object], ...]
    database_bytes_after: int
    wal_bytes_after: int
    database_mtime_ns_after: int
    wal_mtime_ns_after: int


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def load_round74_cohort_operator_plan(
    repository: Path,
) -> Round74EventCohortPlan:
    path = repository.resolve() / ROUND74_EVENT_COHORT_PLAN_RELATIVE_PATH
    plan = load_round74_event_cohort_plan(path.read_text(encoding="utf-8"))
    if plan.plan_sha256 != ROUND74_EVENT_COHORT_PLAN_SHA256:
        raise ValueError("Round 74 cohort operator plan digest differs")
    return plan


def select_round74_cohort_slot(
    plan: Round74EventCohortPlan,
    *,
    now_wall_ns: int,
) -> Round74CohortSlotSelection:
    """Select only an on-time slot; never shift or replace the schedule."""

    plan.validate()
    if isinstance(now_wall_ns, bool) or not isinstance(now_wall_ns, int):
        raise ValueError("Round 74 cohort wall time must be an integer")
    if now_wall_ns < plan.scheduled_start_wall_ns:
        return Round74CohortSlotSelection("before_campaign", None, None)
    offset = now_wall_ns - plan.scheduled_start_wall_ns
    ordinal = offset // plan.slot_period_ns
    if ordinal >= plan.total_slots:
        return Round74CohortSlotSelection("after_campaign", None, None)
    within = offset % plan.slot_period_ns
    if within <= plan.start_tolerance_ns:
        return Round74CohortSlotSelection("open", int(ordinal), int(within))
    return Round74CohortSlotSelection("between_slots", int(ordinal), int(within))


def _active_result_path(repository: Path) -> Path:
    return (
        repository
        / "data"
        / "round74-v10-active-qualification"
        / ROUND74_ACTIVE_PREFLIGHT_SHA256
        / "result.json"
    )


def validate_round74_active_prerequisite(
    repository: Path,
) -> dict[str, object]:
    """Require the exact raw result and deterministic adjudication."""

    root = repository.resolve()
    path = _active_result_path(root)
    if _sha256_file(path) != ROUND74_ACTIVE_RESULT_FILE_SHA256:
        raise ValueError("Round 74 active prerequisite file identity differs")
    result = _strict_json_object(
        path.read_text(encoding="utf-8"),
        "Round 74 active prerequisite result",
    )
    claimed = str(result.get("result_sha256", ""))
    canonical = dict(result)
    canonical.pop("result_sha256", None)
    verdict = result.get("verdict")
    audits = result.get("fresh_process_audits")
    supervisor = result.get("supervisor_report")
    if (
        claimed != _canonical_sha256(canonical)
        or claimed != ROUND74_ACTIVE_RESULT_SHA256
        or result.get("schema_version") != ROUND74_ACTIVE_RESULT_SCHEMA_VERSION
        or result.get("preflight_sha256") != ROUND74_ACTIVE_PREFLIGHT_SHA256
        or not isinstance(verdict, Mapping)
        or verdict.get("outcome") != "failed"
        or verdict.get("active_qualified") is not False
        or verdict.get("capture_data_passed") is not True
        or verdict.get("activity_label") != "active"
        or verdict.get("errors") != ["fresh_audit_or_report_identity_failed"]
        or result.get("capture_return_code") != 0
        or result.get("watchdog_breaches") != []
        or result.get("automatic_retry_permitted") is not False
        or result.get("orders_submitted") is not False
        or result.get("credentials_used") is not False
        or not isinstance(supervisor, Mapping)
        or supervisor.get("status") != "completed"
        or supervisor.get("qualification_passed") is not True
        or supervisor.get("reconnect_count") != 0
        or not isinstance(audits, list)
        or len(audits) != 1
        or not isinstance(audits[0], Mapping)
        or audits[0].get("return_code") != 0
        or not isinstance(audits[0].get("audit"), Mapping)
        or audits[0]["audit"].get("passed") is not True
    ):
        raise ValueError("Round 74 active prerequisite did not pass exactly")

    adjudication_path = root / ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH
    adjudication = _strict_json_object(
        adjudication_path.read_text(encoding="utf-8"),
        "Round 74 active prerequisite adjudication",
    )
    adjudication_claimed = str(adjudication.get("artifact_sha256", ""))
    adjudicated_at = datetime.fromisoformat(
        str(adjudication.get("adjudicated_at_utc", "")).replace("Z", "+00:00")
    )
    expected = build_round74_active_adjudication(
        root,
        adjudicated_at_utc=adjudicated_at,
    )
    if (
        adjudication.get("schema_version") != ROUND74_ACTIVE_ADJUDICATION_SCHEMA_VERSION
        or adjudication_claimed != ROUND74_ACTIVE_ADJUDICATION_SHA256
        or adjudication != expected
    ):
        raise ValueError(
            "Round 74 active prerequisite adjudication did not pass exactly"
        )
    return {
        "source_result": result,
        "adjudication": adjudication,
    }


def _campaign_root(repository: Path) -> Path:
    return (
        repository / "data" / "round74-event-cohort" / ROUND74_EVENT_COHORT_PLAN_SHA256
    )


def _slot_root(repository: Path, ordinal: int) -> Path:
    return _campaign_root(repository) / "slots" / f"{ordinal:03d}"


def _binding_path(repository: Path, ordinal: int) -> Path:
    return _campaign_root(repository) / "bindings" / f"{ordinal:03d}.json"


def _load_contiguous_bindings(
    repository: Path,
    plan: Round74EventCohortPlan,
    *,
    before_ordinal: int,
) -> list[Round74EventCohortRunBinding]:
    bindings: list[Round74EventCohortRunBinding] = []
    for ordinal in range(before_ordinal):
        path = _binding_path(repository, ordinal)
        if not path.is_file():
            raise ValueError(f"Round 74 cohort prior slot {ordinal} is missing")
        binding = load_round74_event_cohort_binding(path.read_text(encoding="utf-8"))
        if binding.slot_ordinal != ordinal or binding.plan_sha256 != plan.plan_sha256:
            raise ValueError(f"Round 74 cohort prior slot {ordinal} binding differs")
        bindings.append(binding)
    return bindings


def _persist_halt(
    repository: Path,
    *,
    slot_ordinal: int,
    reason: str,
) -> None:
    path = _campaign_root(repository) / "halt.json"
    if path.exists():
        return
    payload = {
        "schema_version": "round-074-event-cohort-halt-v1",
        "plan_sha256": ROUND74_EVENT_COHORT_PLAN_SHA256,
        "slot_ordinal": slot_ordinal,
        "observed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "reason": reason[:2_000],
        "automatic_retry_permitted": False,
        "failed_or_missed_slot_replacement_permitted": False,
        "orders_submitted": False,
        "credentials_used": False,
    }
    payload["halt_sha256"] = _canonical_sha256(payload)
    try:
        reserve_round74_active_attempt(path, payload)
    except FileExistsError:
        pass


def inspect_round74_cohort_readiness(
    repository: Path,
    *,
    now_wall_ns: int | None = None,
    process_provider=_active_capture_processes,
) -> dict[str, object]:
    """Inspect current slot readiness without reserving or launching."""

    root = repository.resolve()
    plan = load_round74_cohort_operator_plan(root)
    observed_ns = time.time_ns() if now_wall_ns is None else int(now_wall_ns)
    selection = select_round74_cohort_slot(plan, now_wall_ns=observed_ns)
    halt_path = _campaign_root(root) / "halt.json"
    prerequisite_passed = False
    prerequisite_error = ""
    try:
        validate_round74_active_prerequisite(root)
        prerequisite_passed = True
    except (OSError, ValueError) as exc:
        prerequisite_error = f"{type(exc).__name__}:{exc}"
    prior_bindings_passed = False
    prior_bindings_error = ""
    if selection.slot_ordinal is not None:
        try:
            _load_contiguous_bindings(
                root,
                plan,
                before_ordinal=selection.slot_ordinal,
            )
            prior_bindings_passed = True
        except (OSError, ValueError) as exc:
            prior_bindings_error = f"{type(exc).__name__}:{exc}"
    database = root / _DATABASE_RELATIVE_PATH
    database_bytes, wal_bytes = _database_and_wal_bytes(database)
    free_bytes = shutil.disk_usage(database.parent).free
    processes = process_provider()
    ordinal = selection.slot_ordinal
    reservation_exists = (
        False
        if ordinal is None
        else (_slot_root(root, ordinal) / "attempt-reservation.json").exists()
    )
    binding_exists = False if ordinal is None else _binding_path(root, ordinal).exists()
    checks = {
        "active_prerequisite_passed": prerequisite_passed,
        "campaign_not_halted": not halt_path.exists(),
        "prior_bindings_passed": prior_bindings_passed,
        "no_slot_reservation": not reservation_exists,
        "no_slot_binding": not binding_exists,
        "no_active_capture_process": not processes,
        "wal_absent": wal_bytes == 0,
        "global_database_cap_headroom": (
            database_bytes + 536_870_912
            < ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES
        ),
        "minimum_free_space": (
            free_bytes >= ROUND74_EVENT_COHORT_FREE_SPACE_MINIMUM_BYTES
        ),
    }
    return {
        "schema_version": "round-074-event-cohort-readiness-v1",
        "observed_wall_ns": observed_ns,
        "plan_sha256": plan.plan_sha256,
        "slot_status": selection.status,
        "slot_ordinal": ordinal,
        "ready_for_current_slot": (selection.status == "open" and all(checks.values())),
        "checks": checks,
        "prerequisite_error": prerequisite_error,
        "prior_bindings_error": prior_bindings_error,
        "database_bytes": database_bytes,
        "wal_bytes": wal_bytes,
        "free_bytes": free_bytes,
        "active_capture_processes": processes,
    }


def _audit_slot(
    repository: Path,
    executable: Path,
    slot_root: Path,
    *,
    run_id: str,
    timeout_seconds: float,
) -> tuple[dict[str, object], float]:
    stdout_path = slot_root / "audit.stdout.json"
    stderr_path = slot_root / "audit.stderr.log"
    command = [
        str(executable),
        "-m",
        "simple_ai_trading",
        "impact-audit",
        "--database",
        str(_DATABASE_RELATIVE_PATH).replace("\\", "/"),
        "--run-id",
        run_id,
        "--json",
    ]
    started = time.monotonic()
    completed = subprocess.run(  # nosec B603
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    elapsed = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    audit = _strict_json_object(completed.stdout, "Round 74 cohort fresh audit")
    audit["_operator_evidence"] = {
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_sha256": _sha256_file(stderr_path),
    }
    return audit, elapsed


def _validate_slot_audit(
    audit: Mapping[str, object],
    binding: Round74EventCohortRunBinding,
    supervisor: Mapping[str, object],
) -> None:
    attempts = supervisor.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != 1
        or not isinstance(attempts[0], Mapping)
    ):
        raise ValueError("Round 74 cohort supervisor attempts differ")
    report = attempts[0]
    evidence = audit.get("_operator_evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("return_code") != 0
        or audit.get("passed") is not True
        or audit.get("errors") != []
        or audit.get("run_id") != binding.run_id
        or audit.get("stored_report_sha256") != binding.report_sha256
        or audit.get("capture_contract_sha256")
        != report.get("capture_contract_sha256")
        or audit.get("frame_count") != binding.frame_count
        or audit.get("message_count") != binding.message_count
        or audit.get("compressed_payload_bytes")
        != binding.compressed_payload_bytes
    ):
        raise ValueError("Round 74 cohort fresh audit identity differs")


def _raise_for_capture_process_failure(
    *,
    return_code: int,
    breaches: list[str],
    stop_method: str,
    stdout_lines: list[str],
) -> None:
    if return_code == 0 and not breaches:
        return
    stdout_state = "present" if "".join(stdout_lines).strip() else "empty"
    raise ValueError(
        "Round 74 cohort capture process failed: "
        f"return_code={return_code} breaches={breaches} "
        f"stop={stop_method or 'not_requested'} stdout={stdout_state}"
    )


def _run_capture_launch(
    repository: Path,
    plan: Round74EventCohortPlan,
    *,
    ordinal: int,
    launch_ordinal: int,
    command: list[str],
    baseline_database: int,
    baseline_wal: int,
    baseline_database_mtime_ns: int,
    baseline_wal_mtime_ns: int,
    state: dict[str, object],
) -> _Round74CaptureLaunch:
    slot = plan.slot(ordinal)
    slot_root = _slot_root(repository, ordinal)
    state_path = slot_root / "state.json"
    stdout_path = slot_root / f"capture-launch-{launch_ordinal:03d}.stdout.json"
    stderr_path = slot_root / f"capture-launch-{launch_ordinal:03d}.stderr.log"
    database = repository / _DATABASE_RELATIVE_PATH
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    monitor_samples: list[dict[str, object]] = []
    breaches: list[str] = []
    stop_method = ""
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with (
        stdout_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        ) as stdout_file,
        stderr_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        ) as stderr_file,
    ):
        process = subprocess.Popen(  # nosec B603
            command,
            cwd=repository,
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
            stop_method = _stop_owned_process(process)
            raise RuntimeError("Round 74 cohort capture pipes were not created")
        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, stdout_file, stdout_lines),
            kwargs={"mirror": False},
            name=f"round74-cohort-{ordinal:03d}-{launch_ordinal:03d}-stdout",
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, stderr_file, stderr_lines),
            kwargs={"mirror": True},
            name=f"round74-cohort-{ordinal:03d}-{launch_ordinal:03d}-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()
        started_monotonic = time.monotonic()
        state.update(
            {
                "phase": "running",
                "process_id": process.pid,
                "current_startup_launch_ordinal": launch_ordinal,
                "current_capture_stdout": stdout_path.name,
                "current_capture_stderr": stderr_path.name,
            }
        )
        _durable_json_replace(state_path, state)
        next_heartbeat = started_monotonic
        hard_deadline_wall_ns = (
            slot.scheduled_end_wall_ns
            + plan.start_tolerance_ns
            + plan.maximum_end_overhead_ns
        )
        while process.poll() is None:
            now_monotonic = time.monotonic()
            elapsed = now_monotonic - started_monotonic
            database_bytes, wal_bytes = _database_and_wal_bytes(database)
            growth = max(
                0,
                database_bytes + wal_bytes - baseline_database - baseline_wal,
            )
            free_bytes = shutil.disk_usage(database.parent).free
            if now_monotonic >= next_heartbeat:
                sample = {
                    "startup_launch_ordinal": launch_ordinal,
                    "elapsed_seconds": round(elapsed, 3),
                    "database_bytes": database_bytes,
                    "wal_bytes": wal_bytes,
                    "database_and_wal_growth_bytes": growth,
                    "free_bytes": free_bytes,
                }
                monitor_samples.append(sample)
                state.update(
                    {
                        "last_progress_at_utc": (
                            datetime.now(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z")
                        ),
                        "last_progress": sample,
                        "monitor_sample_count": len(monitor_samples),
                    }
                )
                _durable_json_replace(state_path, state)
                print(
                    "round74-cohort-progress: "
                    f"slot={ordinal} launch={launch_ordinal} "
                    f"elapsed={elapsed:.1f}s "
                    f"database_bytes={database_bytes} wal_bytes={wal_bytes} "
                    f"growth_bytes={growth} free_bytes={free_bytes}",
                    file=sys.stderr,
                    flush=True,
                )
                next_heartbeat += _HEARTBEAT_INTERVAL_SECONDS
            if growth > ROUND74_EVENT_COHORT_PER_SLOT_GROWTH_LIMIT_BYTES:
                breaches.append("database_and_wal_growth_limit_exceeded")
            if free_bytes < ROUND74_EVENT_COHORT_FREE_SPACE_MINIMUM_BYTES:
                breaches.append("minimum_free_space_breached")
            if time.time_ns() > hard_deadline_wall_ns:
                breaches.append("slot_capture_deadline_exceeded")
            if breaches:
                stop_method = _stop_owned_process(process)
                break
            time.sleep(_MONITOR_INTERVAL_SECONDS)
        return_code = process.wait(timeout=60)
        stdout_thread.join(timeout=30)
        stderr_thread.join(timeout=30)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise RuntimeError("Round 74 cohort output drain did not terminate")

    final_database, final_wal = _database_and_wal_bytes(database)
    final_database_mtime_ns = _path_mtime_ns(database)
    final_wal_mtime_ns = _path_mtime_ns(Path(f"{database}.wal"))
    return _Round74CaptureLaunch(
        launch_ordinal=launch_ordinal,
        return_code=return_code,
        breaches=tuple(breaches),
        stop_method=stop_method,
        stdout_text="".join(stdout_lines),
        stderr_text="".join(stderr_lines),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        monitor_samples=tuple(monitor_samples),
        database_bytes_after=final_database,
        wal_bytes_after=final_wal,
        database_mtime_ns_after=final_database_mtime_ns,
        wal_mtime_ns_after=final_wal_mtime_ns,
    )


def _pre_admission_startup_failure(
    launch: _Round74CaptureLaunch,
    *,
    baseline_database: int,
    baseline_wal: int,
    baseline_database_mtime_ns: int,
    baseline_wal_mtime_ns: int,
    active_capture_processes: list[dict[str, object]],
) -> dict[str, object] | None:
    """Return an exact zero-evidence transient startup failure, otherwise none."""

    try:
        supervisor = _strict_json_object(
            launch.stdout_text,
            "Round 74 cohort failed-startup supervisor",
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    startup_errors = supervisor.get("startup_errors")
    terminal_error = supervisor.get("terminal_error")
    if (
        launch.return_code == 0
        or launch.breaches
        or launch.stop_method
        or baseline_wal != 0
        or hashlib.sha256(launch.stdout_text.encode("utf-8")).hexdigest()
        != _sha256_file(launch.stdout_path)
        or hashlib.sha256(launch.stderr_text.encode("utf-8")).hexdigest()
        != _sha256_file(launch.stderr_path)
        or launch.database_bytes_after != baseline_database
        or launch.wal_bytes_after != baseline_wal
        or launch.database_mtime_ns_after != baseline_database_mtime_ns
        or launch.wal_mtime_ns_after != baseline_wal_mtime_ns
        or active_capture_processes
        or supervisor.get("schema_version")
        != "round-074-capture-supervisor-report-v1"
        or supervisor.get("design_sha256") != ROUND74_CAPTURE_DESIGN_SHA256
        or supervisor.get("capture_schema_version")
        != IMPACT_CAPTURE_V10_SCHEMA_VERSION
        or supervisor.get("capture_contract_sha256")
        != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or supervisor.get("status") != "failed"
        or supervisor.get("qualification_passed") is not False
        or supervisor.get("selected_run_id") != ""
        or supervisor.get("attempts") != []
        or supervisor.get("attempt_count") != 1
        or supervisor.get("reconnect_count") != 0
        or supervisor.get("reconnect_delays_seconds") != []
        or supervisor.get("attempt_evidence_combined") is not False
        or not isinstance(startup_errors, list)
        or len(startup_errors) != 1
        or not isinstance(startup_errors[0], str)
        or terminal_error != startup_errors[0]
        or not startup_errors[0].startswith("startup:")
        or not any(
            source in startup_errors[0]
            for source in ("public_source:", "market_source:")
        )
        or not any(
            marker in startup_errors[0]
            for marker in _TRANSIENT_STARTUP_ERROR_MARKERS
        )
    ):
        return None
    return supervisor


def _startup_relaunch_fits_slot(
    plan: Round74EventCohortPlan,
    *,
    ordinal: int,
    now_wall_ns: int,
) -> bool:
    backoff_ns = int(
        ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS * 1_000_000_000
    )
    return (
        now_wall_ns + backoff_ns + ROUND74_EVENT_COHORT_RELAUNCH_START_HEADROOM_NS
        <= plan.slot(ordinal).start_window_end_wall_ns
    )


def _persist_startup_launch_evidence(
    repository: Path,
    plan: Round74EventCohortPlan,
    launch: _Round74CaptureLaunch,
    *,
    ordinal: int,
    baseline_database: int,
    baseline_wal: int,
    baseline_database_mtime_ns: int,
    baseline_wal_mtime_ns: int,
    active_capture_processes: list[dict[str, object]],
    supervisor: Mapping[str, object] | None,
    disposition: str,
    relaunch_permitted: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ROUND74_EVENT_COHORT_STARTUP_LAUNCH_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": ordinal,
        "role": plan.slot(ordinal).role,
        "launch_ordinal": launch.launch_ordinal,
        "disposition": disposition,
        "return_code": launch.return_code,
        "breaches": list(launch.breaches),
        "stop_method": launch.stop_method,
        "stdout_sha256": _sha256_file(launch.stdout_path),
        "stderr_sha256": _sha256_file(launch.stderr_path),
        "supervisor_sha256": (
            "" if supervisor is None else _canonical_sha256(dict(supervisor))
        ),
        "selected_run_id": (
            "" if supervisor is None else str(supervisor.get("selected_run_id", ""))
        ),
        "supervisor_attempts": (
            None if supervisor is None else supervisor.get("attempts")
        ),
        "startup_errors": (
            None if supervisor is None else supervisor.get("startup_errors")
        ),
        "database_bytes_before": baseline_database,
        "wal_bytes_before": baseline_wal,
        "database_mtime_ns_before": baseline_database_mtime_ns,
        "wal_mtime_ns_before": baseline_wal_mtime_ns,
        "database_bytes_after": launch.database_bytes_after,
        "wal_bytes_after": launch.wal_bytes_after,
        "database_mtime_ns_after": launch.database_mtime_ns_after,
        "wal_mtime_ns_after": launch.wal_mtime_ns_after,
        "database_or_wal_changed": (
            launch.database_bytes_after != baseline_database
            or launch.wal_bytes_after != baseline_wal
            or launch.database_mtime_ns_after != baseline_database_mtime_ns
            or launch.wal_mtime_ns_after != baseline_wal_mtime_ns
        ),
        "active_capture_processes_after": active_capture_processes,
        "slot_retry_permitted": False,
        "pre_admission_startup_relaunch_permitted": relaunch_permitted,
        "credentials_used": False,
        "orders_submitted": False,
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    path = (
        _slot_root(repository, ordinal)
        / f"startup-launch-{launch.launch_ordinal:03d}.json"
    )
    _durable_json_replace(path, payload)
    return payload


def _run_slot_process(
    repository: Path,
    plan: Round74EventCohortPlan,
    *,
    ordinal: int,
) -> tuple[dict[str, object], dict[str, object]]:
    slot = plan.slot(ordinal)
    slot_root = _slot_root(repository, ordinal)
    state_path = slot_root / "state.json"
    database = repository / _DATABASE_RELATIVE_PATH
    executable = repository / ".venv" / "Scripts" / "python.exe"
    command = [str(executable), *_CAPTURE_ARGUMENTS]
    baseline_database, baseline_wal = _database_and_wal_bytes(database)
    baseline_database_mtime_ns = _path_mtime_ns(database)
    baseline_wal_mtime_ns = _path_mtime_ns(Path(f"{database}.wal"))
    reserved_at = datetime.now(timezone.utc)
    reservation = {
        "schema_version": ROUND74_EVENT_COHORT_OPERATOR_STATE_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": ordinal,
        "role": slot.role,
        "reserved_at_utc": reserved_at.isoformat().replace("+00:00", "Z"),
        "command": command,
        "baseline_database_bytes": baseline_database,
        "baseline_wal_bytes": baseline_wal,
        "baseline_database_mtime_ns": baseline_database_mtime_ns,
        "baseline_wal_mtime_ns": baseline_wal_mtime_ns,
        "automatic_retry_permitted": False,
        "maximum_pre_admission_startup_launches": (
            ROUND74_EVENT_COHORT_MAXIMUM_STARTUP_LAUNCHES
        ),
        "pre_admission_startup_relaunch_backoff_seconds": (
            ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS
        ),
    }
    reserve_round74_active_attempt(
        slot_root / "attempt-reservation.json",
        reservation,
    )
    state: dict[str, object] = {**reservation, "phase": "reserved"}
    _durable_json_replace(state_path, state)

    startup_launch_evidence: list[dict[str, object]] = []
    accepted_launch: _Round74CaptureLaunch | None = None
    supervisor: dict[str, object] | None = None
    for launch_ordinal in range(
        1,
        ROUND74_EVENT_COHORT_MAXIMUM_STARTUP_LAUNCHES + 1,
    ):
        if launch_ordinal > 1:
            current_database, current_wal = _database_and_wal_bytes(database)
            current_database_mtime_ns = _path_mtime_ns(database)
            current_wal_mtime_ns = _path_mtime_ns(Path(f"{database}.wal"))
            if (
                current_database != baseline_database
                or current_wal != baseline_wal
                or current_database_mtime_ns != baseline_database_mtime_ns
                or current_wal_mtime_ns != baseline_wal_mtime_ns
                or _active_capture_processes()
                or not _startup_relaunch_fits_slot(
                    plan,
                    ordinal=ordinal,
                    now_wall_ns=time.time_ns(),
                )
            ):
                raise ValueError(
                    "Round 74 cohort pre-admission startup relaunch "
                    "readiness changed"
                )
        launch = _run_capture_launch(
            repository,
            plan,
            ordinal=ordinal,
            launch_ordinal=launch_ordinal,
            command=command,
            baseline_database=baseline_database,
            baseline_wal=baseline_wal,
            baseline_database_mtime_ns=baseline_database_mtime_ns,
            baseline_wal_mtime_ns=baseline_wal_mtime_ns,
            state=state,
        )
        if launch.return_code == 0 and not launch.breaches:
            try:
                supervisor = _strict_json_object(
                    launch.stdout_text,
                    "Round 74 cohort supervisor",
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                active_processes = _active_capture_processes()
                evidence = _persist_startup_launch_evidence(
                    repository,
                    plan,
                    launch,
                    ordinal=ordinal,
                    baseline_database=baseline_database,
                    baseline_wal=baseline_wal,
                    baseline_database_mtime_ns=baseline_database_mtime_ns,
                    baseline_wal_mtime_ns=baseline_wal_mtime_ns,
                    active_capture_processes=active_processes,
                    supervisor=None,
                    disposition="invalid_completed_supervisor",
                    relaunch_permitted=False,
                )
                startup_launch_evidence.append(evidence)
                raise
            accepted_launch = launch
            break

        active_processes = _active_capture_processes()
        failed_supervisor = _pre_admission_startup_failure(
            launch,
            baseline_database=baseline_database,
            baseline_wal=baseline_wal,
            baseline_database_mtime_ns=baseline_database_mtime_ns,
            baseline_wal_mtime_ns=baseline_wal_mtime_ns,
            active_capture_processes=active_processes,
        )
        relaunch_permitted = (
            failed_supervisor is not None
            and launch_ordinal < ROUND74_EVENT_COHORT_MAXIMUM_STARTUP_LAUNCHES
            and _startup_relaunch_fits_slot(
                plan,
                ordinal=ordinal,
                now_wall_ns=time.time_ns(),
            )
        )
        evidence = _persist_startup_launch_evidence(
            repository,
            plan,
            launch,
            ordinal=ordinal,
            baseline_database=baseline_database,
            baseline_wal=baseline_wal,
            baseline_database_mtime_ns=baseline_database_mtime_ns,
            baseline_wal_mtime_ns=baseline_wal_mtime_ns,
            active_capture_processes=active_processes,
            supervisor=failed_supervisor,
            disposition=(
                "excluded_transient_pre_admission_startup"
                if relaunch_permitted
                else "terminal_capture_failure"
            ),
            relaunch_permitted=relaunch_permitted,
        )
        startup_launch_evidence.append(evidence)
        state.update(
            {
                "phase": (
                    "pre_admission_startup_relaunch_wait"
                    if relaunch_permitted
                    else "running"
                ),
                "pre_admission_startup_launches": startup_launch_evidence,
            }
        )
        _durable_json_replace(state_path, state)
        if relaunch_permitted:
            time.sleep(ROUND74_EVENT_COHORT_STARTUP_RELAUNCH_BACKOFF_SECONDS)
            continue
        _raise_for_capture_process_failure(
            return_code=launch.return_code,
            breaches=list(launch.breaches),
            stop_method=launch.stop_method,
            stdout_lines=[launch.stdout_text],
        )

    if accepted_launch is None or supervisor is None:
        raise RuntimeError("Round 74 cohort has no admissible capture launch")
    attempts = supervisor.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != 1
        or not isinstance(attempts[0], Mapping)
        or isinstance(attempts[0].get("process_io_delta_write_bytes"), bool)
        or not isinstance(
            attempts[0].get("process_io_delta_write_bytes"),
            int,
        )
        or int(attempts[0]["process_io_delta_write_bytes"])
        > ROUND74_EVENT_COHORT_PROCESS_IO_LIMIT_BYTES
    ):
        raise ValueError("Round 74 cohort process I/O evidence differs")
    binding = bind_round74_event_cohort_supervisor(
        plan,
        slot_ordinal=ordinal,
        supervisor_payload=supervisor,
    )
    audit, audit_elapsed = _audit_slot(
        repository,
        executable,
        slot_root,
        run_id=binding.run_id,
        timeout_seconds=plan.fresh_audit_timeout_ns / 1_000_000_000,
    )
    _validate_slot_audit(audit, binding, supervisor)
    if audit_elapsed > plan.fresh_audit_timeout_ns / 1_000_000_000:
        raise ValueError("Round 74 cohort fresh audit exceeded its bound")
    final_database, final_wal = _database_and_wal_bytes(database)
    if final_wal != 0:
        raise ValueError("Round 74 cohort WAL remains after fresh audit")
    report = {
        "schema_version": ROUND74_EVENT_COHORT_OPERATOR_RESULT_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "slot_ordinal": ordinal,
        "role": slot.role,
        "capture_return_code": accepted_launch.return_code,
        "capture_stdout_sha256": _sha256_file(accepted_launch.stdout_path),
        "capture_stderr_sha256": _sha256_file(accepted_launch.stderr_path),
        "admitted_startup_launch_ordinal": accepted_launch.launch_ordinal,
        "pre_admission_startup_launches": startup_launch_evidence,
        "supervisor_sha256": _canonical_sha256(supervisor),
        "fresh_audit": audit,
        "monitor_samples": list(accepted_launch.monitor_samples),
        "database_bytes_before": baseline_database,
        "wal_bytes_before": baseline_wal,
        "database_bytes_after": final_database,
        "wal_bytes_after": final_wal,
        "binding": binding.as_dict(),
        "orders_submitted": False,
        "credentials_used": False,
        "automatic_retry_permitted": False,
    }
    report["result_sha256"] = _canonical_sha256(report)
    _durable_json_replace(slot_root / "result.json", report)
    _durable_json_replace(_binding_path(repository, ordinal), binding.as_dict())
    state.update(
        {
            "phase": "terminal",
            "outcome": "admitted",
            "run_id": binding.run_id,
            "binding_sha256": binding.binding_sha256,
            "result_sha256": report["result_sha256"],
        }
    )
    _durable_json_replace(state_path, state)
    return report, binding.as_dict()


def _complete_partition_if_ready(
    repository: Path,
    plan: Round74EventCohortPlan,
) -> None:
    bindings = _load_contiguous_bindings(
        repository,
        plan,
        before_ordinal=plan.total_slots,
    )
    partition = build_round74_event_run_partition(plan, bindings)
    _durable_json_replace(
        _campaign_root(repository) / "partition.json",
        partition.as_dict(),
    )


def run_round74_cohort_current_slot(repository: Path) -> int:
    """Capture and admit exactly the current immutable slot."""

    root = repository.resolve()
    plan = load_round74_cohort_operator_plan(root)
    now_wall_ns = time.time_ns()
    selection = select_round74_cohort_slot(plan, now_wall_ns=now_wall_ns)
    if selection.status != "open" or selection.slot_ordinal is None:
        raise RuntimeError(f"Round 74 cohort has no open slot: {selection.status}")
    ordinal = selection.slot_ordinal
    halt_path = _campaign_root(root) / "halt.json"
    if halt_path.exists():
        raise RuntimeError("Round 74 cohort campaign is already halted")
    binding_path = _binding_path(root, ordinal)
    reservation_path = _slot_root(root, ordinal) / "attempt-reservation.json"
    if binding_path.exists():
        binding = load_round74_event_cohort_binding(
            binding_path.read_text(encoding="utf-8")
        )
        if (
            not reservation_path.exists()
            or binding.plan_sha256 != plan.plan_sha256
            or binding.slot_ordinal != ordinal
        ):
            raise ValueError("Round 74 cohort completed slot identity differs")
        return 0
    if reservation_path.exists():
        _persist_halt(
            root,
            slot_ordinal=ordinal,
            reason="reserved slot has no admitted binding",
        )
        raise RuntimeError("Round 74 cohort slot was already attempted")
    try:
        validate_round74_active_prerequisite(root)
        _load_contiguous_bindings(root, plan, before_ordinal=ordinal)
        readiness = inspect_round74_cohort_readiness(
            root,
            now_wall_ns=now_wall_ns,
        )
        if readiness["ready_for_current_slot"] is not True:
            raise ValueError(f"Round 74 cohort slot readiness failed: {readiness}")
        report, _binding = _run_slot_process(root, plan, ordinal=ordinal)
        if ordinal + 1 == plan.total_slots:
            _complete_partition_if_ready(root, plan)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        slot_root = _slot_root(root, ordinal)
        if (slot_root / "attempt-reservation.json").exists():
            state_path = slot_root / "state.json"
            prior_state: dict[str, object] = {}
            if state_path.exists():
                try:
                    prior_state = _strict_json_object(
                        state_path.read_text(encoding="utf-8"),
                        "Round 74 cohort slot state",
                    )
                except (OSError, ValueError):
                    prior_state = {}
            launches = prior_state.get("pre_admission_startup_launches")
            startup_launch_sha256s = (
                [
                    str(item["evidence_sha256"])
                    for item in launches
                    if isinstance(item, Mapping)
                    and isinstance(item.get("evidence_sha256"), str)
                ]
                if isinstance(launches, list)
                else []
            )
            failure = {
                "schema_version": ROUND74_EVENT_COHORT_OPERATOR_RESULT_SCHEMA_VERSION,
                "plan_sha256": plan.plan_sha256,
                "slot_ordinal": ordinal,
                "outcome": "failed",
                "error": f"{type(exc).__name__}:{exc}"[:2_000],
                "startup_launch_evidence_sha256s": startup_launch_sha256s,
                "automatic_retry_permitted": False,
                "orders_submitted": False,
                "credentials_used": False,
            }
            failure["result_sha256"] = _canonical_sha256(failure)
            if not (slot_root / "result.json").exists():
                _durable_json_replace(slot_root / "result.json", failure)
            terminal_state = {
                **prior_state,
                "schema_version": ROUND74_EVENT_COHORT_OPERATOR_STATE_SCHEMA_VERSION,
                "plan_sha256": plan.plan_sha256,
                "slot_ordinal": ordinal,
                "role": plan.slot(ordinal).role,
                "phase": "terminal",
                "outcome": "failed",
                "error": failure["error"],
                "result_sha256": failure["result_sha256"],
                "automatic_retry_permitted": False,
            }
            _durable_json_replace(state_path, terminal_state)
        _persist_halt(
            root,
            slot_ordinal=ordinal,
            reason=f"{type(exc).__name__}:{exc}",
        )
        raise


__all__ = [
    "ROUND74_EVENT_COHORT_FRESH_AUDIT_TIMEOUT_SECONDS",
    "ROUND74_EVENT_COHORT_GLOBAL_DATABASE_CAP_BYTES",
    "ROUND74_EVENT_COHORT_OPERATOR_RESULT_SCHEMA_VERSION",
    "ROUND74_EVENT_COHORT_OPERATOR_STATE_SCHEMA_VERSION",
    "ROUND74_EVENT_COHORT_PLAN_RELATIVE_PATH",
    "ROUND74_EVENT_COHORT_PLAN_SHA256",
    "Round74CohortSlotSelection",
    "inspect_round74_cohort_readiness",
    "load_round74_cohort_operator_plan",
    "run_round74_cohort_current_slot",
    "select_round74_cohort_slot",
    "validate_round74_active_prerequisite",
]
