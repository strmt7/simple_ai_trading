"""One-attempt operator for the frozen Round 74 active qualification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from typing import TextIO

from .impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from .impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)


ROUND74_ACTIVE_PREFLIGHT_RELATIVE_PATH = Path(
    "docs/model-research/action-value/"
    "round-074-v10-active-regime-qualification-preflight-2026-07-27.json"
)
ROUND74_ACTIVE_PREFLIGHT_SHA256 = (
    "5d5afce7f2ab42fa6dd3f31ea3460997b620225df216fd2d0534b42c33059428"
)
ROUND74_ACTIVE_ATTEMPT_STATE_SCHEMA_VERSION = (
    "round-074-active-qualification-attempt-state-v1"
)
ROUND74_ACTIVE_RESULT_SCHEMA_VERSION = (
    "round-074-active-qualification-operator-result-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_CAPTURE_SOURCE_PATHS = (
    "src/simple_ai_trading/__main__.py",
    "src/simple_ai_trading/assets.py",
    "src/simple_ai_trading/cli.py",
    "src/simple_ai_trading/impact_absorption.py",
    "src/simple_ai_trading/impact_absorption_capture.py",
    "src/simple_ai_trading/impact_absorption_store.py",
    "src/simple_ai_trading/impact_capture_frame.py",
)
_EXPECTED_ARGUMENTS = (
    "-m",
    "simple_ai_trading",
    "impact-capture",
    "--database",
    "data/microstructure.duckdb",
    "--mode",
    "qualification",
    "--schema-version",
    "v10",
    "--duration-seconds",
    "3600",
    "--maximum-reconnects",
    "0",
    "--database-size-cap-bytes",
    "10507268096",
    "--progress-interval-seconds",
    "30",
    "--json",
)
_MONITOR_INTERVAL_SECONDS = 5.0
_HEARTBEAT_INTERVAL_SECONDS = 30.0
_FINALIZATION_ALLOWANCE_SECONDS = 600.0


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


def _strict_json_object(raw_text: str, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(
        raw_text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} has non-finite JSON constant {value!r}")
        ),
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be an object")
    return parsed


def _parse_utc(value: object, label: str) -> datetime:
    selected = str(value)
    try:
        parsed = datetime.fromisoformat(selected.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    normalized = parsed.astimezone(timezone.utc)
    if normalized.isoformat().replace("+00:00", "Z") != selected:
        raise ValueError(f"{label} must use canonical UTC form")
    return normalized


def _resolved_inside(root: Path, relative: str | Path, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    return candidate


@dataclass(frozen=True)
class Round74ActivePreflight:
    """Validated execution values extracted from the immutable preflight."""

    payload: Mapping[str, object]
    artifact_sha256: str
    implementation_git_commit: str
    executable: Path
    arguments: tuple[str, ...]
    database: Path
    earliest_start_utc: datetime
    latest_start_utc: datetime
    duration_seconds: int
    free_space_minimum_bytes: int
    database_growth_limit_bytes: int
    process_io_limit_bytes: int
    baseline_database_bytes: int
    baseline_wal_bytes: int
    configured_database_size_cap_bytes: int
    active_minimum_messages_per_second: float
    quiet_maximum_messages_per_second: float

    @property
    def state_directory_name(self) -> str:
        return self.artifact_sha256


def load_round74_active_preflight(
    repository: Path,
    path: Path | None = None,
) -> Round74ActivePreflight:
    """Load and validate every execution-affecting preflight field."""

    root = repository.resolve()
    selected_path = (
        _resolved_inside(root, ROUND74_ACTIVE_PREFLIGHT_RELATIVE_PATH, "preflight")
        if path is None
        else _resolved_inside(
            root,
            path.resolve().relative_to(root),
            "preflight",
        )
    )
    payload = _strict_json_object(
        selected_path.read_text(encoding="utf-8"),
        "Round 74 active preflight",
    )
    claimed = str(payload.get("artifact_sha256", ""))
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    if (
        claimed != ROUND74_ACTIVE_PREFLIGHT_SHA256
        or claimed != _canonical_sha256(canonical)
        or payload.get("artifact_schema_version")
        != "round-074-v10-active-regime-qualification-preflight-v3"
    ):
        raise ValueError("Round 74 active preflight identity differs")

    scope = _mapping(payload.get("scope"), "scope")
    window = _mapping(payload.get("fixed_execution_window"), "execution window")
    invocation = _mapping(payload.get("capture_invocation"), "capture invocation")
    resources = _mapping(payload.get("resource_contract"), "resource contract")
    classification = _mapping(
        payload.get("frozen_classification"),
        "classification contract",
    )
    authorization = _mapping(payload.get("authorization"), "authorization")
    correction = _mapping(
        payload.get("pre_attempt_correction"),
        "pre-attempt correction",
    )

    arguments = tuple(
        str(value)
        for value in _sequence(invocation.get("arguments"), "capture arguments")
    )
    executable_text = str(invocation.get("executable", ""))
    executable = _resolved_inside(root, executable_text, "capture executable")
    database = _resolved_inside(root, "data/microstructure.duckdb", "database")
    baseline_database = _integer(
        resources.get("baseline_database_bytes"),
        "baseline database bytes",
    )
    baseline_wal = _integer(
        resources.get("baseline_wal_bytes"),
        "baseline WAL bytes",
    )
    growth_limit = _integer(
        resources.get("database_and_wal_growth_limit_bytes_per_hour"),
        "database growth limit",
    )
    configured_cap = _integer(
        resources.get("configured_database_size_cap_bytes"),
        "configured database cap",
    )

    if (
        arguments != _EXPECTED_ARGUMENTS
        or payload.get("implementation_git_commit")
        != "a986a2e3d6c69abe9d6f497b432645545c6825e6"
        or scope.get("capture_schema_version") != IMPACT_CAPTURE_V10_SCHEMA_VERSION
        or scope.get("capture_contract_sha256") != IMPACT_CAPTURE_V10_CONTRACT_SHA256
        or scope.get("qualification_only") is not True
        or scope.get("order_submission_permitted") is not False
        or window.get("maximum_reconnects") != 0
        or window.get("automatic_retry_permitted") is not False
        or window.get("reschedule_after_observing_market_activity_permitted")
        is not False
        or invocation.get("stdout_and_stderr_must_be_persisted") is not True
        or invocation.get("operator_progress_check_interval_maximum_seconds") != 120
        or correction.get("prior_command_started") is not False
        or correction.get("prior_market_stream_observed") is not False
        or correction.get("correction_selected_from_market_outcome") is not False
        or resources.get("inherited_default_would_reject_before_stream_start")
        is not True
        or configured_cap
        != baseline_database + baseline_wal + growth_limit + 536_870_912
        or authorization.get(
            "one_v10_active_regime_qualification_attempt_in_fixed_window"
        )
        is not True
        or authorization.get("round_074_model_training_or_evaluation") is not False
        or authorization.get("paper_order_authority") is not False
        or authorization.get("testnet_order_authority") is not False
        or authorization.get("live_trading_authority") is not False
    ):
        raise ValueError("Round 74 active preflight static policy differs")

    return Round74ActivePreflight(
        payload=payload,
        artifact_sha256=claimed,
        implementation_git_commit=str(payload["implementation_git_commit"]),
        executable=executable,
        arguments=arguments,
        database=database,
        earliest_start_utc=_parse_utc(
            window.get("earliest_start_utc"),
            "earliest start",
        ),
        latest_start_utc=_parse_utc(
            window.get("latest_start_utc"),
            "latest start",
        ),
        duration_seconds=_integer(window.get("duration_seconds"), "duration"),
        free_space_minimum_bytes=_integer(
            resources.get("preflight_free_space_minimum_bytes"),
            "minimum free space",
        ),
        database_growth_limit_bytes=growth_limit,
        process_io_limit_bytes=_integer(
            resources.get("process_io_transfer_limit_bytes_per_hour"),
            "process I/O limit",
        ),
        baseline_database_bytes=baseline_database,
        baseline_wal_bytes=baseline_wal,
        configured_database_size_cap_bytes=configured_cap,
        active_minimum_messages_per_second=float(
            classification["active_minimum_messages_per_second"]
        ),
        quiet_maximum_messages_per_second=float(
            classification["quiet_maximum_messages_per_second"]
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 74 active {label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"Round 74 active {label} must be an array")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Round 74 active {label} must be a non-negative integer")
    return value


def round74_active_window_status(
    preflight: Round74ActivePreflight,
    now_utc: datetime,
) -> str:
    """Classify wall time without mutating attempt state."""

    if now_utc.tzinfo is None:
        raise ValueError("Round 74 active wall time must be timezone-aware")
    selected = now_utc.astimezone(timezone.utc)
    if selected < preflight.earliest_start_utc:
        return "before_window"
    if selected <= preflight.latest_start_utc:
        return "open"
    return "missed"


def classify_round74_activity(
    *,
    message_count: int,
    elapsed_seconds: float,
    active_minimum: float,
    quiet_maximum: float,
) -> tuple[str, float]:
    """Apply the frozen activity rule without changing capture validity."""

    if (
        isinstance(message_count, bool)
        or message_count < 0
        or not 0.0 < float(elapsed_seconds)
        or not 0.0 <= float(quiet_maximum) < float(active_minimum)
    ):
        raise ValueError("Round 74 active classification inputs differ")
    rate = int(message_count) / float(elapsed_seconds)
    if rate >= float(active_minimum):
        return "active", rate
    if rate <= float(quiet_maximum):
        return "quiet", rate
    return "middle", rate


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _database_and_wal_bytes(database: Path) -> tuple[int, int]:
    return _path_bytes(database), _path_bytes(Path(f"{database}.wal"))


def _git_capture_source_matches(
    repository: Path,
    implementation_commit: str,
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for relative_path in _CAPTURE_SOURCE_PATHS:
        result = subprocess.run(  # nosec B603
            [
                "git",
                "diff",
                "--quiet",
                implementation_commit,
                "--",
                relative_path,
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            mismatches.append(relative_path)
    return not mismatches, mismatches


def _active_capture_processes() -> list[dict[str, object]]:
    if os.name == "nt":
        command = (
            "$ErrorActionPreference='Stop';"
            "$items=Get-CimInstance Win32_Process | "
            "Where-Object {$_.CommandLine} | "
            "Select-Object ProcessId,CommandLine;"
            "@($items)|ConvertTo-Json -Compress"
        )
        result = subprocess.run(  # nosec B603
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
        raw = json.loads(result.stdout or "[]")
        rows = raw if isinstance(raw, list) else [raw]
        output: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            command_line = str(row.get("CommandLine", ""))
            normalized = command_line.lower().replace("_", "-")
            if "simple-ai-trading" in normalized and "impact-capture" in normalized:
                output.append(
                    {
                        "process_id": int(row.get("ProcessId", 0)),
                        "command_line": command_line,
                    }
                )
        return output

    output = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("process inventory provider is unavailable")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        normalized = command_line.decode("utf-8", errors="replace").lower()
        if "simple_ai_trading" in normalized and "impact-capture" in normalized:
            output.append(
                {
                    "process_id": int(entry.name),
                    "command_line": normalized,
                }
            )
    return output


def inspect_round74_active_readiness(
    repository: Path,
    *,
    now_utc: datetime | None = None,
    process_provider: Callable[[], list[dict[str, object]]] = (
        _active_capture_processes
    ),
) -> dict[str, object]:
    """Return an evidence-rich, non-mutating launch readiness report."""

    root = repository.resolve()
    preflight = load_round74_active_preflight(root)
    observed = (
        datetime.now(timezone.utc)
        if now_utc is None
        else now_utc.astimezone(timezone.utc)
    )
    state_root = (
        root
        / "data"
        / "round74-v10-active-qualification"
        / preflight.state_directory_name
    )
    reservation = state_root / "attempt-reservation.json"
    database_bytes, wal_bytes = _database_and_wal_bytes(preflight.database)
    free_bytes = shutil.disk_usage(preflight.database.parent).free
    source_matches, source_mismatches = _git_capture_source_matches(
        root,
        preflight.implementation_git_commit,
    )
    processes = process_provider()
    baseline_matches = (
        database_bytes == preflight.baseline_database_bytes
        and wal_bytes == preflight.baseline_wal_bytes
    )
    checks = {
        "preflight_identity_passed": True,
        "capture_executable_passed": preflight.executable.is_file(),
        "frozen_capture_source_passed": source_matches,
        "baseline_database_and_wal_passed": baseline_matches,
        "free_space_passed": free_bytes >= preflight.free_space_minimum_bytes,
        "no_existing_reservation_passed": not reservation.exists(),
        "no_active_capture_process_passed": not processes,
    }
    return {
        "schema_version": "round-074-active-qualification-readiness-v1",
        "observed_at_utc": observed.isoformat().replace("+00:00", "Z"),
        "preflight_sha256": preflight.artifact_sha256,
        "window_status": round74_active_window_status(preflight, observed),
        "ready_for_window": all(checks.values()),
        "can_start_now": (
            all(checks.values())
            and round74_active_window_status(preflight, observed) == "open"
        ),
        "checks": checks,
        "frozen_source_mismatches": source_mismatches,
        "database_bytes": database_bytes,
        "wal_bytes": wal_bytes,
        "baseline_database_bytes": preflight.baseline_database_bytes,
        "baseline_wal_bytes": preflight.baseline_wal_bytes,
        "free_bytes": free_bytes,
        "free_space_minimum_bytes": preflight.free_space_minimum_bytes,
        "active_capture_processes": processes,
        "reservation_path": str(reservation),
    }


def _durable_json_replace(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def reserve_round74_active_attempt(
    reservation_path: Path,
    payload: Mapping[str, object],
) -> None:
    """Create the permanent no-retry marker before process creation."""

    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(reservation_path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial marker intentionally remains and still forbids a retry.
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_reader(
    source: TextIO,
    destination: TextIO,
    sink: list[str],
    *,
    mirror: bool,
) -> None:
    try:
        for line in iter(source.readline, ""):
            sink.append(line)
            destination.write(line)
            destination.flush()
            if mirror:
                print(line, end="", file=sys.stderr, flush=True)
    finally:
        source.close()


def _stop_owned_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is not None:
        return "already_terminal"
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=30)
        return "interrupt"
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.terminate()
        process.wait(timeout=15)
        return "terminate"
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait(timeout=15)
        return "kill"


def _capture_run_ids(supervisor: Mapping[str, object]) -> list[str]:
    attempts = supervisor.get("attempts")
    if not isinstance(attempts, list):
        return []
    run_ids: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        run_id = str(attempt.get("run_id", ""))
        if _RUN_ID.fullmatch(run_id) is not None and run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids


def _run_fresh_audits(
    preflight: Round74ActivePreflight,
    repository: Path,
    state_root: Path,
    supervisor: Mapping[str, object],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for run_id in _capture_run_ids(supervisor):
        stdout_path = state_root / f"audit-{run_id}.stdout.json"
        stderr_path = state_root / f"audit-{run_id}.stderr.log"
        command = [
            str(preflight.executable),
            "-m",
            "simple_ai_trading",
            "impact-audit",
            "--database",
            str(preflight.database.relative_to(repository)),
            "--run-id",
            run_id,
            "--json",
        ]
        completed = subprocess.run(  # nosec B603
            command,
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
        stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
        audit: dict[str, object] | None = None
        parse_error = ""
        try:
            audit = _strict_json_object(
                completed.stdout,
                f"fresh audit {run_id}",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}:{exc}"
        results.append(
            {
                "run_id": run_id,
                "return_code": completed.returncode,
                "audit": audit,
                "parse_error": parse_error,
                "stdout_path": str(stdout_path.relative_to(repository)),
                "stdout_sha256": _sha256_file(stdout_path),
                "stderr_path": str(stderr_path.relative_to(repository)),
                "stderr_sha256": _sha256_file(stderr_path),
            }
        )
    return results


def _evaluate_operator_result(
    preflight: Round74ActivePreflight,
    *,
    capture_return_code: int,
    supervisor: Mapping[str, object] | None,
    supervisor_parse_error: str,
    audits: Sequence[Mapping[str, object]],
    watchdog_breaches: Sequence[str],
) -> dict[str, object]:
    errors: list[str] = []
    if supervisor_parse_error:
        errors.append(f"supervisor_parse:{supervisor_parse_error}")
    if supervisor is None:
        errors.append("supervisor_report_missing")
        attempts: list[object] = []
    else:
        raw_attempts = supervisor.get("attempts")
        attempts = raw_attempts if isinstance(raw_attempts, list) else []
        if (
            supervisor.get("schema_version") != "round-074-capture-supervisor-report-v1"
            or supervisor.get("capture_schema_version")
            != IMPACT_CAPTURE_V10_SCHEMA_VERSION
            or supervisor.get("capture_contract_sha256")
            != IMPACT_CAPTURE_V10_CONTRACT_SHA256
            or supervisor.get("design_sha256") != ROUND74_CAPTURE_DESIGN_SHA256
            or supervisor.get("attempt_count") != 1
            or supervisor.get("reconnect_count") != 0
            or supervisor.get("reconnect_delays_seconds") != []
            or supervisor.get("attempt_evidence_combined") is not False
            or len(attempts) > 1
        ):
            errors.append("supervisor_static_contract_mismatch")
    if watchdog_breaches:
        errors.extend(f"watchdog:{value}" for value in watchdog_breaches)

    audit_by_run = {
        str(value.get("run_id", "")): value
        for value in audits
        if isinstance(value, Mapping)
    }
    activity_label = "unavailable"
    messages_per_second: float | None = None
    report_sha256 = ""
    capture_data_passed = False
    if len(attempts) == 1 and isinstance(attempts[0], Mapping):
        attempt = attempts[0]
        run_id = str(attempt.get("run_id", ""))
        report_sha256 = _canonical_sha256(dict(attempt))
        audit_wrapper = audit_by_run.get(run_id)
        audit = (
            audit_wrapper.get("audit") if isinstance(audit_wrapper, Mapping) else None
        )
        if (
            not isinstance(audit, Mapping)
            or audit_wrapper.get("return_code") != 0
            or audit.get("passed") is not True
            or audit.get("errors") != []
            or audit.get("run_id") != run_id
            or audit.get("run_status") != attempt.get("status")
            or audit.get("capture_contract_sha256")
            != attempt.get("capture_contract_sha256")
            or audit.get("stored_report_schema_version")
            != IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
            or audit.get("stored_report_sha256") != report_sha256
            or audit.get("frame_count") != attempt.get("writer_frame_count")
            or audit.get("message_count") != attempt.get("writer_message_count")
            or audit.get("compressed_payload_bytes")
            != attempt.get("writer_compressed_payload_bytes")
            or _SHA256.fullmatch(str(audit.get("last_frame_sha256", ""))) is None
        ):
            errors.append("fresh_audit_or_report_identity_failed")
        try:
            activity_label, messages_per_second = classify_round74_activity(
                message_count=int(attempt["writer_message_count"]),
                elapsed_seconds=float(attempt["elapsed_seconds"]),
                active_minimum=preflight.active_minimum_messages_per_second,
                quiet_maximum=preflight.quiet_maximum_messages_per_second,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"activity_classification:{type(exc).__name__}:{exc}")
        capture_data_passed = (
            attempt.get("schema_version") == IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
            and attempt.get("status") == "completed"
            and attempt.get("capture_gate_passed") is True
            and attempt.get("qualification_passed") is True
            and attempt.get("data_qualification_passed") is True
            and attempt.get("resource_safety_passed") is True
            and attempt.get("resource_safety_errors") == []
            and attempt.get("audit_passed") is True
            and attempt.get("audit_errors") == []
            and attempt.get("process_io_delta_write_bytes") is not None
            and int(attempt["process_io_delta_write_bytes"])
            <= preflight.process_io_limit_bytes
        )
        if not capture_data_passed:
            errors.append("capture_data_or_resource_gate_failed")

    active_qualified = (
        capture_return_code == 0
        and not errors
        and capture_data_passed
        and activity_label == "active"
    )
    if active_qualified:
        outcome = "active_qualified"
    elif capture_data_passed and activity_label in {"quiet", "middle"}:
        outcome = f"valid_{activity_label}_not_active"
    else:
        outcome = "failed"
    return {
        "outcome": outcome,
        "active_qualified": active_qualified,
        "capture_data_passed": capture_data_passed,
        "activity_label": activity_label,
        "messages_per_second": messages_per_second,
        "capture_report_sha256": report_sha256,
        "errors": errors,
        "automatic_retry_permitted": False,
    }


def _run_round74_active_qualification(repository: Path) -> int:
    """Run the only permitted attempt and retain every observed outcome."""

    root = repository.resolve()
    preflight = load_round74_active_preflight(root)
    now = datetime.now(timezone.utc)
    readiness = inspect_round74_active_readiness(root, now_utc=now)
    if readiness["window_status"] != "open":
        raise RuntimeError(
            f"Round 74 active window is {readiness['window_status']}; "
            "no reservation was created"
        )
    if readiness["can_start_now"] is not True:
        raise RuntimeError(f"Round 74 active readiness failed: {readiness}")

    state_root = (
        root
        / "data"
        / "round74-v10-active-qualification"
        / preflight.state_directory_name
    )
    reservation_path = state_root / "attempt-reservation.json"
    state_path = state_root / "state.json"
    capture_stdout_path = state_root / "capture.stdout.json"
    capture_stderr_path = state_root / "capture.stderr.log"
    reservation = {
        "schema_version": ROUND74_ACTIVE_ATTEMPT_STATE_SCHEMA_VERSION,
        "preflight_sha256": preflight.artifact_sha256,
        "reserved_at_utc": now.isoformat().replace("+00:00", "Z"),
        "command": [str(preflight.executable), *preflight.arguments],
        "automatic_retry_permitted": False,
    }
    reserve_round74_active_attempt(reservation_path, reservation)
    state: dict[str, object] = {
        **reservation,
        "phase": "reserved",
        "readiness": readiness,
    }
    _durable_json_replace(state_path, state)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    command = [str(preflight.executable), *preflight.arguments]
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    monitor_samples: list[dict[str, object]] = []
    watchdog_breaches: list[str] = []
    stop_method = ""
    with (
        capture_stdout_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        ) as stdout_file,
        capture_stderr_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        ) as stderr_file,
    ):
        process = subprocess.Popen(  # nosec B603
            command,
            cwd=root,
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
            raise RuntimeError("capture process pipes were not created")
        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, stdout_file, stdout_lines),
            kwargs={"mirror": False},
            name="round74-capture-stdout",
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, stderr_file, stderr_lines),
            kwargs={"mirror": True},
            name="round74-capture-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()
        state.update(
            {
                "phase": "running",
                "process_id": process.pid,
                "started_at_utc": started_utc.isoformat().replace("+00:00", "Z"),
            }
        )
        _durable_json_replace(state_path, state)
        next_heartbeat = started_monotonic
        while process.poll() is None:
            elapsed = time.monotonic() - started_monotonic
            database_bytes, wal_bytes = _database_and_wal_bytes(preflight.database)
            growth = max(
                0,
                database_bytes
                + wal_bytes
                - preflight.baseline_database_bytes
                - preflight.baseline_wal_bytes,
            )
            free_bytes = shutil.disk_usage(preflight.database.parent).free
            if time.monotonic() >= next_heartbeat:
                sample = {
                    "elapsed_seconds": round(elapsed, 3),
                    "database_bytes": database_bytes,
                    "wal_bytes": wal_bytes,
                    "database_and_wal_growth_bytes": growth,
                    "free_bytes": free_bytes,
                }
                monitor_samples.append(sample)
                print(
                    "round74-active-progress: "
                    f"elapsed={elapsed:.1f}s database_bytes={database_bytes} "
                    f"wal_bytes={wal_bytes} growth_bytes={growth} "
                    f"free_bytes={free_bytes}",
                    file=sys.stderr,
                    flush=True,
                )
                next_heartbeat += _HEARTBEAT_INTERVAL_SECONDS
            if growth > preflight.database_growth_limit_bytes:
                watchdog_breaches.append("database_and_wal_growth_limit_exceeded")
            if free_bytes < preflight.free_space_minimum_bytes:
                watchdog_breaches.append("minimum_free_space_breached")
            if elapsed > preflight.duration_seconds + _FINALIZATION_ALLOWANCE_SECONDS:
                watchdog_breaches.append("capture_finalization_deadline_exceeded")
            if watchdog_breaches:
                stop_method = _stop_owned_process(process)
                break
            time.sleep(_MONITOR_INTERVAL_SECONDS)
        return_code = process.wait(timeout=60)
        stdout_thread.join(timeout=30)
        stderr_thread.join(timeout=30)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise RuntimeError("capture output drain did not terminate")

    supervisor: dict[str, object] | None = None
    supervisor_parse_error = ""
    try:
        supervisor = _strict_json_object(
            "".join(stdout_lines),
            "capture supervisor report",
        )
    except (json.JSONDecodeError, ValueError) as exc:
        supervisor_parse_error = f"{type(exc).__name__}:{exc}"
    audits = (
        []
        if supervisor is None
        else _run_fresh_audits(preflight, root, state_root, supervisor)
    )
    verdict = _evaluate_operator_result(
        preflight,
        capture_return_code=return_code,
        supervisor=supervisor,
        supervisor_parse_error=supervisor_parse_error,
        audits=audits,
        watchdog_breaches=watchdog_breaches,
    )
    result: dict[str, object] = {
        "schema_version": ROUND74_ACTIVE_RESULT_SCHEMA_VERSION,
        "preflight_sha256": preflight.artifact_sha256,
        "started_at_utc": started_utc.isoformat().replace("+00:00", "Z"),
        "ended_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_command": command,
        "capture_return_code": return_code,
        "capture_stdout_path": str(capture_stdout_path.relative_to(root)),
        "capture_stdout_sha256": _sha256_file(capture_stdout_path),
        "capture_stderr_path": str(capture_stderr_path.relative_to(root)),
        "capture_stderr_sha256": _sha256_file(capture_stderr_path),
        "supervisor_report": supervisor,
        "supervisor_parse_error": supervisor_parse_error,
        "fresh_process_audits": audits,
        "monitor_samples": monitor_samples,
        "watchdog_breaches": watchdog_breaches,
        "watchdog_stop_method": stop_method,
        "verdict": verdict,
        "orders_submitted": False,
        "credentials_used": False,
        "automatic_retry_permitted": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    result_path = state_root / "result.json"
    _durable_json_replace(result_path, result)
    state.update(
        {
            "phase": "terminal",
            "ended_at_utc": result["ended_at_utc"],
            "result_path": str(result_path.relative_to(root)),
            "result_sha256": result["result_sha256"],
            "outcome": verdict["outcome"],
        }
    )
    _durable_json_replace(state_path, state)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if verdict["active_qualified"] is True else 2


def _persist_unexpected_terminal_failure(
    repository: Path,
    error: BaseException,
) -> None:
    state_root = (
        repository
        / "data"
        / "round74-v10-active-qualification"
        / ROUND74_ACTIVE_PREFLIGHT_SHA256
    )
    reservation_path = state_root / "attempt-reservation.json"
    result_path = state_root / "result.json"
    if not reservation_path.exists() or result_path.exists():
        return
    ended = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result: dict[str, object] = {
        "schema_version": ROUND74_ACTIVE_RESULT_SCHEMA_VERSION,
        "preflight_sha256": ROUND74_ACTIVE_PREFLIGHT_SHA256,
        "ended_at_utc": ended,
        "capture_return_code": None,
        "supervisor_report": None,
        "fresh_process_audits": [],
        "watchdog_breaches": [],
        "verdict": {
            "outcome": "operator_exception",
            "active_qualified": False,
            "capture_data_passed": False,
            "activity_label": "unavailable",
            "messages_per_second": None,
            "capture_report_sha256": "",
            "errors": [f"{type(error).__name__}:{error}"[:2_000]],
            "automatic_retry_permitted": False,
        },
        "orders_submitted": False,
        "credentials_used": False,
        "automatic_retry_permitted": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    _durable_json_replace(result_path, result)
    state_path = state_root / "state.json"
    state: dict[str, object] = {
        "schema_version": ROUND74_ACTIVE_ATTEMPT_STATE_SCHEMA_VERSION,
        "preflight_sha256": ROUND74_ACTIVE_PREFLIGHT_SHA256,
    }
    if state_path.exists():
        try:
            state.update(
                _strict_json_object(
                    state_path.read_text(encoding="utf-8"),
                    "attempt state",
                )
            )
        except (OSError, ValueError):
            pass
    state.update(
        {
            "phase": "terminal",
            "ended_at_utc": ended,
            "result_path": str(result_path.relative_to(repository)),
            "result_sha256": result["result_sha256"],
            "outcome": "operator_exception",
        }
    )
    _durable_json_replace(state_path, state)


def run_round74_active_qualification(repository: Path) -> int:
    """Run once and make every post-reservation failure terminal."""

    root = repository.resolve()
    try:
        return _run_round74_active_qualification(root)
    except BaseException as exc:
        _persist_unexpected_terminal_failure(root, exc)
        raise


__all__ = [
    "ROUND74_ACTIVE_ATTEMPT_STATE_SCHEMA_VERSION",
    "ROUND74_ACTIVE_PREFLIGHT_RELATIVE_PATH",
    "ROUND74_ACTIVE_PREFLIGHT_SHA256",
    "ROUND74_ACTIVE_RESULT_SCHEMA_VERSION",
    "Round74ActivePreflight",
    "classify_round74_activity",
    "inspect_round74_active_readiness",
    "load_round74_active_preflight",
    "reserve_round74_active_attempt",
    "round74_active_window_status",
    "run_round74_active_qualification",
]
