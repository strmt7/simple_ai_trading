"""Exact-ownership supervisor for the Round 75 continuous capture service."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal

# Fixed argv is used after package startup removes unsafe executable search entries.
import subprocess  # nosec B404
import time
from typing import Callable, Mapping, Sequence

from .impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
    load_round74_segmented_cohort_plan,
)
from .round75_continuous_capture import (
    ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION,
    ROUND75_SHARD_PREFIX,
    Round75ContinuousCaptureConfig,
    _canonical_sha256,
    load_round75_continuous_capture_contract,
)


ROUND75_CAPTURE_SUPERVISOR_SCHEMA_VERSION = "round-075-capture-supervisor-v3"
ROUND75_SERVICE_STALE_AFTER_NS = 90 * 1_000_000_000
ROUND75_SERVICE_STARTUP_GRACE_NS = 90 * 1_000_000_000
_WINDOWS_VENV_LAUNCHER_SEMANTICS = os.name == "nt"
_WINDOWS_VENV_LAUNCHER_CREATION_SKEW_NS = 5 * 1_000_000_000


@dataclass(frozen=True)
class Round75ProcessRecord:
    process_id: int
    parent_process_id: int
    command_line: str
    creation_wall_ns: int | None = None

    def validate(self) -> None:
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id <= 0
            or isinstance(self.parent_process_id, bool)
            or not isinstance(self.parent_process_id, int)
            or self.parent_process_id < 0
            or not isinstance(self.command_line, str)
            or not self.command_line.strip()
            or (
                self.creation_wall_ns is not None
                and (
                    isinstance(self.creation_wall_ns, bool)
                    or not isinstance(self.creation_wall_ns, int)
                    or self.creation_wall_ns <= 0
                )
            )
        ):
            raise ValueError("Round 75 process inventory row differs")


@dataclass(frozen=True)
class Round75CaptureSupervisorConfig:
    capture: Round75ContinuousCaptureConfig
    python_executable: Path
    service_tool_path: Path
    capture_tool_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    stale_after_ns: int = ROUND75_SERVICE_STALE_AFTER_NS
    startup_grace_ns: int = ROUND75_SERVICE_STARTUP_GRACE_NS

    def validate(self) -> None:
        self.capture.validate()
        repository = self.capture.repository.resolve()
        paths = (
            self.python_executable,
            self.service_tool_path,
            self.capture_tool_path,
            self.stdout_log_path,
            self.stderr_log_path,
        )
        integer_fields = (
            (self.stale_after_ns, ROUND75_SERVICE_STALE_AFTER_NS),
            (self.startup_grace_ns, ROUND75_SERVICE_STARTUP_GRACE_NS),
        )
        if (
            not self.python_executable.is_file()
            or not self.service_tool_path.is_file()
            or not self.capture_tool_path.is_file()
            or self.service_tool_path.resolve()
            != (repository / "tools/run_round75_continuous_capture.py").resolve()
            or self.capture_tool_path.resolve()
            != (repository / "tools/run_round74_segmented_capture.py").resolve()
            or any(path.is_symlink() for path in paths)
            or any(
                path != self.python_executable
                and not (
                    path.resolve() == repository or repository in path.resolve().parents
                )
                for path in paths
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
                for value, expected in integer_fields
            )
        ):
            raise ValueError("Round 75 capture supervisor config differs")


def _windows_argv(command_line: str) -> tuple[str, ...]:
    argc = ctypes.c_int()
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.CommandLineToArgvW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    )
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel32.LocalFree.restype = ctypes.c_void_p
    argv_pointer = shell32.CommandLineToArgvW(command_line, ctypes.byref(argc))
    if not argv_pointer:
        raise OSError("CommandLineToArgvW failed")
    try:
        return tuple(argv_pointer[index] for index in range(argc.value))
    finally:
        kernel32.LocalFree(ctypes.cast(argv_pointer, ctypes.c_void_p))


def _command_argv(command_line: str) -> tuple[str, ...]:
    if os.name == "nt":
        return _windows_argv(command_line)
    return tuple(shlex.split(command_line, posix=True))


def _argument_value(arguments: Sequence[str], name: str) -> str | None:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        return None
    return arguments[positions[0] + 1]


def _path_argument_matches(arguments: Sequence[str], expected: Path) -> bool:
    expected_resolved = expected.resolve()
    for argument in arguments:
        try:
            if Path(argument).resolve() == expected_resolved:
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def _resolved_argument_path(argument: str) -> Path | None:
    try:
        return Path(argument).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _windows_venv_base_executable(
    config: Round75CaptureSupervisorConfig,
) -> Path | None:
    if not _WINDOWS_VENV_LAUNCHER_SEMANTICS:
        return None
    environment = config.python_executable.parent.parent
    configuration = environment / "pyvenv.cfg"
    if (
        config.python_executable.parent.name.casefold() != "scripts"
        or configuration.is_symlink()
        or not configuration.is_file()
    ):
        return None
    values: dict[str, str] = {}
    try:
        for raw_line in configuration.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator or not key.strip() or not value.strip():
                return None
            values[key.strip().casefold()] = value.strip()
    except (OSError, UnicodeError):
        return None
    configured = values.get("executable")
    home = values.get("home")
    if configured:
        candidate = Path(configured)
    elif home:
        candidate = Path(home) / config.python_executable.name
    else:
        return None
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        return None
    resolved = candidate.resolve()
    if resolved == config.python_executable.resolve():
        return None
    return resolved


def _collapse_windows_venv_launcher_pairs(
    config: Round75CaptureSupervisorConfig,
    processes: Sequence[Round75ProcessRecord],
) -> tuple[tuple[Round75ProcessRecord, ...], tuple[Round75ProcessRecord, ...]]:
    """Return logical interpreters and exact redirect launchers.

    CPython's Windows venv redirector remains as a parent process for the real
    base interpreter. Both expose the same script arguments. Collapse only the
    standard pair proven by pyvenv.cfg, process ancestry, argument equality,
    and near-identical creation time; every other duplicate remains ambiguous.
    """

    physical = tuple(processes)
    base_executable = _windows_venv_base_executable(config)
    if base_executable is None or len(physical) < 2:
        return physical, ()
    parsed: dict[int, tuple[str, ...]] = {}
    for process in physical:
        try:
            arguments = _command_argv(process.command_line)
        except (OSError, ValueError):
            continue
        if arguments:
            parsed[process.process_id] = arguments
    launchers: list[Round75ProcessRecord] = []
    for parent in physical:
        parent_arguments = parsed.get(parent.process_id)
        if (
            not parent_arguments
            or _resolved_argument_path(parent_arguments[0])
            != config.python_executable.resolve()
        ):
            continue
        candidates: list[Round75ProcessRecord] = []
        for child in physical:
            child_arguments = parsed.get(child.process_id)
            if (
                child.parent_process_id != parent.process_id
                or not child_arguments
                or _resolved_argument_path(child_arguments[0]) != base_executable
                or child_arguments[1:] != parent_arguments[1:]
                or parent.creation_wall_ns is None
                or child.creation_wall_ns is None
                or abs(child.creation_wall_ns - parent.creation_wall_ns)
                > _WINDOWS_VENV_LAUNCHER_CREATION_SKEW_NS
            ):
                continue
            candidates.append(child)
        if len(candidates) == 1:
            launchers.append(parent)
    launcher_ids = {process.process_id for process in launchers}
    logical = tuple(
        process for process in physical if process.process_id not in launcher_ids
    )
    return logical, tuple(launchers)


def _effective_parent_process_id(
    process: Round75ProcessRecord,
    launchers: Sequence[Round75ProcessRecord],
) -> int:
    launcher = next(
        (
            candidate
            for candidate in launchers
            if candidate.process_id == process.parent_process_id
        ),
        None,
    )
    return (
        launcher.parent_process_id
        if launcher is not None
        else process.parent_process_id
    )


def _service_process_matches(
    config: Round75CaptureSupervisorConfig,
    process: Round75ProcessRecord,
) -> bool:
    process.validate()
    try:
        arguments = _command_argv(process.command_line)
    except (OSError, ValueError):
        return False
    contract = _argument_value(arguments, "--contract")
    plan = _argument_value(arguments, "--plan")
    repository = _argument_value(arguments, "--repository")
    if contract is None or plan is None or repository is None:
        return False
    return (
        _path_argument_matches(arguments, config.service_tool_path)
        and Path(contract).resolve() == config.capture.contract_path.resolve()
        and Path(plan).resolve() == config.capture.plan_path.resolve()
        and Path(repository).resolve() == config.capture.repository.resolve()
    )


def _capture_child_matches(
    config: Round75CaptureSupervisorConfig,
    process: Round75ProcessRecord,
) -> bool:
    process.validate()
    try:
        arguments = _command_argv(process.command_line)
    except (OSError, ValueError):
        return False
    database = _argument_value(arguments, "--database")
    if database is None or not _path_argument_matches(
        arguments,
        config.capture_tool_path,
    ):
        return False
    selected = Path(database).resolve()
    return (
        selected.parent == config.capture.data_root.resolve()
        and selected.name.startswith(ROUND75_SHARD_PREFIX)
        and selected.name.endswith(".duckdb")
        and selected.name[len(ROUND75_SHARD_PREFIX) : -len(".duckdb")].isdigit()
    )


def inventory_round75_processes() -> tuple[Round75ProcessRecord, ...]:
    if os.name == "nt":
        command = (
            "$rows=Get-CimInstance Win32_Process | Where-Object "
            "{$_.Name -match '^(python|pythonw)'} | ForEach-Object {"
            "$created=0; try {$created=([DateTimeOffset]$_.CreationDate)."
            "ToUnixTimeMilliseconds()*1000000} catch {};"
            "[pscustomobject]@{ProcessId=$_.ProcessId;"
            "ParentProcessId=$_.ParentProcessId;CommandLine=$_.CommandLine;"
            "CreationWallNs=$created}};@($rows)|ConvertTo-Json -Compress"
        )
        completed = subprocess.run(  # nosec B603, B607
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
        decoded = json.loads(completed.stdout or "[]")
        rows = decoded if isinstance(decoded, list) else [decoded]
        result: list[Round75ProcessRecord] = []
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("CommandLine"):
                continue
            creation = int(row.get("CreationWallNs", 0))
            record = Round75ProcessRecord(
                process_id=int(row["ProcessId"]),
                parent_process_id=int(row.get("ParentProcessId", 0)),
                command_line=str(row["CommandLine"]),
                creation_wall_ns=creation if creation > 0 else None,
            )
            record.validate()
            result.append(record)
        return tuple(result)
    result = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Round 75 process inventory is unavailable")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            stat = (entry / "stat").read_text(encoding="ascii").split()
        except OSError:
            continue
        decoded = command_line.decode("utf-8", errors="replace").strip()
        if not decoded:
            continue
        result.append(
            Round75ProcessRecord(
                process_id=int(entry.name),
                parent_process_id=int(stat[3]),
                command_line=decoded,
            )
        )
    return tuple(result)


def _load_service_state(
    config: Round75CaptureSupervisorConfig,
    *,
    plan_sha256: str,
    contract_sha256: str,
) -> dict[str, object] | None:
    path = config.capture.service_state_path
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Round 75 service state path differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Round 75 service state root differs")
    canonical = dict(value)
    claimed = str(canonical.pop("state_sha256", ""))
    if (
        claimed != _canonical_sha256(canonical)
        or value.get("schema_version")
        != ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION
        or value.get("plan_sha256") != plan_sha256
        or value.get("contract_sha256") != contract_sha256
        or value.get("credentials_used") is not False
        or value.get("orders_submitted") is not False
        or value.get("trading_authority") is not False
    ):
        raise ValueError("Round 75 service state differs")
    return value


def inspect_round75_capture_supervisor(
    config: Round75CaptureSupervisorConfig,
    *,
    processes: Sequence[Round75ProcessRecord] | None = None,
    now_wall_ns: int | None = None,
) -> dict[str, object]:
    config.validate()
    plan = load_round74_segmented_cohort_plan(
        config.capture.plan_path.read_text(encoding="utf-8")
    )
    _contract, contract_sha256 = load_round75_continuous_capture_contract(
        config.capture.contract_path
    )
    observed_wall_ns = time.time_ns() if now_wall_ns is None else int(now_wall_ns)
    campaign_boundary_wall_ns = (
        plan.scheduled_start_wall_ns
        + plan.total_slots * ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS
    )
    campaign_ended = observed_wall_ns >= campaign_boundary_wall_ns
    inventory = inventory_round75_processes() if processes is None else tuple(processes)
    physical_services = tuple(
        process for process in inventory if _service_process_matches(config, process)
    )
    physical_children = tuple(
        process for process in inventory if _capture_child_matches(config, process)
    )
    services, service_launchers = _collapse_windows_venv_launcher_pairs(
        config,
        physical_services,
    )
    children, child_launchers = _collapse_windows_venv_launcher_pairs(
        config,
        physical_children,
    )
    state_error = ""
    try:
        state = _load_service_state(
            config,
            plan_sha256=plan.plan_sha256,
            contract_sha256=contract_sha256,
        )
    except (OSError, TypeError, ValueError) as exc:
        state = None
        state_error = f"{type(exc).__name__}: {exc}"

    classification = "ambiguous_fail_closed"
    repairable_process_ids: list[int] = []
    if len(services) > 1 or len(children) > 1 or state_error:
        pass
    elif campaign_ended:
        classification = (
            "campaign_terminal_process_present_fail_closed"
            if services or children
            else "campaign_terminal"
        )
    elif not services:
        if children:
            classification = "stale_owned_orphan_child"
            repairable_process_ids = [children[0].process_id]
        else:
            classification = "service_missing"
    else:
        service = services[0]
        if any(
            _effective_parent_process_id(child, child_launchers) != service.process_id
            for child in children
        ):
            pass
        elif state is None:
            creation = service.creation_wall_ns
            if (
                creation is not None
                and observed_wall_ns - creation <= config.startup_grace_ns
            ):
                classification = "service_starting"
            else:
                classification = "ambiguous_fail_closed"
        elif (
            state.get("service_process_id") != service.process_id
            or state.get("service_parent_process_id") != service.parent_process_id
        ):
            pass
        else:
            heartbeat = state.get("heartbeat_wall_ns")
            if isinstance(heartbeat, bool) or not isinstance(heartbeat, int):
                pass
            elif observed_wall_ns - heartbeat <= config.stale_after_ns:
                classification = "service_healthy"
            else:
                classification = "stale_owned_service"
                repairable_process_ids = [
                    *(child.process_id for child in children),
                    *(launcher.process_id for launcher in child_launchers),
                    service.process_id,
                    *(launcher.process_id for launcher in service_launchers),
                ]
    payload: dict[str, object] = {
        "schema_version": ROUND75_CAPTURE_SUPERVISOR_SCHEMA_VERSION,
        "observed_wall_ns": observed_wall_ns,
        "campaign_boundary_wall_ns": campaign_boundary_wall_ns,
        "campaign_ended": campaign_ended,
        "plan_sha256": plan.plan_sha256,
        "contract_sha256": contract_sha256,
        "classification": classification,
        "service_process_ids": [process.process_id for process in services],
        "service_launcher_process_ids": [
            process.process_id for process in service_launchers
        ],
        "physical_service_process_ids": [
            process.process_id for process in physical_services
        ],
        "capture_child_process_ids": [process.process_id for process in children],
        "capture_child_launcher_process_ids": [
            process.process_id for process in child_launchers
        ],
        "physical_capture_child_process_ids": [
            process.process_id for process in physical_children
        ],
        "repairable_process_ids_in_child_first_order": repairable_process_ids,
        "state_error": state_error,
        "automatic_start_permitted": classification
        in {"service_missing", "stale_owned_orphan_child", "stale_owned_service"},
        "automatic_repair_permitted": classification
        in {"stale_owned_orphan_child", "stale_owned_service"},
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    payload["inspection_sha256"] = _canonical_sha256(payload)
    return payload


def _service_command(config: Round75CaptureSupervisorConfig) -> list[str]:
    capture = config.capture
    return [
        str(config.python_executable),
        str(config.service_tool_path),
        "--repository",
        str(capture.repository),
        "--contract",
        str(capture.contract_path),
        "--plan",
        str(capture.plan_path),
        "--prerequisite",
        str(capture.prerequisite_path),
        "--data-root",
        str(capture.data_root),
        "--state-root",
        str(capture.state_root),
        "--service-state",
        str(capture.service_state_path),
        "--lease",
        str(capture.lease_path),
        "--stop-request",
        str(capture.stop_request_path),
    ]


def start_round75_capture_service(config: Round75CaptureSupervisorConfig) -> int:
    config.validate()
    config.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    with (
        config.stdout_log_path.open("a", encoding="utf-8", newline="\n") as stdout,
        config.stderr_log_path.open("a", encoding="utf-8", newline="\n") as stderr,
    ):
        process = subprocess.Popen(  # nosec B603
            _service_command(config),
            cwd=config.capture.repository,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )
    return int(process.pid)


def terminate_round75_owned_process(process_id: int) -> None:
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError("Round 75 termination process id differs")
    os.kill(process_id, signal.SIGTERM)


def supervise_round75_capture(
    config: Round75CaptureSupervisorConfig,
    *,
    repair_stale_owned: bool,
    processes: Sequence[Round75ProcessRecord] | None = None,
    now_wall_ns: int | None = None,
    terminate: Callable[[int], None] = terminate_round75_owned_process,
    start_service: Callable[[Round75CaptureSupervisorConfig], int] = (
        start_round75_capture_service
    ),
    refresh_inventory: Callable[[], Sequence[Round75ProcessRecord]] = (
        inventory_round75_processes
    ),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    inspection = inspect_round75_capture_supervisor(
        config,
        processes=processes,
        now_wall_ns=now_wall_ns,
    )
    classification = str(inspection["classification"])
    terminated: list[int] = []
    started_process_id: int | None = None
    if classification in {"service_healthy", "service_starting"}:
        action = "none"
    elif classification == "campaign_terminal":
        action = "none_campaign_terminal"
    elif classification == "campaign_terminal_process_present_fail_closed":
        action = "blocked_campaign_terminal_process_present"
    elif classification == "ambiguous_fail_closed":
        action = "blocked_ambiguous"
    elif classification == "service_missing":
        started_process_id = start_service(config)
        action = "service_started"
    elif not repair_stale_owned:
        action = "repair_required"
    else:
        refreshed_before_repair = tuple(refresh_inventory())
        reinspection = inspect_round75_capture_supervisor(
            config,
            processes=refreshed_before_repair,
            now_wall_ns=now_wall_ns,
        )
        if reinspection["classification"] not in {
            "stale_owned_orphan_child",
            "stale_owned_service",
        }:
            action = "repair_cancelled_state_changed"
            result = {
                "schema_version": ROUND75_CAPTURE_SUPERVISOR_SCHEMA_VERSION,
                "inspection_sha256": inspection["inspection_sha256"],
                "reinspection_sha256": reinspection["inspection_sha256"],
                "classification": classification,
                "action": action,
                "terminated_process_ids": terminated,
                "started_process_id": started_process_id,
                "repair_stale_owned_requested": repair_stale_owned,
                "credentials_used": False,
                "orders_submitted": False,
                "trading_authority": False,
            }
            result["result_sha256"] = _canonical_sha256(result)
            return result
        expected = [
            int(value)
            for value in reinspection["repairable_process_ids_in_child_first_order"]
        ]
        for process_id in expected:
            refreshed = tuple(refresh_inventory())
            current = next(
                (row for row in refreshed if row.process_id == process_id),
                None,
            )
            if current is None:
                continue
            if not (
                _service_process_matches(config, current)
                or _capture_child_matches(config, current)
            ):
                raise RuntimeError("Round 75 process identity changed before repair")
            terminate(process_id)
            terminated.append(process_id)
            deadline = time.monotonic() + 15.0
            while any(row.process_id == process_id for row in refresh_inventory()):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Round 75 owned process did not terminate before deadline"
                    )
                sleep(0.1)
        remaining = tuple(refresh_inventory())
        if any(
            _service_process_matches(config, process)
            or _capture_child_matches(config, process)
            for process in remaining
        ):
            raise RuntimeError("Round 75 owned process remained after repair")
        started_process_id = start_service(config)
        action = "stale_owned_repaired_and_service_started"
    result: dict[str, object] = {
        "schema_version": ROUND75_CAPTURE_SUPERVISOR_SCHEMA_VERSION,
        "inspection_sha256": inspection["inspection_sha256"],
        "classification": classification,
        "action": action,
        "terminated_process_ids": terminated,
        "started_process_id": started_process_id,
        "repair_stale_owned_requested": repair_stale_owned,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "ROUND75_CAPTURE_SUPERVISOR_SCHEMA_VERSION",
    "ROUND75_SERVICE_STALE_AFTER_NS",
    "ROUND75_SERVICE_STARTUP_GRACE_NS",
    "Round75CaptureSupervisorConfig",
    "Round75ProcessRecord",
    "inspect_round75_capture_supervisor",
    "inventory_round75_processes",
    "start_round75_capture_service",
    "supervise_round75_capture",
    "terminate_round75_owned_process",
]
