from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import simple_ai_trading.round75_capture_supervisor as supervisor_module
from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortPlan,
)
from simple_ai_trading.round75_capture_supervisor import (
    ROUND75_SERVICE_STALE_AFTER_NS,
    Round75CaptureSupervisorConfig,
    Round75ProcessRecord,
    _service_command,
    inspect_round75_capture_supervisor,
    supervise_round75_capture,
)
from simple_ai_trading.round75_continuous_capture import (
    ROUND75_CAMPAIGN_SIZE_CAP_BYTES,
    ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION,
    ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION,
    ROUND75_SHARD_PREFIX,
    ROUND75_SHARD_SIZE_CAP_BYTES,
    ROUND75_SHARD_SLOT_COUNT,
    Round75ContinuousCaptureConfig,
    _canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/simple_ai_trading/round75_continuous_capture.py"
PREREQUISITE = (
    ROOT / "docs/model-research/action-value/"
    "round-074-segmented-prerequisite-attempt-003-success-2026-07-28.json"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(
    tmp_path: Path,
) -> tuple[Round75CaptureSupervisorConfig, Round74SegmentedCohortPlan, str]:
    repository = tmp_path / "repository"
    tools = repository / "tools"
    tools.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "tools/run_round75_continuous_capture.py",
        tools / "run_round75_continuous_capture.py",
    )
    shutil.copyfile(
        ROOT / "tools/run_round74_segmented_capture.py",
        tools / "run_round74_segmented_capture.py",
    )
    prerequisite = repository / "docs/prerequisite.json"
    prerequisite.parent.mkdir(parents=True)
    shutil.copyfile(PREREQUISITE, prerequisite)
    prerequisite_payload = json.loads(prerequisite.read_text(encoding="utf-8"))
    plan = Round74SegmentedCohortPlan(
        scheduled_start_wall_ns=1_900_000_000_000_000_000,
        implementation_git_commit="0" * 40,
        prerequisite_artifact_sha256=prerequisite_payload["artifact_sha256"],
        prerequisite_window_start_wall_ns=prerequisite_payload["capture"][
            "started_wall_ns"
        ],
        prerequisite_window_end_wall_ns=prerequisite_payload["capture"][
            "ended_wall_ns"
        ],
    )
    plan_path = repository / "docs/plan.json"
    _write_json(plan_path, plan.as_dict())
    contract: dict[str, object] = {
        "schema_version": ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION,
        "implementation": {
            "continuous_service_path": (
                "src/simple_ai_trading/round75_continuous_capture.py"
            ),
            "continuous_service_file_sha256": hashlib.sha256(
                SOURCE.read_bytes()
            ).hexdigest(),
        },
        "resource_contract": {
            "shard_slot_count": ROUND75_SHARD_SLOT_COUNT,
            "shard_size_cap_bytes": ROUND75_SHARD_SIZE_CAP_BYTES,
            "campaign_size_cap_bytes": ROUND75_CAMPAIGN_SIZE_CAP_BYTES,
            "minimum_free_bytes": 100 * 1024 * 1024 * 1024,
            "slot_growth_cap_bytes": 512 * 1024 * 1024,
            "duckdb_memory_limit": "2GB",
            "duckdb_threads": 2,
        },
        "activated_plan": {
            "plan_path": "docs/plan.json",
            "plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "plan_sha256": plan.plan_sha256,
        },
    }
    contract["artifact_sha256"] = _canonical_sha256(contract)
    contract_path = repository / "docs/contract.json"
    _write_json(contract_path, contract)
    capture = Round75ContinuousCaptureConfig(
        repository=repository,
        contract_path=contract_path,
        plan_path=plan_path,
        prerequisite_path=prerequisite,
        data_root=repository / "data/campaign",
        state_root=repository / "data/state",
        service_state_path=repository / "data/service-state.json",
        lease_path=repository / "data/service.lock",
        stop_request_path=repository / "data/stop.request",
    )
    config = Round75CaptureSupervisorConfig(
        capture=capture,
        python_executable=Path(sys.executable),
        service_tool_path=tools / "run_round75_continuous_capture.py",
        capture_tool_path=tools / "run_round74_segmented_capture.py",
        stdout_log_path=repository / "data/service.stdout.log",
        stderr_log_path=repository / "data/service.stderr.log",
    )
    return config, plan, str(contract["artifact_sha256"])


def _service_process(
    config: Round75CaptureSupervisorConfig,
    *,
    process_id: int,
    creation_wall_ns: int,
) -> Round75ProcessRecord:
    return Round75ProcessRecord(
        process_id=process_id,
        parent_process_id=1,
        command_line=subprocess.list2cmdline(_service_command(config)),
        creation_wall_ns=creation_wall_ns,
    )


def _child_process(
    config: Round75CaptureSupervisorConfig,
    *,
    process_id: int,
    parent_process_id: int,
) -> Round75ProcessRecord:
    command = [
        str(config.python_executable),
        str(config.capture_tool_path),
        "--database",
        str(config.capture.data_root / f"{ROUND75_SHARD_PREFIX}000.duckdb"),
        "--json",
    ]
    return Round75ProcessRecord(
        process_id=process_id,
        parent_process_id=parent_process_id,
        command_line=subprocess.list2cmdline(command),
    )


def _write_state(
    config: Round75CaptureSupervisorConfig,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
    *,
    process_id: int,
    parent_process_id: int = 1,
    heartbeat_wall_ns: int,
) -> None:
    payload: dict[str, object] = {
        "schema_version": ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION,
        "service_schema_version": "round-075-continuous-capture-service-v1",
        "plan_sha256": plan.plan_sha256,
        "contract_sha256": contract_sha256,
        "service_instance_id": "a" * 32,
        "service_process_id": process_id,
        "service_parent_process_id": parent_process_id,
        "service_started_wall_ns": heartbeat_wall_ns - 1,
        "heartbeat_wall_ns": heartbeat_wall_ns,
        "phase": "capturing",
        "slot_ordinal": 0,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    payload["state_sha256"] = _canonical_sha256(payload)
    _write_json(config.capture.service_state_path, payload)


def _windows_venv_config(
    config: Round75CaptureSupervisorConfig,
) -> tuple[Round75CaptureSupervisorConfig, Path]:
    environment = config.capture.repository / ".venv"
    launcher = environment / "Scripts/python.exe"
    base = config.capture.repository / "base-python/python.exe"
    launcher.parent.mkdir(parents=True)
    base.parent.mkdir(parents=True)
    launcher.write_bytes(b"launcher")
    base.write_bytes(b"base")
    (environment / "pyvenv.cfg").write_text(
        f"home = {base.parent}\nversion_info = 3.12.10\n",
        encoding="utf-8",
    )
    return replace(config, python_executable=launcher), base


def _redirected_process(
    command: list[str],
    *,
    executable: Path,
    process_id: int,
    parent_process_id: int,
    creation_wall_ns: int,
) -> Round75ProcessRecord:
    return Round75ProcessRecord(
        process_id=process_id,
        parent_process_id=parent_process_id,
        command_line=subprocess.list2cmdline([str(executable), *command[1:]]),
        creation_wall_ns=creation_wall_ns,
    )


def test_capture_supervisor_accepts_one_healthy_exact_process_family(
    tmp_path: Path,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    now = 1_900_000_100_000_000_000
    service = _service_process(config, process_id=4000, creation_wall_ns=now - 10)
    child = _child_process(config, process_id=4001, parent_process_id=4000)
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4000,
        heartbeat_wall_ns=now - 1,
    )
    result = inspect_round75_capture_supervisor(
        config,
        processes=(service, child),
        now_wall_ns=now,
    )
    assert result["classification"] == "service_healthy"
    assert result["service_process_ids"] == [4000]
    assert result["capture_child_process_ids"] == [4001]
    assert result["automatic_repair_permitted"] is False


def test_capture_supervisor_collapses_exact_windows_venv_redirect_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    config, base = _windows_venv_config(config)
    monkeypatch.setattr(
        supervisor_module,
        "_WINDOWS_VENV_LAUNCHER_SEMANTICS",
        True,
    )
    now = 1_900_000_100_000_000_000
    command = _service_command(config)
    launcher = Round75ProcessRecord(
        process_id=4050,
        parent_process_id=1,
        command_line=subprocess.list2cmdline(command),
        creation_wall_ns=now - 10,
    )
    interpreter = _redirected_process(
        command,
        executable=base,
        process_id=4051,
        parent_process_id=4050,
        creation_wall_ns=now - 10,
    )
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4051,
        parent_process_id=4050,
        heartbeat_wall_ns=now - 1,
    )
    result = inspect_round75_capture_supervisor(
        config,
        processes=(launcher, interpreter),
        now_wall_ns=now,
    )
    assert result["classification"] == "service_healthy"
    assert result["service_process_ids"] == [4051]
    assert result["service_launcher_process_ids"] == [4050]
    assert result["physical_service_process_ids"] == [4050, 4051]


def test_capture_supervisor_repairs_redirected_family_in_physical_child_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    config, base = _windows_venv_config(config)
    monkeypatch.setattr(
        supervisor_module,
        "_WINDOWS_VENV_LAUNCHER_SEMANTICS",
        True,
    )
    now = 1_900_000_100_000_000_000
    service_command = _service_command(config)
    service_launcher = _redirected_process(
        service_command,
        executable=config.python_executable,
        process_id=4060,
        parent_process_id=1,
        creation_wall_ns=now - 10,
    )
    service = _redirected_process(
        service_command,
        executable=base,
        process_id=4061,
        parent_process_id=4060,
        creation_wall_ns=now - 10,
    )
    child_command = [
        str(config.python_executable),
        str(config.capture_tool_path),
        "--database",
        str(config.capture.data_root / f"{ROUND75_SHARD_PREFIX}000.duckdb"),
        "--json",
    ]
    child_launcher = _redirected_process(
        child_command,
        executable=config.python_executable,
        process_id=4062,
        parent_process_id=4061,
        creation_wall_ns=now - 10,
    )
    child = _redirected_process(
        child_command,
        executable=base,
        process_id=4063,
        parent_process_id=4062,
        creation_wall_ns=now - 10,
    )
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4061,
        parent_process_id=4060,
        heartbeat_wall_ns=now - ROUND75_SERVICE_STALE_AFTER_NS - 1,
    )
    current = [service_launcher, service, child_launcher, child]
    terminated: list[int] = []

    def terminate(process_id: int) -> None:
        terminated.append(process_id)
        current[:] = [row for row in current if row.process_id != process_id]

    result = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=tuple(current),
        now_wall_ns=now,
        terminate=terminate,
        start_service=lambda _config: 4099,
        refresh_inventory=lambda: tuple(current),
    )
    assert terminated == [4063, 4062, 4061, 4060]
    assert result["terminated_process_ids"] == terminated
    assert result["started_process_id"] == 4099


def test_capture_supervisor_repairs_stale_exact_family_child_first(
    tmp_path: Path,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    now = 1_900_000_100_000_000_000
    service = _service_process(config, process_id=4100, creation_wall_ns=now - 10)
    child = _child_process(config, process_id=4101, parent_process_id=4100)
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4100,
        heartbeat_wall_ns=now - ROUND75_SERVICE_STALE_AFTER_NS - 1,
    )
    current = [service, child]
    terminated: list[int] = []

    def terminate(process_id: int) -> None:
        terminated.append(process_id)
        current[:] = [row for row in current if row.process_id != process_id]

    result = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=(service, child),
        now_wall_ns=now,
        terminate=terminate,
        start_service=lambda _config: 4199,
        refresh_inventory=lambda: tuple(current),
    )
    assert terminated == [4101, 4100]
    assert result["terminated_process_ids"] == [4101, 4100]
    assert result["started_process_id"] == 4199
    assert result["action"] == "stale_owned_repaired_and_service_started"


def test_capture_supervisor_does_not_repair_stale_family_without_flag(
    tmp_path: Path,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    now = 1_900_000_100_000_000_000
    service = _service_process(config, process_id=4200, creation_wall_ns=now - 10)
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4200,
        heartbeat_wall_ns=now - ROUND75_SERVICE_STALE_AFTER_NS - 1,
    )
    result = supervise_round75_capture(
        config,
        repair_stale_owned=False,
        processes=(service,),
        now_wall_ns=now,
        terminate=lambda _process_id: (_ for _ in ()).throw(AssertionError()),
        start_service=lambda _config: (_ for _ in ()).throw(AssertionError()),
    )
    assert result["action"] == "repair_required"
    assert result["terminated_process_ids"] == []


def test_capture_supervisor_cancels_repair_when_heartbeat_recovers(
    tmp_path: Path,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    now = 1_900_000_100_000_000_000
    service = _service_process(config, process_id=4250, creation_wall_ns=now - 10)
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4250,
        heartbeat_wall_ns=now - ROUND75_SERVICE_STALE_AFTER_NS - 1,
    )

    def refresh() -> tuple[Round75ProcessRecord, ...]:
        _write_state(
            config,
            plan,
            contract_sha256,
            process_id=4250,
            heartbeat_wall_ns=now,
        )
        return (service,)

    result = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=(service,),
        now_wall_ns=now,
        terminate=lambda _process_id: (_ for _ in ()).throw(AssertionError()),
        start_service=lambda _config: (_ for _ in ()).throw(AssertionError()),
        refresh_inventory=refresh,
    )
    assert result["action"] == "repair_cancelled_state_changed"
    assert result["terminated_process_ids"] == []


def test_capture_supervisor_starts_missing_service_but_ignores_unrelated_python(
    tmp_path: Path,
) -> None:
    config, _plan, _contract_sha256 = _fixture(tmp_path)
    unrelated = Round75ProcessRecord(
        process_id=4300,
        parent_process_id=1,
        command_line=subprocess.list2cmdline(
            [str(config.python_executable), "unrelated.py", "--database", "other.db"]
        ),
    )
    result = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=(unrelated,),
        now_wall_ns=1_900_000_100_000_000_000,
        terminate=lambda _process_id: (_ for _ in ()).throw(AssertionError()),
        start_service=lambda _config: 4399,
    )
    assert result["classification"] == "service_missing"
    assert result["action"] == "service_started"
    assert result["started_process_id"] == 4399


def test_capture_supervisor_never_restarts_an_ended_campaign(tmp_path: Path) -> None:
    config, plan, _contract_sha256 = _fixture(tmp_path)
    campaign_boundary = (
        plan.slot(plan.total_slots - 1).scheduled_start_wall_ns
        + plan.slot(1).scheduled_start_wall_ns
        - plan.slot(0).scheduled_start_wall_ns
    )
    inspection = inspect_round75_capture_supervisor(
        config,
        processes=(),
        now_wall_ns=campaign_boundary,
    )
    assert inspection["classification"] == "campaign_terminal"
    assert inspection["campaign_ended"] is True
    assert inspection["campaign_boundary_wall_ns"] == campaign_boundary
    assert inspection["automatic_start_permitted"] is False

    result = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=(),
        now_wall_ns=campaign_boundary,
        terminate=lambda _process_id: (_ for _ in ()).throw(AssertionError()),
        start_service=lambda _config: (_ for _ in ()).throw(AssertionError()),
    )
    assert result["action"] == "none_campaign_terminal"
    assert result["started_process_id"] is None


def test_capture_supervisor_blocks_terminal_process_without_touching_it(
    tmp_path: Path,
) -> None:
    config, plan, _contract_sha256 = _fixture(tmp_path)
    campaign_boundary = (
        plan.slot(plan.total_slots - 1).scheduled_start_wall_ns
        + plan.slot(1).scheduled_start_wall_ns
        - plan.slot(0).scheduled_start_wall_ns
    )
    service = _service_process(
        config,
        process_id=4398,
        creation_wall_ns=campaign_boundary - 1,
    )
    result = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=(service,),
        now_wall_ns=campaign_boundary,
        terminate=lambda _process_id: (_ for _ in ()).throw(AssertionError()),
        start_service=lambda _config: (_ for _ in ()).throw(AssertionError()),
    )
    assert result["classification"] == ("campaign_terminal_process_present_fail_closed")
    assert result["action"] == "blocked_campaign_terminal_process_present"
    assert result["terminated_process_ids"] == []


def test_capture_supervisor_fails_closed_on_duplicate_or_tampered_state(
    tmp_path: Path,
) -> None:
    config, plan, contract_sha256 = _fixture(tmp_path)
    now = 1_900_000_100_000_000_000
    service1 = _service_process(config, process_id=4400, creation_wall_ns=now - 10)
    service2 = _service_process(config, process_id=4401, creation_wall_ns=now - 10)
    _write_state(
        config,
        plan,
        contract_sha256,
        process_id=4400,
        heartbeat_wall_ns=now - 1,
    )
    duplicate = supervise_round75_capture(
        config,
        repair_stale_owned=True,
        processes=(service1, service2),
        now_wall_ns=now,
        terminate=lambda _process_id: (_ for _ in ()).throw(AssertionError()),
        start_service=lambda _config: (_ for _ in ()).throw(AssertionError()),
    )
    assert duplicate["classification"] == "ambiguous_fail_closed"
    assert duplicate["action"] == "blocked_ambiguous"

    state = json.loads(config.capture.service_state_path.read_text(encoding="utf-8"))
    state["heartbeat_wall_ns"] = now
    _write_json(config.capture.service_state_path, state)
    tampered = inspect_round75_capture_supervisor(
        config,
        processes=(service1,),
        now_wall_ns=now,
    )
    assert tampered["classification"] == "ambiguous_fail_closed"
    assert tampered["state_error"].startswith("ValueError:")
