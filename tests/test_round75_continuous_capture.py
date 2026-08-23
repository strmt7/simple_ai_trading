from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import pytest

from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortPlan,
)
from simple_ai_trading.round75_continuous_capture import (
    ROUND75_CAMPAIGN_SIZE_CAP_BYTES,
    ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION,
    ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION,
    ROUND75_SHARD_PREFIX,
    ROUND75_SHARD_SIZE_CAP_BYTES,
    ROUND75_SHARD_SLOT_COUNT,
    Round75ContinuousCaptureConfig,
    Round75ExclusiveLease,
    _canonical_sha256,
    inspect_round75_storage,
    load_round75_continuous_capture_contract,
    run_round75_continuous_capture,
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


def _contract(path: Path, *, plan_path: Path, plan_sha256: str) -> str:
    payload: dict[str, object] = {
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
            "plan_path": str(plan_path.relative_to(path.parents[1])).replace("\\", "/"),
            "plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "plan_sha256": plan_sha256,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    _write_json(path, payload)
    return str(payload["artifact_sha256"])


def _fixture(
    tmp_path: Path,
    *,
    scheduled_start_wall_ns: int,
) -> tuple[Round75ContinuousCaptureConfig, Round74SegmentedCohortPlan, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    prerequisite = repository / "docs/prerequisite.json"
    prerequisite.parent.mkdir(parents=True)
    shutil.copyfile(PREREQUISITE, prerequisite)
    prerequisite_payload = json.loads(prerequisite.read_text(encoding="utf-8"))
    plan = Round74SegmentedCohortPlan(
        scheduled_start_wall_ns=scheduled_start_wall_ns,
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
    contract_path = repository / "docs/contract.json"
    contract_sha256 = _contract(
        contract_path,
        plan_path=plan_path,
        plan_sha256=plan.plan_sha256,
    )
    config = Round75ContinuousCaptureConfig(
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
    return config, plan, contract_sha256


def test_continuous_capture_waits_without_shifting_a_future_plan(
    tmp_path: Path,
) -> None:
    start = 1_900_000_000_000_000_000
    config, plan, contract_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    state = run_round75_continuous_capture(
        config,
        once=True,
        clock_ns=lambda: start - 1,
        sleep=lambda _seconds: None,
    )
    assert state["phase"] == "waiting_for_campaign_start"
    assert state["slot_ordinal"] is None
    assert state["plan_sha256"] == plan.plan_sha256
    assert state["contract_sha256"] == contract_sha256
    assert state["orders_submitted"] is False
    assert state["trading_authority"] is False
    claimed = state.pop("state_sha256")
    assert claimed == _canonical_sha256(state)


def test_continuous_capture_terminalizes_every_elapsed_unstarted_slot_once(
    tmp_path: Path,
) -> None:
    start = 1_900_000_000_000_000_000
    config, plan, _contract_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    now = plan.slot(2).start_window_end_wall_ns + 1
    state = run_round75_continuous_capture(
        config,
        once=True,
        clock_ns=lambda: now,
        sleep=lambda _seconds: None,
    )
    assert state["phase"] == "waiting_for_next_fixed_slot"
    receipts = sorted((config.state_root / "missed-slots").glob("*.json"))
    assert [path.name for path in receipts] == ["000.json", "001.json", "002.json"]
    values = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
    assert all(
        value["schema_version"] == ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION
        for value in values
    )
    assert all(value["automatic_retry_permitted"] is False for value in values)
    assert all(value["exact_host_cause_established"] is False for value in values)

    second = run_round75_continuous_capture(
        config,
        once=True,
        clock_ns=lambda: now,
        sleep=lambda _seconds: None,
    )
    assert second["detail"]["missed_receipts_created"] == 0
    assert len(list((config.state_root / "missed-slots").glob("*.json"))) == 3


def test_continuous_capture_runs_one_open_slot_on_its_deterministic_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = 1_900_000_000_000_000_000
    config, _plan, _contract_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    host_usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        "simple_ai_trading.round75_continuous_capture.shutil.disk_usage",
        lambda _path: host_usage._replace(free=config.minimum_free_bytes),
    )
    observed: list[Path] = []

    def fake_run_slot(runner_config) -> dict[str, object]:
        observed.append(runner_config.database_path)
        return {"result_sha256": "a" * 64}

    state = run_round75_continuous_capture(
        config,
        once=True,
        clock_ns=lambda: start,
        sleep=lambda _seconds: time.sleep(0.001),
        run_slot=fake_run_slot,
    )
    assert state["phase"] == "slot_terminal"
    assert state["slot_ordinal"] == 0
    assert observed == [config.data_root / f"{ROUND75_SHARD_PREFIX}000.duckdb"]
    assert state["detail"]["result_sha256"] == "a" * 64
    assert state["detail"]["storage_before_slot"]["database_opened"] is False


def test_continuous_capture_stop_request_prevents_slot_start(tmp_path: Path) -> None:
    start = 1_900_000_000_000_000_000
    config, _plan, _contract_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    config.stop_request_path.parent.mkdir(parents=True, exist_ok=True)
    config.stop_request_path.write_text("stop\n", encoding="ascii")

    def forbidden_runner(_config) -> dict[str, object]:
        raise AssertionError("stop request must prevent a new capture")

    state = run_round75_continuous_capture(
        config,
        once=True,
        clock_ns=lambda: start,
        sleep=lambda _seconds: None,
        run_slot=forbidden_runner,
    )
    assert state["phase"] == "stopped"
    assert state["orders_submitted"] is False


def test_continuous_capture_storage_inspection_fails_closed_on_any_wal(
    tmp_path: Path,
) -> None:
    start = 1_900_000_000_000_000_000
    config, _plan, _contract_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    config.data_root.mkdir(parents=True)
    shard = config.shard_path(0)
    Path(f"{shard}.wal").write_bytes(b"pending")
    evidence = inspect_round75_storage(config, slot_ordinal=0)
    assert evidence["passed"] is False
    assert evidence["checks"]["all_campaign_wals_absent"] is False
    assert evidence["database_opened"] is False


def test_continuous_capture_contract_and_lease_reject_tampering(
    tmp_path: Path,
) -> None:
    start = 1_900_000_000_000_000_000
    config, _plan, expected_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    _value, claimed = load_round75_continuous_capture_contract(config.contract_path)
    assert claimed == expected_sha256
    with Round75ExclusiveLease(config.lease_path):
        with pytest.raises(RuntimeError, match="lease is held"):
            with Round75ExclusiveLease(config.lease_path):
                pass

    payload = json.loads(config.contract_path.read_text(encoding="utf-8"))
    payload["resource_contract"]["duckdb_threads"] = 3
    _write_json(config.contract_path, payload)
    with pytest.raises(ValueError, match="contract differs"):
        load_round75_continuous_capture_contract(config.contract_path)


def test_continuous_capture_rejects_a_substituted_valid_plan(tmp_path: Path) -> None:
    start = 1_900_000_000_000_000_000
    config, plan, _contract_sha256 = _fixture(
        tmp_path,
        scheduled_start_wall_ns=start,
    )
    substituted = Round74SegmentedCohortPlan(
        scheduled_start_wall_ns=start + 300_000_000_000,
        implementation_git_commit=plan.implementation_git_commit,
        prerequisite_artifact_sha256=plan.prerequisite_artifact_sha256,
        prerequisite_window_start_wall_ns=plan.prerequisite_window_start_wall_ns,
        prerequisite_window_end_wall_ns=plan.prerequisite_window_end_wall_ns,
    )
    _write_json(config.plan_path, substituted.as_dict())
    with pytest.raises(ValueError, match="activated plan differs"):
        run_round75_continuous_capture(
            config,
            once=True,
            clock_ns=lambda: start,
            sleep=lambda _seconds: None,
        )
