"""Leased continuous scheduler for prospective Round 75 transport epochs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping
import uuid

from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortPlan,
    load_round74_segmented_cohort_plan,
)
from .round74_segmented_campaign_runner import (
    ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES,
    ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES,
    ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES,
    Round74SegmentedCampaignRunnerConfig,
    run_round74_segmented_campaign_current_slot,
    select_round74_segmented_campaign_slot,
)
from .storage import write_json_atomic


ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION = (
    "round-075-continuous-capture-contract-v4"
)
ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION = (
    "round-075-continuous-capture-state-v1"
)
ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION = (
    "round-075-continuous-capture-missed-slot-v1"
)
ROUND75_CONTINUOUS_CAPTURE_SERVICE_SCHEMA_VERSION = (
    "round-075-continuous-capture-service-v1"
)
ROUND75_SHARD_SLOT_COUNT = 24
ROUND75_SHARD_SIZE_CAP_BYTES = 2 * 1024 * 1024 * 1024
ROUND75_CAMPAIGN_SIZE_CAP_BYTES = 40 * 1024 * 1024 * 1024
ROUND75_HEARTBEAT_INTERVAL_SECONDS = 5.0
ROUND75_IDLE_POLL_MAXIMUM_SECONDS = 5.0
ROUND75_SHARD_PREFIX = "round75-prospective-event-cohort-shard-"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root differs")
    return value


def _is_within(root: Path, path: Path) -> bool:
    selected = path.resolve()
    return selected == root or root in selected.parents


@dataclass(frozen=True)
class Round75ContinuousCaptureConfig:
    """All mutable host paths and immutable service resource ceilings."""

    repository: Path
    contract_path: Path
    plan_path: Path
    prerequisite_path: Path
    data_root: Path
    state_root: Path
    service_state_path: Path
    lease_path: Path
    stop_request_path: Path
    shard_slot_count: int = ROUND75_SHARD_SLOT_COUNT
    shard_size_cap_bytes: int = ROUND75_SHARD_SIZE_CAP_BYTES
    campaign_size_cap_bytes: int = ROUND75_CAMPAIGN_SIZE_CAP_BYTES
    minimum_free_bytes: int = ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES
    slot_growth_cap_bytes: int = ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 2

    def validate(self) -> None:
        repository = self.repository.resolve()
        paths = (
            self.contract_path,
            self.plan_path,
            self.prerequisite_path,
            self.data_root,
            self.state_root,
            self.service_state_path,
            self.lease_path,
            self.stop_request_path,
        )
        integer_fields = (
            (self.shard_slot_count, ROUND75_SHARD_SLOT_COUNT),
            (self.shard_size_cap_bytes, ROUND75_SHARD_SIZE_CAP_BYTES),
            (self.campaign_size_cap_bytes, ROUND75_CAMPAIGN_SIZE_CAP_BYTES),
            (
                self.minimum_free_bytes,
                ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES,
            ),
            (
                self.slot_growth_cap_bytes,
                ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES,
            ),
            (self.duckdb_threads, 2),
        )
        if (
            not repository.is_dir()
            or not self.contract_path.is_file()
            or not self.plan_path.is_file()
            or not self.prerequisite_path.is_file()
            or any(path.is_symlink() for path in paths)
            or any(not _is_within(repository, path) for path in paths)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
                for value, expected in integer_fields
            )
            or self.duckdb_memory_limit != "2GB"
        ):
            raise ValueError("Round 75 continuous capture config differs")

    def shard_path(self, slot_ordinal: int) -> Path:
        if (
            isinstance(slot_ordinal, bool)
            or not isinstance(slot_ordinal, int)
            or slot_ordinal < 0
        ):
            raise ValueError("Round 75 shard slot ordinal differs")
        shard_ordinal = slot_ordinal // self.shard_slot_count
        return self.data_root / f"{ROUND75_SHARD_PREFIX}{shard_ordinal:03d}.duckdb"


class Round75ExclusiveLease:
    """Cross-platform advisory lease held for the service process lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> Round75ExclusiveLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Round 75 continuous capture lease is held") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def load_round75_continuous_capture_contract(
    path: Path,
) -> tuple[dict[str, object], str]:
    value = _load_json_object(path, "Round 75 continuous capture contract")
    canonical = dict(value)
    claimed = str(canonical.pop("artifact_sha256", ""))
    implementation = value.get("implementation")
    resources = value.get("resource_contract")
    activated_plan = value.get("activated_plan")
    if (
        value.get("schema_version")
        != ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION
        or claimed != _canonical_sha256(canonical)
        or not isinstance(implementation, dict)
        or not isinstance(resources, dict)
        or not isinstance(activated_plan, dict)
        or implementation.get("continuous_service_path")
        != "src/simple_ai_trading/round75_continuous_capture.py"
        or implementation.get("continuous_service_file_sha256")
        != _file_sha256(Path(__file__))
        or resources.get("shard_slot_count") != ROUND75_SHARD_SLOT_COUNT
        or resources.get("shard_size_cap_bytes") != ROUND75_SHARD_SIZE_CAP_BYTES
        or resources.get("campaign_size_cap_bytes") != ROUND75_CAMPAIGN_SIZE_CAP_BYTES
        or resources.get("minimum_free_bytes")
        != ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES
        or resources.get("slot_growth_cap_bytes")
        != ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES
        or resources.get("duckdb_memory_limit") != "2GB"
        or resources.get("duckdb_threads") != 2
        or activated_plan.get("plan_sha256") is None
        or activated_plan.get("plan_file_sha256") is None
        or activated_plan.get("plan_path") is None
    ):
        raise ValueError("Round 75 continuous capture contract differs")
    return value, claimed


def _load_plan(config: Round75ContinuousCaptureConfig) -> Round74SegmentedCohortPlan:
    config.validate()
    contract, _contract_sha256 = load_round75_continuous_capture_contract(
        config.contract_path
    )
    plan = load_round74_segmented_cohort_plan(
        config.plan_path.read_text(encoding="utf-8")
    )
    activated_plan = contract["activated_plan"]
    expected_path = (config.repository / str(activated_plan["plan_path"])).resolve()
    if (
        expected_path != config.plan_path.resolve()
        or activated_plan["plan_file_sha256"] != _file_sha256(config.plan_path)
        or activated_plan["plan_sha256"] != plan.plan_sha256
    ):
        raise ValueError("Round 75 activated plan differs")
    return plan


def inspect_round75_storage(
    config: Round75ContinuousCaptureConfig,
    *,
    slot_ordinal: int,
) -> dict[str, object]:
    """Inspect only file metadata; never open a campaign database or WAL."""

    shard = config.shard_path(slot_ordinal)
    database_paths = sorted(
        config.data_root.glob(f"{ROUND75_SHARD_PREFIX}*.duckdb"),
        key=lambda path: path.name,
    )
    wal_paths = sorted(
        config.data_root.glob(f"{ROUND75_SHARD_PREFIX}*.duckdb.wal"),
        key=lambda path: path.name,
    )
    if any(path.is_symlink() for path in (*database_paths, *wal_paths)):
        raise ValueError("Round 75 storage contains a symbolic link")
    database_bytes = sum(path.stat().st_size for path in database_paths)
    wal_bytes = sum(path.stat().st_size for path in wal_paths)
    shard_bytes = shard.stat().st_size if shard.is_file() else 0
    selected_wal = Path(f"{shard}.wal")
    selected_wal_bytes = selected_wal.stat().st_size if selected_wal.is_file() else 0
    free_bytes = shutil.disk_usage(config.data_root).free
    checks = {
        "campaign_size_cap_passed": (
            database_bytes + wal_bytes < config.campaign_size_cap_bytes
        ),
        "selected_shard_size_cap_passed": (
            shard_bytes + selected_wal_bytes < config.shard_size_cap_bytes
        ),
        "minimum_free_space_passed": free_bytes >= config.minimum_free_bytes,
        "all_campaign_wals_absent": wal_bytes == 0,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "database_file_count": len(database_paths),
        "database_bytes": database_bytes,
        "wal_file_count": len(wal_paths),
        "wal_bytes": wal_bytes,
        "selected_shard_ordinal": slot_ordinal // config.shard_slot_count,
        "selected_shard_bytes": shard_bytes,
        "selected_wal_bytes": selected_wal_bytes,
        "free_bytes": free_bytes,
        "database_opened": False,
    }


def _state_payload(
    *,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
    instance_id: str,
    started_wall_ns: int,
    phase: str,
    slot_ordinal: int | None,
    detail: Mapping[str, object] | None = None,
    now_wall_ns: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION,
        "service_schema_version": ROUND75_CONTINUOUS_CAPTURE_SERVICE_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "contract_sha256": contract_sha256,
        "service_instance_id": instance_id,
        "service_process_id": os.getpid(),
        "service_parent_process_id": os.getppid(),
        "service_started_wall_ns": started_wall_ns,
        "heartbeat_wall_ns": time.time_ns() if now_wall_ns is None else now_wall_ns,
        "phase": phase,
        "slot_ordinal": slot_ordinal,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    if detail is not None:
        payload["detail"] = dict(detail)
    payload["state_sha256"] = _canonical_sha256(payload)
    return payload


def _write_state(
    config: Round75ContinuousCaptureConfig,
    **kwargs: object,
) -> dict[str, object]:
    payload = _state_payload(**kwargs)
    write_json_atomic(config.service_state_path, payload, indent=2, sort_keys=True)
    return payload


def _terminalize_missed_slots(
    config: Round75ContinuousCaptureConfig,
    *,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
    observed_wall_ns: int,
) -> int:
    missed_root = config.state_root / "missed-slots"
    missed_root.mkdir(parents=True, exist_ok=True)
    created = 0
    for ordinal in range(plan.total_slots):
        slot = plan.slot(ordinal)
        if observed_wall_ns <= slot.start_window_end_wall_ns:
            break
        slot_root = config.state_root / f"slot-{ordinal:03d}"
        if (slot_root / "reservation.json").exists():
            continue
        path = missed_root / f"{ordinal:03d}.json"
        payload: dict[str, object] = {
            "schema_version": ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "contract_sha256": contract_sha256,
            "slot_ordinal": ordinal,
            "role": slot.role,
            "scheduled_start_wall_ns": slot.scheduled_start_wall_ns,
            "start_window_end_wall_ns": slot.start_window_end_wall_ns,
            "first_observed_missed_wall_ns": observed_wall_ns,
            "reason_code": "service_observed_start_window_elapsed_without_reservation",
            "exact_host_cause_established": False,
            "automatic_retry_permitted": False,
            "replacement_or_time_shift_permitted": False,
            "credentials_used": False,
            "orders_submitted": False,
            "model_data_admitted": False,
            "trading_authority": False,
        }
        payload["missed_sha256"] = _canonical_sha256(payload)
        if path.exists():
            existing = _load_json_object(path, "Round 75 missed-slot receipt")
            canonical = dict(existing)
            claimed = str(canonical.pop("missed_sha256", ""))
            if (
                claimed != _canonical_sha256(canonical)
                or existing.get("plan_sha256") != plan.plan_sha256
                or existing.get("contract_sha256") != contract_sha256
                or existing.get("slot_ordinal") != ordinal
            ):
                raise ValueError("Round 75 missed-slot receipt differs")
            continue
        write_json_atomic(path, payload, indent=2, sort_keys=True)
        created += 1
    return created


def _runner_config(
    config: Round75ContinuousCaptureConfig,
    *,
    slot_ordinal: int,
) -> Round74SegmentedCampaignRunnerConfig:
    return Round74SegmentedCampaignRunnerConfig(
        repository=config.repository,
        plan_path=config.plan_path,
        prerequisite_path=config.prerequisite_path,
        database_path=config.shard_path(slot_ordinal),
        state_root=config.state_root,
        database_cap_bytes=ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES,
        slot_growth_cap_bytes=config.slot_growth_cap_bytes,
        minimum_free_bytes=config.minimum_free_bytes,
        duckdb_memory_limit=config.duckdb_memory_limit,
        duckdb_threads=config.duckdb_threads,
    )


def _run_slot_with_heartbeat(
    config: Round75ContinuousCaptureConfig,
    *,
    plan: Round74SegmentedCohortPlan,
    contract_sha256: str,
    instance_id: str,
    started_wall_ns: int,
    slot_ordinal: int,
    run_slot: Callable[..., dict[str, object]],
    clock_ns: Callable[[], int],
    sleep: Callable[[float], None],
) -> dict[str, object]:
    runner_config = _runner_config(config, slot_ordinal=slot_ordinal)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="round75-slot") as pool:
        future: Future[dict[str, object]] = pool.submit(run_slot, runner_config)
        while not future.done():
            stop_requested = config.stop_request_path.exists()
            _write_state(
                config,
                plan=plan,
                contract_sha256=contract_sha256,
                instance_id=instance_id,
                started_wall_ns=started_wall_ns,
                phase=(
                    "stop_requested_finishing_current_slot"
                    if stop_requested
                    else "capturing"
                ),
                slot_ordinal=slot_ordinal,
                detail={"stop_requested": stop_requested},
                now_wall_ns=clock_ns(),
            )
            sleep(ROUND75_HEARTBEAT_INTERVAL_SECONDS)
        return future.result()


def run_round75_continuous_capture(
    config: Round75ContinuousCaptureConfig,
    *,
    once: bool = False,
    clock_ns: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
    run_slot: Callable[..., dict[str, object]] = (
        run_round74_segmented_campaign_current_slot
    ),
) -> dict[str, object]:
    """Run the prospective schedule without shifting or retrying a market window."""

    config.data_root.mkdir(parents=True, exist_ok=True)
    config.state_root.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(config)
    _contract, contract_sha256 = load_round75_continuous_capture_contract(
        config.contract_path
    )
    instance_id = uuid.uuid4().hex
    started_wall_ns = clock_ns()
    with Round75ExclusiveLease(config.lease_path):
        while True:
            now_wall_ns = clock_ns()
            created_missed = _terminalize_missed_slots(
                config,
                plan=plan,
                contract_sha256=contract_sha256,
                observed_wall_ns=now_wall_ns,
            )
            selection = select_round74_segmented_campaign_slot(
                plan,
                now_wall_ns=now_wall_ns,
            )
            if config.stop_request_path.exists():
                return _write_state(
                    config,
                    plan=plan,
                    contract_sha256=contract_sha256,
                    instance_id=instance_id,
                    started_wall_ns=started_wall_ns,
                    phase="stopped",
                    slot_ordinal=selection.slot_ordinal,
                    detail={"missed_receipts_created": created_missed},
                    now_wall_ns=now_wall_ns,
                )
            if selection.status == "after_campaign":
                return _write_state(
                    config,
                    plan=plan,
                    contract_sha256=contract_sha256,
                    instance_id=instance_id,
                    started_wall_ns=started_wall_ns,
                    phase="campaign_terminal",
                    slot_ordinal=None,
                    detail={"missed_receipts_created": created_missed},
                    now_wall_ns=now_wall_ns,
                )
            if selection.status == "open" and selection.slot_ordinal is not None:
                ordinal = selection.slot_ordinal
                slot_root = config.state_root / f"slot-{ordinal:03d}"
                reservation = slot_root / "reservation.json"
                if not reservation.exists():
                    storage = inspect_round75_storage(config, slot_ordinal=ordinal)
                    if storage["passed"] is not True:
                        raise RuntimeError("Round 75 storage resource gate failed")
                    result = _run_slot_with_heartbeat(
                        config,
                        plan=plan,
                        contract_sha256=contract_sha256,
                        instance_id=instance_id,
                        started_wall_ns=started_wall_ns,
                        slot_ordinal=ordinal,
                        run_slot=run_slot,
                        clock_ns=clock_ns,
                        sleep=sleep,
                    )
                    _write_state(
                        config,
                        plan=plan,
                        contract_sha256=contract_sha256,
                        instance_id=instance_id,
                        started_wall_ns=started_wall_ns,
                        phase="slot_terminal",
                        slot_ordinal=ordinal,
                        detail={
                            "result_sha256": result.get("result_sha256", ""),
                            "storage_before_slot": storage,
                        },
                        now_wall_ns=clock_ns(),
                    )
                else:
                    _write_state(
                        config,
                        plan=plan,
                        contract_sha256=contract_sha256,
                        instance_id=instance_id,
                        started_wall_ns=started_wall_ns,
                        phase="slot_already_reserved_no_retry",
                        slot_ordinal=ordinal,
                        detail={"missed_receipts_created": created_missed},
                        now_wall_ns=now_wall_ns,
                    )
                if once:
                    return _load_json_object(
                        config.service_state_path,
                        "Round 75 service state",
                    )
            else:
                _write_state(
                    config,
                    plan=plan,
                    contract_sha256=contract_sha256,
                    instance_id=instance_id,
                    started_wall_ns=started_wall_ns,
                    phase=(
                        "waiting_for_campaign_start"
                        if selection.status == "before_campaign"
                        else "waiting_for_next_fixed_slot"
                    ),
                    slot_ordinal=selection.slot_ordinal,
                    detail={"missed_receipts_created": created_missed},
                    now_wall_ns=now_wall_ns,
                )
                if once:
                    return _load_json_object(
                        config.service_state_path,
                        "Round 75 service state",
                    )
            sleep(ROUND75_IDLE_POLL_MAXIMUM_SECONDS)


__all__ = [
    "ROUND75_CAMPAIGN_SIZE_CAP_BYTES",
    "ROUND75_CONTINUOUS_CAPTURE_CONTRACT_SCHEMA_VERSION",
    "ROUND75_CONTINUOUS_CAPTURE_MISSED_SCHEMA_VERSION",
    "ROUND75_CONTINUOUS_CAPTURE_SERVICE_SCHEMA_VERSION",
    "ROUND75_CONTINUOUS_CAPTURE_STATE_SCHEMA_VERSION",
    "ROUND75_HEARTBEAT_INTERVAL_SECONDS",
    "ROUND75_IDLE_POLL_MAXIMUM_SECONDS",
    "ROUND75_SHARD_PREFIX",
    "ROUND75_SHARD_SIZE_CAP_BYTES",
    "ROUND75_SHARD_SLOT_COUNT",
    "Round75ContinuousCaptureConfig",
    "Round75ExclusiveLease",
    "inspect_round75_storage",
    "load_round75_continuous_capture_contract",
    "run_round75_continuous_capture",
]
