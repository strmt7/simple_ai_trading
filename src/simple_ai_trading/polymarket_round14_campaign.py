"""Bounded coordinator for the immutable Round 14 BTC capture campaign."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import BinaryIO

from .polymarket_recorder import PolymarketEvidenceStore, PolymarketPublicRecorder
from .polymarket_round14_capture import (
    POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS,
    build_round14_capture_manifest,
)
from .polymarket_round14_contract import load_round14_contract
from .storage import write_bytes_atomic


POLYMARKET_ROUND14_CAMPAIGN_PLAN_SCHEMA_VERSION = (
    "polymarket-round14-campaign-plan-v1"
)
POLYMARKET_ROUND14_CAMPAIGN_SLOT_RESULT_SCHEMA_VERSION = (
    "polymarket-round14-campaign-slot-result-v1"
)
POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION = (
    "polymarket-round14-campaign-state-v1"
)
POLYMARKET_ROUND14_TOTAL_SLOTS = 1_440
POLYMARKET_ROUND14_START_TOLERANCE_MS = 60_000
POLYMARKET_ROUND14_DATABASE_CAP_BYTES = 512 * 1024 * 1024 * 1024
POLYMARKET_ROUND14_MINIMUM_FREE_BYTES = 512 * 1024 * 1024 * 1024
POLYMARKET_ROUND14_QUEUE_CAPACITY = 100_000
POLYMARKET_ROUND14_MEMORY_LIMIT = "1GB"
POLYMARKET_ROUND14_DATABASE_THREADS = 2
POLYMARKET_ROUND14_MAXIMUM_CONCURRENT_UNITS = 2
_PLAN_MAX_BYTES = 64 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 14 campaign JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 14 campaign JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _read_strict_json(path: Path, *, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unavailable")
    size = path.stat().st_size
    if size < 2 or size > _PLAN_MAX_BYTES:
        raise ValueError(f"{label} size is invalid")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not an object")
    return payload


@dataclass(frozen=True, slots=True)
class PolymarketRound14CampaignPlan:
    created_at_ms: int
    scheduled_start_ms: int
    scheduled_end_ms: int
    contract_repository_path: str
    contract_sha256: str
    total_slots: int
    plan_sha256: str

    def scheduled_slot_ms(self, slot_index: int) -> int:
        index = int(slot_index)
        if index < 0 or index >= self.total_slots:
            raise ValueError("Round 14 campaign slot index is invalid")
        return self.scheduled_start_ms + (
            index * POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS * 1_000
        )


def create_round14_campaign_plan(
    *,
    created_at_ms: int,
    scheduled_start_ms: int,
    contract_repository_path: str,
    contract_sha256: str,
) -> dict[str, object]:
    start = int(scheduled_start_ms)
    unit_ms = POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS * 1_000
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND14_CAMPAIGN_PLAN_SCHEMA_VERSION,
        "created_at_ms": int(created_at_ms),
        "scheduled_start_ms": start,
        "scheduled_end_ms": start + POLYMARKET_ROUND14_TOTAL_SLOTS * unit_ms,
        "capture_unit_seconds": POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS,
        "total_slots": POLYMARKET_ROUND14_TOTAL_SLOTS,
        "contract_repository_path": str(contract_repository_path),
        "contract_sha256": str(contract_sha256).strip().lower(),
        "required_assets": ["BTC"],
        "required_streams": [
            "binance_futures",
            "binance_spot",
            "clob_market",
            "polymarket_rtds",
        ],
        "decision_cadence_ms": 250,
        "queue_capacity": POLYMARKET_ROUND14_QUEUE_CAPACITY,
        "memory_limit": POLYMARKET_ROUND14_MEMORY_LIMIT,
        "database_threads": POLYMARKET_ROUND14_DATABASE_THREADS,
        "database_cap_bytes": POLYMARKET_ROUND14_DATABASE_CAP_BYTES,
        "minimum_free_bytes": POLYMARKET_ROUND14_MINIMUM_FREE_BYTES,
        "maximum_concurrent_units": POLYMARKET_ROUND14_MAXIMUM_CONCURRENT_UNITS,
        "labels_consulted": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    validate_round14_campaign_plan(payload)
    return payload


def validate_round14_campaign_plan(
    value: Mapping[str, object],
) -> PolymarketRound14CampaignPlan:
    payload = dict(value)
    claimed = str(payload.pop("plan_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "created_at_ms",
        "scheduled_start_ms",
        "scheduled_end_ms",
        "capture_unit_seconds",
        "total_slots",
        "contract_repository_path",
        "contract_sha256",
        "required_assets",
        "required_streams",
        "decision_cadence_ms",
        "queue_capacity",
        "memory_limit",
        "database_threads",
        "database_cap_bytes",
        "minimum_free_bytes",
        "maximum_concurrent_units",
        "labels_consulted",
        "outcomes_consulted",
        "model_scores_consulted",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    start = payload.get("scheduled_start_ms")
    end = payload.get("scheduled_end_ms")
    created = payload.get("created_at_ms")
    unit_ms = POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS * 1_000
    booleans = (
        "labels_consulted",
        "outcomes_consulted",
        "model_scores_consulted",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        set(payload) != expected_keys
        or payload["schema_version"]
        != POLYMARKET_ROUND14_CAMPAIGN_PLAN_SCHEMA_VERSION
        or type(created) is not int
        or type(start) is not int
        or type(end) is not int
        or created <= 0
        or start <= created
        or start % unit_ms
        or end != start + POLYMARKET_ROUND14_TOTAL_SLOTS * unit_ms
        or payload["capture_unit_seconds"]
        != POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS
        or payload["total_slots"] != POLYMARKET_ROUND14_TOTAL_SLOTS
        or not isinstance(payload["contract_repository_path"], str)
        or not payload["contract_repository_path"].startswith(
            "docs/model-research/polymarket/"
        )
        or "\\" in payload["contract_repository_path"]
        or ".." in Path(payload["contract_repository_path"]).parts
        or _SHA256.fullmatch(str(payload["contract_sha256"])) is None
        or payload["required_assets"] != ["BTC"]
        or payload["required_streams"]
        != [
            "binance_futures",
            "binance_spot",
            "clob_market",
            "polymarket_rtds",
        ]
        or payload["decision_cadence_ms"] != 250
        or payload["queue_capacity"] != POLYMARKET_ROUND14_QUEUE_CAPACITY
        or payload["memory_limit"] != POLYMARKET_ROUND14_MEMORY_LIMIT
        or payload["database_threads"] != POLYMARKET_ROUND14_DATABASE_THREADS
        or payload["database_cap_bytes"] != POLYMARKET_ROUND14_DATABASE_CAP_BYTES
        or payload["minimum_free_bytes"] != POLYMARKET_ROUND14_MINIMUM_FREE_BYTES
        or payload["maximum_concurrent_units"]
        != POLYMARKET_ROUND14_MAXIMUM_CONCURRENT_UNITS
        or any(payload[name] is not False for name in booleans)
        or _SHA256.fullmatch(claimed) is None
        or _canonical_sha256(payload) != claimed
    ):
        raise ValueError("Round 14 campaign plan is invalid")
    return PolymarketRound14CampaignPlan(
        created_at_ms=created,
        scheduled_start_ms=start,
        scheduled_end_ms=end,
        contract_repository_path=payload["contract_repository_path"],
        contract_sha256=str(payload["contract_sha256"]),
        total_slots=POLYMARKET_ROUND14_TOTAL_SLOTS,
        plan_sha256=claimed,
    )


def load_round14_campaign_plan(path: str | Path) -> PolymarketRound14CampaignPlan:
    return validate_round14_campaign_plan(
        _read_strict_json(Path(path), label="Round 14 campaign plan")
    )


@dataclass(frozen=True, slots=True)
class PolymarketRound14CampaignConfig:
    repository: Path
    plan_path: Path
    database_path: Path
    state_root: Path

    def validate(self) -> None:
        root = self.repository.resolve()
        plan = self.plan_path.resolve()
        database = self.database_path.resolve()
        state = self.state_root.resolve()
        if (
            not root.is_dir()
            or not plan.is_file()
            or root not in plan.parents
            or database == plan
            or state == plan
            or database == state
            or database in state.parents
            or state in database.parents
        ):
            raise ValueError("Round 14 campaign configuration is invalid")


class _CampaignFileLock(AbstractContextManager["_CampaignFileLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self) -> "_CampaignFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Round 14 campaign is already running") from exc
        self.handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        handle = self.handle
        if handle is None:
            return
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self.handle = None


def _write_hashed_json(path: Path, payload: Mapping[str, object]) -> str:
    output = dict(payload)
    digest = _canonical_sha256(output)
    output["artifact_sha256"] = digest
    encoded = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(path, encoded)
    return digest


def _slot_result_path(state_root: Path, slot_index: int) -> Path:
    return state_root / "slots" / f"slot-{slot_index:04d}.json"


def _write_slot_result(
    state_root: Path,
    *,
    plan: PolymarketRound14CampaignPlan,
    slot_index: int,
    scheduled_start_ms: int,
    status: str,
    observed_at_ms: int,
    details: Mapping[str, object],
) -> dict[str, object]:
    if status not in {
        "complete",
        "degraded",
        "failed",
        "missed",
        "resource_blocked",
    }:
        raise ValueError("Round 14 campaign slot status is invalid")
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND14_CAMPAIGN_SLOT_RESULT_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "slot_index": int(slot_index),
        "scheduled_start_ms": int(scheduled_start_ms),
        "observed_at_ms": int(observed_at_ms),
        "status": status,
        "condition_level_admission_required": status in {"complete", "degraded"},
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "details": dict(details),
    }
    path = _slot_result_path(state_root, slot_index)
    if path.exists():
        existing = _read_strict_json(path, label="Round 14 slot result")
        claimed = str(existing.get("artifact_sha256") or "")
        body = dict(existing)
        body.pop("artifact_sha256", None)
        if claimed != _canonical_sha256(body):
            raise ValueError("existing Round 14 slot result is invalid")
        return dict(existing)
    _write_hashed_json(path, payload)
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def _completed_slot_indexes(
    state_root: Path,
    plan: PolymarketRound14CampaignPlan,
) -> frozenset[int]:
    output: set[int] = set()
    slot_root = state_root / "slots"
    if not slot_root.exists():
        return frozenset()
    for path in sorted(slot_root.glob("slot-*.json")):
        payload = _read_strict_json(path, label="Round 14 slot result")
        claimed = str(payload.get("artifact_sha256") or "")
        body = dict(payload)
        body.pop("artifact_sha256", None)
        index = body.get("slot_index")
        if (
            claimed != _canonical_sha256(body)
            or body.get("schema_version")
            != POLYMARKET_ROUND14_CAMPAIGN_SLOT_RESULT_SCHEMA_VERSION
            or body.get("plan_sha256") != plan.plan_sha256
            or type(index) is not int
            or index in output
            or not 0 <= index < plan.total_slots
            or path != _slot_result_path(state_root, index)
        ):
            raise ValueError("Round 14 slot result set is invalid")
        output.add(index)
    return frozenset(output)


def _resource_reason(config: PolymarketRound14CampaignConfig) -> str | None:
    database_size = (
        config.database_path.stat().st_size
        if config.database_path.is_file()
        else 0
    )
    if database_size >= POLYMARKET_ROUND14_DATABASE_CAP_BYTES:
        return "database_cap_reached"
    probe = config.database_path.parent
    probe.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(probe).free < POLYMARKET_ROUND14_MINIMUM_FREE_BYTES:
        return "minimum_free_space_not_met"
    return None


def _recover_orphaned_runs(config: PolymarketRound14CampaignConfig) -> int:
    if not config.database_path.is_file():
        return 0
    now_ms = time.time_ns() // 1_000_000
    recovered = 0
    with PolymarketEvidenceStore(
        config.database_path,
        memory_limit=POLYMARKET_ROUND14_MEMORY_LIMIT,
        threads=POLYMARKET_ROUND14_DATABASE_THREADS,
    ) as store:
        rows = store.connect().execute(
            """
            SELECT run_id, started_at_ms
            FROM polymarket_recorder_run
            WHERE status = 'running'
            ORDER BY started_at_ms, run_id
            """
        ).fetchall()
        for run_id, started_at_ms in rows:
            store.fail_run(
                str(run_id),
                started_at_ms=int(started_at_ms),
                ended_at_ms=max(int(started_at_ms), now_ms),
                database=str(config.database_path),
                errors=("campaign_restart_interrupted_run",),
            )
            recovered += 1
    return recovered


def _load_and_verify(
    config: PolymarketRound14CampaignConfig,
) -> tuple[PolymarketRound14CampaignPlan, Path]:
    config.validate()
    plan = load_round14_campaign_plan(config.plan_path)
    contract = (config.repository / plan.contract_repository_path).resolve()
    if config.repository.resolve() not in contract.parents:
        raise ValueError("Round 14 campaign contract is outside the repository")
    program = load_round14_contract(contract)
    if program.contract_sha256 != plan.contract_sha256:
        raise ValueError("Round 14 campaign contract identity differs")
    return plan, contract


async def _capture_slot(
    config: PolymarketRound14CampaignConfig,
    plan: PolymarketRound14CampaignPlan,
    contract_path: Path,
    *,
    slot_index: int,
    scheduled_start_ms: int,
) -> dict[str, object]:
    heartbeat_path = (
        config.state_root / "heartbeats" / f"slot-{slot_index:04d}.json"
    )

    def progress(_phase: str, details: Mapping[str, object]) -> None:
        payload = {
            "schema_version": POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "slot_index": slot_index,
            "scheduled_start_ms": scheduled_start_ms,
            "details": dict(details),
            "profitability_claim": False,
            "trading_authority": False,
        }
        _write_hashed_json(heartbeat_path, payload)

    additional_files = (
        config.plan_path.resolve().relative_to(config.repository.resolve()).as_posix(),
        "src/simple_ai_trading/polymarket_round14_campaign.py",
        "tools/run_polymarket_round14_campaign.py",
        "tests/test_polymarket_round14_campaign.py",
    )

    def manifest_factory(run_id: str, started_at_ms: int) -> Mapping[str, object]:
        return build_round14_capture_manifest(
            contract_path,
            run_id=run_id,
            created_at_ms=started_at_ms,
            capture_duration_seconds=POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS,
            purpose="prospective",
            slot_index=slot_index,
            scheduled_start_ms=scheduled_start_ms,
            campaign_plan_sha256=plan.plan_sha256,
            additional_required_files=additional_files,
        )

    recorder = PolymarketPublicRecorder(
        config.database_path,
        queue_capacity=POLYMARKET_ROUND14_QUEUE_CAPACITY,
        memory_limit=POLYMARKET_ROUND14_MEMORY_LIMIT,
        database_threads=POLYMARKET_ROUND14_DATABASE_THREADS,
        assets=("BTC",),
        include_binance_futures=True,
    )
    try:
        report = await recorder.run(
            duration_seconds=POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS,
            progress=progress,
            progress_interval_seconds=30,
            preregistration_manifest_factory=manifest_factory,
        )
        status = (
            report.status
            if report.status in {"complete", "degraded"}
            else "failed"
        )
        details: dict[str, object] = {
            "run_id": report.run_id,
            "recorder_status": report.status,
            "report_sha256": report.report_sha256,
            "duration_seconds": report.duration_seconds,
            "raw_message_count": report.raw_message_count,
            "normalized_event_count": report.normalized_event_count,
            "market_snapshot_count": report.market_snapshot_count,
            "stream_gap_count": report.stream_gap_count,
            "stream_counts": dict(report.stream_counts),
            "errors": list(report.errors),
            "integrity_errors": list(report.integrity_errors),
            "evidence_manifest_sha256": report.evidence_manifest_sha256,
        }
    except Exception as exc:
        status = "failed"
        details = {
            "failure_type": type(exc).__name__,
            "failure": str(exc)[:2_000],
        }
    return _write_slot_result(
        config.state_root,
        plan=plan,
        slot_index=slot_index,
        scheduled_start_ms=scheduled_start_ms,
        status=status,
        observed_at_ms=time.time_ns() // 1_000_000,
        details=details,
    )


def inspect_round14_campaign(
    config: PolymarketRound14CampaignConfig,
    *,
    now_ms: int | None = None,
) -> dict[str, object]:
    plan, _contract = _load_and_verify(config)
    observed = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
    completed = _completed_slot_indexes(config.state_root, plan)
    if observed < plan.scheduled_start_ms:
        relation = "before_campaign"
        current_slot = None
    elif observed >= plan.scheduled_end_ms:
        relation = "after_campaign"
        current_slot = None
    else:
        current_slot = (
            observed - plan.scheduled_start_ms
        ) // (POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS * 1_000)
        relation = (
            "open"
            if observed - plan.scheduled_slot_ms(current_slot)
            <= POLYMARKET_ROUND14_START_TOLERANCE_MS
            else "between_slots"
        )
    return {
        "schema_version": POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "observed_at_ms": observed,
        "relation": relation,
        "current_slot_index": current_slot,
        "terminal_slot_count": len(completed),
        "resource_block": _resource_reason(config),
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


async def run_round14_campaign(
    config: PolymarketRound14CampaignConfig,
    *,
    poll_interval_seconds: float = 1.0,
) -> dict[str, object]:
    plan, contract_path = _load_and_verify(config)
    interval = float(poll_interval_seconds)
    if not 0.1 <= interval <= 30.0:
        raise ValueError("Round 14 campaign poll interval is invalid")
    config.state_root.mkdir(parents=True, exist_ok=True)
    active: dict[int, asyncio.Task[dict[str, object]]] = {}
    with _CampaignFileLock(config.state_root / "campaign.lock"):
        recovered = _recover_orphaned_runs(config)
        terminal = set(_completed_slot_indexes(config.state_root, plan))
        next_index = 0
        while next_index in terminal:
            next_index += 1
        next_state_write = 0.0
        while True:
            done_indexes = [
                index for index, task in active.items() if task.done()
            ]
            for index in done_indexes:
                await active.pop(index)
                terminal.add(index)

            now_ms = time.time_ns() // 1_000_000
            if now_ms >= plan.scheduled_end_ms and not active:
                break
            while next_index < plan.total_slots:
                index = next_index
                if index in terminal:
                    next_index += 1
                    continue
                scheduled = plan.scheduled_slot_ms(next_index)
                if scheduled > now_ms:
                    break
                delay = now_ms - scheduled
                if delay > POLYMARKET_ROUND14_START_TOLERANCE_MS:
                    _write_slot_result(
                        config.state_root,
                        plan=plan,
                        slot_index=index,
                        scheduled_start_ms=scheduled,
                        status="missed",
                        observed_at_ms=now_ms,
                        details={"reason": "start_tolerance_exceeded"},
                    )
                    terminal.add(index)
                    next_index += 1
                    continue
                resource = _resource_reason(config)
                if resource is not None:
                    _write_slot_result(
                        config.state_root,
                        plan=plan,
                        slot_index=index,
                        scheduled_start_ms=scheduled,
                        status="resource_blocked",
                        observed_at_ms=now_ms,
                        details={"reason": resource},
                    )
                    terminal.add(index)
                    next_index += 1
                    continue
                if len(active) >= POLYMARKET_ROUND14_MAXIMUM_CONCURRENT_UNITS:
                    break
                active[index] = asyncio.create_task(
                    _capture_slot(
                        config,
                        plan,
                        contract_path,
                        slot_index=index,
                        scheduled_start_ms=scheduled,
                    )
                )
                next_index += 1

            monotonic_now = time.monotonic()
            if done_indexes or monotonic_now >= next_state_write:
                state = {
                    "schema_version": (
                        POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION
                    ),
                    "plan_sha256": plan.plan_sha256,
                    "observed_at_ms": now_ms,
                    "active_slot_indexes": sorted(active),
                    "next_slot_index": next_index,
                    "terminal_slot_count": len(terminal),
                    "recovered_interrupted_run_count": recovered,
                    "database_bytes": (
                        config.database_path.stat().st_size
                        if config.database_path.is_file()
                        else 0
                    ),
                    "profitability_claim": False,
                    "paper_trading_authority": False,
                    "live_trading_authority": False,
                }
                _write_hashed_json(
                    config.state_root / "campaign-state.json",
                    state,
                )
                next_state_write = monotonic_now + 30.0
            await asyncio.sleep(interval)

        statuses: dict[str, int] = {}
        for index in sorted(terminal):
            payload = _read_strict_json(
                _slot_result_path(config.state_root, index),
                label="Round 14 slot result",
            )
            status = str(payload["status"])
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "schema_version": POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "terminal_slot_count": len(terminal),
            "status_counts": dict(sorted(statuses.items())),
            "recovered_interrupted_run_count": recovered,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }


__all__ = [
    "POLYMARKET_ROUND14_CAMPAIGN_PLAN_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_CAMPAIGN_SLOT_RESULT_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_TOTAL_SLOTS",
    "PolymarketRound14CampaignConfig",
    "PolymarketRound14CampaignPlan",
    "create_round14_campaign_plan",
    "inspect_round14_campaign",
    "load_round14_campaign_plan",
    "run_round14_campaign",
    "validate_round14_campaign_plan",
]
