"""Fixed-window, storage-bounded Round 27 Stage 1 capture campaign."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable, Mapping, Sequence

from .polymarket_recorder import PolymarketPublicRecorder, RecorderReport
from .polymarket_round27_capture import (
    ROUND27_PROGRESS_INTERVAL_SECONDS,
    _AUTHORITY,
    _CAPTURE_SCOPE,
    _CaptureFileLock,
    _audit_sources,
    _create_recorder,
    _database_footprint_bytes,
)
from .storage import write_bytes_atomic, write_json_atomic


ROUND27_STAGE1_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round27-stage1-campaign-contract-v1"
)
ROUND27_STAGE1_RESULT_SCHEMA_VERSION = "polymarket-round27-stage1-slot-result-v1"
ROUND27_STAGE1_STATE_SCHEMA_VERSION = "polymarket-round27-stage1-supervisor-state-v1"
ROUND27_STAGE1_MISSED_SCHEMA_VERSION = "polymarket-round27-stage1-missed-slot-v1"
ROUND27_STAGE1_SLOT_DURATION_SECONDS = 37_800
ROUND27_STAGE1_PRIMARY_SLOT_COUNT = 3
ROUND27_STAGE1_CONTINGENCY_SLOT_COUNT = 1
ROUND27_STAGE1_MINIMUM_ELIGIBLE_MARKETS = 300
ROUND27_STAGE1_MINIMUM_UTC_DATES = 3
ROUND27_STAGE1_START_TOLERANCE_MS = 30_000
ROUND27_STAGE1_END_TOLERANCE_MS = 5_000
ROUND27_STAGE1_DATABASE_CAP_BYTES = 2_500 * 1024**2
ROUND27_STAGE1_DATABASE_STOP_RESERVE_BYTES = 256 * 1024**2
ROUND27_STAGE1_MINIMUM_FREE_BYTES = 32 * 1024**3
ROUND27_STAGE1_HEARTBEAT_SECONDS = 30
ROUND27_STAGE1_SOURCE_RELATIVES = (
    Path("src/simple_ai_trading/polymarket.py"),
    Path("src/simple_ai_trading/polymarket_recorder.py"),
    Path("src/simple_ai_trading/polymarket_source_quality.py"),
    Path("src/simple_ai_trading/polymarket_round27_capture.py"),
    Path("src/simple_ai_trading/polymarket_round27_stage1_capture.py"),
    Path("tools/run_polymarket_round27_stage1.py"),
)
_LINEAGE_ARTIFACTS = {
    "source_qualification_sha256": (
        Path(
            "docs/model-research/polymarket/"
            "binance-usdm-aggregate-trade-source-qualification-v1-2026-08-15.json"
        ),
        "qualification_sha256",
    ),
    "preregistration_sha256": (
        Path(
            "docs/model-research/polymarket/"
            "round-027-execution-hypothesis-preregistration-v3.json"
        ),
        "preregistration_sha256",
    ),
    "stage0_capture_result_sha256": (
        Path(
            "docs/model-research/polymarket/"
            "round-027-stage0-mechanics-capture-result-v1-2026-08-15.json"
        ),
        "result_sha256",
    ),
    "stage0_condition_audit_sha256": (
        Path(
            "docs/model-research/polymarket/latest/"
            "round-027-stage0-condition-audit/condition-replay-audit.json"
        ),
        "audit_sha256",
    ),
    "stage0_mechanics_sha256": (
        Path(
            "docs/model-research/polymarket/latest/"
            "round-027-mechanics-diagnostic/mechanics-diagnostic.json"
        ),
        "mechanics_sha256",
    ),
    "stage0_resolution_audit_sha256": (
        Path(
            "docs/model-research/polymarket/latest/"
            "round-027-stage0-resolution-mechanics/"
            "settlement-mechanics-audit.json"
        ),
        "audit_sha256",
    ),
}
_RESOURCE_POLICY = {
    "database_cap_bytes_per_slot": ROUND27_STAGE1_DATABASE_CAP_BYTES,
    "database_stop_reserve_bytes": ROUND27_STAGE1_DATABASE_STOP_RESERVE_BYTES,
    "minimum_free_bytes": ROUND27_STAGE1_MINIMUM_FREE_BYTES,
    "memory_limit": "1GB",
    "database_threads": 2,
    "queue_capacity": 100_000,
    "progress_interval_seconds": ROUND27_PROGRESS_INTERVAL_SECONDS,
    "one_database_per_slot": True,
}
_CAMPAIGN_POLICY = {
    "minimum_eligible_markets_after_target_free_audit": (
        ROUND27_STAGE1_MINIMUM_ELIGIBLE_MARKETS
    ),
    "minimum_distinct_utc_market_start_dates": ROUND27_STAGE1_MINIMUM_UTC_DATES,
    "selection_unit": "whole_condition_local_executable_interval",
    "target_access_during_capture_or_admission": False,
    "outcome_or_market_state_dependent_replacement": False,
    "primary_slots_required": ROUND27_STAGE1_PRIMARY_SLOT_COUNT,
    "primary_scheduled_five_minute_intervals": (
        ROUND27_STAGE1_PRIMARY_SLOT_COUNT * ROUND27_STAGE1_SLOT_DURATION_SECONDS // 300
    ),
    "contingency_activation_rule": (
        "only_after_all_primary_target_free_replay_audits_if_eligible_total_below_300"
    ),
    "model_fitting_before_campaign_gate": False,
    "economic_claim_before_sealed_evaluation": False,
}
_SUCCESS_GATE = {
    "terminal_recorder_status": "complete_or_gap_degraded",
    "required_stream_message_count_positive": True,
    "transport_gap_policy": "condition_local_exclusion",
    "recorder_error_count": 0,
    "integrity_error_count": 0,
    "documented_spot_and_futures_trade_quality_passed": True,
    "database_footprint_at_or_below_slot_cap": True,
    "scheduled_window_observed": True,
    "condition_local_target_free_audit_required": True,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


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


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _artifact_claim(path: Path, *, field: str, label: str) -> str:
    value = _read_json(path, label=label)
    claimed = str(value.pop(field, "")).lower()
    if _SHA256.fullmatch(claimed) is None or not hmac.compare_digest(
        claimed,
        _canonical_sha256(value),
    ):
        raise ValueError(f"{label} claim differs")
    return claimed


def _repository_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Round 27 Stage 1 repository HEAD differs")
    return commit


@dataclass(frozen=True, slots=True)
class Round27Stage1Slot:
    slot_id: str
    role: str
    scheduled_start_ms: int
    scheduled_end_ms: int

    def validated(self) -> Round27Stage1Slot:
        if (
            re.fullmatch(r"stage1-[a-z]", self.slot_id) is None
            or self.role not in {"primary", "contingency"}
            or type(self.scheduled_start_ms) is not int
            or type(self.scheduled_end_ms) is not int
            or self.scheduled_start_ms <= 0
            or self.scheduled_end_ms - self.scheduled_start_ms
            != ROUND27_STAGE1_SLOT_DURATION_SECONDS * 1_000
            or self.scheduled_start_ms % 300_000 != 0
            or self.scheduled_end_ms % 300_000 != 0
        ):
            raise ValueError("Round 27 Stage 1 slot differs")
        return self

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Round27Stage1Contract:
    contract_sha256: str
    created_at_ms: int
    repository_commit: str
    source_text_sha256: dict[str, str]
    lineage: dict[str, str]
    slots: tuple[Round27Stage1Slot, ...]

    def slot(self, slot_id: str) -> Round27Stage1Slot:
        matches = [slot for slot in self.slots if slot.slot_id == slot_id]
        if len(matches) != 1:
            raise ValueError("Round 27 Stage 1 slot ID is unavailable")
        return matches[0]


def _validate_slots(
    slots: Sequence[Round27Stage1Slot],
) -> tuple[Round27Stage1Slot, ...]:
    selected = tuple(slot.validated() for slot in slots)
    primary = tuple(slot for slot in selected if slot.role == "primary")
    contingency = tuple(slot for slot in selected if slot.role == "contingency")
    starts = [slot.scheduled_start_ms for slot in selected]
    primary_dates = {
        datetime.fromtimestamp(
            slot.scheduled_start_ms / 1_000,
            tz=timezone.utc,
        ).date()
        for slot in primary
    }
    if (
        len(selected)
        != ROUND27_STAGE1_PRIMARY_SLOT_COUNT + ROUND27_STAGE1_CONTINGENCY_SLOT_COUNT
        or len(primary) != ROUND27_STAGE1_PRIMARY_SLOT_COUNT
        or len(contingency) != ROUND27_STAGE1_CONTINGENCY_SLOT_COUNT
        or len(set(slot.slot_id for slot in selected)) != len(selected)
        or starts != sorted(starts)
        or any(
            left.scheduled_end_ms > right.scheduled_start_ms
            for left, right in zip(selected, selected[1:], strict=False)
        )
        or len(primary_dates) < ROUND27_STAGE1_MINIMUM_UTC_DATES
        or contingency[0] != selected[-1]
    ):
        raise ValueError("Round 27 Stage 1 campaign schedule differs")
    return selected


def create_round27_stage1_contract(
    repository: str | Path,
    *,
    created_at_ms: int,
    slots: Sequence[Round27Stage1Slot],
) -> dict[str, object]:
    root = Path(repository).resolve()
    selected_slots = _validate_slots(slots)
    if (
        not root.is_dir()
        or type(created_at_ms) is not int
        or created_at_ms <= 0
        or selected_slots[0].scheduled_start_ms < created_at_ms + 15 * 60 * 1_000
    ):
        raise ValueError("Round 27 Stage 1 contract inputs differ")
    lineage = {
        name: _artifact_claim(root / relative, field=field, label=name)
        for name, (relative, field) in _LINEAGE_ARTIFACTS.items()
    }
    payload: dict[str, object] = {
        "schema_version": ROUND27_STAGE1_CONTRACT_SCHEMA_VERSION,
        "created_at_ms": created_at_ms,
        "phase": "model_development_stage1",
        "asset": "BTC",
        "market": "Polymarket BTC five-minute Up/Down",
        "slot_duration_seconds": ROUND27_STAGE1_SLOT_DURATION_SECONDS,
        "slots": [slot.asdict() for slot in selected_slots],
        "capture_scope": copy.deepcopy(_CAPTURE_SCOPE),
        "resource_policy": copy.deepcopy(_RESOURCE_POLICY),
        "campaign_policy": copy.deepcopy(_CAMPAIGN_POLICY),
        "success_gate": copy.deepcopy(_SUCCESS_GATE),
        "repository_commit": _repository_head(root),
        "source_text_sha256": {
            path.as_posix(): _text_sha256(root / path)
            for path in ROUND27_STAGE1_SOURCE_RELATIVES
        },
        "lineage": lineage,
        "authority": copy.deepcopy(_AUTHORITY),
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def validate_round27_stage1_contract(
    value: Mapping[str, object],
    *,
    repository: str | Path,
) -> Round27Stage1Contract:
    root = Path(repository).resolve()
    payload = dict(value)
    claimed = str(payload.pop("contract_sha256", "")).lower()
    raw_slots = payload.get("slots")
    source_hashes = payload.get("source_text_sha256")
    lineage = payload.get("lineage")
    if (
        _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version") != ROUND27_STAGE1_CONTRACT_SCHEMA_VERSION
        or payload.get("phase") != "model_development_stage1"
        or payload.get("asset") != "BTC"
        or payload.get("market") != "Polymarket BTC five-minute Up/Down"
        or payload.get("slot_duration_seconds") != ROUND27_STAGE1_SLOT_DURATION_SECONDS
        or payload.get("capture_scope") != _CAPTURE_SCOPE
        or payload.get("resource_policy") != _RESOURCE_POLICY
        or payload.get("campaign_policy") != _CAMPAIGN_POLICY
        or payload.get("success_gate") != _SUCCESS_GATE
        or payload.get("authority") != _AUTHORITY
        or type(payload.get("created_at_ms")) is not int
        or not isinstance(raw_slots, list)
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes)
        != {path.as_posix() for path in ROUND27_STAGE1_SOURCE_RELATIVES}
        or not isinstance(lineage, Mapping)
        or set(lineage) != set(_LINEAGE_ARTIFACTS)
        or re.fullmatch(r"[0-9a-f]{40}", str(payload.get("repository_commit", "")))
        is None
    ):
        raise ValueError("Round 27 Stage 1 contract differs")
    try:
        slots = _validate_slots(
            tuple(
                Round27Stage1Slot(
                    slot_id=str(item["slot_id"]),
                    role=str(item["role"]),
                    scheduled_start_ms=int(item["scheduled_start_ms"]),
                    scheduled_end_ms=int(item["scheduled_end_ms"]),
                )
                for item in raw_slots
                if isinstance(item, Mapping)
                and set(item)
                == {"slot_id", "role", "scheduled_start_ms", "scheduled_end_ms"}
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 27 Stage 1 slots differ") from exc
    if len(slots) != len(raw_slots):
        raise ValueError("Round 27 Stage 1 slots differ")
    normalized_sources: dict[str, str] = {}
    for relative, expected_value in source_hashes.items():
        expected = str(expected_value).lower()
        if _SHA256.fullmatch(expected) is None or expected != _text_sha256(
            root / str(relative)
        ):
            raise ValueError(f"Round 27 Stage 1 source differs: {relative}")
        normalized_sources[str(relative)] = expected
    normalized_lineage: dict[str, str] = {}
    for name, (relative, field) in _LINEAGE_ARTIFACTS.items():
        expected = _artifact_claim(root / relative, field=field, label=name)
        if lineage.get(name) != expected:
            raise ValueError(f"Round 27 Stage 1 lineage differs: {name}")
        normalized_lineage[name] = expected
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            str(payload["repository_commit"]),
            "HEAD",
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("Round 27 Stage 1 source commit is not an ancestor of HEAD")
    return Round27Stage1Contract(
        contract_sha256=claimed,
        created_at_ms=int(payload["created_at_ms"]),
        repository_commit=str(payload["repository_commit"]),
        source_text_sha256=normalized_sources,
        lineage=normalized_lineage,
        slots=slots,
    )


def load_round27_stage1_contract(
    path: str | Path,
    *,
    repository: str | Path,
) -> Round27Stage1Contract:
    return validate_round27_stage1_contract(
        _read_json(Path(path), label="Round 27 Stage 1 contract"),
        repository=repository,
    )


def write_round27_stage1_contract(
    path: str | Path, value: Mapping[str, object]
) -> None:
    write_bytes_atomic(
        Path(path),
        (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


@dataclass(frozen=True, slots=True)
class Round27Stage1SlotConfig:
    repository: Path
    contract_path: Path
    slot_id: str
    database_path: Path
    result_path: Path
    lock_path: Path

    def validated(self) -> Round27Stage1SlotConfig:
        root = self.repository.resolve()
        paths = tuple(
            path.resolve()
            for path in (
                self.contract_path,
                self.database_path,
                self.result_path,
                self.lock_path,
            )
        )
        if (
            not root.is_dir()
            or not paths[0].is_file()
            or len(set(paths)) != len(paths)
            or re.fullmatch(r"stage1-[a-z]", self.slot_id) is None
        ):
            raise ValueError("Round 27 Stage 1 slot configuration differs")
        return Round27Stage1SlotConfig(root, paths[0], self.slot_id, *paths[1:])


def _resource_block(database: Path) -> str | None:
    footprint = _database_footprint_bytes(database)
    if footprint >= (
        ROUND27_STAGE1_DATABASE_CAP_BYTES - ROUND27_STAGE1_DATABASE_STOP_RESERVE_BYTES
    ):
        return "database_cap_reserve_reached"
    database.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(database.parent).free < ROUND27_STAGE1_MINIMUM_FREE_BYTES:
        return "minimum_free_space_breached"
    return None


def _require_fresh_paths(database: Path, result: Path) -> None:
    occupied = [
        candidate
        for candidate in (
            database,
            Path(f"{database}.wal"),
            Path(f"{database}.tmp"),
            result,
        )
        if candidate.exists()
    ]
    if occupied:
        raise RuntimeError(
            "Round 27 Stage 1 slot requires fresh paths: "
            + ", ".join(str(path) for path in occupied)
        )


def _manifest(
    contract: Round27Stage1Contract,
    slot: Round27Stage1Slot,
    *,
    run_id: str,
    started_at_ms: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ROUND27_STAGE1_CONTRACT_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "run_id": run_id,
        "created_at_ms": started_at_ms,
        "phase": "model_development_stage1",
        "slot": slot.asdict(),
        "required_assets": ["BTC"],
        "required_streams": list(_CAPTURE_SCOPE["required_streams"]),
        "required_clob_lanes": list(_CAPTURE_SCOPE["required_clob_lanes"]),
        "required_rtds_topics": list(_CAPTURE_SCOPE["required_rtds_topics"]),
        "binance_futures_profile": "documented_aggregate_trades",
        "lineage": contract.lineage,
        **_AUTHORITY,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _slot_result(
    contract: Round27Stage1Contract,
    slot: Round27Stage1Slot,
    report: RecorderReport,
    source_quality: Mapping[str, object],
    *,
    footprint: int,
    resource_stop_reason: str,
) -> dict[str, object]:
    scheduled_window_observed = (
        report.started_at_ms
        >= slot.scheduled_start_ms - ROUND27_STAGE1_START_TOLERANCE_MS
        and report.started_at_ms
        <= slot.scheduled_start_ms + ROUND27_STAGE1_START_TOLERANCE_MS
        and report.ended_at_ms
        <= slot.scheduled_end_ms + ROUND27_STAGE1_END_TOLERANCE_MS
        and report.ended_at_ms
        >= slot.scheduled_end_ms - ROUND27_STAGE1_END_TOLERANCE_MS
    )
    checks = {
        "terminal_recorder_status_accepted": report.status in {"complete", "degraded"},
        "required_stream_message_count_positive": all(
            int(report.stream_counts.get(stream, 0)) > 0
            for stream in _CAPTURE_SCOPE["required_streams"]
        ),
        "recorder_error_count_zero": not report.errors,
        "integrity_error_count_zero": not report.integrity_errors,
        "documented_spot_and_futures_trade_quality_passed": source_quality.get("passed")
        is True,
        "database_footprint_at_or_below_slot_cap": footprint
        <= ROUND27_STAGE1_DATABASE_CAP_BYTES,
        "resource_stop_not_triggered": not resource_stop_reason,
        "scheduled_window_observed": scheduled_window_observed,
    }
    failures = [name for name, passed in checks.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": ROUND27_STAGE1_RESULT_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "slot": slot.asdict(),
        "status": "passed" if not failures else "failed",
        "run_id": report.run_id,
        "capture_report": report.asdict(),
        "source_quality": dict(source_quality),
        "gate_checks": checks,
        "failure_reasons": failures,
        "database_footprint_bytes": footprint,
        "database_cap_bytes": ROUND27_STAGE1_DATABASE_CAP_BYTES,
        "resource_stop_reason": resource_stop_reason,
        "analysis_policy": {
            "condition_local_target_free_audit_required": True,
            "model_data_eligible_before_audit": False,
            "target_accessed": False,
            "captured_condition_count": len(report.conditions),
            "stream_gap_count": report.stream_gap_count,
        },
        "authority": copy.deepcopy(_AUTHORITY),
        "conclusion": (
            "slot source gate passed; target-free condition audit remains required"
            if not failures
            else "slot source gate failed; this slot is inadmissible"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def validate_round27_stage1_slot_result(
    value: Mapping[str, object],
    *,
    contract: Round27Stage1Contract,
    slot: Round27Stage1Slot,
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("result_sha256", "")).lower()
    authority = payload.get("authority")
    analysis = payload.get("analysis_policy")
    if (
        _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version") != ROUND27_STAGE1_RESULT_SCHEMA_VERSION
        or payload.get("contract_sha256") != contract.contract_sha256
        or payload.get("slot") != slot.asdict()
        or payload.get("status") not in {"passed", "failed"}
        or not isinstance(authority, Mapping)
        or not isinstance(analysis, Mapping)
        or authority != _AUTHORITY
        or analysis.get("target_accessed") is not False
        or analysis.get("model_data_eligible_before_audit") is not False
    ):
        raise ValueError("Round 27 Stage 1 slot result differs")
    return {**payload, "result_sha256": claimed}


def load_round27_stage1_slot_result(
    path: str | Path,
    *,
    contract: Round27Stage1Contract,
    slot: Round27Stage1Slot,
) -> dict[str, object]:
    return validate_round27_stage1_slot_result(
        _read_json(Path(path), label="Round 27 Stage 1 slot result"),
        contract=contract,
        slot=slot,
    )


async def run_round27_stage1_slot(
    config: Round27Stage1SlotConfig,
    *,
    progress: ProgressCallback | None = None,
    recorder_factory: Callable[[Path], PolymarketPublicRecorder] = _create_recorder,
    source_audit: Callable[[Path, str], Mapping[str, object]] = _audit_sources,
    clock_ms: Callable[[], int] | None = None,
) -> dict[str, object]:
    selected = config.validated()
    contract = load_round27_stage1_contract(
        selected.contract_path,
        repository=selected.repository,
    )
    slot = contract.slot(selected.slot_id)
    now = clock_ms or (lambda: time.time_ns() // 1_000_000)
    selected.database_path.parent.mkdir(parents=True, exist_ok=True)
    selected.result_path.parent.mkdir(parents=True, exist_ok=True)
    with _CaptureFileLock(selected.lock_path):
        started = int(now())
        if not (
            slot.scheduled_start_ms
            <= started
            <= slot.scheduled_start_ms + ROUND27_STAGE1_START_TOLERANCE_MS
        ):
            raise RuntimeError("Round 27 Stage 1 slot is outside its launch tolerance")
        _require_fresh_paths(selected.database_path, selected.result_path)
        initial_block = _resource_block(selected.database_path)
        if initial_block:
            raise RuntimeError(f"Round 27 Stage 1 resource block: {initial_block}")
        resource_stop_reason = ""

        def stop_requested() -> str | None:
            nonlocal resource_stop_reason
            reason = _resource_block(selected.database_path)
            if reason:
                resource_stop_reason = reason
            return reason

        def manifest_factory(run_id: str, started_at_ms: int) -> Mapping[str, object]:
            return _manifest(
                contract,
                slot,
                run_id=run_id,
                started_at_ms=started_at_ms,
            )

        remaining_seconds = max(1, (slot.scheduled_end_ms - started) // 1_000)
        report = await recorder_factory(selected.database_path).run(
            duration_seconds=remaining_seconds,
            progress=progress,
            progress_interval_seconds=ROUND27_PROGRESS_INTERVAL_SECONDS,
            stop_requested=stop_requested,
            preregistration_manifest_factory=manifest_factory,
        )
        if progress is not None:
            progress("source-audit-started", {"run_id": report.run_id})
        try:
            quality = dict(source_audit(selected.database_path, report.run_id))
        except Exception as exc:
            quality = {"passed": False, "error": f"{type(exc).__name__}:{exc}"}
        result = _slot_result(
            contract,
            slot,
            report,
            quality,
            footprint=_database_footprint_bytes(selected.database_path),
            resource_stop_reason=resource_stop_reason,
        )
        write_bytes_atomic(
            selected.result_path,
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii"),
        )
        if progress is not None:
            progress(
                "source-audit-finalized",
                {"status": result["status"], "result_sha256": result["result_sha256"]},
            )
        return result


def _slot_paths(root: Path, slot: Round27Stage1Slot) -> dict[str, Path]:
    return {
        "database": root / f"round27-{slot.slot_id}.duckdb",
        "result": root / f"round27-{slot.slot_id}-result.json",
        "progress": root / f"round27-{slot.slot_id}-progress.json",
        "lock": root / f"round27-{slot.slot_id}.lock",
        "missed": root / f"round27-{slot.slot_id}-missed.json",
    }


def _write_state(
    path: Path,
    contract: Round27Stage1Contract,
    *,
    phase: str,
    slot: Round27Stage1Slot | None,
    detail: Mapping[str, object],
    clock_ms: Callable[[], int],
) -> None:
    payload: dict[str, object] = {
        "schema_version": ROUND27_STAGE1_STATE_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "heartbeat_wall_ms": int(clock_ms()),
        "phase": phase,
        "slot_id": None if slot is None else slot.slot_id,
        "detail": dict(detail),
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    payload["state_sha256"] = _canonical_sha256(payload)
    write_json_atomic(path, payload, indent=2, sort_keys=True)


def _write_missed(
    path: Path,
    contract: Round27Stage1Contract,
    slot: Round27Stage1Slot,
    *,
    reason: str,
    observed_ms: int,
) -> None:
    payload: dict[str, object] = {
        "schema_version": ROUND27_STAGE1_MISSED_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "slot": slot.asdict(),
        "observed_at_ms": observed_ms,
        "reason": reason,
        "model_data_eligible": False,
        "target_accessed": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    write_json_atomic(path, payload, indent=2, sort_keys=True)


def supervise_round27_stage1_primary(
    *,
    repository: str | Path,
    contract_path: str | Path,
    data_root: str | Path,
    state_path: str | Path,
    lease_path: str | Path,
    clock_ms: Callable[[], int] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    slot_runner: Callable[
        [Round27Stage1SlotConfig, ProgressCallback], Mapping[str, object]
    ]
    | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    contract_file = Path(contract_path).resolve()
    data = Path(data_root).resolve()
    state = Path(state_path).resolve()
    lease = Path(lease_path).resolve()
    contract = load_round27_stage1_contract(contract_file, repository=root)
    now = clock_ms or (lambda: time.time_ns() // 1_000_000)
    data.mkdir(parents=True, exist_ok=True)

    def default_runner(
        config: Round27Stage1SlotConfig, progress: ProgressCallback
    ) -> Mapping[str, object]:
        return asyncio.run(run_round27_stage1_slot(config, progress=progress))

    run_slot = slot_runner or default_runner
    outcomes: dict[str, str] = {}
    with _CaptureFileLock(lease):
        for slot in (item for item in contract.slots if item.role == "primary"):
            paths = _slot_paths(data, slot)
            if paths["result"].is_file():
                result = load_round27_stage1_slot_result(
                    paths["result"],
                    contract=contract,
                    slot=slot,
                )
                outcomes[slot.slot_id] = f"preexisting_{result['status']}"
                continue
            while int(now()) < slot.scheduled_start_ms:
                remaining_ms = slot.scheduled_start_ms - int(now())
                _write_state(
                    state,
                    contract,
                    phase="waiting_for_fixed_slot",
                    slot=slot,
                    detail={"remaining_seconds": max(0, remaining_ms // 1_000)},
                    clock_ms=now,
                )
                sleeper(min(ROUND27_STAGE1_HEARTBEAT_SECONDS, remaining_ms / 1_000))
            observed = int(now())
            if observed > slot.scheduled_start_ms + ROUND27_STAGE1_START_TOLERANCE_MS:
                _write_missed(
                    paths["missed"],
                    contract,
                    slot,
                    reason="supervisor_observed_slot_after_launch_tolerance",
                    observed_ms=observed,
                )
                outcomes[slot.slot_id] = "missed"
                continue

            def progress(phase: str, value: Mapping[str, object]) -> None:
                write_json_atomic(
                    paths["progress"],
                    {"phase": phase, **dict(value)},
                    indent=2,
                    sort_keys=True,
                )
                _write_state(
                    state,
                    contract,
                    phase="slot_capture_running",
                    slot=slot,
                    detail={"capture_phase": phase},
                    clock_ms=now,
                )

            try:
                result = run_slot(
                    Round27Stage1SlotConfig(
                        repository=root,
                        contract_path=contract_file,
                        slot_id=slot.slot_id,
                        database_path=paths["database"],
                        result_path=paths["result"],
                        lock_path=paths["lock"],
                    ),
                    progress,
                )
                outcomes[slot.slot_id] = str(result.get("status", "unknown"))
            except Exception as exc:
                _write_missed(
                    paths["missed"],
                    contract,
                    slot,
                    reason=f"{type(exc).__name__}:{exc}",
                    observed_ms=int(now()),
                )
                outcomes[slot.slot_id] = "failed"
        _write_state(
            state,
            contract,
            phase="awaiting_primary_target_free_audits",
            slot=None,
            detail={"outcomes": outcomes},
            clock_ms=now,
        )
    return {
        "contract_sha256": contract.contract_sha256,
        "phase": "awaiting_primary_target_free_audits",
        "outcomes": outcomes,
        "profitability_claim": False,
        "trading_authority": False,
    }


__all__ = [
    "ROUND27_STAGE1_CONTRACT_SCHEMA_VERSION",
    "ROUND27_STAGE1_DATABASE_CAP_BYTES",
    "ROUND27_STAGE1_MINIMUM_ELIGIBLE_MARKETS",
    "ROUND27_STAGE1_RESULT_SCHEMA_VERSION",
    "ROUND27_STAGE1_SLOT_DURATION_SECONDS",
    "Round27Stage1Contract",
    "Round27Stage1Slot",
    "Round27Stage1SlotConfig",
    "create_round27_stage1_contract",
    "load_round27_stage1_slot_result",
    "load_round27_stage1_contract",
    "run_round27_stage1_slot",
    "supervise_round27_stage1_primary",
    "validate_round27_stage1_contract",
    "validate_round27_stage1_slot_result",
    "write_round27_stage1_contract",
]
