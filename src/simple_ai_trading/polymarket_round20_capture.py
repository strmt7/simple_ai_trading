"""Attested capture and terminal qualification for Polymarket Round 20."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from .polymarket import PolymarketPublicClient
from .polymarket_recorder import (
    PolymarketEvidenceStore,
    PolymarketPublicRecorder,
    RawStreamMessage,
    RecorderReport,
    StreamGap,
)
from .polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
    PolymarketUnionEvent,
)
from .polymarket_round20_contract import (
    POLYMARKET_ROUND20_CONTRACT_SHA256,
    POLYMARKET_ROUND20_PARENT_RESULT_SHA256,
    PolymarketRound20Program,
)


POLYMARKET_ROUND20_CAPTURE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round20-capture-manifest-v1"
)
POLYMARKET_ROUND20_QUALIFICATION_SCHEMA_VERSION = (
    "polymarket-round20-capture-qualification-v1"
)
POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS = 1_200
POLYMARKET_ROUND20_FRESH_EVENT_SECONDS = 10
POLYMARKET_ROUND20_QUALIFICATION_WARMUP_SECONDS = 10
POLYMARKET_ROUND20_MAXIMUM_JOINT_UNHEALTHY_MS = 2_000
_ROUND20_LANES = ("clob-a", "clob-b")
_ROUND20_REQUIRED_FILES = (
    "docs/model-research/polymarket/"
    "round-020-independent-redundant-corpus-contract-v1.json",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_redundant_union.py",
    "src/simple_ai_trading/polymarket_round20_capture.py",
    "src/simple_ai_trading/polymarket_round20_contract.py",
    "tests/test_polymarket_recorder.py",
    "tests/test_polymarket_redundant_union.py",
    "tests/test_polymarket_round20_capture.py",
    "tests/test_polymarket_round20_contract.py",
    "tests/test_polymarket_round20_recorder.py",
    "tools/qualify_round20_capture.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 20 capture contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 20 capture contains {value}")


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if result.returncode:
        raise ValueError(
            "Round 20 capture Git operation failed: "
            + (result.stderr.strip() or result.stdout.strip())[:500]
        )
    return result.stdout.strip()


def _repository_attestation(repository: Path) -> tuple[str, str, dict[str, str]]:
    root = repository.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Round 20 capture requires a clean Git worktree")
    commit_oid = _git(root, "rev-parse", "HEAD").lower()
    tree_oid = _git(root, "rev-parse", "HEAD^{tree}").lower()
    files: dict[str, str] = {}
    for relative in _ROUND20_REQUIRED_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Round 20 required file is unavailable: {relative}")
        files[relative] = _file_sha256(path)
    return commit_oid, tree_oid, files


def create_round20_capture_manifest(
    *,
    run_id: str,
    created_at_ms: int,
    repository_commit_oid: str,
    repository_tree_oid: str,
    repository_file_sha256: Mapping[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND20_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "round20_contract_sha256": POLYMARKET_ROUND20_CONTRACT_SHA256,
        "parent_round19_result_sha256": POLYMARKET_ROUND20_PARENT_RESULT_SHA256,
        "run_id": str(run_id),
        "created_at_ms": int(created_at_ms),
        "capture_duration_seconds": (
            POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS
        ),
        "purpose": "storage_transport_qualification",
        "required_assets": ["BTC"],
        "required_streams": ["clob_market", "polymarket_rtds"],
        "required_clob_lanes": list(_ROUND20_LANES),
        "required_rtds_topics": ["crypto_prices_chainlink"],
        "optional_predictor_sources_captured": [],
        "binance_credentials_used": False,
        "binance_execution_connected": False,
        "labels_consulted": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "repository_commit_oid": str(repository_commit_oid).lower(),
        "repository_tree_oid": str(repository_tree_oid).lower(),
        "repository_file_sha256": dict(sorted(repository_file_sha256.items())),
        "clean_worktree_before_capture": True,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round20_capture_manifest(payload)


def validate_round20_capture_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    files = payload.get("repository_file_sha256")
    booleans_false = (
        "binance_credentials_used",
        "binance_execution_connected",
        "labels_consulted",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    expected_keys = {
        "schema_version",
        "round20_contract_sha256",
        "parent_round19_result_sha256",
        "run_id",
        "created_at_ms",
        "capture_duration_seconds",
        "purpose",
        "required_assets",
        "required_streams",
        "required_clob_lanes",
        "required_rtds_topics",
        "optional_predictor_sources_captured",
        "binance_credentials_used",
        "binance_execution_connected",
        "labels_consulted",
        "outcomes_consulted",
        "model_scores_consulted",
        "repository_commit_oid",
        "repository_tree_oid",
        "repository_file_sha256",
        "clean_worktree_before_capture",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload["schema_version"]
        != POLYMARKET_ROUND20_CAPTURE_MANIFEST_SCHEMA_VERSION
        or payload["round20_contract_sha256"]
        != POLYMARKET_ROUND20_CONTRACT_SHA256
        or payload["parent_round19_result_sha256"]
        != POLYMARKET_ROUND20_PARENT_RESULT_SHA256
        or re.fullmatch(r"[0-9a-f]{32}", str(payload["run_id"])) is None
        or type(payload["created_at_ms"]) is not int
        or payload["created_at_ms"] <= 0
        or payload["capture_duration_seconds"]
        != POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS
        or payload["purpose"] != "storage_transport_qualification"
        or payload["required_assets"] != ["BTC"]
        or payload["required_streams"]
        != ["clob_market", "polymarket_rtds"]
        or payload["required_clob_lanes"] != list(_ROUND20_LANES)
        or payload["required_rtds_topics"] != ["crypto_prices_chainlink"]
        or payload["optional_predictor_sources_captured"] != []
        or any(payload[name] is not False for name in booleans_false)
        or payload["clean_worktree_before_capture"] is not True
        or _GIT_OID.fullmatch(str(payload["repository_commit_oid"])) is None
        or _GIT_OID.fullmatch(str(payload["repository_tree_oid"])) is None
        or not isinstance(files, Mapping)
        or set(files) != set(_ROUND20_REQUIRED_FILES)
        or any(_SHA256.fullmatch(str(value)) is None for value in files.values())
    ):
        raise ValueError("Round 20 capture manifest differs")
    return {**payload, "manifest_sha256": claimed}


def build_round20_capture_manifest(
    repository: str | Path,
    *,
    run_id: str,
    created_at_ms: int,
) -> dict[str, object]:
    root = Path(repository).resolve()
    commit_oid, tree_oid, files = _repository_attestation(root)
    return create_round20_capture_manifest(
        run_id=run_id,
        created_at_ms=created_at_ms,
        repository_commit_oid=commit_oid,
        repository_tree_oid=tree_oid,
        repository_file_sha256=files,
    )


def verify_round20_repository_attestation(
    repository: str | Path,
    value: Mapping[str, object],
) -> None:
    manifest = validate_round20_capture_manifest(value)
    root = Path(repository).resolve()
    if _git(root, "rev-parse", "HEAD").lower() != manifest["repository_commit_oid"]:
        raise ValueError("Round 20 capture commit differs")
    if (
        _git(root, "rev-parse", "HEAD^{tree}").lower()
        != manifest["repository_tree_oid"]
    ):
        raise ValueError("Round 20 capture tree differs")
    files = manifest["repository_file_sha256"]
    if not isinstance(files, Mapping):
        raise AssertionError("validated Round 20 file attestation is unavailable")
    for relative, expected in files.items():
        path = root / str(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or not hmac.compare_digest(_file_sha256(path), str(expected))
        ):
            raise ValueError(f"Round 20 captured file bytes differ: {relative}")


def create_round20_recorder(
    database: str | Path,
    *,
    client: PolymarketPublicClient | None = None,
) -> PolymarketPublicRecorder:
    return PolymarketPublicRecorder(
        database,
        client=client,
        queue_capacity=100_000,
        discovery_interval_seconds=30,
        memory_limit="1GB",
        database_threads=2,
        assets=("BTC",),
        include_binance_futures=False,
        include_binance_spot=False,
        include_rtds_binance=False,
        clob_lane_ids=_ROUND20_LANES,
    )


@dataclass(frozen=True, slots=True)
class _LaneTimeline:
    event_wall_ms: tuple[int, ...]
    gaps: tuple[StreamGap, ...]


def _lane_id(connection_id: str) -> str | None:
    lane = str(connection_id).partition(":")[0]
    return lane if lane in _ROUND20_LANES else None


def _is_market_event(message: RawStreamMessage) -> bool:
    return message.raw_text != "PONG"


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


def _unhealthy_intervals(
    timeline: _LaneTimeline,
    *,
    started_at_ms: int,
    ended_at_ms: int,
) -> tuple[tuple[int, int], ...]:
    freshness_ms = POLYMARKET_ROUND20_FRESH_EVENT_SECONDS * 1_000
    events = timeline.event_wall_ms
    intervals: list[tuple[int, int]] = []
    if not events:
        return ((started_at_ms, ended_at_ms),)
    intervals.append(
        (started_at_ms, min(max(events[0], started_at_ms), ended_at_ms))
    )
    for previous, current in zip(events, events[1:], strict=False):
        intervals.append(
            (
                max(previous + freshness_ms, started_at_ms),
                min(current, ended_at_ms),
            )
        )
    intervals.append(
        (max(events[-1] + freshness_ms, started_at_ms), ended_at_ms)
    )
    for gap in timeline.gaps:
        later = next(
            (receipt for receipt in events if receipt >= gap.opened_at_ms),
            ended_at_ms,
        )
        intervals.append(
            (
                max(gap.opened_at_ms, started_at_ms),
                min(later, ended_at_ms),
            )
        )
    return _merge_intervals(intervals)


def _intersection_milliseconds(
    first: Sequence[tuple[int, int]],
    second: Sequence[tuple[int, int]],
) -> int:
    left = 0
    right = 0
    total = 0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        end = min(first[left][1], second[right][1])
        total += max(0, end - start)
        if first[left][1] <= second[right][1]:
            left += 1
        else:
            right += 1
    return total


def _chain_events(
    previous_sha256: str,
    events: Sequence[PolymarketUnionEvent],
) -> str:
    chain = previous_sha256
    for event in events:
        chain = hashlib.sha256(
            f"{chain}:{event.event_sha256}".encode("ascii")
        ).hexdigest()
    return chain


def evaluate_round20_capture(
    *,
    database: str | Path,
    report: RecorderReport,
    program: PolymarketRound20Program,
    capture_manifest_sha256: str,
) -> dict[str, object]:
    """Rebuild the exact CLOB union once and emit a non-authoritative result."""

    if (
        program.contract_sha256 != POLYMARKET_ROUND20_CONTRACT_SHA256
        or report.run_id == ""
        or not _SHA256.fullmatch(capture_manifest_sha256)
    ):
        raise ValueError("Round 20 qualification inputs differ")
    builder = PolymarketRedundantUnionBuilder(
        pairing_window_ms=program.pairing_window_ms,
        maximum_pending_events=100_000,
    )
    lane_events: dict[str, list[int]] = defaultdict(list)
    lane_connections: dict[str, set[str]] = defaultdict(set)
    union_chain = ""
    emitted_union_events = 0
    decoded_clob_receipts = 0
    rtds_chainlink_receipts = 0
    database_path = Path(database)
    with PolymarketEvidenceStore(
        database_path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for message in store.iter_terminal_capture_messages(
            report.run_id,
            streams=("clob_market", "polymarket_rtds"),
        ):
            if message.stream == "polymarket_rtds":
                if "crypto_prices_chainlink" in message.raw_text:
                    rtds_chainlink_receipts += 1
                continue
            lane = _lane_id(message.connection_id)
            if lane is None:
                raise ValueError("Round 20 CLOB receipt has an unknown lane")
            decoded_clob_receipts += 1
            lane_connections[lane].add(message.connection_id)
            if _is_market_event(message):
                lane_events[lane].append(message.received_wall_ms)
            ready = builder.add(
                PolymarketClobLaneReceipt(
                    lane_id=lane,
                    connection_id=message.connection_id,
                    sequence_number=message.sequence_number,
                    received_wall_ms=message.received_wall_ms,
                    received_monotonic_ns=message.received_monotonic_ns,
                    raw_text=message.raw_text,
                )
            )
            emitted_union_events += len(ready)
            union_chain = _chain_events(union_chain, ready)
        trailing, union_audit = builder.finish()
        emitted_union_events += len(trailing)
        union_chain = _chain_events(union_chain, trailing)
        gaps = tuple(store.iter_terminal_stream_gaps(report.run_id))
    lane_gaps = {
        lane: tuple(
            gap
            for gap in gaps
            if gap.stream == "clob_market"
            and _lane_id(gap.connection_id) == lane
        )
        for lane in _ROUND20_LANES
    }
    timelines = {
        lane: _LaneTimeline(
            event_wall_ms=tuple(lane_events[lane]),
            gaps=lane_gaps[lane],
        )
        for lane in _ROUND20_LANES
    }
    unhealthy = {
        lane: _unhealthy_intervals(
            timelines[lane],
            started_at_ms=(
                report.started_at_ms
                + POLYMARKET_ROUND20_QUALIFICATION_WARMUP_SECONDS * 1_000
            ),
            ended_at_ms=report.ended_at_ms,
        )
        for lane in _ROUND20_LANES
    }
    joint_unhealthy_ms = _intersection_milliseconds(
        unhealthy["clob-a"],
        unhealthy["clob-b"],
    )
    expected_clob_receipts = int(report.stream_counts.get("clob_market", 0))
    duration_complete = (
        report.duration_seconds
        >= POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS * 0.99
    )
    gates = {
        "duration_complete": duration_complete,
        "terminal_status_usable": report.status in {"complete", "degraded"},
        "zero_integrity_errors": not report.integrity_errors,
        "zero_recorder_errors": not report.errors,
        "exact_clob_receipt_count_reconciled": (
            decoded_clob_receipts == expected_clob_receipts
            and decoded_clob_receipts > 0
        ),
        "both_clob_lanes_observed": all(lane_events[lane] for lane in _ROUND20_LANES),
        "both_clob_lanes_connected": all(
            len(lane_connections[lane]) >= 1 for lane in _ROUND20_LANES
        ),
        "multiple_market_conditions_observed": len(report.conditions) >= 4,
        "minimum_lane_coverage": all(
            union_audit.lane_coverage_fraction[lane] >= 0.9
            for lane in _ROUND20_LANES
        ),
        "minimum_shared_fraction": union_audit.shared_fraction >= 0.9,
        "joint_unhealthy_within_limit": (
            joint_unhealthy_ms <= program.maximum_joint_unhealthy_ms
        ),
        "chainlink_rtds_observed": rtds_chainlink_receipts > 0,
        "union_count_reconciled": (
            emitted_union_events == union_audit.union_event_count
        ),
        "union_pending_bound_preserved": (
            union_audit.maximum_pending_events_observed <= 100_000
            and union_audit.terminal_pending_event_count == 0
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": POLYMARKET_ROUND20_QUALIFICATION_SCHEMA_VERSION,
        "round20_contract_sha256": program.contract_sha256,
        "parent_round19_result_sha256": program.parent_result_sha256,
        "capture_manifest_sha256": capture_manifest_sha256,
        "run_id": report.run_id,
        "recorder_report_sha256": report.report_sha256,
        "started_at_ms": report.started_at_ms,
        "ended_at_ms": report.ended_at_ms,
        "duration_seconds": report.duration_seconds,
        "recorder_status": report.status,
        "database_bytes": database_path.stat().st_size,
        "reported_clob_receipts": expected_clob_receipts,
        "decoded_clob_receipts": decoded_clob_receipts,
        "rtds_chainlink_receipts": rtds_chainlink_receipts,
        "market_condition_count": len(report.conditions),
        "emitted_union_events": emitted_union_events,
        "lane_connection_counts": {
            lane: len(lane_connections[lane]) for lane in _ROUND20_LANES
        },
        "lane_gap_counts": {
            lane: len(lane_gaps[lane]) for lane in _ROUND20_LANES
        },
        "joint_unhealthy_ms": joint_unhealthy_ms,
        "integrity_errors": list(report.integrity_errors),
        "recorder_errors": list(report.errors),
        "union_event_chain_sha256": union_chain,
        "union_audit": union_audit.as_dict(),
        "gates": gates,
        "qualified": all(gates.values()),
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return validate_round20_qualification(payload)


def validate_round20_qualification(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("result_sha256", "")).strip().lower()
    gates = payload.get("gates")
    union_audit = payload.get("union_audit")
    expected_gates = {
        "duration_complete",
        "terminal_status_usable",
        "zero_integrity_errors",
        "zero_recorder_errors",
        "exact_clob_receipt_count_reconciled",
        "both_clob_lanes_observed",
        "both_clob_lanes_connected",
        "multiple_market_conditions_observed",
        "minimum_lane_coverage",
        "minimum_shared_fraction",
        "joint_unhealthy_within_limit",
        "chainlink_rtds_observed",
        "union_count_reconciled",
        "union_pending_bound_preserved",
    }
    expected_union_keys = {
        "schema_version",
        "pairing_window_ms",
        "union_event_count",
        "shared_event_count",
        "single_lane_event_count",
        "lane_event_counts",
        "lane_coverage_fraction",
        "shared_fraction",
        "event_type_counts",
        "receipt_difference_ms",
        "maximum_pending_events_observed",
        "terminal_pending_event_count",
        "audit_sha256",
    }
    expected_keys = {
        "schema_version",
        "round20_contract_sha256",
        "parent_round19_result_sha256",
        "capture_manifest_sha256",
        "run_id",
        "recorder_report_sha256",
        "started_at_ms",
        "ended_at_ms",
        "duration_seconds",
        "recorder_status",
        "database_bytes",
        "reported_clob_receipts",
        "decoded_clob_receipts",
        "rtds_chainlink_receipts",
        "market_condition_count",
        "emitted_union_events",
        "lane_connection_counts",
        "lane_gap_counts",
        "joint_unhealthy_ms",
        "integrity_errors",
        "recorder_errors",
        "union_event_chain_sha256",
        "union_audit",
        "gates",
        "qualified",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
        "binance_credentials_used",
        "binance_execution_connected",
    }
    false_fields = (
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
        "binance_credentials_used",
        "binance_execution_connected",
    )
    union_hash_valid = False
    union_semantics_valid = False
    if isinstance(union_audit, Mapping):
        unhashed_union = dict(union_audit)
        union_claimed = str(unhashed_union.pop("audit_sha256", "")).lower()
        union_hash_valid = (
            set(union_audit) == expected_union_keys
            and _SHA256.fullmatch(union_claimed) is not None
            and hmac.compare_digest(_canonical_sha256(unhashed_union), union_claimed)
        )
        union_count = union_audit.get("union_event_count")
        shared_count = union_audit.get("shared_event_count")
        single_count = union_audit.get("single_lane_event_count")
        lane_event_counts = union_audit.get("lane_event_counts")
        lane_coverage = union_audit.get("lane_coverage_fraction")
        shared_fraction = union_audit.get("shared_fraction")
        event_type_counts = union_audit.get("event_type_counts")
        receipt_difference = union_audit.get("receipt_difference_ms")
        pending_high_water = union_audit.get(
            "maximum_pending_events_observed"
        )
        terminal_pending = union_audit.get("terminal_pending_event_count")
        numeric_receipt_differences = (
            tuple(receipt_difference.values())
            if isinstance(receipt_difference, Mapping)
            else ()
        )
        union_semantics_valid = (
            union_audit.get("schema_version")
            == "polymarket-redundant-union-audit-v1"
            and union_audit.get("pairing_window_ms") == 2_000
            and type(union_count) is int
            and union_count >= 0
            and type(shared_count) is int
            and 0 <= shared_count <= union_count
            and type(single_count) is int
            and single_count == union_count - shared_count
            and isinstance(lane_event_counts, Mapping)
            and set(lane_event_counts) == set(_ROUND20_LANES)
            and all(
                type(count) is int and 0 <= count <= union_count
                for count in lane_event_counts.values()
            )
            and isinstance(lane_coverage, Mapping)
            and set(lane_coverage) == set(_ROUND20_LANES)
            and all(
                type(value) is float and math.isfinite(value)
                for value in lane_coverage.values()
            )
            and all(
                math.isclose(
                    float(lane_coverage[lane]),
                    (
                        0.0
                        if union_count == 0
                        else int(lane_event_counts[lane]) / union_count
                    ),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for lane in _ROUND20_LANES
            )
            and type(shared_fraction) is float
            and math.isfinite(shared_fraction)
            and math.isclose(
                shared_fraction,
                0.0 if union_count == 0 else shared_count / union_count,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and isinstance(event_type_counts, Mapping)
            and all(
                isinstance(name, str)
                and name
                and type(count) is int
                and count >= 0
                for name, count in event_type_counts.items()
            )
            and sum(event_type_counts.values()) == union_count
            and isinstance(receipt_difference, Mapping)
            and set(receipt_difference) == {"median", "p95", "maximum"}
            and all(
                value is None
                or (
                    type(value) is float
                    and math.isfinite(value)
                    and value >= 0.0
                )
                for value in numeric_receipt_differences
            )
            and type(pending_high_water) is int
            and 0 <= pending_high_water <= 100_000
            and type(terminal_pending) is int
            and terminal_pending >= 0
        )
    lane_connections = payload.get("lane_connection_counts")
    lane_gaps = payload.get("lane_gap_counts")
    integrity_errors = payload.get("integrity_errors")
    recorder_errors = payload.get("recorder_errors")
    duration = payload.get("duration_seconds")
    duration_valid = (
        type(duration) is float
        and math.isfinite(duration)
        and duration >= 0.0
        and type(payload.get("started_at_ms")) is int
        and type(payload.get("ended_at_ms")) is int
        and math.isclose(
            duration,
            (payload["ended_at_ms"] - payload["started_at_ms"]) / 1_000,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    expected_gate_values: dict[str, bool] | None = None
    if (
        union_semantics_valid
        and isinstance(lane_connections, Mapping)
        and set(lane_connections) == set(_ROUND20_LANES)
        and all(
            type(value) is int and value >= 0
            for value in lane_connections.values()
        )
        and isinstance(integrity_errors, list)
        and isinstance(recorder_errors, list)
    ):
        lane_event_counts = union_audit["lane_event_counts"]  # type: ignore[index]
        lane_coverage = union_audit["lane_coverage_fraction"]  # type: ignore[index]
        expected_gate_values = {
            "duration_complete": bool(
                duration_valid
                and duration
                >= POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS * 0.99
            ),
            "terminal_status_usable": (
                payload.get("recorder_status") in {"complete", "degraded"}
            ),
            "zero_integrity_errors": not integrity_errors,
            "zero_recorder_errors": not recorder_errors,
            "exact_clob_receipt_count_reconciled": (
                payload.get("decoded_clob_receipts")
                == payload.get("reported_clob_receipts")
                and type(payload.get("decoded_clob_receipts")) is int
                and payload["decoded_clob_receipts"] > 0
            ),
            "both_clob_lanes_observed": all(
                int(lane_event_counts[lane]) > 0 for lane in _ROUND20_LANES
            ),
            "both_clob_lanes_connected": all(
                int(lane_connections[lane]) >= 1 for lane in _ROUND20_LANES
            ),
            "multiple_market_conditions_observed": (
                type(payload.get("market_condition_count")) is int
                and payload["market_condition_count"] >= 4
            ),
            "minimum_lane_coverage": all(
                float(lane_coverage[lane]) >= 0.9 for lane in _ROUND20_LANES
            ),
            "minimum_shared_fraction": (
                float(union_audit["shared_fraction"]) >= 0.9
            ),
            "joint_unhealthy_within_limit": (
                type(payload.get("joint_unhealthy_ms")) is int
                and payload["joint_unhealthy_ms"]
                <= POLYMARKET_ROUND20_MAXIMUM_JOINT_UNHEALTHY_MS
            ),
            "chainlink_rtds_observed": (
                type(payload.get("rtds_chainlink_receipts")) is int
                and payload["rtds_chainlink_receipts"] > 0
            ),
            "union_count_reconciled": (
                payload.get("emitted_union_events")
                == union_audit["union_event_count"]
            ),
            "union_pending_bound_preserved": (
                union_audit["maximum_pending_events_observed"] <= 100_000
                and union_audit["terminal_pending_event_count"] == 0
            ),
        }
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or payload["schema_version"]
        != POLYMARKET_ROUND20_QUALIFICATION_SCHEMA_VERSION
        or payload["round20_contract_sha256"]
        != POLYMARKET_ROUND20_CONTRACT_SHA256
        or payload["parent_round19_result_sha256"]
        != POLYMARKET_ROUND20_PARENT_RESULT_SHA256
        or not isinstance(gates, Mapping)
        or set(gates) != expected_gates
        or any(type(gate) is not bool for gate in gates.values())
        or payload["qualified"] is not all(gates.values())
        or any(payload[field] is not False for field in false_fields)
        or not union_hash_valid
        or not union_semantics_valid
        or _SHA256.fullmatch(str(payload["capture_manifest_sha256"])) is None
        or _SHA256.fullmatch(str(payload["recorder_report_sha256"])) is None
        or _SHA256.fullmatch(str(payload["union_event_chain_sha256"])) is None
        or re.fullmatch(r"[0-9a-f]{32}", str(payload["run_id"])) is None
        or payload["recorder_status"] not in {"complete", "degraded"}
        or not duration_valid
        or payload["ended_at_ms"] < payload["started_at_ms"]
        or type(payload["database_bytes"]) is not int
        or payload["database_bytes"] <= 0
        or type(payload["reported_clob_receipts"]) is not int
        or payload["reported_clob_receipts"] < 0
        or type(payload["decoded_clob_receipts"]) is not int
        or payload["decoded_clob_receipts"] < 0
        or type(payload["rtds_chainlink_receipts"]) is not int
        or payload["rtds_chainlink_receipts"] < 0
        or type(payload["market_condition_count"]) is not int
        or payload["market_condition_count"] < 0
        or type(payload["emitted_union_events"]) is not int
        or payload["emitted_union_events"] < 0
        or type(payload["joint_unhealthy_ms"]) is not int
        or payload["joint_unhealthy_ms"] < 0
        or not isinstance(lane_connections, Mapping)
        or set(lane_connections) != set(_ROUND20_LANES)
        or any(
            type(value) is not int or value < 0
            for value in lane_connections.values()
        )
        or not isinstance(lane_gaps, Mapping)
        or set(lane_gaps) != set(_ROUND20_LANES)
        or any(type(value) is not int or value < 0 for value in lane_gaps.values())
        or not isinstance(integrity_errors, list)
        or any(not isinstance(error, str) for error in integrity_errors)
        or not isinstance(recorder_errors, list)
        or any(not isinstance(error, str) for error in recorder_errors)
        or expected_gate_values is None
        or dict(gates) != expected_gate_values
    ):
        raise ValueError("Round 20 qualification result differs")
    return {**payload, "result_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND20_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS",
    "POLYMARKET_ROUND20_QUALIFICATION_SCHEMA_VERSION",
    "build_round20_capture_manifest",
    "create_round20_capture_manifest",
    "create_round20_recorder",
    "evaluate_round20_capture",
    "validate_round20_capture_manifest",
    "validate_round20_qualification",
    "verify_round20_repository_attestation",
]
