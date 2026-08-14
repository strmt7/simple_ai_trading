"""Target-free, condition-isolated integrity audit for Polymarket CLOB replay."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay, PolymarketRecordedBook
from .storage import write_bytes_atomic


POLYMARKET_CONDITION_REPLAY_AUDIT_SCHEMA_VERSION = (
    "polymarket-condition-replay-audit-v1"
)
POLYMARKET_CONDITION_REPLAY_MINIMUM_INTERVAL_MS = 250


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


def _failure_class(error: ValueError) -> str:
    message = str(error).lower()
    if "checksum" in message or "best bid/ask" in message:
        return "book_integrity"
    if "causal reorder" in message or "receipt clock" in message:
        return "causal_order"
    if "baseline" in message or "active tick" in message:
        return "segment_baseline"
    if "integrity" in message or "manifest" in message or "hash" in message:
        return "storage_integrity"
    return "replay_validation"


def _segment_intervals(
    books: tuple[PolymarketRecordedBook, ...],
    *,
    event_start_ms: int,
    end_ms: int,
    connection_last_wall_ms: Mapping[str, int],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[PolymarketRecordedBook]] = defaultdict(list)
    for book in books:
        grouped[(book.segment_id, book.connection_id)].append(book)
    intervals: list[dict[str, object]] = []
    for (segment_id, connection_id), values in sorted(grouped.items()):
        first_by_outcome: dict[str, int] = {}
        book_count_by_outcome: dict[str, int] = defaultdict(int)
        for book in values:
            first_by_outcome[book.outcome] = min(
                first_by_outcome.get(book.outcome, book.received_wall_ms),
                book.received_wall_ms,
            )
            book_count_by_outcome[book.outcome] += 1
        has_two_outcome_baseline = set(first_by_outcome) == {"Up", "Down"}
        interval_start = (
            max(event_start_ms, *first_by_outcome.values())
            if has_two_outcome_baseline
            else None
        )
        connection_end = connection_last_wall_ms.get(connection_id)
        interval_end = (
            None if connection_end is None else min(end_ms - 1, connection_end)
        )
        interval_duration_ms = (
            interval_end - interval_start
            if interval_start is not None and interval_end is not None
            else 0
        )
        eligible = bool(
            has_two_outcome_baseline
            and interval_start is not None
            and interval_end is not None
            and interval_duration_ms
            >= POLYMARKET_CONDITION_REPLAY_MINIMUM_INTERVAL_MS
        )
        intervals.append(
            {
                "segment_id": segment_id,
                "connection_id": connection_id,
                "has_two_outcome_baseline": has_two_outcome_baseline,
                "book_count_by_outcome": dict(sorted(book_count_by_outcome.items())),
                "interval_start_ms": interval_start,
                "interval_end_ms": interval_end,
                "interval_duration_ms": max(0, interval_duration_ms),
                "eligible": eligible,
            }
        )
    return intervals


def audit_polymarket_condition_replay(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    condition_ids: tuple[str, ...] | None = None,
    build_condition_cache: bool = False,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Audit each condition independently without loading resolution labels."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("condition replay audit requires a run ID")
    run = store.connect().execute(
        """
        SELECT status, error, report_sha256, started_at_ms, ended_at_ms
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
        [selected],
    ).fetchone()
    if run is None:
        raise ValueError("unknown condition replay audit run")
    if str(run[0]) not in {"complete", "degraded"} or str(run[1] or "").strip():
        raise ValueError("condition replay audit requires a finished error-free run")
    if run[4] is None:
        raise ValueError("condition replay audit run has no terminal timestamp")
    integrity_errors = store.resume_integrity_errors(
        selected,
        progress=progress,
        progress_interval_seconds=10,
    )
    if integrity_errors:
        raise ValueError(
            "condition replay audit storage integrity failed: "
            + "; ".join(integrity_errors)
        )
    markets = PolymarketEvidenceReplay.load_markets(store, run_id=selected)
    if condition_ids is not None:
        requested = tuple(
            sorted({str(condition or "").strip().lower() for condition in condition_ids})
        )
        if not requested or any(not condition for condition in requested):
            raise ValueError("condition replay audit selection is invalid")
        available = {market.condition_id for market in markets}
        if not set(requested).issubset(available):
            raise ValueError("condition replay audit selection is unknown")
        markets = tuple(
            market for market in markets if market.condition_id in set(requested)
        )
    cache = store.connect().execute(
        """
        SELECT state FROM polymarket_condition_cache_build WHERE run_id = ?
        """,
        [selected],
    ).fetchone()
    if not build_condition_cache and (
        cache is None or str(cache[0]) != "complete"
    ):
        raise ValueError(
            "condition replay audit requires an existing condition cache; "
            "construction must be explicitly enabled"
        )
    store.ensure_condition_message_cache(
        selected,
        condition_ids=tuple(market.condition_id for market in markets),
        progress=progress,
    )
    lane_summaries = store.raw_message_lane_summaries(
        selected,
        streams=("clob_market",),
    )
    connection_last_wall_ms = {
        summary.connection_id: int(summary.last_received_wall_ms)
        for summary in lane_summaries
    }
    gap_count = int(
        store.connect().execute(
            "SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?",
            [selected],
        ).fetchone()[0]
    )
    conditions: list[dict[str, object]] = []
    ordered_markets = tuple(sorted(markets, key=lambda item: item.event_start_ms))
    for index, market in enumerate(ordered_markets, start=1):
        try:
            replay = PolymarketEvidenceReplay.load(
                store,
                run_id=selected,
                allow_segmented_gaps=True,
                include_resolutions=False,
                book_sample_interval_ms=50,
                condition_ids=(market.condition_id,),
                maximum_received_wall_ms_by_condition={
                    market.condition_id: market.end_ms - 1
                },
                materialized_minimum_depth_levels=1,
                cap_materialized_depth_to_minimum_order_size=True,
            )
        except ValueError as exc:
            conditions.append(
                {
                    "condition_id": market.condition_id,
                    "slug": market.slug,
                    "event_start_ms": market.event_start_ms,
                    "end_ms": market.end_ms,
                    "eligible": False,
                    "failure_class": _failure_class(exc),
                    "failure": str(exc),
                    "segments": [],
                }
            )
            if progress is not None:
                progress(
                    "condition-replay",
                    {
                        "completed_condition_count": index,
                        "condition_count": len(ordered_markets),
                        "condition_id": market.condition_id,
                        "eligible": False,
                    },
                )
            continue
        intervals = _segment_intervals(
            replay.books,
            event_start_ms=market.event_start_ms,
            end_ms=market.end_ms,
            connection_last_wall_ms=connection_last_wall_ms,
        )
        eligible = any(bool(item["eligible"]) for item in intervals)
        conditions.append(
            {
                "condition_id": market.condition_id,
                "slug": market.slug,
                "event_start_ms": market.event_start_ms,
                "end_ms": market.end_ms,
                "eligible": eligible,
                "failure_class": None if eligible else "no_executable_interval",
                "failure": None,
                "materialized_book_count": len(replay.books),
                "replay_diagnostics": replay.diagnostics.asdict(),
                "segments": intervals,
            }
        )
        if progress is not None:
            progress(
                "condition-replay",
                {
                    "completed_condition_count": index,
                    "condition_count": len(ordered_markets),
                    "condition_id": market.condition_id,
                    "eligible": eligible,
                },
            )
    eligible_ids = [
        str(item["condition_id"]) for item in conditions if bool(item["eligible"])
    ]
    failed_ids = [
        str(item["condition_id"]) for item in conditions if not bool(item["eligible"])
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_CONDITION_REPLAY_AUDIT_SCHEMA_VERSION,
        "run_id": selected,
        "run_report_sha256": str(run[2]),
        "run_status": str(run[0]),
        "run_started_at_ms": int(run[3]),
        "run_ended_at_ms": int(run[4]),
        "stream_gap_count": gap_count,
        "target_free": True,
        "minimum_executable_interval_ms": (
            POLYMARKET_CONDITION_REPLAY_MINIMUM_INTERVAL_MS
        ),
        "condition_count": len(conditions),
        "eligible_condition_count": len(eligible_ids),
        "failed_condition_count": len(failed_ids),
        "eligible_condition_ids": eligible_ids,
        "failed_condition_ids": failed_ids,
        "conditions": conditions,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    return {**body, "audit_sha256": _canonical_sha256(body)}


def write_polymarket_condition_replay_audit(
    output_path: str | Path,
    report: Mapping[str, object],
) -> None:
    body = dict(report)
    claimed = str(body.pop("audit_sha256", "")).lower()
    if claimed != _canonical_sha256(body):
        raise ValueError("condition replay audit hash differs")
    write_bytes_atomic(
        Path(output_path),
        (json.dumps(dict(report), indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


__all__ = [
    "POLYMARKET_CONDITION_REPLAY_AUDIT_SCHEMA_VERSION",
    "POLYMARKET_CONDITION_REPLAY_MINIMUM_INTERVAL_MS",
    "audit_polymarket_condition_replay",
    "write_polymarket_condition_replay_audit",
]
