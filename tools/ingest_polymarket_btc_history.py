#!/usr/bin/env python3
"""Resume checksummed BTC-only one-second corpus ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_btc_history import (  # noqa: E402
    POLYMARKET_BTC_HISTORY_RESEARCH_ROUND,
    POLYMARKET_BTC_HISTORY_SYMBOLS,
    load_polymarket_btc_history_contract,
)
from simple_ai_trading.spot_perpetual_corpus import (  # noqa: E402
    SpotPerpetualCorpusStore,
)


DEFAULT_INVENTORY = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-015-btc-5m-full-history-inventory-v1.json"
)
DEFAULT_DATABASE = ROOT / "data" / "polymarket-btc-flow-history-v1.duckdb"
DEFAULT_CACHE = ROOT / "data" / "archive-cache-polymarket-btc"


def _emit(event: str, **payload: object) -> None:
    print(
        json.dumps(
            {"event": event, "observed_at_ms": time.time_ns() // 1_000_000, **payload},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest the frozen BTC-only archive inventory one resumable UTC "
            "day at a time; raw ZIPs are deleted after each atomic commit."
        )
    )
    parser.add_argument("phase", choices=("status", "run"))
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--maximum-days", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=4)
    values = parser.parse_args()
    if values.maximum_days < 0:
        parser.error("--maximum-days cannot be negative")
    return values


def _status(
    store: SpotPerpetualCorpusStore,
    *,
    expected_days: int,
) -> dict[str, object]:
    row = store.connect().execute(
        """
        SELECT count(*)::UBIGINT,
               coalesce(sum(flow_rows), 0)::UBIGINT,
               coalesce(sum(compressed_bytes), 0)::UBIGINT,
               min(period),
               max(period)
        FROM spot_perpetual_flow_day_manifest
        WHERE research_round = ? AND status = 'complete' AND is_current
        """,
        [POLYMARKET_BTC_HISTORY_RESEARCH_ROUND],
    ).fetchone()
    completed = int(row[0])
    return {
        "expected_days": expected_days,
        "completed_days": completed,
        "remaining_days": expected_days - completed,
        "flow_rows": int(row[1]),
        "compressed_source_bytes": int(row[2]),
        "first_completed_day": row[3],
        "last_completed_day": row[4],
        "raw_archive_retained": False,
    }


def main() -> int:
    args = _arguments()
    contract = load_polymarket_btc_history_contract(args.inventory)
    with SpotPerpetualCorpusStore(
        args.database,
        cache_root=args.cache_root,
        memory_limit=args.memory_limit,
        threads=args.threads,
        symbols=POLYMARKET_BTC_HISTORY_SYMBOLS,
        research_round=POLYMARKET_BTC_HISTORY_RESEARCH_ROUND,
    ) as store:
        _emit("history_status", **_status(store, expected_days=len(contract.days)))
        if args.phase == "status":
            return 0
        processed = 0
        last_progress_at = 0.0
        last_phase = ""

        def progress(phase: str, current: int, total: int | None) -> None:
            nonlocal last_progress_at, last_phase
            now = time.monotonic()
            if phase == last_phase and now - last_progress_at < 2.0:
                return
            last_phase = phase
            last_progress_at = now
            _emit(
                "history_progress",
                phase=phase,
                current=int(current),
                total=None if total is None else int(total),
            )

        for index, day in enumerate(contract.days):
            if args.maximum_days and processed >= args.maximum_days:
                break
            result = store.ingest_day(
                day,
                inventory_sha256=contract.inventory_sha256,
                timeout_seconds=args.timeout_seconds,
                progress=progress,
            )
            if result.status == "skipped":
                continue
            processed += 1
            _emit(
                "history_day_complete",
                day_index=index,
                period=result.period,
                source_count=result.source_count,
                flow_rows=result.flow_rows,
                compressed_bytes=result.compressed_bytes,
                combined_flow_sha256=result.combined_flow_sha256,
            )
        status = _status(store, expected_days=len(contract.days))
        _emit("history_run_complete", processed_days=processed, **status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
