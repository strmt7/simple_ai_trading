#!/usr/bin/env python3
"""Resume checksummed BTC-only one-second corpus ingestion."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
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
DEFAULT_STATUS_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-015-btc-5m-history-ingestion-status.json"
)


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
    parser.add_argument("phase", choices=("status", "run", "publish-status"))
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--maximum-days", type=int, default=0)
    parser.add_argument(
        "--period",
        default="",
        help="ingest exactly one frozen YYYY-MM-DD period",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    values = parser.parse_args()
    if values.maximum_days < 0:
        parser.error("--maximum-days cannot be negative")
    if values.period and values.maximum_days:
        parser.error("--period and --maximum-days are mutually exclusive")
    return values


def _status(
    store: SpotPerpetualCorpusStore,
    *,
    expected_days: int,
    inventory_sha256: str,
) -> dict[str, object]:
    row = store.connect().execute(
        """
        SELECT count(*)::UBIGINT,
               coalesce(sum(flow_rows), 0)::UBIGINT,
               coalesce(sum(compressed_bytes), 0)::UBIGINT,
               min(period),
               max(period)
        FROM spot_perpetual_flow_day_manifest
        WHERE research_round = ? AND inventory_sha256 = ?
          AND status = 'complete' AND is_current
        """,
        [POLYMARKET_BTC_HISTORY_RESEARCH_ROUND, inventory_sha256],
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


def _latest_batch(
    store: SpotPerpetualCorpusStore,
    *,
    inventory_sha256: str,
) -> tuple[dict[str, object], ...]:
    rows = store.connect().execute(
        """
        SELECT period, source_count, flow_rows, compressed_bytes,
               combined_flow_sha256
        FROM spot_perpetual_flow_day_manifest
        WHERE research_round = ? AND inventory_sha256 = ?
          AND status = 'complete' AND is_current
        ORDER BY period DESC
        LIMIT 3
        """,
        [POLYMARKET_BTC_HISTORY_RESEARCH_ROUND, inventory_sha256],
    ).fetchall()
    return tuple(
        {
            "period": str(row[0]),
            "source_count": int(row[1]),
            "flow_rows": int(row[2]),
            "compressed_bytes": int(row[3]),
            "combined_flow_sha256": str(row[4]),
        }
        for row in reversed(rows)
    )


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _repository_path(path: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the repository") from exc


def _status_artifact(
    *,
    status: dict[str, object],
    latest_batch: tuple[dict[str, object], ...],
    inventory_path: Path,
    inventory_sha256: str,
    database_path: Path,
    cache_root: Path,
    generated_at_utc: str,
) -> dict[str, object]:
    completed = int(status["completed_days"])
    expected = int(status["expected_days"])
    remaining = int(status["remaining_days"])
    flow_rows = int(status["flow_rows"])
    if completed + remaining != expected:
        raise ValueError("BTC history status day arithmetic differs")
    if flow_rows != completed * 86_400:
        raise ValueError("BTC history status row arithmetic differs")
    expected_latest = min(3, completed)
    if len(latest_batch) != expected_latest:
        raise ValueError(
            "BTC history latest batch count differs from completed days"
        )
    periods = tuple(str(item["period"]) for item in latest_batch)
    if periods != tuple(sorted(periods)) or len(periods) != len(set(periods)):
        raise ValueError("BTC history latest batch chronology differs")
    if periods and periods[-1] != str(status["last_completed_day"]):
        raise ValueError("BTC history latest batch does not reach status end")
    for item in latest_batch:
        digest = str(item["combined_flow_sha256"])
        if (
            int(item["source_count"]) != 2
            or int(item["flow_rows"]) != 86_400
            or int(item["compressed_bytes"]) <= 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("BTC history latest batch manifest differs")

    cache_files = (
        sum(1 for path in cache_root.rglob("*") if path.is_file())
        if cache_root.exists()
        else 0
    )
    if cache_files != 0:
        raise ValueError("BTC history archive cache is not empty")
    database_bytes = database_path.stat().st_size
    if database_bytes <= 0:
        raise ValueError("BTC history database is empty")

    artifact: dict[str, object] = {
        "schema_version": "polymarket-btc-flow-history-ingestion-status-v1",
        "generated_at_utc": generated_at_utc,
        "inventory_path": _repository_path(inventory_path, label="inventory"),
        "inventory_sha256": inventory_sha256,
        "local_database_path": _repository_path(database_path, label="database"),
        "local_database_tracked": False,
        "database_bytes": database_bytes,
        "expected_days": expected,
        "completed_days": completed,
        "remaining_days": remaining,
        "flow_rows": flow_rows,
        "compressed_source_bytes": int(status["compressed_source_bytes"]),
        "first_completed_day": status["first_completed_day"],
        "last_completed_day": status["last_completed_day"],
        "raw_archive_retained": False,
        "archive_cache_files": cache_files,
        "status_probe_changed_database_mtime": False,
        "latest_batch": list(latest_batch),
        "stderr_bytes": 0,
        "dataset_frozen": False,
        "model_training_started": False,
        "test_targets_accessed": False,
        "profitability_claim": False,
        "paper_authority": False,
        "live_authority": False,
    }
    return {**artifact, "artifact_sha256": _canonical_sha256(artifact)}


def _write_status_artifact(path: Path, artifact: dict[str, object]) -> None:
    if path.is_symlink():
        raise ValueError("BTC history status output cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact,
        ensure_ascii=True,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_status_destination(
    *,
    output: Path,
    inventory: Path,
    database: Path,
    cache_root: Path,
) -> None:
    if output.resolve() != DEFAULT_STATUS_OUTPUT.resolve():
        return
    actual = tuple(
        path.resolve() for path in (inventory, database, cache_root)
    )
    expected = tuple(
        path.resolve()
        for path in (DEFAULT_INVENTORY, DEFAULT_DATABASE, DEFAULT_CACHE)
    )
    if actual != expected:
        raise ValueError(
            "canonical BTC history status requires canonical inventory, "
            "database, and cache paths"
        )


def main() -> int:
    args = _arguments()
    if args.phase in {"run", "publish-status"}:
        _validate_status_destination(
            output=args.status_output,
            inventory=args.inventory,
            database=args.database,
            cache_root=args.cache_root,
        )
    contract = load_polymarket_btc_history_contract(args.inventory)
    if args.period:
        selected_days = tuple(
            day for day in contract.days if day.period == str(args.period)
        )
        if len(selected_days) != 1:
            raise ValueError("--period is outside the frozen BTC history")
    else:
        selected_days = contract.days
    processed = 0
    publication_status: dict[str, object] | None = None
    publication_batch: tuple[dict[str, object], ...] = ()
    with SpotPerpetualCorpusStore(
        args.database,
        cache_root=args.cache_root,
        memory_limit=args.memory_limit,
        threads=args.threads,
        symbols=POLYMARKET_BTC_HISTORY_SYMBOLS,
        research_round=POLYMARKET_BTC_HISTORY_RESEARCH_ROUND,
        read_only=args.phase in {"status", "publish-status"},
    ) as store:
        current_status = _status(
            store,
            expected_days=len(contract.days),
            inventory_sha256=contract.inventory_sha256,
        )
        _emit(
            "history_status",
            **current_status,
        )
        if args.phase == "status":
            return 0
        if args.phase == "publish-status":
            publication_status = current_status
            publication_batch = _latest_batch(
                store,
                inventory_sha256=contract.inventory_sha256,
            )
        else:
            processed = 0
        last_progress_at = 0.0
        last_phase = ""
        active_period = ""

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
                period=active_period,
                current=int(current),
                total=None if total is None else int(total),
            )

        if args.phase == "run":
            for day in selected_days:
                if args.maximum_days and processed >= args.maximum_days:
                    break
                active_period = day.period
                inventory_index = contract.days.index(day)
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
                    day_index=inventory_index,
                    period=result.period,
                    source_count=result.source_count,
                    flow_rows=result.flow_rows,
                    compressed_bytes=result.compressed_bytes,
                    combined_flow_sha256=result.combined_flow_sha256,
                )
            publication_status = _status(
                store,
                expected_days=len(contract.days),
                inventory_sha256=contract.inventory_sha256,
            )
            publication_batch = _latest_batch(
                store,
                inventory_sha256=contract.inventory_sha256,
            )
            _emit(
                "history_run_complete",
                processed_days=processed,
                **publication_status,
            )
    if publication_status is not None and (
        args.phase == "publish-status" or processed > 0
    ):
        generated_at = (
            datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        artifact = _status_artifact(
            status=publication_status,
            latest_batch=publication_batch,
            inventory_path=args.inventory,
            inventory_sha256=contract.inventory_sha256,
            database_path=args.database,
            cache_root=args.cache_root,
            generated_at_utc=generated_at,
        )
        _write_status_artifact(args.status_output, artifact)
        _emit(
            "history_status_published",
            artifact_sha256=artifact["artifact_sha256"],
            completed_days=artifact["completed_days"],
            output=str(args.status_output),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
