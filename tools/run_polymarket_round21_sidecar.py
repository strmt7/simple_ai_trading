from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Mapping

from simple_ai_trading.polymarket_round21_sidecar import (
    POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES,
    POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES,
    POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
    build_round21_sidecar_manifest,
    create_round21_sidecar_recorder,
    round21_sidecar_state,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _resource_snapshot(database: Path) -> tuple[int, int, int]:
    database_bytes = _size(database)
    wal_bytes = _size(Path(f"{database}.wal"))
    free_bytes = shutil.disk_usage(database.parent.resolve()).free
    return database_bytes, wal_bytes, free_bytes


async def _run(arguments: argparse.Namespace) -> int:
    repository = Path(arguments.repository).resolve()
    database = Path(arguments.database).resolve()
    state_path = Path(arguments.state).resolve()
    scheduled_end_ms = int(arguments.scheduled_end_ms)
    if scheduled_end_ms != POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS:
        raise ValueError("Round 21 sidecar scheduled end differs")
    if database.exists() or Path(f"{database}.wal").exists():
        raise ValueError("Round 21 sidecar database already exists")
    database.parent.mkdir(parents=True, exist_ok=True)
    started_at_ms = int(time.time() * 1_000)
    duration_seconds = int(
        math.floor((scheduled_end_ms - started_at_ms) / 1_000)
    )
    if not 5 <= duration_seconds <= 30 * 86_400:
        raise ValueError("Round 21 sidecar duration is outside the frozen window")
    recorder = create_round21_sidecar_recorder(database)

    def stop_requested() -> str | None:
        database_bytes, wal_bytes, free_bytes = _resource_snapshot(database)
        if database_bytes + wal_bytes > POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES:
            return "database_and_wal_cap_exceeded"
        if free_bytes < POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES:
            return "minimum_free_space_crossed"
        return None

    def progress(phase: str, details: Mapping[str, object]) -> None:
        database_bytes, wal_bytes, free_bytes = _resource_snapshot(database)
        state = round21_sidecar_state(
            phase=phase,
            observed_at_ms=int(details.get("observed_at_ms", time.time() * 1_000)),
            database_bytes=database_bytes,
            wal_bytes=wal_bytes,
            free_bytes=free_bytes,
            details=details,
        )
        _write_json_atomic(state_path, state)
        print(_canonical_json(state), flush=True)

    def manifest_factory(run_id: str, created_at_ms: int) -> Mapping[str, object]:
        return build_round21_sidecar_manifest(
            repository,
            run_id=run_id,
            created_at_ms=created_at_ms,
            capture_duration_seconds=duration_seconds,
            scheduled_end_ms=scheduled_end_ms,
        )

    report = await recorder.run(
        duration_seconds=duration_seconds,
        progress=progress,
        progress_interval_seconds=30,
        stop_requested=stop_requested,
        preregistration_manifest_factory=manifest_factory,
    )
    return 0 if report.status in {"complete", "degraded"} else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the independent Round 21 public Binance predictor sidecar."
    )
    parser.add_argument(
        "--repository",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--database",
        default="data/round21-binance-sidecar.duckdb",
    )
    parser.add_argument(
        "--state",
        default="data/round21-binance-sidecar-state.json",
    )
    parser.add_argument(
        "--scheduled-end-ms",
        type=int,
        default=POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
    )
    return parser


def main() -> int:
    try:
        return asyncio.run(_run(_parser().parse_args()))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Round 21 sidecar failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
