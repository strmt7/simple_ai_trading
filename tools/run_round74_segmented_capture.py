"""Run one exact Round 74 transport unit with bounded progress heartbeats."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from simple_ai_trading.round74_segmented_capture import (
    Round74SegmentedCaptureConfig,
    capture_round74_segmented,
)


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


async def _capture_with_progress(
    config: Round74SegmentedCaptureConfig,
    *,
    progress_interval_seconds: float,
) -> object:
    database = Path(config.database)
    wal = Path(f"{database}.wal")
    task = asyncio.create_task(capture_round74_segmented(config))
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        while not task.done():
            done, _pending = await asyncio.wait(
                (task,),
                timeout=progress_interval_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if done:
                break
            elapsed = loop.time() - started
            print(
                "round74-segmented-capture-progress: "
                f"state={'finalizing' if elapsed >= config.duration_seconds else 'running'} "
                f"wall_elapsed={elapsed:.1f}s "
                f"stream_target={config.duration_seconds:.1f}s "
                f"database_bytes={_path_bytes(database)} "
                f"wal_bytes={_path_bytes(wal)}",
                file=sys.stderr,
                flush=True,
            )
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one zero-reconnect Round 74 transport unit.",
    )
    parser.add_argument("--database", required=True)
    parser.add_argument(
        "--database-size-cap-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
    )
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--database-threads", type=int, default=2)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    interval = float(args.progress_interval_seconds)
    if not 5.0 <= interval <= 120.0:
        print(
            "round74-segmented-capture failed: progress interval must be "
            "between 5 and 120 seconds",
            file=sys.stderr,
        )
        return 2
    config = Round74SegmentedCaptureConfig(
        database=str(args.database),
        database_size_cap_bytes=int(args.database_size_cap_bytes),
        duckdb_memory_limit=str(args.memory_limit),
        duckdb_threads=int(args.database_threads),
    )
    try:
        config.validate()
        report = asyncio.run(
            _capture_with_progress(
                config,
                progress_interval_seconds=interval,
            )
        )
    except KeyboardInterrupt:
        print("round74-segmented-capture cancelled", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"round74-segmented-capture failed: {exc}", file=sys.stderr)
        return 2
    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "round74-segmented-capture: "
            f"status={report.status} "
            f"run_id={report.selected_run_id or 'none'} "
            f"attempts={report.attempt_count} "
            f"reconnects={report.reconnect_count}"
        )
    return 0 if report.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
