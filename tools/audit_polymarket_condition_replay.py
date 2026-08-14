"""Build and publish a target-free condition-isolated Polymarket replay audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.polymarket_condition_replay_audit import (
    audit_polymarket_condition_replay,
    write_polymarket_condition_replay_audit,
)
from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.storage import write_bytes_atomic


def _progress_callback(
    path: Path | None,
):
    def callback(phase: str, payload: Mapping[str, object]) -> None:
        message = {"phase": phase, **payload}
        print(json.dumps(message, sort_keys=True), flush=True)
        if path is not None:
            write_bytes_atomic(
                path,
                (json.dumps(message, indent=2, sort_keys=True) + "\n").encode(
                    "ascii"
                ),
            )

    return callback


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--condition-id", action="append", default=[])
    parser.add_argument("--build-condition-cache", action="store_true")
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    progress = _progress_callback(args.progress)
    with PolymarketEvidenceStore(
        args.database,
        memory_limit=args.memory_limit,
        threads=args.threads,
    ) as store:
        report = audit_polymarket_condition_replay(
            store,
            run_id=args.run_id,
            condition_ids=(
                tuple(args.condition_id) if args.condition_id else None
            ),
            build_condition_cache=args.build_condition_cache,
            progress=progress,
        )
    write_polymarket_condition_replay_audit(args.output, report)
    progress(
        "complete",
        {
            "audit_sha256": report["audit_sha256"],
            "condition_count": report["condition_count"],
            "eligible_condition_count": report["eligible_condition_count"],
            "failed_condition_count": report["failed_condition_count"],
        },
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "audit_sha256",
                    "condition_count",
                    "eligible_condition_count",
                    "failed_condition_count",
                    "eligible_condition_ids",
                    "failed_condition_ids",
                )
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
