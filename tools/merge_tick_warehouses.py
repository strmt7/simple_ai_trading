from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from simple_ai_trading.tick_warehouse_merge import merge_certified_tick_warehouse


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transactionally merge certified tick partitions into one warehouse."
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--data-type",
        choices=("bookTicker", "trades", "bookDepth"),
        required=True,
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    def progress(phase: str, completed: int, total: int) -> None:
        print(f"merge-tick-warehouses {phase}: {completed}/{total}", flush=True)

    try:
        evidence = merge_certified_tick_warehouse(
            destination_path=args.destination,
            source_path=args.source,
            symbol=args.symbol,
            data_type=args.data_type,
            start_date=args.start_date,
            end_date=args.end_date,
            memory_limit=args.memory_limit,
            threads=args.threads,
            progress=progress,
        )
    except Exception as exc:
        print(
            f"merge-tick-warehouses failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    if args.output is not None:
        _write_atomic(args.output, evidence)
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
