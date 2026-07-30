"""Run the preregistered Round 18 redundant public CLOB qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

from simple_ai_trading.polymarket_round18_transport import (
    run_round18_transport_qualification,
)
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-018-redundant-clob-transport-qualification-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "evidence"
    / "round-018-redundant-clob-transport-qualification-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded dual-lane BTC five-minute Polymarket CLOB "
            "transport qualification with no trading authority."
        )
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _progress(value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {"round": 18, **dict(value)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested_output = args.output
    output = requested_output.resolve()
    if requested_output.is_symlink() or output.exists():
        print(
            "round18-redundant-clob failed: output already exists or is a symlink",
            file=sys.stderr,
        )
        return 2
    try:
        result = run_round18_transport_qualification(
            args.contract.resolve(),
            progress=_progress,
        )
        payload = (
            json.dumps(
                result,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        write_bytes_atomic(output, payload)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"round18-redundant-clob failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "round": 18,
                "qualified": result["qualified"],
                "result_sha256": result["result_sha256"],
                "output": str(output),
                "model_data_eligible": result["model_data_eligible"],
                "profitability_claim": result["profitability_claim"],
                "live_trading_authority": result["live_trading_authority"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
