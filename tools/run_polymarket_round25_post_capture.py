from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.compute import SUPPORTED_COMPUTE_BACKENDS  # noqa: E402
from simple_ai_trading.polymarket_round25_post_capture_runner import (  # noqa: E402
    Round25PostCaptureRunnerConfig,
    run_round25_post_capture,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _emit(event: str, values: Mapping[str, object]) -> None:
    print(_canonical_json({"event": event, **dict(values)}), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Advance one bounded, resumable Round 25 exact-TWAP post-capture pass."
        )
    )
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--maximum-resolution-conditions",
        type=int,
        choices=range(1, 513),
        default=128,
    )
    parser.add_argument(
        "--lightgbm-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
    )
    parser.add_argument(
        "--tcn-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_round25_post_capture(
        Round25PostCaptureRunnerConfig(
            repository=args.repository,
            source_database=args.source_database,
            plan_path=args.plan,
            state_root=args.state_root,
            output_root=args.output_root,
            source_commit_oid=args.source_commit,
            maximum_resolution_conditions=args.maximum_resolution_conditions,
            lightgbm_backend=args.lightgbm_backend,
            tcn_backend=args.tcn_backend,
        ),
        progress=_emit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
