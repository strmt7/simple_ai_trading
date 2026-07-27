"""Check or execute the one permitted Round 74 active qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round74_active_qualification import (  # noqa: E402
    inspect_round74_active_readiness,
    run_round74_active_qualification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the frozen one-attempt Round 74 qualification."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="inspect readiness without reserving or starting the attempt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_only:
            report = inspect_round74_active_readiness(REPOSITORY)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ready_for_window"] is True else 2
        return run_round74_active_qualification(REPOSITORY)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"round74-active-qualification failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
