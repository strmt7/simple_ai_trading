"""Inspect or run the current predeclared Round 74 cohort slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round74_event_cohort_operator import (  # noqa: E402
    inspect_round74_cohort_readiness,
    run_round74_cohort_current_slot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate only the current immutable Round 74 cohort slot."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="inspect current readiness without reserving or capturing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.check_only:
            report = inspect_round74_cohort_readiness(REPOSITORY)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ready_for_current_slot"] is True else 2
        return run_round74_cohort_current_slot(REPOSITORY)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"round74-event-cohort-slot failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
