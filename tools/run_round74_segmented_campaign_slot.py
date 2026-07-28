"""Run exactly one currently open slot from a frozen segmented campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from simple_ai_trading.round74_segmented_campaign_runner import (
    Round74SegmentedCampaignRunnerConfig,
    inspect_round74_segmented_campaign_readiness,
    run_round74_segmented_campaign_current_slot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one immutable Round 74 segmented campaign slot.",
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prerequisite", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--readiness-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path(args.repository).resolve()
    config = Round74SegmentedCampaignRunnerConfig(
        repository=repository,
        plan_path=(repository / args.plan).resolve(),
        prerequisite_path=(repository / args.prerequisite).resolve(),
        database_path=(repository / args.database).resolve(),
        state_root=(repository / args.state_root).resolve(),
    )
    try:
        if args.readiness_only:
            result = inspect_round74_segmented_campaign_readiness(config)
            exit_code = 0 if result["can_start_now"] else 2
        else:
            result = run_round74_segmented_campaign_current_slot(config)
            exit_code = 0
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"round74-segmented-campaign failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
