"""Run or inspect the immutable Round 14 BTC prospective campaign."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from simple_ai_trading.polymarket_round14_campaign import (
    PolymarketRound14CampaignConfig,
    inspect_round14_campaign,
    run_round14_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the independent BTC-only Polymarket Round 14 capture.",
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--readiness-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path(args.repository).resolve()
    config = PolymarketRound14CampaignConfig(
        repository=repository,
        plan_path=(repository / args.plan).resolve(),
        database_path=(repository / args.database).resolve(),
        state_root=(repository / args.state_root).resolve(),
    )
    try:
        result = (
            inspect_round14_campaign(config)
            if args.readiness_only
            else asyncio.run(run_round14_campaign(config))
        )
    except KeyboardInterrupt:
        print("polymarket-round14-campaign interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"polymarket-round14-campaign failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
