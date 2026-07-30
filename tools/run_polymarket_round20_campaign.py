"""Initialize, inspect, or run the independent Round 20 corpus campaign."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from simple_ai_trading.polymarket_round20_campaign import (
    PolymarketRound20CampaignConfig,
    build_round20_campaign_plan,
    inspect_round20_campaign,
    run_round20_campaign,
    write_round20_campaign_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data" / "round20-campaign-plan.json"
DEFAULT_DATABASE = ROOT / "data" / "round20-campaign.duckdb"
DEFAULT_STATE = ROOT / "data" / "round20-campaign-state"


def _config(arguments: argparse.Namespace) -> PolymarketRound20CampaignConfig:
    return PolymarketRound20CampaignConfig(
        repository=ROOT,
        plan_path=arguments.plan.resolve(),
        database_path=arguments.database.resolve(),
        state_root=arguments.state_root.resolve(),
    )


def _print(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--scheduled-start-ms", type=int, required=True)
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--poll-interval-seconds", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.command == "init":
        if arguments.plan.exists():
            raise FileExistsError(f"Round 20 plan already exists: {arguments.plan}")
        value = build_round20_campaign_plan(
            ROOT,
            scheduled_start_ms=arguments.scheduled_start_ms,
        )
        write_round20_campaign_plan(arguments.plan, value)
        _print(value)
        return 0
    config = _config(arguments)
    if arguments.command == "status":
        _print(inspect_round20_campaign(config))
        return 0
    _print(
        asyncio.run(
            run_round20_campaign(
                config,
                poll_interval_seconds=arguments.poll_interval_seconds,
                progress=_print,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
