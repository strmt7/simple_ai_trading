"""Build, inspect, or run the frozen Polymarket Round 25 core campaign."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from simple_ai_trading.polymarket_round25_campaign import (
    PolymarketRound25CampaignConfig,
    build_round25_campaign_plan,
    inspect_round25_campaign,
    run_round25_campaign,
    write_round25_campaign_plan,
)


ACKNOWLEDGEMENT = "start-frozen-round25-twap-core-capture-v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the source-pinned BTC Polymarket core capture."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/round25-twap-core-v2-campaign-plan.json"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/round25-twap-core-v2.duckdb"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("data/round25-twap-core-v2-state"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument(
        "--source-qualification",
        type=Path,
        default=Path(
            "docs/model-research/polymarket/"
            "round-025-twap-wire-source-qualification-v2-2026-08-10.json"
        ),
    )
    commands.add_parser("inspect")
    run = commands.add_parser("run")
    run.add_argument("--acknowledgement", required=True)
    return parser


def _config(arguments: argparse.Namespace) -> PolymarketRound25CampaignConfig:
    repository = arguments.repository.resolve()

    def resolved(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repository / path).resolve()

    return PolymarketRound25CampaignConfig(
        repository=repository,
        plan_path=resolved(arguments.plan),
        database_path=resolved(arguments.database),
        state_root=resolved(arguments.state_root),
    )


def main() -> int:
    arguments = _parser().parse_args()
    config = _config(arguments)
    if arguments.command == "build-plan":
        qualification = arguments.source_qualification
        if not qualification.is_absolute():
            qualification = config.repository / qualification
        plan = build_round25_campaign_plan(
            repository=config.repository,
            source_qualification=qualification,
        )
        write_round25_campaign_plan(config.plan_path, plan)
        print(json.dumps(plan, sort_keys=True))
        return 0
    if arguments.command == "inspect":
        print(json.dumps(inspect_round25_campaign(config), sort_keys=True))
        return 0
    if arguments.acknowledgement != ACKNOWLEDGEMENT:
        raise SystemExit("exact Round 25 acknowledgement is required")

    def progress(value: object) -> None:
        print(json.dumps(value, sort_keys=True), flush=True)

    result = asyncio.run(run_round25_campaign(config, progress=progress))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
