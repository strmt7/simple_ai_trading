"""Initialize, inspect, or run the resumable Round 21 Binance sidecar."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from simple_ai_trading.polymarket_round21_sidecar_campaign import (
    Round21SidecarCampaignConfig,
    build_round21_sidecar_campaign_plan,
    inspect_round21_sidecar_campaign,
    run_round21_sidecar_campaign,
    write_round21_sidecar_campaign_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data" / "round21-binance-sidecar-campaign-plan-v2.json"
DEFAULT_DATABASE = ROOT / "data" / "round21-binance-sidecar.duckdb"
DEFAULT_STATE_ROOT = ROOT / "data" / "round21-binance-sidecar-campaign-state-v2"
DEFAULT_LEGACY_STATE = ROOT / "data" / "round21-binance-sidecar-state.json"


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


def _config(arguments: argparse.Namespace) -> Round21SidecarCampaignConfig:
    return Round21SidecarCampaignConfig(
        repository=ROOT,
        plan_path=arguments.plan.resolve(),
        database_path=arguments.database.resolve(),
        state_root=arguments.state_root.resolve(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reboot-resumable Round 21 public Binance sidecar."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument(
        "--legacy-state",
        type=Path,
        default=DEFAULT_LEGACY_STATE,
    )
    initialize.add_argument("--scheduled-start-ms", type=int, required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("run")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "init":
            if arguments.plan.exists():
                raise FileExistsError(
                    f"Round 21 sidecar campaign plan exists: {arguments.plan}"
                )
            plan = build_round21_sidecar_campaign_plan(
                ROOT,
                legacy_state_path=arguments.legacy_state.resolve(),
                scheduled_start_ms=arguments.scheduled_start_ms,
            )
            write_round21_sidecar_campaign_plan(arguments.plan, plan)
            _print(plan)
            return 0
        config = _config(arguments)
        if arguments.command == "status":
            _print(inspect_round21_sidecar_campaign(config))
            return 0
        result = asyncio.run(
            run_round21_sidecar_campaign(config, progress=_print)
        )
        _print(result)
        return 0 if result["status"] == "campaign_window_ended" else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"Round 21 sidecar campaign failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
