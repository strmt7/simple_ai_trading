#!/usr/bin/env python3
"""Resolve and evaluate one preregistered Polymarket shadow-hour log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket import PolymarketPublicClient  # noqa: E402
from simple_ai_trading.polymarket_historical_shadow_campaign import (  # noqa: E402
    build_shadow_hour_evaluation,
    load_shadow_hour_log,
    load_shadow_hour_policy,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen public BTC Polymarket shadow hour."
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    policy = load_shadow_hour_policy(args.policy)
    records, source_log_sha = load_shadow_hour_log(
        args.event_log,
        policy=policy,
    )
    representatives: dict[int, Mapping[str, object]] = {}
    for record in records:
        if record.get("event") != "opportunity":
            continue
        payload = record["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("Polymarket shadow opportunity payload differs")
        event_start = int(payload["event_start_ms"])
        if (
            int(policy["eligible_event_start_ms"])
            <= event_start
            < int(policy["expected_run_end_ms"])
        ):
            representatives.setdefault(event_start, payload)
    client = PolymarketPublicClient(timeout_seconds=args.timeout_seconds)
    terminal_sources = {}
    for event_start, opportunity in sorted(representatives.items()):
        gamma = client.gamma_market(str(opportunity["gamma_market_id"]))
        clob = client.clob_market(str(opportunity["condition_id"]))
        terminal_sources[event_start] = (
            gamma,
            clob,
            time.time_ns() // 1_000_000,
        )
    artifact, artifact_sha = build_shadow_hour_evaluation(
        policy=policy,
        records=records,
        source_log_sha256=source_log_sha,
        terminal_sources=terminal_sources,
        source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            artifact,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.output)
    counterfactual = artifact["first_candidate_counterfactual"]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha,
                "resolved_events": artifact["population"]["resolved_events"],
                "observed_prediction_rows": artifact["population"][
                    "observed_prediction_rows"
                ],
                "selected_events": counterfactual["selected_events"],
                "counterfactual_net_pnl_quote": counterfactual[
                    "net_pnl_quote"
                ],
                "real_fills_observed": 0,
                "profitability_claim": False,
                "trading_authority": False,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
