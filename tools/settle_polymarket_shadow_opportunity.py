#!/usr/bin/env python3
"""Settle one hash-verified Polymarket shadow opportunity."""

from __future__ import annotations

import argparse
import hashlib
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
from simple_ai_trading.polymarket_historical_shadow_settlement import (  # noqa: E402
    settle_shadow_opportunity,
)


_MAXIMUM_LOG_BYTES = 2 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Polymarket shadow log contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket shadow log contains {value}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Settle one public, no-authority BTC shadow opportunity."
    )
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--opportunity-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def _load_opportunity(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, object], str, Mapping[str, object]]:
    if path.is_symlink():
        raise ValueError("Polymarket shadow log cannot be a symlink")
    raw = path.read_bytes()
    if not raw or len(raw) > _MAXIMUM_LOG_BYTES:
        raise ValueError("Polymarket shadow log size is invalid")
    records = []
    for line in raw.splitlines():
        if not line:
            continue
        try:
            record = json.loads(
                line.decode("utf-8", errors="strict"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Polymarket shadow log is not strict JSONL") from exc
        if not isinstance(record, Mapping):
            raise ValueError("Polymarket shadow log record is not an object")
        if (
            record.get("trading_authority") is not False
            or record.get("profitability_claim") is not False
        ):
            raise ValueError("Polymarket shadow log contains authority")
        records.append(record)
    selected = [
        record["payload"]
        for record in records
        if record.get("event") == "opportunity"
        and isinstance(record.get("payload"), Mapping)
        and record["payload"].get("artifact_sha256") == expected_sha256
    ]
    stopped = [
        record["payload"]
        for record in records
        if record.get("event") == "stopped"
        and isinstance(record.get("payload"), Mapping)
    ]
    if len(selected) != 1 or len(stopped) != 1:
        raise ValueError("Polymarket shadow log evidence set differs")
    health = stopped[0]
    if (
        health.get("running") is not False
        or any(str(item) for item in health.get("last_errors", {}).values())
        or any(int(item) != 0 for item in health.get("reconnect_counts", {}).values())
        or any(
            int(item) != 0
            for item in health.get("stale_epoch_discard_counts", {}).values()
        )
    ):
        raise ValueError("Polymarket shadow log ended with feed degradation")
    return (
        selected[0],
        hashlib.sha256(raw).hexdigest(),
        health,
    )


def main() -> int:
    args = _arguments()
    expected = str(args.opportunity_sha256 or "").strip().lower()
    opportunity, log_sha, health = _load_opportunity(
        args.event_log,
        expected_sha256=expected,
    )
    client = PolymarketPublicClient(timeout_seconds=args.timeout_seconds)
    gamma = client.gamma_market(str(opportunity["gamma_market_id"]))
    clob = client.clob_market(str(opportunity["condition_id"]))
    artifact, artifact_sha = settle_shadow_opportunity(
        opportunity,
        gamma_market=gamma,
        clob_market=clob,
        resolution_observed_at_ms=time.time_ns() // 1_000_000,
        source_log_sha256=log_sha,
        source_commit=args.source_commit,
    )
    artifact = {**artifact, "terminal_feed_health": dict(health)}
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256")
    artifact_sha = hashlib.sha256(
        json.dumps(
            unhashed,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    artifact["artifact_sha256"] = artifact_sha
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
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha,
                "output": str(args.output),
                "winner": artifact["official_resolution"]["winner"],
                "selected_outcome": artifact["prediction"][
                    "selected_outcome"
                ],
                "selected_outcome_won": artifact["prediction"][
                    "selected_outcome_won"
                ],
                "counterfactual_net_pnl_quote": artifact[
                    "displayed_depth_counterfactual"
                ]["net_pnl_quote"],
                "real_fill_observed": False,
                "trading_authority": False,
                "profitability_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
