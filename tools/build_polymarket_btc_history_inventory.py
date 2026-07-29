#!/usr/bin/env python3
"""Freeze all official BTC spot/perpetual archive days for Polymarket research."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.binance_archive import list_archive_items  # noqa: E402
from simple_ai_trading.polymarket_btc_history import (  # noqa: E402
    POLYMARKET_BTC_HISTORY_MARKET_TYPES,
    build_polymarket_btc_history_inventory,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-015-btc-5m-full-history-inventory-v1.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze every official BTCUSDT spot and USD-M aggTrades day in "
            "the requested interval without reading prices or outcomes."
        )
    )
    parser.add_argument("--first-day", default="2026-02-12")
    parser.add_argument("--last-day", default="2026-07-15")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    listings = {}
    for market_type in POLYMARKET_BTC_HISTORY_MARKET_TYPES:
        print(
            json.dumps(
                {
                    "event": "listing_started",
                    "market_type": market_type,
                    "symbol": "BTCUSDT",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        listings[market_type] = list_archive_items(
            symbol="BTCUSDT",
            interval="1s",
            market_type=market_type,
            cadence="daily",
            data_type="aggTrades",
            timeout=int(args.timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "event": "listing_complete",
                    "market_type": market_type,
                    "listed_days": len(listings[market_type]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    artifact = build_polymarket_btc_history_inventory(
        listings,
        first_day=args.first_day,
        last_day=args.last_day,
        observed_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "event": "inventory_complete",
                "artifact_sha256": artifact["artifact_sha256"],
                "day_count": artifact["range"]["day_count"],
                "source_count": artifact["source_count"],
                "selected_compressed_bytes": artifact[
                    "selected_compressed_bytes"
                ],
                "raw_archive_retained": False,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
