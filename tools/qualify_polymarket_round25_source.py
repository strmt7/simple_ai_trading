"""Qualify the exact public BTC five-minute resolution-source regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from simple_ai_trading.polymarket import PolymarketPublicClient
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    qualify_round25_source,
)
from simple_ai_trading.storage import write_bytes_atomic


DEFAULT_OUTPUT = Path(
    "docs/model-research/polymarket/"
    "round-025-twap-source-qualification-2026-08-10.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe two consecutive public BTC 5m markets and CLOB metadata."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    client = PolymarketPublicClient(
        required_five_minute_resolution_sources={
            "BTC": POLYMARKET_ROUND25_RESOLUTION_SOURCE
        }
    )
    result = qualify_round25_source(
        client,
        observed_at_ms=time.time_ns() // 1_000_000,
    )
    write_bytes_atomic(
        arguments.output,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
