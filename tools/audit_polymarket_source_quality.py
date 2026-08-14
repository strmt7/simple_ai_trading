#!/usr/bin/env python3
"""Audit recorded Binance predictor trades without loading target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_source_quality import audit_binance_trade_quality
from simple_ai_trading.storage import write_bytes_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with PolymarketEvidenceStore(
        arguments.database.resolve(),
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        result = audit_binance_trade_quality(store, run_id=arguments.run_id)
    write_bytes_atomic(
        arguments.output.resolve(),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
