#!/usr/bin/env python3
"""Compatibility entry point for the frozen Round 14 historical screen."""

from __future__ import annotations

from pathlib import Path

from run_polymarket_historical_screen import main


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_contract=(
                ROOT
                / "docs"
                / "model-research"
                / "polymarket"
                / "round-014-btc-5m-historical-screen-v2.json"
            ),
            default_database=(
                ROOT / "data" / "polymarket-round14-historical-screen-v2.duckdb"
            ),
            default_microstructure=(
                ROOT.parent
                / "simple_ai_trading"
                / "data"
                / "microstructure.duckdb"
            ),
        )
    )
