"""Build a synthetic collateral-stress demonstration without venue access."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

from simple_ai_trading.collateral_stress import CollateralState, stress_linear_hedge

ROOT = Path(__file__).resolve().parents[1]


def review() -> dict:
    states = tuple(
        CollateralState(str(p), D(p), D(p), D(".95"), D(p) / 10, D(0))
        for p in (100, 200, 1000)
    )
    result = stress_linear_hedge(
        owned_asset_quantity=D(1),
        short_base_quantity=D(1),
        future_entry_price=D(100),
        quote_cash=D(0),
        required_quote_buffer=D(0),
        states=states,
    )
    sources = (
        "src/simple_ai_trading/collateral_stress.py",
        "tools/review_collateral_stress.py",
        "tests/test_collateral_stress.py",
        "docs/review/2026-09-05/collateral-semantics-discovery-plan.json",
        "docs/review/2026-09-05/collateral-semantics-discovery-extraction.json",
        "docs/review/2026-09-05/collateral-semantics-source-plan.json",
        "docs/review/2026-09-05/collateral-semantics-source-extraction.json",
    )
    return {
        "schema_version": "conditional-collateral-stress-review-v1",
        "classification": "synthetic_accounting_demonstration_not_market_evidence",
        "inputs": {
            "owned_asset_quantity": "1",
            "short_base_quantity": "1",
            "future_entry_price": "100",
            "quote_cash": "0",
            "required_quote_buffer": "0",
            "states": [asdict(s) for s in states],
        },
        "result": asdict(result),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
        },
        "source_adjudication": "The independently opened educational Portfolio Margin Pro limits guide explains wallet combination and collateral calculation but its 95 percent BTC rate is illustrative, not a current asset-rate table. Program and contract applicability remain unqualified. The same number in this synthetic demonstration is not adopted as an actual rate.",
        "limitations": "Supplied marks, credit ratio and margin requirements; fully credited nonborrowed quote cash and full short-PnL recognition assumed. No probability distribution, complete risk envelope, current account eligibility, actual margin engine, execution, profit, model uplift or new economic capture. Original results and registry unchanged.",
    }


if __name__ == "__main__":
    print(
        json.dumps(
            review(),
            default=str,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
