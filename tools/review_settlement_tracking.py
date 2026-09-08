"""Publish fixed conditional settlement examples without venue or account access."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal as D
from pathlib import Path

from simple_ai_trading.settlement_tracking import (
    SettlementObservation,
    evaluate_linear_settlement_tracking,
)


def build_review(root: Path) -> dict:
    scenarios = (
        ("matched_rising", ("90", "110"), ("0.5", "0.5"), "0", "0"),
        ("matched_falling", ("110", "90"), ("0.5", "0.5"), "0", "0"),
        ("endpoint_exit_falling", ("110", "90"), ("0", "1"), "0", "0"),
        ("matched_but_index_basis", ("90", "110"), ("0.5", "0.5"), "2", "0"),
        ("matched_but_execution_loss", ("90", "110"), ("0.5", "0.5"), "0", "3"),
    )
    rows = []
    for label, prices, exits, index_addition, execution_loss in scenarios:
        observations = tuple(
            SettlementObservation(
                timestamp_ms=i + 1,
                index_price=D(price) + D(index_addition),
                spot_reference_price=D(price),
                spot_execution_price=D(price) - D(execution_loss)
                if D(weight)
                else None,
                settlement_weight_units=D("1"),
                spot_exit_weight=D(weight),
            )
            for i, (price, weight) in enumerate(zip(prices, exits, strict=True))
        )
        inputs = dict(
            quantity=D("1"),
            spot_entry_price=D("100"),
            futures_entry_price=D("101"),
            total_cost_quote=D("0.2"),
            declared_timestamps_ms=(1, 2),
            observations=observations,
            spot_price_lower_bound=D("90"),
            spot_price_upper_bound=D("110"),
        )
        result = evaluate_linear_settlement_tracking(**inputs)
        rows.append(
            {
                "label": label,
                "inputs": {
                    **inputs,
                    "observations": [asdict(row) for row in observations],
                },
                "result": asdict(result),
            }
        )
    paths = (
        "src/simple_ai_trading/settlement_tracking.py",
        "tools/review_settlement_tracking.py",
        "tests/test_settlement_tracking.py",
        "docs/review/2026-09-04/inverse-clearing-source-review.json",
    )
    payload = {
        "schema_version": 1,
        "classification": "conditional_linear_settlement_accounting_not_market_outcomes",
        "base_commit": "538792180379e66578b208094baa9d54ba859839",
        "source_bindings": {
            path: hashlib.sha256(
                (root / path).read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for path in paths
        },
        "source_binding_encoding": "UTF-8 with CRLF normalized to LF",
        "retained_pdf_sha256": "53197b612332da02c20b5b7d19b81ff53ee5f4938c6330c72a30a1ca4f91049f",
        "source_applicability_verified": False,
        "qualified_edge": False,
        "new_market_requests": 0,
        "new_account_requests": 0,
        "historical_results_changed": False,
        "rounding": "exact rational internal arithmetic; recurring result fields rounded to 60 decimal significant digits",
        "examples": rows,
    }
    return json.loads(json.dumps(payload, default=str, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_review(Path(__file__).resolve().parents[1])
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"examples": len(result["examples"]), "qualified_edge": False}))


if __name__ == "__main__":
    main()
