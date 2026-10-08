"""Reproduce synthetic transition cash controls; never capture market outcomes."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_inventory import InventoryCashPath
from simple_ai_trading.inventory_cash_actions import (
    InventoryCashAction,
    evaluate_inventory_cash_actions,
)


ROOT = Path(__file__).resolve().parents[1]


def control_report() -> dict[str, object]:
    """Bind exact synthetic cash controls to the implementation and frozen intent."""
    controls = []
    for name, incumbent, exit_price, with_funding in (
        ("existing_unit_flat_prices", 1, 100, False),
        ("new_unit_flat_prices", 0, 100, False),
        ("existing_unit_price_doubles", 1, 200, False),
        ("existing_unit_entry_funding", 1, 100, True),
    ):
        events = (
            (
                LinearFundingSettlement(
                    "BTCUSDT", "USDT", 10, Fraction("0.01"), Fraction(150), "c" * 64
                ),
            )
            if with_funding
            else ()
        )
        path = InventoryCashPath(
            "BTCUSDT",
            (10, 20),
            (Fraction(100), Fraction(exit_price)),
            (0,),
            Fraction(100),
            Fraction(6, 10_000),
            "a" * 64,
            events,
            tuple((event.funding_time_ms, event.rate) for event in events),
            0,
            30,
            "b" * 64,
            initial_signed_base_quantity=Fraction(incumbent),
        )
        table = evaluate_inventory_cash_actions(
            path,
            tuple(
                InventoryCashAction(action_id, target)
                for action_id, target in (
                    ("flatten", Fraction(0)),
                    ("long", Fraction(1)),
                    ("reduce_long", Fraction(1, 2)),
                    ("short", Fraction(-1)),
                    ("increase_long", Fraction(2)),
                )
            ),
            state_as_of_ms=8,
            decision_time_ms=9,
            state_source_sha256="d" * 64,
        )
        controls.append(
            {
                "name": name,
                "input_sha256": table.input_sha256,
                "initial_signed_base_quantity": str(table.initial_signed_base_quantity),
                "values": [
                    {
                        "action": value.action.action_id,
                        "quantity": str(value.action.target_signed_base_quantity),
                        "traded_quote": str(value.cash_replay.total_traded_quote),
                        "execution_cost": str(value.cash_replay.total_execution_cost),
                        "cash_lower": str(value.cash_replay.total_net_cash_lower),
                        "cash_upper": str(value.cash_replay.total_net_cash_upper),
                        "surplus_lower": str(value.surplus_cash_lower),
                        "surplus_upper": str(value.surplus_cash_upper),
                    }
                    for value in table.values
                ],
            }
        )
    return {
        "schema": "synthetic-inventory-action-controls-v1",
        "baseline_commit": "5c0a95698e5dbbf9760c637542b5cbc988388038",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/simple_ai_trading/funding_cash.py",
                "src/simple_ai_trading/funding_cash_inventory.py",
                "src/simple_ai_trading/inventory_cash_actions.py",
                "tests/test_inventory_cash_actions.py",
                "tools/reproduce_inventory_cash_actions.py",
                "docs/review/2026-10-08/inventory-action-intent.json",
            )
        },
        "controls": controls,
        "financially_qualified": False,
        "model_fit_or_market_access": False,
        "historical_results_modified": False,
        "limitations": [
            "Synthetic single-instrument horizon-liquidation labels, not an edge or continuous Bellman objective",
            "Supplied state/source digests do not authenticate origin or decision-time availability",
            "Monotone net quantity transitions do not qualify owned partial executions",
            "Hedge legs, basis, all-in costs, financing, margin and chronological roles remain required before fits",
        ],
        "financial_counts_unchanged": {
            "observations": 202,
            "hypotheses": 65,
            "scoped_mechanisms": 37,
            "qualified_stable_edges": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Exclusive creation preserves existing evidence. This runner performs no
    # network access or fitting; numeric artifacts are generated, not hand-edited.
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(control_report(), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Retained synthetic controls: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
