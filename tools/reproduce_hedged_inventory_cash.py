"""Retain synthetic paired-cash accounting controls without market access."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path

from simple_ai_trading.funding_cash import LinearFundingSettlement
from simple_ai_trading.funding_cash_inventory import InventoryCashPath
from simple_ai_trading.hedged_inventory_cash import (
    HedgedInventoryCashPath,
    replay_hedged_inventory_cash,
)
from simple_ai_trading.spot_inventory_cash import SpotCashTrade, SpotInventoryCashPath


ROOT = Path(__file__).resolve().parents[1]
INTENT = "docs/review/2026-11-09/paired-inventory-cash/intent.json"


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _fraction_text(value: object) -> str:
    if not isinstance(value, F):
        raise TypeError("synthetic control contains an unsupported value")
    return str(value)


def _trade(
    index: int, side: str, quantity: str, cash: str, fee: str = "0"
) -> SpotCashTrade:
    return SpotCashTrade(
        f"fill-{index}-{side}",
        index,
        side,
        F(quantity),
        F(cash),
        F(fee),
        F(0),
        "a" * 64,
    )


def _path(exit_price: int, hedge_quantity: F) -> HedgedInventoryCashPath:
    perpetual = InventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (F(101), F(exit_price + 1)),
        (-1,),
        F(100),
        F(0),
        "b" * 64,
        (),
        (),
        0,
        30,
        "c" * 64,
        target_signed_base_quantity=(-hedge_quantity,),
    )
    spot = SpotInventoryCashPath(
        "BTCUSDT",
        (10, 20),
        (F(100), F(exit_price)),
        F(0),
        (_trade(0, "BUY", "1", "100"), _trade(1, "SELL", "1", str(exit_price))),
        "d" * 64,
    )
    return HedgedInventoryCashPath(perpetual, spot, (F(0),), (F(0),), (F(0),), "e" * 64)


def control_report() -> dict[str, object]:
    """Reproduce exact accounting controls, distinct from economic validation."""
    intent = json.loads((ROOT / INTENT).read_bytes())
    if intent["sha256"] != _canonical(
        {k: v for k, v in intent.items() if k != "sha256"}
    ):
        raise ValueError("paired-cash intent digest mismatch")
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
        for name, expected in intent["source_sha256"].items()
    ):
        raise ValueError("frozen pre-existing cash sources changed")
    cases = []
    for exit_price in (80, 120):
        cases.extend(
            [
                (f"matched_base_exit_{exit_price}", _path(exit_price, F(1))),
                (f"equal_quote_exit_{exit_price}", _path(exit_price, F(100, 101))),
            ]
        )
    path = _path(120, F(1))
    fee_spot = replace(
        path.spot,
        trades=(
            _trade(0, "BUY", "1", "100", "0.01"),
            _trade(1, "SELL", "0.99", "118.8"),
        ),
    )
    cases.extend(
        [
            ("base_fee_unmatched", replace(path, spot=fee_spot)),
            (
                "base_fee_matched",
                replace(
                    path,
                    spot=fee_spot,
                    perpetual=replace(
                        path.perpetual, target_signed_base_quantity=(-F("0.99"),)
                    ),
                ),
            ),
            (
                "incumbent_matched",
                replace(
                    path,
                    spot=replace(
                        path.spot,
                        initial_base_quantity=F(1),
                        trades=path.spot.trades[1:],
                    ),
                    perpetual=replace(
                        path.perpetual, initial_signed_base_quantity=-F(1)
                    ),
                ),
            ),
        ]
    )
    event = LinearFundingSettlement("BTCUSDT", "USDT", 15, F("0.001"), F(110), "f" * 64)
    costs = replace(
        path,
        perpetual=replace(
            path.perpetual,
            settlements=(event,),
            expected_events=((15, event.rate),),
            one_way_cost_fraction=F(6, 10_000),
        ),
        spot=replace(
            path.spot,
            trades=(
                replace(path.spot.trades[0], quote_commission=F("0.1")),
                replace(path.spot.trades[1], quote_commission=F("0.12")),
            ),
        ),
        financing_debit_quote=(F("0.02"),),
        opportunity_debit_quote=(F("0.03"),),
        other_debit_quote=(F("0.04"),),
    )
    cases.append(("funding_after_every_declared_debit", costs))
    rows = []
    for name, supplied in cases:
        result = replay_hedged_inventory_cash(supplied)
        rows.append(
            {
                "name": name,
                "input_sha256": result.input_sha256,
                "inputs": asdict(supplied),
                "spot_incremental_wealth": result.spot.incremental_wealth_change,
                "perpetual_net_lower": result.perpetual.total_net_cash_lower,
                "perpetual_net_upper": result.perpetual.total_net_cash_upper,
                "external_debits": result.total_external_debit_quote,
                "paired_net_lower": result.total_net_cash_lower,
                "paired_net_upper": result.total_net_cash_upper,
                "residual_base_by_interval": [
                    r.residual_base_quantity for r in result.intervals
                ],
                "boundary_orphan_ranges": result.boundary_orphan_base_ranges,
                "base_matched_between_boundaries": result.base_matched_in_every_supplied_interval,
            }
        )
    # Normalize only rationals; never round or serialize a private host mapping.
    rows = json.loads(json.dumps(rows, default=_fraction_text, allow_nan=False))
    report = {
        "schema": "synthetic-paired-inventory-cash-controls-v1",
        "intent_sha256": intent["sha256"],
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/simple_ai_trading/funding_cash.py",
                "src/simple_ai_trading/funding_cash_inventory.py",
                "src/simple_ai_trading/spot_inventory_cash.py",
                "src/simple_ai_trading/hedged_inventory_cash.py",
                "tools/reproduce_hedged_inventory_cash.py",
                INTENT,
            )
        },
        "controls": rows,
        "market_observations": 0,
        "financially_qualified": False,
        "training_admitted": False,
        "qualified_edge": False,
        "limitations": [
            "Constructed accounting identities, not native fills, data admission or positive market economics",
            "Perpetual executions and costs retain the existing supplied proportional model",
            "Boundary orphan ranges enclose unknown cross-leg ordering, not cash-loss or fill-probability bounds",
            "Native paired coverage, every fee asset, causal timing, financing, equal-wealth comparator and continuous margin remain open",
        ],
    }
    report["result_sha256"] = _canonical(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = control_report()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("Retained synthetic paired-cash controls; no market access or qualification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
