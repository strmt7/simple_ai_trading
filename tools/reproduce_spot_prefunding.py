"""Retain exact synthetic spot-funding controls without market access."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path

from simple_ai_trading.spot_inventory_cash import SpotCashTrade, SpotInventoryCashPath
from simple_ai_trading.spot_prefunding import evaluate_spot_prefunding


ROOT = Path(__file__).resolve().parents[1]
INTENT = "docs/review/2026-11-09/spot-prefunding/intent.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _rational(value: object) -> str:
    if not isinstance(value, F):
        raise TypeError("unsupported synthetic control value")
    return str(value)


def control_report() -> dict[str, object]:
    intent = json.loads((ROOT / INTENT).read_bytes())
    if (
        intent["sha256"]
        != hashlib.sha256(
            _canonical({k: v for k, v in intent.items() if k != "sha256"})
        ).hexdigest()
    ):
        raise ValueError("prefunding intent digest mismatch")
    for name, expected in intent["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen accounting source changed")

    def trade(name, index, side, quantity, cash):
        return SpotCashTrade(
            name, index, side, F(quantity), F(cash), F(0), F(0), "a" * 64
        )

    buy = trade("buy", 0, "BUY", 1, 100)
    sell = trade("sell", 0, "SELL", 1, 100)
    close = trade("close", 1, "SELL", 1, 120)
    baseline = SpotInventoryCashPath(
        "BTCUSDT", (10, 20), (F(100), F(120)), F(0), (buy, close), "b" * 64
    )
    incumbent = replace(baseline, initial_base_quantity=F(1), trades=(sell, buy, close))
    cases = [
        ("ending_profit_without_initial_cash", baseline, F(0), F(0)),
        ("exact_acquisition_cash", baseline, F(100), F(0)),
        ("buffer_preserved", baseline, F(101), F(1)),
        (
            "zero_net_same_boundary_cash",
            replace(baseline, trades=(buy, sell)),
            F(0),
            F(0),
        ),
        ("incumbent_sale_before_repurchase", incumbent, F(0), F(0)),
        (
            "incumbent_repurchase_before_sale",
            replace(incumbent, trades=(buy, sell, close)),
            F(0),
            F(0),
        ),
        ("flat_abstention", replace(baseline, trades=()), F(0), F(0)),
    ]
    controls = []
    for name, path, cash, buffer in cases:
        result = evaluate_spot_prefunding(
            path, initial_quote_cash=cash, required_quote_buffer=buffer
        )
        controls.append(
            {"name": name, "spot_path": asdict(path), "result": asdict(result)}
        )
    controls = json.loads(json.dumps(controls, default=_rational, allow_nan=False))
    report = {
        "schema": "synthetic-spot-prefunding-controls-v1",
        "intent_sha256": intent["sha256"],
        "source_binding": "SHA-256 of LF-normalized source bytes; no host path",
        "source_sha256": {
            p: hashlib.sha256(
                (ROOT / p).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in [
                "src/simple_ai_trading/spot_inventory_cash.py",
                "src/simple_ai_trading/hedged_inventory_cash.py",
                "src/simple_ai_trading/spot_prefunding.py",
                "tools/reproduce_spot_prefunding.py",
                "tests/test_spot_prefunding.py",
                INTENT,
            ]
        },
        "controls": controls,
        "market_observations": 0,
        "native_capital_qualified": False,
        "qualified_edge": False,
        "limitations": [
            "Constructed exact cash identities, not native execution or profitability",
            "Immediate fill-proceeds settlement and no outside cash or cross-wallet credit are explicit scenario assumptions",
            "Order locks, native cash-source provenance, financing, reserves and continuous derivative margin remain unqualified",
        ],
    }
    report["result_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = control_report()
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        "Retained synthetic spot-funding controls; no market access or qualification."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
