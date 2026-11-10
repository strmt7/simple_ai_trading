"""Reproduce synthetic payout controls to a new file; never read market data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from simple_ai_trading import option_long_floor


def exact(value):
    if isinstance(value, Fraction):
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [exact(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    source = root / "src/simple_ai_trading/option_long_floor.py"
    if Path(option_long_floor.__file__).resolve() != source.resolve():
        raise RuntimeError("Run with this checkout's isolated project environment")
    cases = []
    for name, call_rate, put_rate in (
        ("interior_fee_minimum", "0.1", "0.1"),
        ("nonterminating_switch", "0.2", "0.2"),
        ("no_exercise_fees", "0", "0"),
        ("unequal_leg_fees", "0", "0.1"),
    ):
        inputs = {
            "lower_call_strike": Decimal("100"),
            "upper_put_strike": Decimal("300"),
            "matched_base_quantity": Decimal("1"),
            "call_exercise_fee": option_long_floor.CappedExerciseFee(
                Decimal(call_rate), Decimal("0.5")
            ),
            "put_exercise_fee": option_long_floor.CappedExerciseFee(
                Decimal(put_rate), Decimal("0.5")
            ),
            "terminal_cost_reserve_quote": Decimal("0"),
        }
        result = option_long_floor.evaluate_fully_paid_long_option_floor(**inputs)
        cases.append(
            {
                "name": name,
                "inputs": {
                    key: asdict(item)
                    if isinstance(item, option_long_floor.CappedExerciseFee)
                    else item
                    for key, item in inputs.items()
                },
                "result": asdict(result),
            }
        )
    payload = {
        "schema": "fully-paid-long-option-synthetic-controls-v1",
        "source_canonical_LF_sha256": hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "scope": "Constructed supplied-law arithmetic only; no vendor coefficients, market observations or independent financial validation.",
        "cases": cases,
        "counterexamples": {
            "positive_payout_is_not_profit": {
                "upfront_cash_debit": Fraction(175),
                "alternative_terminal_multiplier": Fraction(51, 50),
                "net_payout_floor": Fraction(150),
                "incremental_floor": Fraction(150) - Fraction(175) * Fraction(51, 50),
            },
            "different_settlement_scalars_break_the_floor": {
                "call_settlement_price": Fraction(0),
                "put_settlement_price": Fraction(400),
                "lower_strike": Fraction(100),
                "upper_strike": Fraction(300),
                "actual_gross_payout": Fraction(0),
                "invalid_common_scalar_floor": Fraction(200),
            },
        },
        "qualified_edge": False,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(exact(payload), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
