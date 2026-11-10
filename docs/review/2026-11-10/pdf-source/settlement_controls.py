"""Construct positive index paths under two possible settlement interpretations.

These are exact counterexamples to assuming a common scalar, not observations
of Binance's actual settlement implementation or proof of an exploitable error.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path


def fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, mean, final in (
        ("both_legs_zero", 100, 400),
        ("small_reference_gap", 200, 210),
    ):
        # A supplied 1,800-point equally weighted path; the procedure does not
        # independently qualify this sampling schedule as the vendor's method.
        count = 1800
        earlier = Fraction(count * mean - final, count - 1)
        actual_mean = (earlier * (count - 1) + final) / count
        call = max(actual_mean - 100, Fraction(0))
        put_on_final_index = max(Fraction(300 - final), Fraction(0))
        put_on_same_mean = max(300 - actual_mean, Fraction(0))
        if earlier <= 0 or actual_mean != mean:
            raise ValueError("constructed positive path failed its exact mean identity")
        rows.append(
            {
                "name": name,
                "samples": count,
                "earlier_identical_positive_index_values": count - 1,
                "earlier_index": fraction(earlier),
                "last_index": fraction(Fraction(final)),
                "arithmetic_mean": fraction(actual_mean),
                "lower_call_strike": "100",
                "upper_put_strike": "300",
                "base_quantity": "1",
                "call_on_mean_gross": fraction(call),
                "put_on_final_index_gross": fraction(put_on_final_index),
                "mixed_reference_gross_total": fraction(call + put_on_final_index),
                "common_mean_gross_total": fraction(call + put_on_same_mean),
                "invalid_assumed_width_floor": "200",
            }
        )
    payload = {
        "schema": "conditional-settlement-reference-counterexamples-v1",
        "scope": "Constructed positive index paths only. Gross payouts, before entry/terminal costs. Final-index put is one literal interpretation of an ambiguous retained source, not a verified venue law.",
        "cases": rows,
        "market_observations": 0,
        "qualified_edge": False,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
