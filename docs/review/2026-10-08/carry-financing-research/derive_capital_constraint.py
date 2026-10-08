"""Derive a post-hoc financing bound from retained cash, without replaying a study."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SOURCE = Path("docs/review/2026-10-07/native-funding-hedge-frontier/result.json")
SOURCE_SHA256 = "11312e21e0d3bf5e3805e0a08e590e9100cb92c28b3fd166eeacbb7b0f32f79e"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def derive(source_bytes: bytes) -> dict[str, object]:
    """Invert the existing cost scenario; do not calculate new funding outcomes."""
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Retained source bytes differ from the reviewed result")
    source = json.loads(source_bytes)
    panels = [panel for panel in source["panels"] if panel["panel"] == "whole_32_days"]
    if len(panels) != 3 or sorted(panel["symbol"] for panel in panels) != sorted(
        SYMBOLS
    ):
        raise ValueError("Require the complete original three-asset whole-period scope")
    rows = []
    for symbol in SYMBOLS:
        panel = next(panel for panel in panels if panel["symbol"] == symbol)
        scenario = [
            item
            for item in panel["sensitivities"]
            if Fraction(item["annual_capital_cost_bps"]) == 325
            and Fraction(item["noncapital_cost_reserve_bps"]) == 32
        ]
        if len(scenario) != 1:
            raise ValueError("Require the exact original 325/32-bps sensitivity")
        days = Fraction(panel["duration_days_at_reference"])
        original_multiple = Fraction(panel["capital_multiple_at_reference"])
        funding = Fraction(panel["funding_bps_at_reference"])
        cost_per_unit = Fraction(325) * days / 365
        if days != 32 or original_multiple != 2:
            raise ValueError("Original duration/capital scope changed")
        if (
            Fraction(scenario[0]["capital_cost_bps"])
            != cost_per_unit * original_multiple
        ):
            raise ValueError("Capital-cost law disagrees with the retained source")
        surplus_before_capital = funding - 32
        threshold = surplus_before_capital / cost_per_unit
        net_at_one = surplus_before_capital - cost_per_unit
        net_at_threshold = surplus_before_capital - cost_per_unit * threshold
        if net_at_threshold != 0:
            raise ArithmeticError("Strict-positive threshold must reconcile exactly")
        rows.append(
            {
                "symbol": symbol,
                "retained_funding_bps": str(funding),
                "capital_cost_bps_per_reference_unit": str(cost_per_unit),
                "strict_positive_net_requires_capital_multiple_below": str(threshold),
                "threshold_below_one_fully_funded_spot_unit": threshold < 1,
                "net_bps_at_one_spot_unit_and_zero_perpetual_margin": str(net_at_one),
                "net_bps_at_one_spot_unit_and_illustrative_one_twentieth_margin": str(
                    surplus_before_capital - cost_per_unit * Fraction(21, 20)
                ),
                "threshold_identity_exact": True,
            }
        )
    return {
        "schema": "posthoc-retained-capital-bound-v1",
        "source_path": SOURCE.as_posix(),
        "source_host_bytes_sha256": digest,
        "source_recorded_result_sha256": source["result_sha256"],
        "classification": "post-hoc algebraic design constraint, not a new replay or independent evaluation",
        "cash_law_bps": "funding - noncapital_cost - basis_deterioration - annual_capital_cost_bps * capital_multiple * days / 365",
        "scenario": {
            "annual_capital_cost_bps": "325",
            "noncapital_cost_reserve_bps": "32",
            "basis_deterioration_bps": "0",
            "duration_days": "32",
            "funding_entitlement": "Original hypothetical HELD case; not owned receipts",
            "capital_rates_and_margin_are_native_terms": False,
            "capital_minimum_one_applies_to": "Fully cash-funded spot purchase normalized to the same reference unit; not every account/collateral structure",
        },
        "rows": rows,
        "conclusion": "Perpetual-margin compression alone cannot make this fully funded spot hedge positive under the declared scenario, even at zero perpetual margin. Different financing, incremental costs, collateral yield or basis outcomes require their own evidence.",
        "new_market_requests": 0,
        "research_observation_count_effect": 0,
        "accepted_edge": False,
        "training_admitted": False,
        "continuous_solvency_qualified": False,
    }


def main() -> None:
    """Write a new derived artifact, refusing to replace any existing result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    payload = derive((repo / SOURCE).read_bytes())
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"assets": len(payload["rows"]), "accepted_edge": False}))


if __name__ == "__main__":
    main()
