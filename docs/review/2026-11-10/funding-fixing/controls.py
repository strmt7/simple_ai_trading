"""Exact supplied-law funding examples; no venue adapter or market evidence."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from simple_ai_trading.funding_cash import (
    FundingEntitlement,
    LinearFundingSettlement,
    funding_cash_bounds_for_events,
)


def exact(value: Q) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def clamp(value: Q, lower: Q, upper: Q) -> Q:
    if lower > upper:
        raise ValueError("The supplied mathematical clamp needs ordered bounds")
    return min(max(value, lower), upper)


def rate(premium: Q, interval_hours: Q) -> Q:
    # Explicit scenario parameters, not installed exchange rules or defaults.
    interest, damper, cap = Q(1, 10000), Q(5, 10000), Q(75, 10000)
    uncapped = (
        (premium + clamp(interest - premium, -damper, damper)) * interval_hours / 8
    )
    return clamp(uncapped, -cap, cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = 0

    def check(condition: bool) -> None:
        nonlocal checks
        if not condition:
            raise ValueError("Constructed funding control failed")
        checks += 1

    # The inner map is increasing outside the damper and constant inside it.
    # Positive interval scaling and the outer ordered clamp preserve monotonicity.
    premiums = [Q(-1), Q(-4, 10000), Q(0), Q(6, 10000), Q(1)]
    check(all(rate(a, Q(8)) <= rate(b, Q(8)) for a, b in zip(premiums, premiums[1:])))
    check(rate(Q(-4, 10000), Q(8)) == rate(Q(6, 10000), Q(8)) == Q(1, 10000))
    prefix_cases = []
    for unknown in (1, 2):
        total, known = 5760, 5760 - unknown
        remaining_weight = 1 - Q(known * (known + 1), total * (total + 1))
        # All supplied known raw premiums are zero; every unknown raw premium
        # is bounded below by -1. This is not a rolling-estimate substitution.
        premium_lower = -remaining_weight
        floor = rate(premium_lower, Q(8))
        check(floor == Q(1, 10000) if unknown == 1 else floor < 0)
        prefix_cases.append(
            {
                "total_samples": total,
                "unknown_final_samples": unknown,
                "known_raw_premium": "0",
                "supplied_unknown_raw_lower": "-1",
                "remaining_weight": exact(remaining_weight),
                "premium_lower": exact(premium_lower),
                "funding_rate_lower": exact(floor),
            }
        )

    # Equal-sized linear spot/perpetual holdings remove the common-price move,
    # but the perpetual-minus-spot basis still changes total hedged cash.
    event = LinearFundingSettlement(
        "BTCUSDT", "USDT", 1000, Q(1, 10000), Q(100), "a" * 64
    )
    held = funding_cash_bounds_for_events(
        (event,),
        (FundingEntitlement.HELD,),
        expected_symbol="BTCUSDT",
        signed_base_quantity=Q(-1),
        entry_price=Q(100),
    )
    unknown = funding_cash_bounds_for_events(
        (event,),
        (FundingEntitlement.UNKNOWN,),
        expected_symbol="BTCUSDT",
        signed_base_quantity=Q(-1),
        entry_price=Q(100),
    )
    check(held.cash_lower == held.cash_upper == Q(1, 100))
    check(unknown.cash_lower == 0 and unknown.cash_upper == held.cash_lower)
    basis_cases = []
    for basis in (Q(0), Q(1, 100), Q(2, 100)):
        spot_entry = spot_exit = futures_entry = Q(100)
        futures_exit = spot_exit + basis
        price_cash = spot_exit - spot_entry - (futures_exit - futures_entry)
        net_before_costs = price_cash + held.cash_lower
        check(net_before_costs == Q(1, 100) - basis)
        basis_cases.append(
            {
                "matched_base_quantity": "1",
                "spot_entry_and_exit": "100",
                "futures_entry": "100",
                "futures_exit": exact(futures_exit),
                "funding_cash": exact(held.cash_lower),
                "net_before_costs": exact(net_before_costs),
                "supplied_total_cost_quote": "0.02",
                "net_after_costs": exact(net_before_costs - Q(2, 100)),
            }
        )
    check(basis_cases[1]["net_before_costs"] == exact(Q(0)))
    check(basis_cases[2]["net_before_costs"] == exact(Q(-1, 100)))

    # Demonstrate the consequence of interpreting the same N as hours versus
    # seconds. Neither interpretation is asserted to be the native venue law.
    hours_rate, seconds_rate = rate(Q(1, 1000), Q(4)), rate(Q(1, 1000), Q(14400))
    check(hours_rate == Q(25, 100000))
    check(seconds_rate == Q(75, 10000))
    payload = {
        "schema": "conditional-funding-fixing-controls-v1",
        "scope": "Constructed exact rational scenarios only; ordered mathematical clamp and declared interval hours. No source-applicability, native sampling/clock/entitlement, financial performance or deployability qualification.",
        "prefix_cases": prefix_cases,
        "coupon_basis_cases": basis_cases,
        "unknown_entitlement_cash": {
            "lower": exact(unknown.cash_lower),
            "upper": exact(unknown.cash_upper),
        },
        "unit_interpretation": {
            "same_supplied_premium": "0.001",
            "interval_hours_interpretation": exact(hours_rate),
            "literal_seconds_substitution": exact(seconds_rate),
        },
        "checks": checks,
        "market_observations": 0,
        "qualified_edge": False,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"constructed_checks": checks, "new_market_observations": 0}))


if __name__ == "__main__":
    main()
