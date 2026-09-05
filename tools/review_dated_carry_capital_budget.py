"""Conditional capital budget on a retained dated-carry snapshot, not ROI proof."""

from __future__ import annotations

from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-05/dated-carry-capital-budget"
YEAR_MS = D("31557600000")  # Same 365.25-day simple annualization as the source.


def capital_budget(
    gross_bips: D,
    duration_ms: int,
    annual_rate: D,
    capital_multiple: D,
    noncapital_reserve_bips: D,
) -> dict[str, str]:
    """Partition gross basis into hypothetical capital cost and all other costs.

    The multiple is committed capital divided by initial spot acquisition cost,
    not futures leverage or a claim that either margin model is feasible.
    """
    values = (gross_bips, annual_rate, capital_multiple, noncapital_reserve_bips)
    if any(not isinstance(x, D) or not x.is_finite() for x in values):
        raise ValueError("finite Decimal inputs required")
    if (
        type(duration_ms) is not int
        or duration_ms <= 0
        or annual_rate <= 0
        or capital_multiple < 1
        or noncapital_reserve_bips < 0
    ):
        raise ValueError(
            "positive duration/rate, fully funded capital and nonnegative reserve required"
        )
    years = D(duration_ms) / YEAR_MS
    cost = annual_rate * years * capital_multiple * 10000
    remaining = gross_bips - cost
    return {
        "capital_cost_bips": str(cost),
        "remaining_noncapital_cost_budget_bips": str(remaining),
        "headroom_after_separate_noncapital_reserve_bips": str(
            remaining - noncapital_reserve_bips
        ),
        "maximum_capital_multiple_after_separate_reserve": str(
            (gross_bips - noncapital_reserve_bips) / (annual_rate * years * 10000)
        ),
    }


def review() -> dict:
    plan = json.loads((BASE / "plan.json").read_bytes())
    for path, expected in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("source binding differs")
    source = json.loads((ROOT / plan["snapshot_path"]).read_bytes())
    rows = []
    for contract in source["screens"]:
        for original in contract["quantity_results"]:
            gross = D(original["gross_basis_bips"])
            horizon = original["delivery_time_ms"] - original["capture_time_ms"]
            if D(original["all_in_cost_hurdle_bips"]) != D(
                plan["separate_noncapital_reserve_bips"]
            ):
                raise ValueError(
                    "declared comparison reserve differs from source sensitivity"
                )
            scenarios = []
            for rate in plan["hypothetical_annual_capital_rates"]:
                for multiple in plan["hypothetical_committed_capital_multiples"]:
                    scenarios.append(
                        {
                            "annual_rate": rate,
                            "capital_multiple": multiple,
                            **capital_budget(
                                gross,
                                horizon,
                                D(rate),
                                D(multiple),
                                D(plan["separate_noncapital_reserve_bips"]),
                            ),
                        }
                    )
            rows.append(
                {
                    "symbol": contract["symbol"],
                    "quantity": original["quantity"],
                    "freshness_passed_at_original_capture": contract[
                        "freshness_passed"
                    ],
                    "capture_time_ms": original["capture_time_ms"],
                    "delivery_time_ms": original["delivery_time_ms"],
                    "original_gross_basis_bips": original["gross_basis_bips"],
                    "original_after_all_in_hurdle_basis_bips": original[
                        "after_hurdle_basis_bips"
                    ],
                    "scenarios": scenarios,
                }
            )
    if len(rows) != 12 or len({x["symbol"] for x in rows}) != 4:
        raise ValueError("original complete population differs")
    return {
        "schema_version": "retained-dated-carry-capital-budget-v1",
        "classification": "retrospective_conditional_sensitivity_not_validation",
        "plan_file_sha256": hashlib.sha256(
            (BASE / "plan.json").read_bytes()
        ).hexdigest(),
        "rows": rows,
        "accepted_edge": False,
        "capital_feasibility_proved": False,
        "current_prices_or_fees_qualified": False,
        "network_requests": 0,
        "original_results_changed": False,
    }


if __name__ == "__main__":
    result = review()
    with (BASE / "result.json").open("x", encoding="ascii", newline="\n") as output:
        json.dump(
            result, output, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        output.write("\n")
    print(
        json.dumps(
            {
                "rows": len(result["rows"]),
                "scenarios": sum(len(x["scenarios"]) for x in result["rows"]),
                "accepted_edge": False,
                "network_requests": 0,
            }
        )
    )
