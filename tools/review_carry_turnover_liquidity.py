"""Retained funding cash paths and turnover sensitivity, never strategy selection."""

from datetime import datetime, timezone
from decimal import Decimal as D
import json
import os
from pathlib import Path

from tools.review_spot_funding_cashflows import canonical, digest, normalize_history
from tools.verify_spot_funding_cashflow_publication import published_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-05/carry-turnover-liquidity"


def funding_path(cashflows):
    if not cashflows or any(not x.is_finite() for x in cashflows):
        raise ValueError("nonempty finite funding cash required")
    cumulative = peak = low = drawdown = D(0)
    negative = 0
    for cash in cashflows:
        cumulative += cash
        low = min(low, cumulative)
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
        negative += cash < 0
    return {
        "net_funding_quote_per_base": cumulative,
        "minimum_retained_funding_prefix_quote_per_base": low,
        "funding_only_prefund_if_all_receipts_retained_quote_per_base": -low,
        "largest_funding_peak_to_trough_quote_per_base": drawdown,
        "negative_settlement_count": negative,
    }


def calculate(plan):
    for p, h in plan["source_sha256"].items():
        if digest((ROOT / p).read_bytes()) != h:
            raise ValueError("frozen source differs: " + p)
    audit = json.loads((ROOT / plan["audit_path"]).read_bytes())
    if canonical(audit, "result_sha256") != audit["result_sha256"]:
        raise ValueError("audit self-hash differs")
    bindings = {x["path"]: x for x in audit["bindings"]}
    prior = json.loads(published_bytes(bindings[plan["original_result_path"]]))
    receipts = {x["name"]: x for x in prior["sources"]["responses"]}
    if sorted({x["symbol"] for x in audit["rows"]}) != plan["symbols"]:
        raise ValueError("complete symbol population differs")
    rows = []
    for symbol in plan["symbols"]:
        roles = [
            next(
                x for x in audit["rows"] if x["symbol"] == symbol and x["role"] == role
            )
            for role in ("training", "validation", "test")
        ]
        raw_path = receipts["funding-" + symbol.lower()]["raw_path"]
        history = normalize_history(
            json.loads(published_bytes(bindings[raw_path])), symbol
        )
        selected = []
        for role in roles:
            part = [
                x
                for x in history
                if role["first_time_ms"] <= x["time"] <= role["last_time_ms"]
            ]
            if len(part) != role["row_count"]:
                raise ValueError("role count differs")
            cash = sum((x["rate"] * x["mark"] for x in part), D(0))
            if cash != D(role["metrics"]["funding_cash_quote_per_base"]):
                raise ValueError("prior role funding differs")
            selected.extend(part)
        if len({x["time"] for x in selected}) != len(selected):
            raise ValueError("overlapping roles")
        full = [
            x
            for x in history
            if roles[0]["first_time_ms"] <= x["time"] <= roles[-1]["last_time_ms"]
        ]
        if selected != full:
            raise ValueError("holding interval has unaccounted funding settlements")
        reference = D(roles[0]["metrics"]["reference_mark_quote_per_base"])
        references = [D(x["metrics"]["reference_mark_quote_per_base"]) for x in roles]
        charge_fraction = D(plan["execution_allowance_bps_per_round_trip"]) / 10000
        single_charge = reference * charge_fraction
        segmented_charge = sum(references, D(0)) * charge_fraction
        original_capital = sum(
            (
                ref * D(role["metrics"]["original_capital_stress_bps"]) / 10000
                for ref, role in zip(references, roles, strict=True)
            ),
            D(0),
        )
        path = funding_path([x["rate"] * x["mark"] for x in full])
        cash = path["net_funding_quote_per_base"]
        values = {
            **path,
            "first_reference_mark_quote_per_base": reference,
            "single_round_trip_allowance_quote_per_base": single_charge,
            "three_role_round_trip_allowance_quote_per_base": segmented_charge,
            "avoided_two_round_trip_allowances_quote_per_base": segmented_charge
            - single_charge,
            "net_funding_less_one_allowance_bps_at_first_reference": (
                cash - single_charge
            )
            / reference
            * 10000,
            "net_funding_less_three_allowances_bps_at_first_reference": (
                cash - segmented_charge
            )
            / reference
            * 10000,
            "original_role_weighted_capital_stress_quote_per_base": original_capital,
            "one_allowance_less_unchanged_role_capital_stress_bps_at_first_reference": (
                cash - single_charge - original_capital
            )
            / reference
            * 10000,
            "funding_only_prefund_bps_at_first_reference": path[
                "funding_only_prefund_if_all_receipts_retained_quote_per_base"
            ]
            / reference
            * 10000,
            "funding_drawdown_bps_at_first_reference": path[
                "largest_funding_peak_to_trough_quote_per_base"
            ]
            / reference
            * 10000,
        }
        rows.append(
            {
                "symbol": symbol,
                "settlements": len(full),
                "first_funding_time_ms": full[0]["time"],
                "last_funding_time_ms": full[-1]["time"],
                "reference_time_ms": roles[0]["reference_time_ms"],
                "metrics": {
                    k: str(v) if isinstance(v, D) else v for k, v in values.items()
                },
            }
        )
    return {
        "schema_version": "carry-turnover-funding-liquidity-review-v1",
        "rows": rows,
        "symbol_count": len(rows),
        "interpretation": "Same one-base-unit funding cash across the original complete role union. Single versus three 32-bp allowances is a transaction-cost sensitivity, not measured commission; references are retained funding marks, not executions. Original role-weighted capital stress is identical in both comparisons. Cash path omits spot/future price PnL, collateral variation, basis, fees, borrowing, liquidation and conversion, so funding-only prefund is not a sufficient margin buffer.",
        "classification": "exploratory consumed history including training; no exit or symbol selection, no revalidation or promotion",
        "accepted_edge": False,
        "profitability_claim": False,
        "new_requests": 0,
        "old_acceptance_changed": False,
        "protected_access": False,
    }


def main():
    plan_path = BASE / "plan.json"
    result_path = BASE / "result.json"
    if result_path.exists():
        raise FileExistsError("audit consumed")
    with (BASE / "journal.jsonl").open("x", encoding="ascii", newline="\n") as journal:

        def record(phase, **extra):
            journal.write(
                json.dumps(
                    {
                        "phase": phase,
                        "utc": datetime.now(timezone.utc).isoformat(),
                        **extra,
                    }
                )
                + "\n"
            )
            journal.flush()
            os.fsync(journal.fileno())

        plan_raw = plan_path.read_bytes()
        record("started", plan_sha256=digest(plan_raw))
        try:
            result = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": digest(plan_raw),
                **calculate(json.loads(plan_raw)),
            }
            result["result_sha256"] = canonical(result, "result_sha256")
            with result_path.open("xb") as out:
                out.write(json.dumps(result, indent=2).encode() + b"\n")
            record("completed", result_sha256=result["result_sha256"])
            print(
                json.dumps(
                    {
                        "result_sha256": result["result_sha256"],
                        "symbol_count": result["symbol_count"],
                    }
                )
            )
        except Exception as exc:
            record("failed", error_type=type(exc).__name__)
            raise


if __name__ == "__main__":
    main()
