"""Offline exposure envelope of a complete retained buy population, not wallet PnL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from tools.update_binance_crypto_option_distinct_population_registry import (
    _canonical_hash as canonical_hash,
)


@dataclass(frozen=True)
class Scope:
    wallet: str
    start: int
    end: int


def envelope(rows: list[dict[str, Any]], scope: Scope) -> dict[str, Any]:
    """Count every scoped buy; bound binary payout without choosing winning outcomes."""
    if not rows or scope.start >= scope.end:
        raise ValueError("nonempty population and ordered window required")
    grouped: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    excluded = 0
    with localcontext() as context:
        context.prec = 50
        for row in rows:
            fingerprint = json.dumps(row, sort_keys=True, allow_nan=False)
            if fingerprint in seen:
                raise ValueError("identical rows have ambiguous fill identity")
            seen.add(fingerprint)
            if row["proxyWallet"].lower() != scope.wallet.lower():
                raise ValueError("wallet differs")
            timestamp = row["timestamp"]
            if type(timestamp) is not int or not scope.start <= timestamp < scope.end:
                raise ValueError("timestamp outside window")
            match = re.fullmatch(
                r"(btc|eth|sol)-updown-(5m|15m|4h)-(\d+)", row["eventSlug"]
            )
            if match is None:
                excluded += 1
                continue
            outcome = row["outcomeIndex"]
            if (
                row["side"] != "BUY"
                or type(outcome) is not int
                or outcome not in (0, 1)
            ):
                raise ValueError("scoped population must be binary buys only")
            if row["outcome"] != ("Up", "Down")[outcome]:
                raise ValueError("outcome label mapping differs")
            quantity, price = Decimal(str(row["size"])), Decimal(str(row["price"]))
            if (
                not quantity.is_finite()
                or not price.is_finite()
                or quantity <= 0
                or not 0 < price < 1
            ):
                raise ValueError("invalid quantity or price")
            condition = row["conditionId"]
            if not condition or not row["asset"]:
                raise ValueError("missing condition or token")
            group = grouped.setdefault(
                condition,
                {
                    "event_slug": row["eventSlug"],
                    "asset": match[1],
                    "quantity": [Decimal(0), Decimal(0)],
                    "tokens": {},
                    "cash": Decimal(0),
                    "rows": 0,
                },
            )
            if group["event_slug"] != row["eventSlug"]:
                raise ValueError("condition has conflicting event identity")
            token = group["tokens"].setdefault(outcome, row["asset"])
            if token != row["asset"] or len(set(group["tokens"].values())) != len(
                group["tokens"]
            ):
                raise ValueError("condition has conflicting token identity")
            group["quantity"][outcome] += quantity
            group["cash"] += quantity * price
            group["rows"] += 1
        if not grouped:
            raise ValueError("no scoped conditions")
        conditions = []
        for condition, group in sorted(grouped.items()):
            q0, q1 = group["quantity"]
            paired, residual = min(q0, q1), abs(q0 - q1)
            conditions.append(
                {
                    "condition_id": condition,
                    "event_slug": group["event_slug"],
                    "asset": group["asset"],
                    "buy_rows": group["rows"],
                    "quantity_up": str(q0),
                    "quantity_down": str(q1),
                    "gross_purchase_cash": str(group["cash"]),
                    "pair_quantity": str(paired),
                    "residual_quantity": str(residual),
                    "residual_outcome": "Up"
                    if q0 > q1
                    else "Down"
                    if q1 > q0
                    else None,
                    "gross_lower_pnl": str(paired - group["cash"]),
                    "gross_upper_pnl": str(paired + residual - group["cash"]),
                }
            )
        total = {
            key: sum((Decimal(c[key]) for c in conditions), Decimal(0))
            for key in (
                "quantity_up",
                "quantity_down",
                "gross_purchase_cash",
                "pair_quantity",
                "residual_quantity",
                "gross_lower_pnl",
                "gross_upper_pnl",
            )
        }
        required = total["gross_purchase_cash"] - total["pair_quantity"]
        return {
            "conditions": conditions,
            "raw_rows": len(rows),
            "excluded_out_of_scope_rows": excluded,
            "scoped_buy_rows": sum(c["buy_rows"] for c in conditions),
            "condition_count": len(conditions),
            "totals": {k: str(v) for k, v in total.items()},
            "positive_lower_condition_count": sum(
                Decimal(c["gross_lower_pnl"]) > 0 for c in conditions
            ),
            "negative_upper_condition_count": sum(
                Decimal(c["gross_upper_pnl"]) < 0 for c in conditions
            ),
            "unbalanced_condition_count": sum(
                Decimal(c["residual_quantity"]) > 0 for c in conditions
            ),
            "residual_payout_needed_for_gross_break_even": str(required),
            "residual_quantity_weighted_fraction_needed": str(
                required / total["residual_quantity"]
            )
            if total["residual_quantity"]
            else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.contract.read_bytes())
    if canonical_hash(plan, "contract_sha256") != plan["contract_sha256"]:
        raise ValueError("contract hash differs")
    for binding in plan["bindings"]:
        if (
            hashlib.sha256((root / binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise ValueError("retained binding differs")
    output = root / plan["output_path"]
    if output.exists():
        raise FileExistsError("offline study already consumed")
    rows = json.loads((root / plan["raw_path"]).read_bytes())
    result = envelope(rows, Scope(**plan["scope"]))
    if (
        result["raw_rows"] != plan["expected_raw_rows"]
        or result["condition_count"] != plan["expected_conditions"]
    ):
        raise ValueError("complete frozen population differs")
    result.update(
        {
            "schema_version": "binary-buy-exposure-envelope-v1",
            "contract_sha256": plan["contract_sha256"],
            "accepted_edge": False,
            "profitability_claim": False,
            "actual_wallet_pnl": False,
            "network_requests": 0,
            "interpretation": plan["interpretation"],
        }
    )
    result["result_sha256"] = canonical_hash(result, "result_sha256")
    with output.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "raw_rows",
                    "condition_count",
                    "totals",
                    "positive_lower_condition_count",
                    "negative_upper_condition_count",
                    "unbalanced_condition_count",
                    "residual_quantity_weighted_fraction_needed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
