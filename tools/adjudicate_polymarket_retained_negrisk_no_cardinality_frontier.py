"""Screen every optimal k-NO frontier in retained complete fixed-NegRisk events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping

from tools import adjudicate_polymarket_retained_negrisk_pairwise_no as base
from tools.adjudicate_polymarket_release_date_deadline_graph import (
    ONE,
    _decimal_text,
)
from tools.adjudicate_polymarket_retained_negrisk_pairwise_no_v2 import (
    _acquisition_v2,
)


CONTRACT_SCHEMA = "polymarket-retained-negrisk-no-cardinality-frontier-contract-v1"
RESULT_SCHEMA = "polymarket-retained-negrisk-no-cardinality-frontier-result-v1"


def _public_leg(leg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (_decimal_text(value) if isinstance(value, Decimal) else value)
        for key, value in leg.items()
        if key != "fee_model"
    }


def _frontier_row(
    event: Mapping[str, Any],
    *,
    cardinality: int,
    legs: list[dict[str, Any]],
    quantity: Decimal,
    stressed: bool,
) -> dict[str, Any]:
    floor = Decimal(cardinality - 1)
    actual_cost = sum((leg["price_pUSD_per_share"] for leg in legs), Decimal("0"))
    stressed_prices = [
        leg["price_pUSD_per_share"] + leg["tick_size_pUSD"] for leg in legs
    ]
    stressed_fee: Decimal | None
    stressed_profit: Decimal | None
    if any(price >= ONE for price in stressed_prices):
        stressed_fee = None
        stressed_profit = None
    else:
        stressed_fee = sum(
            (
                leg["fee_model"](price, quantity, "taker")
                for leg, price in zip(legs, stressed_prices, strict=True)
            ),
            Decimal("0"),
        )
        stressed_profit = (
            quantity * (floor - sum(stressed_prices, Decimal("0"))) - stressed_fee
        )
    metadata_profit = quantity * (floor - actual_cost)
    return {
        "event_id": str(event["id"]),
        "event_slug": str(event["slug"]),
        "event_title": str(event["title"]),
        "cardinality": cardinality,
        "frontier_basis": (
            "lowest_after_fee_one_tick_unit_cost"
            if stressed
            else "lowest_metadata_cost"
        ),
        "guaranteed_payout_floor_pUSD_per_share": _decimal_text(floor),
        "quantity_shares_each_leg": _decimal_text(quantity),
        "legs": [_public_leg(leg) for leg in legs],
        "metadata_cost_pUSD_per_share": _decimal_text(actual_cost),
        "metadata_profit_floor_pUSD": _decimal_text(metadata_profit),
        "one_adverse_tick_per_leg_prices_pUSD": [
            _decimal_text(value) for value in stressed_prices
        ],
        "stressed_taker_fee_pUSD": _decimal_text(stressed_fee),
        "after_fee_one_tick_profit_floor_pUSD": _decimal_text(stressed_profit),
        "passes_strict_metadata_gate": metadata_profit > 0,
        "passes_fee_and_one_tick_gate": (
            stressed_profit is not None and stressed_profit > 0
        ),
    }


def _stressed_unit_cost(leg: Mapping[str, Any], quantity: Decimal) -> Decimal:
    price = leg["price_pUSD_per_share"] + leg["tick_size_pUSD"]
    if price >= ONE:
        return Decimal("Infinity")
    return price + leg["fee_model"](price, quantity, "taker") / quantity


def _frontiers(
    event: Mapping[str, Any], markets: list[dict[str, Any]], quantity: Decimal
) -> tuple[list[dict[str, Any]], int]:
    acquisitions = [
        acquisition
        for market in markets
        if (acquisition := _acquisition_v2(market, outcome="No")) is not None
    ]
    metadata_order = sorted(
        acquisitions,
        key=lambda leg: (leg["price_pUSD_per_share"], int(leg["market_id"])),
    )
    stressed_order = sorted(
        acquisitions,
        key=lambda leg: (_stressed_unit_cost(leg, quantity), int(leg["market_id"])),
    )
    rows: list[dict[str, Any]] = []
    for cardinality in range(2, len(acquisitions) + 1):
        metadata_legs = metadata_order[:cardinality]
        stressed_legs = stressed_order[:cardinality]
        rows.append(
            _frontier_row(
                event,
                cardinality=cardinality,
                legs=metadata_legs,
                quantity=quantity,
                stressed=False,
            )
        )
        if [leg["market_id"] for leg in stressed_legs] != [
            leg["market_id"] for leg in metadata_legs
        ]:
            rows.append(
                _frontier_row(
                    event,
                    cardinality=cardinality,
                    legs=stressed_legs,
                    quantity=quantity,
                    stressed=True,
                )
            )
    return rows, len(acquisitions)


def _headroom(row: Mapping[str, Any], key: str) -> Decimal:
    value = row.get(key)
    return Decimal(str(value)) if value is not None else Decimal("-Infinity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = base._mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    base.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    retained, _, selected = base._validate_contract(contract, contract_path)
    market_counts = {str(event["id"]): len(markets) for event, markets in selected}
    if market_counts != contract["expected_market_counts"]:
        raise RuntimeError("frozen market counts changed")
    theoretical_subset_count = sum(
        (1 << len(markets)) - len(markets) - 1 for _, markets in selected
    )
    if theoretical_subset_count != contract["expected_theoretical_subset_count"]:
        raise RuntimeError("frozen theoretical subset count changed")
    quantity = Decimal(str(contract["common_quantity_shares_each_leg"]))
    if quantity <= 0:
        raise RuntimeError("frozen common quantity must be positive")
    for _, markets in selected:
        for market in markets:
            if Decimal(str(market["orderMinSize"])) != quantity:
                raise RuntimeError("frozen common minimum order size changed")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "event_count": len(selected),
                    "market_count": sum(market_counts.values()),
                    "payloads_printed": 0,
                    "preflight_only": True,
                    "retained_input": str(retained.relative_to(base._root_path("."))),
                    "theoretical_subset_count": theoretical_subset_count,
                },
                sort_keys=True,
            )
        )
        return

    result_path = base._root_path(str(contract["output_path"]))
    if result_path.exists():
        raise RuntimeError("one-use result already exists")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    priced_market_counts: dict[str, int] = {}
    for event, markets in selected:
        event_rows, priced_count = _frontiers(event, markets, quantity)
        rows.extend(event_rows)
        priced_market_counts[str(event["id"])] = priced_count
    metadata_candidates = [row for row in rows if row["passes_strict_metadata_gate"]]
    stressed_candidates = [row for row in rows if row["passes_fee_and_one_tick_gate"]]
    best_metadata = max(
        rows, key=lambda row: _headroom(row, "metadata_profit_floor_pUSD")
    )
    best_stressed = max(
        rows, key=lambda row: _headroom(row, "after_fee_one_tick_profit_floor_pUSD")
    )

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "retained_input": contract["retained_input"],
        "population": {
            "event_count": len(selected),
            "market_counts": market_counts,
            "priced_market_counts": priced_market_counts,
            "theoretical_subset_count": theoretical_subset_count,
            "frontier_row_count": len(rows),
        },
        "screen": {
            "payout_identity": "NO on any k distinct outcomes in one complete fixed-NegRisk event pays at least k minus one pUSD",
            "optimization_identity": "for each cardinality, a profitable subset exists iff its cheapest complete acquisition frontier is profitable",
            "price_gate": "conservative one minus direct YES bestBid; Gamma is rejection-only and missing prices are not free",
            "zero_fee_fallback": "feesEnabled false with absent feeSchedule and empty feeType means zero configured taker fee; enabled markets require the supported exact schedule",
            "rows": rows,
            "strict_metadata_candidate_count": len(metadata_candidates),
            "after_fee_one_tick_candidate_count": len(stressed_candidates),
            "best_metadata_frontier": best_metadata,
            "best_after_fee_one_tick_frontier": best_stressed,
        },
        "adjudication": {
            "status": (
                "retrospective_hypothesis_generated_requires_distinct_prospective_confirmation"
                if stressed_candidates
                else "retained_population_rejected_before_books_accounts_and_orders"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "promotion_eligible": False,
            "next_action": (
                "freeze a distinct unconsumed complete fixed-NegRisk event before prices and require prospective recurrence"
                if stressed_candidates
                else "do not refetch reprice or request books for this retained exact population"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = base._canonical_hash(result, "result_sha256")
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "after_fee_one_tick_candidate_count": len(stressed_candidates),
                "best_after_fee_one_tick_profit_floor_pUSD": best_stressed.get(
                    "after_fee_one_tick_profit_floor_pUSD"
                ),
                "best_cardinality": best_stressed["cardinality"],
                "best_event_id": best_stressed["event_id"],
                "event_count": len(selected),
                "frontier_row_count": len(rows),
                "payloads_printed": 0,
                "strict_metadata_candidate_count": len(metadata_candidates),
                "theoretical_subset_count": theoretical_subset_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
