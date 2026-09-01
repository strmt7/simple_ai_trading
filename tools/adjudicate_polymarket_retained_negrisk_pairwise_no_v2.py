"""Correctly adjudicate retained pairwise-NO floors with explicit zero-fee rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from tools import adjudicate_polymarket_retained_negrisk_pairwise_no as base
from tools.adjudicate_polymarket_release_date_deadline_graph import (
    ONE,
    _decimal,
    _decimal_text,
    _fee_model,
)


CONTRACT_SCHEMA = "polymarket-retained-negrisk-pairwise-no-contract-v2"
RESULT_SCHEMA = "polymarket-retained-negrisk-pairwise-no-result-v2"


def _fee_model_v2(market: dict[str, Any]) -> PolymarketFeeModel:
    if market.get("feesEnabled") is False:
        if market.get("feeSchedule") is not None or market.get("feeType") not in (
            None,
            "",
        ):
            raise RuntimeError(
                f"market {market.get('id')} disabled fee representation changed"
            )
        return PolymarketFeeModel(False, Decimal("0"), 1, True)
    return _fee_model(market)


def _acquisition_v2(market: dict[str, Any], *, outcome: str) -> dict[str, Any] | None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        return None
    if outcome != "No":
        raise RuntimeError("v2 supports only frozen NO acquisition")
    raw_bid = market.get("bestBid")
    if raw_bid is None:
        return None
    price = ONE - _decimal(raw_bid, name="YES bestBid")
    if price <= 0 or price >= ONE:
        return None
    tokens = base._json_array(market.get("clobTokenIds"), "clobTokenIds")
    return {
        "market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "outcome": "No",
        "token_id": str(tokens[1]),
        "price_pUSD_per_share": price,
        "price_source": "one_minus_direct_YES_bestBid",
        "tick_size_pUSD": _decimal(market["orderPriceMinTickSize"], name="tick size"),
        "minimum_order_shares": _decimal(market["orderMinSize"], name="minimum order"),
        "fee_model": _fee_model_v2(market),
    }


def _pair_row(
    event: Mapping[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    identity = {
        "event_id": str(event["id"]),
        "event_slug": str(event["slug"]),
        "event_title": str(event["title"]),
        "first_market_id": str(first["id"]),
        "first_label": str(first.get("groupItemTitle") or first.get("question")),
        "second_market_id": str(second["id"]),
        "second_label": str(second.get("groupItemTitle") or second.get("question")),
        "package": "NO(first mutually exclusive outcome) plus NO(second mutually exclusive outcome)",
        "guaranteed_payout_floor_pUSD_per_share": "1",
    }
    first_no = _acquisition_v2(first, outcome="No")
    second_no = _acquisition_v2(second, outcome="No")
    if first_no is None or second_no is None:
        return {
            **identity,
            "status": "missing_side_specific_acquisition_evidence",
            "passes_strict_metadata_gate": False,
            "passes_fee_and_one_tick_gate": False,
        }

    quantity = max(first_no["minimum_order_shares"], second_no["minimum_order_shares"])
    actual_prices = [
        first_no["price_pUSD_per_share"],
        second_no["price_pUSD_per_share"],
    ]
    stressed_prices = [
        first_no["price_pUSD_per_share"] + first_no["tick_size_pUSD"],
        second_no["price_pUSD_per_share"] + second_no["tick_size_pUSD"],
    ]
    actual_sum = sum(actual_prices, Decimal("0"))
    gross_headroom = quantity * (ONE - actual_sum)
    if any(price >= ONE for price in stressed_prices):
        stressed_fee: Decimal | None = None
        stressed_headroom: Decimal | None = None
    else:
        stressed_fee = first_no["fee_model"](
            stressed_prices[0], quantity, "taker"
        ) + second_no["fee_model"](stressed_prices[1], quantity, "taker")
        stressed_headroom = (
            quantity * (ONE - sum(stressed_prices, Decimal("0"))) - stressed_fee
        )
    return {
        **identity,
        "status": "priced",
        "quantity_shares_each_leg": _decimal_text(quantity),
        "legs": [
            {
                key: (_decimal_text(value) if isinstance(value, Decimal) else value)
                for key, value in leg.items()
                if key != "fee_model"
            }
            for leg in (first_no, second_no)
        ],
        "metadata_cost_pUSD_per_share": _decimal_text(actual_sum),
        "metadata_gross_headroom_pUSD": _decimal_text(gross_headroom),
        "one_adverse_tick_per_leg_prices_pUSD": [
            _decimal_text(value) for value in stressed_prices
        ],
        "stressed_taker_fee_pUSD": _decimal_text(stressed_fee),
        "after_fee_one_tick_profit_floor_pUSD": _decimal_text(stressed_headroom),
        "passes_strict_metadata_gate": actual_sum < ONE,
        "passes_fee_and_one_tick_gate": (
            stressed_headroom is not None and stressed_headroom > 0
        ),
    }


def _row_key(row: Mapping[str, Any]) -> tuple[bool, Decimal, int, int, int]:
    return (
        row["status"] != "priced",
        Decimal(str(row.get("metadata_cost_pUSD_per_share", "Infinity"))),
        int(row["event_id"]),
        int(row["first_market_id"]),
        int(row["second_market_id"]),
    )


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
    pair_count = sum(len(markets) * (len(markets) - 1) // 2 for _, markets in selected)
    if pair_count != contract["expected_pair_count"]:
        raise RuntimeError("frozen pair count changed")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "event_count": len(selected),
                    "pair_count": pair_count,
                    "payloads_printed": 0,
                    "preflight_only": True,
                    "retained_input": str(retained.relative_to(base._root_path("."))),
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
    for event, markets in selected:
        rows.extend(
            _pair_row(event, first, second)
            for first, second in combinations(markets, 2)
        )
    rows.sort(key=_row_key)
    priced = [row for row in rows if row["status"] == "priced"]
    metadata_candidates = [row for row in priced if row["passes_strict_metadata_gate"]]
    stressed_candidates = [row for row in priced if row["passes_fee_and_one_tick_gate"]]
    best = priced[0] if priced else None

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "superseded_failure": contract["superseded_failure"],
        "retained_input": contract["retained_input"],
        "population": {
            "event_count": len(selected),
            "market_counts": {
                str(event["id"]): len(markets) for event, markets in selected
            },
            "pair_count": len(rows),
            "priced_pair_count": len(priced),
            "price_incomplete_pair_count": len(rows) - len(priced),
        },
        "screen": {
            "price_gate": "conservative one minus direct YES bestBid for NO acquisition; Gamma is rejection-only",
            "payout_identity": "two distinct outcomes in one fixed-NegRisk event are mutually exclusive, so NO(A) plus NO(B) pays at least one pUSD",
            "zero_fee_fallback": "feesEnabled false with absent feeSchedule and empty feeType means zero configured taker fee; enabled markets require the supported exact schedule",
            "rows": rows,
            "best_priced_pair": best,
            "strict_metadata_candidate_count": len(metadata_candidates),
            "after_fee_one_tick_candidate_count": len(stressed_candidates),
            "best_after_fee_one_tick_candidate": (
                stressed_candidates[0] if stressed_candidates else None
            ),
        },
        "adjudication": {
            "status": (
                "retained_candidate_requires_separately_frozen_exact_depth_batch"
                if stressed_candidates
                else "retained_population_rejected_before_books_accounts_and_orders"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze one exact current depth batch for the deterministic strongest candidate"
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
                "best_cost_pUSD_per_share": (
                    best["metadata_cost_pUSD_per_share"] if best else None
                ),
                "event_count": len(selected),
                "pair_count": len(rows),
                "payloads_printed": 0,
                "price_incomplete_pair_count": len(rows) - len(priced),
                "strict_metadata_candidate_count": len(metadata_candidates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
