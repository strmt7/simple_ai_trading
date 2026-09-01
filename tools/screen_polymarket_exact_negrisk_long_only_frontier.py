"""Capture one exact fixed-NegRisk event and screen its complete long-only basis."""

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
    _decimal,
    _decimal_text,
)
from tools.adjudicate_polymarket_retained_negrisk_no_cardinality_frontier import (
    _frontier_row,
    _headroom,
    _public_leg,
    _stressed_unit_cost,
)
from tools.adjudicate_polymarket_retained_negrisk_pairwise_no_v2 import (
    _acquisition_v2,
    _fee_model_v2,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_negrisk_complete_set_catalog import _eligible_event


CONTRACT_SCHEMA = "polymarket-exact-negrisk-long-only-frontier-contract-v1"
RESULT_SCHEMA = "polymarket-exact-negrisk-long-only-frontier-result-v1"


def _validate_contract(contract: Mapping[str, Any], path: Path) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("unexpected contract schema")
    if base._canonical_hash(dict(contract), "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if path != base._root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    slug = contract.get("event_slug")
    if not isinstance(slug, str) or not slug:
        raise RuntimeError("event slug is missing")
    count = contract.get("expected_market_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 100:
        raise RuntimeError("expected market count is invalid")
    if contract.get("request") != {
        "body_sha256": base._sha256(b""),
        "count": 1,
        "method": "GET",
        "url": f"https://gamma-api.polymarket.com/events/slug/{slug}",
    }:
        raise RuntimeError("request boundary changed")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        implementation_path = base._root_path(str(implementation["path"]))
        if base._sha256(implementation_path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(
                f"implementation hash mismatch: {implementation_path.name}"
            )


def _markets(event: Mapping[str, Any], expected_count: int) -> list[dict[str, Any]]:
    if not _eligible_event(dict(event)):
        raise RuntimeError("event is not an eligible fixed-NegRisk event")
    values = event.get("markets")
    if not isinstance(values, list) or len(values) != expected_count:
        raise RuntimeError("exact event market count changed")
    neg_risk_market_id = str(event.get("negRiskMarketID") or "").lower()
    if len(neg_risk_market_id) != 66 or not neg_risk_market_id.startswith("0x"):
        raise RuntimeError("event NegRisk market ID is invalid")
    markets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    seen_tokens: set[str] = set()
    for value in values:
        market = base._mapping(value, name="event market")
        market_id = str(market.get("id") or "")
        label = str(market.get("groupItemTitle") or market.get("question") or "")
        if (
            not market_id.isdigit()
            or market_id in seen_ids
            or not label
            or label in seen_labels
        ):
            raise RuntimeError("market identity or label is invalid or duplicated")
        if (
            market.get("active") is not True
            or market.get("closed") is not False
            or market.get("acceptingOrders") is not True
            or market.get("enableOrderBook") is not True
            or market.get("negRisk") is not True
            or str(market.get("negRiskMarketID") or "").lower() != neg_risk_market_id
        ):
            raise RuntimeError("event contains an unavailable market")
        outcomes = base._json_array(market.get("outcomes"), "outcomes")
        tokens = [
            str(item)
            for item in base._json_array(market.get("clobTokenIds"), "clobTokenIds")
        ]
        if outcomes != ["Yes", "No"] or len(tokens) != 2:
            raise RuntimeError("market binary mapping changed")
        if any(not token.isdigit() or token in seen_tokens for token in tokens):
            raise RuntimeError("market token identity is invalid or duplicated")
        seen_ids.add(market_id)
        seen_labels.add(label)
        seen_tokens.update(tokens)
        markets.append(market)
    return markets


def _yes_acquisition(market: dict[str, Any]) -> dict[str, Any] | None:
    raw_ask = market.get("bestAsk")
    if raw_ask is None:
        return None
    price = _decimal(raw_ask, name="YES bestAsk")
    if price <= 0 or price >= ONE:
        return None
    tokens = base._json_array(market.get("clobTokenIds"), "clobTokenIds")
    return {
        "market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "label": str(market.get("groupItemTitle") or market["question"]),
        "outcome": "Yes",
        "token_id": str(tokens[0]),
        "price_pUSD_per_share": price,
        "price_source": "direct_YES_bestAsk",
        "tick_size_pUSD": _decimal(market["orderPriceMinTickSize"], name="tick size"),
        "minimum_order_shares": _decimal(market["orderMinSize"], name="minimum order"),
        "fee_model": _fee_model_v2(market),
    }


def _package_row(
    event: Mapping[str, Any],
    *,
    package: str,
    floor: Decimal,
    legs: list[dict[str, Any]],
    quantity: Decimal,
) -> dict[str, Any]:
    actual_cost = sum((leg["price_pUSD_per_share"] for leg in legs), Decimal("0"))
    stressed_prices = [
        leg["price_pUSD_per_share"] + leg["tick_size_pUSD"] for leg in legs
    ]
    if any(price >= ONE for price in stressed_prices):
        stressed_fee: Decimal | None = None
        stressed_profit: Decimal | None = None
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
        "package": package,
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


def _screen(
    event: Mapping[str, Any], markets: list[dict[str, Any]], quantity: Decimal
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    yes_legs = [
        leg for market in markets if (leg := _yes_acquisition(market)) is not None
    ]
    no_legs = [
        leg
        for market in markets
        if (leg := _acquisition_v2(market, outcome="No")) is not None
    ]
    rows: list[dict[str, Any]] = []
    if len(yes_legs) == len(markets):
        rows.append(
            _package_row(
                event,
                package="all YES complete set",
                floor=ONE,
                legs=yes_legs,
                quantity=quantity,
            )
        )
    yes_by_market = {leg["market_id"]: leg for leg in yes_legs}
    no_by_market = {leg["market_id"]: leg for leg in no_legs}
    for market_id in sorted(yes_by_market.keys() & no_by_market.keys(), key=int):
        rows.append(
            _package_row(
                event,
                package="same-market YES plus NO binary straddle",
                floor=ONE,
                legs=[yes_by_market[market_id], no_by_market[market_id]],
                quantity=quantity,
            )
        )
    metadata_order = sorted(
        no_legs,
        key=lambda leg: (leg["price_pUSD_per_share"], int(leg["market_id"])),
    )
    stressed_order = sorted(
        no_legs,
        key=lambda leg: (_stressed_unit_cost(leg, quantity), int(leg["market_id"])),
    )
    for cardinality in range(2, len(no_legs) + 1):
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
    return rows, {
        "market_count": len(markets),
        "yes_price_complete_market_count": len(yes_legs),
        "no_price_complete_market_count": len(no_legs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = base._mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract, contract_path)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "event_slug": contract["event_slug"],
                    "expected_market_count": contract["expected_market_count"],
                    "payloads_printed": 0,
                    "preflight_only": True,
                    "requests_made": 0,
                },
                sort_keys=True,
            )
        )
        return

    raw_path = base._root_path(str(contract["outputs"]["raw_path"]))
    journal_path = base._root_path(str(contract["outputs"]["journal_path"]))
    result_path = base._root_path(str(contract["outputs"]["result_path"]))
    for output in (raw_path, journal_path, result_path):
        if output.exists():
            raise RuntimeError(f"one-use output already exists: {output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
    raw, receipt = _request(
        method="GET",
        url=str(contract["request"]["url"]),
        body=b"",
        name="prospective-exact-negrisk-long-only-frontier",
        raw_path=raw_path,
        raw_relative_path=str(contract["outputs"]["raw_path"]),
        journal_path=journal_path,
    )
    event = base._mapping(json.loads(raw), name="exact event response")
    if event.get("slug") != contract["event_slug"]:
        raise RuntimeError("exact event slug mismatch")
    markets = _markets(event, int(contract["expected_market_count"]))
    quantity = Decimal(str(contract["common_quantity_shares_each_leg"]))
    for market in markets:
        if Decimal(str(market["orderMinSize"])) != quantity:
            raise RuntimeError("frozen common minimum order size changed")
    rows, population = _screen(event, markets, quantity)
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
        "capture": {"receipt": receipt},
        "event": {
            "id": str(event["id"]),
            "slug": str(event["slug"]),
            "title": str(event["title"]),
            "end_date_utc": event.get("endDate"),
            **population,
        },
        "screen": {
            "basis": "all-YES complete set, every same-market YES-plus-NO straddle, and every optimal k-NO cardinality frontier",
            "rows": rows,
            "strict_metadata_candidate_count": len(metadata_candidates),
            "after_fee_one_tick_candidate_count": len(stressed_candidates),
            "best_metadata_package": best_metadata,
            "best_after_fee_one_tick_package": best_stressed,
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_one_frozen_exact_book_batch"
                if stressed_candidates
                else "prospective_event_rejected_before_books_accounts_and_orders"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze one exact current CLOB batch for every leg of the deterministic strongest stressed package"
                if stressed_candidates
                else "do not refetch reprice or request books for this exact event"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementations"][0],
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
                "event_id": str(event["id"]),
                "market_count": population["market_count"],
                "payloads_printed": 0,
                "strict_metadata_candidate_count": len(metadata_candidates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
