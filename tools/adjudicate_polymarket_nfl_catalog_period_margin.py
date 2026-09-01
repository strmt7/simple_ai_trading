from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_nfl_catalog_period_graph import (
    _canonical_hash,
    _pair,
    _root_path,
    _sha256,
    _side_price,
)


ONE = Decimal("1")
SCOPES = {
    "1Q": ("q1_moneyline", "q1_spreads"),
    "2Q": ("q2_moneyline", "q2_spreads"),
    "3Q": ("q3_moneyline", "q3_spreads"),
    "4Q": ("q4_moneyline", "q4_spreads"),
    "2H": ("second_half_moneyline", "second_half_spreads"),
}


def _market_gate(market: dict[str, Any]) -> None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError(f"inactive market entered graph: {market.get('id')}")
    description = str(market.get("description") or "")
    if "overtime is not included" not in description:
        raise RuntimeError(f"period overtime rule mismatch: {market.get('id')}")
    if "game is canceled entirely" not in description or "resolve 50-50" not in description:
        raise RuntimeError(f"cancellation rule mismatch: {market.get('id')}")
    prices = _pair(market.get("outcomePrices"), "outcomePrices")
    if any(not Decimal("0") <= Decimal(value) <= ONE for value in prices):
        raise RuntimeError(f"diagnostic price out of range: {market.get('id')}")


def _relation(
    event: dict[str, Any],
    scope: str,
    moneyline: dict[str, Any],
    spread: dict[str, Any],
) -> dict[str, Any]:
    _market_gate(moneyline)
    _market_gate(spread)
    ml_outcomes = _pair(moneyline.get("outcomes"), "moneyline outcomes")
    spread_outcomes = _pair(spread.get("outcomes"), "spread outcomes")
    if set(ml_outcomes) != set(spread_outcomes):
        raise RuntimeError(f"team identity mismatch: {spread.get('id')}")
    ml_description = str(moneyline.get("description") or "")
    if "If both teams score the same number of points" not in ml_description:
        raise RuntimeError(f"moneyline tie rule mismatch: {moneyline.get('id')}")
    line = Decimal(str(spread.get("line")))
    if line >= 0 or (-line % 1) != Decimal("0.5"):
        raise RuntimeError(f"spread is not negative half-point: {spread.get('id')}")
    favorite, opponent = spread_outcomes
    required_margin = int(-line + Decimal("0.5"))
    spread_description = str(spread.get("description") or "")
    if (
        f"{favorite} outscore the {opponent} by {required_margin} or more points"
        not in spread_description
    ):
        raise RuntimeError(f"spread line rule mismatch: {spread.get('id')}")
    ml_side = ml_outcomes.index(favorite)
    spread_side = spread_outcomes.index(opponent)
    ml_price, ml_source = _side_price(moneyline, ml_side)
    spread_price, spread_source = _side_price(spread, spread_side)
    complete = ml_price is not None and spread_price is not None
    side_sum = ml_price + spread_price if complete else None
    ml_diagnostic = Decimal(_pair(moneyline["outcomePrices"], "outcomePrices")[ml_side])
    spread_diagnostic = Decimal(_pair(spread["outcomePrices"], "outcomePrices")[spread_side])
    return {
        "event_id": str(event["id"]),
        "event_slug": str(event["slug"]),
        "event_title": str(event["title"]),
        "scope": scope,
        "favorite": favorite,
        "opponent": opponent,
        "favorite_required_margin": required_margin,
        "guaranteed_floor_pUSD": "1",
        "tie_payout_pUSD": "1.5",
        "price_complete": complete,
        "side_specific_rejection_sum_pUSD": None if side_sum is None else str(side_sum),
        "strict_side_specific_subfloor": complete and side_sum < ONE,
        "diagnostic_sum_pUSD": str(ml_diagnostic + spread_diagnostic),
        "legs": [
            {
                "market_id": str(moneyline["id"]),
                "outcome": favorite,
                "price_pUSD": None if ml_price is None else str(ml_price),
                "price_source": ml_source,
                "question": str(moneyline["question"]),
            },
            {
                "market_id": str(spread["id"]),
                "outcome": opponent,
                "price_pUSD": None if spread_price is None else str(spread_price),
                "price_source": spread_source,
                "question": str(spread["question"]),
            },
        ],
    }


def _rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = list(event.get("markets", []))
    rows: list[dict[str, Any]] = []
    for scope, (moneyline_type, spread_type) in SCOPES.items():
        moneylines = [
            market
            for market in markets
            if market.get("sportsMarketType") == moneyline_type
        ]
        spreads = [
            market for market in markets if market.get("sportsMarketType") == spread_type
        ]
        if not moneylines and not spreads:
            continue
        if len(moneylines) != 1:
            raise RuntimeError(
                f"expected one {scope} moneyline in event {event.get('id')}"
            )
        rows.extend(
            _relation(event, scope, moneylines[0], spread) for spread in spreads
        )
    return rows


def _row_key(row: dict[str, Any]) -> tuple:
    return (
        row["event_id"],
        row["scope"],
        row["favorite_required_margin"],
        [leg["market_id"] for leg in row["legs"]],
    )


def _price_key(row: dict[str, Any]) -> tuple:
    return (Decimal(row["side_specific_rejection_sum_pUSD"]), _row_key(row))


def adjudicate(contract: dict[str, Any]) -> dict[str, Any]:
    body = dict(contract)
    expected_hash = body.pop("contract_sha256", None)
    if _canonical_hash(body) != expected_hash:
        raise RuntimeError("contract hash mismatch")
    raw_path = _root_path(str(contract["source"]["raw_path"]))
    metadata_path = _root_path(str(contract["source"]["metadata_path"]))
    raw_bytes = raw_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    if _sha256(raw_bytes) != contract["source"]["raw_file_sha256"]:
        raise RuntimeError("raw source hash mismatch")
    if _sha256(metadata_bytes) != contract["source"]["metadata_file_sha256"]:
        raise RuntimeError("metadata file hash mismatch")
    metadata = json.loads(metadata_bytes)
    if metadata["result_sha256"] != contract["source"]["metadata_result_sha256"]:
        raise RuntimeError("metadata result identity mismatch")
    if metadata["capture"]["receipt"]["response_sha256"] != _sha256(raw_bytes):
        raise RuntimeError("raw receipt binding mismatch")
    for bound in [contract["implementation"], *contract["dependencies"]]:
        path = _root_path(str(bound["path"]))
        if _sha256(path.read_bytes()) != bound["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")
    payload = json.loads(raw_bytes)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or len(events) != contract["population"]["event_count"]:
        raise RuntimeError("catalog population mismatch")
    if sorted(str(event["id"]) for event in events) != contract["population"]["event_ids"]:
        raise RuntimeError("catalog event ids mismatch")
    rows = [row for event in events for row in _rows(event)]
    complete = [row for row in rows if row["price_complete"]]
    incomplete = [row for row in rows if not row["price_complete"]]
    candidates = [row for row in complete if row["strict_side_specific_subfloor"]]
    scope_summaries: dict[str, Any] = {}
    for scope in SCOPES:
        scoped = [row for row in rows if row["scope"] == scope]
        scoped_complete = [row for row in scoped if row["price_complete"]]
        if not scoped:
            continue
        scope_summaries[scope] = {
            "best_complete_relation": min(scoped_complete, key=_price_key),
            "price_complete_relation_count": len(scoped_complete),
            "price_incomplete_relation_count": len(scoped) - len(scoped_complete),
            "relation_count": len(scoped),
            "strict_side_specific_subfloor_count": sum(
                row["strict_side_specific_subfloor"] for row in scoped_complete
            ),
        }
    population_complete = not incomplete
    result: dict[str, Any] = {
        "schema_version": "polymarket-nfl-catalog-period-moneyline-spread-result-v1",
        "created_at_utc": contract["frozen_at_utc"],
        "contract": {"path": contract["contract_path"], "sha256": expected_hash},
        "source_binding": contract["source"],
        "population": contract["population"],
        "proof_contract": contract["proof_contract"],
        "pricing_contract": contract["pricing_contract"],
        "scope_screens": scope_summaries,
        "aggregate_screen": {
            "best_complete_relation": min(complete, key=_price_key),
            "price_complete_relation_count": len(complete),
            "price_incomplete_relation_count": len(incomplete),
            "relation_count": len(rows),
            "relations_sha256": _canonical_hash(sorted(rows, key=_row_key)),
            "strict_side_specific_subfloor_count": len(candidates),
        },
        "adjudication": {
            "accepted_edge": False,
            "book_or_fee_request_permitted": population_complete and bool(candidates),
            "deployment_ready": False,
            "market_direction_forecast_required": False,
            "next_action": (
                "freeze_one_exact_book_and_fee_batch_for_the_best_candidate"
                if population_complete and candidates
                else "do_not_request_books_or_fees_for_this_retained_population"
            ),
            "profitability_claim": False,
            "status": (
                "candidate_requires_separately_frozen_exact_depth_screen"
                if population_complete and candidates
                else "retained_population_price_incomplete_no_escalation"
                if incomplete
                else "terminal_retained_population_rejected_before_books_and_fees"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
        "dependencies": contract["dependencies"],
    }
    result["result_sha256"] = _canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen retained NFL period moneyline-spread covers."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("output already exists")
    result = adjudicate(contract)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "candidate_count": result["aggregate_screen"]["strict_side_specific_subfloor_count"],
                "incomplete_count": result["aggregate_screen"]["price_incomplete_relation_count"],
                "network_requests": 0,
                "relation_count": result["aggregate_screen"]["relation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
