"""Adjudicate pairwise-NO floors in retained fixed-NegRisk events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_release_date_deadline_graph import (
    ONE,
    _acquisition,
    _decimal_text,
)
from tools.screen_polymarket_negrisk_complete_set_catalog import (
    _eligible_event,
    _json_array,
)


CONTRACT_SCHEMA = "polymarket-retained-negrisk-pairwise-no-contract-v1"
RESULT_SCHEMA = "polymarket-retained-negrisk-pairwise-no-result-v1"


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be an object")
    return dict(value)


def _load_retained(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name="population")


def _event_map(population: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = population.get("events")
    if not isinstance(values, list):
        raise RuntimeError("retained population events must be a list")
    events: dict[str, dict[str, Any]] = {}
    for value in values:
        event = _mapping(value, name="retained event")
        event_id = str(event.get("id") or "")
        if not event_id.isdigit() or event_id in events:
            raise RuntimeError("retained event identity is invalid or duplicated")
        events[event_id] = event
    return events


def _event_markets(
    event: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not _eligible_event(dict(event)):
        raise RuntimeError("frozen event is no longer an eligible fixed-NegRisk event")
    for field in ("id", "slug", "title"):
        if str(event.get(field) or "") != str(expected[field]):
            raise RuntimeError(f"frozen event {field} changed")
    values = event.get("markets")
    if not isinstance(values, list) or len(values) != expected["market_count"]:
        raise RuntimeError("frozen event market count changed")
    neg_risk_market_id = str(event.get("negRiskMarketID") or "").lower()
    if len(neg_risk_market_id) != 66 or not neg_risk_market_id.startswith("0x"):
        raise RuntimeError("event NegRisk market ID is invalid")

    markets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    seen_tokens: set[str] = set()
    for value in values:
        market = _mapping(value, name="retained market")
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
            raise RuntimeError("event contains an unavailable or incompatible market")
        outcomes = _json_array(market.get("outcomes"), "outcomes")
        tokens = [
            str(item)
            for item in _json_array(market.get("clobTokenIds"), "clobTokenIds")
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


def _validate_contract(
    contract: Mapping[str, Any], path: Path
) -> tuple[Path, dict[str, Any], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("unexpected contract schema")
    if contract.get("status") != "frozen_before_one_zero_network_retained_adjudication":
        raise RuntimeError("contract status changed")
    if _canonical_hash(dict(contract), "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")

    authority = {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 0,
        "signed_requests": 0,
        "trading_authority": False,
    }
    if contract.get("authority") != authority:
        raise RuntimeError("authority boundary changed")

    retained = _root_path(str(contract["retained_input"]["path"]))
    if _sha256(retained.read_bytes()) != contract["retained_input"]["sha256"]:
        raise RuntimeError("retained input hash mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")

    population = _load_retained(retained)
    events = _event_map(population)
    expected_events = contract.get("events")
    if not isinstance(expected_events, list) or len(expected_events) != 3:
        raise RuntimeError("contract must freeze exactly three events")
    selected: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    seen_event_ids: set[str] = set()
    for value in expected_events:
        expected = _mapping(value, name="frozen event")
        event_id = str(expected.get("id") or "")
        if event_id in seen_event_ids or event_id not in events:
            raise RuntimeError("frozen event is missing or duplicated")
        if not isinstance(expected.get("market_count"), int):
            raise RuntimeError("frozen market count must be an integer")
        seen_event_ids.add(event_id)
        event = events[event_id]
        selected.append((event, _event_markets(event, expected)))
    return retained, population, selected


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
    first_no = _acquisition(first, outcome="No")
    second_no = _acquisition(second, outcome="No")
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
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    retained, _, selected = _validate_contract(contract, contract_path)
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
                    "retained_input": str(retained.relative_to(_root_path("."))),
                },
                sort_keys=True,
            )
        )
        return

    result_path = _root_path(str(contract["output_path"]))
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
            "price_gate": "direct NO bestAsk when available or conservative 1 minus direct YES bestBid; Gamma is rejection-only",
            "payout_identity": "two distinct outcomes in one fixed-NegRisk event are mutually exclusive, so NO(A) plus NO(B) pays at least one pUSD",
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
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
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
