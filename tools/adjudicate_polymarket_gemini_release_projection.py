"""Adjudicate one retained Gemini exact-date to deadline projection package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_release_date_deadline_graph import (
    _acquisition,
    _canonical_hash,
    _root_path,
    _sha256,
    _validate_market,
)


SCHEMA = "polymarket-gemini-release-projection-adjudication-v1"
ONE = Decimal("1")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _validate_contract(contract: dict[str, Any], path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid")
    implementation = contract["implementation"]
    implementation_path = _root_path(str(implementation["path"]))
    if _sha256(implementation_path.read_bytes()) != implementation["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if contract.get("authority") != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    if contract.get("promotion_eligible") is not False:
        raise RuntimeError("leaked discovery population must remain ineligible")


def _load_bound(path_text: str, expected_sha256: str) -> dict[str, Any]:
    path = _root_path(path_text)
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise RuntimeError(f"source hash mismatch: {path.name}")
    return _load_object(path)


def _event_from_catalog(
    catalog: dict[str, Any], *, event_id: str, event_slug: str, market_count: int
) -> dict[str, Any]:
    events = catalog.get("events")
    if not isinstance(events, list):
        raise RuntimeError("catalog lacks events")
    matches = [
        event
        for event in events
        if isinstance(event, dict)
        and str(event.get("id")) == event_id
        and event.get("slug") == event_slug
    ]
    if len(matches) != 1 or len(matches[0].get("markets") or []) != market_count:
        raise RuntimeError("exact-date event identity differs")
    return matches[0]


def _direct_event(
    event: dict[str, Any], *, event_id: str, event_slug: str, market_count: int
) -> dict[str, Any]:
    if not (
        str(event.get("id")) == event_id
        and event.get("slug") == event_slug
        and isinstance(event.get("markets"), list)
        and len(event["markets"]) == market_count
    ):
        raise RuntimeError("deadline event identity differs")
    return event


def _market_by_group(event: dict[str, Any], group: str) -> dict[str, Any]:
    matches = [
        market
        for market in event["markets"]
        if isinstance(market, dict) and market.get("groupItemTitle") == group
    ]
    if len(matches) != 1:
        raise RuntimeError(f"group is not unique: {group}")
    return matches[0]


def adjudicate(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    _validate_contract(contract, contract_path)
    source = contract["sources"]
    catalog = _load_bound(source["catalog"]["path"], source["catalog"]["sha256"])
    exact_event = _event_from_catalog(
        catalog,
        event_id=source["exact_event"]["event_id"],
        event_slug=source["exact_event"]["event_slug"],
        market_count=source["exact_event"]["market_count"],
    )
    deadline_event = _direct_event(
        _load_bound(
            source["deadline_event"]["path"], source["deadline_event"]["sha256"]
        ),
        event_id=source["deadline_event"]["event_id"],
        event_slug=source["deadline_event"]["event_slug"],
        market_count=source["deadline_event"]["market_count"],
    )

    fragments = list(contract["required_common_rule_fragments"])
    no_release_group = str(contract["exact_partition"]["no_release_group"])
    exact_markets = [
        market for market in exact_event["markets"] if isinstance(market, dict)
    ]
    no_release = _market_by_group(exact_event, no_release_group)
    release_markets = [market for market in exact_markets if market is not no_release]
    if len(release_markets) != int(contract["exact_partition"]["release_leg_count"]):
        raise RuntimeError("release-leg population differs")
    if len({str(market.get("groupItemTitle")) for market in exact_markets}) != len(
        exact_markets
    ):
        raise RuntimeError("exact partition contains duplicate groups")
    deadline_market = _market_by_group(
        deadline_event, str(contract["deadline_leg"]["group"])
    )

    for market in [*exact_markets, deadline_market]:
        _validate_market(market, required_fragments=fragments)
    if not (
        str(exact_event.get("startDate")) == source["exact_event"]["start_date"]
        and str(deadline_event.get("startDate"))
        == source["deadline_event"]["start_date"]
    ):
        raise RuntimeError("event start-date lineage differs")

    acquisitions = [_acquisition(market, outcome="Yes") for market in release_markets]
    acquisitions.append(_acquisition(deadline_market, outcome="No"))
    if any(row is None for row in acquisitions):
        missing = sum(row is None for row in acquisitions)
        return {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "contract": {
                "path": contract["contract_path"],
                "sha256": contract["contract_sha256"],
            },
            "source_binding": source,
            "scope": {
                "release_leg_count": len(release_markets),
                "total_leg_count": len(acquisitions),
                "missing_side_specific_leg_count": missing,
            },
            "adjudication": {
                "status": "terminal_missing_side_specific_acquisition_evidence",
                "accepted_edge": False,
                "deployment_ready": False,
                "profitability_claim": False,
                "book_request_justified": False,
            },
            "authority": contract["authority"],
        }
    legs = [row for row in acquisitions if row is not None]
    quantity = max(row["minimum_order_shares"] for row in legs)
    prices = [row["price_pUSD_per_share"] for row in legs]
    stressed_prices = [
        row["price_pUSD_per_share"] + row["tick_size_pUSD"] for row in legs
    ]
    metadata_cost = sum(prices, Decimal("0"))
    gross = quantity * (ONE - metadata_cost)
    if any(price >= ONE for price in stressed_prices):
        fee = None
        stressed = None
    else:
        fee = sum(
            (
                row["fee_model"](price, quantity, "taker")
                for row, price in zip(legs, stressed_prices, strict=True)
            ),
            Decimal("0"),
        )
        stressed = quantity * (ONE - sum(stressed_prices, Decimal("0"))) - fee
    public_legs = [
        {
            key: (_decimal_text(value) if isinstance(value, Decimal) else value)
            for key, value in row.items()
            if key != "fee_model"
        }
        for row in legs
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source_binding": source,
        "payoff_identity": {
            "optimistic_package": "YES on every exact-date release row except No release, plus NO on the cumulative September 30 deadline",
            "optimistic_floor_pUSD_per_share": "1",
            "prehistory_gap_seconds": int(contract["prehistory_gap_seconds"]),
            "source_proved_floor": False,
            "reason_not_source_proved": "the deadline event began before the exact-date partition; no qualifying release in that exclusive prehistory was not source-proved",
            "use": "rejection_only; an optimistic unproved floor may reject spend but never authorize books, promotion, or trading",
        },
        "scope": {
            "release_leg_count": len(release_markets),
            "deadline_no_leg_count": 1,
            "total_leg_count": len(legs),
            "quantity_shares_each_leg": _decimal_text(quantity),
            "promotion_eligible": False,
        },
        "economics": {
            "legs": public_legs,
            "metadata_cost_pUSD_per_share": _decimal_text(metadata_cost),
            "optimistic_metadata_gross_headroom_pUSD": _decimal_text(gross),
            "one_adverse_tick_per_leg_prices_pUSD": [
                _decimal_text(value) for value in stressed_prices
            ],
            "stressed_taker_fee_pUSD": _decimal_text(fee),
            "after_fee_one_tick_profit_floor_pUSD": _decimal_text(stressed),
            "passes_strict_metadata_gate": metadata_cost < ONE,
            "passes_fee_and_one_tick_gate": stressed is not None and stressed > 0,
        },
        "adjudication": {
            "status": "terminal_before_books_fee_and_tick_stress_rejection",
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "book_request_justified": False,
        },
        "authority": contract["authority"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    result = adjudicate(contract, contract_path)
    if "result_sha256" not in result:
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output = _root_path(str(contract["output_path"]))
    if output.exists():
        raise RuntimeError("one-use output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "total_leg_count": result["scope"]["total_leg_count"],
                "passes_fee_and_one_tick_gate": result.get("economics", {}).get(
                    "passes_fee_and_one_tick_gate", False
                ),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
