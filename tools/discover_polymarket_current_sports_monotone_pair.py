"""Freeze one current sports event inventory request without accessing books."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


TEMPLATE_SCHEMA = "polymarket-current-sports-monotone-pair-discovery-template-v1"
CONTRACT_SCHEMA = "polymarket-current-sports-monotone-pair-discovery-contract-v1"
RESULT_SCHEMA = "polymarket-current-sports-monotone-pair-discovery-result-v1"
EXPECTED_TITLE = "Colorado Rockies vs. Washington Nationals"
EXPECTED_ENDPOINT = "https://gamma-api.polymarket.com/events/keyset"
EXPECTED_QUERY = {
    "closed": "false",
    "title_search": EXPECTED_TITLE,
    "limit": "20",
    "order": "volume24hr",
    "ascending": "false",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _freeze_contract(*, template_path: Path, contract_path: Path) -> dict[str, object]:
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen contract: {contract_path}")
    template_bytes = template_path.read_bytes()
    template = _mapping(json.loads(template_bytes), name="contract template")
    if (
        template.get("schema_version") != TEMPLATE_SCHEMA
        or template.get("status") != "prefreeze_template_no_api_access_yet"
    ):
        raise ValueError("unexpected or already-consumed template")
    source = _mapping(template.get("source_contract"), name="source contract")
    query = _mapping(source.get("query"), name="source query")
    if (
        source.get("method") != "GET"
        or source.get("endpoint") != EXPECTED_ENDPOINT
        or query != EXPECTED_QUERY
        or source.get("maximum_requests") != 1
        or source.get("retry_permitted") is not False
    ):
        raise ValueError("template differs from the exact one-request source contract")
    frozen_at = datetime.now(timezone.utc)
    contract = dict(template)
    contract["schema_version"] = CONTRACT_SCHEMA
    contract["status"] = "frozen_before_one_public_Gamma_request"
    contract["frozen_at_utc"] = frozen_at.isoformat().replace("+00:00", "Z")
    contract["template"] = {
        "path": template_path.as_posix(),
        "sha256": _sha256(template_bytes),
    }
    contract["contract_sha256"] = ""
    contract["contract_sha256"] = _canonical_hash(contract, field="contract_sha256")
    write_bytes_atomic(
        contract_path,
        (json.dumps(contract, indent=2, ensure_ascii=True) + "\n").encode("ascii"),
    )
    retained = _mapping(json.loads(contract_path.read_bytes()), name="frozen contract")
    if _canonical_hash(retained, field="contract_sha256") != retained.get(
        "contract_sha256"
    ):
        raise ValueError("persisted contract hash differs")
    retained_time = datetime.fromisoformat(
        str(retained.get("frozen_at_utc")).replace("Z", "+00:00")
    )
    if retained_time.tzinfo is None or retained_time > datetime.now(timezone.utc):
        raise ValueError("persisted contract timestamp is invalid or future")
    return retained


def _market_summary(raw_market: object) -> dict[str, object]:
    market = _mapping(raw_market, name="embedded market")
    return {
        key: market.get(key)
        for key in (
            "id",
            "question",
            "description",
            "active",
            "closed",
            "acceptingOrders",
            "acceptingOrdersTimestamp",
            "endDate",
            "eventStartTime",
            "gameId",
            "sportsMarketType",
            "line",
            "groupItemTitle",
            "outcomes",
            "clobTokenIds",
            "enableOrderBook",
            "negRisk",
            "feesEnabled",
            "feeSchedule",
            "secondsDelay",
            "orderMinSize",
            "orderPriceMinTickSize",
        )
    }


def _capture(
    *,
    contract: Mapping[str, object],
    raw_path: Path,
    journal_path: Path,
) -> tuple[object, dict[str, object]]:
    source = _mapping(contract.get("source_contract"), name="source contract")
    query = _mapping(source.get("query"), name="source query")
    started_ms = time.time_ns() // 1_000_000
    response = requests.get(
        str(source["endpoint"]),
        params=query,
        headers={
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-public-sports-research/1.0",
        },
        timeout=30,
    )
    completed_ms = time.time_ns() // 1_000_000
    raw = response.content
    write_bytes_atomic(raw_path, raw)
    receipt = {
        "name": "polymarket-current-sports-title-search",
        "transport": "HTTPS",
        "method": "GET",
        "url": response.url,
        "status_code": response.status_code,
        "requested_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "raw_path": raw_path.as_posix(),
    }
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("xb") as stream:
        stream.write((_canonical_json(receipt) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    response.raise_for_status()
    return response.json(), receipt


def _evaluate(
    *,
    contract_path: Path,
    contract: Mapping[str, object],
    payload: object,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    body = _mapping(payload, name="keyset response")
    events = [
        _mapping(value, name="event")
        for value in _list(body.get("events"), name="events")
    ]
    exact_events = [
        event
        for event in events
        if str(event.get("title") or "").strip().casefold() == EXPECTED_TITLE.casefold()
    ]
    summaries = []
    for event in exact_events:
        markets = [
            _market_summary(value)
            for value in _list(event.get("markets"), name="event markets")
        ]
        active_markets = [
            market
            for market in markets
            if market["active"] is True
            and market["closed"] is False
            and market["acceptingOrders"] is True
        ]
        summaries.append(
            {
                "id": event.get("id"),
                "slug": event.get("slug"),
                "title": event.get("title"),
                "description": event.get("description"),
                "resolutionSource": event.get("resolutionSource"),
                "active": event.get("active"),
                "closed": event.get("closed"),
                "endDate": event.get("endDate"),
                "eventDate": event.get("eventDate"),
                "gameId": event.get("gameId"),
                "active_accepting_market_count": len(active_markets),
                "observed_sports_market_types": sorted(
                    {
                        str(market["sportsMarketType"])
                        for market in active_markets
                        if market["sportsMarketType"] is not None
                    }
                ),
                "markets": active_markets,
            }
        )
    partial = body.get("next_cursor") not in (None, "", "LTE=")
    possible_pair_inventory = any(
        len(summary["observed_sports_market_types"]) >= 2 for summary in summaries
    )
    if not summaries:
        status = "no_exact_title_event_in_frozen_page"
    elif not possible_pair_inventory:
        status = "exact_event_without_multiple_sports_market_types"
    else:
        status = "exact_event_has_multiple_market_types_rules_review_required"
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract_path.as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "authority": {
            "public_unauthenticated_GET_requests": 1,
            "authenticated_requests": 0,
            "book_or_price_requests": 0,
            "account_state_accessed": False,
            "orders_wallets_or_transactions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "receipt": dict(receipt),
            "returned_event_count": len(events),
            "next_cursor": body.get("next_cursor"),
            "population_partial": partial,
            "exact_title_event_count": len(summaries),
        },
        "discovery": {
            "status": status,
            "possible_pair_inventory": possible_pair_inventory,
            "exact_events": summaries,
        },
        "adjudication": {
            "accepted_edge": False,
            "profitability_claim": False,
            "public_after_cost_profit_floor_pUSD": "0",
            "deployment_ready": False,
            "trading_authority": False,
            "next_action": (
                "manually_prove_exact_joint_payoffs_from_retained_rules_before_"
                "separately_freezing_any_book_request"
                if possible_pair_inventory
                else "stop_this_exact_displayed_lead_without_an_adaptive_request"
            ),
        },
        "implementation": {
            "path": "tools/discover_polymarket_current_sports_monotone_pair.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, field="result_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.contract_output, args.raw, args.journal, args.output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite one-use output: {path}")
    contract = _freeze_contract(
        template_path=args.template,
        contract_path=args.contract_output,
    )
    payload, receipt = _capture(
        contract=contract,
        raw_path=args.raw,
        journal_path=args.journal,
    )
    result = _evaluate(
        contract_path=args.contract_output,
        contract=contract,
        payload=payload,
        receipt=receipt,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["capture"], indent=2))
    print(json.dumps(result["discovery"], indent=2))
    print(json.dumps(result["adjudication"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
