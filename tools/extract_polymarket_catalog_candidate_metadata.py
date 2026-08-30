"""Extract the precommitted best catalog candidate from retained Gamma bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)


MARKET_FIELDS = (
    "id",
    "slug",
    "question",
    "description",
    "active",
    "closed",
    "acceptingOrders",
    "endDate",
    "sportsMarketType",
    "line",
    "groupItemTitle",
    "outcomes",
    "outcomePrices",
    "conditionId",
    "clobTokenIds",
    "enableOrderBook",
    "negRisk",
    "feesEnabled",
    "feeSchedule",
    "takerBaseFee",
    "secondsDelay",
    "orderMinSize",
    "orderPriceMinTickSize",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = _load(contract_path)
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    implementation = _root_path(contract["implementation"]["path"])
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")

    source_result_path = _root_path(contract["source_result"]["path"])
    source_result = _load(source_result_path)
    if (
        _canonical_hash(source_result, "result_sha256")
        != contract["source_result"]["result_sha256"]
    ):
        raise RuntimeError("source result hash mismatch")
    selected = source_result["screen"]["depth_candidate"]
    if selected is None or {
        "event_id": selected["event_id"],
        "event_slug": selected["event_slug"],
    } != contract["selected_event"]:
        raise RuntimeError("selected event differs from precommitted candidate")

    raw_path = _root_path(contract["raw_source"]["path"])
    raw = raw_path.read_bytes()
    if _sha256(raw) != contract["raw_source"]["sha256"]:
        raise RuntimeError("raw source hash mismatch")
    payload = json.loads(raw)
    matches = [
        event
        for event in payload["events"]
        if str(event.get("id")) == selected["event_id"]
        and event.get("slug") == selected["event_slug"]
    ]
    if len(matches) != 1:
        raise RuntimeError("selected event is not unique in retained catalog")
    event = matches[0]
    active = [
        {key: market.get(key) for key in MARKET_FIELDS}
        for market in event.get("markets", [])
        if market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
    ]
    required_ids = set(contract["required_market_ids"])
    if not required_ids.issubset({str(row["id"]) for row in active}):
        raise RuntimeError("selected package market is absent")

    result: dict[str, Any] = {
        "schema_version": "polymarket-catalog-candidate-metadata-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source": {
            "result": contract["source_result"],
            "raw": contract["raw_source"],
        },
        "event": {
            key: event.get(key)
            for key in (
                "id",
                "slug",
                "title",
                "startTime",
                "active",
                "closed",
            )
        },
        "selection": selected,
        "discovery": {
            "active_accepting_market_count": len(active),
            "active_accepting_markets": active,
        },
        "authority": {
            "network_requests": 0,
            "credentials_used": False,
            "orders_or_transactions": 0,
        },
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output = _root_path(contract["output_path"])
    if output.exists():
        raise RuntimeError("metadata output already exists")
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "event_slug": event["slug"],
                "active_accepting_market_count": len(active),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
