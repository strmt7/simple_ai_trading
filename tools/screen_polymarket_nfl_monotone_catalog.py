from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _json_ready,
    _relations,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_exact_nfl_monotone_prefilter import (
    _margin_markets,
    _total_markets,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_mlb_cross_period_catalog import (
    _frozen_instant,
    _instant,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-nfl-monotone-catalog-result-v1"


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    frozen = _frozen_instant(contract.get("frozen_at_utc"))
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    if _instant(contract["capture"]["start_time_min"], "start_time_min") <= frozen:
        raise RuntimeError("catalog must begin after the frozen instant")
    if _instant(contract["capture"]["start_time_max"], "start_time_max") <= _instant(
        contract["capture"]["start_time_min"], "start_time_min"
    ):
        raise RuntimeError("catalog time window is empty")
    if contract["capture"]["limit"] != 500:
        raise RuntimeError("catalog must use the documented maximum page size")
    if contract["capture"]["request_count"] != 1:
        raise RuntimeError("catalog must freeze exactly one request")
    if contract["authority"] != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("catalog authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _event_series_matches(event: dict[str, Any], series_id: str) -> bool:
    return any(str(row.get("id")) == series_id for row in event.get("series", []))


def _screen_event(event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    markets = [
        row
        for row in event.get("markets", [])
        if row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
    ]
    moneylines = [row for row in markets if row.get("sportsMarketType") == "moneyline"]
    if len(moneylines) != 1:
        raise RuntimeError(f"expected one moneyline, found {len(moneylines)}")
    team_a, team_b, margins = _margin_markets(markets, moneylines[0])
    totals = _total_markets(markets)
    families = {"full_game_margin": margins, "full_game_total": totals}
    relations = [
        relation
        for family, rows in families.items()
        for relation in _relations(
            family, rows, minimum_state=-8 if family.endswith("margin") else 0
        )
    ]
    if not relations:
        raise RuntimeError("no exact monotone relations")
    event_ref = {
        "event_id": str(event["id"]),
        "event_slug": str(event["slug"]),
        "event_title": str(event["title"]),
        "start_time_utc": _instant(event.get("startTime"), "startTime")
        .isoformat()
        .replace("+00:00", "Z"),
    }
    return (
        [{**event_ref, **relation} for relation in relations],
        {
            **event_ref,
            "team_a": team_a,
            "team_b": team_b,
            "active_accepting_market_count": len(markets),
            "margin_threshold_count": len(margins),
            "total_threshold_count": len(totals),
            "relation_count": len(relations),
        },
    )


def _candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        Decimal(str(row["displayed_price_sum_per_share_pUSD"])),
        row["start_time_utc"],
        row["event_slug"],
        row["family"],
        int(row["superset_threshold"]),
        int(row["subset_threshold"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen one bounded future NFL catalog for monotone payoffs."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)

    result_path = _root_path(str(contract["outputs"]["result_path"]))
    raw_path = _root_path(str(contract["outputs"]["raw_path"]))
    journal_path = _root_path(str(contract["outputs"]["journal_path"]))
    if journal_path.parent.exists() or result_path.exists():
        raise RuntimeError("one-use output already exists")
    raw_path.parent.mkdir(parents=True)
    raw, receipt = _request(
        method="GET",
        url=str(contract["capture"]["url"]),
        body=b"",
        name="future-nfl-monotone-catalog",
        raw_path=raw_path,
        raw_relative_path=str(contract["outputs"]["raw_path"]),
        journal_path=journal_path,
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise RuntimeError("keyset catalog response shape changed")
    events = payload["events"]
    if len(events) > int(contract["capture"]["limit"]):
        raise RuntimeError("catalog exceeded its frozen limit")
    population_complete = "next_cursor" not in payload
    if not population_complete and not isinstance(payload.get("next_cursor"), str):
        raise RuntimeError("catalog cursor shape changed")

    minimum = _instant(contract["capture"]["start_time_min"], "start_time_min")
    maximum = _instant(contract["capture"]["start_time_max"], "start_time_max")
    series_id = str(contract["capture"]["series_id"])
    relations: list[dict[str, Any]] = []
    included_events: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for event in events:
        event_slug = str(event.get("slug"))
        try:
            start = _instant(event.get("startTime"), "startTime")
            if not (
                event.get("active") is True
                and event.get("closed") is False
                and minimum <= start <= maximum
                and _event_series_matches(event, series_id)
            ):
                raise RuntimeError("event fails exact frozen population filter")
            event_relations, event_summary = _screen_event(event)
        except (KeyError, RuntimeError, ValueError, ArithmeticError) as exc:
            exclusions.append({"event_slug": event_slug, "reason": str(exc)})
            continue
        relations.extend(event_relations)
        included_events.append(event_summary)

    candidates = [row for row in relations if row["passes_strictly_below_payout_gate"]]
    candidates.sort(key=_candidate_key)
    best = candidates[0] if candidates else None
    depth_candidate = best if population_complete else None
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "returned_event_count": len(events),
            "limit": contract["capture"]["limit"],
            "next_cursor_present": "next_cursor" in payload,
            "population_complete_under_frozen_filter": population_complete,
            "frozen_time_window": {
                "start_time_min": contract["capture"]["start_time_min"],
                "start_time_max": contract["capture"]["start_time_max"],
            },
        },
        "screen": {
            "included_event_count": len(included_events),
            "excluded_event_count": len(exclusions),
            "included_events": included_events,
            "exclusions": exclusions,
            "complete_relation_count": len(relations),
            "relations": relations,
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "best_candidate": best,
            "depth_candidate": depth_candidate,
        },
        "adjudication": {
            "status": (
                "incomplete_catalog_no_depth_escalation"
                if not population_complete
                else (
                    "candidate_requires_separately_frozen_exact_depth_screen"
                    if candidates
                    else "frozen_catalog_window_rejected_before_books_and_fees"
                )
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "stop_without_adaptive_pagination_or_depth_access"
                if not population_complete
                else (
                    "freeze_one_exact_depth_screen_for_only_the_best_distinct_candidate"
                    if candidates
                    else "stop_without_any_book_or_fee_request"
                )
            ),
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/screen_polymarket_nfl_monotone_catalog.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "returned_event_count": len(events),
                "included_event_count": len(included_events),
                "excluded_event_count": len(exclusions),
                "relation_count": len(relations),
                "candidate_count": len(candidates),
                "population_complete": population_complete,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
