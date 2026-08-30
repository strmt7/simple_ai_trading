"""Adjudicate one frozen CFB catalog from already-retained Gamma bytes."""

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
    _root_path,
    _sha256,
)
from tools.screen_polymarket_cfb_monotone_catalog import (
    _candidate_key,
    _event_series_matches,
    _screen_event,
)
from tools.screen_polymarket_mlb_cross_period_catalog import _instant


SCHEMA = "polymarket-cfb-catalog-retained-adjudication-v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _screen_catalog(
    payload: dict[str, Any], population: dict[str, Any]
) -> dict[str, Any]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise RuntimeError("keyset catalog response shape changed")
    limit = int(population["limit"])
    if len(events) > limit:
        raise RuntimeError("catalog exceeded its frozen limit")
    population_complete = "next_cursor" not in payload
    if not population_complete and not isinstance(payload.get("next_cursor"), str):
        raise RuntimeError("catalog cursor shape changed")

    minimum = _instant(population["start_time_min"], "start_time_min")
    maximum = _instant(population["start_time_max"], "start_time_max")
    series_id = str(population["series_id"])
    consumed = set(population["excluded_consumed_event_slugs"])
    relations: list[dict[str, Any]] = []
    included_events: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for event in events:
        event_slug = str(event.get("slug"))
        if event_slug in consumed:
            exclusions.append(
                {"event_slug": event_slug, "reason": "excluded_consumed_event"}
            )
            continue
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
    best = min(relations, key=_candidate_key) if relations else None
    depth_candidate = candidates[0] if candidates and population_complete else None
    return {
        "returned_event_count": len(events),
        "population_complete_under_frozen_filter": population_complete,
        "next_cursor_present": "next_cursor" in payload,
        "excluded_consumed_event_count": sum(
            row["reason"] == "excluded_consumed_event" for row in exclusions
        ),
        "included_event_count": len(included_events),
        "excluded_event_count": len(exclusions),
        "included_events": included_events,
        "exclusions": exclusions,
        "complete_relation_count": len(relations),
        "relations": relations,
        "candidate_count_strictly_below_payout_floor": len(candidates),
        "best_relation": best,
        "depth_candidate": depth_candidate,
    }


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    population = contract["population"]
    if population != {
        "ascending": True,
        "closed": False,
        "excluded_consumed_event_slugs": [
            "cfb-ballst-ohiost-2026-09-05",
            "cfb-clmsn-lsu-2026-09-05",
        ],
        "limit": 500,
        "order": "startTime",
        "series_id": "12756",
        "start_time_max": "2026-09-05T23:59:59Z",
        "start_time_min": "2026-09-05T00:00:00Z",
        "tag_slug": "cfb",
    }:
        raise RuntimeError("population contract changed")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = _load_object(contract_path)
    _validate_contract(contract, contract_path)

    source_result_path = _root_path(contract["outputs"]["result_path"])
    raw_path = _root_path(contract["outputs"]["raw_path"])
    result_path = _root_path(contract["adjudication_output_path"])
    if result_path.exists():
        raise RuntimeError("one-use adjudication output already exists")
    source_result = _load_object(source_result_path)
    if _canonical_hash(source_result, "result_sha256") != source_result.get(
        "result_sha256"
    ):
        raise RuntimeError("source result hash mismatch")
    if source_result["contract"] != {
        "path": contract["contract_path"],
        "sha256": contract["contract_sha256"],
    }:
        raise RuntimeError("source result contract mismatch")
    if source_result["source_gate"]["passed"] is not True:
        raise RuntimeError("public source gate did not pass")
    raw = raw_path.read_bytes()
    receipt = source_result["capture"]["receipt"]
    if (
        receipt["url"] != contract["request"]["url"]
        or receipt["response_sha256"] != _sha256(raw)
        or receipt["response_bytes"] != len(raw)
        or receipt["status_code"] != 200
    ):
        raise RuntimeError("retained source receipt mismatch")

    payload = json.loads(raw)
    screen = _screen_catalog(payload, contract["population"])
    complete = screen["population_complete_under_frozen_filter"]
    candidate = screen["depth_candidate"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "retained_source": {
            "source_result_path": contract["outputs"]["result_path"],
            "source_result_sha256": source_result["result_sha256"],
            "raw_path": contract["outputs"]["raw_path"],
            "raw_sha256": _sha256(raw),
            "raw_bytes": len(raw),
        },
        "screen": screen,
        "adjudication": {
            "status": (
                "incomplete_catalog_no_depth_escalation"
                if not complete
                else "candidate_requires_separately_frozen_exact_depth_screen"
                if candidate is not None
                else "frozen_catalog_window_rejected_before_books_and_fees"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "stop_without_adaptive_pagination_or_depth_access"
                if not complete
                else "freeze_one_exact_depth_screen_for_only_the_best_distinct_candidate"
                if candidate is not None
                else "stop_without_any_book_or_fee_request"
            ),
        },
        "authority": contract["economic_authority"],
        "implementation": {
            "path": "tools/adjudicate_polymarket_cfb_catalog_retained.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
        "diagnostic": {
            "best_displayed_headroom_pUSD": (
                str(
                    Decimal("1")
                    - Decimal(str(screen["best_relation"]["displayed_price_sum_per_share_pUSD"]))
                )
                if screen["best_relation"] is not None
                else None
            )
        },
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "returned_event_count": screen["returned_event_count"],
                "included_event_count": screen["included_event_count"],
                "excluded_consumed_event_count": screen[
                    "excluded_consumed_event_count"
                ],
                "relation_count": screen["complete_relation_count"],
                "candidate_count": screen[
                    "candidate_count_strictly_below_payout_floor"
                ],
                "population_complete": complete,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
