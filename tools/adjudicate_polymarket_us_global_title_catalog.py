"""Join a frozen Polymarket US event catalog to retained Global titles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _validate_capture(contract: dict[str, Any]) -> dict[str, Any]:
    outputs = contract["outputs"]
    source_result = json.loads(
        _root_path(outputs["result_path"]).read_text(encoding="ascii")
    )
    if _canonical_hash(source_result, "result_sha256") != source_result[
        "result_sha256"
    ]:
        raise RuntimeError("source result hash mismatch")
    if not source_result["source_gate"]["passed"]:
        raise RuntimeError("Polymarket US catalog source gate failed")
    journal = [
        json.loads(line)
        for line in _root_path(outputs["journal_path"])
        .read_text(encoding="ascii")
        .splitlines()
        if line.strip()
    ]
    if len(journal) != 2 or journal[-1].get("status_code") != 200:
        raise RuntimeError("catalog journal is not one successful GET")
    raw = _root_path(outputs["raw_path"]).read_bytes()
    if journal[-1].get("response_sha256") != _sha256(raw):
        raise RuntimeError("catalog raw bytes differ from journal receipt")
    return source_result


def _global_events(contract: dict[str, Any]) -> list[dict[str, Any]]:
    source = contract["global_population"]
    path = _root_path(source["path"])
    if _sha256(path.read_bytes()) != source["sha256"]:
        raise RuntimeError("retained Global catalog hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload if isinstance(payload, list) else payload.get("events", [])
    selected = [
        event
        for event in events
        if event.get("active") is True
        and event.get("closed") is False
        and event.get("endDate") == source["exact_end_date_utc"]
    ]
    if len(selected) != source["expected_event_count"]:
        raise RuntimeError("retained Global event population changed")
    if len({event.get("title") for event in selected}) != len(selected):
        raise RuntimeError("retained Global titles are not unique")
    return selected


def _market_signature(market: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": market.get("question"),
        "description": market.get("description"),
        "endDate": market.get("endDate"),
        "outcomes": market.get("outcomes"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "archived": market.get("archived"),
        "hidden": market.get("hidden"),
    }


def _event_candidates(
    global_events: list[dict[str, Any]], us_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    global_by_title = {event.get("title"): event for event in global_events}
    candidates: list[dict[str, Any]] = []
    for us_event in us_events:
        title = us_event.get("title")
        global_event = global_by_title.get(title)
        if global_event is None:
            continue
        global_markets = global_event.get("markets") or []
        us_markets = us_event.get("markets") or []
        global_by_question = {
            market.get("question"): market for market in global_markets
        }
        exact_market_rows: list[dict[str, Any]] = []
        for us_market in us_markets:
            question = us_market.get("question")
            global_market = global_by_question.get(question)
            if global_market is None:
                continue
            global_signature = _market_signature(global_market)
            us_signature = _market_signature(us_market)
            comparisons = {
                key: global_signature[key] == us_signature[key]
                for key in global_signature
                if key not in {"archived", "hidden"}
            }
            comparisons["both_unarchived"] = (
                global_market.get("archived") in (None, False)
                and us_market.get("archived") in (None, False)
            )
            comparisons["both_visible"] = (
                global_market.get("hidden") in (None, False)
                and us_market.get("hidden") in (None, False)
            )
            exact_market_rows.append(
                {
                    "global_market_id": str(global_market.get("id")),
                    "us_market_id": str(us_market.get("id")),
                    "question": question,
                    "comparisons": comparisons,
                    "exact_source_identity": all(comparisons.values()),
                }
            )
        event_comparisons = {
            "description_exact": global_event.get("description")
            == us_event.get("description"),
            "end_date_exact": global_event.get("endDate")
            == us_event.get("endDate"),
            "both_active": global_event.get("active") is True
            and us_event.get("active") is True,
            "both_open": global_event.get("closed") is False
            and us_event.get("closed") is False,
            "both_unarchived": global_event.get("archived") in (None, False)
            and us_event.get("archived") in (None, False),
            "market_count_exact": len(global_markets) == len(us_markets),
            "every_market_question_joined": len(exact_market_rows)
            == len(global_markets)
            == len(us_markets),
        }
        candidates.append(
            {
                "title": title,
                "global_event_id": str(global_event.get("id")),
                "us_event_id": str(us_event.get("id")),
                "event_comparisons": event_comparisons,
                "market_rows": exact_market_rows,
                "exact_source_identity": all(event_comparisons.values())
                and bool(exact_market_rows)
                and all(row["exact_source_identity"] for row in exact_market_rows),
            }
        )
    candidates.sort(key=lambda row: row["title"])
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()

    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    source_result = _validate_capture(contract)
    output_path = _root_path(contract["outputs"]["adjudication_path"])
    if output_path.exists():
        raise RuntimeError("one-use adjudication output already exists")

    global_events = _global_events(contract)
    us_payload = json.loads(
        _root_path(contract["outputs"]["raw_path"]).read_text(encoding="utf-8")
    )
    us_events = us_payload.get("events")
    if not isinstance(us_events, list):
        raise RuntimeError("Polymarket US events array missing")
    requested_limit = contract["us_population"]["requested_limit"]
    population_complete = len(us_events) < requested_limit
    filter_exact = all(
        event.get("active") is True
        and event.get("closed") is False
        and event.get("archived") in (None, False)
        for event in us_events
    )
    candidates = _event_candidates(global_events, us_events)
    exact = [row for row in candidates if row["exact_source_identity"]]
    result: dict[str, Any] = {
        "schema_version": "polymarket-us-global-title-catalog-adjudication-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source_result": {
            "path": contract["outputs"]["result_path"],
            "sha256": source_result["result_sha256"],
        },
        "population": {
            "global_event_count": len(global_events),
            "us_event_count": len(us_events),
            "us_requested_limit": requested_limit,
            "us_population_complete_by_strict_below_limit": population_complete,
            "us_filters_exact": filter_exact,
        },
        "join": {
            "exact_event_title_match_count": len(candidates),
            "exact_source_identity_count": len(exact),
            "candidates": candidates,
            "economic_fields_read_or_printed": 0,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "candidate_for_separate_frozen_economics_gate": bool(exact)
            and population_complete
            and filter_exact,
            "status": (
                "exact_source_identity_candidate_requires_separate_economics_gate"
                if exact and population_complete and filter_exact
                else "no_exact_cross_venue_source_identity_in_frozen_populations"
            ),
            "next_action": (
                "freeze one exact candidate BBO and fee comparison without orders"
                if exact and population_complete and filter_exact
                else "stop without any BBO book fee account credential or order request"
            ),
        },
        "limitations": contract["limitations"],
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/adjudicate_polymarket_us_global_title_catalog.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "exact_source_identity_count": len(exact),
                "exact_title_match_count": len(candidates),
                "global_event_count": len(global_events),
                "payloads_printed": 0,
                "us_event_count": len(us_events),
                "us_population_complete": population_complete,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
