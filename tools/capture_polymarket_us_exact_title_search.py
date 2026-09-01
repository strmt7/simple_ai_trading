"""Capture one frozen Polymarket US exact-title search without exposing economics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(encoded)


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _frozen_instant(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError("frozen_at_utc is unparsable") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError("frozen_at_utc must carry the UTC offset")
    if parsed > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is in the future")
    return parsed


def _journal(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(
    *, url: str, raw_path: Path, journal_path: Path, raw_relative_path: str
) -> tuple[bytes, dict[str, Any]]:
    intent = {
        "method": "GET",
        "name": "polymarket-us-exact-title-search",
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": time.time_ns() // 1_000_000,
        "url": url,
    }
    _journal(journal_path, intent)
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": "simple-ai-trading-public-research/1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
            status_code = response.status
    except HTTPError as exc:
        raw = exc.read()
        status_code = exc.code
        raw_path.write_bytes(raw)
        _journal(
            journal_path,
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": raw_relative_path,
                "response_bytes": len(raw),
                "response_sha256": _sha256(raw),
                "status_code": status_code,
            },
        )
        raise
    raw_path.write_bytes(raw)
    receipt = {
        **intent,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "phase": "completed",
        "raw_path": raw_relative_path,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "status_code": status_code,
    }
    _journal(journal_path, receipt)
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    return raw, receipt


def _market_metadata(market: dict[str, Any]) -> dict[str, Any]:
    sides = market.get("marketSides")
    side_metadata = []
    if isinstance(sides, list):
        side_metadata = [
            {
                key: side.get(key)
                for key in ("id", "identifier", "description", "long", "tradable")
            }
            for side in sides
            if isinstance(side, dict)
        ]
    return {
        key: market.get(key)
        for key in (
            "id",
            "slug",
            "question",
            "title",
            "description",
            "resolutionSource",
            "rulesDisclaimer",
            "active",
            "closed",
            "archived",
            "hidden",
            "endDate",
            "outcomes",
            "marketType",
        )
    } | {"marketSides": side_metadata}


def _event_metadata(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: event.get(key)
        for key in (
            "id",
            "ticker",
            "slug",
            "title",
            "subtitle",
            "description",
            "resolutionSource",
            "startDate",
            "endDate",
            "active",
            "closed",
            "archived",
            "hidden",
        )
    }


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _load_global_event(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["global_source"]
    path = _root_path(source["path"])
    raw = path.read_bytes()
    if _sha256(raw) != source["sha256"]:
        raise RuntimeError("global source hash mismatch")
    payload = json.loads(raw)
    events = payload if isinstance(payload, list) else payload.get("events", [])
    matches = [row for row in events if str(row.get("id")) == source["event_id"]]
    if len(matches) != 1:
        raise RuntimeError("global source does not contain exactly one bound event")
    event = matches[0]
    if event.get("title") != contract["exact_title"]:
        raise RuntimeError("global event title differs from frozen exact title")
    return event


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    _frozen_instant(contract.get("frozen_at_utc"))
    implementation = _root_path(contract["implementation"]["path"])
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    expected_url = "https://gateway.polymarket.us/v1/search?" + urlencode(
        {"query": contract["exact_title"], "limit": contract["limit"]}
    )
    if contract["capture"] != {
        "method": "GET",
        "public_unauthenticated": True,
        "read_only": True,
        "retry_count": 0,
        "url": expected_url,
    }:
        raise RuntimeError("capture contract is not the exact public one-GET search")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract_path does not bind the loaded contract")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one frozen Polymarket US exact-title search."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()

    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)
    global_event = _load_global_event(contract)
    result_path = _root_path(contract["outputs"]["result_path"])
    raw_path = _root_path(contract["outputs"]["raw_path"])
    journal_path = _root_path(contract["outputs"]["journal_path"])
    if journal_path.parent.exists() or result_path.exists():
        raise RuntimeError("one-use output already exists")
    raw_path.parent.mkdir(parents=True)

    raw, receipt = _capture(
        url=contract["capture"]["url"],
        raw_path=raw_path,
        journal_path=journal_path,
        raw_relative_path=contract["outputs"]["raw_path"],
    )
    payload = json.loads(raw)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise RuntimeError("Polymarket US search response lacks an events list")
    exact = [row for row in events if row.get("title") == contract["exact_title"]]
    us_event = exact[0] if len(exact) == 1 else None
    us_markets = (
        [_market_metadata(row) for row in us_event.get("markets", [])]
        if isinstance(us_event, dict)
        else []
    )
    global_markets = [
        _market_metadata(row) for row in global_event.get("markets", [])
    ]
    us_market = us_markets[0] if len(us_markets) == 1 else None
    global_market = global_markets[0] if len(global_markets) == 1 else None

    comparisons = {
        "event_title_exact": bool(us_event),
        "single_market_each": us_market is not None and global_market is not None,
        "event_description_exact": bool(us_event)
        and _normalize_text(us_event.get("description"))
        == _normalize_text(global_event.get("description")),
        "event_resolution_source_exact": bool(us_event)
        and _normalize_text(us_event.get("resolutionSource"))
        == _normalize_text(global_event.get("resolutionSource")),
        "event_end_date_exact": bool(us_event)
        and us_event.get("endDate") == global_event.get("endDate"),
        "market_question_exact": bool(us_market and global_market)
        and _normalize_text(us_market.get("question"))
        == _normalize_text(global_market.get("question")),
        "market_description_exact": bool(us_market and global_market)
        and _normalize_text(us_market.get("description"))
        == _normalize_text(global_market.get("description")),
        "market_resolution_source_exact": bool(us_market and global_market)
        and _normalize_text(us_market.get("resolutionSource"))
        == _normalize_text(global_market.get("resolutionSource")),
        "market_end_date_exact": bool(us_market and global_market)
        and us_market.get("endDate") == global_market.get("endDate"),
        "us_has_no_additional_rules_disclaimer": bool(us_market)
        and not _normalize_text(us_market.get("rulesDisclaimer")),
    }
    exact_payoff_identity = all(comparisons.values())
    both_open = bool(
        us_event
        and us_event.get("active") is True
        and us_event.get("closed") is False
        and global_event.get("active") is True
        and global_event.get("closed") is False
    )
    result: dict[str, Any] = {
        "schema_version": "polymarket-us-global-exact-title-search-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "returned_event_count": len(events),
            "exact_title_match_count": len(exact),
            "economic_fields_printed": 0,
        },
        "global_event": _event_metadata(global_event),
        "global_markets": global_markets,
        "polymarket_us_event": _event_metadata(us_event) if us_event else None,
        "polymarket_us_markets": us_markets,
        "payoff_identity": {
            "comparisons": comparisons,
            "exact_text_and_time_identity": exact_payoff_identity,
            "both_events_open": both_open,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "candidate_for_separate_frozen_economics_gate": exact_payoff_identity
            and both_open,
            "next_action": (
                "freeze_separate_cross_venue_economics_contract_before_any_price_or_book_access"
                if exact_payoff_identity and both_open
                else "stop_without_any_price_book_account_or_order_request"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "candidate_for_separate_frozen_economics_gate": exact_payoff_identity
                and both_open,
                "exact_title_match_count": len(exact),
                "economic_fields_printed": 0,
                "returned_event_count": len(events),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
