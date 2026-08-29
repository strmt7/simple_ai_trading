from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
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
        "name": "exact-sports-event-metadata",
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


def _market_summary(market: dict[str, Any]) -> dict[str, Any]:
    return {
        key: market.get(key)
        for key in (
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
    }


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    _frozen_instant(contract.get("frozen_at_utc"))
    implementation = _root_path(contract["implementation"]["path"])
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    capture = contract["capture"]
    expected_url = (
        "https://gamma-api.polymarket.com/events/slug/" + contract["event_slug"]
    )
    if capture != {
        "method": "GET",
        "public_unauthenticated": True,
        "read_only": True,
        "retry_count": 0,
        "url": expected_url,
    }:
        raise RuntimeError("capture contract is not the exact public one-GET endpoint")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract_path does not bind the loaded contract")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one frozen public Polymarket sports event by exact slug."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()

    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)
    result_path = _root_path(contract["outputs"]["result_path"])
    raw_path = _root_path(contract["outputs"]["raw_path"])
    journal_path = _root_path(contract["outputs"]["journal_path"])
    data_root = journal_path.parent
    if data_root.exists() or result_path.exists():
        raise RuntimeError("one-use output already exists")
    raw_path.parent.mkdir(parents=True)

    raw, receipt = _capture(
        url=contract["capture"]["url"],
        raw_path=raw_path,
        journal_path=journal_path,
        raw_relative_path=contract["outputs"]["raw_path"],
    )
    event = json.loads(raw)
    exact_slug = event.get("slug") == contract["event_slug"]
    active_event = exact_slug and event.get("active") is True and event.get("closed") is False
    markets = [_market_summary(row) for row in event.get("markets", [])]
    active = [
        row
        for row in markets
        if row["active"] is True
        and row["closed"] is False
        and row["acceptingOrders"] is True
    ]
    types = sorted({str(row["sportsMarketType"]) for row in active})
    candidate = active_event and "moneyline" in types and "spreads" in types
    result: dict[str, Any] = {
        "schema_version": "polymarket-exact-sports-event-metadata-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "exact_slug_match": exact_slug,
            "event_active_and_open": active_event,
            "embedded_market_count": len(markets),
            "active_accepting_market_count": len(active),
        },
        "event": {
            key: event.get(key)
            for key in (
                "id",
                "slug",
                "title",
                "description",
                "resolutionSource",
                "active",
                "closed",
                "endDate",
                "eventDate",
                "startTime",
                "gameId",
                "gameStatus",
            )
        },
        "discovery": {
            "active_sports_market_types": types,
            "active_accepting_markets": active,
            "exact_moneyline_spread_candidate": candidate,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "offline_rejection_only_payoff_and_Gamma_price_sum_prefilter"
                if candidate
                else "stop_without_any_book_request"
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
                "active_accepting_market_count": len(active),
                "active_sports_market_types": types,
                "candidate": candidate,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
