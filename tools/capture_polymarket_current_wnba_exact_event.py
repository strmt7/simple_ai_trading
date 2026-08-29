from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-current-wnba-exact-event-contract-v1-2026-08-29.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-current-wnba-exact-event-result-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/polymarket-current-wnba-exact-event-v1"
RAW_PATH = DATA_ROOT / "raw/event.json"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"


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


def _journal(payload: dict[str, Any]) -> None:
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(url: str) -> tuple[bytes, dict[str, Any]]:
    intent = {
        "method": "GET",
        "name": "exact-current-wnba-event",
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": time.time_ns() // 1_000_000,
        "url": url,
    }
    _journal(intent)
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
        RAW_PATH.write_bytes(raw)
        _journal(
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
                "response_bytes": len(raw),
                "response_sha256": _sha256(raw),
                "status_code": status_code,
            }
        )
        raise
    RAW_PATH.write_bytes(raw)
    receipt = {
        **intent,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "phase": "completed",
        "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "status_code": status_code,
    }
    _journal(receipt)
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


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if DATA_ROOT.exists() or RESULT_PATH.exists():
        raise RuntimeError("one-use output already exists")
    RAW_PATH.parent.mkdir(parents=True)

    raw, receipt = _capture(contract["capture"]["url"])
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
        "schema_version": "polymarket-current-wnba-exact-event-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
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
                "prove_exact_joint_payoffs_offline_then_freeze_one_book_batch"
                if candidate
                else "stop_without_any_book_request"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
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
