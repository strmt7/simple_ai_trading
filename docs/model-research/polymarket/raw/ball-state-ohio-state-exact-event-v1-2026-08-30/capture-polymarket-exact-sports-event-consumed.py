"""Capture one exact public Polymarket sports event before any book access."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-ball-state-ohio-state-exact-event-contract-v1-2026-08-30.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-ball-state-ohio-state-exact-event-result-v1-2026-08-30.json"
)
RAW_DIR = ROOT / (
    "docs/model-research/polymarket/raw/"
    "ball-state-ohio-state-exact-event-v1-2026-08-30"
)
RAW_PATH = RAW_DIR / "event.raw.json"
JOURNAL_PATH = RAW_DIR / "request-journal.jsonl"


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


def _canonical_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def _append_journal(payload: Mapping[str, object]) -> None:
    with JOURNAL_PATH.open("ab") as stream:
        stream.write((_canonical_json(payload) + "\n").encode("ascii"))
        stream.flush()


def _market_summary(market: Mapping[str, object]) -> dict[str, object]:
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


def _load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    if contract.get("schema_version") != (
        "polymarket-exact-sports-event-source-contract-v1"
    ):
        raise ValueError("contract schema differs")
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    implementation = contract["implementation"]
    implementation_path = ROOT / str(implementation["path"])
    if _sha256(implementation_path.read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation hash mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("frozen timestamp invalid or future")
    return contract


def _preflight() -> dict[str, Any]:
    contract = _load_contract()
    if RAW_DIR.exists() or RESULT_PATH.exists():
        raise FileExistsError("one-use raw directory or result already exists")
    if not RAW_DIR.parent.is_dir():
        raise FileNotFoundError("raw parent is missing")
    return contract


def run() -> dict[str, object]:
    contract = _preflight()
    RAW_DIR.mkdir(parents=False, exist_ok=False)
    probe = RAW_DIR / ".write-probe"
    write_bytes_atomic(probe, b"ready\n")
    probe.unlink()
    url = str(contract["capture"]["url"])
    request_id = "exact-sports-event"
    intent = {
        "phase": "intent",
        "request_id": request_id,
        "method": "GET",
        "url": url,
        "request_body_sha256": _sha256(b""),
        "planned_at_ms": time.time_ns() // 1_000_000,
    }
    _append_journal(intent)
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        },
    )
    started_ms = time.time_ns() // 1_000_000
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(int(contract["capture"]["maximum_response_bytes"]) + 1)
            status = response.status
    except HTTPError as exc:
        raw = exc.read()
        status = exc.code
        write_bytes_atomic(RAW_PATH, raw)
        _append_journal(
            {
                **intent,
                "phase": "completed",
                "started_at_ms": started_ms,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "status_code": status,
                "response_bytes": len(raw),
                "response_sha256": _sha256(raw),
                "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
            }
        )
        raise
    write_bytes_atomic(RAW_PATH, raw)
    receipt = {
        **intent,
        "phase": "completed",
        "started_at_ms": started_ms,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "status_code": status,
        "response_bytes": len(raw),
        "response_sha256": _sha256(raw),
        "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
    }
    _append_journal(receipt)
    if status != 200:
        raise RuntimeError(f"unexpected HTTP status {status}")
    if len(raw) > int(contract["capture"]["maximum_response_bytes"]):
        raise ValueError("response exceeds frozen byte ceiling")
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise ValueError("event response must be an object")
    exact_slug = event.get("slug") == contract["event_slug"]
    event_open = (
        exact_slug
        and event.get("active") is True
        and event.get("closed") is False
    )
    markets = [
        _market_summary(value)
        for value in event.get("markets", [])
        if isinstance(value, Mapping)
    ]
    active = [
        value
        for value in markets
        if value["active"] is True
        and value["closed"] is False
        and value["acceptingOrders"] is True
    ]
    types = sorted(
        {
            str(value["sportsMarketType"])
            for value in active
            if value["sportsMarketType"] is not None
        }
    )
    source_candidate = event_open and "moneyline" in types and "spreads" in types
    result: dict[str, object] = {
        "schema_version": "polymarket-exact-sports-event-source-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "exact_slug_match": exact_slug,
            "event_active_and_open": event_open,
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
            "exact_moneyline_spread_source_candidate": source_candidate,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "adjudicate_exact_joint_payoffs_and_displayed_prices_offline"
                if source_candidate
                else "stop_without_any_book_request"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    _write_json(RESULT_PATH, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        contract = _preflight()
        print(
            _canonical_json(
                {
                    "status": "preflight_passed",
                    "contract_sha256": contract["contract_sha256"],
                }
            )
        )
        return
    result = run()
    print(
        _canonical_json(
            {
                "result_sha256": result["result_sha256"],
                "event": result["event"],
                "active_market_types": result["discovery"][
                    "active_sports_market_types"
                ],
                "active_market_count": result["capture"][
                    "active_accepting_market_count"
                ],
                "source_candidate": result["discovery"][
                    "exact_moneyline_spread_source_candidate"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
