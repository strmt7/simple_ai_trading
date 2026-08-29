from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
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


def _instant(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is not a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} is unparsable") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} is offset-free")
    return parsed.astimezone(timezone.utc)


def _journal(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(
    *, url: str, raw_path: Path, raw_relative_path: str, journal_path: Path
) -> tuple[bytes, dict[str, Any]]:
    intent = {
        "method": "GET",
        "name": "future-mlb-cross-period-catalog",
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


def _pair(value: object, label: str) -> list[str]:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a JSON string")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise RuntimeError(f"{label} must contain exactly two values")
    return [str(item) for item in parsed]


def _market_map(event: dict[str, Any], market_type: str) -> dict[Decimal, dict[str, Any]]:
    rows: dict[Decimal, dict[str, Any]] = {}
    for market in event.get("markets", []):
        if market.get("sportsMarketType") != market_type:
            continue
        if not (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("acceptingOrders") is True
            and market.get("enableOrderBook") is True
        ):
            continue
        outcomes = _pair(market.get("outcomes"), "outcomes")
        prices = _pair(market.get("outcomePrices"), "outcomePrices")
        tokens = _pair(market.get("clobTokenIds"), "clobTokenIds")
        if outcomes != ["Over", "Under"]:
            raise RuntimeError(f"unexpected outcomes for market {market.get('id')}")
        line = Decimal(str(market.get("line")))
        if line < 0 or line % 1 != Decimal("0.5"):
            raise RuntimeError(f"unexpected total line for market {market.get('id')}")
        threshold = int(line + Decimal("0.5"))
        description = str(market.get("description") or "")
        scope_fragment = (
            f"score {threshold} or more runs in this game"
            if market_type == "totals"
            else f"score {threshold} or more runs by the conclusion of the 5th inning"
        )
        required = [
            scope_fragment,
            f"combined total is less than {threshold}",
            "canceled entirely, with no make-up game, this market will resolve 50-50",
        ]
        if not all(fragment in description for fragment in required):
            raise RuntimeError(f"incomplete rules for market {market.get('id')}")
        if line in rows:
            raise RuntimeError(f"duplicate {market_type} line {line}")
        rows[line] = {
            "market_id": str(market["id"]),
            "market_slug": str(market["slug"]),
            "condition_id": str(market["conditionId"]),
            "line": str(line),
            "over_price_pUSD": prices[0],
            "under_price_pUSD": prices[1],
            "over_token_id": tokens[0],
            "under_token_id": tokens[1],
            "fee_schedule": market.get("feeSchedule"),
            "taker_base_fee": market.get("takerBaseFee"),
            "tick_size": str(market.get("orderPriceMinTickSize")),
            "minimum_order_size": str(market.get("orderMinSize")),
        }
    return rows


def _event_has_mlb_tag(event: dict[str, Any], tag_id: str) -> bool:
    return any(str(tag.get("id")) == tag_id for tag in event.get("tags", []))


def _relations(
    events: list[dict[str, Any]], *, completed_at: datetime, tag_id: str
) -> tuple[list[dict[str, Any]], int]:
    relations: list[dict[str, Any]] = []
    future_event_count = 0
    for event in events:
        if not (
            event.get("active") is True
            and event.get("closed") is False
            and _event_has_mlb_tag(event, tag_id)
        ):
            continue
        start = _instant(event.get("startTime"), "event startTime")
        if start <= completed_at:
            continue
        future_event_count += 1
        full_game = _market_map(event, "totals")
        first_five = _market_map(event, "baseball_team_first_five_total")
        for line in sorted(set(full_game) & set(first_five)):
            full = full_game[line]
            first = first_five[line]
            price_sum = Decimal(full["over_price_pUSD"]) + Decimal(
                first["under_price_pUSD"]
            )
            relations.append(
                {
                    "event_id": str(event["id"]),
                    "event_slug": str(event["slug"]),
                    "event_title": str(event["title"]),
                    "start_time_utc": start.isoformat().replace("+00:00", "Z"),
                    "line": str(line),
                    "full_game_over": full,
                    "first_five_under": first,
                    "guaranteed_floor_per_share_pUSD": "1",
                    "gamma_displayed_price_sum_pUSD": str(price_sum),
                    "optimistic_profit_floor_per_share_before_execution_costs_pUSD": str(
                        Decimal("1") - price_sum
                    ),
                    "passes_strictly_below_payout_gate": price_sum < 1,
                }
            )
    return relations, future_event_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen one bounded future MLB Gamma catalog for cross-period totals."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    _frozen_instant(contract.get("frozen_at_utc"))
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    implementation = _root_path(contract["implementation"]["path"])
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")

    result_path = _root_path(contract["outputs"]["result_path"])
    raw_path = _root_path(contract["outputs"]["raw_path"])
    journal_path = _root_path(contract["outputs"]["journal_path"])
    if journal_path.parent.exists() or result_path.exists():
        raise RuntimeError("one-use output already exists")
    raw_path.parent.mkdir(parents=True)
    raw, receipt = _capture(
        url=contract["capture"]["url"],
        raw_path=raw_path,
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=journal_path,
    )
    events = json.loads(raw)
    if not isinstance(events, list):
        raise RuntimeError("catalog response is not a list")
    limit = int(contract["capture"]["limit"])
    if len(events) > limit:
        raise RuntimeError("catalog exceeded its frozen limit")
    completed_at = datetime.fromtimestamp(
        receipt["completed_at_ms"] / 1000, tz=timezone.utc
    )
    relations, future_event_count = _relations(
        events,
        completed_at=completed_at,
        tag_id=str(contract["capture"]["tag_id"]),
    )
    candidates = [row for row in relations if row["passes_strictly_below_payout_gate"]]
    candidates.sort(
        key=lambda row: (
            -Decimal(
                row[
                    "optimistic_profit_floor_per_share_before_execution_costs_pUSD"
                ]
            ),
            row["start_time_utc"],
            row["event_slug"],
            Decimal(row["line"]),
        )
    )
    result: dict[str, Any] = {
        "schema_version": "polymarket-mlb-cross-period-catalog-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "returned_event_count": len(events),
            "limit": limit,
            "offset": contract["capture"]["offset"],
            "population_complete_under_frozen_filter": len(events) < limit,
        },
        "screen": {
            "future_event_count_at_completed_request": future_event_count,
            "exact_cross_period_relation_count": len(relations),
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "candidates": candidates,
            "best_candidate": candidates[0] if candidates else None,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_exact_depth_screen_for_only_the_deterministic_best_candidate"
                if candidates
                else "stop_without_any_book_or_fee_request"
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
                "returned_event_count": len(events),
                "future_event_count": future_event_count,
                "relation_count": len(relations),
                "candidate_count": len(candidates),
                "population_complete": len(events) < limit,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
