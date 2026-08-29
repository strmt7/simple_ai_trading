"""Screen one bounded Gamma page for fixed NegRisk all-YES complete sets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_mlb_cross_period_catalog import _frozen_instant, _instant


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-negrisk-complete-set-catalog-result-v1"


def _json_array(value: object, name: str) -> list[Any]:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be encoded JSON")
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise RuntimeError(f"{name} must decode to a list")
    return decoded


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen = _frozen_instant(contract.get("frozen_at_utc"))
    minimum = _instant(contract["capture"]["end_date_min"], "end_date_min")
    maximum = _instant(contract["capture"]["end_date_max"], "end_date_max")
    if minimum <= frozen or maximum <= minimum:
        raise RuntimeError("catalog end-date window must be future and nonempty")
    if contract["capture"] != {
        "ascending": True,
        "closed": False,
        "end_date_max": contract["capture"]["end_date_max"],
        "end_date_min": contract["capture"]["end_date_min"],
        "limit": 500,
        "order": "endDate",
        "request_count": 1,
        "url": contract["capture"]["url"],
    }:
        raise RuntimeError("catalog capture boundary changed")
    if contract["authority"] != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "onchain_requests": 0,
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


def _eligible_event(event: dict[str, Any]) -> bool:
    return (
        event.get("active") is True
        and event.get("closed") is False
        and event.get("negRisk") is True
        and event.get("enableNegRisk") is True
        and event.get("negRiskAugmented") is False
    )


def _screen_event(event: dict[str, Any]) -> dict[str, Any]:
    event_id = str(event.get("id") or "")
    event_slug = str(event.get("slug") or "")
    if not event_id.isdigit() or not event_slug:
        raise RuntimeError("event identity is invalid")
    market_id = str(event.get("negRiskMarketID") or "").lower()
    if len(market_id) != 66 or not market_id.startswith("0x"):
        raise RuntimeError("event NegRisk market ID is invalid")
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) < 2:
        raise RuntimeError("event lacks a multi-outcome market set")

    legs: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    total = Decimal("0")
    for market in markets:
        if not isinstance(market, dict):
            raise RuntimeError("event market is not an object")
        if (
            market.get("active") is not True
            or market.get("closed") is not False
            or market.get("acceptingOrders") is not True
            or market.get("enableOrderBook") is not True
            or market.get("negRisk") is not True
            or str(market.get("negRiskMarketID") or "").lower() != market_id
        ):
            raise RuntimeError("event contains an unavailable or incompatible market")
        outcomes = _json_array(market.get("outcomes"), "outcomes")
        prices = _json_array(market.get("outcomePrices"), "outcomePrices")
        tokens = [str(value) for value in _json_array(market.get("clobTokenIds"), "clobTokenIds")]
        if outcomes != ["Yes", "No"] or len(prices) != 2 or len(tokens) != 2:
            raise RuntimeError("event market binary mapping differs")
        if any(not token.isdigit() or token in seen_tokens for token in tokens):
            raise RuntimeError("event token identity is invalid or duplicated")
        seen_tokens.update(tokens)
        yes_price = Decimal(str(prices[0]))
        no_price = Decimal(str(prices[1]))
        if any(not value.is_finite() or value < 0 or value > 1 for value in (yes_price, no_price)):
            raise RuntimeError("event Gamma price is invalid")
        total += yes_price
        legs.append(
            {
                "gamma_market_id": str(market.get("id") or ""),
                "label": str(market.get("groupItemTitle") or market.get("question") or ""),
                "yes_price_pUSD": _decimal_text(yes_price),
                "yes_token_id": tokens[0],
                "no_price_pUSD": _decimal_text(no_price),
                "no_token_id": tokens[1],
            }
        )
    if any(not leg["gamma_market_id"].isdigit() or not leg["label"] for leg in legs):
        raise RuntimeError("event market identity or label is invalid")

    floor = Decimal("1")
    return {
        "event_id": event_id,
        "event_slug": event_slug,
        "event_title": str(event.get("title") or ""),
        "end_date_utc": event.get("endDate"),
        "neg_risk_market_id": market_id,
        "market_count": len(legs),
        "legs": legs,
        "displayed_all_yes_sum_pUSD": _decimal_text(total),
        "guaranteed_payout_floor_pUSD_pending_onchain_proof": "1",
        "optimistic_profit_before_execution_costs_pUSD": _decimal_text(floor - total),
        "passes_strictly_below_payout_gate": total < floor,
        "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
    }


def _candidate_key(row: dict[str, Any]) -> tuple[Decimal, int]:
    return Decimal(row["displayed_all_yes_sum_pUSD"]), int(row["event_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    raw_path = _root_path(str(contract["outputs"]["raw_path"]))
    journal_path = _root_path(str(contract["outputs"]["journal_path"]))
    result_path = _root_path(str(contract["outputs"]["result_path"]))
    for path in (raw_path, journal_path, result_path):
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    raw, receipt = _request(
        method="GET",
        url=str(contract["capture"]["url"]),
        body=b"",
        name="near-expiry-negrisk-complete-set-catalog",
        raw_path=raw_path,
        raw_relative_path=str(contract["outputs"]["raw_path"]),
        journal_path=journal_path,
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise RuntimeError("keyset catalog response shape changed")
    events = payload["events"]
    if len(events) > 500:
        raise RuntimeError("catalog exceeded frozen page limit")
    population_complete = "next_cursor" not in payload
    if not population_complete and not isinstance(payload.get("next_cursor"), str):
        raise RuntimeError("catalog cursor shape changed")

    minimum = _instant(contract["capture"]["end_date_min"], "end_date_min")
    maximum = _instant(contract["capture"]["end_date_max"], "end_date_max")
    classifications: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id") or "")
        event_slug = str(event.get("slug") or "")
        try:
            end = _instant(event.get("endDate"), "endDate")
            if not (event.get("closed") is False and minimum <= end <= maximum):
                raise RuntimeError("event fails exact frozen population filter")
            if not _eligible_event(event):
                classifications.append(
                    {"event_id": event_id, "event_slug": event_slug, "classification": "not_fixed_negrisk"}
                )
                continue
            row = _screen_event(event)
        except (KeyError, RuntimeError, ValueError, ArithmeticError) as exc:
            classifications.append(
                {"event_id": event_id, "event_slug": event_slug, "classification": "excluded", "reason": str(exc)}
            )
            continue
        screened.append(row)
        classifications.append(
            {"event_id": event_id, "event_slug": event_slug, "classification": "screened_fixed_negrisk"}
        )

    candidates = sorted(
        (row for row in screened if row["passes_strictly_below_payout_gate"]),
        key=_candidate_key,
    )
    best = candidates[0] if candidates else None
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {"path": contract["contract_path"], "sha256": contract["contract_sha256"]},
        "capture": {
            "receipt": receipt,
            "returned_event_count": len(events),
            "next_cursor_present": "next_cursor" in payload,
            "population_complete_under_frozen_filter": population_complete,
            "frozen_end_date_window": {
                "end_date_min": contract["capture"]["end_date_min"],
                "end_date_max": contract["capture"]["end_date_max"],
            },
        },
        "screen": {
            "classifications": classifications,
            "fixed_negrisk_event_count": len(screened),
            "events": screened,
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "best_candidate": best,
            "proof_candidate": best if population_complete else None,
        },
        "adjudication": {
            "status": (
                "incomplete_catalog_no_escalation"
                if not population_complete
                else (
                    "candidate_requires_separately_frozen_onchain_and_exact_depth_proof"
                    if best is not None
                    else "complete_window_rejected_before_onchain_books_and_fees"
                )
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/screen_polymarket_negrisk_complete_set_catalog.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "returned_event_count": len(events),
                "fixed_negrisk_event_count": len(screened),
                "candidate_count": len(candidates),
                "population_complete": population_complete,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
