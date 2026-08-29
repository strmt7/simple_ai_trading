"""Select one price-blind BTC/ETH/SOL series, then screen its fixed NegRisk events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_mlb_cross_period_catalog import _frozen_instant, _instant
from tools.screen_polymarket_negrisk_complete_set_catalog import (
    _candidate_key,
    _eligible_event,
    _screen_event,
)


SCHEMA = "polymarket-source-selected-crypto-negrisk-series-result-v1"
CRYPTO_SCOPE = re.compile(r"\b(?:bitcoin|btc|ethereum|eth|solana|sol)\b", re.IGNORECASE)


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen = _frozen_instant(contract.get("frozen_at_utc"))
    minimum = _instant(contract["event_capture"]["end_date_min"], "end_date_min")
    maximum = _instant(contract["event_capture"]["end_date_max"], "end_date_max")
    if minimum <= frozen or maximum <= minimum:
        raise RuntimeError("event end-date window must be future and nonempty")
    if contract["series_discovery"] != {
        "ascending": False,
        "closed": False,
        "exclude_events": True,
        "limit": 100,
        "offset": 0,
        "order": "volume24hr",
        "request_count": 1,
        "url": "https://gamma-api.polymarket.com/series?limit=100&offset=0&order=volume24hr&ascending=false&closed=false&exclude_events=true",
    }:
        raise RuntimeError("series discovery boundary changed")
    expected_event_capture = {
        "ascending": True,
        "closed": False,
        "end_date_max": contract["event_capture"]["end_date_max"],
        "end_date_min": contract["event_capture"]["end_date_min"],
        "limit": 500,
        "order": "endDate",
        "request_count_if_series_selected": 1,
        "url_template": contract["event_capture"]["url_template"],
    }
    if contract["event_capture"] != expected_event_capture:
        raise RuntimeError("event capture boundary changed")
    if "{series_id}" not in contract["event_capture"]["url_template"]:
        raise RuntimeError("event URL template lacks series placeholder")
    if contract["authority"] != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "maximum_public_unauthenticated_read_only_requests": 2,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _series_is_eligible(series: dict[str, Any]) -> bool:
    identity_text = " ".join(
        str(series.get(field) or "")
        for field in ("slug", "title", "subtitle", "ticker")
    )
    return (
        series.get("active") is True
        and series.get("closed") is False
        and series.get("archived") is not True
        and isinstance(series.get("recurrence"), str)
        and bool(series["recurrence"].strip())
        and CRYPTO_SCOPE.search(identity_text) is not None
        and str(series.get("id") or "").isdigit()
    )


def _select_series(
    payload: object,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(payload, list) or len(payload) > 100:
        raise RuntimeError("series response shape or limit changed")
    classifications: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for ordinal, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError("series row is not an object")
        if "events" in item:
            raise RuntimeError("exclude_events=true returned outcome-sensitive events")
        eligible = _series_is_eligible(item)
        classifications.append(
            {
                "ordinal": ordinal,
                "series_id": str(item.get("id") or ""),
                "slug": str(item.get("slug") or ""),
                "title": str(item.get("title") or ""),
                "recurrence": item.get("recurrence"),
                "eligible": eligible,
            }
        )
        if selected is None and eligible:
            selected = item
    return selected, classifications


def _event_series_matches(event: dict[str, Any], series_id: str) -> bool:
    series = event.get("series")
    return isinstance(series, list) and any(
        isinstance(row, dict) and str(row.get("id") or "") == series_id
        for row in series
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    paths = {
        name: _root_path(str(value)) for name, value in contract["outputs"].items()
    }
    for path in paths.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    series_raw, series_receipt = _request(
        method="GET",
        url=contract["series_discovery"]["url"],
        body=b"",
        name="source-selected-crypto-series-discovery",
        raw_path=paths["series_raw_path"],
        raw_relative_path=contract["outputs"]["series_raw_path"],
        journal_path=paths["journal_path"],
    )
    selected, series_classifications = _select_series(json.loads(series_raw))

    event_receipt: dict[str, Any] | None = None
    event_classifications: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    next_cursor_present: bool | None = None
    population_complete: bool | None = None
    if selected is not None:
        series_id = str(selected["id"])
        event_url = contract["event_capture"]["url_template"].replace(
            "{series_id}", series_id
        )
        event_raw, event_receipt = _request(
            method="GET",
            url=event_url,
            body=b"",
            name="source-selected-crypto-negrisk-events",
            raw_path=paths["events_raw_path"],
            raw_relative_path=contract["outputs"]["events_raw_path"],
            journal_path=paths["journal_path"],
        )
        payload = json.loads(event_raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            raise RuntimeError("event keyset response shape changed")
        events = payload["events"]
        if len(events) > 500:
            raise RuntimeError("event catalog exceeded frozen page limit")
        next_cursor_present = "next_cursor" in payload
        population_complete = not next_cursor_present
        if next_cursor_present and not isinstance(payload.get("next_cursor"), str):
            raise RuntimeError("event cursor shape changed")
        minimum = _instant(contract["event_capture"]["end_date_min"], "end_date_min")
        maximum = _instant(contract["event_capture"]["end_date_max"], "end_date_max")
        for event in events:
            event_id = str(event.get("id") or "")
            event_slug = str(event.get("slug") or "")
            try:
                end = _instant(event.get("endDate"), "endDate")
                if not (minimum <= end <= maximum):
                    raise RuntimeError("event fails exact frozen end-date filter")
                if not _event_series_matches(event, series_id):
                    raise RuntimeError("event does not bind selected series")
                if not _eligible_event(event):
                    event_classifications.append(
                        {
                            "event_id": event_id,
                            "event_slug": event_slug,
                            "classification": "not_fixed_negrisk",
                        }
                    )
                    continue
                row = _screen_event(event)
            except (KeyError, RuntimeError, ValueError, ArithmeticError) as exc:
                event_classifications.append(
                    {
                        "event_id": event_id,
                        "event_slug": event_slug,
                        "classification": "excluded",
                        "reason": str(exc),
                    }
                )
                continue
            screened.append(row)
            event_classifications.append(
                {
                    "event_id": event_id,
                    "event_slug": event_slug,
                    "classification": "screened_fixed_negrisk",
                }
            )

    candidates = sorted(
        (row for row in screened if row["passes_strictly_below_payout_gate"]),
        key=_candidate_key,
    )
    best = candidates[0] if candidates else None
    if selected is None:
        status = "no_eligible_price_blind_crypto_series_selected"
    elif population_complete is False:
        status = "selected_series_event_catalog_incomplete_no_escalation"
    elif best is None:
        status = "selected_series_rejected_before_onchain_books_and_fees"
    else:
        status = "candidate_requires_separately_frozen_onchain_and_exact_depth_proof"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "series_receipt": series_receipt,
            "event_receipt": event_receipt,
            "event_next_cursor_present": next_cursor_present,
            "event_population_complete_under_frozen_filter": population_complete,
        },
        "selection": {
            "rule": "first_server_ordered_eligible_recurring_BTC_ETH_SOL_series_without_events_or_prices",
            "series_classifications": series_classifications,
            "selected": None
            if selected is None
            else {
                "series_id": str(selected["id"]),
                "slug": str(selected.get("slug") or ""),
                "title": str(selected.get("title") or ""),
                "recurrence": selected.get("recurrence"),
            },
        },
        "screen": {
            "event_classifications": event_classifications,
            "fixed_negrisk_event_count": len(screened),
            "events": screened,
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "best_candidate": best,
            "proof_candidate": best if population_complete is True else None,
        },
        "adjudication": {
            "status": status,
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_separate_onchain_question_count_conversion_fee_and_exact_all_token_depth_proof"
                if best is not None and population_complete is True
                else "stop_without_onchain_book_fee_account_order_or_fund_requests"
            ),
        },
        "authority": {
            "public_unauthenticated_read_only_requests": 1
            + int(event_receipt is not None),
            "credentials_used": False,
            "signed_requests": 0,
            "account_requests": 0,
            "onchain_requests": 0,
            "book_requests": 0,
            "fee_requests": 0,
            "orders_or_transactions": 0,
            "funds_used": False,
            "trading_authority": False,
            "protected_capture_touched": False,
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    paths["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": status,
                "selected_series_id": None if selected is None else str(selected["id"]),
                "fixed_negrisk_event_count": len(screened),
                "candidate_count": len(candidates),
                "best_sum": None
                if best is None
                else best["displayed_all_yes_sum_pUSD"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
