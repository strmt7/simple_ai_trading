"""Screen creation-window-safe deadline implications from retained Gamma events."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_release_date_deadline_graph import _package_row
from tools.screen_polymarket_exact_crypto_threshold_ladder_v2 import _list, _mapping


CONTRACT_SCHEMA = "polymarket-retained-deadline-implications-contract-v1"
RESULT_SCHEMA = "polymarket-retained-deadline-implications-result-v1"


def _load_population(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name="population")


def _utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{name} must be an exact UTC instant")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _date(value: object, *, name: str) -> date:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an ISO date") from exc


def _select_event(population: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    events = population.get("events")
    if not isinstance(events, list):
        raise RuntimeError("retained population events must be a list")
    matches = [
        _mapping(value, name="retained event")
        for value in events
        if isinstance(value, Mapping) and str(value.get("id")) == event_id
    ]
    if len(matches) != 1:
        raise RuntimeError("exact retained event must occur once")
    return matches[0]


def _market_by_id(event: Mapping[str, Any], market_id: str) -> dict[str, Any]:
    markets = event.get("markets")
    if not isinstance(markets, list):
        raise RuntimeError("exact event markets must be a list")
    matches = [
        _mapping(value, name="exact event market")
        for value in markets
        if isinstance(value, Mapping) and str(value.get("id")) == market_id
    ]
    if len(matches) != 1:
        raise RuntimeError("exact retained market must occur once")
    return matches[0]


def _preflight_pair(
    population: Mapping[str, Any], pair: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    event = _select_event(population, str(pair["event_id"]))
    if event.get("slug") != pair["event_slug"]:
        raise RuntimeError("exact event slug changed")
    if event.get("title") != pair["event_title"]:
        raise RuntimeError("exact event title changed")
    earlier = _market_by_id(event, str(pair["earlier_market_id"]))
    later = _market_by_id(event, str(pair["later_market_id"]))
    if earlier.get("groupItemTitle") != pair["earlier_label"]:
        raise RuntimeError("earlier deadline label changed")
    if later.get("groupItemTitle") != pair["later_label"]:
        raise RuntimeError("later deadline label changed")
    if earlier.get("description") != later.get("description"):
        raise RuntimeError("deadline rules differ")
    description = str(earlier.get("description") or "")
    if not all(fragment in description for fragment in pair["required_rule_fragments"]):
        raise RuntimeError("required deadline rule changed")
    earlier_start = _utc(earlier.get("startDate"), name="earlier market start")
    later_start = _utc(later.get("startDate"), name="later market start")
    if earlier_start.isoformat().replace("+00:00", "Z") != pair["earlier_start_utc"]:
        raise RuntimeError("earlier market start changed")
    if later_start.isoformat().replace("+00:00", "Z") != pair["later_start_utc"]:
        raise RuntimeError("later market start changed")
    earlier_deadline = _date(pair["earlier_deadline"], name="earlier deadline")
    later_deadline = _date(pair["later_deadline"], name="later deadline")
    if not (earlier_deadline < later_deadline and later_start <= earlier_start):
        raise RuntimeError("later deadline does not cover the earlier creation window")
    for market in (earlier, later):
        if _list(market.get("outcomes"), name="outcomes") != ["Yes", "No"]:
            raise RuntimeError("binary outcome representation changed")
        if len(_list(market.get("clobTokenIds"), name="CLOB token IDs")) != 2:
            raise RuntimeError("binary token representation changed")
        if not (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("acceptingOrders") is True
            and market.get("enableOrderBook") is True
        ):
            raise RuntimeError("selected deadline market is not executable")
        for field in ("bestAsk", "bestBid"):
            raw = market.get(field)
            if raw is not None and not isinstance(raw, (str, int, float)):
                raise RuntimeError(f"{field} representation changed")
    return earlier, later


def _screen(
    population: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in contract["pairs"]:
        pair = _mapping(value, name="frozen deadline pair")
        earlier, later = _preflight_pair(population, pair)
        row = _package_row(
            family=str(pair["pair"]),
            relation="earlier_YES_implies_later_YES_so_earlier_NO_plus_later_YES_has_one_pUSD_floor",
            exact_date=None,
            deadline_date=_date(pair["later_deadline"], name="later deadline"),
            first_market=earlier,
            first_outcome="No",
            second_market=later,
            second_outcome="Yes",
        )
        row.update(
            {
                "earlier_label": pair["earlier_label"],
                "later_label": pair["later_label"],
                "earlier_start_utc": pair["earlier_start_utc"],
                "later_start_utc": pair["later_start_utc"],
                "creation_window_proof": "later_start_at_or_before_earlier_start",
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["status"] != "priced",
            Decimal(row["metadata_cost_pUSD_per_share"])
            if row["status"] == "priced"
            else Decimal("Infinity"),
            row["family"],
        )
    )
    return rows


def _validate_contract(
    contract: Mapping[str, Any], path: Path, *, preflight_only: bool
) -> tuple[Path, dict[str, Any]]:
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_one_zero_network_retained_adjudication"
    )
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("unexpected contract schema")
    if contract.get("status") != expected_status:
        raise RuntimeError("contract status does not match invocation mode")
    if _canonical_hash(dict(contract), "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen = _utc(contract.get("frozen_at_utc"), name="frozen_at_utc")
    if frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is in the future")
    retained = _root_path(str(contract["retained_input"]["path"]))
    if _sha256(retained.read_bytes()) != contract["retained_input"]["sha256"]:
        raise RuntimeError("retained input hash mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "network_requests": 0,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    pairs = contract.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        raise RuntimeError("exact four-pair population changed")
    population = _load_population(retained)
    for pair in pairs:
        _preflight_pair(population, _mapping(pair, name="frozen deadline pair"))
    return retained, population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    retained, population = _validate_contract(
        contract, contract_path, preflight_only=args.preflight_only
    )
    if args.preflight_only:
        print("preflight_passed=true")
        return

    result_path = _root_path(str(contract["result_path"]))
    if result_path.exists():
        raise RuntimeError("one-use result already exists")
    rows = _screen(population, contract)
    priced = [row for row in rows if row["status"] == "priced"]
    metadata_candidates = [row for row in rows if row["passes_strict_metadata_gate"]]
    stressed_candidates = [row for row in rows if row["passes_fee_and_one_tick_gate"]]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "retained_input": {
            "path": contract["retained_input"]["path"],
            "sha256": _sha256(retained.read_bytes()),
        },
        "payoff_identity": contract["payoff_identity"],
        "screen": {
            "pair_count": len(rows),
            "side_specific_price_available_count": len(priced),
            "strict_metadata_candidate_count": len(metadata_candidates),
            "fee_and_one_tick_candidate_count": len(stressed_candidates),
            "pairs_ranked_by_side_specific_sum": rows,
            "best_pair": priced[0] if priced else None,
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "candidate_requires_separately_frozen_exact_depth_batch"
                if stressed_candidates
                else "rejected_before_books_and_onchain_requests"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "pair_count": len(rows),
                "side_specific_price_available_count": len(priced),
                "best_displayed_sum_pUSD": (
                    priced[0]["metadata_cost_pUSD_per_share"] if priced else None
                ),
                "strict_metadata_candidate_count": len(metadata_candidates),
                "fee_and_one_tick_candidate_count": len(stressed_candidates),
                "network_requests": 0,
                "payloads_printed": 0,
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
