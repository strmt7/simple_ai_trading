"""Screen creation-safe AI Arena score threshold covers from retained Gamma."""

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


CONTRACT_SCHEMA = "polymarket-retained-ai-arena-threshold-ladder-contract-v1"
RESULT_SCHEMA = "polymarket-retained-ai-arena-threshold-ladder-result-v1"


def _utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{name} must be an exact UTC instant")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _load_population(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name="population")


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


def _preflight_market(
    event: Mapping[str, Any], spec: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    market = _market_by_id(event, str(spec["market_id"]))
    if str(market.get("groupItemTitle")) != str(spec["threshold"]):
        raise RuntimeError("threshold label changed")
    start = _utc(market.get("startDate"), name="market start")
    expected_start = _utc(spec.get("start_utc"), name="expected market start")
    if start != expected_start:
        raise RuntimeError("market start changed")
    description = str(market.get("description") or "")
    if not all(
        fragment in description for fragment in contract["required_rule_fragments"]
    ):
        raise RuntimeError("AI Arena threshold rules changed")
    if _list(market.get("outcomes"), name="outcomes") != ["Yes", "No"]:
        raise RuntimeError("binary outcome representation changed")
    if len(_list(market.get("clobTokenIds"), name="CLOB token IDs")) != 2:
        raise RuntimeError("binary token representation changed")
    for field in ("active", "closed", "acceptingOrders", "enableOrderBook"):
        if market.get(field) is not spec[field]:
            raise RuntimeError(f"market {field} state changed")
    for field in ("bestAsk", "bestBid"):
        raw = market.get(field)
        if raw is not None and not isinstance(raw, (str, int, float)):
            raise RuntimeError(f"{field} representation changed")
    return market


def _preflight_pair(
    event: Mapping[str, Any], pair: Mapping[str, Any], by_id: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    lower = _mapping(by_id[str(pair["lower_market_id"])], name="lower market")
    higher = _mapping(by_id[str(pair["higher_market_id"])], name="higher market")
    if not (
        Decimal(str(pair["lower_threshold"])) < Decimal(str(pair["higher_threshold"]))
    ):
        raise RuntimeError("threshold order changed")
    if str(lower.get("groupItemTitle")) != str(pair["lower_threshold"]):
        raise RuntimeError("lower threshold mapping changed")
    if str(higher.get("groupItemTitle")) != str(pair["higher_threshold"]):
        raise RuntimeError("higher threshold mapping changed")
    lower_start = _utc(lower.get("startDate"), name="lower market start")
    higher_start = _utc(higher.get("startDate"), name="higher market start")
    if lower_start > higher_start:
        raise RuntimeError("lower threshold does not cover the higher creation window")
    for market in (lower, higher):
        if not (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("acceptingOrders") is True
            and market.get("enableOrderBook") is True
        ):
            raise RuntimeError("selected threshold market is not executable")
    return lower, higher


def _preflight_event(
    population: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    event = _select_event(population, str(contract["event_id"]))
    if event.get("slug") != contract["event_slug"]:
        raise RuntimeError("exact event slug changed")
    if event.get("title") != contract["event_title"]:
        raise RuntimeError("exact event title changed")
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) != len(contract["markets"]):
        raise RuntimeError("exact event market population changed")
    by_id: dict[str, dict[str, Any]] = {}
    for value in contract["markets"]:
        spec = _mapping(value, name="frozen market spec")
        market = _preflight_market(event, spec, contract)
        by_id[str(spec["market_id"])] = market
    if set(by_id) != {str(value["id"]) for value in markets}:
        raise RuntimeError("exact event market IDs changed")
    for value in contract["pairs"]:
        _preflight_pair(event, _mapping(value, name="frozen pair"), by_id)
    return event, by_id


def _screen(
    population: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    event, by_id = _preflight_event(population, contract)
    rows: list[dict[str, Any]] = []
    for value in contract["pairs"]:
        pair = _mapping(value, name="frozen pair")
        lower, higher = _preflight_pair(event, pair, by_id)
        row = _package_row(
            family=str(pair["pair"]),
            relation="higher_YES_implies_lower_YES_so_lower_YES_plus_higher_NO_has_one_pUSD_floor",
            exact_date=None,
            deadline_date=date.fromisoformat(str(contract["cutoff_date"])),
            first_market=lower,
            first_outcome="Yes",
            second_market=higher,
            second_outcome="No",
        )
        row.update(
            {
                "lower_threshold": str(pair["lower_threshold"]),
                "higher_threshold": str(pair["higher_threshold"]),
                "lower_start_utc": lower["startDate"],
                "higher_start_utc": higher["startDate"],
                "creation_window_proof": "lower_start_at_or_before_higher_start",
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["status"] != "priced",
            Decimal(row["metadata_cost_pUSD_per_share"])
            if row["status"] == "priced"
            else Decimal("Infinity"),
            row["lower_threshold"],
            row["higher_threshold"],
        )
    )
    return rows


def _validate_contract(
    contract: Mapping[str, Any], path: Path
) -> tuple[Path, dict[str, Any]]:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("unexpected contract schema")
    if contract.get("status") != "frozen_before_one_zero_network_retained_adjudication":
        raise RuntimeError("unexpected contract status")
    if _canonical_hash(dict(contract), "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    if _utc(contract.get("frozen_at_utc"), name="frozen_at_utc") > datetime.now(
        timezone.utc
    ):
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
    if len(contract.get("markets", [])) != 5 or len(contract.get("pairs", [])) != 4:
        raise RuntimeError("frozen population size changed")
    population = _load_population(retained)
    _preflight_event(population, contract)
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
    retained, population = _validate_contract(contract, contract_path)
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
            "market_count": len(contract["markets"]),
            "active_market_count": sum(
                value["closed"] is False for value in contract["markets"]
            ),
            "excluded_creation_gap_pair_count": 2,
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
                "candidate_requires_separately_frozen_fresh_exact_depth_batch"
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
