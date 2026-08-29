from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    HALF,
    ONE,
    _canonical_hash,
    _common_market_gate,
    _json_pair,
    _json_ready,
    _market_ref,
    _relations,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_exact_nfl_monotone_prefilter import _total_markets


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-nfl-catalog-tie-collision-correction-v1"


def _collision_aware_margin_markets(
    markets: list[dict[str, Any]], moneyline: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    _common_market_gate(moneyline)
    team_a, team_b = _json_pair(moneyline.get("outcomes"), "moneyline outcomes")
    if "ends in a tie" not in str(moneyline["description"]):
        raise RuntimeError("moneyline lacks exact half-half tie semantics")
    rows = [
        _market_ref(
            moneyline,
            threshold=1,
            positive_outcome=team_a,
            complement_outcome=team_b,
            resolver="moneyline_with_half_half_tie",
        )
    ]
    for market in markets:
        if market.get("sportsMarketType") != "spreads":
            continue
        _common_market_gate(market)
        outcomes = _json_pair(market.get("outcomes"), "spread outcomes")
        if set(outcomes) != {team_a, team_b}:
            raise RuntimeError("spread team identity mismatch")
        line = Decimal(str(market.get("line")))
        if line >= 0 or (-line % 1) != HALF:
            raise RuntimeError("spread line is not a negative half-point")
        required_margin = int(-line + HALF)
        favored = outcomes[0]
        if (
            f'resolve to "{favored}" if the {favored} win the game by '
            f"{required_margin} or more points"
        ) not in str(market["description"]):
            raise RuntimeError("spread description does not bind its exact line")
        if favored == team_a:
            threshold = required_margin
        elif favored == team_b:
            threshold = -(required_margin - 1)
        else:  # pragma: no cover - set equality above proves this unreachable
            raise RuntimeError("spread favored team is unknown")
        prices = dict(
            zip(
                outcomes,
                _json_pair(market.get("outcomePrices"), "spread prices"),
            )
        )
        rows.append(
            {
                **_market_ref(
                    market,
                    threshold=threshold,
                    positive_outcome=team_a,
                    complement_outcome=team_b,
                    resolver="integer_margin_threshold",
                ),
                "positive_price_pUSD": Decimal(prices[team_a]),
                "complement_price_pUSD": Decimal(prices[team_b]),
            }
        )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["threshold"]), []).append(row)
    for threshold, duplicates in grouped.items():
        resolvers = {str(row["resolver"]) for row in duplicates}
        if len(duplicates) > 1 and not (
            threshold == 1
            and len(duplicates) == 2
            and resolvers
            == {"moneyline_with_half_half_tie", "integer_margin_threshold"}
        ):
            raise RuntimeError("unproved duplicate NFL margin threshold")
    return (
        team_a,
        team_b,
        sorted(
            rows,
            key=lambda row: (
                int(row["threshold"]),
                0 if row["resolver"] == "moneyline_with_half_half_tie" else 1,
            ),
        ),
    )


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    if contract["authority"] != {
        "book_requests": 0,
        "fee_requests": 0,
        "network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "trading_authority": False,
    }:
        raise RuntimeError("offline authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")
    source = contract["source"]
    result_path = _root_path(str(source["catalog_result_path"]))
    result_raw = result_path.read_bytes()
    if _sha256(result_raw) != source["catalog_result_file_sha256"]:
        raise RuntimeError("catalog result file hash mismatch")
    catalog_result = json.loads(result_raw)
    if (
        _canonical_hash(catalog_result, "result_sha256")
        != source["catalog_result_sha256"]
    ):
        raise RuntimeError("catalog result canonical hash mismatch")
    expected_exclusion = {
        "event_slug": source["event_slug"],
        "reason": "NFL margin lattice contains duplicate logical thresholds",
    }
    if expected_exclusion not in catalog_result["screen"]["exclusions"]:
        raise RuntimeError("catalog does not preserve the exact collision")
    raw_path = _root_path(str(source["catalog_raw_path"]))
    raw = raw_path.read_bytes()
    if _sha256(raw) != source["catalog_raw_sha256"]:
        raise RuntimeError("catalog raw hash mismatch")
    payload = json.loads(raw)
    event = next(
        (
            row
            for row in payload.get("events", [])
            if str(row.get("slug")) == source["event_slug"]
        ),
        None,
    )
    if event is None:
        raise RuntimeError("excluded event is absent from retained raw catalog")
    return event


def adjudicate(event: dict[str, Any]) -> dict[str, Any]:
    markets = [
        row
        for row in event.get("markets", [])
        if row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
    ]
    moneylines = [row for row in markets if row.get("sportsMarketType") == "moneyline"]
    if len(moneylines) != 1:
        raise RuntimeError("expected exactly one retained moneyline")
    team_a, team_b, margins = _collision_aware_margin_markets(markets, moneylines[0])
    totals = _total_markets(markets)
    families = {"full_game_margin": margins, "full_game_total": totals}
    relations = [
        relation
        for family, rows in families.items()
        for relation in _relations(
            family, rows, minimum_state=-8 if family.endswith("margin") else 0
        )
    ]
    if not relations or any(
        row["minimum_terminal_payout_per_share_pUSD"] < ONE for row in relations
    ):
        raise RuntimeError("corrected relation set lacks a complete payout proof")
    candidates = [row for row in relations if row["passes_strictly_below_payout_gate"]]
    best = min(
        relations,
        key=lambda row: (
            row["displayed_price_sum_per_share_pUSD"],
            row["family"],
            row["superset_threshold"],
            row["subset_threshold"],
        ),
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": {
            "id": str(event["id"]),
            "slug": str(event["slug"]),
            "title": str(event["title"]),
            "start_time_utc": str(event["startTime"]),
        },
        "correction": {
            "preserved_failure": "v1 excluded this event because moneyline and favorite-minus-0.5 share integer win threshold one but have different actual-tie payouts",
            "correct_order": "half-half-tie moneyline positive payout weakly contains favorite-minus-0.5 positive payout at the shared threshold",
            "promotion_or_depth_access_from_correction": False,
        },
        "payoff_proof": {
            "team_a": team_a,
            "team_b": team_b,
            "margin_thresholds": [
                {
                    "market_id": row["market_id"],
                    "threshold": row["threshold"],
                    "resolver": row["resolver"],
                    "positive_outcome": row["positive_outcome"],
                    "complement_outcome": row["complement_outcome"],
                }
                for row in margins
            ],
            "total_threshold_count": len(totals),
            "complete_relation_count": len(relations),
            "relations": relations,
        },
        "rejection_only_gamma_prefilter": {
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "best_relation": best,
            "gamma_can_support_acceptance_or_promotion": False,
        },
        "adjudication": {
            "status": "retained_catalog_coverage_corrected_without_depth_escalation",
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": "do_not_adaptively_request_books_for_this_outcome_aware_correction",
        },
        "authority": {
            "network_requests": 0,
            "book_requests": 0,
            "fee_requests": 0,
            "orders_or_transactions": 0,
            "protected_capture_touched": False,
            "trading_authority": False,
        },
        "implementation": {
            "path": "tools/adjudicate_polymarket_nfl_catalog_tie_collision.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct one retained NFL moneyline/spread tie collision offline."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    event = _validate_contract(contract, contract_path)
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("correction output already exists")
    result = adjudicate(event)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "event_slug": result["event"]["slug"],
                "relation_count": result["payoff_proof"]["complete_relation_count"],
                "candidate_count": result["rejection_only_gamma_prefilter"][
                    "candidate_count_strictly_below_payout_floor"
                ],
                "network_requests": 0,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
