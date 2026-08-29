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


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-exact-nfl-monotone-prefilter-adjudication-v1"


def _margin_markets(
    markets: list[dict[str, Any]], moneyline: dict[str, Any]
) -> tuple[str, str, list[dict[str, Any]]]:
    _common_market_gate(moneyline)
    teams = _json_pair(moneyline.get("outcomes"), "moneyline outcomes")
    team_a, team_b = teams
    description = str(moneyline["description"])
    if "ends in a tie" not in description:
        raise RuntimeError("NFL moneyline lacks exact half-half tie semantics")
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
                _json_pair(market.get("outcomes"), "spread outcomes"),
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
                "favored_team": favored,
                "line": line,
            }
        )
    thresholds = [int(row["threshold"]) for row in rows]
    if len(thresholds) != len(set(thresholds)):
        raise RuntimeError("NFL margin lattice contains duplicate logical thresholds")
    return team_a, team_b, sorted(rows, key=lambda row: int(row["threshold"]))


def _total_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        if market.get("sportsMarketType") != "totals":
            continue
        _common_market_gate(market)
        if _json_pair(market.get("outcomes"), "total outcomes") != [
            "Over",
            "Under",
        ]:
            raise RuntimeError("NFL total outcomes are not Over/Under")
        line = Decimal(str(market.get("line")))
        if line < 0 or (line % 1) != HALF:
            raise RuntimeError("NFL total line is not a positive half-point")
        threshold = int(line + HALF)
        description = str(market["description"])
        if f"combine to score {threshold} or more points" not in description:
            raise RuntimeError("NFL total description does not bind its exact line")
        rows.append(
            _market_ref(
                market,
                threshold=threshold,
                positive_outcome="Over",
                complement_outcome="Under",
                resolver="integer_total_threshold",
            )
        )
    thresholds = [int(row["threshold"]) for row in rows]
    if len(thresholds) != len(set(thresholds)):
        raise RuntimeError("NFL total lattice contains duplicate logical thresholds")
    return sorted(rows, key=lambda row: int(row["threshold"]))


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    if contract["authority"] != {
        "network_requests": 0,
        "book_requests": 0,
        "fee_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "trading_authority": False,
    }:
        raise RuntimeError("offline authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")
    source = contract["metadata_source"]
    source_path = _root_path(str(source["path"]))
    raw = source_path.read_bytes()
    if _sha256(raw) != source["file_sha256"]:
        raise RuntimeError("metadata file hash mismatch")
    metadata = json.loads(raw)
    if _canonical_hash(metadata, "result_sha256") != source["result_sha256"]:
        raise RuntimeError("metadata canonical hash mismatch")
    if (
        metadata["capture"]["receipt"]["response_sha256"]
        != source["raw_response_sha256"]
    ):
        raise RuntimeError("raw response binding mismatch")
    return metadata


def adjudicate(metadata: dict[str, Any]) -> dict[str, Any]:
    if _canonical_hash(metadata, "result_sha256") != metadata.get("result_sha256"):
        raise RuntimeError("metadata result hash mismatch")
    if not (
        metadata["capture"]["exact_slug_match"] is True
        and metadata["capture"]["event_active_and_open"] is True
        and metadata["discovery"]["exact_moneyline_spread_candidate"] is True
    ):
        raise RuntimeError("metadata does not prove an exact active NFL candidate")
    markets = metadata["discovery"]["active_accepting_markets"]
    moneylines = [row for row in markets if row.get("sportsMarketType") == "moneyline"]
    if len(moneylines) != 1:
        raise RuntimeError("expected exactly one NFL moneyline")
    team_a, team_b, margins = _margin_markets(markets, moneylines[0])
    totals = _total_markets(markets)
    families = {"full_game_margin": margins, "full_game_total": totals}
    relations = [
        relation
        for family, rows in families.items()
        for relation in _relations(
            family, rows, minimum_state=-8 if family.endswith("margin") else 0
        )
    ]
    if not relations:
        raise RuntimeError("no exact NFL monotone relations were proved")
    if any(row["minimum_terminal_payout_per_share_pUSD"] < ONE for row in relations):
        raise RuntimeError("a generated NFL package lacks its payout floor")
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
        "source_metadata": {
            "event_slug": metadata["event"]["slug"],
            "event_title": metadata["event"]["title"],
            "result_sha256": metadata["result_sha256"],
            "raw_response_sha256": metadata["capture"]["receipt"]["response_sha256"],
            "public_request_count": 1,
        },
        "payoff_proof": {
            "team_a": team_a,
            "team_b": team_b,
            "state_variable": f"integer final score margin {team_a} minus {team_b}",
            "actual_game_tie_state": 0,
            "cancellation_payout_per_leg_pUSD": HALF,
            "family_market_thresholds": {
                family: [
                    {
                        "market_id": row["market_id"],
                        "threshold": row["threshold"],
                        "positive_outcome": row["positive_outcome"],
                        "complement_outcome": row["complement_outcome"],
                        "resolver": row["resolver"],
                    }
                    for row in rows
                ]
                for family, rows in families.items()
            },
            "complete_relation_count": len(relations),
            "relations": relations,
        },
        "rejection_only_gamma_prefilter": {
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "all_displayed_price_sums_at_or_above_payout_floor": not candidates,
            "best_relation": best,
            "best_optimistic_profit_floor_at_five_shares_before_execution_costs_pUSD": (
                best["optimistic_profit_floor_per_share_before_execution_costs_pUSD"]
                * Decimal("5")
            ),
            "gamma_can_support_acceptance_or_promotion": False,
        },
        "adjudication": {
            "status": (
                "candidate_requires_separately_frozen_exact_depth_screen"
                if candidates
                else "terminal_exact_event_rejected_before_books_and_fees"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_exact_token_depth_and_fee_screen_without_trading"
                if candidates
                else "do_not_request_books_or_fees_for_this_exact_event"
            ),
        },
        "authority": {
            **metadata["authority"],
            "offline_adjudication": True,
            "book_requests": 0,
            "fee_requests": 0,
            "orders_or_transactions": 0,
            "protected_capture_touched": False,
            "trading_authority": False,
        },
        "implementation": {
            "path": "tools/adjudicate_polymarket_exact_nfl_monotone_prefilter.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
            "dependency_path": "tools/adjudicate_polymarket_exact_mlb_monotone_prefilter.py",
            "dependency_sha256": _sha256(
                (
                    ROOT / "tools/adjudicate_polymarket_exact_mlb_monotone_prefilter.py"
                ).read_bytes()
            ),
        },
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjudicate one frozen exact NFL Gamma event without network access."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    metadata = _validate_contract(contract, contract_path)
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("adjudication output already exists")
    result = adjudicate(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    best = result["rejection_only_gamma_prefilter"]["best_relation"]
    print(
        json.dumps(
            {
                "candidate_count": result["rejection_only_gamma_prefilter"][
                    "candidate_count_strictly_below_payout_floor"
                ],
                "complete_relation_count": result["payoff_proof"][
                    "complete_relation_count"
                ],
                "best_displayed_price_sum_pUSD": best[
                    "displayed_price_sum_per_share_pUSD"
                ],
                "best_family": best["family"],
                "network_requests": 0,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
