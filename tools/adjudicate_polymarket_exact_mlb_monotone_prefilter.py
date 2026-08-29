from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-exact-mlb-monotone-prefilter-adjudication-v1"
HALF = Decimal("0.5")
ONE = Decimal("1")


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _json_pair(value: object, label: str) -> list[str]:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a JSON string")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise RuntimeError(f"{label} must contain exactly two values")
    return [str(item) for item in parsed]


def _prices(market: dict[str, Any]) -> dict[str, Decimal]:
    outcomes = _json_pair(market.get("outcomes"), "outcomes")
    prices = _json_pair(market.get("outcomePrices"), "outcomePrices")
    mapped = {outcome: Decimal(price) for outcome, price in zip(outcomes, prices)}
    if any(price < 0 or price > 1 for price in mapped.values()):
        raise RuntimeError("Gamma outcome price is outside [0, 1]")
    return mapped


def _common_market_gate(market: dict[str, Any]) -> None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
    ):
        raise RuntimeError("included market is not active, open, and accepting")
    description = market.get("description")
    if not isinstance(description, str) or "resolve 50-50" not in description:
        raise RuntimeError("included market lacks exact 50-50 cancellation semantics")


def _market_ref(
    market: dict[str, Any],
    *,
    threshold: int,
    positive_outcome: str,
    complement_outcome: str,
    resolver: str,
) -> dict[str, Any]:
    prices = _prices(market)
    return {
        "market_id": str(market["id"]),
        "market_slug": str(market["slug"]),
        "question": str(market["question"]),
        "threshold": threshold,
        "positive_outcome": positive_outcome,
        "positive_price_pUSD": prices[positive_outcome],
        "complement_outcome": complement_outcome,
        "complement_price_pUSD": prices[complement_outcome],
        "resolver": resolver,
    }


def _margin_markets(
    markets: list[dict[str, Any]],
    *,
    moneyline: dict[str, Any] | None,
    market_type: str,
    team_a: str,
    team_b: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if moneyline is not None:
        _common_market_gate(moneyline)
        outcomes = _json_pair(moneyline.get("outcomes"), "moneyline outcomes")
        if outcomes != [team_a, team_b]:
            raise RuntimeError("moneyline outcomes changed order or identity")
        description = str(moneyline["description"])
        if "ends in a tie" not in description:
            raise RuntimeError("moneyline lacks exact tie semantics")
        rows.append(
            _market_ref(
                moneyline,
                threshold=1,
                positive_outcome=team_a,
                complement_outcome=team_b,
                resolver="moneyline_with_half_half_tie",
            )
        )

    for market in markets:
        if market.get("sportsMarketType") != market_type:
            continue
        _common_market_gate(market)
        outcomes = _json_pair(market.get("outcomes"), f"{market_type} outcomes")
        if set(outcomes) != {team_a, team_b}:
            raise RuntimeError(f"{market_type} team identity mismatch")
        line = Decimal(str(market.get("line")))
        if line >= 0 or (-line % 1) != HALF:
            raise RuntimeError(f"{market_type} line is not a negative half-run")
        required_margin = int(-line + HALF)
        favored = outcomes[0]
        description = str(market["description"])
        if f"win the game by {required_margin} or more runs" not in description:
            if market_type != "baseball_team_first_five_spread" or (
                f"winning the game by {required_margin} or more runs" not in description
            ):
                raise RuntimeError(f"{market_type} description does not bind its line")
        if favored == team_a:
            threshold = required_margin
            positive, complement = team_a, team_b
        elif favored == team_b:
            threshold = -(required_margin - 1)
            positive, complement = team_a, team_b
        else:
            raise RuntimeError(f"{market_type} favored team is unknown")
        rows.append(
            _market_ref(
                market,
                threshold=threshold,
                positive_outcome=positive,
                complement_outcome=complement,
                resolver="integer_margin_threshold",
            )
        )
    thresholds = [row["threshold"] for row in rows]
    if len(thresholds) != len(set(thresholds)):
        raise RuntimeError(f"{market_type} contains duplicate logical thresholds")
    return sorted(rows, key=lambda row: row["threshold"])


def _total_markets(
    markets: list[dict[str, Any]], market_type: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        if market.get("sportsMarketType") != market_type:
            continue
        _common_market_gate(market)
        outcomes = _json_pair(market.get("outcomes"), f"{market_type} outcomes")
        if outcomes != ["Over", "Under"]:
            raise RuntimeError(f"{market_type} outcomes are not Over/Under")
        line = Decimal(str(market.get("line")))
        if line < 0 or (line % 1) != HALF:
            raise RuntimeError(f"{market_type} line is not a positive half-run")
        threshold = int(line + HALF)
        description = str(market["description"])
        if f"score {threshold} or more runs" not in description:
            raise RuntimeError(f"{market_type} description does not bind its line")
        rows.append(
            _market_ref(
                market,
                threshold=threshold,
                positive_outcome="Over",
                complement_outcome="Under",
                resolver="integer_total_threshold",
            )
        )
    thresholds = [row["threshold"] for row in rows]
    if len(thresholds) != len(set(thresholds)):
        raise RuntimeError(f"{market_type} contains duplicate logical thresholds")
    return sorted(rows, key=lambda row: row["threshold"])


def _positive_payout(row: dict[str, Any], state: int | str) -> Decimal:
    if state == "canceled":
        return HALF
    if row["resolver"] == "moneyline_with_half_half_tie" and state == 0:
        return HALF
    return ONE if int(state) >= int(row["threshold"]) else Decimal("0")


def _relations(
    family: str, rows: list[dict[str, Any]], *, minimum_state: int
) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    maximum = max(int(row["threshold"]) for row in rows) + 2
    minimum = min(minimum_state, min(int(row["threshold"]) for row in rows) - 2)
    states: list[int | str] = [*range(minimum, maximum + 1), "canceled"]
    relations: list[dict[str, Any]] = []
    for lower_index, superset in enumerate(rows[:-1]):
        for subset in rows[lower_index + 1 :]:
            payouts = []
            for state in states:
                superset_positive = _positive_payout(superset, state)
                subset_complement = ONE - _positive_payout(subset, state)
                payouts.append(
                    {
                        "state": state,
                        "package_payout_pUSD": superset_positive + subset_complement,
                    }
                )
            minimum_payout = min(row["package_payout_pUSD"] for row in payouts)
            price_sum = (
                superset["positive_price_pUSD"] + subset["complement_price_pUSD"]
            )
            relations.append(
                {
                    "family": family,
                    "superset_threshold": superset["threshold"],
                    "superset_positive_market_id": superset["market_id"],
                    "superset_positive_outcome": superset["positive_outcome"],
                    "superset_positive_price_pUSD": superset[
                        "positive_price_pUSD"
                    ],
                    "subset_threshold": subset["threshold"],
                    "subset_complement_market_id": subset["market_id"],
                    "subset_complement_outcome": subset["complement_outcome"],
                    "subset_complement_price_pUSD": subset[
                        "complement_price_pUSD"
                    ],
                    "minimum_terminal_payout_per_share_pUSD": minimum_payout,
                    "displayed_price_sum_per_share_pUSD": price_sum,
                    "optimistic_profit_floor_per_share_before_execution_costs_pUSD": (
                        minimum_payout - price_sum
                    ),
                    "passes_strictly_below_payout_gate": price_sum < minimum_payout,
                    "minimum_payout_states": [
                        row["state"]
                        for row in payouts
                        if row["package_payout_pUSD"] == minimum_payout
                    ],
                }
            )
    return relations


def adjudicate(metadata: dict[str, Any]) -> dict[str, Any]:
    if _canonical_hash(metadata, "result_sha256") != metadata.get("result_sha256"):
        raise RuntimeError("metadata result hash mismatch")
    if not (
        metadata["capture"]["exact_slug_match"] is True
        and metadata["capture"]["event_active_and_open"] is True
        and metadata["discovery"]["exact_moneyline_spread_candidate"] is True
    ):
        raise RuntimeError("metadata does not prove an exact active MLB candidate")
    markets = metadata["discovery"]["active_accepting_markets"]
    moneylines = [row for row in markets if row.get("sportsMarketType") == "moneyline"]
    if len(moneylines) != 1:
        raise RuntimeError("expected exactly one moneyline")
    teams = _json_pair(moneylines[0].get("outcomes"), "moneyline outcomes")
    team_a, team_b = teams
    families = {
        "full_game_margin": _margin_markets(
            markets,
            moneyline=moneylines[0],
            market_type="spreads",
            team_a=team_a,
            team_b=team_b,
        ),
        "first_five_margin": _margin_markets(
            markets,
            moneyline=None,
            market_type="baseball_team_first_five_spread",
            team_a=team_a,
            team_b=team_b,
        ),
        "full_game_total": _total_markets(markets, "totals"),
        "first_five_total": _total_markets(
            markets, "baseball_team_first_five_total"
        ),
    }
    relations = []
    for family, rows in families.items():
        relations.extend(
            _relations(
                family,
                rows,
                minimum_state=-8 if "margin" in family else 0,
            )
        )
    if not relations:
        raise RuntimeError("no exact monotone relations were proved")
    if any(row["minimum_terminal_payout_per_share_pUSD"] < ONE for row in relations):
        raise RuntimeError("a generated package lacks its guaranteed payout floor")
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
    authority = dict(metadata["authority"])
    authority.update(
        {
            "offline_adjudication": True,
            "book_requests": 0,
            "fee_requests": 0,
            "orders_or_transactions": 0,
            "protected_capture_touched": False,
            "trading_authority": False,
        }
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "source_metadata": {
            "event_slug": metadata["event"]["slug"],
            "event_title": metadata["event"]["title"],
            "result_sha256": metadata["result_sha256"],
            "raw_response_sha256": metadata["capture"]["receipt"][
                "response_sha256"
            ],
            "public_request_count": 1,
        },
        "payoff_proof": {
            "team_a": team_a,
            "team_b": team_b,
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
            "cancellation_payout_per_leg_pUSD": str(HALF),
        },
        "rejection_only_gamma_prefilter": {
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "all_displayed_price_sums_at_or_above_payout_floor": not candidates,
            "best_relation": best,
            "best_optimistic_profit_floor_at_five_shares_before_execution_costs_pUSD": (
                best[
                    "optimistic_profit_floor_per_share_before_execution_costs_pUSD"
                ]
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
        "authority": authority,
        "implementation": {
            "path": "tools/adjudicate_polymarket_exact_mlb_monotone_prefilter.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjudicate retained exact MLB Gamma metadata without network access."
    )
    parser.add_argument("--metadata-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata_path = _root_path(args.metadata_result)
    output_path = _root_path(args.output)
    if output_path.exists():
        raise RuntimeError("adjudication output already exists")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
                "best_family": best["family"],
                "best_displayed_price_sum_pUSD": str(
                    best["displayed_price_sum_per_share_pUSD"]
                ),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
