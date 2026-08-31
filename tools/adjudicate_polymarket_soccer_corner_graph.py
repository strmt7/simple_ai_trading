"""Exhaust retained soccer corner-count structural payoff relations."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic
from tools import adjudicate_polymarket_soccer_structural_graph as graph


ROOT = Path(__file__).resolve().parents[1]
TOTAL_TYPES = {
    "total_corners",
    "soccer_first_half_total_corners",
    "soccer_second_half_total_corners",
    "soccer_team_total_corners",
}


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


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract.get("schema_version") != "polymarket-soccer-corner-graph-contract-v1":
        raise ValueError("unexpected contract schema")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise ValueError("contract path mismatch")
    if _canonical_hash(contract, field="contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("contract timestamp is invalid or future")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    dependency = _root_path(str(contract["implementation"]["dependency"]["path"]))
    if (
        _sha256(dependency.read_bytes())
        != contract["implementation"]["dependency"]["sha256"]
    ):
        raise ValueError("dependency hash mismatch")


def _description(market: Mapping[str, Any]) -> str:
    return " ".join(str(market.get("description") or "").split())


def _floor_line(market: Mapping[str, Any]) -> int:
    return int(Decimal(str(market["line"])).to_integral_value(rounding=ROUND_FLOOR))


def _require_corner_rules(market: Mapping[str, Any]) -> None:
    market_type = str(market.get("sportsMarketType"))
    description = _description(market)
    common = (
        "corners taken",
        "not corners awarded",
        "canceled entirely, with no make-up game, this market will resolve 50",
        "official statistics published by the governing body or event organizers",
        "If no acceptable data is available within 48 hours",
        "this market will resolve 50-50",
    )
    if market_type in TOTAL_TYPES:
        outcomes = json.loads(str(market.get("outcomes")))
        if outcomes != ["Over", "Under"] or not all(
            phrase in description for phrase in common
        ):
            raise ValueError(f"corner-total rule changed: {market.get('id')}")
        return
    if market_type == "soccer_game_corners_odd_even":
        outcomes = json.loads(str(market.get("outcomes")))
        parity = (
            'resolve to "Odd"',
            'resolve to "Even"',
            "Zero corners is considered even",
        )
        if outcomes != ["Odd", "Even"] or not all(
            phrase in description for phrase in (*common, *parity)
        ):
            raise ValueError(f"corner-parity rule changed: {market.get('id')}")
        return
    raise ValueError(f"unexpected corner market type: {market_type}")


def _price(market: Mapping[str, Any], *, side: str) -> Decimal | None:
    if side == "first_outcome":
        value = market.get("bestAsk")
        return None if value is None else Decimal(str(value))
    value = market.get("bestBid")
    return None if value is None else Decimal("1") - Decimal(str(value))


def _relation(
    *,
    family: str,
    base_slug: str,
    legs: list[tuple[Mapping[str, Any], str]],
    proof: str,
    cancellation_floor_pusd: str,
) -> dict[str, Any]:
    prices: list[Decimal | None] = []
    identities: list[dict[str, Any]] = []
    for market, side in legs:
        graph._require_active(market)
        prices.append(_price(market, side=side))
        identities.append(graph._market_identity(market, side=side))
    complete = all(value is not None for value in prices)
    proxy_sum = sum((value for value in prices if value is not None), Decimal("0")) if complete else None
    return {
        "family": family,
        "base_slug": base_slug,
        "legs": identities,
        "payoff_proof": proof,
        "common_rule_floor_pUSD": "1",
        "cancellation_floor_pUSD": cancellation_floor_pusd,
        "side_specific_prices_complete": complete,
        "rejection_proxy_sum_pUSD": (
            graph._decimal_text(proxy_sum) if proxy_sum is not None else None
        ),
        "optimistic_proxy_headroom_pUSD": (
            graph._decimal_text(Decimal("1") - proxy_sum)
            if proxy_sum is not None
            else None
        ),
        "passes_strict_side_specific_rejection_gate": bool(
            proxy_sum is not None and proxy_sum < 1
        ),
    }


def _monotone_relations(
    *,
    base_slug: str,
    family: str,
    markets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(markets, key=lambda market: Decimal(str(market["line"])))
    rows: list[dict[str, Any]] = []
    for index, lower in enumerate(ordered):
        for higher in ordered[index + 1 :]:
            rows.append(
                _relation(
                    family=family,
                    base_slug=base_slug,
                    legs=[
                        (lower, "first_outcome"),
                        (higher, "second_outcome"),
                    ],
                    proof=(
                        f"Over {lower['line']} plus Under {higher['line']} "
                        "covers every integer corner count"
                    ),
                    cancellation_floor_pusd="1",
                )
            )
    return rows


def _partition_relations(
    *,
    base_slug: str,
    partition_name: str,
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    full: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first_market in first:
        for second_market in second:
            min_sum_when_both_over = _floor_line(first_market) + _floor_line(second_market) + 2
            max_sum_when_both_under = _floor_line(first_market) + _floor_line(second_market)
            for full_market in full:
                full_line = Decimal(str(full_market["line"]))
                if Decimal(min_sum_when_both_over) > full_line:
                    rows.append(
                        _relation(
                            family=f"{partition_name}_both_over_implies_full_over",
                            base_slug=base_slug,
                            legs=[
                                (first_market, "second_outcome"),
                                (second_market, "second_outcome"),
                                (full_market, "first_outcome"),
                            ],
                            proof=(
                                f"both component counts above {first_market['line']} and "
                                f"{second_market['line']} sum to at least "
                                f"{min_sum_when_both_over}, implying full Over {full_line}"
                            ),
                            cancellation_floor_pusd="1.5",
                        )
                    )
                if Decimal(max_sum_when_both_under) < full_line:
                    rows.append(
                        _relation(
                            family=f"{partition_name}_both_under_implies_full_under",
                            base_slug=base_slug,
                            legs=[
                                (first_market, "first_outcome"),
                                (second_market, "first_outcome"),
                                (full_market, "second_outcome"),
                            ],
                            proof=(
                                f"both component counts at or below {_floor_line(first_market)} "
                                f"and {_floor_line(second_market)} sum to at most "
                                f"{max_sum_when_both_under}, implying full Under {full_line}"
                            ),
                            cancellation_floor_pusd="1.5",
                        )
                    )
    return rows


def _enumerate(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("retained events must be a list")
    relations: list[dict[str, Any]] = []
    complete_event_count = 0

    for event_value in sorted(events, key=lambda row: str(row.get("slug"))):
        event = _mapping(event_value, name="event")
        base_slug = str(event.get("slug"))
        if not base_slug.endswith("-total-corners"):
            continue
        market_values = event.get("markets")
        if not isinstance(market_values, list) or len(market_values) != 23:
            raise ValueError(f"corner event cardinality changed: {base_slug}")
        markets = [_mapping(value, name="corner market") for value in market_values]
        for market in markets:
            if market.get("sportsMarketType") != "soccer_first_corner":
                _require_corner_rules(market)
        full = [market for market in markets if market.get("sportsMarketType") == "total_corners"]
        first_half = [
            market
            for market in markets
            if market.get("sportsMarketType") == "soccer_first_half_total_corners"
        ]
        second_half = [
            market
            for market in markets
            if market.get("sportsMarketType") == "soccer_second_half_total_corners"
        ]
        team_markets = [
            market
            for market in markets
            if market.get("sportsMarketType") == "soccer_team_total_corners"
        ]
        parity = [
            market
            for market in markets
            if market.get("sportsMarketType") == "soccer_game_corners_odd_even"
        ]
        if not (
            len(full) == 7
            and len(first_half) == 3
            and len(second_half) == 3
            and len(team_markets) == 8
            and len(parity) == 1
        ):
            raise ValueError(f"corner subfamily cardinality changed: {base_slug}")
        team_groups: dict[str, list[dict[str, Any]]] = {}
        for market in team_markets:
            title = str(market.get("groupItemTitle"))
            team = title.split(" Corners: O/U ", 1)[0]
            team_groups.setdefault(team, []).append(market)
        if sorted(len(group) for group in team_groups.values()) != [4, 4]:
            raise ValueError(f"team-corner grouping changed: {base_slug}")
        complete_event_count += 1

        relations.extend(
            _monotone_relations(
                base_slug=base_slug,
                family="full_corner_total_monotone",
                markets=full,
            )
        )
        relations.extend(
            _monotone_relations(
                base_slug=base_slug,
                family="first_half_corner_total_monotone",
                markets=first_half,
            )
        )
        relations.extend(
            _monotone_relations(
                base_slug=base_slug,
                family="second_half_corner_total_monotone",
                markets=second_half,
            )
        )
        for group in team_groups.values():
            relations.extend(
                _monotone_relations(
                    base_slug=base_slug,
                    family="team_corner_total_monotone",
                    markets=group,
                )
            )
        relations.extend(
            _partition_relations(
                base_slug=base_slug,
                partition_name="half_partition",
                first=first_half,
                second=second_half,
                full=full,
            )
        )
        team_names = sorted(team_groups)
        relations.extend(
            _partition_relations(
                base_slug=base_slug,
                partition_name="team_partition",
                first=team_groups[team_names[0]],
                second=team_groups[team_names[1]],
                full=full,
            )
        )

        ordered_full = sorted(full, key=lambda market: Decimal(str(market["line"])))
        for lower, higher in zip(ordered_full, ordered_full[1:], strict=True):
            if Decimal(str(higher["line"])) - Decimal(str(lower["line"])) != 1:
                raise ValueError(f"nonadjacent full-corner ladder: {base_slug}")
            exact_count = _floor_line(higher)
            parity_side = "first_outcome" if exact_count % 2 else "second_outcome"
            relations.append(
                _relation(
                    family="adjacent_full_corner_interval_implies_parity",
                    base_slug=base_slug,
                    legs=[
                        (lower, "second_outcome"),
                        (higher, "first_outcome"),
                        (parity[0], parity_side),
                    ],
                    proof=(
                        f"Over {lower['line']} and Under {higher['line']} fix total "
                        f"corners at {exact_count}, implying its parity"
                    ),
                    cancellation_floor_pusd="1.5",
                )
            )

    identities = {
        tuple((leg["market_id"], leg["outcome"]) for leg in relation["legs"])
        for relation in relations
    }
    if len(identities) != len(relations):
        raise ValueError("duplicate corner relation emitted")
    family_counts = Counter(str(relation["family"]) for relation in relations)
    return relations, {
        "complete_event_count": complete_event_count,
        "relation_count": len(relations),
        **{f"{family}_count": count for family, count in sorted(family_counts.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = _mapping(json.loads(contract_path.read_bytes()), name="contract")
    _validate_contract(contract, contract_path)
    source = _mapping(contract["retained_source"], name="retained source")
    source_path = _root_path(str(source["path"]))
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != source["sha256"]:
        raise ValueError("retained source hash mismatch")
    payload = _mapping(json.loads(source_bytes), name="retained payload")
    relations, population = _enumerate(payload)
    complete = [row for row in relations if row["side_specific_prices_complete"]]
    ranked = sorted(
        complete,
        key=lambda row: (
            Decimal(str(row["rejection_proxy_sum_pUSD"])),
            str(row["family"]),
            str(row["base_slug"]),
            tuple(str(leg["market_id"]) for leg in row["legs"]),
        ),
    )
    strict = [row for row in ranked if row["passes_strict_side_specific_rejection_gate"]]
    result: dict[str, Any] = {
        "schema_version": "polymarket-soccer-corner-graph-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": args.contract,
            "sha256": contract["contract_sha256"],
        },
        "retained_source": source,
        "population": {
            **population,
            "side_specific_price_complete_count": len(complete),
            "side_specific_price_missing_count": len(relations) - len(complete),
            "strict_sub_floor_count": len(strict),
        },
        "best_complete_relation": ranked[0] if ranked else None,
        "strict_sub_floor_relations": strict[:20],
        "adjudication": {
            "status": (
                "historical_proxy_candidate_requires_future_distinct_prospective_event"
                if strict
                else "no_strict_side_specific_sub_floor_candidate"
            ),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "current_book_request_authorized": False,
            "next_action": (
                "freeze_one_future_distinct_active_equivalent_relation_before_any_exact_book_request"
                if strict
                else "terminalize_this_retained_population_without_any_market_request"
            ),
        },
        "authority": contract["authority"],
        "result_sha256": "",
    }
    result["result_sha256"] = _canonical_hash(result, field="result_sha256")
    output_path = _root_path(str(contract["output_path"]))
    write_bytes_atomic(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        _canonical_json(
            {
                "result_path": str(contract["output_path"]),
                "result_sha256": result["result_sha256"],
                "relations": len(relations),
                "complete": len(complete),
                "strict": len(strict),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
