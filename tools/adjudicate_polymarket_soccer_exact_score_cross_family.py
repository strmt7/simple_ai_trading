"""Exhaust retained soccer exact-score cross-family payoff implications."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic
from tools import adjudicate_polymarket_soccer_structural_graph as graph


ROOT = Path(__file__).resolve().parents[1]


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
    if (
        contract.get("schema_version")
        != "polymarket-soccer-exact-score-cross-family-contract-v1"
    ):
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


def _require_rules(
    *,
    exact_markets: list[dict[str, Any]],
    first_markets: list[dict[str, Any]],
    more_markets: list[dict[str, Any]],
) -> None:
    for market in exact_markets:
        description = _description(market)
        if not (
            'canceled with no make-up game, the market resolves to "0-0."'
            in description
            and "90 minutes of regulation plus stoppage time" in description
        ):
            raise ValueError(f"exact-score rule changed: {market.get('id')}")
    for market in first_markets:
        description = _description(market)
        if not (
            'canceled entirely, with no make-up game, this market will resolve to "Neither"'
            in description
            and "first 90 minutes of regular play plus stoppage time" in description
        ):
            raise ValueError(f"first-to-score rule changed: {market.get('id')}")
    for market in more_markets:
        description = _description(market)
        if not (
            "canceled entirely, with no make-up game" in description
            and "resolve 50" in description
            and "first 90 minutes of regular play plus stoppage time" in description
        ):
            raise ValueError(f"cross-family market rule changed: {market.get('id')}")


def _exact_scores(
    event: Mapping[str, Any], *, home: str, away: str
) -> dict[tuple[int, int], dict[str, Any]]:
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) != 17:
        raise ValueError(f"exact-score cardinality changed: {event.get('slug')}")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for value in markets:
        market = _mapping(value, name="exact-score market")
        label = str(market.get("groupItemTitle"))
        if label == "Any Other Score":
            continue
        match = graph.SCORE_RE.fullmatch(label)
        if (
            match is None
            or match.group("home") != home
            or match.group("away") != away
        ):
            raise ValueError(f"unparseable exact score: {label}")
        score = (int(match.group("h")), int(match.group("a")))
        if score in result:
            raise ValueError(f"duplicate exact score: {score}")
        result[score] = market
    if len(result) != 16 or (0, 0) not in result:
        raise ValueError("explicit exact-score grid changed")
    return result


def _event_markets(event: Mapping[str, Any], *, name: str) -> list[dict[str, Any]]:
    values = event.get("markets")
    if not isinstance(values, list):
        raise ValueError(f"{name} markets must be a list")
    return [_mapping(value, name=f"{name} market") for value in values]


def _relation(
    *,
    family: str,
    base_slug: str,
    first_market: Mapping[str, Any],
    first_side: str,
    second_market: Mapping[str, Any],
    second_side: str,
    proof: str,
    cancellation_floor_pusd: str,
) -> dict[str, Any]:
    return graph._relation(
        family=family,
        base_slug=base_slug,
        first_market=first_market,
        first_side=first_side,
        second_market=second_market,
        second_side=second_side,
        proof=proof,
        cancellation_floor_pUSD=cancellation_floor_pusd,
    )


def _enumerate(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("retained events must be a list")
    by_slug = {str(event["slug"]): _mapping(event, name="event") for event in events}
    relations: list[dict[str, Any]] = []
    complete_base_count = 0

    for base_slug, main in sorted(by_slug.items()):
        if any(
            base_slug.endswith(suffix)
            for suffix in ("-exact-score", "-first-to-score", "-more-markets")
        ):
            continue
        exact_event = by_slug.get(f"{base_slug}-exact-score")
        first_event = by_slug.get(f"{base_slug}-first-to-score")
        more_event = by_slug.get(f"{base_slug}-more-markets")
        if exact_event is None or first_event is None or more_event is None:
            continue
        complete_base_count += 1
        home, away, _ = graph._main_team_markets(main)
        exact_by_score = _exact_scores(exact_event, home=home, away=away)
        first_markets = _event_markets(first_event, name="first-to-score")
        if len(first_markets) != 3:
            raise ValueError(f"first-to-score cardinality changed: {base_slug}")
        first_by_label = {
            str(market.get("groupItemTitle")): market for market in first_markets
        }
        if set(first_by_label) != {home, away, "Neither"}:
            raise ValueError(f"first-to-score labels changed: {base_slug}")
        more_markets = _event_markets(more_event, name="more")
        full_totals = sorted(
            (
                market
                for market in more_markets
                if market.get("sportsMarketType") == "totals"
            ),
            key=lambda market: Decimal(str(market["line"])),
        )
        team_totals = [
            market
            for market in more_markets
            if market.get("sportsMarketType") == "soccer_team_totals"
        ]
        btts = next(
            (
                market
                for market in more_markets
                if market.get("sportsMarketType") == "both_teams_to_score"
            ),
            None,
        )
        over_half = next(
            (
                market
                for market in full_totals
                if Decimal(str(market["line"])) == Decimal("0.5")
            ),
            None,
        )
        if btts is None or over_half is None or not full_totals or not team_totals:
            raise ValueError(f"required cross-family markets absent: {base_slug}")
        _require_rules(
            exact_markets=list(exact_by_score.values()),
            first_markets=first_markets,
            more_markets=[*full_totals, *team_totals, btts],
        )
        zero_zero = exact_by_score[(0, 0)]
        neither = first_by_label["Neither"]
        relations.extend(
            [
                _relation(
                    family="exact_zero_zero_equivalent_neither_first_to_score",
                    base_slug=base_slug,
                    first_market=zero_zero,
                    first_side="second_outcome",
                    second_market=neither,
                    second_side="first_outcome",
                    proof="not exact 0-0 plus neither scores first pays one in every completed score state",
                    cancellation_floor_pusd="1",
                ),
                _relation(
                    family="neither_first_to_score_equivalent_exact_zero_zero",
                    base_slug=base_slug,
                    first_market=neither,
                    first_side="second_outcome",
                    second_market=zero_zero,
                    second_side="first_outcome",
                    proof="not neither plus exact 0-0 pays one in every completed score state",
                    cancellation_floor_pusd="1",
                ),
                _relation(
                    family="under_zero_point_five_implies_exact_zero_zero",
                    base_slug=base_slug,
                    first_market=zero_zero,
                    first_side="first_outcome",
                    second_market=over_half,
                    second_side="first_outcome",
                    proof="exact 0-0 plus Over 0.5 pays one because Under 0.5 is exactly 0-0",
                    cancellation_floor_pusd="1.5",
                ),
            ]
        )

        for (home_score, away_score), exact in sorted(exact_by_score.items()):
            if home_score == 0 and away_score == 0:
                continue
            if home_score > 0 and away_score == 0:
                relations.append(
                    _relation(
                        family="one_sided_exact_score_implies_first_scorer",
                        base_slug=base_slug,
                        first_market=exact,
                        first_side="second_outcome",
                        second_market=first_by_label[home],
                        second_side="first_outcome",
                        proof=f"score {home_score}-0 implies {home} scored first",
                        cancellation_floor_pusd="1",
                    )
                )
            elif away_score > 0 and home_score == 0:
                relations.append(
                    _relation(
                        family="one_sided_exact_score_implies_first_scorer",
                        base_slug=base_slug,
                        first_market=exact,
                        first_side="second_outcome",
                        second_market=first_by_label[away],
                        second_side="first_outcome",
                        proof=f"score 0-{away_score} implies {away} scored first",
                        cancellation_floor_pusd="1",
                    )
                )

            btts_side = (
                "first_outcome"
                if home_score > 0 and away_score > 0
                else "second_outcome"
            )
            relations.append(
                _relation(
                    family="nonzero_exact_score_implies_btts_side",
                    base_slug=base_slug,
                    first_market=exact,
                    first_side="second_outcome",
                    second_market=btts,
                    second_side=btts_side,
                    proof=(
                        f"score {home_score}-{away_score} fixes whether both teams score"
                    ),
                    cancellation_floor_pusd="1.5",
                )
            )

            total_goals = home_score + away_score
            for total_market in full_totals:
                line = Decimal(str(total_market["line"]))
                total_side = (
                    "first_outcome"
                    if Decimal(total_goals) > line
                    else "second_outcome"
                )
                relations.append(
                    _relation(
                        family="nonzero_exact_score_implies_full_game_total_side",
                        base_slug=base_slug,
                        first_market=exact,
                        first_side="second_outcome",
                        second_market=total_market,
                        second_side=total_side,
                        proof=(
                            f"score {home_score}-{away_score} fixes the {line} full-game total side"
                        ),
                        cancellation_floor_pusd="1.5",
                    )
                )

            for team_total in team_totals:
                title = str(team_total.get("groupItemTitle") or "")
                line = Decimal(str(team_total["line"]))
                if title.startswith(f"{home} O/U "):
                    team_score = home_score
                    team = home
                elif title.startswith(f"{away} O/U "):
                    team_score = away_score
                    team = away
                else:
                    raise ValueError(f"team-total title changed: {title}")
                total_side = (
                    "first_outcome"
                    if Decimal(team_score) > line
                    else "second_outcome"
                )
                relations.append(
                    _relation(
                        family="nonzero_exact_score_implies_team_total_side",
                        base_slug=base_slug,
                        first_market=exact,
                        first_side="second_outcome",
                        second_market=team_total,
                        second_side=total_side,
                        proof=(
                            f"score {home_score}-{away_score} fixes {team} {line} team-total side"
                        ),
                        cancellation_floor_pusd="1.5",
                    )
                )

    identities = {
        tuple((leg["market_id"], leg["outcome"]) for leg in relation["legs"])
        for relation in relations
    }
    if len(identities) != len(relations):
        raise ValueError("duplicate cross-family relation emitted")
    family_counts = Counter(str(relation["family"]) for relation in relations)
    return relations, {
        "complete_base_count": complete_base_count,
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
    strict = [row for row in complete if row["passes_strict_side_specific_rejection_gate"]]
    ranked = sorted(
        complete,
        key=lambda row: (
            Decimal(str(row["rejection_proxy_sum_pUSD"])),
            str(row["family"]),
            str(row["base_slug"]),
        ),
    )
    strict_ranked = [row for row in ranked if row["passes_strict_side_specific_rejection_gate"]]
    result: dict[str, Any] = {
        "schema_version": "polymarket-soccer-exact-score-cross-family-result-v1",
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
        "strict_sub_floor_relations": strict_ranked[:20],
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
    write_bytes_atomic(output_path, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
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
