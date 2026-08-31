"""Exhaust retained soccer goalscorer-to-total payoff implications."""

from __future__ import annotations

import argparse
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
    if contract.get("schema_version") != "polymarket-soccer-goalscorer-total-contract-v1":
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


def _require_player_rules(market: Mapping[str, Any]) -> None:
    description = _description(market)
    required = (
        'resolve to "Yes" if',
        "is credited with a goal in the official box score",
        "first 90 minutes of regular play plus stoppage time",
        'canceled entirely, with no make-up game, this market will resolve "50-50"',
        'listed as inactive or otherwise does not play, the market will resolve "No"',
    )
    if market.get("sportsMarketType") != "soccer_anytime_goalscorer" or not all(
        phrase in description for phrase in required
    ):
        raise ValueError(f"goalscorer rule changed: {market.get('id')}")


def _require_total_rules(market: Mapping[str, Any]) -> None:
    description = _description(market)
    required = (
        'resolve to "Over" if',
        "combine to score 1 or more goals",
        "first 90 minutes of regular play plus stoppage time",
        "canceled entirely, with no make-up game, this market will resolve 50",
    )
    if (
        market.get("sportsMarketType") != "totals"
        or Decimal(str(market.get("line"))) != Decimal("0.5")
        or not all(phrase in description for phrase in required)
    ):
        raise ValueError(f"Over 0.5 rule changed: {market.get('id')}")


def _event_markets(event: Mapping[str, Any], *, name: str) -> list[dict[str, Any]]:
    values = event.get("markets")
    if not isinstance(values, list):
        raise ValueError(f"{name} markets must be a list")
    return [_mapping(value, name=f"{name} market") for value in values]


def _enumerate(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("retained events must be a list")
    by_slug = {str(event["slug"]): _mapping(event, name="event") for event in events}
    relations: list[dict[str, Any]] = []
    complete_base_count = 0

    for player_slug, player_event in sorted(by_slug.items()):
        if not player_slug.endswith("-player-props"):
            continue
        base_slug = player_slug.removesuffix("-player-props")
        more_event = by_slug.get(f"{base_slug}-more-markets")
        if more_event is None:
            continue
        player_markets = _event_markets(player_event, name="player-props")
        total_markets = [
            market
            for market in _event_markets(more_event, name="more")
            if market.get("sportsMarketType") == "totals"
            and Decimal(str(market.get("line"))) == Decimal("0.5")
        ]
        if len(total_markets) != 1 or not player_markets:
            raise ValueError(f"required goalscorer/Over 0.5 population changed: {base_slug}")
        total_market = total_markets[0]
        _require_total_rules(total_market)
        complete_base_count += 1
        for player_market in player_markets:
            _require_player_rules(player_market)
            relations.append(
                graph._relation(
                    family="anytime_goalscorer_implies_over_zero_point_five",
                    base_slug=base_slug,
                    first_market=player_market,
                    first_side="second_outcome",
                    second_market=total_market,
                    second_side="first_outcome",
                    proof=(
                        "NO anytime goalscorer plus YES Over 0.5 pays at least one "
                        "because any credited player goal makes the match total positive"
                    ),
                    cancellation_floor_pUSD="1",
                )
            )

    identities = {
        tuple((leg["market_id"], leg["outcome"]) for leg in relation["legs"])
        for relation in relations
    }
    if len(identities) != len(relations):
        raise ValueError("duplicate goalscorer-total relation emitted")
    return relations, complete_base_count


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
    relations, complete_base_count = _enumerate(payload)
    complete = [row for row in relations if row["side_specific_prices_complete"]]
    ranked = sorted(
        complete,
        key=lambda row: (
            Decimal(str(row["rejection_proxy_sum_pUSD"])),
            str(row["base_slug"]),
            str(row["legs"][0]["market_id"]),
        ),
    )
    strict = [row for row in ranked if row["passes_strict_side_specific_rejection_gate"]]
    result: dict[str, Any] = {
        "schema_version": "polymarket-soccer-goalscorer-total-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": args.contract,
            "sha256": contract["contract_sha256"],
        },
        "retained_source": source,
        "population": {
            "complete_base_count": complete_base_count,
            "relation_count": len(relations),
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
