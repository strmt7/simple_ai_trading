from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ONE = Decimal("1")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: object) -> str:
    return _sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _pair(value: object, label: str) -> list[str]:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a JSON string")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise RuntimeError(f"{label} must contain exactly two values")
    return [str(item) for item in parsed]


def _threshold(market: dict[str, Any]) -> int:
    line = Decimal(str(market.get("line")))
    if line % 1 != Decimal("0.5"):
        raise RuntimeError(f"non-half-point total entered graph: {market.get('id')}")
    return int(line) + 1


def _gate_total(market: dict[str, Any], period: bool) -> None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError(f"inactive market entered graph: {market.get('id')}")
    if _pair(market.get("outcomes"), "outcomes") != ["Over", "Under"]:
        raise RuntimeError(f"outcome mismatch: {market.get('id')}")
    threshold = _threshold(market)
    description = str(market.get("description") or "")
    if f"score {threshold} or more points" not in description:
        raise RuntimeError(f"threshold rule mismatch: {market.get('id')}")
    if "resolve 50-50" not in description:
        raise RuntimeError(f"cancellation rule missing: {market.get('id')}")
    if period and "overtime is not included" not in description:
        raise RuntimeError(f"period overtime rule mismatch: {market.get('id')}")
    prices = _pair(market.get("outcomePrices"), "outcomePrices")
    if any(not (Decimal("0") <= Decimal(value) <= ONE) for value in prices):
        raise RuntimeError(f"diagnostic price out of range: {market.get('id')}")


def _team(question: str, teams: tuple[str, str]) -> str:
    matches = [team for team in teams if question.startswith(f"{team} ")]
    if len(matches) != 1:
        raise RuntimeError(f"team total cannot be assigned: {question}")
    return matches[0]


def _inventory(event: dict[str, Any]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    aliases = tuple(str(row["alias"]) for row in event.get("teams", []))
    if len(aliases) != 2 or len(set(aliases)) != 2:
        raise RuntimeError(f"event does not have two unique teams: {event.get('id')}")
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    mappings = {
        "totals": ("full", False),
        "first_half_totals": ("1H", True),
        "second_half_totals": ("2H", True),
        "q1_totals": ("1Q", True),
        "q2_totals": ("2Q", True),
        "q3_totals": ("3Q", True),
        "q4_totals": ("4Q", True),
    }
    team_mappings = {
        "team_totals": ("team_full", False),
        "first_half_team_totals": ("team_1H", True),
        "second_half_team_totals": ("team_2H", True),
    }
    for market in event.get("markets", []):
        market_type = str(market.get("sportsMarketType") or "")
        if market_type in mappings:
            scope, period = mappings[market_type]
            _gate_total(market, period)
            groups[scope].append((_threshold(market), market))
        elif market_type in team_mappings:
            scope, period = team_mappings[market_type]
            _gate_total(market, period)
            team = _team(str(market.get("question") or ""), aliases)
            groups[f"{scope}:{team}"].append((_threshold(market), market))
    for name, rows in groups.items():
        thresholds = [threshold for threshold, _ in rows]
        if len(thresholds) != len(set(thresholds)):
            raise RuntimeError(
                f"duplicate logical threshold in {event.get('id')} {name}"
            )
        rows.sort(key=lambda row: row[0])
    groups["__teams__"] = [(0, {"alias": alias}) for alias in aliases]
    return groups


def _side_price(market: dict[str, Any], side: int) -> tuple[Decimal | None, str]:
    field = "bestAsk" if side == 0 else "bestBid"
    value = market.get(field)
    if value is None or value == "":
        return None, field
    quote = Decimal(str(value))
    price = quote if side == 0 else ONE - quote
    if not Decimal("0") <= price <= ONE:
        raise RuntimeError(f"side-specific price out of range: {market.get('id')}")
    return price, "bestAsk" if side == 0 else "1-bestBid"


def _relation(
    event: dict[str, Any],
    family: str,
    legs: list[tuple[dict[str, Any], int]],
    thresholds: list[object],
) -> dict[str, Any]:
    priced_legs: list[dict[str, Any]] = []
    side_sum = Decimal("0")
    diagnostic_sum = Decimal("0")
    complete = True
    for market, side in legs:
        outcomes = _pair(market["outcomes"], "outcomes")
        price, source = _side_price(market, side)
        if price is None:
            complete = False
        else:
            side_sum += price
        diagnostic_sum += Decimal(_pair(market["outcomePrices"], "outcomePrices")[side])
        priced_legs.append(
            {
                "market_id": str(market["id"]),
                "outcome": outcomes[side],
                "price_pUSD": None if price is None else str(price),
                "price_source": source,
                "question": str(market["question"]),
            }
        )
    return {
        "event_id": str(event["id"]),
        "event_slug": str(event["slug"]),
        "event_title": str(event["title"]),
        "family": family,
        "guaranteed_floor_pUSD": "1",
        "legs": priced_legs,
        "outcome_prices_diagnostic_sum_pUSD": str(diagnostic_sum),
        "price_complete": complete,
        "side_specific_rejection_sum_pUSD": str(side_sum) if complete else None,
        "strict_side_specific_subfloor": complete and side_sum < ONE,
        "thresholds": thresholds,
    }


def _ladders(
    event: dict[str, Any], family: str, groups: Iterable[tuple[str, list]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, group in groups:
        for (lower, lower_market), (higher, higher_market) in itertools.combinations(
            group, 2
        ):
            rows.append(
                _relation(
                    event,
                    family,
                    [(lower_market, 0), (higher_market, 1)],
                    [scope, lower, higher],
                )
            )
    return rows


def _additive(
    event: dict[str, Any],
    family: str,
    left: list,
    right: list,
    combined: list,
    scope: list[object],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_threshold, left_market in left:
        for right_threshold, right_market in right:
            for combined_threshold, combined_market in combined:
                if combined_threshold <= left_threshold + right_threshold:
                    rows.append(
                        _relation(
                            event,
                            family,
                            [(left_market, 1), (right_market, 1), (combined_market, 0)],
                            [*scope, left_threshold, right_threshold, combined_threshold],
                        )
                    )
    return rows


def _event_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    groups = _inventory(event)
    teams = [row[1]["alias"] for row in groups.pop("__teams__")]
    rows = _ladders(
        event,
        "total_ladders",
        [(scope, group) for scope, group in sorted(groups.items())],
    )
    rows += _additive(
        event,
        "full_team_additive_covers",
        groups.get(f"team_full:{teams[0]}", []),
        groups.get(f"team_full:{teams[1]}", []),
        groups.get("full", []),
        ["full_game"],
    )
    for half in ("1H", "2H"):
        rows += _additive(
            event,
            "half_team_additive_covers",
            groups.get(f"team_{half}:{teams[0]}", []),
            groups.get(f"team_{half}:{teams[1]}", []),
            groups.get(half, []),
            [half],
        )
    rows += _additive(
        event,
        "halves_to_full_covers",
        groups.get("1H", []),
        groups.get("2H", []),
        groups.get("full", []),
        ["game"],
    )
    for team in teams:
        rows += _additive(
            event,
            "team_halves_to_full_covers",
            groups.get(f"team_1H:{team}", []),
            groups.get(f"team_2H:{team}", []),
            groups.get(f"team_full:{team}", []),
            [team],
        )
    for half, first, second in (("1H", "1Q", "2Q"), ("2H", "3Q", "4Q")):
        rows += _additive(
            event,
            "quarters_to_halves_covers",
            groups.get(first, []),
            groups.get(second, []),
            groups.get(half, []),
            [half],
        )
    quarter_groups = [groups.get(scope, []) for scope in ("1Q", "2Q", "3Q", "4Q")]
    for quarters in itertools.product(*quarter_groups):
        threshold_sum = sum(threshold for threshold, _ in quarters)
        for full_threshold, full_market in groups.get("full", []):
            if full_threshold <= threshold_sum:
                rows.append(
                    _relation(
                        event,
                        "quarters_to_full_covers",
                        [(market, 1) for _, market in quarters] + [(full_market, 0)],
                        [*(threshold for threshold, _ in quarters), full_threshold],
                    )
                )
    return rows


def _row_key(row: dict[str, Any]) -> tuple:
    return (
        row["family"],
        row["event_id"],
        row["thresholds"],
        [leg["market_id"] for leg in row["legs"]],
    )


def _priced_key(row: dict[str, Any]) -> tuple:
    return (Decimal(row["side_specific_rejection_sum_pUSD"]), _row_key(row))


def adjudicate(contract: dict[str, Any]) -> dict[str, Any]:
    body = dict(contract)
    expected_contract_hash = body.pop("contract_sha256", None)
    if _canonical_hash(body) != expected_contract_hash:
        raise RuntimeError("contract hash mismatch")
    raw_path = _root_path(str(contract["source"]["raw_path"]))
    metadata_path = _root_path(str(contract["source"]["metadata_path"]))
    raw_bytes = raw_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    if _sha256(raw_bytes) != contract["source"]["raw_file_sha256"]:
        raise RuntimeError("raw source hash mismatch")
    if _sha256(metadata_bytes) != contract["source"]["metadata_file_sha256"]:
        raise RuntimeError("metadata file hash mismatch")
    metadata = json.loads(metadata_bytes)
    if metadata["result_sha256"] != contract["source"]["metadata_result_sha256"]:
        raise RuntimeError("metadata result binding mismatch")
    if metadata["capture"]["receipt"]["response_sha256"] != _sha256(raw_bytes):
        raise RuntimeError("raw receipt binding mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    payload = json.loads(raw_bytes)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or len(events) != contract["population"]["event_count"]:
        raise RuntimeError("catalog population mismatch")
    if sorted(str(event["id"]) for event in events) != contract["population"]["event_ids"]:
        raise RuntimeError("catalog event ids mismatch")
    rows = [row for event in events for row in _event_rows(event)]
    canonical = sorted(rows, key=_row_key)
    complete = [row for row in rows if row["price_complete"]]
    incomplete = [row for row in rows if not row["price_complete"]]
    candidates = [row for row in complete if row["strict_side_specific_subfloor"]]
    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        family_complete = [row for row in family_rows if row["price_complete"]]
        family_candidates = [row for row in family_complete if row["strict_side_specific_subfloor"]]
        families[family] = {
            "best_side_specific_relation": min(family_complete, key=_priced_key),
            "price_complete_relation_count": len(family_complete),
            "price_incomplete_relation_count": len(family_rows) - len(family_complete),
            "relation_count": len(family_rows),
            "strict_side_specific_subfloor_count": len(family_candidates),
        }
    population_complete = len(incomplete) == 0
    result: dict[str, Any] = {
        "schema_version": "polymarket-nfl-catalog-team-period-graph-result-v1",
        "created_at_utc": contract["frozen_at_utc"],
        "contract": {"path": contract["contract_path"], "sha256": contract["contract_sha256"]},
        "source_binding": contract["source"],
        "proof_contract": contract["proof_contract"],
        "pricing_contract": contract["pricing_contract"],
        "population": contract["population"],
        "family_screens": families,
        "aggregate_screen": {
            "best_complete_relation": min(complete, key=_priced_key),
            "price_complete_relation_count": len(complete),
            "price_incomplete_relation_count": len(incomplete),
            "relation_count": len(rows),
            "relations_sha256": _canonical_hash(canonical),
            "strict_side_specific_subfloor_count": len(candidates),
        },
        "adjudication": {
            "accepted_edge": False,
            "book_or_fee_request_permitted": population_complete and bool(candidates),
            "deployment_ready": False,
            "market_direction_forecast_required": False,
            "next_action": (
                "freeze_one_exact_book_batch_for_the_precommitted_best_candidate"
                if population_complete and candidates
                else "do_not_request_books_or_fees_for_this_retained_population"
            ),
            "profitability_claim": False,
            "status": (
                "candidate_requires_separately_frozen_exact_depth_screen"
                if population_complete and candidates
                else "retained_population_price_incomplete_no_escalation"
                if incomplete
                else "terminal_retained_population_rejected_before_books_and_fees"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaust one retained NFL catalog team and period payoff graph."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--contract")
    modes.add_argument(
        "--hash-template",
        help="Print the canonical hash of a frozen template without contract_sha256.",
    )
    args = parser.parse_args()
    if args.hash_template:
        template = json.loads(_root_path(args.hash_template).read_text(encoding="utf-8"))
        if "contract_sha256" in template:
            raise RuntimeError("hash template already contains contract_sha256")
        print(_canonical_hash(template))
        return
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("output already exists")
    result = adjudicate(contract)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "candidate_count": result["aggregate_screen"]["strict_side_specific_subfloor_count"],
                "incomplete_count": result["aggregate_screen"]["price_incomplete_relation_count"],
                "network_requests": 0,
                "relation_count": result["aggregate_screen"]["relation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
