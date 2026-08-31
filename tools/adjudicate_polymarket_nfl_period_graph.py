from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ONE = Decimal("1")
HALF = Decimal("0.5")


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


def _market_gate(market: dict[str, Any], outcomes: list[str]) -> None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError(f"inactive market entered graph: {market.get('id')}")
    if _pair(market.get("outcomes"), "outcomes") != outcomes:
        raise RuntimeError(f"outcome mismatch: {market.get('id')}")
    prices = _pair(market.get("outcomePrices"), "outcomePrices")
    if any(not (Decimal("0") <= Decimal(value) <= ONE) for value in prices):
        raise RuntimeError(f"diagnostic price out of range: {market.get('id')}")
    best_ask = Decimal(str(market.get("bestAsk")))
    best_bid = Decimal(str(market.get("bestBid")))
    if not (Decimal("0") <= best_bid <= best_ask <= ONE):
        raise RuntimeError(f"side-specific quote out of range: {market.get('id')}")
    if "resolve 50-50" not in str(market.get("description") or ""):
        raise RuntimeError(f"cancellation rule missing: {market.get('id')}")


def _threshold(match: re.Match[str], group: int) -> int:
    return int(match.group(group)) + 1


def _inventory(markets: list[dict[str, Any]]) -> dict[str, Any]:
    full: list[tuple[int, dict[str, Any]]] = []
    team_full = {team: [] for team in ("Patriots", "Seahawks")}
    period = {scope: [] for scope in ("1Q", "2Q", "3Q", "4Q", "1H", "2H")}
    team_period = {
        (team, scope): []
        for team in ("Patriots", "Seahawks")
        for scope in ("1H", "2H")
    }
    spread = {scope: [] for scope in ("1Q", "2Q", "3Q", "4Q", "2H")}
    for market in markets:
        question = str(market.get("question") or "")
        description = str(market.get("description") or "")
        match = re.fullmatch(r"Patriots vs\. Seahawks: O/U ([0-9]+)\.5", question)
        if match:
            threshold = _threshold(match, 1)
            _market_gate(market, ["Over", "Under"])
            if f"combine to score {threshold} or more points in this game" not in description:
                raise RuntimeError(f"full-game total rule mismatch: {market['id']}")
            full.append((threshold, market))
            continue
        match = re.fullmatch(
            r"(Patriots|Seahawks) Team Total: O/U ([0-9]+)\.5", question
        )
        if match:
            team, threshold = match.group(1), _threshold(match, 2)
            _market_gate(market, ["Over", "Under"])
            if f"{team} score {threshold} or more points in this game" not in description:
                raise RuntimeError(f"team total rule mismatch: {market['id']}")
            team_full[team].append((threshold, market))
            continue
        match = re.fullmatch(
            r"Patriots vs\. Seahawks: (1Q|2Q|3Q|4Q|1H|2H) O/U ([0-9]+)\.5",
            question,
        )
        if match:
            scope, threshold = match.group(1), _threshold(match, 2)
            _market_gate(market, ["Over", "Under"])
            if f"score {threshold} or more points" not in description:
                raise RuntimeError(f"period total rule mismatch: {market['id']}")
            if scope == "1H":
                if not (
                    "score at halftime only" in description
                    or "overtime is not included" in description
                ):
                    raise RuntimeError(f"first-half scope mismatch: {market['id']}")
            elif "overtime is not included" not in description:
                raise RuntimeError(f"period overtime rule mismatch: {market['id']}")
            period[scope].append((threshold, market))
            continue
        match = re.fullmatch(
            r"(Patriots|Seahawks) (1H|2H) Team Total: O/U ([0-9]+)\.5",
            question,
        )
        if match:
            team, scope, threshold = (
                match.group(1),
                match.group(2),
                _threshold(match, 3),
            )
            _market_gate(market, ["Over", "Under"])
            if not (
                f"{team} score {threshold} or more points" in description
                and "overtime is not included" in description
            ):
                raise RuntimeError(f"team-period rule mismatch: {market['id']}")
            team_period[(team, scope)].append((threshold, market))
            continue
        match = re.fullmatch(
            r"(1Q|2Q|3Q|4Q|2H) Spread: Seahawks \(-([0-9]+)\.5\)",
            question,
        )
        if match:
            scope, threshold = match.group(1), _threshold(match, 2)
            _market_gate(market, ["Seahawks", "Patriots"])
            if not (
                f"Seahawks outscore the Patriots by {threshold} or more points"
                in description
                and "overtime is not included" in description
            ):
                raise RuntimeError(f"period spread rule mismatch: {market['id']}")
            spread[scope].append((threshold, market))
    groups = [full, *team_full.values(), *period.values(), *team_period.values(), *spread.values()]
    for group in groups:
        thresholds = [threshold for threshold, _ in group]
        if len(thresholds) != len(set(thresholds)):
            raise RuntimeError("duplicate logical threshold")
        group.sort(key=lambda row: row[0])
    return {
        "full": full,
        "team_full": team_full,
        "period": period,
        "team_period": team_period,
        "spread": spread,
    }


def _side_price(market: dict[str, Any], side: int) -> tuple[Decimal, str]:
    if side == 0:
        return Decimal(str(market["bestAsk"])), "bestAsk"
    return ONE - Decimal(str(market["bestBid"])), "1-bestBid"


def _diagnostic_price(market: dict[str, Any], side: int) -> Decimal:
    return Decimal(_pair(market["outcomePrices"], "outcomePrices")[side])


def _relation(
    family: str,
    legs: list[tuple[dict[str, Any], int]],
    thresholds: list[object],
) -> dict[str, Any]:
    priced_legs: list[dict[str, str]] = []
    side_sum = Decimal("0")
    diagnostic_sum = Decimal("0")
    for market, side in legs:
        outcomes = _pair(market["outcomes"], "outcomes")
        side_price, source = _side_price(market, side)
        side_sum += side_price
        diagnostic_sum += _diagnostic_price(market, side)
        priced_legs.append(
            {
                "market_id": str(market["id"]),
                "outcome": outcomes[side],
                "price_pUSD": str(side_price),
                "price_source": source,
                "question": str(market["question"]),
            }
        )
    return {
        "family": family,
        "guaranteed_floor_pUSD": "1",
        "legs": priced_legs,
        "outcome_prices_diagnostic_sum_pUSD": str(diagnostic_sum),
        "side_specific_rejection_sum_pUSD": str(side_sum),
        "strict_side_specific_subfloor": side_sum < ONE,
        "thresholds": thresholds,
    }


def _ladders(
    family: str, groups: list[tuple[str, list[tuple[int, dict[str, Any]]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, group in groups:
        for (lower, lower_market), (higher, higher_market) in itertools.combinations(
            group, 2
        ):
            rows.append(
                _relation(
                    family,
                    [(lower_market, 0), (higher_market, 1)],
                    [scope, lower, higher],
                )
            )
    return rows


def _additive(
    family: str,
    left: list[tuple[int, dict[str, Any]]],
    right: list[tuple[int, dict[str, Any]]],
    combined: list[tuple[int, dict[str, Any]]],
    scope: list[object],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_threshold, left_market in left:
        for right_threshold, right_market in right:
            for combined_threshold, combined_market in combined:
                if combined_threshold <= left_threshold + right_threshold:
                    rows.append(
                        _relation(
                            family,
                            [(left_market, 1), (right_market, 1), (combined_market, 0)],
                            [
                                *scope,
                                left_threshold,
                                right_threshold,
                                combined_threshold,
                            ],
                        )
                    )
    return rows


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            Decimal(row["side_specific_rejection_sum_pUSD"]),
            row["thresholds"],
            [leg["market_id"] for leg in row["legs"]],
        ),
    )
    diagnostic_best = min(
        rows,
        key=lambda row: (
            Decimal(row["outcome_prices_diagnostic_sum_pUSD"]),
            row["thresholds"],
        ),
    )
    canonical = sorted(
        rows,
        key=lambda row: (
            row["family"],
            row["thresholds"],
            [leg["market_id"] for leg in row["legs"]],
        ),
    )
    return {
        "best_side_specific_relation": ordered[0],
        "diagnostic_best_sum_pUSD": diagnostic_best[
            "outcome_prices_diagnostic_sum_pUSD"
        ],
        "relation_count": len(rows),
        "relations_sha256": _canonical_hash(canonical),
        "strict_diagnostic_subfloor_count": sum(
            Decimal(row["outcome_prices_diagnostic_sum_pUSD"]) < ONE for row in rows
        ),
        "strict_side_specific_subfloor_count": sum(
            row["strict_side_specific_subfloor"] for row in rows
        ),
    }


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
    if (
        metadata["capture"]["receipt"]["response_sha256"]
        != contract["source"]["raw_file_sha256"]
    ):
        raise RuntimeError("raw receipt binding mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or not isinstance(raw.get("markets"), list):
        raise RuntimeError("raw source is not one exact Gamma event")
    inventory = _inventory(raw["markets"])
    full = inventory["full"]
    team_full = inventory["team_full"]
    period = inventory["period"]
    team_period = inventory["team_period"]
    spread = inventory["spread"]

    families: dict[str, list[dict[str, Any]]] = {}
    families["full_team_total_ladders"] = _ladders(
        "full_team_total_ladders",
        [(team, team_full[team]) for team in ("Patriots", "Seahawks")],
    )
    families["full_team_additive_covers"] = _additive(
        "full_team_additive_covers",
        team_full["Patriots"],
        team_full["Seahawks"],
        full,
        ["full_game"],
    )
    families["period_ladders"] = _ladders(
        "period_ladders",
        [(scope, period[scope]) for scope in period]
        + [
            (f"{team}_{scope}", team_period[(team, scope)])
            for team in ("Patriots", "Seahawks")
            for scope in ("1H", "2H")
        ]
        + [(f"spread_{scope}", spread[scope]) for scope in spread],
    )
    families["half_team_additive_covers"] = [
        row
        for scope in ("1H", "2H")
        for row in _additive(
            "half_team_additive_covers",
            team_period[("Patriots", scope)],
            team_period[("Seahawks", scope)],
            period[scope],
            [scope],
        )
    ]
    families["halves_to_full_covers"] = _additive(
        "halves_to_full_covers", period["1H"], period["2H"], full, ["game"]
    )
    families["team_halves_to_full_covers"] = [
        row
        for team in ("Patriots", "Seahawks")
        for row in _additive(
            "team_halves_to_full_covers",
            team_period[(team, "1H")],
            team_period[(team, "2H")],
            team_full[team],
            [team],
        )
    ]
    families["quarters_to_halves_covers"] = [
        row
        for half, first, second in (("1H", "1Q", "2Q"), ("2H", "3Q", "4Q"))
        for row in _additive(
            "quarters_to_halves_covers",
            period[first],
            period[second],
            period[half],
            [half],
        )
    ]
    quarter_rows: list[dict[str, Any]] = []
    for quarters in itertools.product(
        period["1Q"], period["2Q"], period["3Q"], period["4Q"]
    ):
        threshold_sum = sum(threshold for threshold, _ in quarters)
        for full_threshold, full_market in full:
            if full_threshold <= threshold_sum:
                quarter_rows.append(
                    _relation(
                        "quarters_to_full_covers",
                        [(market, 1) for _, market in quarters] + [(full_market, 0)],
                        [*(threshold for threshold, _ in quarters), full_threshold],
                    )
                )
    families["quarters_to_full_covers"] = quarter_rows
    summaries = {name: _family_summary(rows) for name, rows in families.items()}
    all_rows = [row for rows in families.values() for row in rows]
    total_candidates = sum(row["strict_side_specific_subfloor"] for row in all_rows)
    result: dict[str, Any] = {
        "schema_version": "polymarket-nfl-team-and-period-structural-graph-correction-v2",
        "created_at_utc": contract["frozen_at_utc"],
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source_binding": contract["source"],
        "supersession": contract["supersession"],
        "proof_contract": contract["proof_contract"],
        "pricing_contract": contract["pricing_contract"],
        "inventory": {
            "full_total_count": len(full),
            "full_team_total_count": {team: len(rows) for team, rows in team_full.items()},
            "period_total_count": {scope: len(rows) for scope, rows in period.items()},
            "team_period_total_count": {
                f"{team}_{scope}": len(team_period[(team, scope)])
                for team in ("Patriots", "Seahawks")
                for scope in ("1H", "2H")
            },
            "period_spread_count": {scope: len(rows) for scope, rows in spread.items()},
        },
        "family_screens": summaries,
        "aggregate_screen": {
            "relation_count": len(all_rows),
            "relations_sha256": _canonical_hash(
                sorted(
                    all_rows,
                    key=lambda row: (
                        row["family"],
                        row["thresholds"],
                        [leg["market_id"] for leg in row["legs"]],
                    ),
                )
            ),
            "strict_diagnostic_subfloor_count": sum(
                Decimal(row["outcome_prices_diagnostic_sum_pUSD"]) < ONE
                for row in all_rows
            ),
            "strict_side_specific_subfloor_count": total_candidates,
        },
        "adjudication": {
            "accepted_edge": False,
            "book_or_fee_request_permitted": total_candidates > 0,
            "deployment_ready": False,
            "market_direction_forecast_required": False,
            "next_action": (
                "freeze_one_exact_book_batch_for_the_precommitted_best_candidate"
                if total_candidates
                else "do_not_rebuild_reprice_or_request_books_for_this_retained_graph"
            ),
            "profitability_claim": False,
            "status": "terminal_corrected_retained_graph_rejected_before_books_and_fees",
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct and exhaust one retained NFL team/period payoff graph."
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("output already exists")
    result = adjudicate(contract)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "candidate_count": result["aggregate_screen"][
                    "strict_side_specific_subfloor_count"
                ],
                "network_requests": 0,
                "relation_count": result["aggregate_screen"]["relation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
