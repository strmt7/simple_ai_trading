"""Enumerate retained soccer half-result/full-result logical superhedges."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("home", "draw", "away")
CONJUNCTIONS = (
    ("home", "home", "home"),
    ("home", "draw", "home"),
    ("draw", "home", "home"),
    ("away", "away", "away"),
    ("away", "draw", "away"),
    ("draw", "away", "away"),
    ("draw", "draw", "draw"),
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical(body))


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    path.relative_to(ROOT)
    return path


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _description(market: Mapping[str, Any]) -> str:
    return " ".join(str(market.get("description") or "").split())


def _load_contract(path: Path) -> dict[str, Any]:
    contract = _mapping(json.loads(path.read_bytes()), name="contract")
    if contract.get("schema_version") != (
        "polymarket-soccer-half-result-superhedge-contract-v1"
    ):
        raise ValueError("unexpected contract schema")
    if _path(contract["contract_path"]) != path.resolve():
        raise ValueError("contract path mismatch")
    if _self_hash(contract, field="contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("invalid or future contract time")
    tool = _path(contract["implementation"]["path"])
    if _sha256(tool.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    return contract


def _load_bound_json(source: Mapping[str, object]) -> dict[str, Any]:
    raw = _path(source["path"]).read_bytes()
    if _sha256(raw) != source["file_sha256"]:
        raise ValueError(f"retained source hash mismatch: {source['path']}")
    payload = _mapping(json.loads(raw), name="retained source")
    expected_result = source.get("result_sha256")
    if expected_result is not None and (
        payload.get("result_sha256") != expected_result
        or _self_hash(payload, field="result_sha256") != expected_result
    ):
        raise ValueError(f"retained result self-hash mismatch: {source['path']}")
    return payload


def _market_map(event: Mapping[str, Any], *, main: bool) -> dict[str, Any]:
    title = str(event["title"])
    if main:
        parts = title.split(" vs. ", 1)
    else:
        parts = title.rsplit(" - ", 1)[0].split(" vs. ", 1)
    if len(parts) != 2:
        raise ValueError(f"unexpected soccer title: {title}")
    home, away = parts
    values = event.get("markets")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"unexpected market cardinality: {event.get('slug')}")
    result: dict[str, Any] = {}
    for value in values:
        market = _mapping(value, name="market")
        label = str(market.get("groupItemTitle"))
        role = (
            "home"
            if label == home
            else "away"
            if label == away
            else "draw"
            if label == "Draw" or label == f"Draw ({home} vs. {away})"
            else ""
        )
        if not role or role in result:
            raise ValueError(f"unexpected or duplicate role label: {label}")
        if not (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("acceptingOrders") is True
        ):
            raise ValueError(f"market was not active and accepting: {market.get('id')}")
        outcomes = json.loads(market["outcomes"])
        tokens = json.loads(market["clobTokenIds"])
        if outcomes != ["Yes", "No"] or len(tokens) != 2:
            raise ValueError(f"binary identity changed: {market.get('id')}")
        result[role] = market
    if set(result) != set(ROLES):
        raise ValueError(f"role population changed: {event.get('slug')}")
    return result


def _validate_rules(
    main: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> None:
    for role in ROLES:
        main_text = _description(main[role])
        first_text = _description(first[role])
        second_text = _description(second[role])
        if not (
            "90 minutes of regular play plus stoppage time" in main_text
            and "first 45 minutes of regular play plus stoppage time" in first_text
            and "second half of regular play plus second-half stoppage time"
            in second_text
            and "If no acceptable data is available within 48 hours"
            in first_text
            and "resolve 50-50" in first_text
            and "If no acceptable data is available within 48 hours"
            in second_text
            and "resolve 50-50" in second_text
        ):
            raise ValueError(f"time-scope or fallback rule changed for {role}")
        if role == "draw":
            if not (
                'canceled entirely, with no make-up game, this market will resolve to "Yes"'
                in main_text
                and 'canceled entirely, with no make-up game, this market will resolve to "Yes"'
                in first_text
                and 'canceled entirely, with no make-up game, this market will resolve to "Draw"'
                in second_text
            ):
                raise ValueError("draw cancellation rule changed")
        elif not (
            'canceled entirely, with no make-up game, this market will resolve "No"'
            in main_text
            and 'canceled entirely, with no make-up game, this market will resolve "No"'
            in first_text
            and 'canceled entirely, with no make-up game, this market will resolve to "Draw"'
            in second_text
        ):
            raise ValueError(f"team cancellation rule changed for {role}")


def _leg(market: Mapping[str, Any], *, outcome: str) -> tuple[dict[str, str], Decimal]:
    tokens = json.loads(market["clobTokenIds"])
    if outcome == "YES":
        price = Decimal(str(market["bestAsk"]))
        token = str(tokens[0])
        kind = "bestAsk"
    else:
        price = Decimal("1") - Decimal(str(market["bestBid"]))
        token = str(tokens[1])
        kind = "one_minus_YES_bestBid"
    return (
        {
            "market_id": str(market["id"]),
            "condition_id": str(market["conditionId"]),
            "question": str(market["question"]),
            "outcome": outcome,
            "token_id": token,
            "price_kind": kind,
            "rejection_price_pUSD": _decimal_text(price),
        },
        price,
    )


def _relation(
    *,
    family: str,
    base_slug: str,
    proof: str,
    markets: tuple[tuple[Mapping[str, Any], str], ...],
) -> dict[str, Any]:
    legs_and_prices = [_leg(market, outcome=outcome) for market, outcome in markets]
    price = sum((value for _, value in legs_and_prices), Decimal())
    return {
        "family": family,
        "base_slug": base_slug,
        "payoff_proof": proof,
        "ordinary_settlement_floor_pUSD": "1",
        "cancellation_floor_pUSD": "1",
        "half_result_no_data_fallback_floor_pUSD": "1",
        "legs": [leg for leg, _ in legs_and_prices],
        "rejection_proxy_sum_pUSD": _decimal_text(price),
        "optimistic_proxy_headroom_pUSD": _decimal_text(Decimal("1") - price),
        "passes_strict_side_specific_rejection_gate": price < 1,
    }


def _enumerate(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("retained events list missing")
    by_slug = {str(value["slug"]): _mapping(value, name="event") for value in events}
    relations: list[dict[str, Any]] = []
    base_count = 0
    for base_slug in sorted(by_slug):
        first_event = by_slug.get(f"{base_slug}-halftime-result")
        second_event = by_slug.get(f"{base_slug}-second-half-result")
        if first_event is None or second_event is None:
            continue
        base_count += 1
        main = _market_map(by_slug[base_slug], main=True)
        first = _market_map(first_event, main=False)
        second = _market_map(second_event, main=False)
        _validate_rules(main, first, second)
        for first_role, second_role, full_role in CONJUNCTIONS:
            relations.append(
                _relation(
                    family="half_result_conjunction_implies_full_game_result",
                    base_slug=base_slug,
                    proof=(
                        f"first_half_{first_role}_and_second_half_{second_role}"
                        f"_implies_full_game_{full_role}"
                    ),
                    markets=(
                        (first[first_role], "NO"),
                        (second[second_role], "NO"),
                        (main[full_role], "YES"),
                    ),
                )
            )
        for role in ("home", "away"):
            relations.append(
                _relation(
                    family="full_game_team_win_implies_at_least_one_half_win",
                    base_slug=base_slug,
                    proof=f"full_game_{role}_win_implies_first_or_second_half_{role}_win",
                    markets=(
                        (main[role], "NO"),
                        (first[role], "YES"),
                        (second[role], "YES"),
                    ),
                )
            )
    relations.sort(
        key=lambda row: (
            Decimal(row["rejection_proxy_sum_pUSD"]),
            row["family"],
            row["base_slug"],
            row["payoff_proof"],
        )
    )
    return relations, base_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _path(args.contract)
    contract = _load_contract(contract_path)
    raw_source = _mapping(contract["retained_raw_source"], name="raw source")
    raw = _load_bound_json(raw_source)
    parent_source = _mapping(contract["retained_parent_result"], name="parent source")
    _load_bound_json(parent_source)
    relations, base_count = _enumerate(raw)
    candidates = [
        row for row in relations if row["passes_strict_side_specific_rejection_gate"]
    ]
    output = _path(contract["output_path"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    result: dict[str, Any] = {
        "schema_version": "polymarket-soccer-half-result-superhedge-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {"path": args.contract, "sha256": contract["contract_sha256"]},
        "retained_sources": {
            "raw": raw_source,
            "parent_result": parent_source,
        },
        "population": {
            "complete_base_halftime_second_half_triple_count": base_count,
            "tested_relation_count": len(relations),
            "conjunction_relation_count": sum(
                row["family"] == "half_result_conjunction_implies_full_game_result"
                for row in relations
            ),
            "full_win_union_relation_count": sum(
                row["family"] == "full_game_team_win_implies_at_least_one_half_win"
                for row in relations
            ),
            "strict_side_specific_candidate_count": len(candidates),
        },
        "screen": {
            "best_relation": relations[0] if relations else None,
            "relations": relations,
            "price_gate": "YES bestAsk and NO conservative proxy one minus YES bestBid; Gamma is rejection-only",
        },
        "adjudication": {
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "market_direction_forecast_required": False,
            "after_all_cost_profit_floor_pUSD": "0",
            "next_action": (
                "freeze_only_one_current_exact_book_batch_for_the_best_candidate"
                if candidates
                else "stop_without_any_venue_request"
            ),
        },
        "authority": {
            "network_requests": 0,
            "credentials_used": False,
            "accounts_orders_funds_or_transactions": 0,
            "trading_authority": False,
        },
        "limitations": [
            "The retained source page was globally incomplete; this audit is exhaustive only for every complete base-halftime-second-half triple present in its immutable response.",
            "Side-specific Gamma fields may reject a package but cannot prove executable asks, synchronized depth, fees, capacity, fills, or profit.",
            "Historical active accepting-order metadata does not establish a current opportunity or recurrence.",
        ],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _self_hash(result, field="result_sha256")
    write_bytes_atomic(output, _canonical(result) + b"\n")
    print(
        json.dumps(
            {
                "population": result["population"],
                "best_relation": result["screen"]["best_relation"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
