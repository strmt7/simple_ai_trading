"""Exhaust a retained exact-date versus cumulative-deadline payoff graph."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from simple_ai_trading.polymarket_fees import PolymarketFeeModel


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-release-date-deadline-graph-adjudication-v1"
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


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a decimal") from exc
    if not parsed.is_finite():
        raise RuntimeError(f"{name} must be finite")
    return parsed


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _utc_instant(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{name} must be an explicit UTC instant")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _json_list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a JSON string")
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise RuntimeError(f"{name} must decode to a list")
    return parsed


def _question_date(question: str, pattern: re.Pattern[str]) -> date | None:
    matched = pattern.fullmatch(question)
    if matched is None:
        return None
    return datetime.strptime(matched.group("date"), "%B %d, %Y").date()


def _fee_model(market: dict[str, Any]) -> PolymarketFeeModel:
    schedule = market.get("feeSchedule")
    if not isinstance(schedule, dict):
        raise RuntimeError(f"market {market.get('id')} lacks a fee schedule")
    rate = _decimal(schedule.get("rate"), name="fee rate")
    exponent = _decimal(schedule.get("exponent"), name="fee exponent")
    if (
        rate < 0
        or rate > 1
        or exponent <= 0
        or exponent != exponent.to_integral_value()
        or schedule.get("takerOnly") is not True
    ):
        raise RuntimeError(f"market {market.get('id')} fee schedule is unsupported")
    enabled = market.get("feesEnabled") is True
    if enabled is not (rate > 0):
        raise RuntimeError(f"market {market.get('id')} fee state is inconsistent")
    return PolymarketFeeModel(enabled, rate, int(exponent), True)


def _validate_market(
    market: dict[str, Any], *, required_fragments: list[str]
) -> None:
    market_id = str(market.get("id") or "")
    if not market_id.isdigit():
        raise RuntimeError("market id is invalid")
    if _json_list(market.get("outcomes"), name="outcomes") != ["Yes", "No"]:
        raise RuntimeError(f"market {market_id} outcomes differ")
    tokens = _json_list(market.get("clobTokenIds"), name="token ids")
    if len(tokens) != 2 or len(set(str(value) for value in tokens)) != 2:
        raise RuntimeError(f"market {market_id} token mapping differs")
    description = str(market.get("description") or "")
    if not all(fragment in description for fragment in required_fragments):
        raise RuntimeError(f"market {market_id} rule fragments differ")
    _fee_model(market)
    tick = _decimal(market.get("orderPriceMinTickSize"), name="tick size")
    minimum = _decimal(market.get("orderMinSize"), name="minimum order")
    if tick <= 0 or minimum <= 0:
        raise RuntimeError(f"market {market_id} execution terms are invalid")


def _acquisition(
    market: dict[str, Any], *, outcome: str
) -> dict[str, Any] | None:
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        return None
    if outcome == "Yes":
        raw_price = market.get("bestAsk")
        if raw_price is None:
            return None
        price = _decimal(raw_price, name="YES bestAsk")
        price_source = "direct_YES_bestAsk"
        token_index = 0
    elif outcome == "No":
        raw_bid = market.get("bestBid")
        if raw_bid is None:
            return None
        price = ONE - _decimal(raw_bid, name="YES bestBid")
        price_source = "one_minus_direct_YES_bestBid"
        token_index = 1
    else:
        raise RuntimeError("unsupported outcome")
    if price <= 0 or price >= 1:
        return None
    tokens = _json_list(market.get("clobTokenIds"), name="token ids")
    return {
        "market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "outcome": outcome,
        "token_id": str(tokens[token_index]),
        "price_pUSD_per_share": price,
        "price_source": price_source,
        "tick_size_pUSD": _decimal(
            market["orderPriceMinTickSize"], name="tick size"
        ),
        "minimum_order_shares": _decimal(
            market["orderMinSize"], name="minimum order"
        ),
        "fee_model": _fee_model(market),
    }


def _package_row(
    *,
    family: str,
    relation: str,
    exact_date: date | None,
    deadline_date: date,
    first_market: dict[str, Any],
    first_outcome: str,
    second_market: dict[str, Any],
    second_outcome: str,
) -> dict[str, Any]:
    identity = {
        "family": family,
        "relation": relation,
        "exact_date": exact_date.isoformat() if exact_date else None,
        "deadline_date": deadline_date.isoformat(),
        "first_market_id": str(first_market["id"]),
        "second_market_id": str(second_market["id"]),
    }
    first = _acquisition(first_market, outcome=first_outcome)
    second = _acquisition(second_market, outcome=second_outcome)
    if first is None or second is None:
        return {
            **identity,
            "status": "missing_side_specific_acquisition_evidence",
            "passes_strict_metadata_gate": False,
            "passes_fee_and_one_tick_gate": False,
        }
    quantity = max(first["minimum_order_shares"], second["minimum_order_shares"])
    actual_prices = [first["price_pUSD_per_share"], second["price_pUSD_per_share"]]
    stressed_prices = [
        first["price_pUSD_per_share"] + first["tick_size_pUSD"],
        second["price_pUSD_per_share"] + second["tick_size_pUSD"],
    ]
    actual_sum = sum(actual_prices, Decimal("0"))
    gross_headroom = quantity * (ONE - actual_sum)
    if any(price >= 1 for price in stressed_prices):
        stressed_fee = None
        stressed_headroom = None
    else:
        stressed_fee = first["fee_model"](
            stressed_prices[0], quantity, "taker"
        ) + second["fee_model"](stressed_prices[1], quantity, "taker")
        stressed_headroom = quantity * (
            ONE - sum(stressed_prices, Decimal("0"))
        ) - stressed_fee
    row = {
        **identity,
        "status": "priced",
        "guaranteed_payout_floor_pUSD_per_share": "1",
        "quantity_shares_each_leg": _decimal_text(quantity),
        "legs": [
            {
                key: (_decimal_text(value) if isinstance(value, Decimal) else value)
                for key, value in first.items()
                if key != "fee_model"
            },
            {
                key: (_decimal_text(value) if isinstance(value, Decimal) else value)
                for key, value in second.items()
                if key != "fee_model"
            },
        ],
        "metadata_cost_pUSD_per_share": _decimal_text(actual_sum),
        "metadata_gross_headroom_pUSD": _decimal_text(gross_headroom),
        "one_adverse_tick_per_leg_prices_pUSD": [
            _decimal_text(value) for value in stressed_prices
        ],
        "stressed_taker_fee_pUSD": _decimal_text(stressed_fee),
        "after_fee_one_tick_profit_floor_pUSD": _decimal_text(stressed_headroom),
        "passes_strict_metadata_gate": actual_sum < ONE,
        "passes_fee_and_one_tick_gate": (
            stressed_headroom is not None and stressed_headroom > 0
        ),
    }
    return row


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen = _utc_instant(contract.get("frozen_at_utc"), name="frozen_at_utc")
    if frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is in the future")
    implementation = contract["implementation"]
    implementation_path = _root_path(str(implementation["path"]))
    if _sha256(implementation_path.read_bytes()) != implementation["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    authority = contract.get("authority")
    if authority != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")


def _load_bound_event(source: dict[str, Any]) -> dict[str, Any]:
    path = _root_path(str(source["path"]))
    raw = path.read_bytes()
    if _sha256(raw) != source["sha256"]:
        raise RuntimeError(f"source hash mismatch: {path.name}")
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise RuntimeError(f"{path.name} must contain an event object")
    if not (
        str(event.get("id")) == str(source["event_id"])
        and event.get("slug") == source["event_slug"]
        and isinstance(event.get("markets"), list)
        and len(event["markets"]) == source["market_count"]
    ):
        raise RuntimeError(f"source event identity differs: {path.name}")
    return event


def adjudicate(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    _validate_contract(contract, contract_path)
    exact_event = _load_bound_event(contract["sources"]["exact_date_event"])
    deadline_event = _load_bound_event(contract["sources"]["deadline_event"])
    parsing = contract["parsing"]
    exact_pattern = re.compile(str(parsing["exact_date_question_regex"]))
    deadline_pattern = re.compile(str(parsing["deadline_question_regex"]))
    eligible_date = date.fromisoformat(str(parsing["eligible_date_on_or_after"]))
    exact_event_start = _utc_instant(
        exact_event.get("startDate"), name="exact event startDate"
    )
    timezone_name = str(parsing["calendar_timezone"])
    calendar_timezone = ZoneInfo(timezone_name)

    exact_markets: list[tuple[date, dict[str, Any]]] = []
    no_release_market: dict[str, Any] | None = None
    for raw_market in exact_event["markets"]:
        if not isinstance(raw_market, dict):
            raise RuntimeError("exact-date market is not an object")
        question = str(raw_market.get("question") or "")
        parsed_date = _question_date(question, exact_pattern)
        if question == parsing["no_release_question"]:
            no_release_market = raw_market
        if parsed_date is None:
            continue
        _validate_market(
            raw_market,
            required_fragments=list(parsing["exact_required_rule_fragments"]),
        )
        if parsed_date >= eligible_date:
            exact_markets.append((parsed_date, raw_market))

    deadline_markets: list[tuple[date, dict[str, Any], datetime]] = []
    for raw_market in deadline_event["markets"]:
        if not isinstance(raw_market, dict):
            raise RuntimeError("deadline market is not an object")
        question = str(raw_market.get("question") or "")
        parsed_date = _question_date(question, deadline_pattern)
        if parsed_date is None:
            continue
        _validate_market(
            raw_market,
            required_fragments=list(parsing["deadline_required_rule_fragments"]),
        )
        if parsed_date >= eligible_date:
            deadline_markets.append(
                (
                    parsed_date,
                    raw_market,
                    _utc_instant(
                        raw_market.get("startDate"), name="deadline market startDate"
                    ),
                )
            )

    if len(exact_markets) != int(parsing["expected_eligible_exact_date_count"]):
        raise RuntimeError("eligible exact-date count differs")
    if len(deadline_markets) != int(parsing["expected_eligible_deadline_count"]):
        raise RuntimeError("eligible deadline count differs")
    if no_release_market is None:
        raise RuntimeError("no-release condition is missing")
    _validate_market(
        no_release_market,
        required_fragments=list(parsing["exact_required_rule_fragments"]),
    )

    rows: list[dict[str, Any]] = []
    excluded = {
        "deadline_created_after_exact_date_started": 0,
        "earlier_deadline_has_unproved_pre_exact_event_history": 0,
    }
    for exact_date, exact_market in sorted(exact_markets, key=lambda item: item[0]):
        exact_day_start = datetime.combine(
            exact_date, time.min, tzinfo=calendar_timezone
        ).astimezone(timezone.utc)
        for deadline_date, deadline_market, deadline_start in sorted(
            deadline_markets, key=lambda item: item[0]
        ):
            if deadline_date >= exact_date:
                if deadline_start > exact_day_start:
                    excluded["deadline_created_after_exact_date_started"] += 1
                    continue
                rows.append(
                    _package_row(
                        family="exact_date_implies_cumulative_deadline",
                        relation="A_exact_date_implies_B_released_by_deadline",
                        exact_date=exact_date,
                        deadline_date=deadline_date,
                        first_market=exact_market,
                        first_outcome="No",
                        second_market=deadline_market,
                        second_outcome="Yes",
                    )
                )
            elif deadline_start >= exact_event_start:
                rows.append(
                    _package_row(
                        family="exact_date_mutually_exclusive_with_earlier_deadline",
                        relation="A_exact_date_and_B_released_by_earlier_deadline_cannot_both_be_true",
                        exact_date=exact_date,
                        deadline_date=deadline_date,
                        first_market=exact_market,
                        first_outcome="No",
                        second_market=deadline_market,
                        second_outcome="No",
                    )
                )
            else:
                excluded[
                    "earlier_deadline_has_unproved_pre_exact_event_history"
                ] += 1

    no_release_date = date.fromisoformat(str(parsing["no_release_through_date"]))
    for deadline_date, deadline_market, deadline_start in deadline_markets:
        if deadline_date <= no_release_date and deadline_start >= exact_event_start:
            rows.append(
                _package_row(
                    family="no_release_state_mutually_exclusive_with_post_creation_deadline",
                    relation="C_no_release_through_horizon_and_B_post_creation_release_by_deadline_cannot_both_be_true",
                    exact_date=None,
                    deadline_date=deadline_date,
                    first_market=no_release_market,
                    first_outcome="No",
                    second_market=deadline_market,
                    second_outcome="No",
                )
            )

    rows.sort(
        key=lambda row: (
            str(row["family"]),
            str(row.get("exact_date") or ""),
            str(row["deadline_date"]),
            str(row["first_market_id"]),
            str(row["second_market_id"]),
        )
    )
    priced = [row for row in rows if row["status"] == "priced"]
    metadata_passes = [row for row in priced if row["passes_strict_metadata_gate"]]
    stressed_passes = [row for row in priced if row["passes_fee_and_one_tick_gate"]]
    best_metadata = min(
        priced,
        key=lambda row: Decimal(str(row["metadata_cost_pUSD_per_share"])),
        default=None,
    )
    best_stressed = max(
        (
            row
            for row in priced
            if row["after_fee_one_tick_profit_floor_pUSD"] is not None
        ),
        key=lambda row: Decimal(str(row["after_fee_one_tick_profit_floor_pUSD"])),
        default=None,
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source_binding": contract["sources"],
        "scope": {
            "eligible_date_on_or_after": eligible_date.isoformat(),
            "eligible_exact_date_market_count": len(exact_markets),
            "eligible_deadline_market_count": len(deadline_markets),
            "valid_relation_count": len(rows),
            "priced_relation_count": len(priced),
            "excluded_relation_counts": excluded,
            "gamma_outcomePrices_field_used_for_economics": False,
        },
        "payoff_proof": {
            "implication": "If the next post-event-start release is exactly D and a deadline condition existed before D begins with L >= D, then release-by-L is true; NO(exact D)+YES(by L) pays at least one.",
            "mutual_exclusion": "If a deadline condition starts no earlier than the exact-date event and L < D, release-by-L and next-release-exactly-D cannot both be true; their two NO tokens pay at least one.",
            "no_release_exclusion": "If a deadline condition starts no earlier than the exact-date event, release-by-L through the no-release horizon and no-release-through-horizon cannot both be true; their two NO tokens pay at least one.",
            "prehistory_guard": "Earlier deadline conditions that predate the exact-date event are excluded from mutual-exclusion and no-release rows unless their complete intervening release history is source-proved.",
        },
        "summary": {
            "strict_metadata_subfloor_count": len(metadata_passes),
            "fee_and_one_tick_positive_count": len(stressed_passes),
            "best_metadata_row": best_metadata,
            "best_fee_and_one_tick_row": best_stressed,
        },
        "relations": rows,
        "adjudication": {
            "status": (
                "candidate_survives_fee_and_one_tick_before_depth"
                if stressed_passes
                else "terminal_before_books_no_relation_survives_fee_and_one_tick"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "book_request_justified": len(stressed_passes) > 0,
            "next_action": (
                "Freeze one exact book request for the deterministic best surviving row."
                if stressed_passes
                else "Do not request books or fee endpoints for this exact retained two-event population."
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = _load_object(contract_path)
    result_path = _root_path(str(contract["outputs"]["result_path"]))
    if result_path.exists():
        raise RuntimeError("one-use output already exists")
    result = adjudicate(contract, contract_path)
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
                "book_request_justified": result["adjudication"][
                    "book_request_justified"
                ],
                "fee_and_one_tick_positive_count": result["summary"][
                    "fee_and_one_tick_positive_count"
                ],
                "payloads_printed": 0,
                "priced_relation_count": result["scope"]["priced_relation_count"],
                "strict_metadata_subfloor_count": result["summary"][
                    "strict_metadata_subfloor_count"
                ],
                "valid_relation_count": result["scope"]["valid_relation_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
