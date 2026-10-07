"""One-use future NFL total-ladder rejection, never midpoint or execution proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _json_ready,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_nfl_catalog_side_specific import _side_specific_price
from tools.capture_public_source_bounded import _record, capture
from tools.screen_polymarket_mlb_cross_period_catalog import _instant


LIMITS = {
    "event_limit": 50,
    "market_limit": 256,
    "threshold_limit": 32,
    "maximum_line": 128,
    "series_id": "12185",
    "selection": "earliest_startTime_then_slug_then_id_before_prices",
    "family": "full_game_total_only",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite native JSON constant")


def _identity(value: object) -> str:
    if not (
        isinstance(value, str)
        and value.strip() == value
        and bool(value)
        or type(value) is int
        and value > 0
    ):
        raise ValueError("native identity is missing or malformed")
    return str(value)


def _pair(raw: object) -> list[str]:
    if not isinstance(raw, str):
        raise ValueError("outcomes must be a native JSON string")
    value = json.loads(raw)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(x, str) or not x for x in value)
        or value[0] == value[1]
    ):
        raise ValueError("outcomes must be two distinct named strings")
    return value


def _active(row: dict[str, Any]) -> bool:
    return (
        row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
    )


def _event_ladder(event: dict[str, Any]) -> dict[str, Any]:
    markets = event.get("markets")
    if not isinstance(markets, list) or not 1 <= len(markets) <= LIMITS["market_limit"]:
        raise ValueError("selected event market resource boundary failed")
    ids = [_identity(m.get("id")) for m in markets if isinstance(m, dict)]
    if len(ids) != len(markets) or "" in ids or len(set(ids)) != len(ids):
        raise ValueError("selected event market identities are missing or ambiguous")
    moneylines = [
        m for m in markets if _active(m) and m.get("sportsMarketType") == "moneyline"
    ]
    if len(moneylines) != 1:
        raise ValueError("selected event must have one active full-game moneyline")
    team_a, team_b = _pair(moneylines[0].get("outcomes"))
    totals = [
        m for m in markets if _active(m) and m.get("sportsMarketType") == "totals"
    ]
    if not 2 <= len(totals) <= LIMITS["threshold_limit"]:
        raise ValueError("selected full-game total threshold population failed")
    rows: list[tuple[int, dict[str, Any]]] = []
    common_schedule = None
    conditions = set()
    token_ids = set()
    for market in totals:
        if _pair(market.get("outcomes")) != ["Over", "Under"]:
            raise ValueError("full-game total outcome identity changed")
        condition = market.get("conditionId")
        tokens = _pair(market.get("clobTokenIds"))
        if (
            not isinstance(condition, str)
            or re.fullmatch(r"0x[0-9a-fA-F]{64}", condition) is None
            or condition.lower() in conditions
            or any(
                not x.isascii()
                or not x.isdecimal()
                or len(x) > 78
                or int(x) >= 2**256
                or str(int(x)) != x
                or x in token_ids
                for x in tokens
            )
        ):
            raise ValueError("native condition/token identity is invalid or aliased")
        conditions.add(condition.lower())
        token_ids.update(tokens)
        line = Decimal(str(market.get("line")))
        if (
            not line.is_finite()
            or not 0 < line <= LIMITS["maximum_line"]
            or line % 1 != Decimal("0.5")
        ):
            raise ValueError(
                "full-game total line resource or half-point contract failed"
            )
        threshold = int(line + Decimal("0.5"))
        description = market.get("description")
        if not isinstance(description, str) or len(description.encode("utf-8")) > 8192:
            raise ValueError("full-game rule description is absent or oversized")
        first, separator, _tail = description.partition("\n")
        schedule = " ".join(first.split())
        if (
            not separator
            or not schedule.startswith("In the upcoming NFL game, scheduled for ")
            or not schedule.endswith(" ET:")
        ):
            raise ValueError("full-game schedule source schema changed")
        if common_schedule is not None and schedule != common_schedule:
            raise ValueError("full-game scheduled observation identity differs")
        common_schedule = schedule
        expected = (
            f'{schedule} This market will resolve to "Over" if the {team_a} and {team_b} '
            f"combine to score {threshold} or more points in this game. "
            f'If the combined total is less than {threshold}, this market will resolve to "Under". '
            "If the game is postponed, this market will remain open until the game has been completed. "
            "If the game is canceled entirely, with no make-up game, this market will resolve 50-50."
        )
        if " ".join(description.split()) != expected:
            raise ValueError(
                "complete full-game resolver, postponement or cancellation rules differ"
            )
        rows.append((threshold, market))
    rows.sort(key=lambda row: row[0])
    if len({t for t, _m in rows}) != len(rows):
        raise ValueError("duplicate logical total thresholds")
    best = None
    incomplete = positive = evaluated = 0
    for index, (lower, superset) in enumerate(rows[:-1]):
        for upper, subset in rows[index + 1 :]:
            try:
                over, _ = _side_specific_price(superset, "Over")
                under, _ = _side_specific_price(subset, "Under")
            except (RuntimeError, ValueError, TypeError, ArithmeticError):
                incomplete += 1
                continue
            evaluated += 1
            cost = over + under
            positive += int(cost < 1)
            candidate = {
                "lower_threshold": lower,
                "upper_threshold": upper,
                "over_market_id": str(superset["id"]),
                "under_market_id": str(subset["id"]),
                "over_condition_id": superset["conditionId"],
                "under_condition_id": subset["conditionId"],
                "over_token_id": _pair(superset["clobTokenIds"])[0],
                "under_token_id": _pair(subset["clobTokenIds"])[1],
                "over_price_source": "bestAsk",
                "under_price_source": "1-bestBid",
                "side_specific_cost_pusd": cost,
                "conditional_payoff_floor_pusd": Decimal(1),
                "gross_surplus_pusd": Decimal(1) - cost,
            }
            if best is None or (cost, lower, upper) < (
                best["side_specific_cost_pusd"],
                best["lower_threshold"],
                best["upper_threshold"],
            ):
                best = candidate
    return {
        "event_slug": event["slug"],
        "event_id": str(event["id"]),
        "start_time": event["startTime"],
        "schedule_rule_identity": common_schedule,
        "threshold_count": len(rows),
        "relation_count": len(rows) * (len(rows) - 1) // 2,
        "price_complete_count": evaluated,
        "price_incomplete_count": incomplete,
        "strict_sub_floor_count": positive,
        "strongest_package": best,
        "payoff_partition": [
            "score < lower: Over(lower)=0, Under(upper)=1",
            "lower <= score < upper: Over(lower)=1, Under(upper)=1",
            "score >= upper: Over(lower)=1, Under(upper)=0",
            "common canceled/no-makeup: 0.5 + 0.5 = 1",
        ],
        "gross_candidate_only": positive > 0 and incomplete == 0,
        "fees_depth_clock_and_recurrence_qualified": False,
    }


def screen(payload: object, *, minimum: str, maximum: str) -> dict[str, Any]:
    """Select the earliest native event before reading any of its economic fields."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("native keyset shape changed")
    if "next_cursor" in payload or len(payload["events"]) > LIMITS["event_limit"]:
        raise ValueError("incomplete or oversized native population; no pagination")
    events = payload["events"]
    if not events:
        return {"status": "no_deployed_event", "returned_event_count": 0}
    event_ids = set()
    event_slugs = set()
    eligible = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("active") is not True
            or event.get("closed") is not False
        ):
            raise ValueError("native event deployment filter failed")
        identity = (_identity(event.get("id")), _identity(event.get("slug")))
        if (
            not isinstance(event["slug"], str)
            or identity[0] in event_ids
            or identity[1] in event_slugs
        ):
            raise ValueError("native event identity is missing or duplicated")
        event_ids.add(identity[0])
        event_slugs.add(identity[1])
        if (
            len(json.dumps(event, ensure_ascii=True, allow_nan=False).encode("ascii"))
            > 1024 * 1024
        ):
            raise ValueError("native event exceeded its per-row resource budget")
        start = _instant(event.get("startTime"), "startTime")
        series = event.get("series")
        if not isinstance(series, list) or not any(
            isinstance(x, dict) and str(x.get("id")) == LIMITS["series_id"]
            for x in series
        ):
            raise ValueError("native NFL series identity failed")
        if not _instant(minimum, "minimum") <= start <= _instant(maximum, "maximum"):
            raise ValueError("native event outside the fixed future window")
        eligible.append((start, identity[1], identity[0], event))
    selected = min(eligible, key=lambda row: row[:3])[3]
    return {
        "status": "screened",
        "returned_event_count": len(events),
        **_event_ladder(selected),
    }


def _validate(plan: dict[str, Any], path: Path) -> None:
    capture(path, preflight=True)
    if plan["screen"] != LIMITS or plan["response_byte_ceiling"] != 50 * 1024 * 1024:
        raise ValueError("frozen screen/resource policy differs")
    window = plan["window"]
    if (
        not _instant(plan["frozen_at_utc"], "freeze")
        < _instant(window["minimum"], "minimum")
        < _instant(window["maximum"], "maximum")
    ):
        raise ValueError("window must be strictly future at freeze")
    if datetime.now(timezone.utc) >= _instant(window["minimum"], "minimum"):
        raise ValueError("future-window capture deadline passed")
    parsed = urlsplit(plan["request"]["url"])
    expected_query = {
        "limit": ["50"],
        "order": ["startTime"],
        "ascending": ["true"],
        "closed": ["false"],
        "start_time_min": [window["minimum"]],
        "start_time_max": [window["maximum"]],
        "series_id": ["12185"],
    }
    if (parsed.scheme, parsed.netloc, parsed.path) != (
        "https",
        "gamma-api.polymarket.com",
        "/events/keyset",
    ) or parse_qs(parsed.query) != expected_query:
        raise ValueError("frozen native endpoint/query differs")
    paths = [_root_path(p) for p in plan["screen_outputs"].values()]
    if (
        set(plan["screen_outputs"]) != {"result_path", "journal_path"}
        or len(set(paths)) != 2
    ):
        raise ValueError("screen outputs differ")
    all_paths = paths + [_root_path(p) for p in plan["outputs"].values()]
    if len(set(all_paths)) != 5 or any(
        p.parent != path.resolve().parent for p in all_paths
    ):
        raise ValueError(
            "all one-use outputs must be distinct and in the contract directory"
        )
    if any(
        p.exists() or not p.parent.is_dir() or not os.access(p.parent, os.W_OK)
        for p in paths
    ):
        raise ValueError("one-use screen outputs exist or are unwritable")


def run(path: Path, *, preflight: bool = False) -> None:
    plan = json.loads(path.read_bytes())
    _validate(plan, path.resolve())
    if preflight:
        return
    with _root_path(plan["screen_outputs"]["journal_path"]).open(
        "x", encoding="ascii", newline="\n"
    ) as journal:
        _record(
            journal, {"phase": "intent", "contract_sha256": plan["contract_sha256"]}
        )
        captured = capture(path)
        assert captured is not None
        try:
            if captured["source_gate"]["passed"] is not True:
                raise ValueError("retained transport/source gate failed")
            raw = _root_path(plan["outputs"]["raw_path"]).read_bytes()
            if _sha256(raw) != captured["capture"]["receipt"]["response_sha256"]:
                raise ValueError("retained native response hash differs")
            outcome = screen(
                json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                ),
                minimum=plan["window"]["minimum"],
                maximum=plan["window"]["maximum"],
            )
        except (
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            ArithmeticError,
        ) as error:
            outcome = {"status": "terminal_unqualified", "reason": str(error)}
        result = _json_ready(
            {
                "schema_version": "future-nfl-total-payoff-floor-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": plan["contract_sha256"],
                "capture_result_sha256": captured["result_sha256"],
                "screen": outcome,
                "accepted_edge": False,
                "profitability_claim": False,
                "book_or_fee_request_authorized": False,
            }
        )
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        with _root_path(plan["screen_outputs"]["result_path"]).open(
            "x", encoding="ascii", newline="\n"
        ) as output:
            output.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        _record(
            journal,
            {
                "phase": "completed",
                "result_sha256": result["result_sha256"],
                "status": outcome["status"],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.contract, preflight=args.preflight)


if __name__ == "__main__":
    main()
