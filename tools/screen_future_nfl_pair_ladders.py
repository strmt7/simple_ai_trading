"""One-use future NFL rejection using exhaustive, unchanged pair validators."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from tools import screen_future_nfl_payoff_floor as legacy
from tools.adjudicate_polymarket_nfl_catalog_side_specific import _side_specific_price
from tools.capture_public_source_bounded import _record, capture


POLICY = {
    "event_limit": 50,
    "eligible_total_limit": 128,
    "series_id": "12185",
    "selection": "earliest_native_full_game_start_slug_id_before_prices",
    "evaluation": "all_unordered_total_pairs_with_unchanged_complete_rule_validator",
}


def candidate_key(row: dict) -> tuple:
    return (
        row["side_specific_cost_pusd"],
        row["lower_threshold"],
        row["upper_threshold"],
    )


def screen(payload: object, *, minimum: str, maximum: str) -> dict:
    """Separate native response bounds from the eligible-ladder computation."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("native keyset shape changed")
    events = payload["events"]
    if (
        any(k in payload for k in ("next_cursor", "nextCursor"))
        or len(events) >= POLICY["event_limit"]
    ):
        raise ValueError(
            "native population is not demonstrably complete; no pagination"
        )
    eligible, ids, slugs = [], set(), set()
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("active") is not True
            or event.get("closed") is not False
        ):
            raise ValueError("native deployment filter failed")
        identifier, slug = (
            legacy._identity(event.get("id")),
            legacy._identity(event.get("slug")),
        )
        if identifier in ids or slug in slugs:
            raise ValueError("duplicate event identity")
        ids.add(identifier)
        slugs.add(slug)
        start = legacy._instant(event.get("startTime"), "startTime")
        if (
            not legacy._instant(minimum, "minimum")
            <= start
            <= legacy._instant(maximum, "maximum")
        ):
            raise ValueError("event lies outside the frozen window")
        if not isinstance(event.get("series"), list) or not any(
            isinstance(s, dict) and str(s.get("id")) == POLICY["series_id"]
            for s in event["series"]
        ):
            raise ValueError("native NFL series identity failed")
        if re.fullmatch(r"nfl-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}", slug):
            eligible.append((start, slug, identifier, event))
    if not eligible:
        return {"status": "no_deployed_full_game", "returned_event_count": len(events)}
    event = min(eligible, key=lambda row: row[:3])[3]
    markets = event.get("markets")
    if not isinstance(markets, list) or not all(isinstance(m, dict) for m in markets):
        raise ValueError("native market shape changed")
    market_ids = [legacy._identity(m.get("id")) for m in markets]
    if len(set(market_ids)) != len(market_ids):
        raise ValueError("duplicate native market identity")
    moneylines = [
        m
        for m in markets
        if legacy._active(m) and m.get("sportsMarketType") == "moneyline"
    ]
    totals = [
        m
        for m in markets
        if legacy._active(m) and m.get("sportsMarketType") == "totals"
    ]
    if len(moneylines) != 1 or not 2 <= len(totals) <= POLICY["eligible_total_limit"]:
        raise ValueError("complete active full-game ladder is outside the frozen scope")
    best, complete, incomplete, positive = None, 0, 0, 0
    for first, second in combinations(totals, 2):
        # Every original row participates. Projection is not a selected subset:
        # each pair reuses the unchanged complete resolver/asset-identity checks.
        pair = legacy._event_ladder(
            {**event, "markets": [moneylines[0], first, second]}
        )
        candidate = pair["strongest_package"]
        if candidate is None:
            incomplete += 1
            continue
        by_id = {str(m["id"]): m for m in (first, second)}
        over, _ = _side_specific_price(by_id[candidate["over_market_id"]], "Over")
        under, _ = _side_specific_price(by_id[candidate["under_market_id"]], "Under")
        if over <= 0 or under <= 0:
            incomplete += 1
            continue
        complete += 1
        positive += int(candidate["side_specific_cost_pusd"] < 1)
        if best is None or candidate_key(candidate) < candidate_key(best):
            best = candidate
    return {
        "status": "screened",
        "returned_event_count": len(events),
        "event_slug": event["slug"],
        "event_id": str(event["id"]),
        "native_market_count": len(markets),
        "eligible_total_count": len(totals),
        "relation_count": len(totals) * (len(totals) - 1) // 2,
        "price_complete_count": complete,
        "price_incomplete_count": incomplete,
        "strict_sub_floor_count": positive,
        "strongest_package": best,
        "gross_candidate_only": positive > 0 and incomplete == 0,
        "fees_depth_clock_and_recurrence_qualified": False,
        "native_oracle_and_collateral_identity_qualified": False,
    }


def validate(plan: dict, path: Path) -> None:
    """Check the entire request and one-use boundary before opening the journal."""
    capture(path, preflight=True)
    if plan["screen"] != POLICY or plan["response_byte_ceiling"] != 50 * 1024 * 1024:
        raise ValueError("frozen computation/transport policy changed")
    window = plan["window"]
    start, end = (legacy._instant(window[k], k) for k in ("minimum", "maximum"))
    if (
        not legacy._instant(plan["frozen_at_utc"], "freeze") < start < end
        or datetime.now(timezone.utc) >= start
    ):
        raise ValueError("window must remain strictly future")
    parsed = urlsplit(plan["request"]["url"])
    expected = {
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
    ) or parse_qs(parsed.query) != expected:
        raise ValueError("native endpoint or query changed")
    if set(plan["screen_outputs"]) != {"result_path", "journal_path"}:
        raise ValueError("screen output identities changed")
    outputs = [
        legacy._root_path(p)
        for p in [*plan["outputs"].values(), *plan["screen_outputs"].values()]
    ]
    if len(set(outputs)) != 5 or any(
        p.parent != path.parent or p.exists() for p in outputs
    ):
        raise ValueError(
            "one-use outputs exist, alias or escape the contract directory"
        )


def run(path: Path, *, preflight: bool = False) -> dict | None:
    """Retain the native response and terminal decision without resampling."""
    path = path.resolve()
    plan = json.loads(path.read_bytes())
    validate(plan, path)
    if preflight:
        return None
    with legacy._root_path(plan["screen_outputs"]["journal_path"]).open(
        "x", encoding="ascii", newline="\n"
    ) as journal:
        _record(
            journal, {"phase": "intent", "contract_sha256": plan["contract_sha256"]}
        )
        captured = None
        try:
            captured = capture(path)
            if captured is None:
                raise RuntimeError("capture returned no receipt")
            if captured["source_gate"]["passed"] is not True:
                raise ValueError("native transport/source gate failed")
            raw = legacy._root_path(plan["outputs"]["raw_path"]).read_bytes()
            if legacy._sha256(raw) != captured["capture"]["receipt"]["response_sha256"]:
                raise ValueError("native response binding failed")
            outcome = screen(
                json.loads(
                    raw,
                    object_pairs_hook=legacy._unique_object,
                    parse_constant=legacy._reject_constant,
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
            OSError,
        ) as error:
            outcome = {"status": "terminal_unqualified", "reason": str(error)}
        result = legacy._json_ready(
            {
                "schema_version": "future-nfl-pair-ladder-v1",
                "contract_sha256": plan["contract_sha256"],
                "capture_result_sha256": None
                if captured is None
                else captured["result_sha256"],
                "screen": outcome,
                "accepted_edge": False,
                "profitability_claim": False,
                "book_or_fee_request_authorized": False,
            }
        )
        result["result_sha256"] = legacy._canonical_hash(result, "result_sha256")
        with legacy._root_path(plan["screen_outputs"]["result_path"]).open(
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args.contract, preflight=args.preflight)
    print(
        json.dumps(
            {"preflight": args.preflight} if result is None else result["screen"],
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
