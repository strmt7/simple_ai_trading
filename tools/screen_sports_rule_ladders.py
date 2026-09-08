"""Rejection-only integer ladders grouped by identical complete retained rules."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import json
import argparse
import hashlib
import os
from pathlib import Path

from tools.adjudicate_polymarket_nfl_catalog_side_specific import _side_specific_price
from tools.capture_public_source_contract import (
    _canonical_hash,
    _load_object,
    _root_path,
)
from tools.adjudicate_sports_deployment import instant
from datetime import datetime, timezone


def rule_row(market: dict) -> dict:
    """Bind a threshold without erasing observation, overtime or fallback differences."""
    kind = market.get("sportsMarketType")
    if kind not in {"spreads", "totals"}:
        raise ValueError("outside declared spread/total ladder scope")
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
    ):
        raise ValueError("market is not active and accepting")
    outcomes = json.loads(market["outcomes"])
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 2
        or not all(isinstance(x, str) and x for x in outcomes)
        or outcomes[0] == outcomes[1]
    ):
        raise ValueError("two distinct exact outcomes required")
    line = Decimal(str(market["line"]))
    if (
        not line.is_finite()
        or abs(line) % 1 != Decimal("0.5")
        or (kind == "spreads" and line >= 0)
        or (kind == "totals" and line < 0)
    ):
        raise ValueError("unsupported integer threshold boundary")
    threshold = int(abs(line) + Decimal("0.5"))
    description = " ".join(market["description"].split())
    cancellation = "If the game is canceled entirely, with no make-up game, this market will resolve 50-50."
    if cancellation not in description:
        raise ValueError("exact common cancellation rule missing")
    if kind == "spreads":
        positive = f'This market will resolve to "{outcomes[0]}" if {outcomes[0]} win the game by {threshold} or more points.'
        negative = f'Otherwise, this market will resolve to "{outcomes[1]}".'
        replacements = [
            (positive, "POSITIVE_IF_MARGIN_AT_LEAST_THRESHOLD"),
            (negative, "COMPLEMENT_OTHERWISE"),
        ]
    else:
        if outcomes != ["Over", "Under"]:
            raise ValueError("total outcomes differ from exact Over/Under")
        replacements = [
            (
                f"combine to score {threshold} or more points",
                "combine to score THRESHOLD or more points",
            ),
            (
                f'If the combined total is less than {threshold}, this market will resolve to "Under".',
                "COMPLEMENT_BELOW_THRESHOLD",
            ),
        ]
        if 'This market will resolve to "Over" if ' not in description:
            raise ValueError("positive total rule missing")
    for old, new in replacements:
        if description.count(old) != 1:
            raise ValueError("exact threshold rule missing or repeated")
        description = description.replace(old, new)
    return {
        "id": str(market["id"]),
        "threshold": threshold,
        "outcomes": outcomes,
        "group": (kind, tuple(outcomes), description, market.get("resolutionSource")),
    }


def evaluate(event: dict) -> dict:
    """Exhaust compatible lower-positive/higher-complement pairs; never authorize books."""
    markets = event.get("markets")
    if not isinstance(markets, list) or not 1 <= len(markets) <= 1000:
        raise ValueError("event market population outside bound")
    groups = defaultdict(list)
    by_id, exclusions = {}, []
    for market in markets:
        identifier = str(market.get("id"))
        if identifier in by_id:
            raise ValueError("duplicate market identity")
        by_id[identifier] = market
        try:
            row = rule_row(market)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            exclusions.append({"market_id": identifier, "reason": str(exc)})
            continue
        groups[row["group"]].append(row)
    relation_count = sum(
        sum(
            a["threshold"] <= b["threshold"] and a["id"] != b["id"]
            for a in rows
            for b in rows
        )
        for rows in groups.values()
    )
    if relation_count > 10000:
        raise ValueError("rule-only relation ceiling exceeded before economics")
    relations = []
    for key, rows in groups.items():
        for lower in rows:
            for upper in rows:
                if (
                    lower["id"] == upper["id"]
                    or lower["threshold"] > upper["threshold"]
                ):
                    continue
                relation = {
                    "family": key[0],
                    "positive_market_id": lower["id"],
                    "complement_market_id": upper["id"],
                    "lower_threshold": lower["threshold"],
                    "upper_threshold": upper["threshold"],
                    "common_rule_payout_floor": "1",
                }
                try:
                    p, ps = _side_specific_price(
                        by_id[lower["id"]], lower["outcomes"][0]
                    )
                    q, qs = _side_specific_price(
                        by_id[upper["id"]], upper["outcomes"][1]
                    )
                    if p <= 0 or q <= 0:
                        raise ValueError(
                            "zero acquisition quote does not prove a free leg"
                        )
                    relation.update(
                        price_complete=True,
                        positive_price=str(p),
                        complement_price=str(q),
                        price_sources=[ps, qs],
                        cost=str(p + q),
                        gross_floor_headroom=str(1 - p - q),
                        strictly_subfloor=p + q < 1,
                    )
                except (RuntimeError, ValueError, ArithmeticError) as exc:
                    relation.update(
                        price_complete=False, strictly_subfloor=False, reason=str(exc)
                    )
                relations.append(relation)
    candidates = [r for r in relations if r["strictly_subfloor"]]
    complete = [r for r in relations if r["price_complete"]]
    best = (
        min(
            complete,
            key=lambda r: (
                Decimal(r["cost"]),
                r["positive_market_id"],
                r["complement_market_id"],
            ),
        )
        if complete
        else None
    )
    return {
        "rule_group_count": len(groups),
        "included_markets": sum(map(len, groups.values())),
        "exclusions": exclusions,
        "relations": relations,
        "relation_count": len(relations),
        "price_complete_count": len(complete),
        "subfloor_count": len(candidates),
        "best_relation": best,
        "book_requests_authorized": False,
        "accepted_edge": False,
        "profitability_claim": False,
    }


def run(path: Path, *, preflight: bool = False) -> dict | None:
    """Verify frozen source/code bindings before one zero-network rejection screen."""
    plan = _load_object(path.resolve())
    if plan["contract_sha256"] != _canonical_hash(plan, "contract_sha256"):
        raise ValueError("contract hash mismatch")
    if instant(plan["frozen_at_utc"]) > datetime.now(timezone.utc):
        raise ValueError("contract is future dated")
    if plan["gate"] != {
        "network_requests": 0,
        "book_requests_authorized": False,
        "maximum_relations": 10000,
        "price_semantics": "bestAsk_or_1-bestBid_rejection_only",
        "families": ["spreads", "totals"],
    }:
        raise ValueError("frozen evaluation scope changed")
    for binding in [plan["source"], *plan["implementations"]]:
        if (
            hashlib.sha256(_root_path(binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise ValueError("source or implementation binding mismatch")
    event = _load_object(_root_path(plan["source"]["path"]))
    if event.get("slug") != plan["event_slug"]:
        raise ValueError("event slug mismatch")
    destination = _root_path(plan["result_path"])
    if destination.exists():
        raise FileExistsError("one-use adjudication exists")
    if preflight:
        for market in event["markets"]:
            if market.get("sportsMarketType") in {"spreads", "totals"}:
                rule_row(market)
        return None
    with destination.open("x", encoding="ascii", newline="\n") as stream:
        try:
            screen = evaluate(event)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            screen = {
                "failed_closed": True,
                "failure_type": type(exc).__name__,
                "book_requests_authorized": False,
            }
        result = {
            "contract_sha256": plan["contract_sha256"],
            "screen": screen,
            "accepted_edge": False,
            "profitability_claim": False,
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args.contract, preflight=args.preflight)
    if result is None:
        print("offline rule-only preflight passed")
    else:
        screen = result["screen"]
        print(
            json.dumps(
                {
                    key: screen.get(key)
                    for key in [
                        "relation_count",
                        "price_complete_count",
                        "subfloor_count",
                        "best_relation",
                        "failed_closed",
                    ]
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
