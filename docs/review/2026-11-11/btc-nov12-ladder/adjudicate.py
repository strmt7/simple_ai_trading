"""Adjudicate one exact BTC threshold ladder from frozen Gamma metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re


ZERO = Decimal(0)
ONE = Decimal(1)
_NUMERAL = r"(?:[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)"


def _normalize(text: object) -> str:
    return " ".join(text.split()) if isinstance(text, str) else ""


def _same_instant(native: object, expected: datetime) -> bool:
    if not isinstance(native, str):
        return False
    observed = datetime.fromisoformat(native)
    return observed.tzinfo is not None and observed == expected


def _identity(event: dict, contract: dict) -> list[tuple[int, dict, list[str]]]:
    title = contract["title"]
    scheduled = datetime.fromisoformat(contract["scheduled_observation_utc"])
    allowed_sources = ("", None, contract["resolution_url"])
    if (
        event.get("slug") != contract["event_slug"]
        or event.get("title") != title
        or event.get("active") is not True
        or event.get("closed") is not False
        or event.get("negRisk") is not False
        or _normalize(event.get("description")) != contract["normalized_rule"]
        or event.get("resolutionSource") not in allowed_sources
        or not _same_instant(event.get("endDate"), scheduled)
    ):
        raise ValueError("exact active event identity or common rule differs")
    markets = event.get("markets")
    if not isinstance(markets, list) or not 3 <= len(markets) <= 32:
        raise ValueError("exact event population outside frozen bounds")

    identities = []
    conditions, tokens, market_ids, thresholds, ordinals = (set() for _ in range(5))
    for market in markets:
        if not isinstance(market, dict):
            raise ValueError("native market row is not an object")
        label = market.get("groupItemTitle")
        if not isinstance(label, str) or re.fullmatch(_NUMERAL, label) is None:
            raise ValueError("threshold label differs from the frozen decimal grammar")
        threshold = int(label.replace(",", ""))
        question = f"Will the price of Bitcoin be above ${label} on November 12?"
        ordinal = market.get("groupItemThreshold")
        if not isinstance(ordinal, str) or re.fullmatch(r"[0-9]{1,2}", ordinal) is None:
            raise ValueError("market ordinal differs")
        ordinals.add(int(ordinal))
        condition = market.get("conditionId")
        market_id = market.get("id")
        native_tokens = market.get("clobTokenIds")
        native_outcomes = market.get("outcomes")
        selected_tokens = (
            json.loads(native_tokens) if isinstance(native_tokens, str) else None
        )
        outcomes = (
            json.loads(native_outcomes) if isinstance(native_outcomes, str) else None
        )
        if (
            market.get("question") != question
            or _normalize(market.get("description")) != contract["normalized_rule"]
            or market.get("active") is not True
            or market.get("closed") is not False
            or market.get("acceptingOrders") is not True
            or market.get("enableOrderBook") is not True
            or market.get("negRisk") is not False
            or not _same_instant(market.get("endDate"), scheduled)
            or not isinstance(condition, str)
            or re.fullmatch(r"0x[0-9a-f]{64}", condition) is None
            or not isinstance(market_id, str)
            or not market_id.isascii()
            or not market_id.isdecimal()
            or outcomes != ["Yes", "No"]
            or not isinstance(selected_tokens, list)
            or len(selected_tokens) != 2
            or any(
                not isinstance(token, str)
                or re.fullmatch(r"[0-9]{1,80}", token) is None
                for token in selected_tokens
            )
        ):
            raise ValueError(
                "threshold question, market state, source or token identity differs"
            )
        if market.get("resolutionSource") not in allowed_sources:
            raise ValueError(
                "structured resolution source conflicts with the common rule"
            )
        conditions.add(condition)
        market_ids.add(market_id)
        tokens.update(selected_tokens)
        thresholds.add(threshold)
        identities.append((threshold, market, selected_tokens))
    if (
        len(conditions) != len(markets)
        or len(market_ids) != len(markets)
        or len(thresholds) != len(markets)
        or len(tokens) != 2 * len(markets)
        or ordinals != set(range(len(markets)))
    ):
        raise ValueError(
            "duplicate instruments, duplicate thresholds or incomplete ordinals"
        )
    return sorted(identities, key=lambda item: item[0])


def _price(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("malformed side-specific price") from exc
    if (
        not price.is_finite()
        or not ZERO <= price <= ONE
        or price.as_tuple().exponent < -12
    ):
        raise ValueError("price outside frozen range or precision")
    return price if ZERO < price < ONE else None


def screen(event: dict, contract: dict) -> dict:
    identities = _identity(event, contract)
    prices = [
        (_price(market.get("bestAsk")), _price(market.get("bestBid")))
        for _, market, _ in identities
    ]
    if any(ask is not None and bid is not None and bid > ask for ask, bid in prices):
        raise ValueError("crossed price within one condition")

    rows = []
    for lower, higher in combinations(range(len(prices)), 2):
        ask, bid = prices[lower][0], prices[higher][1]
        if ask is not None and bid is not None:
            rows.append((ask + ONE - bid, lower, higher))
    rows.sort()
    survivors = sum(value < ONE for value, _, _ in rows)

    # The suffix maximum is the smallest decreasing YES valuation above all bids.
    values = [ZERO] * len(prices)
    suffix = ZERO
    for index in range(len(prices) - 1, -1, -1):
        suffix = max(suffix, prices[index][1] or ZERO)
        values[index] = suffix
    consistent = all(
        ask is None or valuation <= ask
        for (ask, _), valuation in zip(prices, values, strict=True)
    )
    if consistent != (survivors == 0):
        raise RuntimeError("strict pair gate and certificate disagree")
    weights = (
        [high - low for high, low in zip([ONE, *values], [*values, ZERO], strict=True)]
        if consistent
        else None
    )
    if weights is not None and (
        any(weight < ZERO for weight in weights) or sum(weights) != ONE
    ):
        raise RuntimeError("state certificate is invalid")

    best = None
    if rows:
        cost, lower, higher = rows[0]
        best = {
            "cost_pUSD": str(cost),
            "lower_yes_threshold": identities[lower][0],
            "higher_no_threshold": identities[higher][0],
            "lower_yes_token": identities[lower][2][0],
            "higher_no_token": identities[higher][2][1],
        }
    return {
        "thresholds_usdt": [row[0] for row in identities],
        "market_count": len(prices),
        "total_pairs": len(prices) * (len(prices) - 1) // 2,
        "price_complete_pairs": len(rows),
        "available_yes_asks": sum(ask is not None for ask, _ in prices),
        "available_no_proxies": sum(bid is not None for _, bid in prices),
        "strict_gross_survivors": survivors,
        "best_complete_pair": best,
        "ordinary_state_certificate_weights": None
        if weights is None
        else [str(weight) for weight in weights],
        "available_proxy_side_constraints": [
            {
                "threshold_usdt": threshold,
                "yes_ask": None if ask is None else str(ask),
                "no_ask_proxy": None if bid is None else str(ONE - bid),
            }
            for (threshold, _, _), (ask, bid) in zip(identities, prices, strict=True)
        ],
        "full_price_coverage": all(
            ask is not None and bid is not None for ask, bid in prices
        ),
    }


def adjudicate(directory: Path, contract: dict) -> dict:
    capture = json.loads((directory / "capture.json").read_bytes())
    receipt = capture["capture"]["receipt"]
    body = (directory / "raw/event.json").read_bytes()
    if (
        capture["contract"]["sha256"] != contract["contract_sha256"]
        or capture["source_gate"]["passed"] is not True
        or receipt["status_code"] != 200
        or receipt["response_bytes"] != len(body)
        or receipt["response_sha256"] != sha256(body).hexdigest()
    ):
        raise ValueError("source receipt differs or did not pass")
    requested = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        milliseconds=receipt["requested_at_ms"]
    )
    if (
        not datetime.fromisoformat(contract["frozen_at_utc"])
        < requested
        < datetime.fromisoformat(contract["scheduled_observation_utc"])
    ):
        raise ValueError("source request was outside the frozen window")
    event = json.loads(body, parse_float=Decimal)
    if not isinstance(event, dict):
        raise ValueError("Gamma did not return one event object")
    return {
        "gamma_event_id": event["id"],
        "raw_sha256": sha256(body).hexdigest(),
        **screen(event, contract),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_directory", type=Path)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            adjudicate(args.receipt_directory, json.loads(args.contract.read_bytes())),
            indent=2,
        )
    )
