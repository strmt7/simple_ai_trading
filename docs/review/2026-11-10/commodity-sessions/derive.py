"""Replay the retained Binance commodity-perpetual session notice."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ARTICLE_CODE = "f69c05ebdba7422e9c614685abe84dc5"
BODY_SHA256 = "2f90231b6a819c33fff0c5bbb16d68a91c39f9e0714cf340861e0b0db1ac5562"


def _text(node: dict) -> str:
    if node.get("node") == "text":
        return node["text"]
    return "".join(_text(child) for child in node.get("child", []))


def _timestamp(value: int) -> str:
    seconds, milliseconds = divmod(value, 1000)
    return (
        datetime.fromtimestamp(seconds, timezone.utc)
        .replace(microsecond=milliseconds * 1000)
        .isoformat()
    )


def derive(directory: Path) -> dict:
    result = json.loads((directory / "source.result.json").read_bytes())
    receipt = result["capture"]["receipt"]
    body = (directory / "source.raw.json").read_bytes()
    if (
        result["source_gate"]["passed"] is not True
        or receipt["status_code"] != 200
        or len(body) != receipt["response_bytes"]
        or sha256(body).hexdigest() != receipt["response_sha256"]
        or receipt["response_sha256"] != BODY_SHA256
    ):
        raise ValueError("native article receipt differs")
    envelope = json.loads(body)
    article = envelope["data"]
    if (
        envelope["success"] is not True
        or envelope["code"] != "000000"
        or article["code"] != ARTICLE_CODE
        or article["id"] != 284483
        or article["version"] != "1"
    ):
        raise ValueError("native publisher identity differs")
    tree = json.loads(article["body"])
    if tree.get("node") != "root":
        raise ValueError("native article body format differs")
    body_text = " ".join(unescape(_text(tree)).split())
    terms = (
        "Effective from 2026-09-15 21:00 (UTC)",
        "From every Sunday 18:00 (ET) to every Friday 17:00 (ET)",
        "The price index is updated every second as a weighted average",
        "constituent prices quoted by third-party data vendors",
        "Weekends and public holidays",
        "from every Friday 17:00 (ET) to every Sunday 18:00 (ET)",
        "Orderbook EWMA mode",
        "will not impact commodity-based TradFi perpetual contracts trading operations",
        "such as XAUUSDT, XAGUSDT",
        "Notice under Binance Clearing Procedures",
    )
    if any(term not in body_text for term in terms):
        raise ValueError("native article semantics differ")

    eastern = ZoneInfo("America/New_York")
    friday = datetime.combine(date(2026, 10, 30), time(17), eastern)
    sunday = datetime.combine(date(2026, 11, 1), time(18), eastern)
    weekend_start = friday.astimezone(timezone.utc)
    weekend_end = sunday.astimezone(timezone.utc)
    # The elapsed duration must use UTC instants across the clock change.
    hours = (weekend_end - weekend_start) // timedelta(hours=1)
    next_friday = datetime.combine(date(2026, 11, 6), time(17), eastern)

    settlement = strike = 100
    short_entry_bid, long_entry_ask = 110, 90
    short_exit_ask, long_exit_bid = 108, 92
    cost = 3
    call_value = max(settlement - strike, 0)
    put_value = max(strike - settlement, 0)
    return {
        "article": {
            "id": article["id"],
            "code": article["code"],
            "version": article["version"],
            "publish_utc": _timestamp(article["publishDate"]),
            "updated_utc": _timestamp(article["lastUpdateTime"]),
            "effective_utc": "2026-09-15T21:00:00+00:00",
            "response_bytes": len(body),
            "response_sha256": BODY_SHA256,
        },
        "constructed_regular_weekend_utc": {
            "switch_to_ewma": weekend_start.isoformat(),
            "switch_to_standard": weekend_end.isoformat(),
            "elapsed_hours": hours,
            "next_friday_switch_to_ewma": next_friday.astimezone(
                timezone.utc
            ).isoformat(),
        },
        "constructed_per_unit_usdt": {
            "settlement_and_strike": strike,
            "all_in_cost": cost,
            "short_entry_bid": short_entry_bid,
            "short_exit_ask": short_exit_ask,
            "call_assuming_equal_exit": call_value
            + short_entry_bid
            - settlement
            - cost,
            "call_at_constructed_exit": call_value
            + short_entry_bid
            - short_exit_ask
            - cost,
            "long_entry_ask": long_entry_ask,
            "long_exit_bid": long_exit_bid,
            "put_assuming_equal_exit": put_value + settlement - long_entry_ask - cost,
            "put_at_constructed_exit": put_value
            + long_exit_bid
            - long_entry_ask
            - cost,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_directory", type=Path)
    print(json.dumps(derive(parser.parse_args().receipt_directory), indent=2))
