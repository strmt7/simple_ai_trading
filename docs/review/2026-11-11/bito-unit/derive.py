"""Reconcile the retained BITO split notice and exchange transition."""

from __future__ import annotations

import argparse
from decimal import Decimal
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re


_ARTICLE = "dfd8cf5e698146ce83b560026529a752"
_EXCHANGE_DIGEST = "738cf0649e1dfa1931d9d8b2a08dde363b209caf3658f7557d0473073941e783"
_ISSUER_DIGEST = "025be4429f349c51995d30b459a312f12befa5d45be5bcc350fc8dfc950c7068"
_ISSUER_URL = (
    "https://www.proshares.com/press-releases/proshares-announces-share-splits"
)


def _text(node: dict) -> str:
    if node.get("node") == "text":
        return node["text"]
    return " ".join(_text(child) for child in node.get("child", []))


def _walk(node: dict):
    yield node
    for child in node.get("child", []):
        yield from _walk(child)


def derive(exchange: Path, issuer: Path) -> dict:
    wire = (exchange / "source.raw.json").read_bytes()
    capture = json.loads((exchange / "source.result.json").read_bytes())
    receipt = capture["capture"]["receipt"]
    if (
        capture["source_gate"]["passed"] is not True
        or receipt["status_code"] != 200
        or len(wire) != receipt["response_bytes"]
        or sha256(wire).hexdigest() != receipt["response_sha256"]
        or receipt["response_sha256"] != _EXCHANGE_DIGEST
    ):
        raise ValueError("exchange receipt differs")
    response = json.loads(wire)
    article = response["data"]
    if (
        response["success"] is not True
        or response["code"] != "000000"
        or (article["id"], article["code"], article["version"])
        != (289544, _ARTICLE, "1")
    ):
        raise ValueError("exchange publisher identity differs")
    tree = json.loads(article["body"])
    if tree.get("node") != "root" or not any(
        node.get("tag") == "a" and node.get("attr", {}).get("href") == _ISSUER_URL
        for node in _walk(tree)
    ):
        raise ValueError("exchange notice issuer link differs")
    article_text = " ".join(unescape(_text(tree)).split())
    required_exchange = (
        "2026-11-19 00:45 (UTC)",
        "2026-11-19 01:00 (UTC)",
        "2026-11-19 03:25 (UTC)",
        "2026-11-19 03:30 (UTC)",
        "2026-11-19 14:30 (UTC)",
        "Adjustment Scale Factor 0.5",
        "will not be reinstated after the Adjustment",
        "including reduce-only orders",
        "funding fee, mark price, price index and liquidation functions will be suspended",
        "the users’ Cross Margin Futures Account",
        "new positions will be re-instated",
        "based on the initial margin requirement",
        "submit, modify or cancel Post-Only orders only",
        "TRADING_HALT",
        "TRADING_CANCEL_ONLY",
        "corrected within 24 hours",
    )
    if any(term not in article_text for term in required_exchange):
        raise ValueError("exchange transition wording differs")

    extracted = (issuer / "issuer.native.json").read_bytes()
    issuer_result = json.loads((issuer / "result.json").read_bytes())
    if (
        issuer_result["source_admitted"] is not True
        or issuer_result["redirected_status_code"] != 200
        or issuer_result["url"] != _ISSUER_URL
        or issuer_result["redirected_url"] != _ISSUER_URL
        or len(extracted) != issuer_result["native_bytes"]
        or sha256(extracted).hexdigest() != issuer_result["native_sha256"]
        or issuer_result["native_sha256"] != _ISSUER_DIGEST
    ):
        raise ValueError("issuer receipt differs")
    native = json.loads(extracted)
    if native["success"] is not True or native["status_code"] != 200:
        raise ValueError("issuer native response differs")
    markdown = native["markdown"]["raw_markdown"]
    rows = [
        [cell.strip() for cell in row.strip().strip("|").split("|")]
        for row in markdown.splitlines()
        if re.match(r"^\|\s*BITO\*\s*\|", row)
    ]
    if rows != [["BITO*", "ProShares Bitcoin ETF", "1:2", "74347G440", "74350X405"]]:
        raise ValueError("issuer BITO identity or split differs")
    if (
        "reverse splits will be effective prior to market open on November 19, 2026"
        not in markdown
        or "fractional shares will be redeemed for cash" not in markdown
    ):
        raise ValueError("issuer effective date or fractional treatment differs")

    conversion = Decimal("0.5")
    old_units = Decimal("10")
    old_price = Decimal("20")
    new_units = old_units * conversion
    new_price = old_price / conversion
    later_price = Decimal("36")
    prior_isolated_margin, new_initial_margin = Decimal("30"), Decimal("20")
    return {
        "issuer_instrument": {
            "ticker": "BITO",
            "old_cusip": rows[0][3],
            "new_cusip": rows[0][4],
            "share_quantity_factor": str(conversion),
            "announced_effective": "before market open on 2026-11-19",
            "issuer_fractional_shares_redeemed_in_cash": True,
        },
        "exchange_schedule_utc_as_announced": {
            "strategy_bot_termination": "2026-11-19T00:45:00Z",
            "halt_start": "2026-11-19T01:00:00Z",
            "nominal_post_only_start": "2026-11-19T03:25:00Z",
            "nominal_continuous_resume": "2026-11-19T03:30:00Z",
            "nominal_end_of_last_price_protection": "2026-11-19T14:30:00Z",
        },
        "constructed": {
            "pre_split_share_value": str(old_units * old_price),
            "equivalent_post_split_share_value": str(new_units * new_price),
            "unhedged_long_value_change": str(new_units * (later_price - new_price)),
            "isolated_reserve_relocation_if_old_30_and_new_20": str(
                prior_isolated_margin - new_initial_margin
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exchange", type=Path)
    parser.add_argument("issuer", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.exchange, args.issuer), indent=2))
