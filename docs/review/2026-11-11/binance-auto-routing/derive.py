"""Reconcile Binance Spot routing terms and compare same-unit proceeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from html import unescape
import json
from pathlib import Path


_ARTICLES = {
    "announcement": (
        286495,
        "2f6fa2926253415a81def145af35014e",
        "66e6f68ac98fe69f1fb5e84765abd74f2a063e4ae231a26d5418ac94f9fa3bb6",
    ),
    "faq": (
        283263,
        "1dce8dc66adc450a8e669676d64a5135",
        "286d58716b5dff60422d12b98bbc694323148c81cc07141ed5a955d200de5b61",
    ),
}


def _text(node: dict) -> str:
    if node.get("node") == "text":
        return node["text"]
    return "".join(_text(child) for child in node.get("child", []))


def _utc(milliseconds: int) -> str:
    return (
        datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=milliseconds)
    ).isoformat()


def _article(directory: Path, role: str) -> tuple[str, dict]:
    article_id, code, digest = _ARTICLES[role]
    raw = (directory / "source.raw.json").read_bytes()
    result = json.loads((directory / "source.result.json").read_bytes())
    receipt = result["capture"]["receipt"]
    if (
        result["source_gate"]["passed"] is not True
        or receipt["status_code"] != 200
        or len(raw) != receipt["response_bytes"]
        or sha256(raw).hexdigest() != digest
        or receipt["response_sha256"] != digest
    ):
        raise ValueError(f"{role} native receipt differs")
    response = json.loads(raw)
    article = response["data"]
    if (
        response["success"] is not True
        or response["code"] != "000000"
        or (article["id"], article["code"], article["version"])
        != (article_id, code, "1")
    ):
        raise ValueError(f"{role} publisher identity differs")
    tree = json.loads(article["body"])
    if tree.get("node") != "root":
        raise ValueError(f"{role} body format differs")
    return " ".join(unescape(_text(tree)).split()), {
        "id": article_id,
        "code": code,
        "version": article["version"],
        "publish_utc": _utc(article["publishDate"]),
        "last_update_utc": _utc(article["lastUpdateTime"])
        if article["lastUpdateTime"]
        else None,
        "response_bytes": len(raw),
        "response_sha256": digest,
    }


def derive(announcement: Path, faq: Path) -> dict:
    announcement_text, publisher_notice = _article(announcement, "announcement")
    faq_text, publisher_faq = _article(faq, "faq")
    for text, phrases in (
        (
            announcement_text,
            (
                "U-quoted symbols now support Auto Order Routing",
                "deducted from the receiving asset of the executed pair",
                "No commission is charged by Binance on the intermediary quote currency conversion",
                "standard spreads will apply",
            ),
        ),
        (
            faq_text,
            (
                "Routing only applies to the portion of your order that fills immediately",
                "the remaining portion stays on your original pair",
                "the synthetic market data shown on the trading interface is indicative",
                "does not guarantee a minimum conversion amount",
                "Conversion rate used may not be at the prevailing Binance Spot rate",
                "U-quoted pairsUSD-quoted pairsUSD1-quoted pairsUSDC-quoted pairsBRL-quoted pairs",
                "original traded prices and the applied conversion rates",
            ),
        ),
    ):
        if any(phrase not in text for phrase in phrases):
            raise ValueError("publisher routing semantics differ")

    direct_gross_u, routed_gross_usdt = Decimal("1000"), Decimal("1001")
    one_fee = Decimal("0.001")
    conversion_u_per_usdt = Decimal("0.9985")
    direct_net_u = direct_gross_u * (1 - one_fee)
    routed_net_usdt = routed_gross_usdt * (1 - one_fee)
    routed_net_u = routed_net_usdt * conversion_u_per_usdt
    manual_net_u = routed_net_u * (1 - one_fee)
    parity = Fraction(direct_net_u) / Fraction(routed_net_usdt)
    return {
        "announcement": publisher_notice,
        "faq": publisher_faq,
        "constructed": {
            "direct_gross_u": str(direct_gross_u),
            "intermediary_gross_usdt": str(routed_gross_usdt),
            "illustrative_fee_fraction_per_commission": str(one_fee),
            "executed_conversion_u_per_usdt": str(conversion_u_per_usdt),
            "parity_conversion_u_per_usdt_exact": str(parity),
            "direct_net_u": str(direct_net_u),
            "automatic_route_net_u": str(routed_net_u),
            "manual_route_net_u_at_same_conversion": str(manual_net_u),
            "automatic_minus_direct_u": str(routed_net_u - direct_net_u),
            "automatic_minus_manual_u": str(routed_net_u - manual_net_u),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("announcement", type=Path)
    parser.add_argument("faq", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.announcement, args.faq), indent=2))
