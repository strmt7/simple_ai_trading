"""Reconcile the exact retained Samsung issuer and Binance CMS source bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re


_ISSUER_URL = "https://www.samsung.com/global/ir/reports-disclosures/public-disclosure-view.84796/"
_ARTICLES = {
    "estimate": "fd194d0fda6e44f2af6bc6333f61600f",
    "reconciliation": "c88b113255244de1953dd416c4806bc3",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text(node: dict) -> str:
    if node.get("node") == "text":
        return node["text"]
    return "".join(_text(child) for child in node.get("child", []))


def _article(directory: Path, name: str) -> tuple[str, dict]:
    result = _json(directory / f"{name}.result.json")
    receipt = result["capture"]["receipt"]
    native_path = directory / f"{name}.raw.json"
    if (
        result["source_gate"]["passed"] is not True
        or receipt["status_code"] != 200
        or receipt["response_sha256"] != _hash(native_path)
        or receipt["response_bytes"] != native_path.stat().st_size
    ):
        raise ValueError(f"{name} source receipt failed")
    native = _json(native_path)
    article = native["data"]
    if (
        native["success"] is not True
        or native["code"] != "000000"
        or article["code"] != _ARTICLES[name]
        or "Dividend Adjustment Process of SAMSUNGUSDT" not in article["title"]
    ):
        raise ValueError(f"{name} article identity differs")
    tree = json.loads(article["body"])
    if tree.get("node") != "root":
        raise ValueError("Binance CMS root format changed")
    text = unescape("\n".join(_text(node) for node in tree["child"]))
    return text, {
        "article_code": article["code"],
        "article_id": article["id"],
        "title": article["title"],
        "reported_publish_time_utc": datetime.fromtimestamp(
            article["publishDate"] / 1_000, timezone.utc
        ).isoformat(),
        "api_version_field": article["version"],
        "api_last_update_field": article["lastUpdateTime"],
        "native_sha256": receipt["response_sha256"],
        "native_bytes": receipt["response_bytes"],
    }


def _number(pattern: str, text: str) -> Decimal:
    values = re.findall(pattern, text, re.IGNORECASE)
    if len(values) != 1:
        raise ValueError(f"expected one source value for {pattern!r}")
    return Decimal(values[0].replace(",", ""))


def reconcile(directory: Path) -> dict:
    estimate, estimate_receipt = _article(directory, "estimate")
    later, later_receipt = _article(directory, "reconciliation")
    issuer_path = directory / "issuer.native.json"
    issuer_receipt = _json(directory / "result.json")
    issuer = _json(issuer_path)
    if (
        issuer_receipt["source_admitted"] is not True
        or issuer_receipt["native_sha256"] != _hash(issuer_path)
        or issuer_receipt["native_bytes"] != issuer_path.stat().st_size
        or issuer["url"] != _ISSUER_URL
    ):
        raise ValueError("issuer source receipt failed")
    issuer_text = issuer["markdown"]["raw_markdown"]
    required = (
        (estimate, "2026-09-29 08:00 (Korea Standard Time)"),
        (estimate, "will not be capped by funding cap/floor"),
        (later, "2026-09-28 23:00 (UTC)"),
        (later, "2026-11-06 06:00 (UTC)"),
        (later, "Special Funding Fee"),
        (later, "funding interval will remain unchanged"),
        (issuer_text, "Quarterly Dividend for Third Quarter 2026"),
        (issuer_text, "Oct 29. 2026"),
        (issuer_text, "Record date: September 30, 2026"),
        (issuer_text, "Scheduled payment date: November 27, 2026"),
    )
    if any(marker not in source for source, marker in required):
        raise ValueError("a required attribution or chronology field differs")
    first = _number(
        r"estimated dividend adjustment amount for SAMSUNGUSDT is ([\d,]+) won per share",
        estimate,
    )
    adjustment = _number(r"will reconcile ([\d,]+) won per share", later)
    adjusted_pct = _number(r"equivalent to (-\d+\.\d+)%", later)
    declared = _number(r"Common shares: KRW ([\d,]+)", issuer_text)
    snapshot = datetime.fromisoformat("2026-09-29T08:00:00+09:00").astimezone(
        timezone.utc
    )
    scheduled = datetime.fromisoformat("2026-11-06T06:00:00+00:00")
    delay = scheduled - snapshot
    if (
        first + adjustment != declared
        or snapshot.isoformat() != "2026-09-28T23:00:00+00:00"
    ):
        raise ValueError("issuer and exchange terms do not reconcile")

    def nominal_adjustment(snapshot_share_units: Decimal) -> str:
        return str(snapshot_share_units * adjustment)

    return {
        "issuer": {
            "url": _ISSUER_URL,
            "reported_approval_date": "2026-10-29",
            "record_date": "2026-09-30",
            "scheduled_payment_date": "2026-11-27",
            "common_share_amount_krw": str(declared),
            "native_sha256": issuer_receipt["native_sha256"],
            "native_bytes": issuer_receipt["native_bytes"],
        },
        "exchange": {
            "estimate_article": estimate_receipt,
            "reconciliation_article": later_receipt,
            "entitlement_snapshot_utc": snapshot.isoformat(),
            "announced_reconciliation_start_utc": scheduled.isoformat(),
            "snapshot_to_reconciliation_start_hours": str(
                (Decimal(delay.days) * 86_400 + delay.seconds) / 3_600
            ),
            "estimated_krw_per_share": str(first),
            "reconciliation_krw_per_share": str(adjustment),
            "announced_reconciliation_percent": str(adjusted_pct),
            "combined_nominal_krw_per_share": str(first + adjustment),
            "ordinary_funding_cap_applied_to_dividend_adjustment": False,
        },
        "constructed_unsigned_krw_examples": {
            "ten_snapshot_units_then_flat": nominal_adjustment(Decimal(10)),
            "flat_snapshot_then_ten_units": nominal_adjustment(Decimal(0)),
        },
        "actual_cash_currency_fx_sign_and_wallet_posting_qualified": False,
        "cross_venue_cash_equivalence_qualified": False,
        "new_economic_observations": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retained_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.retained_directory), indent=2, sort_keys=True))
