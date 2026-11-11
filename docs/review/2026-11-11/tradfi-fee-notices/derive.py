"""Reconcile the retained Binance TradFi fee notice versions."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re


ARTICLES = {
    "original": (
        288534,
        "4f406308840549ffb8fdaac9873c25a8",
        "dee25b61d003d8564e6c431198db27de98b98048061899f22833366ced301d91",
    ),
    "superseding": (
        288946,
        "e3424d3a781841eb8b507bc694c0ca3a",
        "bca195e4d75f6ab9ccb73570807166cff0e31b16d50c5e6ac99f918e3370b6f6",
    ),
}
TIERS = ("Regular User", *(f"VIP {tier}" for tier in range(1, 10)))


def _text(node: dict) -> str:
    if node.get("node") == "text":
        return node["text"]
    return "".join(_text(child) for child in node.get("child", []))


def _rows(node: dict):
    if node.get("tag") == "tr":
        yield tuple(_text(cell).strip() for cell in node["child"])
        return
    for child in node.get("child", []):
        yield from _rows(child)


def _percent(value: str) -> Decimal:
    if re.fullmatch(r"\d+\.\d{4}%", value) is None:
        raise ValueError("native table rate format differs")
    return Decimal(value[:-1])


def _utc(milliseconds: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        milliseconds=milliseconds
    )


def _read(directory: Path, role: str) -> tuple[dict, str, list[tuple[Decimal, ...]]]:
    article_id, code, digest = ARTICLES[role]
    raw = (directory / f"{role}.raw.json").read_bytes()
    result = json.loads((directory / f"{role}.result.json").read_bytes())
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
    rows = list(_rows(tree))
    if tree.get("node") != "root" or tuple(row[0] for row in rows[1:]) != TIERS:
        raise ValueError(f"{role} native fee table differs")
    if rows[0] != (
        "VIP Level",
        "Maker Fee",
        "Taker Fee",
        "Taker Fee (With 10% Discount When Paying With BNB)",
    ) or any(len(row) != 4 for row in rows):
        raise ValueError(f"{role} native fee column scope differs")
    return (
        article,
        " ".join(unescape(_text(tree)).split()),
        [tuple(_percent(rate) for rate in row[1:]) for row in rows[1:]],
    )


def reconcile(directory: Path) -> dict:
    original, old_body, old_rates = _read(directory, "original")
    current, body, rates = _read(directory, "superseding")
    planned = datetime(2026, 11, 5, 8, tzinfo=timezone.utc)
    publication = _utc(current["publishDate"])
    if (
        "has been superseded by the announcement published on 2026-11-04"
        not in old_body
        or "That change will not be implemented." not in body
        or "This notice supersedes the announcement of 2026-10-28." not in body
        or "The promotion announced on 2026-03-31 therefore continues" not in body
        or "2026-11-05 08:00 (UTC)" not in old_body
        or "2026-11-05 08:00 (UTC)" not in body
        or not _utc(original["publishDate"]) < publication < planned
    ):
        raise ValueError("publisher supersession or timeline differs")
    results = {}
    for tier, previous, retained in zip(TIERS, old_rates, rates, strict=True):
        if previous[0] != 0 or retained[0] != 0:
            raise ValueError("maker schedule differs")
        delta_bps = (previous[1] - retained[1]) * 100
        results[tier] = {
            "cancelled_taker_fee_bps": format(previous[1] * 100, ".2f"),
            "retained_taker_fee_bps": format(retained[1] * 100, ".2f"),
            "retained_bnb_taker_fee_bps_as_displayed": format(retained[2] * 100, ".2f"),
            "cancelled_increase_bps_per_side": format(delta_bps, ".2f"),
            "cancelled_increase_bps_two_equal_notionals": format(2 * delta_bps, ".2f"),
        }
    return {
        "original_publish_utc": _utc(original["publishDate"]).isoformat(),
        "original_reported_update_utc": _utc(original["lastUpdateTime"]).isoformat(),
        "superseding_publish_utc": publication.isoformat(),
        "cancelled_planned_effective_utc": planned.isoformat(),
        "superseding_notice_replaces_prospective_fee_change": True,
        "displayed_maker_fee_bps": "0.00",
        "tiers": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_directory", type=Path)
    print(json.dumps(reconcile(parser.parse_args().receipt_directory), indent=2))
