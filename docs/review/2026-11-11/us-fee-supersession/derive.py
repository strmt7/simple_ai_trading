"""Reconcile retained US fee sources and exact constructed fill accounting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo


SOURCES = {
    "changelog": (
        "native-result.json",
        "https://docs.polymarket.us/changelog",
        "Changelog - Polymarket US Documentation",
        "9912b4a2ee599c9ef47192aac66377fa6f917759cd407c890b2a52340f8b8ffe",
    ),
    "fees": (
        "fees/native-result.json",
        "https://docs.polymarket.us/fees",
        "Fee Schedule - Polymarket US Documentation",
        "eaf057d31a543cd69d66978cb914ecf7865f0e88bb2008063ff3881ced1c6fe9",
    ),
}
TAKER_COEFFICIENTS = ("0.095", "0.085", "0.075", "0.0695")
MAKER_REBATE_COEFFICIENTS = ("0", "0.0035", "0.0070", "0.0104")


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def derive(directory: Path) -> dict:
    sources = {}
    texts = {}
    for role, (relative, url, title, expected) in SOURCES.items():
        body = (directory / relative).read_bytes()
        if sha256(body).hexdigest() != expected:
            raise ValueError("Retained native source digest differs")
        native = json.loads(body)
        if (
            native["url"] != url
            or native["redirected_url"] != url
            or native["status_code"] != 200
            or native["redirected_status_code"] != 200
            or native["success"] is not True
            or native["error_message"]
            or native["metadata"]["title"] != title
            or native["cache_status"] != "miss"
        ):
            raise ValueError("Native source admission differs")
        text = native["markdown"]["raw_markdown"]
        if len(text) < 1_000:
            raise ValueError("Native source text is incomplete")
        texts[role] = text
        sources[role] = {
            "url": url,
            "status_code": 200,
            "redirected_status_code": 200,
            "success": True,
            "error_message": "",
            "title": title,
            "native_bytes": len(body),
            "native_sha256": expected,
            "markdown_characters": len(text),
            "markdown_sha256": sha256(text.encode("utf-8")).hexdigest(),
            "cache_status": native["cache_status"],
        }
    required = (
        "30 completed Eastern Time",
        "Below $10,000,000",
        "No maker rebate",
        "Maker rebates are computed per fill, independently.",
        "commission collected across the order",
        "The adjustment can only reduce",
        "November 11, 2026",
        "November 14, 2026",
        "0.0695 × (1 - p) + 0.085 × (1 - p)^4",
    )
    if not all(phrase in texts["fees"] for phrase in required):
        raise ValueError("Selected fee terms are absent")
    if "October 19, 2026" not in texts["changelog"]:
        raise ValueError("Independent rebate supersession trigger is absent")
    if not all(
        coefficient in texts["fees"]
        for coefficient in TAKER_COEFFICIENTS + MAKER_REBATE_COEFFICIENTS[1:]
    ):
        raise ValueError("Selected fee coefficients are absent")

    count, price = Decimal("100"), Decimal("0.50")
    uncertainty = count * price * (1 - price)
    taker = {
        str(tier): money(Decimal(coefficient) * uncertainty)
        for tier, coefficient in enumerate(TAKER_COEFFICIENTS, 1)
    }
    maker = {
        str(tier): money(Decimal(coefficient) * uncertainty)
        for tier, coefficient in enumerate(MAKER_REBATE_COEFFICIENTS, 1)
    }
    current_combo = (
        count
        * price
        * (Decimal("0.0695") * (1 - price) + Decimal("0.085") * (1 - price) ** 4)
    )
    future_combo = {
        str(tier): money(
            count
            * price
            * (Decimal(coefficient) * (1 - price) + Decimal("0.06") * (1 - price) ** 4)
        )
        for tier, coefficient in enumerate(TAKER_COEFFICIENTS, 1)
    }
    fill_price = Decimal("0.10")
    fill_uncertainty = count * fill_price * (1 - fill_price)
    exact_taker_fill = Decimal("0.0695") * fill_uncertainty
    exact_maker_fill = Decimal("0.0104") * fill_uncertainty
    new_york = ZoneInfo("America/New_York")
    times = {
        "combo_taker_announced_effective_utc": datetime(
            2026, 11, 11, 13, tzinfo=new_york
        )
        .astimezone(timezone.utc)
        .isoformat(),
        "combo_maker_announced_effective_utc": datetime(2026, 11, 14, tzinfo=new_york)
        .astimezone(timezone.utc)
        .isoformat(),
    }
    return {
        "sources": sources,
        "standard_taker_coefficients_by_tier": list(TAKER_COEFFICIENTS),
        "standard_maker_rebate_magnitudes_by_tier": list(MAKER_REBATE_COEFFICIENTS),
        "announced_times": times,
        "constructed_100_contracts_at_0_50": {
            "standard_taker_usd_by_tier": taker,
            "standard_maker_rebate_usd_by_tier": maker,
            "historical_flat_maker_rebate_usd": money(Decimal("0.0125") * uncertainty),
            "combo_taker_pre_change_usd": money(current_combo),
            "combo_taker_post_change_usd_by_tier": future_combo,
            "combo_maker_announced_post_change_usd": money(
                Decimal("0.03") * uncertainty
            ),
        },
        "constructed_two_100_contract_fills_at_0_10": {
            "taker_exact_usd_per_fill": str(exact_taker_fill),
            "unadjusted_rounded_taker_usd_sum": str(
                2 * Decimal(money(exact_taker_fill))
            ),
            "rounded_cumulative_exact_taker_usd_ceiling": money(2 * exact_taker_fill),
            "minimum_required_taker_adjustment_usd": str(
                2 * Decimal(money(exact_taker_fill))
                - Decimal(money(2 * exact_taker_fill))
            ),
            "maker_rebate_usd_per_fill_sum": str(2 * Decimal(money(exact_maker_fill))),
            "incorrect_coalesced_maker_rebate_usd": money(2 * exact_maker_fill),
            "actual_taker_adjustment_algorithm_or_owned_receipts_proved": False,
        },
        "rounding_bound": {
            "absolute_usd_per_eligible_maker_fill": "0.005",
            "aggregate_absolute_usd_bound": "0.005 times eligible maker fill count",
            "historical_rounded_taker_share_grid_applied_to_new_schedule": False,
        },
        "account_eligibility_or_assigned_coefficients_proved": False,
        "actual_fee_activation_or_owned_receipt_proved": False,
        "new_orders_or_economic_observations": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.retained_directory), sort_keys=True, indent=2))
