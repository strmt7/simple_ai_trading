from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.binance_archive import ArchiveListingItem, archive_file_url
from tools.build_round72_spot_perpetual_inventory import (
    MARKET_TYPES,
    SCHEMA_VERSION,
    SYMBOLS,
    _canonical_sha256,
    build_inventory_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "docs/model-research/action-value/round-072-spot-perpetual-price-discovery-design.json"
INVENTORY_PATH = ROOT / "docs/model-research/action-value/round-072-spot-perpetual-inventory.json"


def _item(market_type: str, symbol: str, day: str) -> ArchiveListingItem:
    return ArchiveListingItem(
        url=archive_file_url(
            symbol=symbol,
            interval="1s",
            period=day,
            market_type=market_type,
            cadence="daily",
            data_type="aggTrades",
        ),
        key=f"key/{market_type}/{symbol}/{day}",
        period=day,
        size_bytes=100 + int(hashlib.sha256(f"{market_type}:{symbol}:{day}".encode()).hexdigest()[:4], 16),
        last_modified=f"{day}T01:00:00.000Z",
        etag="a" * 32,
        checksum_size_bytes=96,
        checksum_last_modified=f"{day}T01:00:01.000Z",
        checksum_etag="b" * 32,
    )


def _listings(start: date, end: date):
    output = {
        (market_type, symbol): []
        for market_type in MARKET_TYPES
        for symbol in SYMBOLS
    }
    cursor = start
    while cursor <= end:
        day = cursor.isoformat()
        for market_type, symbol in output:
            output[(market_type, symbol)].append(_item(market_type, symbol, day))
        cursor += timedelta(days=1)
    return output


def test_round72_inventory_selection_is_deterministic_and_return_independent() -> None:
    listings = _listings(date(2024, 1, 1), date(2024, 2, 29))
    first = build_inventory_artifact(
        listings,
        observed_at_utc="2026-07-22T00:00:00+00:00",
        start_month="2024-01",
        end_month="2024-02",
        selection_seed="frozen-seed",
        minimum_complete_months=2,
    )
    reversed_listings = {
        key: list(reversed(value)) for key, value in listings.items()
    }
    second = build_inventory_artifact(
        reversed_listings,
        observed_at_utc="2026-07-22T00:00:00+00:00",
        start_month="2024-01",
        end_month="2024-02",
        selection_seed="frozen-seed",
        minimum_complete_months=2,
    )

    assert first == second
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["complete_months"] == 2
    assert first["selected_files"] == 12
    assert first["price_or_return_data_used_for_selection"] is False
    assert first["inventory_sha256"] == _canonical_sha256(
        {key: value for key, value in first.items() if key != "inventory_sha256"}
    )
    for month in first["selected_months"]:
        assert len(month["files"]) == 6
        assert {value["period"] for value in month["files"]} == {
            month["selected_day"]
        }


def test_round72_inventory_excludes_any_incomplete_six_stream_month() -> None:
    listings = _listings(date(2024, 1, 1), date(2024, 2, 29))
    listings[("spot", "ETHUSDT")] = [
        item
        for item in listings[("spot", "ETHUSDT")]
        if item.period != "2024-01-17"
    ]

    artifact = build_inventory_artifact(
        listings,
        observed_at_utc="2026-07-22T00:00:00+00:00",
        start_month="2024-01",
        end_month="2024-02",
        selection_seed="frozen-seed",
        minimum_complete_months=1,
    )

    assert artifact["complete_months"] == 1
    assert artifact["selected_months"][0]["month"] == "2024-02"
    assert artifact["excluded_months"] == [
        {
            "month": "2024-01",
            "reason": "incomplete_six_stream_calendar_month",
            "missing_by_stream": {"spot:ETHUSDT": ["2024-01-17"]},
        }
    ]


def test_round72_inventory_fails_closed_on_bad_metadata_and_insufficient_months() -> None:
    listings = _listings(date(2024, 1, 1), date(2024, 1, 31))
    bad = listings[("futures", "BTCUSDT")][0]
    listings[("futures", "BTCUSDT")][0] = ArchiveListingItem(
        **{**bad.asdict(), "checksum_size_bytes": 0}
    )

    with pytest.raises(ValueError, match="object size is missing"):
        build_inventory_artifact(
            listings,
            observed_at_utc="2026-07-22T00:00:00+00:00",
            start_month="2024-01",
            end_month="2024-01",
            selection_seed="frozen-seed",
            minimum_complete_months=1,
        )

    complete = _listings(date(2024, 1, 1), date(2024, 1, 31))
    with pytest.raises(ValueError, match="too few complete"):
        build_inventory_artifact(
            complete,
            observed_at_utc="2026-07-22T00:00:00+00:00",
            start_month="2024-01",
            end_month="2024-01",
            selection_seed="frozen-seed",
            minimum_complete_months=2,
        )


def test_round72_frozen_design_binds_the_real_inventory() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    canonical_design = dict(design)
    design_sha256 = canonical_design.pop("design_sha256")
    canonical_inventory = dict(inventory)
    inventory_sha256 = canonical_inventory.pop("inventory_sha256")
    source = design["source_contract"]

    assert design_sha256 == _canonical_sha256(canonical_design)
    assert inventory_sha256 == _canonical_sha256(canonical_inventory)
    assert source["inventory_canonical_sha256"] == inventory_sha256
    assert source["inventory_file_sha256"] == hashlib.sha256(
        INVENTORY_PATH.read_bytes()
    ).hexdigest()
    assert source["complete_months"] == inventory["complete_months"] == 69
    assert source["selected_files"] == inventory["selected_files"] == 414
    assert (
        source["selected_compressed_bytes"]
        == inventory["selected_compressed_bytes"]
        == 5_964_131_852
    )
    assert design["governance"]["profitability_claim_permitted"] is False
    assert design["governance"]["trading_authority_permitted"] is False
    assert design["feature_contract"]["crypto_market_close_feature"] is False
