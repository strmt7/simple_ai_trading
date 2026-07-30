from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.binance_archive import (
    ArchiveListingItem,
    archive_file_url,
)
from simple_ai_trading.polymarket_btc_history import (
    build_polymarket_btc_history_inventory,
    load_polymarket_btc_history_contract,
)


ROOT = Path(__file__).parents[1]
INGESTION_STATUS = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-015-btc-5m-history-ingestion-status.json"
)


def _item(market_type: str, period: str, index: int) -> ArchiveListingItem:
    url = archive_file_url(
        symbol="BTCUSDT",
        interval="1s",
        period=period,
        market_type=market_type,
        cadence="daily",
        data_type="aggTrades",
    )
    return ArchiveListingItem(
        url=url,
        key=Path(url).name,
        period=period,
        size_bytes=1_000 + index,
        last_modified="2026-02-14T00:00:00+00:00",
        etag=f"{index + 1:032x}",
        checksum_size_bytes=99,
        checksum_last_modified="2026-02-14T00:00:01+00:00",
        checksum_etag=f"{index + 11:032x}",
    )


def test_btc_history_inventory_is_contiguous_target_blind_and_loadable(
    tmp_path: Path,
) -> None:
    periods = ("2026-02-12", "2026-02-13")
    listings = {
        market_type: tuple(
            _item(market_type, period, market_index * 10 + period_index)
            for period_index, period in enumerate(periods)
        )
        for market_index, market_type in enumerate(("spot", "futures"))
    }
    artifact = build_polymarket_btc_history_inventory(
        listings,
        first_day=periods[0],
        last_day=periods[-1],
        observed_at_utc=datetime(2026, 2, 14, tzinfo=UTC).isoformat(),
    )
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = load_polymarket_btc_history_contract(path)

    assert artifact["range"]["day_count"] == 2
    assert artifact["source_count"] == 4
    assert artifact["storage_policy"]["raw_archive_retained"] is False
    assert artifact["training_authority"] is False
    assert artifact["trading_authority"] is False
    assert contract.symbols == ("BTCUSDT",)
    assert contract.expected_files == 4
    assert contract.expected_rows == 2 * 86_400
    assert tuple(day.period for day in contract.days) == periods


def test_btc_history_inventory_rejects_any_missing_official_day() -> None:
    listings = {
        "spot": (
            _item("spot", "2026-02-12", 0),
            _item("spot", "2026-02-13", 1),
        ),
        "futures": (_item("futures", "2026-02-12", 2),),
    }
    with pytest.raises(ValueError, match="misses 1 days"):
        build_polymarket_btc_history_inventory(
            listings,
            first_day="2026-02-12",
            last_day="2026-02-13",
            observed_at_utc="2026-02-14T00:00:00Z",
        )


def test_tracked_ingestion_status_is_integral_and_non_authoritative() -> None:
    status = json.loads(INGESTION_STATUS.read_text(encoding="utf-8"))
    claimed = status.pop("artifact_sha256")
    canonical = json.dumps(
        status,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    assert hashlib.sha256(canonical).hexdigest() == claimed
    assert status["completed_days"] + status["remaining_days"] == status[
        "expected_days"
    ]
    assert status["flow_rows"] == status["completed_days"] * 86_400
    assert status["archive_cache_files"] == 0
    assert status["raw_archive_retained"] is False
    assert status["dataset_frozen"] is False
    assert status["model_training_started"] is False
    assert status["test_targets_accessed"] is False
    assert status["profitability_claim"] is False
    assert status["paper_authority"] is False
    assert status["live_authority"] is False
