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
from tools.ingest_polymarket_btc_history import (
    DEFAULT_STATUS_OUTPUT,
    _status_artifact,
    _validate_status_destination,
    _write_status_artifact,
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


def test_ingestion_status_publisher_is_atomic_hashed_and_non_authoritative(
    tmp_path: Path,
) -> None:
    database = ROOT / "data" / f"{tmp_path.name}-history-status.duckdb"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"database")
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "status.json"
    try:
        artifact = _status_artifact(
            status={
                "expected_days": 154,
                "completed_days": 3,
                "remaining_days": 151,
                "flow_rows": 259_200,
                "compressed_source_bytes": 1234,
                "first_completed_day": "2026-02-12",
                "last_completed_day": "2026-02-14",
                "raw_archive_retained": False,
            },
            latest_batch=(
                {
                    "period": "2026-02-12",
                    "source_count": 2,
                    "flow_rows": 86_400,
                    "compressed_bytes": 400,
                    "combined_flow_sha256": "1" * 64,
                },
                {
                    "period": "2026-02-13",
                    "source_count": 2,
                    "flow_rows": 86_400,
                    "compressed_bytes": 410,
                    "combined_flow_sha256": "2" * 64,
                },
                {
                    "period": "2026-02-14",
                    "source_count": 2,
                    "flow_rows": 86_400,
                    "compressed_bytes": 424,
                    "combined_flow_sha256": "3" * 64,
                },
            ),
            inventory_path=ROOT
            / "docs"
            / "model-research"
            / "polymarket"
            / "round-015-btc-5m-full-history-inventory-v1.json",
            inventory_sha256="4" * 64,
            database_path=database,
            cache_root=cache,
            generated_at_utc="2026-07-30T03:00:00Z",
        )
        _write_status_artifact(output, artifact)
        restored = json.loads(output.read_text(encoding="utf-8"))
    finally:
        database.unlink(missing_ok=True)

    claimed = restored.pop("artifact_sha256")
    canonical = json.dumps(
        restored,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == claimed
    assert restored["completed_days"] == 3
    assert restored["latest_batch"][-1]["period"] == "2026-02-14"
    assert restored["database_bytes"] == 8
    assert restored["archive_cache_files"] == 0
    assert restored["dataset_frozen"] is False
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_ingestion_status_publisher_rejects_retained_archive(
    tmp_path: Path,
) -> None:
    database = ROOT / "data" / f"{tmp_path.name}-history-cache.duckdb"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"database")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stale.zip").write_bytes(b"archive")
    try:
        with pytest.raises(ValueError, match="archive cache is not empty"):
            _status_artifact(
                status={
                    "expected_days": 1,
                    "completed_days": 1,
                    "remaining_days": 0,
                    "flow_rows": 86_400,
                    "compressed_source_bytes": 10,
                    "first_completed_day": "2026-02-12",
                    "last_completed_day": "2026-02-12",
                },
                latest_batch=(
                    {
                        "period": "2026-02-12",
                        "source_count": 2,
                        "flow_rows": 86_400,
                        "compressed_bytes": 10,
                        "combined_flow_sha256": "5" * 64,
                    },
                ),
                inventory_path=ROOT
                / "docs"
                / "model-research"
                / "polymarket"
                / "round-015-btc-5m-full-history-inventory-v1.json",
                inventory_sha256="6" * 64,
                database_path=database,
                cache_root=cache,
                generated_at_utc="2026-07-30T03:00:00Z",
            )
    finally:
        database.unlink(missing_ok=True)


def test_ingestion_status_publisher_rejects_inconsistent_evidence(
    tmp_path: Path,
) -> None:
    database = ROOT / "data" / f"{tmp_path.name}-history-invalid.duckdb"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"database")
    cache = tmp_path / "cache"
    cache.mkdir()
    base_status = {
        "expected_days": 2,
        "completed_days": 2,
        "remaining_days": 0,
        "flow_rows": 172_800,
        "compressed_source_bytes": 20,
        "first_completed_day": "2026-02-12",
        "last_completed_day": "2026-02-13",
    }
    base_latest = (
        {
            "period": "2026-02-12",
            "source_count": 2,
            "flow_rows": 86_400,
            "compressed_bytes": 10,
            "combined_flow_sha256": "7" * 64,
        },
        {
            "period": "2026-02-13",
            "source_count": 2,
            "flow_rows": 86_400,
            "compressed_bytes": 10,
            "combined_flow_sha256": "8" * 64,
        },
    )

    def build(
        *,
        status: dict[str, object] | None = None,
        latest: tuple[dict[str, object], ...] | None = None,
        inventory: Path | None = None,
    ) -> dict[str, object]:
        return _status_artifact(
            status=base_status if status is None else status,
            latest_batch=base_latest if latest is None else latest,
            inventory_path=(
                ROOT
                / "docs"
                / "model-research"
                / "polymarket"
                / "round-015-btc-5m-full-history-inventory-v1.json"
                if inventory is None
                else inventory
            ),
            inventory_sha256="9" * 64,
            database_path=database,
            cache_root=cache,
            generated_at_utc="2026-07-30T03:00:00Z",
        )

    try:
        with pytest.raises(ValueError, match="day arithmetic"):
            build(status={**base_status, "remaining_days": 1})
        with pytest.raises(ValueError, match="row arithmetic"):
            build(status={**base_status, "flow_rows": 1})
        with pytest.raises(ValueError, match="batch count differs"):
            build(latest=())
        with pytest.raises(ValueError, match="batch count differs"):
            build(latest=base_latest + base_latest)
        with pytest.raises(ValueError, match="chronology differs"):
            build(latest=tuple(reversed(base_latest)))
        with pytest.raises(ValueError, match="manifest differs"):
            build(
                latest=(
                    {
                        **base_latest[0],
                        "combined_flow_sha256": "not-a-digest",
                    },
                    base_latest[1],
                )
            )
        with pytest.raises(ValueError, match="does not reach status end"):
            build(
                status={**base_status, "last_completed_day": "2026-02-14"}
            )
        with pytest.raises(ValueError, match="must remain inside"):
            build(inventory=tmp_path / "outside-inventory.json")

        database.write_bytes(b"")
        with pytest.raises(ValueError, match="database is empty"):
            build()
    finally:
        database.unlink(missing_ok=True)


def test_ingestion_status_writer_rejects_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "status.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="cannot be a symlink"):
        _write_status_artifact(link, {"artifact_sha256": "a" * 64})


def test_canonical_status_rejects_custom_data_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires canonical"):
        _validate_status_destination(
            output=DEFAULT_STATUS_OUTPUT,
            inventory=tmp_path / "inventory.json",
            database=ROOT / "data" / "custom.duckdb",
            cache_root=ROOT / "data" / "custom-cache",
        )

    _validate_status_destination(
        output=tmp_path / "custom-status.json",
        inventory=tmp_path / "inventory.json",
        database=tmp_path / "custom.duckdb",
        cache_root=tmp_path / "custom-cache",
    )
