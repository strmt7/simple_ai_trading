from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
import zipfile

import pytest

from simple_ai_trading import microstructure_warehouse as warehouse_module
from simple_ai_trading.microstructure_warehouse import (
    MicrostructureWarehouse,
    official_tick_archive_url,
)
from simple_ai_trading.tick_warehouse_merge import merge_certified_tick_warehouse


_ZIP_ETAG = "a" * 32
_CHECKSUM_ETAG = "b" * 32
_CHECKSUM_BYTES = 100


def _inventory_item(*, period: str, size_bytes: int) -> dict[str, object]:
    return {
        "period": period,
        "url": official_tick_archive_url(
            symbol="BTCUSDT",
            data_type="trades",
            period=period,
        ),
        "size_bytes": size_bytes,
        "last_modified": "2026-07-10T00:00:00Z",
        "etag": _ZIP_ETAG,
        "checksum_size_bytes": _CHECKSUM_BYTES,
        "checksum_last_modified": "2026-07-10T00:00:00Z",
        "checksum_etag": _CHECKSUM_ETAG,
    }


def _record_inventory(path: Path, item: dict[str, object], period: str) -> None:
    with MicrostructureWarehouse(
        path,
        cache_root=path.parent / "cache",
        memory_limit="256MB",
        threads=1,
    ) as warehouse:
        warehouse.record_official_archive_inventory(
            symbol="BTCUSDT",
            data_type="trades",
            items=[item],
            full_history=False,
            scope_start_period=period,
            scope_end_period=period,
        )


def _build_source(path: Path, period: str, monkeypatch) -> dict[str, object]:
    cache_root = path.parent / "source-cache"
    archive_path = (
        cache_root
        / "binance"
        / "usdm"
        / "trades"
        / "BTCUSDT"
        / f"BTCUSDT-trades-{period}.zip"
    )
    archive_path.parent.mkdir(parents=True)
    start_ms = int(
        datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"BTCUSDT-trades-{period}.csv",
            "id,price,qty,quote_qty,time,is_buyer_maker\n"
            f"1,100.0,1.0,100.0,{start_ms + 1000},false\n"
            f"2,101.0,2.0,202.0,{start_ms + 2000},true\n",
        )
    source_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        warehouse_module,
        "_fetch_checksum",
        lambda *_args, **_kwargs: source_sha256,
    )
    item = _inventory_item(period=period, size_bytes=archive_path.stat().st_size)
    with MicrostructureWarehouse(
        path,
        cache_root=cache_root,
        memory_limit="256MB",
        threads=1,
    ) as warehouse:
        warehouse.record_official_archive_inventory(
            symbol="BTCUSDT",
            data_type="trades",
            items=[item],
            full_history=False,
            scope_start_period=period,
            scope_end_period=period,
        )
        result = warehouse.ingest_public_archive(
            symbol="BTCUSDT",
            data_type="trades",
            period=period,
            expected_bytes=int(item["size_bytes"]),
            official_last_modified=str(item["last_modified"]),
            official_etag=str(item["etag"]),
            checksum_object_size_bytes=int(item["checksum_size_bytes"]),
            checksum_last_modified=str(item["checksum_last_modified"]),
            checksum_etag=str(item["checksum_etag"]),
            session=object(),
        )
    assert result.status == "complete"
    return item


def test_certified_merge_is_atomic_verified_and_idempotent(tmp_path, monkeypatch) -> None:
    period = "2026-07-09"
    source = tmp_path / "source.duckdb"
    destination = tmp_path / "destination.duckdb"
    item = _build_source(source, period, monkeypatch)
    _record_inventory(destination, item, period)

    evidence = merge_certified_tick_warehouse(
        destination_path=destination,
        source_path=source,
        symbol="BTCUSDT",
        data_type="trades",
        start_date=period,
        end_date=period,
        memory_limit="256MB",
        threads=1,
    )
    repeated = merge_certified_tick_warehouse(
        destination_path=destination,
        source_path=source,
        symbol="BTCUSDT",
        data_type="trades",
        start_date=period,
        end_date=period,
        memory_limit="256MB",
        threads=1,
    )

    with MicrostructureWarehouse(
        destination,
        memory_limit="256MB",
        threads=1,
        read_only=True,
    ) as warehouse:
        raw_rows = warehouse.connect().execute("SELECT count(*) FROM trade_raw").fetchone()[0]
        derived_rows = warehouse.connect().execute("SELECT count(*) FROM trade_1s").fetchone()[0]

    assert evidence["status"] == "complete"
    assert evidence["inserted_manifest_count"] == 1
    assert evidence["reused_manifest_count"] == 0
    assert evidence["copied_rows"] == {"trade_raw": 2, "trade_1s": 2}
    assert len(str(evidence["merge_sha256"])) == 64
    assert repeated["inserted_manifest_count"] == 0
    assert repeated["reused_manifest_count"] == 1
    assert raw_rows == 2
    assert derived_rows == 2


def test_certified_merge_rejects_inventory_mismatch(tmp_path, monkeypatch) -> None:
    period = "2026-07-09"
    source = tmp_path / "source.duckdb"
    destination = tmp_path / "destination.duckdb"
    item = _build_source(source, period, monkeypatch)
    mismatched = dict(item)
    mismatched["etag"] = "c" * 32
    _record_inventory(destination, mismatched, period)

    with pytest.raises(ValueError, match="official inventories differ"):
        merge_certified_tick_warehouse(
            destination_path=destination,
            source_path=source,
            symbol="BTCUSDT",
            data_type="trades",
            start_date=period,
            end_date=period,
            memory_limit="256MB",
            threads=1,
        )


def test_read_only_warehouse_requires_existing_file(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "missing.duckdb",
        read_only=True,
        memory_limit="256MB",
        threads=1,
    )
    with pytest.raises(FileNotFoundError, match="read-only warehouse does not exist"):
        warehouse.connect()
