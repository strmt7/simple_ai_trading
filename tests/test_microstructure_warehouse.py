from __future__ import annotations

import duckdb
import pytest

from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.microstructure_warehouse import official_tick_archive_url


def _insert_complete_manifest(
    warehouse: MicrostructureWarehouse,
    *,
    archive_id: str,
    period: str,
    source_hash: str,
    rows: int,
    first_ms: int,
    last_ms: int,
    data_type: str = "bookTicker",
) -> None:
    warehouse.connect().execute(
        """
        INSERT INTO archive_manifest (
            archive_id, schema_version, provider, market_type, symbol, data_type,
            period, url, archive_path, status, is_current, expected_bytes,
            compressed_bytes, uncompressed_bytes, source_sha256, expected_sha256,
            checksum_status, rows_read, derived_rows, first_exchange_time_ms,
            last_exchange_time_ms, invalid_rows, duplicate_ids, update_id_regressions,
            event_time_regressions, out_of_order_rows, crossed_books, ingested_at_ms, error
        ) VALUES (
            ?, 'binance-usdm-tick-v6', 'binance', 'futures', 'BTCUSDT',
            ?, ?, ?, '', 'complete', true, 0, 0, 0, ?, ?,
            'verified', ?, 0, ?, ?, 0, 0, 0, 0, 0, 0, 1, ''
        )
        """,
        [
            archive_id,
            data_type,
            period,
            f"https://data.binance.vision/{archive_id}.zip",
            source_hash,
            source_hash,
            rows,
            first_ms,
            last_ms,
        ],
    )


def _insert_certified_manifest(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    data_type: str,
    period: str,
    size_bytes: int,
    hash_character: str,
) -> None:
    from datetime import UTC, datetime

    start_ms = int(datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
    first_ms = start_ms + 1_000
    last_ms = start_ms + 86_399_000
    source_hash = hash_character * 64
    url = official_tick_archive_url(symbol=symbol, data_type=data_type, period=period)
    archive_id = f"{symbol}-{data_type}-{period}"
    raw_rows = 500 if data_type == "bookDepth" else 100
    derived_rows = 50
    warehouse.connect().execute(
        """
        INSERT INTO archive_manifest (
            archive_id, schema_version, provider, market_type, symbol, data_type,
            period, url, archive_path, status, is_current, expected_bytes,
            compressed_bytes, uncompressed_bytes, source_sha256, expected_sha256,
            checksum_status, rows_read, derived_rows, first_exchange_time_ms,
            last_exchange_time_ms, invalid_rows, duplicate_ids, update_id_regressions,
            event_time_regressions, out_of_order_rows, crossed_books, ingested_at_ms, error
        ) VALUES (
            ?, 'binance-usdm-tick-v6', 'binance', 'futures', ?, ?, ?, ?, '',
            'complete', true, ?, ?, ?, ?, ?, 'verified', ?, ?, ?, ?,
            0, 0, 0, 0, 0, 0, ?, ''
        )
        """,
        [
            archive_id,
            symbol,
            data_type,
            period,
            url,
            size_bytes,
            size_bytes,
            size_bytes * 4,
            source_hash,
            source_hash,
            raw_rows,
            derived_rows,
            first_ms,
            last_ms,
            start_ms + 172_800_000,
        ],
    )
    conn = warehouse.connect()
    if data_type == "bookTicker":
        conn.execute(
            """
            INSERT INTO book_ticker_raw
            SELECT ?, ?, i::UBIGINT, 100.0, 1.0, 101.0, 1.0,
                   ? + ((? - ?) * i) // 99,
                   ? + ((? - ?) * i) // 99
            FROM range(100) AS rows(i)
            """,
            [archive_id, symbol, first_ms, last_ms, first_ms, first_ms, last_ms, first_ms],
        )
        conn.execute(
            """
            INSERT INTO book_ticker_path_1s
            SELECT ?, ?, ? + (((? - ?) // 1000 * i) // 49) * 1000,
                   100.0, 100.0, 100.0, 101.0, 101.0, 101.0
            FROM range(50) AS rows(i)
            """,
            [
                archive_id,
                symbol,
                (first_ms // 1_000) * 1_000,
                (last_ms // 1_000) * 1_000,
                (first_ms // 1_000) * 1_000,
            ],
        )
        conn.execute(
            """
            INSERT INTO book_ticker_100ms
            SELECT ?, ?, ? + (((? - ?) // 100 * i) // 49) * 100,
                   100.0, 100.0, 100.0, 1.0,
                   101.0, 101.0, 101.0, 1.0,
                   ? + (((? - ?) // 100 * i) // 49) * 100,
                   ? + (((? - ?) // 100 * i) // 49) * 100
            FROM range(50) AS rows(i)
            """,
            [
                archive_id,
                symbol,
                (first_ms // 100) * 100,
                (last_ms // 100) * 100,
                (first_ms // 100) * 100,
                (first_ms // 100) * 100,
                (last_ms // 100) * 100,
                (first_ms // 100) * 100,
                (first_ms // 100) * 100,
                (last_ms // 100) * 100,
                (first_ms // 100) * 100,
            ],
        )
    elif data_type == "trades":
        conn.execute(
            """
            INSERT INTO trade_raw
            SELECT ?, ?, i::UBIGINT, 100.0, 1.0, 100.0,
                   ? + ((? - ?) * i) // 99, false
            FROM range(100) AS rows(i)
            """,
            [archive_id, symbol, first_ms, last_ms, first_ms],
        )
        conn.execute(
            """
            INSERT INTO trade_1s
            SELECT ?, ?, ? + (((? - ?) // 1000 * i) // 49) * 1000,
                   100.0, 100.0, 100.0, 100.0,
                   1.0, 100.0, 0.5, 0.5, 0.0, 2
            FROM range(50) AS rows(i)
            """,
            [
                archive_id,
                symbol,
                (first_ms // 1_000) * 1_000,
                (last_ms // 1_000) * 1_000,
                (first_ms // 1_000) * 1_000,
            ],
        )
    else:
        conn.execute(
            """
            INSERT INTO book_depth_aggregate_raw
            SELECT ?, ?, ? + ((? - ?) * i) // 49,
                   percentage::DECIMAL(4,2), 1.0, 100.0
            FROM range(50) AS rows(i),
                 unnest([-5.0,-4.0,-3.0,-2.0,-1.0,1.0,2.0,3.0,4.0,5.0]) bands(percentage)
            """,
            [archive_id, symbol, first_ms, last_ms, first_ms],
        )


def test_corpus_certificate_binds_official_inventory_and_every_daily_manifest(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "certificate.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    periods = ("2026-07-08", "2026-07-09")
    data_types = ("bookTicker", "trades", "bookDepth")
    try:
        for type_index, data_type in enumerate(data_types, start=1):
            items = []
            for period_index, period in enumerate(periods, start=1):
                size_bytes = 100 * type_index + period_index
                items.append(
                    {
                        "period": period,
                        "url": official_tick_archive_url(
                            symbol="BTCUSDT",
                            data_type=data_type,
                            period=period,
                        ),
                        "size_bytes": size_bytes,
                        "last_modified": f"2026-07-{period_index:02d}T00:00:00Z",
                    }
                )
                _insert_certified_manifest(
                    warehouse,
                    symbol="BTCUSDT",
                    data_type=data_type,
                    period=period,
                    size_bytes=size_bytes,
                    hash_character=str(type_index),
                )
            first = warehouse.record_official_archive_inventory(
                symbol="BTCUSDT",
                data_type=data_type,
                items=items,
                full_history=True,
            )
            repeated = warehouse.record_official_archive_inventory(
                symbol="BTCUSDT",
                data_type=data_type,
                items=items,
                full_history=True,
            )
            assert repeated["snapshot_id"] == first["snapshot_id"]
            assert repeated["observed_at_ms"] == first["observed_at_ms"]

        evidence = warehouse.require_corpus_certificate(
            "BTCUSDT",
            required_start_ms=1_783_468_800_000,
            required_end_ms=1_783_641_599_999,
        )
    finally:
        warehouse.close()

    assert evidence["status"] == "pass"
    assert evidence["verified"] is True
    assert evidence["common_period_count"] == 2
    assert evidence["common_calendar_gaps"] == []
    assert len(str(evidence["certificate_sha256"])) == 64
    assert all(
        evidence["data_types"][data_type]["verified_scope_archive_count"] == 2
        for data_type in data_types
    )


def test_corpus_certificate_rejects_missing_and_mutated_partitions(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "incomplete-certificate.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    period = "2026-07-09"
    try:
        for type_index, data_type in enumerate(("bookTicker", "trades", "bookDepth"), start=1):
            size_bytes = 100 + type_index
            item = {
                "period": period,
                "url": official_tick_archive_url(
                    symbol="BTCUSDT",
                    data_type=data_type,
                    period=period,
                ),
                "size_bytes": size_bytes,
                "last_modified": "2026-07-10T00:00:00Z",
            }
            warehouse.record_official_archive_inventory(
                symbol="BTCUSDT",
                data_type=data_type,
                items=[item],
                full_history=True,
            )
            if data_type != "bookDepth":
                _insert_certified_manifest(
                    warehouse,
                    symbol="BTCUSDT",
                    data_type=data_type,
                    period=period,
                    size_bytes=size_bytes,
                    hash_character=str(type_index),
                )

        missing = warehouse.corpus_certificate("BTCUSDT")
        assert missing["status"] == "fail"
        assert any("bookDepth:missing_manifests" in reason for reason in missing["reasons"])

        _insert_certified_manifest(
            warehouse,
            symbol="BTCUSDT",
            data_type="bookDepth",
            period=period,
            size_bytes=103,
            hash_character="3",
        )
        warehouse.connect().execute(
            "UPDATE archive_manifest SET compressed_bytes = compressed_bytes + 1 "
            "WHERE data_type = 'trades'"
        )
        mutated = warehouse.corpus_certificate("BTCUSDT")
    finally:
        warehouse.close()

    assert mutated["status"] == "fail"
    assert any("trades:invalid_manifests" in reason for reason in mutated["reasons"])


def test_verified_unchanged_reuse_requires_intact_physical_partition(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "reusable-certificate.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    period = "2026-07-09"
    item = {
        "period": period,
        "url": official_tick_archive_url(
            symbol="BTCUSDT",
            data_type="trades",
            period=period,
        ),
        "size_bytes": 101,
        "last_modified": "2026-07-10T00:00:00Z",
    }
    try:
        _insert_certified_manifest(
            warehouse,
            symbol="BTCUSDT",
            data_type="trades",
            period=period,
            size_bytes=101,
            hash_character="a",
        )
        warehouse.record_official_archive_inventory(
            symbol="BTCUSDT",
            data_type="trades",
            items=[item],
            full_history=True,
        )
        reusable = warehouse.reusable_official_archives(
            symbol="BTCUSDT",
            data_type="trades",
            items=[item],
        )
        assert reusable[period].status == "skipped_verified_unchanged"

        warehouse.connect().execute(
            "DELETE FROM trade_raw WHERE archive_id = ? AND trade_id = 0",
            [f"BTCUSDT-trades-{period}"],
        )
        corrupted = warehouse.reusable_official_archives(
            symbol="BTCUSDT",
            data_type="trades",
            items=[item],
        )
        certificate = warehouse.corpus_certificate(
            "BTCUSDT",
            required_data_types=("trades",),
        )
    finally:
        warehouse.close()

    assert corrupted == {}
    assert certificate["status"] == "fail"
    corruption_reasons = certificate["data_types"]["trades"]["invalid_details"][period]
    assert "physical_raw_row_count_mismatch" in corruption_reasons
    assert "physical_raw_time_bounds_mismatch" in corruption_reasons


def test_book_ticker_ingest_canonicalizes_interleaved_source_rows(tmp_path) -> None:
    csv_path = tmp_path / "BTCUSDT-bookTicker-2024-02-15.csv"
    csv_path.write_text(
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,"
        "transaction_time,event_time\n"
        "100,50000,1,50001,2,1707955200024,1707955200033\n"
        "300,50100,1,50101,2,1708017636626,1708017636633\n"
        "200,50002,1,50003,2,1707955200026,1707955200034\n",
        encoding="ascii",
    )
    warehouse = MicrostructureWarehouse(
        tmp_path / "ticks.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    try:
        metrics = warehouse._ingest_book_ticker_csv(
            csv_path,
            has_header=True,
            symbol="BTCUSDT",
            archive_id="interleaved",
            period="2024-02-15",
        )
        persisted = warehouse.connect().execute(
            """
            SELECT update_id, transaction_time_ms
            FROM book_ticker_raw
            WHERE archive_id = 'interleaved'
            ORDER BY transaction_time_ms, event_time_ms, update_id
            """
        ).fetchall()
    finally:
        warehouse.close()

    assert metrics["rows_read"] == 3
    assert metrics["duplicate_ids"] == 0
    assert metrics["invalid_rows"] == 0
    assert metrics["crossed_books"] == 0
    assert metrics["update_id_regressions"] == 1
    assert metrics["out_of_order_rows"] == 1
    assert persisted == [
        (100, 1_707_955_200_024),
        (200, 1_707_955_200_026),
        (300, 1_708_017_636_626),
    ]


def test_book_depth_ingest_preserves_official_point_two_percent_bands(tmp_path) -> None:
    csv_path = tmp_path / "BTCUSDT-bookDepth-2026-07-09.csv"
    percentages = (-5, -4, -3, -2, -1, -0.20, 0.20, 1, 2, 3, 4, 5)
    csv_path.write_text(
        "timestamp,percentage,depth,notional\n"
        + "".join(
            f"2026-07-09 00:00:06,{percentage:.2f},100.0,10000.0\n"
            for percentage in percentages
        ),
        encoding="ascii",
    )
    warehouse = MicrostructureWarehouse(
        tmp_path / "depth.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    try:
        metrics = warehouse._ingest_book_depth_csv(
            csv_path,
            symbol="BTCUSDT",
            archive_id="depth-fixture",
            period="2026-07-09",
        )
        persisted = warehouse.connect().execute(
            "SELECT percentage::VARCHAR FROM book_depth_aggregate_raw "
            "WHERE archive_id = 'depth-fixture' ORDER BY percentage"
        ).fetchall()
    finally:
        warehouse.close()

    assert metrics["rows_read"] == 12
    assert metrics["derived_rows"] == 1
    assert ("-0.20",) in persisted
    assert ("0.20",) in persisted


def test_book_depth_ingest_accepts_exact_historical_ten_band_schema(tmp_path) -> None:
    csv_path = tmp_path / "BTCUSDT-bookDepth-2023-01-01.csv"
    percentages = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    csv_path.write_text(
        "timestamp,percentage,depth,notional\n"
        + "".join(
            f"2023-01-01 00:00:00,{percentage},100.0,10000.0\n"
            for percentage in percentages
        ),
        encoding="ascii",
    )
    warehouse = MicrostructureWarehouse(
        tmp_path / "historical-depth.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    try:
        metrics = warehouse._ingest_book_depth_csv(
            csv_path,
            symbol="BTCUSDT",
            archive_id="historical-depth",
            period="2023-01-01",
        )
        _insert_complete_manifest(
            warehouse,
            archive_id="historical-depth",
            period="2023-01-01",
            source_hash="c" * 64,
            rows=10,
            first_ms=1_672_531_200_000,
            last_ms=1_672_531_200_000,
            data_type="bookDepth",
        )
        snapshot = warehouse.connect().execute(
            "SELECT band_count, bid_notional_0_2, ask_notional_0_2 "
            "FROM current_book_depth_snapshots"
        ).fetchone()
    finally:
        warehouse.close()

    assert metrics["rows_read"] == 10
    assert metrics["derived_rows"] == 1
    assert snapshot == (10, None, None)


def test_book_depth_ingest_rejects_mixed_incomplete_band_schema(tmp_path) -> None:
    csv_path = tmp_path / "BTCUSDT-bookDepth-2023-01-01.csv"
    percentages = (-5, -4, -3, -2, -1, -0.20, 0.20, 1, 2, 3)
    csv_path.write_text(
        "timestamp,percentage,depth,notional\n"
        + "".join(
            f"2023-01-01 00:00:00,{percentage},100.0,10000.0\n"
            for percentage in percentages
        ),
        encoding="ascii",
    )
    warehouse = MicrostructureWarehouse(
        tmp_path / "mixed-depth.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    try:
        with pytest.raises(ValueError, match="incomplete_groups=1"):
            warehouse._ingest_book_depth_csv(
                csv_path,
                symbol="BTCUSDT",
                archive_id="mixed-depth",
                period="2023-01-01",
            )
    finally:
        warehouse.close()


def test_warehouse_migrates_legacy_integer_book_depth_percentage(tmp_path) -> None:
    database = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(
        "CREATE TABLE book_depth_aggregate_raw ("
        "archive_id VARCHAR, symbol VARCHAR, timestamp_ms BIGINT, "
        "percentage SMALLINT, depth DOUBLE, notional DOUBLE)"
    )
    connection.execute(
        "INSERT INTO book_depth_aggregate_raw VALUES "
        "('legacy', 'BTCUSDT', 1, -1, 10.0, 100.0)"
    )
    connection.close()

    warehouse = MicrostructureWarehouse(
        database,
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    try:
        column_type = {
            row[1]: row[2]
            for row in warehouse.connect().execute(
                "PRAGMA table_info('book_depth_aggregate_raw')"
            ).fetchall()
        }["percentage"]
        value = warehouse.connect().execute(
            "SELECT percentage::VARCHAR FROM book_depth_aggregate_raw"
        ).fetchone()
    finally:
        warehouse.close()

    assert column_type == "DECIMAL(4,2)"
    assert value == ("-1.00",)


def test_trade_depth_view_uses_only_latest_snapshot_available_at_second(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "asof.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    try:
        _insert_complete_manifest(
            warehouse,
            archive_id="trades",
            period="2026-07-09",
            source_hash="a" * 64,
            rows=2,
            first_ms=1_000,
            last_ms=2_999,
            data_type="trades",
        )
        _insert_complete_manifest(
            warehouse,
            archive_id="depth",
            period="2026-07-09",
            source_hash="b" * 64,
            rows=12,
            first_ms=1_500,
            last_ms=1_500,
            data_type="bookDepth",
        )
        warehouse.connect().executemany(
            "INSERT INTO trade_1s VALUES "
            "('trades', 'BTCUSDT', ?, 100, 101, 99, 100, 10, 1000, 6, 4, 0.2, 2)",
            [(1_000,), (2_000,)],
        )
        percentages = (-5, -4, -3, -2, -1, -0.20, 0.20, 1, 2, 3, 4, 5)
        warehouse.connect().executemany(
            "INSERT INTO book_depth_aggregate_raw VALUES "
            "('depth', 'BTCUSDT', 1500, ?, ?, ?)",
            [
                (
                    percentage,
                    200.0 if percentage < 0 else 100.0,
                    20_000.0 if percentage < 0 else 10_000.0,
                )
                for percentage in percentages
            ],
        )
        rows = warehouse.connect().execute(
            "SELECT second_ms, depth_time_ms, depth_age_ms, "
            "bid_depth_0_2, ask_depth_0_2, depth_imbalance_0_2 "
            "FROM current_trade_depth_1s ORDER BY second_ms"
        ).fetchall()
    finally:
        warehouse.close()

    assert rows[0] == (1_000, None, None, None, None, None)
    assert rows[1][:5] == (2_000, 1_500, 500, 200.0, 100.0)
    assert rows[1][5] == pytest.approx(1.0 / 3.0)


def test_causal_feature_rebuild_aggregates_event_time_across_daily_archives(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "ticks.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    boundary_ms = 1_704_153_600_000
    try:
        _insert_complete_manifest(
            warehouse,
            archive_id="day-one",
            period="2024-01-01",
            source_hash="a" * 64,
            rows=2,
            first_ms=boundary_ms - 100,
            last_ms=boundary_ms - 10,
        )
        _insert_complete_manifest(
            warehouse,
            archive_id="day-two",
            period="2024-01-02",
            source_hash="b" * 64,
            rows=1,
            first_ms=boundary_ms + 10,
            last_ms=boundary_ms + 10,
        )
        warehouse.connect().executemany(
            "INSERT INTO book_ticker_raw VALUES (?, 'BTCUSDT', ?, ?, ?, ?, ?, ?, ?)",
            [
                ("day-one", 1, 100.0, 2.0, 102.0, 2.0, boundary_ms - 100, boundary_ms - 80),
                ("day-one", 2, 110.0, 2.0, 112.0, 2.0, boundary_ms - 10, boundary_ms + 5),
                ("day-two", 3, 120.0, 2.0, 122.0, 2.0, boundary_ms + 10, boundary_ms + 20),
            ],
        )

        evidence = warehouse.rebuild_causal_feature_bars("BTCUSDT")
        bars = warehouse.connect().execute(
            """
            SELECT second_ms, open_mid, close_mid, quote_updates, source_archive_count
            FROM current_book_ticker_1s
            ORDER BY second_ms
            """
        ).fetchall()

        assert evidence["verified"] is True
        assert evidence["source_raw_rows"] == 3
        assert evidence["feature_rows"] == 2
        assert bars == [
            (boundary_ms - 1_000, 101.0, 101.0, 1, 1),
            (boundary_ms, 111.0, 121.0, 2, 2),
        ]
        assert len({row[0] for row in bars}) == len(bars)

        warehouse.connect().execute(
            "UPDATE archive_manifest SET rows_read = rows_read + 1 WHERE archive_id = 'day-two'"
        )
        with pytest.raises(ValueError, match="stale relative to current source manifests"):
            warehouse.require_causal_feature_bars("BTCUSDT")
    finally:
        warehouse.close()


def test_terminal_holdout_reservation_blocks_all_overlapping_reuse(tmp_path) -> None:
    warehouse = MicrostructureWarehouse(
        tmp_path / "ticks.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    arguments = {
        "symbol": "BTCUSDT",
        "candidate_sha256": "a" * 64,
        "source_manifest_fingerprint": "b" * 64,
        "source_feature_build_id": "c" * 64,
            "feature_version": "l1-tape-causal-v5",
            "model_schema_version": "microstructure-action-value-v9",
            "prequential_report_sha256": "d" * 64,
        }
    try:
        first = warehouse.reserve_terminal_holdout(
            first_utc_day=100,
            last_utc_day=110,
            **arguments,
        )
        with pytest.raises(ValueError, match="overlaps a previously consumed or reserved period"):
            warehouse.reserve_terminal_holdout(
                first_utc_day=110,
                last_utc_day=120,
                **arguments,
            )
        completed = warehouse.finalize_terminal_holdout(
            str(first["reservation_id"]),
            result_status="accepted",
        )
        assert completed["status"] == "complete"
        with pytest.raises(ValueError, match="overlaps a previously consumed or reserved period"):
            warehouse.reserve_terminal_holdout(
                first_utc_day=101,
                last_utc_day=102,
                **arguments,
            )

        second = warehouse.reserve_terminal_holdout(
            first_utc_day=111,
            last_utc_day=120,
            **arguments,
        )
        failed = warehouse.finalize_terminal_holdout(
            str(second["reservation_id"]),
            result_status="evaluation_error",
            error="simulated interruption",
        )
        assert failed["status"] == "failed"
        with pytest.raises(ValueError, match="already been finalized"):
            warehouse.finalize_terminal_holdout(
                str(second["reservation_id"]),
                result_status="rejected",
            )
    finally:
        warehouse.close()
