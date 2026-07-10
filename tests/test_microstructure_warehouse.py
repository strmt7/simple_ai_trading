from __future__ import annotations

import duckdb
import pytest

from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse


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
            ?, 'binance-usdm-tick-v5', 'binance', 'futures', 'BTCUSDT',
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
