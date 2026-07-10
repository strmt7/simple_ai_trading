from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from simple_ai_trading import tape_depth_features as feature_module
from simple_ai_trading import tape_depth_cache as cache_module
from simple_ai_trading.microstructure_warehouse import (
    MicrostructureWarehouse,
    TICK_WAREHOUSE_SCHEMA_VERSION,
    official_tick_archive_url,
)
from simple_ai_trading.tape_depth_cache import (
    load_tape_depth_dataset_cache,
    save_tape_depth_dataset_cache,
)
from simple_ai_trading.tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_TARGET_MODE,
    build_tape_depth_forecast_dataset,
    slice_tape_depth_forecast_dataset,
    tape_depth_dataset_source_evidence,
    tape_depth_dataset_fingerprint,
    tape_depth_source_evidence,
)


def _manifest(
    warehouse: MicrostructureWarehouse,
    *,
    archive_id: str,
    symbol: str = "BTCUSDT",
    data_type: str,
    period: str = "2026-07-09",
    rows: int,
    derived_rows: int,
    first_ms: int,
    last_ms: int,
) -> None:
    source_hash = ("a" if data_type == "trades" else "b") * 64
    source_url = official_tick_archive_url(
        symbol=symbol,
        data_type=data_type,
        period=period,
    )
    warehouse.connect().execute(
        """
        INSERT INTO archive_manifest (
            archive_id, schema_version, provider, market_type, symbol, data_type,
            period, url, archive_path, status, is_current, expected_bytes,
            compressed_bytes, uncompressed_bytes, source_sha256, expected_sha256,
            checksum_status, rows_read, derived_rows, first_exchange_time_ms,
            last_exchange_time_ms, invalid_rows, duplicate_ids, update_id_regressions,
            event_time_regressions, out_of_order_rows, crossed_books, ingested_at_ms,
            error, official_etag, checksum_object_size_bytes,
            checksum_last_modified, checksum_etag
        ) VALUES (
            ?, ?, 'binance', 'futures', ?, ?, ?, ?, '',
            'complete', true, 1, 1, 1, ?, ?, 'verified', ?, ?, ?, ?,
            0, 0, 0, 0, 0, 0, ?, '', ?, 100, ?, ?
        )
        """,
        [
            archive_id,
            TICK_WAREHOUSE_SCHEMA_VERSION,
            symbol,
            data_type,
            period,
            source_url,
            source_hash,
            source_hash,
            rows,
            derived_rows,
            first_ms,
            last_ms,
            last_ms + 86_400_000,
            "c" * 32,
            f"{period}T00:00:00Z",
            "d" * 32,
        ],
    )
    inventory_rows = warehouse.connect().execute(
        """
        SELECT period, url
        FROM archive_manifest
        WHERE symbol = ? AND data_type = ? AND status = 'complete' AND is_current
        ORDER BY period
        """,
        [symbol, data_type],
    ).fetchall()
    warehouse.record_official_archive_inventory(
        symbol=symbol,
        data_type=data_type,
        items=[
            {
                "period": item_period,
                "url": item_url,
                "size_bytes": 1,
                "last_modified": f"{item_period}T00:00:00Z",
                "etag": "c" * 32,
                "checksum_size_bytes": 100,
                "checksum_last_modified": f"{item_period}T00:00:00Z",
                "checksum_etag": "d" * 32,
            }
            for item_period, item_url in inventory_rows
        ],
        full_history=True,
    )


def _warehouse_fixture(tmp_path) -> tuple[MicrostructureWarehouse, int, int]:
    warehouse = MicrostructureWarehouse(
        tmp_path / "tape-depth.duckdb",
        cache_root=tmp_path / "cache",
        memory_limit="256MB",
        threads=1,
    )
    base_ms = 1_783_555_200_000
    seconds = 1_020
    _manifest(
        warehouse,
        archive_id="trades",
        data_type="trades",
        rows=seconds,
        derived_rows=seconds - 1,
        first_ms=base_ms,
        last_ms=base_ms + (seconds - 1) * 1_000,
    )
    trade_rows = []
    for index in range(seconds):
        if index == 974:
            continue
        price = 100.0 + index * 0.002 + 0.02 * math.sin(index / 10.0)
        buy = 6.0 + 0.5 * math.sin(index / 7.0)
        sell = 4.0 + 0.5 * math.cos(index / 9.0)
        volume = buy + sell
        trade_rows.append(
            (
                "trades",
                "BTCUSDT",
                base_ms + index * 1_000,
                price - 0.01,
                price + 0.02,
                price - 0.02,
                price,
                volume,
                price * volume,
                buy,
                sell,
                (buy - sell) / volume,
                20 + index % 5,
            )
        )
    warehouse.connect().executemany(
        "INSERT INTO trade_1s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        trade_rows,
    )
    warehouse.connect().executemany(
        "INSERT INTO trade_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "trades",
                "BTCUSDT",
                index,
                100.0 + index * 0.002 + 0.02 * math.sin(index / 10.0),
                1.0,
                100.0 + index * 0.002 + 0.02 * math.sin(index / 10.0),
                base_ms + index * 1_000,
                False,
            )
            for index in range(seconds)
        ],
    )
    for peer_index, peer in enumerate(("ETHUSDT", "SOLUSDT"), start=1):
        archive_id = f"trades-{peer.lower()}"
        _manifest(
            warehouse,
            archive_id=archive_id,
            symbol=peer,
            data_type="trades",
            rows=seconds,
            derived_rows=seconds,
            first_ms=base_ms,
            last_ms=base_ms + (seconds - 1) * 1_000,
        )
        peer_rows = []
        for index in range(seconds):
            price = 50.0 * peer_index + index * (0.001 + peer_index * 0.0002)
            buy = 4.0 + peer_index * 0.5 + 0.2 * math.sin(index / 8.0)
            sell = 3.5 + 0.2 * math.cos(index / 11.0)
            volume = buy + sell
            peer_rows.append(
                (
                    archive_id,
                    peer,
                    base_ms + index * 1_000,
                    price - 0.01,
                    price + 0.01,
                    price - 0.01,
                    price,
                    volume,
                    price * volume,
                    buy,
                    sell,
                    (buy - sell) / volume,
                    12 + index % 4,
                )
            )
        warehouse.connect().executemany(
            "INSERT INTO trade_1s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            peer_rows,
        )
        warehouse.connect().executemany(
            "INSERT INTO trade_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    archive_id,
                    peer,
                    peer_index * 10_000 + index,
                    50.0 * peer_index + index * (0.001 + peer_index * 0.0002),
                    1.0,
                    50.0 * peer_index + index * (0.001 + peer_index * 0.0002),
                    base_ms + index * 1_000,
                    False,
                )
                for index in range(seconds)
            ],
        )
    depth_start_index = 950
    depth_indexes = list(range(depth_start_index, seconds, 30))
    _manifest(
        warehouse,
        archive_id="depth",
        data_type="bookDepth",
        rows=len(depth_indexes) * 12,
        derived_rows=len(depth_indexes),
        first_ms=base_ms + depth_indexes[0] * 1_000,
        last_ms=base_ms + depth_indexes[-1] * 1_000,
    )
    percentages = (-5, -4, -3, -2, -1, -0.20, 0.20, 1, 2, 3, 4, 5)
    depth_rows = []
    for index in depth_indexes:
        for percentage in percentages:
            side_multiplier = 1.2 if percentage < 0 else 0.8
            depth_rows.append(
                (
                    "depth",
                    "BTCUSDT",
                    base_ms + index * 1_000,
                    percentage,
                    100.0 * abs(percentage) * side_multiplier,
                    10_000.0 * abs(percentage) * side_multiplier,
                )
            )
    warehouse.connect().executemany(
        "INSERT INTO book_depth_aggregate_raw VALUES (?, ?, ?, ?, ?, ?)",
        depth_rows,
    )
    return warehouse, base_ms, seconds


def test_tape_depth_dataset_is_causal_bounded_and_has_no_execution_claim(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        dataset = build_tape_depth_forecast_dataset(
            warehouse,
            symbol="BTCUSDT",
            start_ms=base_ms + 901_000,
            end_ms=base_ms + 1_000_000,
            horizon_seconds=5,
            total_latency_ms=750,
            decision_cadence_seconds=5,
            maximum_rows=100,
        )
    finally:
        warehouse.close()

    assert dataset.target_mode == TAPE_DEPTH_TARGET_MODE
    assert dataset.summary()["execution_claim"] is False
    assert dataset.feature_names == TAPE_DEPTH_FEATURE_NAMES
    assert dataset.features.shape == (20, len(TAPE_DEPTH_FEATURE_NAMES))
    assert np.all(np.diff(dataset.decision_time_ms) == 5_000)
    assert np.all(dataset.target_entry_time_ms - dataset.decision_time_ms == 1_000)
    assert np.all(dataset.target_exit_time_ms - dataset.target_entry_time_ms == 5_000)
    assert dataset.summary()["effective_entry_delay_ms"] == 1_000
    assert dataset.summary()["target_span_ms"] == 6_000
    assert dataset.summary()["cross_asset_context_available_ratio"] == 1.0
    assert np.all(np.isfinite(dataset.gross_return_bps))
    depth_available = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("depth_available")
    ]
    assert np.any(depth_available == 0.0)
    assert np.any(depth_available == 1.0)
    trade_observed = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("trade_observed")
    ]
    trade_age = dataset.features[:, TAPE_DEPTH_FEATURE_NAMES.index("trade_age_seconds")]
    assert np.count_nonzero(trade_observed == 0.0) == 1
    assert np.max(trade_age) == 1.0
    for window in (60, 300, 900):
        efficiency = dataset.features[
            :, TAPE_DEPTH_FEATURE_NAMES.index(f"price_efficiency_{window}")
        ]
        observation_rate = dataset.features[
            :, TAPE_DEPTH_FEATURE_NAMES.index(f"trade_observation_rate_{window}")
        ]
        assert np.all((efficiency >= 0.0) & (efficiency <= 1.0 + 1e-6))
        assert np.all((observation_rate >= 0.0) & (observation_rate <= 1.0))
    first_depth = dataset.features[
        0, TAPE_DEPTH_FEATURE_NAMES.index("depth_imbalance_0_2")
    ]
    assert np.isnan(first_depth)
    assert dataset.source_evidence["verified"] is True
    assert dataset.source_evidence["schema_version"] == TICK_WAREHOUSE_SCHEMA_VERSION
    assert set(dataset.source_evidence["cross_asset_context"]["peers"]) == {
        "ETHUSDT",
        "SOLUSDT",
    }
    cross_available = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("cross_asset_context_available")
    ]
    peer_return = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("peer_mean_return_bps_60")
    ]
    relative_return = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("relative_return_vs_peers_bps_60")
    ]
    assert np.all(cross_available == 1.0)
    assert np.all(np.isfinite(peer_return))
    assert np.any(np.abs(peer_return) > 0.0)
    assert np.any(np.abs(relative_return) > 0.0)


def test_tape_depth_dataset_cache_is_transactional_bound_and_replayable(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    options = {
        "symbol": "BTCUSDT",
        "start_ms": base_ms + 901_000,
        "end_ms": base_ms + 1_000_000,
        "horizon_seconds": 5,
        "decision_cadence_seconds": 5,
        "maximum_rows": 100,
    }
    try:
        dataset = build_tape_depth_forecast_dataset(warehouse, **options)
        cache_key = save_tape_depth_dataset_cache(warehouse, dataset)
        assert save_tape_depth_dataset_cache(warehouse, dataset) == cache_key

        load_options = {
            "symbol": dataset.symbol,
            "requested_start_ms": int(dataset.decision_time_ms[0]),
            "requested_end_ms": int(dataset.decision_time_ms[-1]),
            "horizon_seconds": dataset.horizon_seconds,
            "total_latency_ms": dataset.total_latency_ms,
            "decision_cadence_seconds": dataset.decision_cadence_seconds,
            "maximum_depth_age_ms": dataset.maximum_depth_age_ms,
            "source_evidence": dataset.source_evidence,
        }
        loaded = load_tape_depth_dataset_cache(warehouse, **load_options)
        assert loaded is not None
        assert tape_depth_dataset_fingerprint(loaded) == tape_depth_dataset_fingerprint(
            dataset
        )
        assert np.array_equal(loaded.features, dataset.features, equal_nan=True)

        changed_evidence = {
            **dataset.source_evidence,
            "manifest_fingerprint": "f" * 64,
        }
        assert (
            load_tape_depth_dataset_cache(
                warehouse,
                **{**load_options, "source_evidence": changed_evidence},
            )
            is None
        )

        rows_table = warehouse.connect().execute(
            "SELECT rows_table FROM tape_depth_dataset_cache_manifest "
            "WHERE cache_key = ?",
            [cache_key],
        ).fetchone()[0]
        warehouse.connect().execute(
            f"UPDATE {rows_table} SET return_bps_1 = return_bps_1 + 1.0 "
            "WHERE cache_key = ? AND decision_time_ms = ?",
            [cache_key, int(dataset.decision_time_ms[0])],
        )
        with pytest.raises(ValueError, match="fingerprint differs"):
            load_tape_depth_dataset_cache(warehouse, **load_options)
    finally:
        warehouse.close()


def test_tape_depth_dataset_cache_rejects_contract_and_storage_drift(
    tmp_path,
    monkeypatch,
) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        dataset = build_tape_depth_forecast_dataset(
            warehouse,
            symbol="BTCUSDT",
            start_ms=base_ms + 901_000,
            end_ms=base_ms + 1_000_000,
            horizon_seconds=5,
            decision_cadence_seconds=5,
            maximum_rows=100,
        )
        with pytest.raises(ValueError, match="verified source evidence"):
            cache_module.tape_depth_dataset_cache_key(
                symbol=dataset.symbol,
                requested_start_ms=int(dataset.decision_time_ms[0]),
                requested_end_ms=int(dataset.decision_time_ms[-1]),
                horizon_seconds=dataset.horizon_seconds,
                total_latency_ms=dataset.total_latency_ms,
                decision_cadence_seconds=dataset.decision_cadence_seconds,
                maximum_depth_age_ms=dataset.maximum_depth_age_ms,
                source_evidence={"verified": False},
            )
        with pytest.raises(ValueError, match="interval is empty"):
            cache_module.tape_depth_dataset_cache_key(
                symbol=dataset.symbol,
                requested_start_ms=2,
                requested_end_ms=1,
                horizon_seconds=dataset.horizon_seconds,
                total_latency_ms=dataset.total_latency_ms,
                decision_cadence_seconds=dataset.decision_cadence_seconds,
                maximum_depth_age_ms=dataset.maximum_depth_age_ms,
                source_evidence=dataset.source_evidence,
            )
        empty = replace(
            dataset,
            decision_time_ms=dataset.decision_time_ms[:0],
            target_entry_time_ms=dataset.target_entry_time_ms[:0],
            target_exit_time_ms=dataset.target_exit_time_ms[:0],
            target_entry_price=dataset.target_entry_price[:0],
            target_exit_price=dataset.target_exit_price[:0],
            gross_return_bps=dataset.gross_return_bps[:0],
            features=dataset.features[:0],
        )
        with pytest.raises(ValueError, match="cannot cache an empty"):
            cache_module.save_tape_depth_dataset_cache(warehouse, empty)
        with pytest.raises(ValueError, match="cannot be stored"):
            cache_module.save_tape_depth_dataset_cache(
                warehouse,
                replace(dataset, features=dataset.features.astype(np.float64)),
            )

        cache_key = cache_module.save_tape_depth_dataset_cache(warehouse, dataset)
        original_fingerprint = tape_depth_dataset_fingerprint(dataset)
        warehouse.connect().execute(
            "UPDATE tape_depth_dataset_cache_manifest "
            "SET dataset_fingerprint = ? WHERE cache_key = ?",
            ["b" * 64, cache_key],
        )
        with pytest.raises(ValueError, match="collides"):
            cache_module.save_tape_depth_dataset_cache(warehouse, dataset)
        warehouse.connect().execute(
            "UPDATE tape_depth_dataset_cache_manifest "
            "SET dataset_fingerprint = ? WHERE cache_key = ?",
            [original_fingerprint, cache_key],
        )

        load_options = {
            "symbol": dataset.symbol,
            "requested_start_ms": int(dataset.decision_time_ms[0]),
            "requested_end_ms": int(dataset.decision_time_ms[-1]),
            "horizon_seconds": dataset.horizon_seconds,
            "total_latency_ms": dataset.total_latency_ms,
            "decision_cadence_seconds": dataset.decision_cadence_seconds,
            "maximum_depth_age_ms": dataset.maximum_depth_age_ms,
            "source_evidence": dataset.source_evidence,
        }
        warehouse.connect().execute(
            "UPDATE tape_depth_dataset_cache_manifest "
            "SET schema_version = 'drifted' WHERE cache_key = ?",
            [cache_key],
        )
        with pytest.raises(ValueError, match="manifest failed"):
            cache_module.load_tape_depth_dataset_cache(warehouse, **load_options)
        warehouse.connect().execute(
            "UPDATE tape_depth_dataset_cache_manifest "
            "SET schema_version = ?, row_count = row_count + 1 WHERE cache_key = ?",
            [cache_module.TAPE_DEPTH_CACHE_SCHEMA_VERSION, cache_key],
        )
        with pytest.raises(ValueError, match="rows are incomplete"):
            cache_module.load_tape_depth_dataset_cache(warehouse, **load_options)

        monkeypatch.setattr(cache_module, "_ROWS_TABLE", "unsafe-table")
        with pytest.raises(ValueError, match="unsafe identifier"):
            cache_module._ensure_cache_schema(warehouse)
        monkeypatch.setattr(cache_module, "_ROWS_TABLE", cache_module._MANIFEST_TABLE)
        monkeypatch.setattr(cache_module, "_table_schema", lambda *_args: ())
        with pytest.raises(ValueError, match="manifest schema drifted"):
            cache_module._ensure_cache_schema(warehouse)

        def row_schema_drift(_warehouse, table):
            if table == cache_module._MANIFEST_TABLE:
                return cache_module._MANIFEST_COLUMNS
            return ()

        monkeypatch.setattr(cache_module, "_ROWS_TABLE", "cache_rows_safe")
        monkeypatch.setattr(cache_module, "_table_schema", row_schema_drift)
        with pytest.raises(ValueError, match="row schema drifted"):
            cache_module._ensure_cache_schema(warehouse)
    finally:
        warehouse.close()


def test_tape_depth_dataset_blocks_unbounded_memory_request(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        with pytest.raises(ValueError, match="maximum_rows=2"):
            build_tape_depth_forecast_dataset(
                warehouse,
                symbol="BTCUSDT",
                start_ms=base_ms + 901_000,
                end_ms=base_ms + 1_000_000,
                horizon_seconds=5,
                decision_cadence_seconds=5,
                maximum_rows=2,
            )
    finally:
        warehouse.close()


def test_cross_asset_helpers_reject_invalid_clock_and_output_contracts(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        with pytest.raises(ValueError, match="no feature seconds"):
            feature_module._peer_return_context(
                warehouse,
                symbol="ETHUSDT",
                feature_seconds_ms=np.asarray([], dtype=np.int64),
            )
        with pytest.raises(ValueError, match="not regularly spaced"):
            feature_module._peer_return_context(
                warehouse,
                symbol="ETHUSDT",
                feature_seconds_ms=np.asarray(
                    [base_ms, base_ms + 1_000, base_ms + 3_000],
                    dtype=np.int64,
                ),
            )
        with pytest.raises(ValueError, match="wrong contract"):
            feature_module._write_cross_asset_features(
                warehouse,
                symbol="BTCUSDT",
                decision_times_ms=np.asarray([base_ms + 901_000], dtype=np.int64),
                target_features=np.zeros(
                    (1, len(feature_module._TAPE_DEPTH_LOCAL_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                output=np.zeros((1, 1), dtype=np.float32),
            )
        with pytest.raises(ValueError, match="outside target evidence"):
            tape_depth_dataset_source_evidence(
                warehouse,
                "BTCUSDT",
                required_start_ms=base_ms,
                required_end_ms=base_ms + 1_000_000,
                peer_feature_start_ms=base_ms + 299_000,
                peer_feature_end_ms=base_ms + 999_000,
            )
    finally:
        warehouse.close()


def test_tape_depth_cross_asset_context_is_strictly_causal(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        before = build_tape_depth_forecast_dataset(
            warehouse,
            symbol="BTCUSDT",
            start_ms=base_ms + 901_000,
            end_ms=base_ms + 1_000_000,
            horizon_seconds=5,
            decision_cadence_seconds=5,
            maximum_rows=100,
        )
        changed_second_ms = int(before.decision_time_ms[10] - 1_000)
        warehouse.connect().execute(
            """
            UPDATE trade_1s
            SET open = open * 1.5, high = high * 1.5, low = low * 1.5,
                close = close * 1.5
            WHERE symbol = 'ETHUSDT' AND second_ms = ?
            """,
            [changed_second_ms],
        )
        after = build_tape_depth_forecast_dataset(
            warehouse,
            symbol="BTCUSDT",
            start_ms=base_ms + 901_000,
            end_ms=base_ms + 1_000_000,
            horizon_seconds=5,
            decision_cadence_seconds=5,
            maximum_rows=100,
        )
    finally:
        warehouse.close()

    context_indexes = [
        TAPE_DEPTH_FEATURE_NAMES.index(name)
        for name in TAPE_DEPTH_FEATURE_NAMES
        if name.startswith(
            (
                "cross_asset_",
                "peer_",
                "relative_return_vs_",
                "btc_anchor_",
            )
        )
    ]
    unaffected = before.decision_time_ms <= changed_second_ms
    affected = before.decision_time_ms > changed_second_ms
    assert np.array_equal(
        before.features[unaffected][:, context_indexes],
        after.features[unaffected][:, context_indexes],
    )
    assert np.any(
        before.features[affected][:, context_indexes]
        != after.features[affected][:, context_indexes]
    )


def test_tape_depth_cross_asset_context_rejects_peer_tail_gap(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        warehouse.connect().execute(
            "DELETE FROM trade_1s WHERE symbol = 'SOLUSDT' AND second_ms > ?",
            [base_ms + 950_000],
        )
        with pytest.raises(ValueError, match="ends before the target interval"):
            build_tape_depth_forecast_dataset(
                warehouse,
                symbol="BTCUSDT",
                start_ms=base_ms + 901_000,
                end_ms=base_ms + 1_000_000,
                horizon_seconds=5,
                decision_cadence_seconds=5,
                maximum_rows=100,
            )
    finally:
        warehouse.close()


def test_cross_asset_context_marks_a_not_yet_listed_peer_unavailable(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        warehouse.connect().execute(
            "UPDATE trade_1s SET second_ms = second_ms + 10000000 "
            "WHERE symbol = 'SOLUSDT'"
        )
        dataset = build_tape_depth_forecast_dataset(
            warehouse,
            symbol="BTCUSDT",
            start_ms=base_ms + 901_000,
            end_ms=base_ms + 1_000_000,
            horizon_seconds=5,
            decision_cadence_seconds=5,
            maximum_rows=100,
        )
    finally:
        warehouse.close()

    sol_evidence = dataset.source_evidence["cross_asset_context"]["peers"][
        "SOLUSDT"
    ]
    assert sol_evidence["status"] == "not_listed_during_interval"
    context_available = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("cross_asset_context_available")
    ]
    peer_mean = dataset.features[
        :, TAPE_DEPTH_FEATURE_NAMES.index("peer_mean_return_bps_60")
    ]
    assert np.all(context_available == 0.0)
    assert np.all(peer_mean == 0.0)


def test_cross_asset_evidence_stops_at_the_last_observable_feature_day(
    tmp_path,
) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    peer_end_ms = base_ms + 86_399_000
    target_day_two_ms = base_ms + 86_400_000
    target_end_ms = base_ms + 86_405_000
    try:
        for peer_index, peer in enumerate(("ETHUSDT", "SOLUSDT"), start=1):
            archive_id = f"trades-{peer.lower()}"
            price = 100.0 * peer_index
            warehouse.connect().execute(
                """
                UPDATE archive_manifest
                SET last_exchange_time_ms = ?, rows_read = rows_read + 1,
                    derived_rows = derived_rows + 1
                WHERE archive_id = ?
                """,
                [peer_end_ms, archive_id],
            )
            warehouse.connect().execute(
                "INSERT INTO trade_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    archive_id,
                    peer,
                    100_000 + peer_index,
                    price,
                    1.0,
                    price,
                    peer_end_ms,
                    False,
                ],
            )
            warehouse.connect().execute(
                "INSERT INTO trade_1s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    archive_id,
                    peer,
                    peer_end_ms,
                    price,
                    price,
                    price,
                    price,
                    1.0,
                    price,
                    0.5,
                    0.5,
                    0.0,
                    1,
                ],
            )
        _manifest(
            warehouse,
            archive_id="trades-target-day-two",
            data_type="trades",
            period="2026-07-10",
            rows=6,
            derived_rows=6,
            first_ms=target_day_two_ms,
            last_ms=target_end_ms,
        )
        for index in range(6):
            timestamp_ms = target_day_two_ms + index * 1_000
            price = 102.0 + index * 0.01
            warehouse.connect().execute(
                "INSERT INTO trade_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "trades-target-day-two",
                    "BTCUSDT",
                    index,
                    price,
                    1.0,
                    price,
                    timestamp_ms,
                    False,
                ],
            )
            warehouse.connect().execute(
                "INSERT INTO trade_1s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "trades-target-day-two",
                    "BTCUSDT",
                    timestamp_ms,
                    price,
                    price,
                    price,
                    price,
                    1.0,
                    price,
                    0.5,
                    0.5,
                    0.0,
                    1,
                ],
            )
        _manifest(
            warehouse,
            archive_id="bookDepth-target-day-two",
            data_type="bookDepth",
            period="2026-07-10",
            rows=24,
            derived_rows=2,
            first_ms=target_day_two_ms,
            last_ms=target_end_ms,
        )
        for timestamp_ms in (target_day_two_ms, target_end_ms):
            for percentage in (-5, -4, -3, -2, -1, -0.20, 0.20, 1, 2, 3, 4, 5):
                warehouse.connect().execute(
                    "INSERT INTO book_depth_aggregate_raw VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        "bookDepth-target-day-two",
                        "BTCUSDT",
                        timestamp_ms,
                        percentage,
                        abs(percentage) + 1.0,
                        (abs(percentage) + 1.0) * 100.0,
                    ],
                )

        evidence = tape_depth_dataset_source_evidence(
            warehouse,
            "BTCUSDT",
            required_start_ms=base_ms,
            required_end_ms=target_end_ms,
            peer_feature_start_ms=base_ms + 300_000,
            peer_feature_end_ms=peer_end_ms,
        )
    finally:
        warehouse.close()

    assert evidence["required_last_period"] == "2026-07-10"
    assert evidence["cross_asset_context"]["required_feature_start_ms"] == (
        base_ms + 300_000
    )
    peer_evidence = evidence["cross_asset_context"]["peers"]
    assert peer_evidence["ETHUSDT"]["required_last_period"] == "2026-07-09"
    assert peer_evidence["SOLUSDT"]["required_last_period"] == "2026-07-09"


def test_tape_depth_source_evidence_rejects_a_missing_trade_archive_day(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        with pytest.raises(ValueError, match="missing"):
            tape_depth_source_evidence(
                warehouse,
                "BTCUSDT",
                required_start_ms=base_ms,
                required_end_ms=base_ms + 86_400_000,
            )
    finally:
        warehouse.close()


def test_tape_depth_slice_reuses_causal_matrix_with_new_source_binding(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        dataset = build_tape_depth_forecast_dataset(
            warehouse,
            symbol="BTCUSDT",
            start_ms=base_ms + 901_000,
            end_ms=base_ms + 1_000_000,
            horizon_seconds=5,
            total_latency_ms=750,
            decision_cadence_seconds=5,
            maximum_rows=100,
        )
    finally:
        warehouse.close()
    evidence = {**dataset.source_evidence, "manifest_fingerprint": "c" * 64}

    sliced = slice_tape_depth_forecast_dataset(
        dataset,
        start_ms=int(dataset.decision_time_ms[5]),
        end_ms=int(dataset.decision_time_ms[14]),
        source_evidence=evidence,
    )

    assert sliced.rows == 10
    assert np.shares_memory(sliced.features, dataset.features)
    assert sliced.source_evidence["manifest_fingerprint"] == "c" * 64
    assert sliced.summary()["dataset_fingerprint"] != dataset.summary()[
        "dataset_fingerprint"
    ]
