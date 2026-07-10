from __future__ import annotations

import math

import numpy as np
import pytest

from simple_ai_trading.microstructure_warehouse import (
    MicrostructureWarehouse,
    TICK_WAREHOUSE_SCHEMA_VERSION,
)
from simple_ai_trading.tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_TARGET_MODE,
    build_tape_depth_forecast_dataset,
    slice_tape_depth_forecast_dataset,
    tape_depth_source_evidence,
)


def _manifest(
    warehouse: MicrostructureWarehouse,
    *,
    archive_id: str,
    data_type: str,
    rows: int,
    derived_rows: int,
    first_ms: int,
    last_ms: int,
) -> None:
    source_hash = ("a" if data_type == "trades" else "b") * 64
    warehouse.connect().execute(
        """
        INSERT INTO archive_manifest (
            archive_id, schema_version, provider, market_type, symbol, data_type,
            period, url, archive_path, status, is_current, expected_bytes,
            compressed_bytes, uncompressed_bytes, source_sha256, expected_sha256,
            checksum_status, rows_read, derived_rows, first_exchange_time_ms,
            last_exchange_time_ms, invalid_rows, duplicate_ids, update_id_regressions,
            event_time_regressions, out_of_order_rows, crossed_books, ingested_at_ms,
            error
        ) VALUES (
            ?, ?, 'binance', 'futures', 'BTCUSDT', ?, '2026-07-09', ?, '',
            'complete', true, 0, 0, 0, ?, ?, 'verified', ?, ?, ?, ?,
            0, 0, 0, 0, 0, 0, 1, ''
        )
        """,
        [
            archive_id,
            TICK_WAREHOUSE_SCHEMA_VERSION,
            data_type,
            f"https://data.binance.vision/{archive_id}.zip",
            source_hash,
            source_hash,
            rows,
            derived_rows,
            first_ms,
            last_ms,
        ],
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


def test_tape_depth_source_evidence_rejects_a_missing_trade_archive_day(tmp_path) -> None:
    warehouse, base_ms, _seconds = _warehouse_fixture(tmp_path)
    try:
        with pytest.raises(ValueError, match="missing day"):
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
