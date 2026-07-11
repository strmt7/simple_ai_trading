"""Tests for the source-bound exact-BBO dataset cache."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import simple_ai_trading.microstructure_cache as cache_module
from simple_ai_trading.microstructure_cache import (
    load_microstructure_dataset_cache,
    microstructure_dataset_cache_key,
    microstructure_dataset_fingerprint,
    save_microstructure_dataset_cache,
)
from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MICROSTRUCTURE_TRADE_EMBARGO_MS,
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse


def _source_evidence() -> dict[str, object]:
    return {
        "verified": True,
        "is_current": True,
        "manifest_fingerprint": "a" * 64,
        "build_id": "b" * 64,
        "corpus_certificate": {
            "status": "pass",
            "verified": True,
            "certificate_sha256": "c" * 64,
        },
    }


def _dataset(rows: int = 8) -> MicrostructureDataset:
    decisions = np.arange(rows, dtype=np.int64) * 5_000 + 10_000
    entry_bid = np.linspace(99.9, 100.6, rows)
    entry_ask = entry_bid + 0.1
    exit_bid = entry_bid + 0.2
    exit_ask = entry_ask + 0.2
    features = np.arange(
        rows * len(MICROSTRUCTURE_FEATURE_NAMES),
        dtype=np.float32,
    ).reshape(rows, len(MICROSTRUCTURE_FEATURE_NAMES))
    return MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=300,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=1.0,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=5,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=decisions,
        long_exit_time_ms=decisions + 300_750,
        short_exit_time_ms=decisions + 300_750,
        features=features,
        long_net_bps=np.linspace(-2.0, 3.0, rows),
        short_net_bps=np.linspace(1.0, -4.0, rows),
        entry_spread_bps=np.full(rows, 10.0),
        exit_spread_bps=np.full(rows, 10.0),
        entry_quote_age_ms=np.full(rows, 25, dtype=np.int64),
        exit_quote_age_ms=np.full(rows, 30, dtype=np.int64),
        entry_bid_price=entry_bid,
        entry_ask_price=entry_ask,
        fixed_exit_bid_price=exit_bid,
        fixed_exit_ask_price=exit_ask,
        entry_bid_qty=np.full(rows, 100.0),
        entry_ask_qty=np.full(rows, 100.0),
        fixed_exit_bid_qty=np.full(rows, 100.0),
        fixed_exit_ask_qty=np.full(rows, 100.0),
        long_l1_participation=np.full(rows, 0.10),
        short_l1_participation=np.full(rows, 0.10),
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
        source_evidence=_source_evidence(),
        trade_feature_embargo_ms=MICROSTRUCTURE_TRADE_EMBARGO_MS,
    )


def _parameters(dataset: MicrostructureDataset) -> dict[str, object]:
    return {
        "symbol": dataset.symbol,
        "requested_start_ms": 0,
        "requested_end_ms": 1_000_000,
        "horizon_seconds": dataset.horizon_seconds,
        "total_latency_ms": dataset.total_latency_ms,
        "taker_fee_bps": dataset.taker_fee_bps,
        "additional_slippage_bps_per_side": (
            dataset.additional_slippage_bps_per_side
        ),
        "reference_order_notional_quote": dataset.reference_order_notional_quote,
        "max_l1_participation": dataset.max_l1_participation,
        "max_quote_age_ms": dataset.max_quote_age_ms,
        "decision_cadence_seconds": dataset.decision_cadence_seconds,
        "require_full_history_inventory": False,
        "source_evidence": dataset.source_evidence,
    }


def _rows_table(warehouse: MicrostructureWarehouse) -> str:
    return next(
        str(row[0])
        for row in warehouse.connect().execute("SHOW TABLES").fetchall()
        if str(row[0]).startswith("microstructure_dataset_cache_rows_")
    )


def test_exact_bbo_cache_round_trip_and_collision_detection(tmp_path) -> None:
    dataset = _dataset()
    with MicrostructureWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        cache_key = save_microstructure_dataset_cache(
            warehouse,
            dataset,
            requested_start_ms=0,
            requested_end_ms=1_000_000,
            require_full_history_inventory=False,
        )
        assert cache_key == microstructure_dataset_cache_key(**_parameters(dataset))
        assert (
            save_microstructure_dataset_cache(
                warehouse,
                dataset,
                requested_start_ms=0,
                requested_end_ms=1_000_000,
                require_full_history_inventory=False,
            )
            == cache_key
        )
        loaded = load_microstructure_dataset_cache(warehouse, **_parameters(dataset))
        assert loaded is not None
        assert microstructure_dataset_fingerprint(loaded) == (
            microstructure_dataset_fingerprint(dataset)
        )
        np.testing.assert_array_equal(loaded.features, dataset.features)
        np.testing.assert_array_equal(
            loaded.long_liquidity_eligible,
            dataset.long_liquidity_eligible,
        )

        changed_features = dataset.features.copy()
        changed_features[0, 0] += 1.0
        with pytest.raises(ValueError, match="collides with different evidence"):
            save_microstructure_dataset_cache(
                warehouse,
                replace(dataset, features=changed_features),
                requested_start_ms=0,
                requested_end_ms=1_000_000,
                require_full_history_inventory=False,
            )


def test_exact_bbo_cache_detects_row_tampering(tmp_path) -> None:
    dataset = _dataset()
    with MicrostructureWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        cache_key = save_microstructure_dataset_cache(
            warehouse,
            dataset,
            requested_start_ms=0,
            requested_end_ms=1_000_000,
            require_full_history_inventory=False,
        )
        rows_table = _rows_table(warehouse)
        warehouse.connect().execute(
            f"UPDATE {rows_table} SET long_net_bps = long_net_bps + 1 "
            "WHERE cache_key = ? AND decision_time_ms = ?",
            [cache_key, int(dataset.decision_time_ms[0])],
        )
        with pytest.raises(ValueError, match="fingerprint differs"):
            load_microstructure_dataset_cache(warehouse, **_parameters(dataset))


def test_exact_bbo_cache_rejects_unverified_source(tmp_path) -> None:
    dataset = replace(_dataset(), source_evidence={"verified": False})
    with MicrostructureWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        with pytest.raises(ValueError, match="current verified source evidence"):
            save_microstructure_dataset_cache(
                warehouse,
                dataset,
                requested_start_ms=0,
                requested_end_ms=1_000_000,
                require_full_history_inventory=False,
            )


def test_exact_bbo_cache_key_rejects_empty_interval() -> None:
    parameters = _parameters(_dataset())
    parameters["requested_start_ms"] = 2
    parameters["requested_end_ms"] = 1
    with pytest.raises(ValueError, match="interval is empty"):
        microstructure_dataset_cache_key(**parameters)


def test_exact_bbo_cache_rejects_missing_evidence_and_non_base_dataset(tmp_path) -> None:
    dataset = _dataset()
    with MicrostructureWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        with pytest.raises(ValueError, match="cannot be cached"):
            save_microstructure_dataset_cache(
                warehouse,
                replace(dataset, target_mode="path-aware"),
                requested_start_ms=0,
                requested_end_ms=1_000_000,
                require_full_history_inventory=False,
            )
        with pytest.raises(ValueError, match="has no source evidence"):
            save_microstructure_dataset_cache(
                warehouse,
                replace(dataset, source_evidence=None),
                requested_start_ms=0,
                requested_end_ms=1_000_000,
                require_full_history_inventory=False,
            )


def test_exact_bbo_cache_miss_and_manifest_drift(tmp_path) -> None:
    dataset = _dataset()
    with MicrostructureWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        assert load_microstructure_dataset_cache(
            warehouse,
            **_parameters(dataset),
        ) is None
        cache_key = save_microstructure_dataset_cache(
            warehouse,
            dataset,
            requested_start_ms=0,
            requested_end_ms=1_000_000,
            require_full_history_inventory=False,
        )
        warehouse.connect().execute(
            "UPDATE microstructure_dataset_cache_manifest "
            "SET schema_version = 'changed' WHERE cache_key = ?",
            [cache_key],
        )
        with pytest.raises(ValueError, match="manifest failed"):
            load_microstructure_dataset_cache(warehouse, **_parameters(dataset))


def test_exact_bbo_cache_detects_incomplete_rows(tmp_path) -> None:
    dataset = _dataset()
    with MicrostructureWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        cache_key = save_microstructure_dataset_cache(
            warehouse,
            dataset,
            requested_start_ms=0,
            requested_end_ms=1_000_000,
            require_full_history_inventory=False,
        )
        warehouse.connect().execute(
            f"DELETE FROM {_rows_table(warehouse)} "
            "WHERE cache_key = ? AND decision_time_ms = ?",
            [cache_key, int(dataset.decision_time_ms[0])],
        )
        with pytest.raises(ValueError, match="rows are incomplete"):
            load_microstructure_dataset_cache(warehouse, **_parameters(dataset))


def test_cache_schema_guards_and_masked_array_conversion(tmp_path, monkeypatch) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(cache_module, "_ROWS_TABLE", "Unsafe-Name")
        with pytest.raises(ValueError, match="unsafe identifier"):
            cache_module._validate_identifiers()

    with MicrostructureWarehouse(tmp_path / "manifest.duckdb") as warehouse:
        with monkeypatch.context() as scoped:
            scoped.setattr(cache_module, "_table_schema", lambda *_args: ())
            with pytest.raises(ValueError, match="manifest schema drifted"):
                cache_module._ensure_cache_schema(warehouse)

    with MicrostructureWarehouse(tmp_path / "rows.duckdb") as warehouse:
        schemas = iter((cache_module._MANIFEST_COLUMNS, ()))
        with monkeypatch.context() as scoped:
            scoped.setattr(cache_module, "_table_schema", lambda *_args: next(schemas))
            with pytest.raises(ValueError, match="row schema drifted"):
                cache_module._ensure_cache_schema(warehouse)

    masked = np.ma.asarray([1.0, 2.0])
    masked[1] = np.ma.masked
    converted = cache_module._numeric_array(masked, "<f8")
    assert converted[0] == 1.0
    assert np.isnan(converted[1])
