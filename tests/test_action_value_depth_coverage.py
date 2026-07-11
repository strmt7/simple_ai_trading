from __future__ import annotations

import json

import duckdb
import pytest

from simple_ai_trading.microstructure_features import (
    AGGREGATE_DEPTH_FEATURE_VERSION,
    microstructure_feature_source_contract,
)
from tools.audit_action_value_depth_coverage import audit_depth_coverage


def _fixture(tmp_path):
    report_path = tmp_path / "report.json"
    warehouse_path = tmp_path / "warehouse.duckdb"
    output_path = tmp_path / "coverage.json"
    certificate_sha = "c" * 64
    cache_key = "a" * 64
    report_path.write_text(
        json.dumps(
            {
                "round": 28,
                "status": "rejected",
                "report_sha256": "b" * 64,
                "corpus_certificate_sha256": certificate_sha,
                "dataset": {"cache_key": cache_key, "rows": 2},
            }
        ),
        encoding="utf-8",
    )
    source_evidence = {
        "feature_source_contract": microstructure_feature_source_contract(
            AGGREGATE_DEPTH_FEATURE_VERSION
        ),
        "corpus_certificate": {
            "certificate_sha256": certificate_sha,
            "required_data_types": ["bookTicker", "trades", "bookDepth"],
        },
    }
    connection = duckdb.connect(str(warehouse_path))
    connection.execute(
        """
        CREATE TABLE microstructure_dataset_cache_manifest (
            cache_key VARCHAR, feature_version VARCHAR, row_count BIGINT,
            source_evidence_json VARCHAR, dataset_fingerprint VARCHAR,
            rows_table VARCHAR
        );
        CREATE TABLE depth_rows (
            cache_key VARCHAR,
            aggregate_depth_available FLOAT,
            aggregate_depth_age_seconds FLOAT,
            log_bid_notional_within_1pct FLOAT,
            log_ask_notional_within_1pct FLOAT,
            log_bid_notional_within_5pct FLOAT,
            log_ask_notional_within_5pct FLOAT,
            aggregate_depth_notional_imbalance_1pct FLOAT,
            aggregate_depth_notional_imbalance_5pct FLOAT,
            bid_depth_concentration_1pct_to_5pct FLOAT,
            ask_depth_concentration_1pct_to_5pct FLOAT,
            aggregate_depth_concentration_skew FLOAT
        );
        """
    )
    connection.execute(
        "INSERT INTO microstructure_dataset_cache_manifest VALUES (?, ?, 2, ?, ?, ?)",
        [
            cache_key,
            AGGREGATE_DEPTH_FEATURE_VERSION,
            json.dumps(source_evidence, sort_keys=True),
            "d" * 64,
            "depth_rows",
        ],
    )
    connection.executemany(
        "INSERT INTO depth_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (cache_key, 1.0, 16.0, 10.0, 9.0, 12.0, 11.0, 0.1, 0.2, 0.5, 0.4, 0.1),
            (cache_key, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
    )
    connection.close()
    return report_path, warehouse_path, output_path, cache_key


def test_depth_coverage_audit_binds_masked_rows_and_source_contract(tmp_path) -> None:
    report, warehouse, output, _cache_key = _fixture(tmp_path)

    result = audit_depth_coverage(report, warehouse, output)

    assert result["rows"] == 2
    assert result["available_rows"] == 1
    assert result["unavailable_rows"] == 1
    assert result["available_ratio"] == 0.5
    assert result["full_l2_order_book"] is False
    assert result["queue_position_evidence"] is False
    assert len(str(result["audit_sha256"])) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_depth_coverage_audit_rejects_nonzero_stale_features(tmp_path) -> None:
    report, warehouse, output, cache_key = _fixture(tmp_path)
    connection = duckdb.connect(str(warehouse))
    connection.execute(
        "UPDATE depth_rows SET log_bid_notional_within_1pct = 1.0 "
        "WHERE cache_key = ? AND aggregate_depth_available = 0.0",
        [cache_key],
    )
    connection.close()

    with pytest.raises(ValueError, match="cached feature coverage is invalid"):
        audit_depth_coverage(report, warehouse, output)
