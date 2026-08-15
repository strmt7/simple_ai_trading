from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import duckdb
import pytest

from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
    Round27FeatureRow,
)


_START_MS = 1_786_784_400_000
_RUN_ID = "round27-stage1-a-test"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _rows() -> tuple[Round27FeatureRow, ...]:
    output: list[Round27FeatureRow] = []
    for condition_index in range(2):
        condition_id = "0x" + format(condition_index + 1, "064x")
        event_start = _START_MS + condition_index * 300_000
        for offset in (30_000, 31_000):
            output.append(
                Round27FeatureRow.create(
                    run_id=_RUN_ID,
                    condition_id=condition_id,
                    event_start_ms=event_start,
                    decision_time_ms=event_start + offset,
                    market_prior_probability=0.45 + condition_index * 0.1,
                    values=(float(condition_index),) * len(
                        POLYMARKET_ROUND27_FEATURE_NAMES
                    ),
                    maximum_receipt_wall_ms=event_start + offset,
                    source_chain_sha256=hashlib.sha256(
                        f"source-{condition_index}-{offset}".encode("ascii")
                    ).hexdigest(),
                )
            )
    return tuple(output)


def _row_chain(rows: tuple[Round27FeatureRow, ...]) -> str:
    chain = hashlib.sha256(b"").hexdigest()
    for row in rows:
        chain = hashlib.sha256(
            bytes.fromhex(chain) + bytes.fromhex(row.row_sha256)
        ).hexdigest()
    return chain


def _evidence(rows: tuple[Round27FeatureRow, ...]):
    condition_ids = list(dict.fromkeys(row.condition_id for row in rows))
    audit = {
        "schema_version": "polymarket-condition-replay-audit-v1",
        "run_id": _RUN_ID,
        "target_free": True,
        "eligible_condition_count": len(condition_ids),
        "eligible_condition_ids": condition_ids,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    report = {
        "schema_version": POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
        "run_id": _RUN_ID,
        "condition_audit_sha256": audit["audit_sha256"],
        "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "feature_count": len(POLYMARKET_ROUND27_FEATURE_NAMES),
        "feature_row_count": len(rows),
        "row_chain_sha256": _row_chain(rows),
        "official_resolution_accessed": False,
        "target_accessed": False,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return audit, report


def test_round27_feature_store_roundtrips_exact_target_free_rows(tmp_path) -> None:
    path = tmp_path / "round27-features.duckdb"
    rows = _rows()
    audit, report = _evidence(rows)

    with Round27FeatureStore(path) as store:
        assert store.put_slot(
            slot_id="stage1-a",
            run_id=_RUN_ID,
            rows=rows,
            condition_audit=audit,
            feature_report=report,
        )
        assert not store.put_slot(
            slot_id="stage1-a",
            run_id=_RUN_ID,
            rows=rows,
            condition_audit=audit,
            feature_report=report,
        )
        assert store.load_rows() == rows
        storage_audit = store.audit()
        raw_bytes, compressed_bytes = store.connection.execute(
            """
            SELECT sum(raw_size_bytes), sum(compressed_size_bytes)
            FROM round27_feature_condition
            """
        ).fetchone()

    assert storage_audit["slot_count"] == 1
    assert storage_audit["condition_count"] == 2
    assert storage_audit["row_count"] == 4
    assert storage_audit["target_columns_present"] is False
    assert int(compressed_bytes) < int(raw_bytes)
    with Round27FeatureStore(path, read_only=True) as store:
        assert store.load_rows(slot_id="stage1-a") == rows


def test_round27_feature_store_detects_compressed_payload_tampering(tmp_path) -> None:
    path = tmp_path / "round27-features.duckdb"
    rows = _rows()
    audit, report = _evidence(rows)
    with Round27FeatureStore(path) as store:
        store.put_slot(
            slot_id="stage1-a",
            run_id=_RUN_ID,
            rows=rows,
            condition_audit=audit,
            feature_report=report,
        )
    connection = duckdb.connect(str(path))
    connection.execute(
        "UPDATE round27_feature_condition SET payload = ? WHERE condition_id = ?",
        [b"tampered", rows[0].condition_id],
    )
    connection.close()

    with Round27FeatureStore(path, read_only=True) as store:
        with pytest.raises(ValueError, match="compressed feature payload differs"):
            store.audit()


def test_round27_feature_store_rejects_redefined_slot(tmp_path) -> None:
    path = tmp_path / "round27-features.duckdb"
    rows = _rows()
    audit, report = _evidence(rows)
    with Round27FeatureStore(path) as store:
        store.put_slot(
            slot_id="stage1-a",
            run_id=_RUN_ID,
            rows=rows,
            condition_audit=audit,
            feature_report=report,
        )
        changed_report = dict(report)
        changed_report.pop("report_sha256")
        changed_report["operator_note"] = "changed"
        changed_report["report_sha256"] = _canonical_sha256(changed_report)

        with pytest.raises(ValueError, match="stored feature slot differs"):
            store.put_slot(
                slot_id="stage1-a",
                run_id=_RUN_ID,
                rows=rows,
                condition_audit=audit,
                feature_report=changed_report,
            )


def test_round27_feature_store_rejects_target_bearing_row(tmp_path) -> None:
    rows = _rows()
    audit, report = _evidence(rows)
    target_bearing = (replace(rows[0], target_accessed=True), *rows[1:])

    with Round27FeatureStore(tmp_path / "round27-features.duckdb") as store:
        with pytest.raises(ValueError, match="feature row differs"):
            store.put_slot(
                slot_id="stage1-a",
                run_id=_RUN_ID,
                rows=target_bearing,
                condition_audit=audit,
                feature_report=report,
            )


def test_round27_feature_store_rejects_diagnostic_subset(tmp_path) -> None:
    rows = _rows()
    audit, report = _evidence(rows)
    source_sha256 = audit.pop("audit_sha256")
    audit["diagnostic_scope"] = "eligible_condition_prefix"
    audit["source_audit_sha256"] = source_sha256
    audit["audit_sha256"] = _canonical_sha256(audit)
    report.pop("report_sha256")
    report["condition_audit_sha256"] = audit["audit_sha256"]
    report["report_sha256"] = _canonical_sha256(report)

    with Round27FeatureStore(tmp_path / "round27-features.duckdb") as store:
        with pytest.raises(ValueError, match="feature slot evidence differs"):
            store.put_slot(
                slot_id="stage1-a",
                run_id=_RUN_ID,
                rows=rows,
                condition_audit=audit,
                feature_report=report,
            )
