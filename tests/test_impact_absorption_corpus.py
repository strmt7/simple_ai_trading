from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest

from simple_ai_trading.impact_absorption import SynchronizedDepthBook
from simple_ai_trading.impact_absorption_corpus import (
    ROUND73_CORPUS_CONTRACT_SHA256,
    ROUND73_CORPUS_MINIMUM_SEGMENT_NS,
    ROUND73_CORPUS_RUN_TABLE,
    ROUND73_CORPUS_SCHEMA_VERSION,
    audit_round73_corpus_manifest,
    index_round73_corpus_run,
    round73_corpus_day_coverage,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactRestEvent,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "a" * 32
WALL_BASE = 1_784_058_600_000_000_000
SEGMENT_START = WALL_BASE + 1_000_000_000
SEGMENT_END = SEGMENT_START + ROUND73_CORPUS_MINIMUM_SEGMENT_NS


def _snapshot() -> dict[str, object]:
    return {
        "lastUpdateId": 100,
        "bids": [[f"{100.0 - index * 0.1:.1f}", "2"] for index in range(25)],
        "asks": [[f"{100.1 + index * 0.1:.1f}", "3"] for index in range(25)],
    }


def _record(
    *,
    stream: str,
    connection_id: str,
    monotonic_ns: int,
    payload: object,
) -> ImpactCaptureFrameRecord:
    return ImpactCaptureFrameRecord(
        stream=stream,
        connection_id=connection_id,
        sequence_number=0,
        received_wall_ns=WALL_BASE + monotonic_ns,
        received_monotonic_ns=monotonic_ns,
        raw_text=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
    )


def _seed_terminal_run(
    database,
    *,
    qualified: bool = True,
    schema_version: str = IMPACT_CAPTURE_SCHEMA_VERSION,
) -> None:
    messages: list[ImpactCaptureMessage] = []
    with ImpactAbsorptionStore(database) as store:
        store.start_run(
            run_id=RUN_ID,
            started_wall_ns=WALL_BASE,
            started_monotonic_ns=1,
            config={"purpose": "corpus-unit-test", "credentials": False},
            schema_version=schema_version,
        )
        for index, symbol in enumerate(IMPACT_CAPTURE_SYMBOLS, start=1):
            segment_id = format(index, "x") * 32
            store.start_segment(
                run_id=RUN_ID,
                segment_id=segment_id,
                symbol=symbol,
                started_wall_ns=SEGMENT_START,
                started_monotonic_ns=index,
                snapshot_update_id=100,
                tick_size=0.1,
                clock_offset_ns=0,
                clock_rtt_ns=1,
                cooldown_until_wall_ns=0,
            )
            snapshot = _snapshot()
            messages.append(
                ImpactCaptureMessage(
                    record=_record(
                        stream="binance_futures_rest",
                        connection_id=f"rest:{symbol}",
                        monotonic_ns=index * 100,
                        payload=snapshot,
                    ),
                    event=ImpactRestEvent(
                        event_type="depthSnapshot",
                        request_path="/fapi/v1/depth",
                        request_parameters={"limit": 1000, "symbol": symbol},
                        response_status=200,
                        request_started_wall_ns=WALL_BASE + index * 100 - 1,
                        request_started_monotonic_ns=index * 100 - 1,
                        symbol=symbol,
                        update_id=100,
                    ),
                    segment_id=segment_id,
                )
            )
            payload = {
                "e": "depthUpdate",
                "E": 1_001 + index,
                "T": 1_000 + index,
                "s": symbol,
                "U": 101,
                "u": 101,
                "pu": 100,
                "b": [["100.0", "4"]],
                "a": [["100.1", "1"]],
                "st": 1,
                "ps": symbol,
            }
            book = SynchronizedDepthBook(symbol, "0.1")
            book.initialize(snapshot)
            pre_state = book.state()
            depth = book.apply(payload, receive_time_ns=index * 100 + 1)
            messages.append(
                ImpactCaptureMessage(
                    record=_record(
                        stream="binance_futures_public",
                        connection_id=f"public:{symbol}",
                        monotonic_ns=index * 100 + 1,
                        payload={
                            "stream": f"{symbol.lower()}@depth@100ms",
                            "data": payload,
                        },
                    ),
                    event=depth,
                    segment_id=segment_id,
                    pre_l2_state=pre_state,
                    l2_state=book.state(),
                )
            )
        store.append_frame(run_id=RUN_ID, messages=messages)
        for index, _symbol in enumerate(IMPACT_CAPTURE_SYMBOLS, start=1):
            store.finish_segment(
                run_id=RUN_ID,
                segment_id=format(index, "x") * 32,
                status="valid",
                ended_wall_ns=SEGMENT_END,
            )
        store.finish_run(
            run_id=RUN_ID,
            status="completed",
            ended_wall_ns=SEGMENT_END + 1,
        )
        frame_count, message_count, compressed_bytes = (
            store.connect()
            .execute(
                "SELECT frame_count, message_count, compressed_payload_bytes "
                "FROM impact_capture_run WHERE run_id = ?",
                [RUN_ID],
            )
            .fetchone()
        )
        store.record_report(
            run_id=RUN_ID,
            report={
                "schema_version": (
                    IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION
                    if schema_version == IMPACT_CAPTURE_V9_SCHEMA_VERSION
                    else IMPACT_CAPTURE_REPORT_SCHEMA_VERSION
                ),
                "run_id": RUN_ID,
                "status": "completed",
                "capture_gate_passed": qualified,
                "qualification_passed": qualified,
                "elapsed_seconds": 3_600.0,
                "writer_frame_count": frame_count,
                "writer_message_count": message_count,
                "writer_compressed_payload_bytes": compressed_bytes,
                "process_io_write_bytes_per_message": 100.0,
                "database_physical_growth_bytes_per_message": 0.0,
                "queue_maximum_utilization": 0.01,
                "negative_corrected_latency_fraction": 0.0,
                "storage_efficiency_passed": qualified,
                "payload_cap_reached": False,
                "database_size_cap_reached": False,
                "audit_passed": True,
                "audit_errors": [],
            },
            recorded_at_wall_ns=SEGMENT_END + 2,
        )


def test_corpus_indexes_audits_and_reuses_one_qualified_run(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_terminal_run(database)

    manifest = index_round73_corpus_run(database, run_id=RUN_ID)
    repeated = index_round73_corpus_run(database, run_id=RUN_ID)
    audit = audit_round73_corpus_manifest(database, run_id=RUN_ID)

    assert manifest == repeated
    assert manifest.coverage_duration_ns == ROUND73_CORPUS_MINIMUM_SEGMENT_NS
    assert manifest.frame_count == 1
    assert manifest.message_count == 6
    assert audit.passed is True
    assert audit.errors == ()
    with ImpactAbsorptionStore(database, read_only=True) as store:
        row = (
            store.connect()
            .execute(
                f"SELECT schema_version, contract_sha256, count(*) OVER () "
                f"FROM {ROUND73_CORPUS_RUN_TABLE} WHERE run_id = ?",
                [RUN_ID],
            )
            .fetchone()
        )
    assert row == (ROUND73_CORPUS_SCHEMA_VERSION, ROUND73_CORPUS_CONTRACT_SHA256, 1)
    assert manifest.as_dict()["authority"]["model_evaluated"] is False


def test_corpus_indexes_v9_exact_wire_replay_without_typed_tables(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_terminal_run(database, schema_version=IMPACT_CAPTURE_V9_SCHEMA_VERSION)

    manifest = index_round73_corpus_run(database, run_id=RUN_ID)
    audit = audit_round73_corpus_manifest(database, run_id=RUN_ID)

    assert audit.passed is True
    assert manifest.frame_count == 1
    with ImpactAbsorptionStore(database, read_only=True) as store:
        connection = store.connect()
        stored = connection.execute(
            f"SELECT manifest_json, feature_source_json "
            f"FROM {ROUND73_CORPUS_RUN_TABLE} WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()
        assert (
            connection.execute(
                "SELECT count(*) FROM impact_event_link_v8 WHERE run_id = ?",
                [RUN_ID],
            ).fetchone()[0]
            == 0
        )
    identity = json.loads(stored[0])
    feature_source = json.loads(stored[1])
    assert identity["capture_schema_version"] == IMPACT_CAPTURE_V9_SCHEMA_VERSION
    assert feature_source["exact_wire_depth_band_replay_passed"] is True
    assert feature_source["stored_depth_band_row_count"] == 0


def test_corpus_rejects_report_without_qualification_authority(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_terminal_run(database, qualified=False)

    with pytest.raises(ValueError, match="did not pass every gate"):
        index_round73_corpus_run(database, run_id=RUN_ID)


def test_corpus_audit_detects_manifest_tampering(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_terminal_run(database)
    index_round73_corpus_run(database, run_id=RUN_ID)
    with ImpactAbsorptionStore(database) as store:
        store.connect().execute(
            f"UPDATE {ROUND73_CORPUS_RUN_TABLE} SET manifest_json = '{{}}' "
            "WHERE run_id = ?",
            [RUN_ID],
        )

    audit = audit_round73_corpus_manifest(database, run_id=RUN_ID)

    assert audit.passed is False
    assert any(error.startswith("manifest:ValueError") for error in audit.errors)


def test_corpus_day_coverage_is_utc_partition_not_crypto_close(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_terminal_run(database)
    index_round73_corpus_run(database, run_id=RUN_ID)
    selected_day = datetime.fromtimestamp(WALL_BASE / 1_000_000_000, UTC).date()
    finalized_at = datetime.combine(
        selected_day + timedelta(days=1),
        datetime.min.time(),
        tzinfo=UTC,
    )

    diagnostic = round73_corpus_day_coverage(
        database,
        utc_day=selected_day.isoformat(),
        now_wall_ns=int(finalized_at.timestamp() * 1_000_000_000),
    )

    assert diagnostic.finalized is True
    assert diagnostic.eligible is False
    assert diagnostic.coverage_ns == ROUND73_CORPUS_MINIMUM_SEGMENT_NS
    assert diagnostic.contributing_run_ids == (RUN_ID,)
    assert diagnostic.as_dict()["crypto_formal_daily_close"] is False


def test_corpus_day_coverage_rejects_index_column_tampering(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_terminal_run(database)
    index_round73_corpus_run(database, run_id=RUN_ID)
    selected_day = datetime.fromtimestamp(WALL_BASE / 1_000_000_000, UTC).date()
    with ImpactAbsorptionStore(database) as store:
        store.connect().execute(
            f"UPDATE {ROUND73_CORPUS_RUN_TABLE} "
            "SET coverage_end_wall_ns = coverage_end_wall_ns + 1 WHERE run_id = ?",
            [RUN_ID],
        )

    with pytest.raises(ValueError, match="columns differ"):
        round73_corpus_day_coverage(database, utc_day=selected_day.isoformat())
