from __future__ import annotations

import json

import pytest

from simple_ai_trading.impact_absorption import SynchronizedDepthBook
from simple_ai_trading.impact_absorption_features import (
    ROUND73_FEATURE_SOURCE_SCHEMA_VERSION,
    ROUND73_FEATURE_SOURCE_V9_SCHEMA_VERSION,
    diagnose_round73_feature_source,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_INITIAL_COOLDOWN_NS,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    IMPACT_DEPTH_BAND_FLOW_TABLE,
    IMPACT_EVENT_LINK_TABLE,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactRestEvent,
    _canonical_sha256,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "f" * 32
WALL_BASE = 1_784_058_600_000_000_000
FEATURE_READY_WALL_NS = WALL_BASE + 400


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
    sequence_number: int = 0,
) -> ImpactCaptureFrameRecord:
    return ImpactCaptureFrameRecord(
        stream=stream,
        connection_id=connection_id,
        sequence_number=sequence_number,
        received_wall_ns=WALL_BASE + monotonic_ns,
        received_monotonic_ns=monotonic_ns,
        raw_text=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
    )


def _seed_qualified_current_run(
    database,
    *,
    schema_version: str = IMPACT_CAPTURE_SCHEMA_VERSION,
    include_pre_ready_depth: bool = False,
) -> None:
    messages: list[ImpactCaptureMessage] = []
    with ImpactAbsorptionStore(database) as store:
        store.start_run(
            run_id=RUN_ID,
            started_wall_ns=WALL_BASE,
            started_monotonic_ns=1,
            config={"purpose": "feature-replay-unit-test", "credentials": False},
            schema_version=schema_version,
        )
        for index, symbol in enumerate(IMPACT_CAPTURE_SYMBOLS, start=1):
            segment_id = format(index, "x") * 32
            store.start_segment(
                run_id=RUN_ID,
                segment_id=segment_id,
                symbol=symbol,
                started_wall_ns=WALL_BASE + index,
                started_monotonic_ns=index + 1,
                snapshot_update_id=100,
                tick_size=0.1,
                clock_offset_ns=0,
                clock_rtt_ns=1,
                cooldown_until_wall_ns=(
                    FEATURE_READY_WALL_NS + IMPACT_CAPTURE_INITIAL_COOLDOWN_NS
                ),
            )
            snapshot = _snapshot()
            book = SynchronizedDepthBook(symbol, "0.1")
            book.initialize(snapshot)
            if include_pre_ready_depth:
                pre_ready_payload = {
                    "e": "depthUpdate",
                    "E": 901 + index,
                    "T": 900 + index,
                    "s": symbol,
                    "U": 101,
                    "u": 101,
                    "pu": 100,
                    "b": [["100.0", "5"]],
                    "a": [["100.1", "4"]],
                    "st": 1,
                    "ps": symbol,
                }
                pre_ready_state = book.state()
                pre_ready = book.apply(
                    pre_ready_payload,
                    receive_time_ns=50 + index,
                )
                messages.append(
                    ImpactCaptureMessage(
                        record=_record(
                            stream="binance_futures_public",
                            connection_id=f"public:{symbol}",
                            monotonic_ns=50 + index,
                            payload={
                                "stream": f"{symbol.lower()}@depth@100ms",
                                "data": pre_ready_payload,
                            },
                        ),
                        event=pre_ready,
                        segment_id=segment_id,
                        pre_l2_state=pre_ready_state,
                        l2_state=book.state(),
                    )
                )
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
                "U": 102 if include_pre_ready_depth else 101,
                "u": 102 if include_pre_ready_depth else 101,
                "pu": 101 if include_pre_ready_depth else 100,
                "b": [["100.0", "4"]],
                "a": [["100.1", "1"]],
                "st": 1,
                "ps": symbol,
            }
            pre_state = book.state()
            eligible_mono = 400 + index
            depth = book.apply(payload, receive_time_ns=eligible_mono)
            messages.append(
                ImpactCaptureMessage(
                    record=_record(
                        stream="binance_futures_public",
                        connection_id=f"public:{symbol}",
                        monotonic_ns=eligible_mono,
                        payload={
                            "stream": f"{symbol.lower()}@depth@100ms",
                            "data": payload,
                        },
                        sequence_number=1 if include_pre_ready_depth else 0,
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
                ended_wall_ns=WALL_BASE + 1_000 + index,
            )
        store.finish_run(
            run_id=RUN_ID,
            status="completed",
            ended_wall_ns=WALL_BASE + 2_000,
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
                "qualification_passed": True,
            },
            recorded_at_wall_ns=WALL_BASE + 2_001,
        )


def test_current_feature_replay_reconciles_every_stored_depth_band_row(
    tmp_path,
) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_qualified_current_run(database)

    diagnostic = diagnose_round73_feature_source(database, run_id=RUN_ID)

    assert diagnostic.as_dict()["schema_version"] == (
        ROUND73_FEATURE_SOURCE_SCHEMA_VERSION
    )
    assert diagnostic.depth_update_count == 3
    assert diagnostic.stored_depth_band_row_count == 3
    assert diagnostic.stored_depth_band_rows_reconciled is True


def test_v9_feature_replay_rebuilds_depth_bands_without_typed_duplicates(
    tmp_path,
) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_qualified_current_run(
        database,
        schema_version=IMPACT_CAPTURE_V9_SCHEMA_VERSION,
        include_pre_ready_depth=True,
    )

    diagnostic = diagnose_round73_feature_source(database, run_id=RUN_ID)
    payload = diagnostic.as_dict()

    assert payload["schema_version"] == ROUND73_FEATURE_SOURCE_V9_SCHEMA_VERSION
    assert diagnostic.depth_update_count == 6
    assert diagnostic.pre_ready_depth_update_count == 3
    assert diagnostic.feature_eligible_depth_update_count == 3
    assert diagnostic.feature_ready_wall_ns == FEATURE_READY_WALL_NS
    assert diagnostic.synchronized_depth_update_count == 6
    assert diagnostic.level_change_count == 6
    assert diagnostic.stored_depth_band_row_count == 0
    assert diagnostic.stored_depth_band_rows_reconciled is False
    assert diagnostic.depth_band_projection_source == "exact_wire_replay"
    assert diagnostic.exact_wire_depth_band_replay_passed is True
    assert diagnostic.causal_exact_wire_depth_band_replay_passed is True
    assert payload["authority"]["exact_wire_depth_band_replay_passed"] is True
    assert payload["authority"]["causal_exact_wire_depth_band_replay_passed"] is True
    assert all(
        state["pre_ready_depth_updates"] == 1
        and state["feature_eligible_depth_updates"] == 1
        and state["feature_eligible_synchronized_depth_updates"] == 1
        and state["level_changes"] == 2
        for state in diagnostic.symbols.values()
    )


def test_current_feature_replay_rejects_hash_consistent_false_band_row(
    tmp_path,
) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_qualified_current_run(database)
    with ImpactAbsorptionStore(database) as store:
        connection = store.connect()
        frame_index, message_index = connection.execute(
            f"SELECT frame_index, message_index FROM {IMPACT_EVENT_LINK_TABLE} "
            "WHERE run_id = ? AND event_type = 'depthUpdate' AND symbol = 'BTCUSDT'",
            [RUN_ID],
        ).fetchone()
        connection.execute(
            f"UPDATE {IMPACT_DEPTH_BAND_FLOW_TABLE} "
            "SET bid_added_quote_levels_1_5 = bid_added_quote_levels_1_5 + 1 "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [RUN_ID, frame_index, message_index],
        )
        typed_row, l2_row = store._stored_typed_rows(
            connection,
            run_id=RUN_ID,
            frame_index=int(frame_index),
            message_index=int(message_index),
            event_type="depthUpdate",
            schema_version=IMPACT_CAPTURE_SCHEMA_VERSION,
        )
        depth_band_row = tuple(
            connection.execute(
                f"SELECT * FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} "
                "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
                [RUN_ID, frame_index, message_index],
            ).fetchone()
        )
        digest = _canonical_sha256(
            {
                "event_type": "depthUpdate",
                "typed_row": typed_row,
                "l2_row": l2_row,
                "depth_band_row": depth_band_row,
            }
        )
        connection.execute(
            f"UPDATE {IMPACT_EVENT_LINK_TABLE} SET typed_event_sha256 = ? "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [bytes.fromhex(digest), RUN_ID, frame_index, message_index],
        )
        assert store.audit_run(RUN_ID).passed is True

    with pytest.raises(ValueError, match="stored depth band"):
        diagnose_round73_feature_source(database, run_id=RUN_ID)
