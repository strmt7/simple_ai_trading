from __future__ import annotations

import hashlib
import json

import pytest

from simple_ai_trading.impact_absorption import (
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_AGGREGATE_TRADE_TABLE,
    IMPACT_BOOK_TICKER_TABLE,
    IMPACT_CAPTURE_FRAME_TABLE,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_DEPTH_BAND_FLOW_TABLE,
    IMPACT_DEPTH_UPDATE_TABLE,
    IMPACT_EVENT_LINK_TABLE,
    IMPACT_L2_STATE_TABLE,
    IMPACT_LIQUIDATION_SNAPSHOT_TABLE,
    IMPACT_MARK_PRICE_TABLE,
    IMPACT_REJECTED_WIRE_EVENT_TABLE,
    IMPACT_REST_EVENT_TABLE,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactRejectedWireEvent,
    ImpactRestEvent,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "0" * 32
SEGMENT_ID = "1" * 32
WALL_BASE = 1_784_058_600_000_000_000


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _rewrite_single_frame_version(connection, schema: str, contract: str) -> str:
    frame = connection.execute(
        """
        SELECT frame_index, previous_frame_sha256, message_count,
               first_message_id, last_message_id, message_manifest_sha256,
               first_received_wall_ns, last_received_wall_ns,
               first_received_monotonic_ns, last_received_monotonic_ns,
               uncompressed_bytes, uncompressed_sha256, compressed_bytes,
               compressed_sha256, stream_counts_json
        FROM impact_capture_frame WHERE run_id = ?
        """,
        [RUN_ID],
    ).fetchone()
    frame_sha256 = _canonical_sha256(
        {
            "schema_version": schema,
            "capture_contract_sha256": contract,
            "run_id": RUN_ID,
            "frame_index": int(frame[0]),
            "previous_frame_sha256": str(frame[1]),
            "message_count": int(frame[2]),
            "first_message_id": str(frame[3]),
            "last_message_id": str(frame[4]),
            "message_manifest_sha256": str(frame[5]),
            "first_received_wall_ns": int(frame[6]),
            "last_received_wall_ns": int(frame[7]),
            "first_received_monotonic_ns": int(frame[8]),
            "last_received_monotonic_ns": int(frame[9]),
            "uncompressed_bytes": int(frame[10]),
            "uncompressed_sha256": str(frame[11]),
            "compressed_bytes": int(frame[12]),
            "compressed_sha256": str(frame[13]),
            "stream_counts_json": str(frame[14]),
        }
    )
    connection.execute(
        "UPDATE impact_capture_frame SET schema_version = ?, frame_sha256 = ? "
        "WHERE run_id = ?",
        [schema, frame_sha256, RUN_ID],
    )
    connection.execute(
        "UPDATE impact_capture_run SET schema_version = ?, "
        "capture_contract_sha256 = ?, last_frame_sha256 = ? WHERE run_id = ?",
        [schema, contract, frame_sha256, RUN_ID],
    )
    return frame_sha256


def _copy_current_frame_to_legacy(connection) -> None:
    connection.execute(
        f"INSERT INTO impact_capture_frame SELECT * "
        f"FROM {IMPACT_CAPTURE_FRAME_TABLE} WHERE run_id = ?",
        [RUN_ID],
    )


def _copy_current_typed_rows_to_v3(connection) -> None:
    for legacy_table, current_table in (
        ("impact_depth_update_v3", IMPACT_DEPTH_UPDATE_TABLE),
        ("impact_l2_state_v3", IMPACT_L2_STATE_TABLE),
        ("impact_book_ticker_v3", IMPACT_BOOK_TICKER_TABLE),
        ("impact_aggregate_trade_v3", IMPACT_AGGREGATE_TRADE_TABLE),
        ("impact_mark_price_v3", IMPACT_MARK_PRICE_TABLE),
        ("impact_liquidation_snapshot_v3", IMPACT_LIQUIDATION_SNAPSHOT_TABLE),
        ("impact_rest_event_v3", IMPACT_REST_EVENT_TABLE),
        ("impact_rejected_wire_event_v3", IMPACT_REJECTED_WIRE_EVENT_TABLE),
    ):
        connection.execute(
            f"INSERT INTO {legacy_table} SELECT * FROM {current_table} "
            "WHERE run_id = ?",
            [RUN_ID],
        )


def _copy_current_rows_to_v5_layout(connection) -> None:
    _copy_current_frame_to_legacy(connection)
    _copy_current_typed_rows_to_v3(connection)
    connection.execute(
        f"INSERT INTO impact_event_link_v5 SELECT * "
        f"FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?",
        [RUN_ID],
    )
    connection.execute(
        f"INSERT INTO impact_depth_band_flow_v5 SELECT * "
        f"FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} WHERE run_id = ?",
        [RUN_ID],
    )


def _legacy_typed_hashes(
    store: ImpactAbsorptionStore,
) -> dict[tuple[int, int], str]:
    links = store.connect().execute(
        f"SELECT frame_index, message_index, event_type "
        f"FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?",
        [RUN_ID],
    ).fetchall()
    output = {}
    for frame_index, message_index, event_type in links:
        typed_row, l2_row = store._stored_typed_rows(
            store.connect(),
            run_id=RUN_ID,
            frame_index=int(frame_index),
            message_index=int(message_index),
            event_type=str(event_type),
            schema_version=IMPACT_CAPTURE_SCHEMA_VERSION,
        )
        output[(int(frame_index), int(message_index))] = _canonical_sha256(
            {
                "event_type": str(event_type),
                "typed_row": typed_row,
                "l2_row": l2_row,
            }
        )
    return output


def _snapshot() -> dict[str, object]:
    return {
        "lastUpdateId": 100,
        "bids": [[f"{100.0 - index * 0.1:.1f}", "2"] for index in range(25)],
        "asks": [[f"{100.1 + index * 0.1:.1f}", "3"] for index in range(25)],
    }


def _record(
    stream: str,
    sequence: int,
    monotonic_ns: int,
    payload: object,
) -> ImpactCaptureFrameRecord:
    return ImpactCaptureFrameRecord(
        stream=stream,
        connection_id=f"{stream}:test",
        sequence_number=sequence,
        received_wall_ns=WALL_BASE + monotonic_ns,
        received_monotonic_ns=monotonic_ns,
        raw_text=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
    )


def _start(store: ImpactAbsorptionStore, *, cap: int = 1_000_000) -> None:
    store.start_run(
        run_id=RUN_ID,
        started_wall_ns=WALL_BASE,
        started_monotonic_ns=1,
        config={"purpose": "unit-test", "credentials": False},
        compressed_payload_cap_bytes=cap,
    )
    store.start_segment(
        run_id=RUN_ID,
        segment_id=SEGMENT_ID,
        symbol="BTCUSDT",
        started_wall_ns=WALL_BASE + 1,
        started_monotonic_ns=2,
        snapshot_update_id=100,
        tick_size=0.1,
        clock_offset_ns=-5_000_000,
        clock_rtt_ns=10_000_000,
        cooldown_until_wall_ns=0,
    )


def _messages() -> tuple[ImpactCaptureMessage, ...]:
    depth_payload = {
        "e": "depthUpdate",
        "E": 1_001,
        "T": 1_000,
        "s": "BTCUSDT",
        "U": 101,
        "u": 102,
        "pu": 100,
        "b": [["100.0", "4"]],
        "a": [["100.1", "1"]],
        "st": 1,
        "ps": "BTCUSDT",
    }
    book = SynchronizedDepthBook("BTCUSDT", "0.1")
    book.initialize(_snapshot())
    pre_state = book.state()
    depth = book.apply(depth_payload, receive_time_ns=100)

    ticker_payload = {
        "e": "bookTicker",
        "E": 1_010,
        "T": 1_009,
        "s": "BTCUSDT",
        "u": 103,
        "b": "100.0",
        "B": "4",
        "a": "100.1",
        "A": "1",
        "st": 1,
        "ps": "BTCUSDT",
    }
    trade_payload = {
        "e": "aggTrade",
        "E": 1_020,
        "T": 1_019,
        "s": "BTCUSDT",
        "a": 500,
        "p": "100.1",
        "q": "2",
        "nq": "2",
        "f": 700,
        "l": 702,
        "m": False,
        "st": 1,
    }
    mark_payload = {
        "e": "markPriceUpdate",
        "E": 1_030,
        "T": 2_000,
        "s": "BTCUSDT",
        "p": "100.0",
        "i": "99.9",
        "P": "0",
        "r": "-0.0001",
        "st": 1,
    }
    liquidation_payload = {
        "e": "forceOrder",
        "E": 2_000,
        "o": {
            "s": "BTCUSDT",
            "ps": "BTCUSDT",
            "st": 1,
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "4",
            "p": "99",
            "ap": "98.5",
            "X": "FILLED",
            "l": "1",
            "z": "4",
            "T": 1_999,
        },
    }
    snapshot = _snapshot()
    return (
        ImpactCaptureMessage(
            record=_record(
                "binance_futures_public",
                0,
                100,
                {"stream": "btcusdt@depth@100ms", "data": depth_payload},
            ),
            event=depth,
            segment_id=SEGMENT_ID,
            pre_l2_state=pre_state,
            l2_state=book.state(),
        ),
        ImpactCaptureMessage(
            record=_record(
                "binance_futures_public",
                1,
                101,
                {"stream": "btcusdt@bookTicker", "data": ticker_payload},
            ),
            event=parse_book_ticker(
                ticker_payload,
                symbol="BTCUSDT",
                receive_time_ns=101,
            ),
            segment_id=SEGMENT_ID,
        ),
        ImpactCaptureMessage(
            record=_record(
                "binance_futures_market",
                0,
                200,
                {"stream": "btcusdt@aggTrade", "data": trade_payload},
            ),
            event=parse_aggregate_trade(
                trade_payload,
                symbol="BTCUSDT",
                receive_time_ns=200,
            ),
            segment_id=SEGMENT_ID,
        ),
        ImpactCaptureMessage(
            record=_record(
                "binance_futures_market",
                1,
                201,
                {"stream": "btcusdt@markPrice@1s", "data": mark_payload},
            ),
            event=parse_mark_price(
                mark_payload,
                symbol="BTCUSDT",
                receive_time_ns=201,
            ),
            segment_id=SEGMENT_ID,
        ),
        ImpactCaptureMessage(
            record=_record(
                "binance_futures_market",
                2,
                202,
                {"stream": "btcusdt@forceOrder", "data": liquidation_payload},
            ),
            event=parse_liquidation_snapshot(
                liquidation_payload,
                symbol="BTCUSDT",
                receive_time_ns=202,
            ),
            segment_id=SEGMENT_ID,
        ),
        ImpactCaptureMessage(
            record=_record("binance_futures_rest", 0, 300, snapshot),
            event=ImpactRestEvent(
                event_type="depthSnapshot",
                request_path="/fapi/v1/depth",
                request_parameters={"limit": 1000, "symbol": "BTCUSDT"},
                response_status=200,
                request_started_wall_ns=WALL_BASE + 290,
                request_started_monotonic_ns=290,
                symbol="BTCUSDT",
                update_id=100,
            ),
            segment_id=SEGMENT_ID,
        ),
    )


def test_store_atomically_links_exact_frames_typed_rows_and_top20_state(
    tmp_path,
) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)

        written = store.append_frame(run_id=RUN_ID, messages=_messages())
        audit = store.audit_run(RUN_ID)

        assert written.frame_index == 0
        assert written.message_count == 6
        assert written.payload_cap_reached is False
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.last_frame_sha256 == written.frame_sha256
        connection = store.connect()
        assert connection.execute(
            "SELECT current_setting('checkpoint_threshold')"
        ).fetchone()[0] == "16.0 MiB"
        assert connection.execute(
            "SELECT current_setting('auto_checkpoint_skip_wal_threshold')"
        ).fetchone()[0] == 100_000
        assert (
            connection.execute(
                f"SELECT count(*) FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?",
                [RUN_ID],
            ).fetchone()[0]
            == 6
        )
        assert (
            connection.execute(
                f"SELECT count(*) FROM {IMPACT_L2_STATE_TABLE} WHERE run_id = ?",
                [RUN_ID],
            ).fetchone()[0]
            == 1
        )
        bid_prices, ask_prices = connection.execute(
            f"SELECT bid_prices, ask_prices FROM {IMPACT_L2_STATE_TABLE} "
            "WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()
        assert len(bid_prices) == len(ask_prices) == 20
        assert connection.execute(
            f"SELECT funding_rate FROM {IMPACT_MARK_PRICE_TABLE} WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()[0] == pytest.approx(-0.0001)
        assert connection.execute(
            f"SELECT count(*) FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()[0] == 1
        band_values = connection.execute(
            f"SELECT bid_added_quote_levels_1_5, "
            "bid_removed_quote_levels_1_5, bid_change_count_levels_1_5, "
            "ask_added_quote_levels_1_5, ask_removed_quote_levels_1_5, "
            f"ask_change_count_levels_1_5 FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} "
            "WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()
        assert band_values == pytest.approx((200.0, 0.0, 1, 0.0, 200.2, 1))
        raw_digest, typed_digest = connection.execute(
            f"SELECT raw_payload_sha256, typed_event_sha256 "
            f"FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ? LIMIT 1",
            [RUN_ID],
        ).fetchone()
        assert isinstance(raw_digest, bytes) and len(raw_digest) == 32
        assert isinstance(typed_digest, bytes) and len(typed_digest) == 32
        link_columns = {
            str(row[1]): str(row[2])
            for row in connection.execute(
                f"PRAGMA table_info('{IMPACT_EVENT_LINK_TABLE}')"
            ).fetchall()
        }
        assert "message_id" not in link_columns
        assert link_columns["event_time_ms"] == "BIGINT"
        assert link_columns["message_index"] == "USMALLINT"
        assert link_columns["raw_payload_sha256"] == "BLOB"
        assert (
            connection.execute(
                "SELECT count(*) FROM duckdb_constraints() "
                "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
                [IMPACT_EVENT_LINK_TABLE],
            ).fetchone()[0]
            == 0
        )
        assert IMPACT_CAPTURE_FRAME_TABLE == "impact_capture_frame_v8"
        assert connection.execute(
            "SELECT count(*) FROM impact_capture_frame WHERE run_id = ?", [RUN_ID]
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT count(*) FROM impact_depth_update_v3 WHERE run_id = ?", [RUN_ID]
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT count(*) FROM duckdb_constraints() "
            "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
            [IMPACT_CAPTURE_FRAME_TABLE],
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT count(*) FROM duckdb_constraints() "
            "WHERE table_name = ? AND constraint_type = 'CHECK'",
            [IMPACT_EVENT_LINK_TABLE],
        ).fetchone()[0] == 2


def test_store_keeps_v2_rows_replay_auditable_without_rewriting_them(tmp_path) -> None:
    v2_schema = "round-073-prospective-evidence-v2"
    v2_contract = "1b46f178e335b3473b86ee71a113e2538a9068e287c50f0867aab13f3230557c"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        messages = _messages()
        store.append_frame(run_id=RUN_ID, messages=messages)
        connection = store.connect()
        legacy_hashes = _legacy_typed_hashes(store)
        _copy_current_frame_to_legacy(connection)

        table_pairs = (
            ("impact_depth_update", IMPACT_DEPTH_UPDATE_TABLE),
            ("impact_l2_state", IMPACT_L2_STATE_TABLE),
            ("impact_book_ticker", IMPACT_BOOK_TICKER_TABLE),
            ("impact_aggregate_trade", IMPACT_AGGREGATE_TRADE_TABLE),
            ("impact_mark_price", IMPACT_MARK_PRICE_TABLE),
            ("impact_liquidation_snapshot", IMPACT_LIQUIDATION_SNAPSHOT_TABLE),
            ("impact_rest_event", IMPACT_REST_EVENT_TABLE),
            ("impact_rejected_wire_event", IMPACT_REJECTED_WIRE_EVENT_TABLE),
        )
        for legacy_table, current_table in table_pairs:
            connection.execute(
                f"INSERT INTO {legacy_table} SELECT * FROM {current_table} "
                "WHERE run_id = ?",
                [RUN_ID],
            )

        links = connection.execute(
            f"SELECT frame_index, message_index, segment_id, stream, connection_id, "
            "sequence_number, received_wall_ns, received_monotonic_ns, "
            "raw_payload_sha256, event_type, symbol, typed_event_sha256 "
            f"FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ? "
            "ORDER BY frame_index, message_index",
            [RUN_ID],
        ).fetchall()
        legacy_rows = []
        for message, link in zip(messages, links, strict=True):
            raw_sha256 = bytes(link[8]).hex()
            message_id = _canonical_sha256(
                {
                    "run_id": RUN_ID,
                    "stream": link[3],
                    "connection_id": link[4],
                    "sequence_number": int(link[5]),
                    "raw_payload_sha256": raw_sha256,
                }
            )
            event_type, symbol, event_time, transaction_time, update_id = (
                store._event_identity(message.event)
            )
            legacy_rows.append(
                (
                    RUN_ID,
                    int(link[0]),
                    int(link[1]),
                    message_id,
                    str(link[2]),
                    str(link[3]),
                    str(link[4]),
                    int(link[5]),
                    int(link[6]),
                    int(link[7]),
                    raw_sha256,
                    event_type,
                    symbol,
                    event_time,
                    transaction_time,
                    update_id,
                    legacy_hashes[(int(link[0]), int(link[1]))],
                )
            )
        connection.executemany(
            "INSERT INTO impact_event_index VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            legacy_rows,
        )

        _rewrite_single_frame_version(connection, v2_schema, v2_contract)
        connection.execute(
            f"DELETE FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?", [RUN_ID]
        )
        for _legacy_table, current_table in table_pairs:
            connection.execute(
                f"DELETE FROM {current_table} WHERE run_id = ?", [RUN_ID]
            )

        audit = store.audit_run(RUN_ID)
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.capture_contract_sha256 == v2_contract


def test_store_keeps_v3_compact_rows_replay_auditable(tmp_path) -> None:
    v3_schema = "round-073-prospective-evidence-v3"
    v3_contract = "9228f8243531e44a264d5f88cf8498282986d1b9cb4a6b64e12ee0cede47dc5b"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        legacy_hashes = _legacy_typed_hashes(store)
        _copy_current_frame_to_legacy(connection)
        _copy_current_typed_rows_to_v3(connection)
        connection.execute(
            f"""
            INSERT INTO impact_event_link_v3
            SELECT run_id, frame_index, message_index, segment_id, stream,
                   connection_id, sequence_number, received_wall_ns,
                   received_monotonic_ns, raw_payload_sha256, event_type, symbol,
                   typed_event_sha256
            FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?
            """,
            [RUN_ID],
        )
        for (frame_index, message_index), digest in legacy_hashes.items():
            connection.execute(
                "UPDATE impact_event_link_v3 SET typed_event_sha256 = ? "
                "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
                [bytes.fromhex(digest), RUN_ID, frame_index, message_index],
            )
        _rewrite_single_frame_version(connection, v3_schema, v3_contract)
        connection.execute(
            f"DELETE FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?", [RUN_ID]
        )

        audit = store.audit_run(RUN_ID)
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.capture_contract_sha256 == v3_contract


def test_store_keeps_v4_event_time_rows_replay_auditable(tmp_path) -> None:
    v4_schema = "round-073-prospective-evidence-v4"
    v4_contract = "c34687c5dff9a4eda98b2e50d6444a12ee1a4f5594806c2410e15cb0242d7529"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        legacy_hashes = _legacy_typed_hashes(store)
        _copy_current_frame_to_legacy(connection)
        _copy_current_typed_rows_to_v3(connection)
        connection.execute(
            f"""
            INSERT INTO impact_event_link_v4
            SELECT run_id, frame_index, message_index, segment_id, stream,
                   connection_id, sequence_number, received_wall_ns,
                   received_monotonic_ns, raw_payload_sha256, event_type, symbol,
                   event_time_ms, typed_event_sha256
            FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?
            """,
            [RUN_ID],
        )
        for (frame_index, message_index), digest in legacy_hashes.items():
            connection.execute(
                "UPDATE impact_event_link_v4 SET typed_event_sha256 = ? "
                "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
                [bytes.fromhex(digest), RUN_ID, frame_index, message_index],
            )
        _rewrite_single_frame_version(connection, v4_schema, v4_contract)
        connection.execute(
            f"DELETE FROM {IMPACT_EVENT_LINK_TABLE} WHERE run_id = ?", [RUN_ID]
        )
        connection.execute(
            f"DELETE FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} WHERE run_id = ?", [RUN_ID]
        )

        audit = store.audit_run(RUN_ID)
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.capture_contract_sha256 == v4_contract


def test_store_keeps_v5_depth_band_rows_replay_auditable(tmp_path) -> None:
    v5_schema = "round-073-prospective-evidence-v5"
    v5_contract = "63a440f1fb875db8ee78bab1631033f24850a65cc7ed80d4fd37078dd6ee9a1b"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        _copy_current_rows_to_v5_layout(connection)

        _rewrite_single_frame_version(connection, v5_schema, v5_contract)

        audit = store.audit_run(RUN_ID)
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.capture_contract_sha256 == v5_contract


def test_store_keeps_v6_telemetry_rows_replay_auditable(tmp_path) -> None:
    v6_schema = "round-073-prospective-evidence-v6"
    v6_contract = "a256f16f1904d6c23b4563e7cbb603353dd7e0fe8253e3c3f2df4a67305da021"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        _copy_current_rows_to_v5_layout(connection)

        _rewrite_single_frame_version(connection, v6_schema, v6_contract)

        audit = store.audit_run(RUN_ID)
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.capture_contract_sha256 == v6_contract


def test_store_keeps_v7_checkpoint_rows_replay_auditable(tmp_path) -> None:
    v7_schema = "round-073-prospective-evidence-v7"
    v7_contract = "18013fc14bad234b241bf05122a6363ad94e6722a598319ae1059cde1941a9f1"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        _copy_current_rows_to_v5_layout(connection)

        _rewrite_single_frame_version(connection, v7_schema, v7_contract)

        audit = store.audit_run(RUN_ID)
        assert audit.passed is True
        assert audit.errors == ()
        assert audit.capture_contract_sha256 == v7_contract


def test_store_rejects_invalid_lane_or_segment_before_any_frame_commits(
    tmp_path,
) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        messages = list(_messages())
        messages[1] = ImpactCaptureMessage(
            **{**messages[1].__dict__, "segment_id": "2" * 32}
        )
        with pytest.raises(ValueError, match="active symbol segment"):
            store.append_frame(run_id=RUN_ID, messages=messages)
        assert (
            store.connect()
            .execute(f"SELECT count(*) FROM {IMPACT_CAPTURE_FRAME_TABLE}")
            .fetchone()[0]
            == 0
        )

        messages = list(_messages())
        messages[1] = ImpactCaptureMessage(
            **{
                **messages[1].__dict__,
                "record": ImpactCaptureFrameRecord(
                    **{**messages[1].record.__dict__, "sequence_number": 2}
                ),
            }
        )
        with pytest.raises(ValueError, match="lane sequence mismatch"):
            store.append_frame(run_id=RUN_ID, messages=messages)
        assert (
            store.connect()
            .execute(f"SELECT count(*) FROM {IMPACT_EVENT_LINK_TABLE}")
            .fetchone()[0]
            == 0
        )


def test_store_detects_payload_and_typed_row_tampering(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        store.connect().execute(
            f"DELETE FROM {IMPACT_AGGREGATE_TRADE_TABLE} WHERE run_id = ?",
            [RUN_ID],
        )
        typed_audit = store.audit_run(RUN_ID)
        assert typed_audit.passed is False
        assert "typed_count_mismatch:aggTrade" in typed_audit.errors

        store.connect().execute(
            f"UPDATE {IMPACT_MARK_PRICE_TABLE} SET funding_rate = 0.5 "
            "WHERE run_id = ?",
            [RUN_ID],
        )
        value_audit = store.audit_run(RUN_ID)
        assert value_audit.passed is False
        assert any(
            error.startswith("typed_sha256_mismatch:") for error in value_audit.errors
        )

        store.connect().execute(
            f"""
            UPDATE {IMPACT_CAPTURE_FRAME_TABLE} SET compressed_payload = from_hex('00')
            WHERE run_id = ? AND frame_index = 0
            """,
            [RUN_ID],
        )
        payload_audit = store.audit_run(RUN_ID)
        assert payload_audit.passed is False
        assert "compressed_size_mismatch:0" in payload_audit.errors
        assert "compressed_sha256_mismatch:0" in payload_audit.errors


def test_store_detects_compact_link_tampering_and_duplicate_rows(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        frame_index, message_index, raw_digest, event_time_ms = connection.execute(
            f"SELECT frame_index, message_index, raw_payload_sha256, event_time_ms "
            f"FROM {IMPACT_EVENT_LINK_TABLE} "
            "WHERE run_id = ? AND event_type = 'aggTrade'",
            [RUN_ID],
        ).fetchone()

        connection.execute(
            f"UPDATE {IMPACT_EVENT_LINK_TABLE} SET event_time_ms = ? "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [event_time_ms + 1, RUN_ID, frame_index, message_index],
        )
        clock_audit = store.audit_run(RUN_ID)
        assert clock_audit.passed is False
        assert f"event_time_link_mismatch:{frame_index}:{message_index}" in (
            clock_audit.errors
        )
        connection.execute(
            f"UPDATE {IMPACT_EVENT_LINK_TABLE} SET event_time_ms = ? "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [event_time_ms, RUN_ID, frame_index, message_index],
        )

        connection.execute(
            f"UPDATE {IMPACT_EVENT_LINK_TABLE} SET raw_payload_sha256 = ? "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [bytes(32), RUN_ID, frame_index, message_index],
        )
        digest_audit = store.audit_run(RUN_ID)
        assert digest_audit.passed is False
        assert f"raw_to_index_mismatch:{frame_index}:{message_index}" in (
            digest_audit.errors
        )

        connection.execute(
            f"UPDATE {IMPACT_EVENT_LINK_TABLE} SET raw_payload_sha256 = ? "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [raw_digest, RUN_ID, frame_index, message_index],
        )
        connection.execute(
            f"INSERT INTO {IMPACT_EVENT_LINK_TABLE} SELECT * "
            f"FROM {IMPACT_EVENT_LINK_TABLE} "
            "WHERE run_id = ? AND frame_index = ? AND message_index = ?",
            [RUN_ID, frame_index, message_index],
        )
        duplicate_audit = store.audit_run(RUN_ID)
        assert duplicate_audit.passed is False
        assert f"event_index_count_mismatch:{frame_index}" in duplicate_audit.errors


def test_store_detects_depth_band_flow_tampering(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        connection = store.connect()
        connection.execute(
            f"UPDATE {IMPACT_DEPTH_BAND_FLOW_TABLE} "
            "SET bid_added_quote_levels_1_5 = bid_added_quote_levels_1_5 + 1 "
            "WHERE run_id = ?",
            [RUN_ID],
        )
        value_audit = store.audit_run(RUN_ID)
        assert value_audit.passed is False
        assert "typed_sha256_mismatch:0:0" in value_audit.errors

        connection.execute(
            f"DELETE FROM {IMPACT_DEPTH_BAND_FLOW_TABLE} WHERE run_id = ?", [RUN_ID]
        )
        count_audit = store.audit_run(RUN_ID)
        assert count_audit.passed is False
        assert "typed_count_mismatch:depthBandFlow" in count_audit.errors


def test_store_allows_one_bounded_cap_crossing_then_blocks_more_frames(
    tmp_path,
) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store, cap=1)
        written = store.append_frame(run_id=RUN_ID, messages=_messages())
        assert written.payload_cap_reached is True
        with pytest.raises(ValueError, match="cap has already been reached"):
            store.append_frame(run_id=RUN_ID, messages=_messages())


def test_store_forbids_secret_bearing_config_and_rest_parameters(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        with pytest.raises(ValueError, match="secret-bearing"):
            store.start_run(
                run_id=RUN_ID,
                started_wall_ns=WALL_BASE,
                started_monotonic_ns=1,
                config={"api_key": "must-not-be-stored"},
            )
        _start(store)
        message = _messages()[-1]
        invalid = ImpactCaptureMessage(
            **{
                **message.__dict__,
                "event": ImpactRestEvent(
                    event_type="depthSnapshot",
                    request_path="/fapi/v1/depth",
                    request_parameters={
                        "symbol": "BTCUSDT",
                        "limit": 1000,
                        "signature": "forbidden",
                    },
                    response_status=200,
                    request_started_wall_ns=WALL_BASE + 290,
                    request_started_monotonic_ns=290,
                    symbol="BTCUSDT",
                    update_id=100,
                ),
            }
        )
        with pytest.raises(ValueError, match="unsupported or secret-bearing"):
            store.append_frame(run_id=RUN_ID, messages=(invalid,))


def test_store_rejects_duplicate_json_and_raw_typed_disagreement(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        ticker = _messages()[1]
        duplicate = ImpactCaptureMessage(
            **{
                **ticker.__dict__,
                "record": ImpactCaptureFrameRecord(
                    **{
                        **ticker.record.__dict__,
                        "sequence_number": 0,
                        "raw_text": '{"stream":"one","stream":"two","data":{}}',
                    }
                ),
            }
        )
        with pytest.raises(ValueError, match="duplicate JSON key"):
            store.append_frame(run_id=RUN_ID, messages=(duplicate,))

        wrapper = json.loads(ticker.record.raw_text)
        wrapper["data"]["a"] = "100.2"
        disagreement = ImpactCaptureMessage(
            **{
                **ticker.__dict__,
                "record": ImpactCaptureFrameRecord(
                    **{
                        **ticker.record.__dict__,
                        "sequence_number": 0,
                        "raw_text": json.dumps(wrapper, separators=(",", ":")),
                    }
                ),
            }
        )
        with pytest.raises(ValueError, match="raw book ticker differs"):
            store.append_frame(run_id=RUN_ID, messages=(disagreement,))

        wrong_stream_wrapper = json.loads(ticker.record.raw_text)
        wrong_stream_wrapper["stream"] = "btcusdt@bookTicker@100ms"
        wrong_stream = ImpactCaptureMessage(
            **{
                **ticker.__dict__,
                "record": ImpactCaptureFrameRecord(
                    **{
                        **ticker.record.__dict__,
                        "sequence_number": 0,
                        "raw_text": json.dumps(
                            wrong_stream_wrapper, separators=(",", ":")
                        ),
                    }
                ),
            }
        )
        with pytest.raises(ValueError, match="combined stream mismatch"):
            store.append_frame(run_id=RUN_ID, messages=(wrong_stream,))
        assert (
            store.connect()
            .execute(f"SELECT count(*) FROM {IMPACT_CAPTURE_FRAME_TABLE}")
            .fetchone()[0]
            == 0
        )


def test_store_hash_chains_valid_json_and_malformed_rejected_wire_evidence(
    tmp_path,
) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        force_raw = (
            '{"stream":"btcusdt@forceOrder","data":{"e":"forceOrder",'
            '"E":2000,"o":{"s":"BTCUSDT","st":2,"ps":"BTCUSDT"}}}'
        )
        records = (
            ImpactCaptureFrameRecord(
                stream="binance_futures_market",
                connection_id="binance-market:rejected-test",
                sequence_number=0,
                received_wall_ns=WALL_BASE + 400,
                received_monotonic_ns=400,
                raw_text=force_raw,
            ),
            ImpactCaptureFrameRecord(
                stream="binance_futures_market",
                connection_id="binance-market:rejected-test",
                sequence_number=1,
                received_wall_ns=WALL_BASE + 401,
                received_monotonic_ns=401,
                raw_text='{"stream":"one","stream":"two","data":{}}',
            ),
        )
        messages = (
            ImpactCaptureMessage(
                record=records[0],
                event=ImpactRejectedWireEvent(
                    observed_stream_name="btcusdt@forceOrder",
                    observed_event_type="forceOrder",
                    observed_symbol="BTCUSDT",
                    rejection_class="feed_integrity",
                    rejection_reason="ImpactFeedIntegrityError:stream type mismatch",
                    receive_time_ns=400,
                ),
                segment_id=SEGMENT_ID,
            ),
            ImpactCaptureMessage(
                record=records[1],
                event=ImpactRejectedWireEvent(
                    observed_stream_name="",
                    observed_event_type="",
                    observed_symbol="",
                    rejection_class="feed_integrity",
                    rejection_reason="ValueError:duplicate JSON key is forbidden",
                    receive_time_ns=401,
                ),
            ),
        )

        written = store.append_frame(run_id=RUN_ID, messages=messages)
        audit = store.audit_run(RUN_ID)

        assert written.message_count == 2
        assert audit.passed is True
        assert (
            store.connect()
            .execute(
                f"SELECT count(*) FROM {IMPACT_REJECTED_WIRE_EVENT_TABLE} "
                "WHERE run_id = ?",
                [RUN_ID],
            )
            .fetchone()[0]
            == 2
        )
        assert (
            store.connect()
            .execute(
                f"SELECT count(*) FROM {IMPACT_EVENT_LINK_TABLE} "
                "WHERE run_id = ? AND event_type = 'rejectedWire'",
                [RUN_ID],
            )
            .fetchone()[0]
            == 2
        )


def test_audit_retains_v1_contract_identity_for_historical_runs(tmp_path) -> None:
    legacy_contract = "f379b53b86d20f16b686132ef8fe4dc5eb47b6a0910e6ba85c38ddf0caa01c7b"
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        store.start_run(
            run_id=RUN_ID,
            started_wall_ns=WALL_BASE,
            started_monotonic_ns=1,
            config={"purpose": "legacy-audit-test"},
        )
        store.connect().execute(
            "UPDATE impact_capture_run SET schema_version = ?, "
            "capture_contract_sha256 = ? WHERE run_id = ?",
            ["round-073-prospective-evidence-v1", legacy_contract, RUN_ID],
        )
        store.finish_run(
            run_id=RUN_ID,
            status="stopped",
            ended_wall_ns=WALL_BASE + 1,
        )

        audit = store.audit_run(RUN_ID)

        assert audit.passed is True
        assert audit.capture_contract_sha256 == legacy_contract


def test_store_persists_exact_typed_open_interest(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        raw = {
            "openInterest": "12345.678",
            "symbol": "BTCUSDT",
            "time": 1_784_058_600_000,
        }
        message = ImpactCaptureMessage(
            record=_record("binance_futures_rest", 0, 300, raw),
            event=ImpactRestEvent(
                event_type="openInterest",
                request_path="/fapi/v1/openInterest",
                request_parameters={"symbol": "BTCUSDT"},
                response_status=200,
                request_started_wall_ns=WALL_BASE + 290,
                request_started_monotonic_ns=290,
                symbol="BTCUSDT",
                exchange_time_ms=1_784_058_600_000,
                open_interest=12345.678,
            ),
            segment_id=SEGMENT_ID,
        )

        store.append_frame(run_id=RUN_ID, messages=(message,))

        assert store.audit_run(RUN_ID).passed is True
        assert store.connect().execute(
            f"SELECT open_interest FROM {IMPACT_REST_EVENT_TABLE} WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()[0] == pytest.approx(12345.678)


def test_terminal_capture_report_is_canonical_hash_bound_and_secret_free(
    tmp_path,
) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.finish_segment(
            run_id=RUN_ID,
            segment_id=SEGMENT_ID,
            status="stopped",
            ended_wall_ns=WALL_BASE + 1_000,
        )
        store.finish_run(
            run_id=RUN_ID,
            status="stopped",
            ended_wall_ns=WALL_BASE + 2_000,
        )
        report = {
            "schema_version": "round-073-capture-report-v8",
            "run_id": RUN_ID,
            "status": "stopped",
            "qualification_passed": False,
        }

        report_sha256 = store.record_report(
            run_id=RUN_ID,
            report=report,
            recorded_at_wall_ns=WALL_BASE + 2_000,
        )

        stored_json, stored_sha256 = (
            store.connect()
            .execute(
                "SELECT report_json, report_sha256 FROM impact_capture_report WHERE run_id = ?",
                [RUN_ID],
            )
            .fetchone()
        )
        assert json.loads(stored_json) == report
        assert stored_sha256 == report_sha256
        with pytest.raises(ValueError, match="secret-bearing"):
            store.record_report(
                run_id=RUN_ID,
                report={**report, "token": "forbidden"},
                recorded_at_wall_ns=WALL_BASE + 3_000,
            )
