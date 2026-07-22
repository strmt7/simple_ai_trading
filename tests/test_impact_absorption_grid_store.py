from __future__ import annotations

import json

from simple_ai_trading.impact_absorption import (
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_mark_price,
)
from simple_ai_trading.impact_absorption_corpus import (
    ROUND73_CORPUS_MINIMUM_SEGMENT_NS,
    index_round73_corpus_run,
)
from simple_ai_trading.impact_absorption_grid_store import (
    ROUND73_GRID_ANCHOR_TABLE,
    ROUND73_GRID_VECTOR_TABLE,
    audit_round73_causal_grid,
    build_round73_causal_grid,
    _CausalClockTimeline,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactRestEvent,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "d" * 32
WALL_BASE = 1_784_058_600_000_000_000
SEGMENT_START = WALL_BASE + 1
SEGMENT_END = SEGMENT_START + ROUND73_CORPUS_MINIMUM_SEGMENT_NS


def test_clock_timeline_never_uses_a_probe_at_or_after_an_event() -> None:
    timeline = _CausalClockTimeline((100, 200), (7, 9))

    assert timeline.offset_strictly_before(100, 5) == 5
    assert timeline.offset_strictly_before(101, 5) == 7
    assert timeline.offset_strictly_before(200, 5) == 7
    assert timeline.offset_strictly_before(201, 5) == 9


def _snapshot(price: float) -> dict[str, object]:
    return {
        "lastUpdateId": 100,
        "bids": [[f"{price - index * 0.1:.1f}", "2"] for index in range(25)],
        "asks": [[f"{price + 0.1 + index * 0.1:.1f}", "3"] for index in range(25)],
    }


def _record(
    *,
    stream: str,
    connection_id: str,
    sequence_number: int,
    monotonic_ns: int,
    payload: object,
) -> ImpactCaptureFrameRecord:
    return ImpactCaptureFrameRecord(
        stream=stream,
        connection_id=connection_id,
        sequence_number=sequence_number,
        received_wall_ns=WALL_BASE + monotonic_ns,
        received_monotonic_ns=monotonic_ns,
        raw_text=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
    )


def _seed_qualified_run(database) -> None:
    messages: list[ImpactCaptureMessage] = []
    sequences: dict[str, int] = {}

    def record(
        stream: str,
        connection_id: str,
        monotonic_ns: int,
        payload: object,
    ) -> ImpactCaptureFrameRecord:
        sequence = sequences.get(connection_id, 0)
        sequences[connection_id] = sequence + 1
        return _record(
            stream=stream,
            connection_id=connection_id,
            sequence_number=sequence,
            monotonic_ns=monotonic_ns,
            payload=payload,
        )

    prices = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0}
    with ImpactAbsorptionStore(database) as store:
        store.start_run(
            run_id=RUN_ID,
            started_wall_ns=WALL_BASE,
            started_monotonic_ns=1,
            config={"purpose": "causal-grid-unit-test", "credentials": False},
        )
        clock_received_mono = 20_000_000
        clock_started_mono = 18_000_000
        clock_exchange_ms = (WALL_BASE + 19_000_000) // 1_000_000
        clock_payload = {"serverTime": clock_exchange_ms}
        messages.append(
            ImpactCaptureMessage(
                record=record(
                    "binance_futures_rest",
                    "rest:server-time",
                    clock_received_mono,
                    clock_payload,
                ),
                event=ImpactRestEvent(
                    event_type="serverTime",
                    request_path="/fapi/v1/time",
                    request_parameters={},
                    response_status=200,
                    request_started_wall_ns=WALL_BASE + clock_started_mono,
                    request_started_monotonic_ns=clock_started_mono,
                    exchange_time_ms=clock_exchange_ms,
                ),
            )
        )
        for symbol_index, symbol in enumerate(IMPACT_CAPTURE_SYMBOLS, start=1):
            segment_id = format(symbol_index, "x") * 32
            price = prices[symbol]
            store.start_segment(
                run_id=RUN_ID,
                segment_id=segment_id,
                symbol=symbol,
                started_wall_ns=SEGMENT_START,
                started_monotonic_ns=2,
                snapshot_update_id=100,
                tick_size=0.1,
                clock_offset_ns=0,
                clock_rtt_ns=1,
                cooldown_until_wall_ns=0,
            )
            book = SynchronizedDepthBook(symbol, "0.1")
            snapshot = _snapshot(price)
            book.initialize(snapshot)
            snapshot_mono = 100 + symbol_index
            messages.append(
                ImpactCaptureMessage(
                    record=record(
                        "binance_futures_rest",
                        f"rest:{symbol}:depth-snapshot",
                        snapshot_mono,
                        snapshot,
                    ),
                    event=ImpactRestEvent(
                        event_type="depthSnapshot",
                        request_path="/fapi/v1/depth",
                        request_parameters={"limit": 1000, "symbol": symbol},
                        response_status=200,
                        request_started_wall_ns=WALL_BASE + snapshot_mono - 1,
                        request_started_monotonic_ns=snapshot_mono - 1,
                        symbol=symbol,
                        update_id=100,
                    ),
                    segment_id=segment_id,
                )
            )
            for second_index in range(5):
                second_base = (3_595 + second_index) * 1_000_000_000
                symbol_offset = symbol_index * 1_000_000
                open_interest_mono = second_base + 50_000_000 + symbol_offset
                depth_mono = second_base + 100_000_000 + symbol_offset
                bbo_mono = second_base + 200_000_000 + symbol_offset
                trade_mono = second_base + 300_000_000 + symbol_offset
                mark_mono = second_base + 400_000_000 + symbol_offset
                if second_index == 0:
                    open_interest_time_ms = (
                        WALL_BASE + open_interest_mono
                    ) // 1_000_000
                    open_interest_payload = {
                        "openInterest": "1000000",
                        "symbol": symbol,
                        "time": open_interest_time_ms,
                    }
                    messages.append(
                        ImpactCaptureMessage(
                            record=record(
                                "binance_futures_rest",
                                f"rest:{symbol}:open-interest",
                                open_interest_mono,
                                open_interest_payload,
                            ),
                            event=ImpactRestEvent(
                                event_type="openInterest",
                                request_path="/fapi/v1/openInterest",
                                request_parameters={"symbol": symbol},
                                response_status=200,
                                request_started_wall_ns=(
                                    WALL_BASE + open_interest_mono - 1
                                ),
                                request_started_monotonic_ns=open_interest_mono - 1,
                                symbol=symbol,
                                exchange_time_ms=open_interest_time_ms,
                                open_interest=1_000_000.0,
                            ),
                            segment_id=segment_id,
                        )
                    )

                update_id = 101 + second_index
                depth_time_ms = (WALL_BASE + depth_mono) // 1_000_000
                depth_payload = {
                    "e": "depthUpdate",
                    "E": depth_time_ms,
                    "T": depth_time_ms,
                    "s": symbol,
                    "U": update_id,
                    "u": update_id,
                    "pu": update_id - 1,
                    "b": [[f"{price:.1f}", f"{4 + second_index}"]],
                    "a": [[f"{price + 0.1:.1f}", f"{2 + second_index}"]],
                    "st": 1,
                    "ps": symbol,
                }
                pre_state = book.state()
                depth = book.apply(depth_payload, receive_time_ns=depth_mono)
                messages.append(
                    ImpactCaptureMessage(
                        record=record(
                            "binance_futures_public",
                            f"public:{symbol}:depth",
                            depth_mono,
                            {
                                "stream": f"{symbol.lower()}@depth@100ms",
                                "data": depth_payload,
                            },
                        ),
                        event=depth,
                        segment_id=segment_id,
                        pre_l2_state=pre_state,
                        l2_state=book.state(),
                    )
                )

                bbo_time_ms = (WALL_BASE + bbo_mono) // 1_000_000
                bbo_payload = {
                    "e": "bookTicker",
                    "E": bbo_time_ms,
                    "T": bbo_time_ms,
                    "s": symbol,
                    "u": 1_000 + second_index,
                    "b": f"{price:.1f}",
                    "B": f"{4 + second_index}",
                    "a": f"{price + 0.1:.1f}",
                    "A": f"{2 + second_index}",
                    "st": 1,
                    "ps": symbol,
                }
                messages.append(
                    ImpactCaptureMessage(
                        record=record(
                            "binance_futures_public",
                            f"public:{symbol}:bbo",
                            bbo_mono,
                            {
                                "stream": f"{symbol.lower()}@bookTicker",
                                "data": bbo_payload,
                            },
                        ),
                        event=parse_book_ticker(
                            bbo_payload,
                            symbol=symbol,
                            receive_time_ns=bbo_mono,
                        ),
                        segment_id=segment_id,
                    )
                )

                trade_time_ms = (WALL_BASE + trade_mono) // 1_000_000
                trade_payload = {
                    "e": "aggTrade",
                    "E": trade_time_ms,
                    "T": trade_time_ms,
                    "s": symbol,
                    "a": 2_000 + second_index,
                    "p": f"{price + 0.1:.1f}",
                    "q": "0.1",
                    "nq": "0.1",
                    "f": 3_000 + second_index,
                    "l": 3_000 + second_index,
                    "m": second_index % 2 == 1,
                    "st": 1,
                }
                messages.append(
                    ImpactCaptureMessage(
                        record=record(
                            "binance_futures_market",
                            f"market:{symbol}:trade",
                            trade_mono,
                            {
                                "stream": f"{symbol.lower()}@aggTrade",
                                "data": trade_payload,
                            },
                        ),
                        event=parse_aggregate_trade(
                            trade_payload,
                            symbol=symbol,
                            receive_time_ns=trade_mono,
                        ),
                        segment_id=segment_id,
                    )
                )

                mark_time_ms = (WALL_BASE + mark_mono) // 1_000_000
                mark_payload = {
                    "e": "markPriceUpdate",
                    "E": mark_time_ms,
                    "s": symbol,
                    "p": f"{price + 0.05:.2f}",
                    "i": f"{price + 0.04:.2f}",
                    "P": "0",
                    "r": "0.00001",
                    "T": mark_time_ms + 3_600_000,
                    "st": 1,
                }
                messages.append(
                    ImpactCaptureMessage(
                        record=record(
                            "binance_futures_market",
                            f"market:{symbol}:mark",
                            mark_mono,
                            {
                                "stream": f"{symbol.lower()}@markPrice@1s",
                                "data": mark_payload,
                            },
                        ),
                        event=parse_mark_price(
                            mark_payload,
                            symbol=symbol,
                            receive_time_ns=mark_mono,
                        ),
                        segment_id=segment_id,
                    )
                )

        messages.sort(key=lambda message: message.record.received_monotonic_ns)
        store.append_frame(run_id=RUN_ID, messages=messages)
        for symbol_index, _symbol in enumerate(IMPACT_CAPTURE_SYMBOLS, start=1):
            store.finish_segment(
                run_id=RUN_ID,
                segment_id=format(symbol_index, "x") * 32,
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
                "schema_version": IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "status": "completed",
                "capture_gate_passed": True,
                "qualification_passed": True,
                "elapsed_seconds": 3_600.0,
                "writer_frame_count": frame_count,
                "writer_message_count": message_count,
                "writer_compressed_payload_bytes": compressed_bytes,
                "process_io_write_bytes_per_message": 100.0,
                "database_physical_growth_bytes_per_message": 0.0,
                "queue_maximum_utilization": 0.01,
                "negative_corrected_latency_fraction": 0.0,
                "storage_efficiency_passed": True,
                "payload_cap_reached": False,
                "database_size_cap_reached": False,
                "audit_passed": True,
                "audit_errors": [],
            },
            recorded_at_wall_ns=SEGMENT_END + 2,
        )


def test_grid_build_is_causal_idempotent_hash_bound_and_tamper_evident(
    tmp_path,
) -> None:
    database = tmp_path / "impact.duckdb"
    _seed_qualified_run(database)
    index_round73_corpus_run(database, run_id=RUN_ID)

    report = build_round73_causal_grid(database, run_id=RUN_ID)
    repeated = build_round73_causal_grid(database, run_id=RUN_ID)
    audit = audit_round73_causal_grid(database, run_id=RUN_ID)

    assert report == repeated
    assert report.anchor_count == 10_620
    assert report.valid_anchor_count == 15
    assert report.vector_count == 15
    assert audit.passed is True
    assert audit.errors == ()
    assert report.as_dict()["target_constructed"] is False
    assert report.as_dict()["trading_authority"] is False

    with ImpactAbsorptionStore(database) as store:
        connection = store.connect()
        original_vector_sha256 = connection.execute(
            f"SELECT vector_sha256 FROM {ROUND73_GRID_VECTOR_TABLE} "
            "WHERE run_id = ? ORDER BY symbol, anchor_index LIMIT 1",
            [RUN_ID],
        ).fetchone()[0]
        connection.execute(
            f"UPDATE {ROUND73_GRID_VECTOR_TABLE} SET vector_sha256 = ? "
            "WHERE run_id = ? AND symbol = 'BTCUSDT' AND anchor_index = 3535",
            ["0" * 64, RUN_ID],
        )
    vector_audit = audit_round73_causal_grid(database, run_id=RUN_ID)
    assert vector_audit.passed is False
    assert any("feature vector hash" in error for error in vector_audit.errors)

    with ImpactAbsorptionStore(database) as store:
        connection = store.connect()
        connection.execute(
            f"UPDATE {ROUND73_GRID_VECTOR_TABLE} SET vector_sha256 = ? "
            "WHERE run_id = ? AND symbol = 'BTCUSDT' AND anchor_index = 3535",
            [original_vector_sha256, RUN_ID],
        )
        connection.execute(
            f"UPDATE {ROUND73_GRID_ANCHOR_TABLE} "
            "SET shock_direction_taker_share = 0.75 "
            "WHERE run_id = ? AND symbol = 'BTCUSDT' AND anchor_index = 3535",
            [RUN_ID],
        )
    anchor_audit = audit_round73_causal_grid(database, run_id=RUN_ID)
    assert anchor_audit.passed is False
    assert any("aggregate hash" in error for error in anchor_audit.errors)
