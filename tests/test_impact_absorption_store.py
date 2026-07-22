from __future__ import annotations

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
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactRestEvent,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "0" * 32
SEGMENT_ID = "1" * 32
WALL_BASE = 1_784_058_600_000_000_000


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
        "st": 1,
        "o": {
            "s": "BTCUSDT",
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
        assert (
            connection.execute(
                "SELECT count(*) FROM impact_event_index WHERE run_id = ?", [RUN_ID]
            ).fetchone()[0]
            == 6
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM impact_l2_state WHERE run_id = ?", [RUN_ID]
            ).fetchone()[0]
            == 1
        )
        bid_prices, ask_prices = connection.execute(
            "SELECT bid_prices, ask_prices FROM impact_l2_state WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()
        assert len(bid_prices) == len(ask_prices) == 20
        assert connection.execute(
            "SELECT funding_rate FROM impact_mark_price WHERE run_id = ?", [RUN_ID]
        ).fetchone()[0] == pytest.approx(-0.0001)


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
            .execute("SELECT count(*) FROM impact_capture_frame")
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
            .execute("SELECT count(*) FROM impact_event_index")
            .fetchone()[0]
            == 0
        )


def test_store_detects_payload_and_typed_row_tampering(tmp_path) -> None:
    with ImpactAbsorptionStore(tmp_path / "impact.duckdb") as store:
        _start(store)
        store.append_frame(run_id=RUN_ID, messages=_messages())
        store.connect().execute(
            "DELETE FROM impact_aggregate_trade WHERE run_id = ?", [RUN_ID]
        )
        typed_audit = store.audit_run(RUN_ID)
        assert typed_audit.passed is False
        assert "typed_count_mismatch:aggTrade" in typed_audit.errors

        store.connect().execute(
            "UPDATE impact_mark_price SET funding_rate = 0.5 WHERE run_id = ?",
            [RUN_ID],
        )
        value_audit = store.audit_run(RUN_ID)
        assert value_audit.passed is False
        assert any(
            error.startswith("typed_sha256_mismatch:") for error in value_audit.errors
        )

        store.connect().execute(
            """
            UPDATE impact_capture_frame SET compressed_payload = from_hex('00')
            WHERE run_id = ? AND frame_index = 0
            """,
            [RUN_ID],
        )
        payload_audit = store.audit_run(RUN_ID)
        assert payload_audit.passed is False
        assert "compressed_size_mismatch:0" in payload_audit.errors
        assert "compressed_sha256_mismatch:0" in payload_audit.errors


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
        assert (
            store.connect()
            .execute("SELECT count(*) FROM impact_capture_frame")
            .fetchone()[0]
            == 0
        )


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
            "SELECT open_interest FROM impact_rest_event WHERE run_id = ?", [RUN_ID]
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
            "schema_version": "round-073-capture-report-v2",
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
