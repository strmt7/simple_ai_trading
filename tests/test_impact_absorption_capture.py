from __future__ import annotations

import asyncio
import json

import pytest

import simple_ai_trading.impact_absorption_capture as impact_capture
from simple_ai_trading.impact_absorption import parse_book_ticker
from simple_ai_trading.impact_absorption_capture import (
    CaptureFailureClass,
    ImpactCaptureConfig,
    ImpactCaptureReport,
    _ImpactFrameWriter,
    _best_effort_wire_identity,
    _is_retriable_capture_failure,
    _market_stream_url,
    _public_stream_url,
    _request_weight_limit,
    _rejected_wire_message,
    _terminal_post_capture_failure_report,
    _tick_sizes,
    capture_round73_supervised,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactRejectedWireEvent,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "a" * 32
SEGMENT_ID = "b" * 32


def _report(
    run_id: str,
    *,
    status: str,
    error: str = "",
    qualification_passed: bool = False,
    failure_class: CaptureFailureClass = "none",
) -> ImpactCaptureReport:
    return ImpactCaptureReport(
        run_id=run_id,
        mode="probe",
        status=status,
        qualification_passed=qualification_passed,
        started_wall_ns=1,
        ended_wall_ns=2,
        elapsed_seconds=1.0,
        queue_high_water_messages=0,
        queue_capacity_messages=65_536,
        queue_maximum_utilization=0.0,
        writer_frame_count=0,
        writer_message_count=0,
        writer_compressed_payload_bytes=0,
        payload_cap_reached=False,
        database_physical_bytes=0,
        database_size_cap_bytes=8 * 1024 * 1024 * 1024,
        database_size_cap_reached=False,
        event_counts={},
        symbol_event_counts={},
        negative_corrected_latency_fraction=None,
        audit_passed=status == "completed",
        audit_errors=(),
        error=error,
        failure_class=failure_class,
    )


def _message() -> ImpactCaptureMessage:
    payload = {
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
    raw = json.dumps(
        {"stream": "btcusdt@bookTicker", "data": payload},
        separators=(",", ":"),
    )
    return ImpactCaptureMessage(
        record=ImpactCaptureFrameRecord(
            stream="binance_futures_public",
            connection_id="binance-public:test",
            sequence_number=0,
            received_wall_ns=1_784_058_600_000_000_000,
            received_monotonic_ns=100,
            raw_text=raw,
        ),
        event=parse_book_ticker(payload, symbol="BTCUSDT", receive_time_ns=100),
        segment_id=SEGMENT_ID,
    )


def _start_store(database) -> None:
    with ImpactAbsorptionStore(database) as store:
        store.start_run(
            run_id=RUN_ID,
            started_wall_ns=1,
            started_monotonic_ns=1,
            config={"mode": "probe"},
        )
        store.start_segment(
            run_id=RUN_ID,
            segment_id=SEGMENT_ID,
            symbol="BTCUSDT",
            started_wall_ns=2,
            started_monotonic_ns=2,
            snapshot_update_id=100,
            tick_size=0.1,
            clock_offset_ns=0,
            clock_rtt_ns=1,
            cooldown_until_wall_ns=0,
        )


def test_capture_config_separates_bounded_probe_from_hour_qualification() -> None:
    ImpactCaptureConfig(mode="probe", duration_seconds=300).validate()
    ImpactCaptureConfig(mode="qualification", duration_seconds=3_600).validate()

    with pytest.raises(ValueError, match="capped at 300"):
        ImpactCaptureConfig(mode="probe", duration_seconds=301).validate()
    with pytest.raises(ValueError, match="at least 3600"):
        ImpactCaptureConfig(mode="qualification", duration_seconds=3_599).validate()
    with pytest.raises(ValueError, match="queue capacity"):
        ImpactCaptureConfig(queue_capacity_messages=65_537).validate()
    with pytest.raises(ValueError, match="maximum reconnects"):
        ImpactCaptureConfig(maximum_reconnects=7).validate()
    with pytest.raises(ValueError, match="threads must be between 1 and 8"):
        ImpactCaptureConfig(duckdb_threads=0).validate()
    with pytest.raises(ValueError, match="positive integer followed by a byte unit"):
        ImpactCaptureConfig(duckdb_memory_limit="unbounded").validate()
    with pytest.raises(ValueError, match="512 MiB safety reserve"):
        ImpactCaptureConfig(database_size_cap_bytes=512 * 1024 * 1024).validate()


def test_stream_topology_is_three_symbols_only_and_uses_specific_liquidations() -> None:
    public_url = _public_stream_url()
    market_url = _market_stream_url()

    for symbol in IMPACT_CAPTURE_SYMBOLS:
        lowered = symbol.lower()
        assert f"{lowered}@depth@100ms" in public_url
        assert f"{lowered}@bookTicker" in public_url
        assert f"{lowered}@aggTrade" in market_url
        assert f"{lowered}@markPrice@1s" in market_url
        assert f"{lowered}@forceOrder" in market_url
    assert "!forceOrder@arr" not in market_url


def test_rejected_wire_message_preserves_current_nested_liquidation_identity() -> None:
    raw = (
        '{"stream":"ethusdt@forceOrder","data":{"e":"forceOrder",'
        '"E":1784734386582,"o":{"s":"ETHUSDT","S":"SELL",'
        '"ps":"ETHUSDT","st":1}}}'
    )
    record = ImpactCaptureFrameRecord(
        stream="binance_futures_market",
        connection_id="binance-market:test",
        sequence_number=0,
        received_wall_ns=1_784_734_386_600_000_000,
        received_monotonic_ns=123,
        raw_text=raw,
    )

    message = _rejected_wire_message(
        record=record,
        error=impact_capture.ImpactFeedIntegrityError("test rejection"),
        segment_ids={"ETHUSDT": SEGMENT_ID},
    )

    assert isinstance(message.event, ImpactRejectedWireEvent)
    assert _best_effort_wire_identity(raw) == (
        "ethusdt@forceOrder",
        "forceOrder",
        "ETHUSDT",
    )
    assert message.event.observed_symbol == "ETHUSDT"
    assert message.event.rejection_class == "feed_integrity"
    assert message.segment_id == SEGMENT_ID
    assert _best_effort_wire_identity('{"stream":"one","stream":"two","data":{}}') == (
        "",
        "",
        "",
    )


def test_one_writer_flushes_atomically_without_blocking_the_caller(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _start_store(database)
    config = ImpactCaptureConfig(
        database=str(database),
        duration_seconds=1,
        queue_capacity_messages=8,
        frame_message_limit=1,
        frame_flush_seconds=0.01,
    )
    config.validate()
    writer = _ImpactFrameWriter(config, RUN_ID)
    writer.start()

    writer.put(_message())
    assert writer.stop(timeout_seconds=5) is True

    assert writer.failed.is_set() is False
    assert writer.frame_count == 1
    assert writer.message_count == 1


def test_writer_stops_at_database_size_reserve_without_silent_success(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "impact.duckdb"
    _start_store(database)
    calls = 0

    def physical_size(_database: str) -> int:
        nonlocal calls
        calls += 1
        return 0 if calls < 2 else 600 * 1024 * 1024

    monkeypatch.setattr(impact_capture, "_database_physical_bytes", physical_size)
    writer = _ImpactFrameWriter(
        ImpactCaptureConfig(
            database=str(database),
            duration_seconds=1,
            queue_capacity_messages=8,
            frame_message_limit=1,
            frame_flush_seconds=0.01,
            database_size_cap_bytes=1024 * 1024 * 1024,
        ),
        RUN_ID,
    )

    writer.start()
    writer.put(_message())
    assert writer.stop(timeout_seconds=5.0) is True
    assert writer.failed.is_set() is False
    assert writer.database_cap_reached.is_set() is True


def test_terminal_post_capture_failure_is_persisted_and_never_retried(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _start_store(database)
    config = ImpactCaptureConfig(
        database=str(database),
        duration_seconds=1,
        queue_capacity_messages=8,
        frame_message_limit=1,
        frame_flush_seconds=0.01,
    )
    writer = _ImpactFrameWriter(config, RUN_ID)
    writer.start()
    writer.put(_message())
    assert writer.stop(timeout_seconds=5.0) is True
    with ImpactAbsorptionStore(database) as store:
        store.finish_segment(
            run_id=RUN_ID,
            segment_id=SEGMENT_ID,
            status="valid",
            ended_wall_ns=10,
        )
        store.finish_run(run_id=RUN_ID, status="completed", ended_wall_ns=11)

    report = _terminal_post_capture_failure_report(
        config,
        run_id=RUN_ID,
        error=RuntimeError("report materialization failed"),
        writer=writer,
    )

    assert report is not None
    assert report.status == "failed"
    assert report.qualification_passed is False
    assert report.failure_class == "post_capture"
    assert report.audit_passed is True
    assert _is_retriable_capture_failure(report.failure_class) is False
    with ImpactAbsorptionStore(database, read_only=True) as store:
        stored = store.connect().execute(
            "SELECT schema_version, report_json FROM impact_capture_report "
            "WHERE run_id = ?",
            [RUN_ID],
        ).fetchone()
    assert stored[0] == "round-073-capture-report-v4"
    assert json.loads(stored[1])["failure_class"] == "post_capture"
    with ImpactAbsorptionStore(database, read_only=True) as store:
        assert store.audit_run(RUN_ID).passed is True


def test_writer_queue_overflow_is_explicit_before_silent_drop(tmp_path) -> None:
    writer = _ImpactFrameWriter(
        ImpactCaptureConfig(
            database=str(tmp_path / "unused.duckdb"),
            duration_seconds=1,
            queue_capacity_messages=1,
        ),
        RUN_ID,
    )
    writer.put(_message())
    with pytest.raises(RuntimeError, match="queue overflow"):
        writer.put(_message())


def test_exchange_metadata_requires_one_unambiguous_minute_limit_and_tick_filter() -> (
    None
):
    exchange_info = {
        "rateLimits": [
            {
                "rateLimitType": "REQUEST_WEIGHT",
                "interval": "MINUTE",
                "intervalNum": 1,
                "limit": 2400,
            }
        ],
        "symbols": [
            {
                "symbol": symbol,
                "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"}],
            }
            for symbol in IMPACT_CAPTURE_SYMBOLS
        ],
    }

    assert _request_weight_limit(exchange_info) == 2400
    assert _tick_sizes(exchange_info) == {
        symbol: "0.10" for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    with pytest.raises(ValueError, match="missing or ambiguous"):
        _request_weight_limit({**exchange_info, "rateLimits": []})


def test_capture_supervisor_restarts_with_exact_backoff_without_pooling_attempts(
    monkeypatch,
) -> None:
    reports = iter(
        (
            _report(
                "1" * 32,
                status="failed",
                error="public_source:ConnectionClosedError:link lost",
                failure_class="transport",
            ),
            _report("2" * 32, status="completed"),
        )
    )
    delays: list[float] = []

    async def fake_capture(_config):
        return next(reports)

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(impact_capture, "capture_round73", fake_capture)
    monkeypatch.setattr(impact_capture, "_sleep_before_reconnect", fake_sleep)

    result = asyncio.run(
        capture_round73_supervised(
            ImpactCaptureConfig(duration_seconds=1, maximum_reconnects=2)
        )
    )

    assert result.status == "completed"
    assert result.selected_run_id == "2" * 32
    assert result.attempt_count == 2
    assert result.reconnect_count == 1
    assert result.reconnect_delays_seconds == (1.0,)
    assert delays == [1.0]
    assert [report.run_id for report in result.attempts] == ["1" * 32, "2" * 32]
    assert result.as_dict()["attempt_evidence_combined"] is False


def test_capture_supervisor_never_combines_failed_qualification_fragments(
    monkeypatch,
) -> None:
    reports = iter(
        (
            _report(
                "3" * 32,
                status="failed",
                error="market_source_stall",
                failure_class="transport",
            ),
            _report("4" * 32, status="completed", qualification_passed=False),
        )
    )

    async def fake_capture(_config):
        return next(reports)

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(impact_capture, "capture_round73", fake_capture)
    monkeypatch.setattr(impact_capture, "_sleep_before_reconnect", fake_sleep)

    result = asyncio.run(
        capture_round73_supervised(
            ImpactCaptureConfig(duration_seconds=1, maximum_reconnects=1)
        )
    )

    assert result.status == "completed"
    assert result.qualification_passed is False
    assert len(result.attempts) == 2
    assert result.as_dict()["attempt_evidence_combined"] is False


def test_capture_supervisor_fails_closed_without_retrying_writer_fault(
    monkeypatch,
) -> None:
    calls = 0

    async def fake_capture(_config):
        nonlocal calls
        calls += 1
        return _report(
            "5" * 32,
            status="failed",
            error="writer_heartbeat_timeout",
            failure_class="writer",
        )

    monkeypatch.setattr(impact_capture, "capture_round73", fake_capture)

    result = asyncio.run(
        capture_round73_supervised(
            ImpactCaptureConfig(duration_seconds=1, maximum_reconnects=6)
        )
    )

    assert calls == 1
    assert result.status == "failed"
    assert result.reconnect_count == 0
    assert result.terminal_error == "writer_heartbeat_timeout"
    assert _is_retriable_capture_failure("writer") is False
    assert _is_retriable_capture_failure("transport") is True


def test_capture_supervisor_retries_bounded_startup_transport_failure(
    monkeypatch,
) -> None:
    calls = 0

    async def fake_capture(_config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("open timeout")
        return _report("6" * 32, status="completed")

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(impact_capture, "capture_round73", fake_capture)
    monkeypatch.setattr(impact_capture, "_sleep_before_reconnect", fake_sleep)

    result = asyncio.run(
        capture_round73_supervised(
            ImpactCaptureConfig(duration_seconds=1, maximum_reconnects=1)
        )
    )

    assert result.status == "completed"
    assert result.attempt_count == 2
    assert result.reconnect_count == 1
    assert len(result.startup_errors) == 1
    assert result.startup_errors[0].startswith("startup:TimeoutError:")


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (impact_capture._ImpactRateLimitGuardError("budget"), "rate_limit"),
        (impact_capture._ImpactRestResponseError("/time", 429), "rate_limit"),
        (impact_capture._ImpactRestResponseError("/time", 503), "rest_transport"),
        (impact_capture._ImpactWriterFault("writer"), "writer"),
        (impact_capture._ImpactWriterQueueOverflow("queue"), "resource_limit"),
        (impact_capture._ImpactPayloadCapReached("cap"), "resource_limit"),
        (impact_capture.ImpactFeedIntegrityError("gap"), "feed_integrity"),
        (RuntimeError("unknown pipeline failure"), "processing"),
    ),
)
def test_capture_failure_classes_are_structured(error, expected) -> None:
    assert impact_capture._failure_class_for_exception(error) == expected
