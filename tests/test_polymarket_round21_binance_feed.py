from __future__ import annotations

import asyncio
import itertools
import json

import pytest

from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket_round21_binance_feed import (
    ROUND21_BINANCE_SPOT_WEBSOCKET,
    ROUND21_BINANCE_USDM_WEBSOCKET,
    Round21BinancePublicSidecar,
)
from simple_ai_trading.polymarket_round21_live_features import (
    Round21PublicSourceGap,
)


SPOT_BOOK = json.dumps(
    {
        "stream": "btcusdt@bookTicker",
        "data": {
            "u": 1,
            "s": "BTCUSDT",
            "b": "60000",
            "B": "2",
            "a": "60001",
            "A": "3",
        },
    },
    separators=(",", ":"),
)
USDM_TRADE = json.dumps(
    {
        "stream": "btcusdt@trade",
        "data": {
            "e": "trade",
            "E": 1_700_000_000_001,
            "s": "BTCUSDT",
            "t": 2,
            "p": "60000",
            "q": "0.1",
            "T": 1_700_000_000_000,
            "m": False,
        },
    },
    separators=(",", ":"),
)


class _Socket:
    def __init__(self, messages: tuple[object, ...]) -> None:
        self.messages = iter(messages)
        self.closed = asyncio.Event()

    async def recv(self) -> object:
        try:
            return next(self.messages)
        except StopIteration:
            await self.closed.wait()
            raise RuntimeError("socket closed")


class _Context:
    def __init__(self, socket: _Socket) -> None:
        self.socket = socket

    async def __aenter__(self) -> _Socket:
        return self.socket

    async def __aexit__(self, *_args) -> None:
        self.socket.closed.set()


class _Connector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.spot = _Socket((SPOT_BOOK,))
        self.usdm = _Socket((USDM_TRADE,))

    def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return _Context(
            self.spot if url == ROUND21_BINANCE_SPOT_WEBSOCKET else self.usdm
        )


def _sidecar(*, connector=None, records=None, gaps=None):
    ticks = itertools.count(1)
    selected_records = [] if records is None else records
    selected_gaps = [] if gaps is None else gaps
    return Round21BinancePublicSidecar(
        markets=("spot", "usdm"),
        record_consumer=selected_records.append,
        gap_consumer=selected_gaps.append,
        connector=connector or _Connector(),
        wall_clock_ms=lambda: 1_700_000_000_100 + next(ticks),
        monotonic_ns=lambda: next(ticks),
        connection_id_factory=lambda market: f"binance:{market}:connection",
    )


def test_binance_sidecar_is_public_data_only() -> None:
    sidecar = _sidecar()
    assert sidecar.credentials_used is False
    assert sidecar.account_connected is False
    assert sidecar.binance_execution_connected is False
    assert sidecar.trading_authority is False
    assert not hasattr(sidecar, "api_key")
    assert not hasattr(sidecar, "order_client")


def test_binance_sidecar_streams_exact_spot_and_usdm_records() -> None:
    async def exercise():
        connector = _Connector()
        records: list[CaptureFrameRecord] = []
        gaps: list[Round21PublicSourceGap] = []
        sidecar = _sidecar(connector=connector, records=records, gaps=gaps)
        stop = asyncio.Event()
        task = asyncio.create_task(sidecar.run(stop))
        for _ in range(100):
            if sidecar.health().processed_message_count >= 2:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        return connector, records, gaps, sidecar

    connector, records, gaps, sidecar = asyncio.run(exercise())
    assert {record.stream for record in records} == {
        "binance_spot",
        "binance_futures",
    }
    assert gaps == []
    assert {url for url, _kwargs in connector.calls} == {
        ROUND21_BINANCE_SPOT_WEBSOCKET,
        ROUND21_BINANCE_USDM_WEBSOCKET,
    }
    health = sidecar.health()
    assert health.running is False
    assert health.received_message_count == 2
    assert health.processed_message_count == 2
    assert health.spot_message_count == 1
    assert health.usdm_message_count == 1
    assert health.gap_count == 0
    assert health.queue_high_watermark <= 2
    assert health.credentials_used is False
    assert health.binance_execution_connected is False


class _FailingSocket(_Socket):
    async def recv(self) -> object:
        raise ConnectionError("offline")


class _FailingConnector:
    def __call__(self, _url: str, **_kwargs):
        return _Context(_FailingSocket(()))


def test_binance_disconnect_is_a_gap_before_retry() -> None:
    async def exercise():
        gaps: list[Round21PublicSourceGap] = []
        stop = asyncio.Event()

        def consume_gap(gap: Round21PublicSourceGap) -> None:
            gaps.append(gap)
            stop.set()

        ticks = itertools.count(1)
        sidecar = Round21BinancePublicSidecar(
            markets=("spot",),
            record_consumer=lambda _record: None,
            gap_consumer=consume_gap,
            connector=_FailingConnector(),
            wall_clock_ms=lambda: 1_700_000_000_100 + next(ticks),
            monotonic_ns=lambda: next(ticks),
            connection_id_factory=lambda market: f"binance:{market}:connection",
        )
        await asyncio.wait_for(sidecar.run(stop), timeout=2.0)
        return gaps, sidecar

    gaps, sidecar = asyncio.run(exercise())
    assert len(gaps) == 1
    assert gaps[0].stream == "binance_spot"
    assert gaps[0].reason == "ConnectionError"
    assert sidecar.health().gap_count == 1


def test_binance_queue_saturation_and_restart_fail_loudly() -> None:
    sidecar = _sidecar()
    output = asyncio.Queue(maxsize=1)
    gap = sidecar._gap(
        stream="binance_spot",
        connection_id="binance:spot:connection",
        last_sequence_number=0,
        error=ConnectionError(),
    )
    sidecar._enqueue(output, gap, market="spot")
    with pytest.raises(MemoryError, match="queue saturated"):
        sidecar._enqueue(output, gap, market="spot")

    async def run_twice() -> None:
        stopped = asyncio.Event()
        stopped.set()
        await sidecar.run(stopped)
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            await sidecar.run(stopped)

    asyncio.run(run_twice())


def test_binance_sidecar_rejects_unknown_market_and_schema_drift() -> None:
    with pytest.raises(ValueError, match="markets are invalid"):
        Round21BinancePublicSidecar(
            markets=("invalid",),  # type: ignore[arg-type]
            record_consumer=lambda _record: None,
            gap_consumer=lambda _gap: None,
        )

    class DriftConnector:
        def __call__(self, _url: str, **_kwargs):
            return _Context(_Socket(("{}",)))

    async def exercise() -> None:
        sidecar = Round21BinancePublicSidecar(
            markets=("spot",),
            record_consumer=lambda _record: None,
            gap_consumer=lambda _gap: None,
            connector=DriftConnector(),
            connection_id_factory=lambda market: f"binance:{market}:connection",
        )
        stop = asyncio.Event()
        task = asyncio.create_task(sidecar.run(stop))
        for _ in range(100):
            if sidecar.health().gap_count:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert sidecar.health().gap_count >= 1

    asyncio.run(exercise())
