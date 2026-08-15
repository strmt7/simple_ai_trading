"""Credential-free Binance predictor sidecar for Polymarket Round 21."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import re
from threading import Lock
import time
from typing import Any, Literal, Protocol
import uuid

from websockets.asyncio.client import connect

from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_round21_binance_features import parse_round21_binance_record
from .polymarket_round21_live_features import Round21PublicSourceGap


POLYMARKET_ROUND21_BINANCE_FEED_SCHEMA_VERSION = (
    "polymarket-round21-binance-public-feed-v1"
)
ROUND21_BINANCE_SPOT_WEBSOCKET = (
    "wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/btcusdt@trade"
)
ROUND21_BINANCE_USDM_WEBSOCKET = (
    "wss://fstream.binance.com/stream?streams=btcusdt@bookTicker/btcusdt@trade"
)
_ENDPOINTS = {
    "spot": ("binance_spot", ROUND21_BINANCE_SPOT_WEBSOCKET),
    "usdm": ("binance_futures", ROUND21_BINANCE_USDM_WEBSOCKET),
}
_MAX_MESSAGE_BYTES = 256 * 1024
_MAX_SOCKET_QUEUE_FRAMES = 2_048
_INACTIVITY_SECONDS = 30.0
_STABLE_CONNECTION_SECONDS = 30.0
_CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9:._-]{1,159}$")


class _WebSocket(Protocol):
    async def recv(self) -> object: ...


def _text_frame(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="strict")
    return str(raw)


@dataclass(frozen=True, slots=True)
class Round21BinancePublicFeedHealth:
    running: bool
    enabled_markets: tuple[str, ...]
    received_message_count: int
    processed_message_count: int
    gap_count: int
    spot_message_count: int
    usdm_message_count: int
    reconnect_counts: dict[str, int]
    queue_capacity: int
    queue_size: int
    queue_high_watermark: int
    started_at_ms: int
    last_receipt_at_ms: int
    last_gap_sha256: str
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    trading_authority: bool = False


FeedItem = CaptureFrameRecord | Round21PublicSourceGap


class Round21BinancePublicSidecar:
    """Stream public BTC spot/USD-M observations without Binance authority."""

    credentials_used = False
    account_connected = False
    binance_execution_connected = False
    trading_authority = False

    def __init__(
        self,
        *,
        markets: tuple[Literal["spot", "usdm"], ...],
        record_consumer: Callable[[CaptureFrameRecord], None],
        gap_consumer: Callable[[Round21PublicSourceGap], None],
        queue_capacity: int = 20_000,
        connector: Callable[..., Any] = connect,
        wall_clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        connection_id_factory: Callable[[str], str] | None = None,
    ) -> None:
        selected_markets = tuple(markets)
        capacity = int(queue_capacity)
        if (
            not selected_markets
            or len(set(selected_markets)) != len(selected_markets)
            or any(market not in _ENDPOINTS for market in selected_markets)
        ):
            raise ValueError("Round 21 Binance sidecar markets are invalid")
        if not 1_000 <= capacity <= 100_000:
            raise ValueError("Round 21 Binance sidecar queue capacity is invalid")
        if not all(
            callable(value)
            for value in (record_consumer, gap_consumer, connector, monotonic_ns)
        ):
            raise TypeError("Round 21 Binance sidecar dependency is invalid")
        if wall_clock_ms is not None and not callable(wall_clock_ms):
            raise TypeError("Round 21 Binance sidecar wall clock is invalid")
        if connection_id_factory is not None and not callable(connection_id_factory):
            raise TypeError("Round 21 Binance sidecar connection factory is invalid")
        self.markets = selected_markets
        self.queue_capacity = capacity
        self._record_consumer = record_consumer
        self._gap_consumer = gap_consumer
        self._connector = connector
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1_000))
        self._monotonic_ns = monotonic_ns
        self._connection_id_factory = connection_id_factory or (
            lambda market: f"binance:{market}:{uuid.uuid4().hex}"
        )
        self._state_lock = Lock()
        self._running = False
        self._started = False
        self._received = 0
        self._processed = 0
        self._gaps = 0
        self._market_messages = {"spot": 0, "usdm": 0}
        self._reconnects = {"spot": 0, "usdm": 0}
        self._queue_size = 0
        self._queue_high_watermark = 0
        self._started_at_ms = 0
        self._last_receipt_at_ms = 0
        self._last_gap_sha256 = ""

    def _connection_id(self, market: str) -> str:
        selected = str(self._connection_id_factory(market) or "").strip().lower()
        if (
            not selected.startswith(f"binance:{market}:")
            or _CONNECTION_ID.fullmatch(selected) is None
        ):
            raise ValueError("Round 21 Binance connection identity is invalid")
        return selected

    def _record_received(
        self,
        item: FeedItem,
        *,
        market: str,
        queue_size: int,
    ) -> None:
        with self._state_lock:
            self._received += 1
            self._queue_size = queue_size
            self._queue_high_watermark = max(self._queue_high_watermark, queue_size)
            if isinstance(item, CaptureFrameRecord):
                self._market_messages[market] += 1
                self._last_receipt_at_ms = item.received_wall_ms
            else:
                self._gaps += 1
                self._last_receipt_at_ms = item.observed_wall_ms
                self._last_gap_sha256 = item.gap_sha256

    def _record_processed(self, *, queue_size: int) -> None:
        with self._state_lock:
            self._processed += 1
            self._queue_size = queue_size

    def health(self) -> Round21BinancePublicFeedHealth:
        with self._state_lock:
            return Round21BinancePublicFeedHealth(
                running=self._running,
                enabled_markets=self.markets,
                received_message_count=self._received,
                processed_message_count=self._processed,
                gap_count=self._gaps,
                spot_message_count=self._market_messages["spot"],
                usdm_message_count=self._market_messages["usdm"],
                reconnect_counts=dict(self._reconnects),
                queue_capacity=self.queue_capacity,
                queue_size=self._queue_size,
                queue_high_watermark=self._queue_high_watermark,
                started_at_ms=self._started_at_ms,
                last_receipt_at_ms=self._last_receipt_at_ms,
                last_gap_sha256=self._last_gap_sha256,
            )

    def _enqueue(
        self,
        output: asyncio.Queue[FeedItem],
        item: FeedItem,
        *,
        market: str,
    ) -> None:
        try:
            output.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise MemoryError(
                "Round 21 Binance queue saturated; optional source is unsafe"
            ) from exc
        self._record_received(item, market=market, queue_size=output.qsize())

    async def _wait_backoff(self, stop: asyncio.Event, seconds: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.1, seconds))
        except TimeoutError:
            return

    def _gap(
        self,
        *,
        stream: str,
        connection_id: str,
        last_sequence_number: int,
        error: BaseException,
    ) -> Round21PublicSourceGap:
        return Round21PublicSourceGap(
            stream=stream,
            connection_id=connection_id,
            observed_wall_ms=int(self._wall_clock_ms()),
            observed_monotonic_ns=int(self._monotonic_ns()),
            last_sequence_number=last_sequence_number,
            reason=error.__class__.__name__,
        )

    async def _stream(
        self,
        market: str,
        output: asyncio.Queue[FeedItem],
        stop: asyncio.Event,
    ) -> None:
        stream, endpoint = _ENDPOINTS[market]
        backoff = 1.0
        first_connection = True
        while not stop.is_set():
            connection_id = self._connection_id(market)
            sequence = 0
            connected_at = 0.0
            if not first_connection:
                with self._state_lock:
                    self._reconnects[market] += 1
            first_connection = False
            try:
                async with self._connector(
                    endpoint,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=_MAX_MESSAGE_BYTES,
                    max_queue=_MAX_SOCKET_QUEUE_FRAMES,
                    compression=None,
                ) as websocket:
                    connected_at = asyncio.get_running_loop().time()
                    while not stop.is_set():
                        receive = asyncio.create_task(websocket.recv())
                        stopping = asyncio.create_task(stop.wait())
                        try:
                            done, _ = await asyncio.wait(
                                {receive, stopping},
                                timeout=_INACTIVITY_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                raise TimeoutError("Binance public-stream inactivity")
                            if stopping in done and stopping.result():
                                return
                            raw_text = _text_frame(receive.result())
                            sequence += 1
                            record = CaptureFrameRecord(
                                stream=stream,
                                connection_id=connection_id,
                                sequence_number=sequence,
                                received_wall_ms=int(self._wall_clock_ms()),
                                received_monotonic_ns=int(self._monotonic_ns()),
                                raw_text=raw_text,
                            )
                            parse_round21_binance_record(record)
                            self._enqueue(
                                output,
                                record,
                                market=market,
                            )
                        finally:
                            for task in (receive, stopping):
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(
                                receive,
                                stopping,
                                return_exceptions=True,
                            )
            except asyncio.CancelledError:
                raise
            except (MemoryError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                self._enqueue(
                    output,
                    self._gap(
                        stream=stream,
                        connection_id=connection_id,
                        last_sequence_number=sequence,
                        error=exc,
                    ),
                    market=market,
                )
                if (
                    connected_at > 0.0
                    and asyncio.get_running_loop().time() - connected_at
                    >= _STABLE_CONNECTION_SECONDS
                ):
                    backoff = 1.0
                await self._wait_backoff(stop, backoff)
                backoff = min(30.0, backoff * 2.0)

    async def _consume(
        self,
        output: asyncio.Queue[FeedItem],
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set() or not output.empty():
            try:
                item = await asyncio.wait_for(output.get(), timeout=0.25)
            except TimeoutError:
                continue
            try:
                if isinstance(item, CaptureFrameRecord):
                    self._record_consumer(item)
                else:
                    self._gap_consumer(item)
                self._record_processed(queue_size=output.qsize())
            finally:
                output.task_done()

    async def run(self, stop: asyncio.Event) -> None:
        """Run until Stop; parsing, queue, and consumer faults propagate."""

        if not isinstance(stop, asyncio.Event):
            raise TypeError("Round 21 Binance sidecar Stop type differs")
        with self._state_lock:
            if self._running:
                raise RuntimeError("Round 21 Binance sidecar is already running")
            if self._started:
                raise RuntimeError("Round 21 Binance sidecar cannot be restarted")
            self._running = True
            self._started = True
            self._started_at_ms = int(self._wall_clock_ms())
        output: asyncio.Queue[FeedItem] = asyncio.Queue(maxsize=self.queue_capacity)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._consume(output, stop))
                for market in self.markets:
                    group.create_task(self._stream(market, output, stop))
        finally:
            with self._state_lock:
                self._running = False
                self._queue_size = output.qsize()


credentials_used = False
account_connected = False
binance_execution_connected = False
trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_BINANCE_FEED_SCHEMA_VERSION",
    "ROUND21_BINANCE_SPOT_WEBSOCKET",
    "ROUND21_BINANCE_USDM_WEBSOCKET",
    "Round21BinancePublicFeedHealth",
    "Round21BinancePublicSidecar",
]
