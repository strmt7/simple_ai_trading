"""Credential-free Polymarket CLOB and Chainlink feed for Round 21."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
import re
from threading import Lock
import time
from typing import Any, Protocol
import uuid

from websockets.asyncio.client import connect

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_redundant_union import PolymarketClobLaneReceipt
from .polymarket_round21_live_features import (
    Round21LiveFeatureCoordinator,
    Round21PublicSourceGap,
)
from .polymarket_round21_core_features import parse_round21_chainlink_wire_text


POLYMARKET_ROUND21_PUBLIC_FEED_SCHEMA_VERSION = (
    "polymarket-round21-public-feed-v1"
)
ROUND21_CLOB_MARKET_WEBSOCKET = (
    "wss://ws-subscriptions-clob.polymarket.com/ws/market"
)
ROUND21_POLYMARKET_RTDS_WEBSOCKET = "wss://ws-live-data.polymarket.com"
ROUND21_PUBLIC_FEED_CLOB_LANES = ("clob-a", "clob-b")
_CLOB_MAX_MESSAGE_BYTES = 512 * 1024
_RTDS_MAX_MESSAGE_BYTES = 64 * 1024
_CLOB_MAX_QUEUE_FRAMES = 2_048
_RTDS_MAX_QUEUE_FRAMES = 1_024
_INACTIVITY_SECONDS = 30.0
_STABLE_CONNECTION_SECONDS = 30.0
_CONNECTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,159}$")


class _WebSocket(Protocol):
    async def send(self, message: str) -> object: ...

    async def recv(self) -> object: ...


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _text_frame(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="strict")
    return str(raw)


@dataclass(frozen=True, slots=True)
class Round21PublicFeedHealth:
    running: bool
    received_message_count: int
    processed_message_count: int
    gap_count: int
    clob_message_count: int
    chainlink_message_count: int
    queue_capacity: int
    queue_size: int
    queue_high_watermark: int
    started_at_ms: int
    last_receipt_at_ms: int
    last_gap_sha256: str
    credentials_used: bool = False
    account_connected: bool = False
    binance_connected: bool = False
    trading_authority: bool = False


FeedItem = PolymarketClobLaneReceipt | CaptureFrameRecord | Round21PublicSourceGap


class Round21PolymarketPublicFeed:
    """Stream one exact BTC five-minute market without persistence or authority."""

    credentials_used = False
    account_connected = False
    binance_connected = False
    trading_authority = False

    def __init__(
        self,
        *,
        market: PolymarketFiveMinuteMarket,
        coordinator: Round21LiveFeatureCoordinator,
        queue_capacity: int = 20_000,
        connector: Callable[..., Any] = connect,
        wall_clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        connection_id_factory: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(market, PolymarketFiveMinuteMarket):
            raise TypeError("Round 21 public-feed market type differs")
        if market.asset != "BTC" or market.horizon_minutes != 5:
            raise ValueError("Round 21 public feed requires a BTC five-minute market")
        if not isinstance(coordinator, Round21LiveFeatureCoordinator):
            raise TypeError("Round 21 public-feed coordinator type differs")
        if coordinator.market.condition_id != market.condition_id:
            raise ValueError("Round 21 public-feed market and coordinator differ")
        capacity = int(queue_capacity)
        if not 1_000 <= capacity <= 100_000:
            raise ValueError("Round 21 public-feed queue capacity is invalid")
        if not callable(connector) or not callable(monotonic_ns):
            raise TypeError("Round 21 public-feed runtime dependency is invalid")
        if wall_clock_ms is not None and not callable(wall_clock_ms):
            raise TypeError("Round 21 public-feed wall clock is invalid")
        if connection_id_factory is not None and not callable(connection_id_factory):
            raise TypeError("Round 21 public-feed connection factory is invalid")
        self.market = market
        self.coordinator = coordinator
        self.queue_capacity = capacity
        self._connector = connector
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1_000))
        self._monotonic_ns = monotonic_ns
        self._connection_id_factory = connection_id_factory or (
            lambda lane: f"{lane}:{uuid.uuid4().hex}"
        )
        self._state_lock = Lock()
        self._running = False
        self._started = False
        self._received = 0
        self._processed = 0
        self._gaps = 0
        self._clob_messages = 0
        self._chainlink_messages = 0
        self._queue_size = 0
        self._queue_high_watermark = 0
        self._started_at_ms = 0
        self._last_receipt_at_ms = 0
        self._last_gap_sha256 = ""

    def _connection_id(self, lane: str) -> str:
        selected = str(self._connection_id_factory(lane) or "").strip().lower()
        if (
            not selected.startswith(f"{lane}:")
            or _CONNECTION_ID.fullmatch(selected) is None
        ):
            raise ValueError("Round 21 public-feed connection identity is invalid")
        return selected

    def _record_received(self, item: FeedItem, *, queue_size: int) -> None:
        with self._state_lock:
            self._received += 1
            self._queue_size = queue_size
            self._queue_high_watermark = max(self._queue_high_watermark, queue_size)
            if isinstance(item, PolymarketClobLaneReceipt):
                self._clob_messages += 1
                self._last_receipt_at_ms = item.received_wall_ms
            elif isinstance(item, CaptureFrameRecord):
                self._chainlink_messages += 1
                self._last_receipt_at_ms = item.received_wall_ms
            else:
                self._gaps += 1
                self._last_receipt_at_ms = item.observed_wall_ms
                self._last_gap_sha256 = item.gap_sha256

    def _record_processed(self, *, queue_size: int) -> None:
        with self._state_lock:
            self._processed += 1
            self._queue_size = queue_size

    def health(self) -> Round21PublicFeedHealth:
        with self._state_lock:
            return Round21PublicFeedHealth(
                running=self._running,
                received_message_count=self._received,
                processed_message_count=self._processed,
                gap_count=self._gaps,
                clob_message_count=self._clob_messages,
                chainlink_message_count=self._chainlink_messages,
                queue_capacity=self.queue_capacity,
                queue_size=self._queue_size,
                queue_high_watermark=self._queue_high_watermark,
                started_at_ms=self._started_at_ms,
                last_receipt_at_ms=self._last_receipt_at_ms,
                last_gap_sha256=self._last_gap_sha256,
            )

    def _enqueue(self, output: asyncio.Queue[FeedItem], item: FeedItem) -> None:
        try:
            output.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise MemoryError(
                "Round 21 public-feed queue saturated; current market is unsafe"
            ) from exc
        self._record_received(item, queue_size=output.qsize())

    async def _heartbeat(
        self,
        websocket: _WebSocket,
        stop: asyncio.Event,
        *,
        interval_seconds: float,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
                return
            except TimeoutError:
                await websocket.send("PING")

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

    async def _clob_lane(
        self,
        lane: str,
        output: asyncio.Queue[FeedItem],
        stop: asyncio.Event,
    ) -> None:
        backoff = 1.0
        while not stop.is_set():
            connection_id = self._connection_id(lane)
            sequence = 0
            connected_at = 0.0
            try:
                async with self._connector(
                    ROUND21_CLOB_MARKET_WEBSOCKET,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=_CLOB_MAX_MESSAGE_BYTES,
                    max_queue=_CLOB_MAX_QUEUE_FRAMES,
                    compression=None,
                ) as websocket:
                    connected_at = asyncio.get_running_loop().time()
                    await websocket.send(
                        _canonical_json(
                            {
                                "assets_ids": sorted(self.market.token_ids),
                                "type": "market",
                                "custom_feature_enabled": True,
                            }
                        )
                    )
                    heartbeat = asyncio.create_task(
                        self._heartbeat(websocket, stop, interval_seconds=10.0)
                    )
                    receive = asyncio.create_task(websocket.recv())
                    stopping = asyncio.create_task(stop.wait())
                    try:
                        while not stop.is_set():
                            done, _ = await asyncio.wait(
                                {receive, stopping, heartbeat},
                                timeout=_INACTIVITY_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                raise TimeoutError("CLOB inactivity")
                            if heartbeat in done:
                                heartbeat.result()
                                if stop.is_set():
                                    return
                                raise RuntimeError("CLOB heartbeat stopped")
                            if stopping in done and stopping.result():
                                return
                            raw = receive.result()
                            sequence += 1
                            item = PolymarketClobLaneReceipt(
                                lane_id=lane,
                                connection_id=connection_id,
                                sequence_number=sequence,
                                received_wall_ms=int(self._wall_clock_ms()),
                                received_monotonic_ns=int(self._monotonic_ns()),
                                raw_text=_text_frame(raw),
                            ).validated()
                            self._enqueue(output, item)
                            receive = asyncio.create_task(websocket.recv())
                    finally:
                        for task in (heartbeat, receive, stopping):
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(
                            heartbeat,
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
                        stream="clob_market",
                        connection_id=connection_id,
                        last_sequence_number=sequence,
                        error=exc,
                    ),
                )
                if (
                    connected_at > 0.0
                    and asyncio.get_running_loop().time() - connected_at
                    >= _STABLE_CONNECTION_SECONDS
                ):
                    backoff = 1.0
                await self._wait_backoff(stop, backoff)
                backoff = min(30.0, backoff * 2.0)

    @staticmethod
    def _validate_chainlink_text(raw_text: str, *, received_at_ms: int) -> None:
        parse_round21_chainlink_wire_text(
            raw_text,
            received_at_ms=received_at_ms,
        )

    async def _chainlink(
        self,
        output: asyncio.Queue[FeedItem],
        stop: asyncio.Event,
    ) -> None:
        subscription = _canonical_json(
            {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "crypto_prices_chainlink",
                        "type": "update",
                        "filters": _canonical_json({"symbol": "BTC/USD"}),
                    }
                ],
            }
        )
        backoff = 1.0
        while not stop.is_set():
            connection_id = self._connection_id("rtds:chainlink:btc")
            sequence = 0
            connected_at = 0.0
            try:
                async with self._connector(
                    ROUND21_POLYMARKET_RTDS_WEBSOCKET,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=_RTDS_MAX_MESSAGE_BYTES,
                    max_queue=_RTDS_MAX_QUEUE_FRAMES,
                    compression=None,
                ) as websocket:
                    connected_at = asyncio.get_running_loop().time()
                    await websocket.send(subscription)
                    heartbeat = asyncio.create_task(
                        self._heartbeat(websocket, stop, interval_seconds=5.0)
                    )
                    receive = asyncio.create_task(websocket.recv())
                    stopping = asyncio.create_task(stop.wait())
                    try:
                        while not stop.is_set():
                            done, _ = await asyncio.wait(
                                {receive, stopping, heartbeat},
                                timeout=_INACTIVITY_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                raise TimeoutError("RTDS inactivity")
                            if heartbeat in done:
                                heartbeat.result()
                                if stop.is_set():
                                    return
                                raise RuntimeError("RTDS heartbeat stopped")
                            if stopping in done and stopping.result():
                                return
                            raw_text = _text_frame(receive.result())
                            sequence += 1
                            wall = int(self._wall_clock_ms())
                            self._validate_chainlink_text(
                                raw_text,
                                received_at_ms=wall,
                            )
                            item = CaptureFrameRecord(
                                stream="polymarket_rtds",
                                connection_id=connection_id,
                                sequence_number=sequence,
                                received_wall_ms=wall,
                                received_monotonic_ns=int(self._monotonic_ns()),
                                raw_text=raw_text,
                            )
                            self._enqueue(output, item)
                            if raw_text == "PING":
                                await websocket.send("PONG")
                            receive = asyncio.create_task(websocket.recv())
                    finally:
                        for task in (heartbeat, receive, stopping):
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(
                            heartbeat,
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
                        stream="polymarket_rtds",
                        connection_id=connection_id,
                        last_sequence_number=sequence,
                        error=exc,
                    ),
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
                if isinstance(item, PolymarketClobLaneReceipt):
                    self.coordinator.ingest_clob_receipt(item)
                elif isinstance(item, CaptureFrameRecord):
                    self.coordinator.ingest_chainlink_record(item)
                else:
                    self.coordinator.record_gap(item)
                self._record_processed(queue_size=output.qsize())
            finally:
                output.task_done()

    async def run(self, stop: asyncio.Event) -> None:
        """Run until requested Stop; any local processing fault propagates."""

        if not isinstance(stop, asyncio.Event):
            raise TypeError("Round 21 public-feed Stop type differs")
        with self._state_lock:
            if self._running:
                raise RuntimeError("Round 21 public feed is already running")
            if self._started:
                raise RuntimeError("Round 21 public feed cannot be restarted")
            self._running = True
            self._started = True
            self._started_at_ms = int(self._wall_clock_ms())
        output: asyncio.Queue[FeedItem] = asyncio.Queue(
            maxsize=self.queue_capacity
        )
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._consume(output, stop))
                for lane in ROUND21_PUBLIC_FEED_CLOB_LANES:
                    group.create_task(self._clob_lane(lane, output, stop))
                group.create_task(self._chainlink(output, stop))
        finally:
            with self._state_lock:
                self._running = False
                self._queue_size = output.qsize()


credentials_used = False
account_connected = False
binance_connected = False
trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_PUBLIC_FEED_SCHEMA_VERSION",
    "ROUND21_CLOB_MARKET_WEBSOCKET",
    "ROUND21_POLYMARKET_RTDS_WEBSOCKET",
    "ROUND21_PUBLIC_FEED_CLOB_LANES",
    "Round21PolymarketPublicFeed",
    "Round21PublicFeedHealth",
]
