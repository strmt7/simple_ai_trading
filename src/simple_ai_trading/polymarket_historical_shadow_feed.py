"""Public dual-Binance transport for non-authoritative Polymarket shadow scoring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import time
from typing import Mapping

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .polymarket_historical_shadow import (
    BtcAggregateTradeObservation,
    PolymarketBtcFlowBuffer,
)


BINANCE_SPOT_AGGREGATE_TRADE_URL = (
    "wss://data-stream.binance.vision/ws/btcusdt@aggTrade"
)
BINANCE_USDM_AGGREGATE_TRADE_URL = (
    "wss://fstream.binance.com/market/ws/btcusdt@aggTrade"
)
_MARKETS = ("spot", "perpetual")
_REQUIRED_KEYS = {
    "spot": frozenset({"e", "E", "s", "a", "p", "q", "f", "l", "T", "m", "M"}),
    "perpetual": frozenset(
        {"e", "E", "s", "a", "p", "q", "nq", "f", "l", "T", "m", "st"}
    ),
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Binance aggregate-trade JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Binance aggregate-trade JSON contains {value}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(parsed) != str(value) or parsed < minimum:
        raise ValueError(f"{name} must be an integer")
    return parsed


def parse_binance_aggregate_trade(
    raw: str | bytes,
    *,
    market: str,
    received_at_ms: int,
) -> BtcAggregateTradeObservation:
    """Strictly decode one current production aggregate-trade payload."""

    normalized = str(market or "").strip().lower()
    if normalized not in _MARKETS:
        raise ValueError("Binance aggregate-trade market is invalid")
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Binance aggregate-trade payload is not UTF-8") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("Binance aggregate-trade payload must be text or bytes")
    if not text or len(text.encode("utf-8")) > 16 * 1024:
        raise ValueError("Binance aggregate-trade payload size is invalid")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Binance aggregate-trade payload is not strict JSON") from exc
    payload = _mapping(value, name="Binance aggregate-trade payload")
    if set(payload) != _REQUIRED_KEYS[normalized]:
        raise ValueError("Binance aggregate-trade schema drifted")
    if payload["e"] != "aggTrade" or payload["s"] != "BTCUSDT":
        raise ValueError("Binance aggregate-trade stream identity differs")
    if type(payload["m"]) is not bool:
        raise ValueError("Binance aggregate-trade maker flag is invalid")
    event_time = _integer(payload["E"], name="event time", minimum=1)
    trade_time = _integer(payload["T"], name="trade time", minimum=1)
    if abs(event_time - trade_time) > 10_000:
        raise ValueError("Binance aggregate-trade event/transaction skew is invalid")
    quantity = _finite_positive(payload["q"], name="quantity")
    if normalized == "spot":
        if type(payload["M"]) is not bool:
            raise ValueError("Binance spot best-match flag is invalid")
    else:
        if _integer(payload["st"], name="symbol type", minimum=1) != 1:
            raise ValueError("Binance futures stream is not USD-M")
        normal_quantity = _finite_nonnegative(
            payload["nq"],
            name="normal quantity",
        )
        if normal_quantity > quantity + 1e-12:
            raise ValueError("Binance futures normal quantity exceeds quantity")
    return BtcAggregateTradeObservation(
        market=normalized,
        source=(
            "BINANCE_SPOT"
            if normalized == "spot"
            else "BINANCE_USD_M_FUTURES"
        ),
        symbol="BTCUSDT",
        event_time_ms=trade_time,
        received_at_ms=int(received_at_ms),
        aggregate_trade_id=_integer(
            payload["a"],
            name="aggregate trade ID",
        ),
        first_trade_id=_integer(payload["f"], name="first trade ID"),
        last_trade_id=_integer(payload["l"], name="last trade ID"),
        price=_finite_positive(payload["p"], name="price"),
        quantity=quantity,
        buyer_is_maker=payload["m"],
    )


@dataclass(frozen=True, slots=True)
class PolymarketShadowFeedHealth:
    running: bool
    queue_size: int
    queue_capacity: int
    queue_high_watermark: int
    received_counts: Mapping[str, int]
    ingested_counts: Mapping[str, int]
    reconnect_counts: Mapping[str, int]
    stale_epoch_discard_counts: Mapping[str, int]
    last_received_at_ms: Mapping[str, int | None]
    last_event_time_ms: Mapping[str, int | None]
    current_epochs: Mapping[str, int]
    last_errors: Mapping[str, str]
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if self.trading_authority:
            raise ValueError("Polymarket shadow feed cannot have trading authority")


@dataclass(frozen=True, slots=True)
class _QueuedMessage:
    market: str
    epoch: int
    received_at_ms: int
    payload: str | bytes


class PolymarketHistoricalShadowFeed:
    """Reconnect public feeds while isolating each incomplete feed epoch."""

    trading_authority = False

    def __init__(
        self,
        *,
        flow: PolymarketBtcFlowBuffer,
        queue_capacity: int = 32_768,
        receive_timeout_seconds: float = 10.0,
        reconnect_initial_seconds: float = 0.25,
        reconnect_maximum_seconds: float = 15.0,
        spot_url: str = BINANCE_SPOT_AGGREGATE_TRADE_URL,
        futures_url: str = BINANCE_USDM_AGGREGATE_TRADE_URL,
    ) -> None:
        if not isinstance(flow, PolymarketBtcFlowBuffer):
            raise TypeError("flow must be PolymarketBtcFlowBuffer")
        self.flow = flow
        self.queue_capacity = int(queue_capacity)
        self.receive_timeout_seconds = float(receive_timeout_seconds)
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_maximum_seconds = float(reconnect_maximum_seconds)
        if not 1_024 <= self.queue_capacity <= 262_144:
            raise ValueError("shadow feed queue capacity must lie in [1024, 262144]")
        if not 1.0 <= self.receive_timeout_seconds <= 60.0:
            raise ValueError("shadow feed receive timeout must lie in [1, 60] seconds")
        if (
            not 0.05 <= self.reconnect_initial_seconds <= 5.0
            or not self.reconnect_initial_seconds
            <= self.reconnect_maximum_seconds
            <= 60.0
        ):
            raise ValueError("shadow feed reconnect interval is invalid")
        self.urls = {
            "spot": str(spot_url),
            "perpetual": str(futures_url),
        }
        if any(not value.startswith("wss://") for value in self.urls.values()):
            raise ValueError("shadow feed endpoints must use wss")
        self._queue: asyncio.Queue[_QueuedMessage] = asyncio.Queue(
            maxsize=self.queue_capacity
        )
        self._running = False
        self._queue_high_watermark = 0
        self._received_counts = {market: 0 for market in _MARKETS}
        self._ingested_counts = {market: 0 for market in _MARKETS}
        self._reconnect_counts = {market: 0 for market in _MARKETS}
        self._stale_epoch_discards = {market: 0 for market in _MARKETS}
        self._last_received_at_ms: dict[str, int | None] = {
            market: None for market in _MARKETS
        }
        self._last_event_time_ms: dict[str, int | None] = {
            market: None for market in _MARKETS
        }
        self._epochs = {market: 0 for market in _MARKETS}
        self._last_errors = {market: "" for market in _MARKETS}

    def health(self) -> PolymarketShadowFeedHealth:
        return PolymarketShadowFeedHealth(
            running=self._running,
            queue_size=self._queue.qsize(),
            queue_capacity=self.queue_capacity,
            queue_high_watermark=self._queue_high_watermark,
            received_counts=dict(self._received_counts),
            ingested_counts=dict(self._ingested_counts),
            reconnect_counts=dict(self._reconnect_counts),
            stale_epoch_discard_counts=dict(self._stale_epoch_discards),
            last_received_at_ms=dict(self._last_received_at_ms),
            last_event_time_ms=dict(self._last_event_time_ms),
            current_epochs=dict(self._epochs),
            last_errors=dict(self._last_errors),
        )

    async def _wait_backoff(self, stop: asyncio.Event, delay: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except TimeoutError:
            pass

    async def _receiver(self, market: str, stop: asyncio.Event) -> None:
        backoff = self.reconnect_initial_seconds
        while not stop.is_set():
            self._epochs[market] += 1
            epoch = self._epochs[market]
            self.flow.reset_market(market)
            try:
                async with connect(
                    self.urls[market],
                    open_timeout=10.0,
                    close_timeout=3.0,
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    max_queue=2_048,
                    max_size=16 * 1024,
                    compression=None,
                ) as websocket:
                    backoff = self.reconnect_initial_seconds
                    self._last_errors[market] = ""
                    while not stop.is_set():
                        raw = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=self.receive_timeout_seconds,
                        )
                        received_at_ms = time.time_ns() // 1_000_000
                        message = _QueuedMessage(
                            market=market,
                            epoch=epoch,
                            received_at_ms=received_at_ms,
                            payload=raw,
                        )
                        try:
                            await asyncio.wait_for(
                                self._queue.put(message),
                                timeout=0.25,
                            )
                        except TimeoutError as exc:
                            raise RuntimeError(
                                "Polymarket shadow feed queue overflow"
                            ) from exc
                        self._received_counts[market] += 1
                        self._last_received_at_ms[market] = received_at_ms
                        self._queue_high_watermark = max(
                            self._queue_high_watermark,
                            self._queue.qsize(),
                        )
            except asyncio.CancelledError:
                raise
            except RuntimeError:
                raise
            except (TimeoutError, OSError, WebSocketException) as exc:
                self._reconnect_counts[market] += 1
                self._last_errors[market] = type(exc).__name__
                self.flow.reset_market(market)
                await self._wait_backoff(stop, backoff)
                backoff = min(backoff * 2.0, self.reconnect_maximum_seconds)

    async def _consumer(self, stop: asyncio.Event) -> None:
        while not stop.is_set() or not self._queue.empty():
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=0.25)
            except TimeoutError:
                continue
            try:
                if message.epoch != self._epochs[message.market]:
                    self._stale_epoch_discards[message.market] += 1
                    continue
                observation = parse_binance_aggregate_trade(
                    message.payload,
                    market=message.market,
                    received_at_ms=message.received_at_ms,
                )
                if self.flow.ingest(observation):
                    self._ingested_counts[message.market] += 1
                    self._last_event_time_ms[
                        message.market
                    ] = observation.event_time_ms
            finally:
                self._queue.task_done()

    async def run(self, stop: asyncio.Event) -> None:
        if self._running:
            raise RuntimeError("Polymarket shadow feed is already running")
        if not isinstance(stop, asyncio.Event):
            raise TypeError("stop must be asyncio.Event")
        self._running = True
        workers = [
            asyncio.create_task(
                self._receiver("spot", stop),
                name="polymarket-shadow-binance-spot",
            ),
            asyncio.create_task(
                self._receiver("perpetual", stop),
                name="polymarket-shadow-binance-usdm",
            ),
            asyncio.create_task(
                self._consumer(stop),
                name="polymarket-shadow-consumer",
            ),
        ]
        stop_waiter = asyncio.create_task(
            stop.wait(),
            name="polymarket-shadow-stop",
        )
        try:
            done, _ = await asyncio.wait(
                [*workers, stop_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            failed = [
                task
                for task in done
                if task is not stop_waiter
                and not task.cancelled()
                and task.exception() is not None
            ]
            if failed:
                raise failed[0].exception()  # type: ignore[misc]
        finally:
            stop.set()
            stop_waiter.cancel()
            for task in workers:
                task.cancel()
            await asyncio.gather(stop_waiter, *workers, return_exceptions=True)
            self._running = False


__all__ = [
    "BINANCE_SPOT_AGGREGATE_TRADE_URL",
    "BINANCE_USDM_AGGREGATE_TRADE_URL",
    "PolymarketHistoricalShadowFeed",
    "PolymarketShadowFeedHealth",
    "parse_binance_aggregate_trade",
]
