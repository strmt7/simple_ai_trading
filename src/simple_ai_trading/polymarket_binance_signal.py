"""Credential-free Binance BTC observations for Polymarket advisory use only."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from threading import RLock
import time
from typing import Mapping

from .polymarket_autonomous import PolymarketAutonomousOpenProposal
from .polymarket_external_signal import (
    BtcPriceDiscoveryTick,
    PolymarketBtcPriceDiscoveryMonitor,
    PolymarketExternalSignalDecision,
)


BINANCE_SPOT_BTC_TICKER_URL = (
    "wss://data-stream.binance.vision/ws/btcusdt@ticker"
)
BINANCE_FUTURES_BTC_BOOK_TICKER_URL = (
    "wss://fstream.binance.com/ws/btcusdt@bookTicker"
)


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value):
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def parse_binance_btc_public_tick(
    source: str,
    payload: Mapping[str, object],
    *,
    received_at_ms: int,
    spot_sequence: int | None = None,
) -> BtcPriceDiscoveryTick:
    """Parse documented public ticker payloads without synthesizing event time."""

    selected = str(source or "").strip().upper()
    event_type = str(payload.get("e") or "").strip()
    symbol = str(payload.get("s") or "").strip().upper()
    event_time_ms = _positive_integer(payload.get("E"), name="event time")
    received = _positive_integer(received_at_ms, name="received time")
    if symbol != "BTCUSDT":
        raise ValueError("Binance public signal symbol differs")
    if selected == "BINANCE_SPOT":
        if event_type != "24hrTicker" or spot_sequence is None:
            raise ValueError("Binance Spot ticker payload differs")
        sequence = _positive_integer(spot_sequence, name="Spot receipt sequence")
    elif selected == "BINANCE_USD_M_FUTURES":
        if event_type != "bookTicker":
            raise ValueError("Binance USD-M book ticker payload differs")
        sequence = _positive_integer(payload.get("u"), name="Futures update ID")
        transaction_time = _positive_integer(
            payload.get("T"),
            name="Futures transaction time",
        )
        if transaction_time > event_time_ms:
            raise ValueError("Binance Futures transaction time is invalid")
    else:
        raise ValueError("unsupported Binance public signal source")
    bid = _positive_decimal(payload.get("b"), name="best bid")
    ask = _positive_decimal(payload.get("a"), name="best ask")
    _positive_decimal(payload.get("B"), name="best bid quantity")
    _positive_decimal(payload.get("A"), name="best ask quantity")
    return BtcPriceDiscoveryTick(
        source=selected,
        symbol=symbol,
        event_time_ms=event_time_ms,
        received_at_ms=received,
        sequence=sequence,
        bid=bid,
        ask=ask,
    )


@dataclass(frozen=True, slots=True)
class BinanceBtcSignalSnapshot:
    spot_connected: bool
    futures_connected: bool
    spot_age_ms: int | None
    futures_age_ms: int | None
    spot_fault: str
    futures_fault: str
    credentials_used: bool
    execution_authority: bool


class BinanceBtcPublicSignalProvider:
    """Maintain two public BTC feeds and expose a fail-closed advisory decision."""

    def __init__(
        self,
        *,
        monitor: PolymarketBtcPriceDiscoveryMonitor | None = None,
        message_timeout_seconds: float = 5.0,
    ) -> None:
        timeout = float(message_timeout_seconds)
        if not 2 <= timeout <= 30:
            raise ValueError("Binance public signal timeout must lie in [2, 30]")
        self.monitor = monitor or PolymarketBtcPriceDiscoveryMonitor()
        self.message_timeout_seconds = timeout
        self._lock = RLock()
        self._ticks: dict[str, BtcPriceDiscoveryTick] = {}
        self._connected = {
            "BINANCE_SPOT": False,
            "BINANCE_USD_M_FUTURES": False,
        }
        self._faults = {
            "BINANCE_SPOT": "not_started",
            "BINANCE_USD_M_FUTURES": "not_started",
        }
        self._spot_sequence = 0

    @staticmethod
    def _abstain(reason: str) -> PolymarketExternalSignalDecision:
        return PolymarketExternalSignalDecision(
            action="abstain",
            maximum_size_multiplier=Decimal("0"),
            reasons=(reason,),
            features=None,
        )

    def _note_disconnected(self, source: str, fault: str) -> None:
        with self._lock:
            self._connected[source] = False
            self._faults[source] = fault

    def _handle_message(
        self,
        source: str,
        raw_message: str | bytes,
        *,
        received_at_ms: int | None = None,
    ) -> BtcPriceDiscoveryTick:
        if isinstance(raw_message, bytes):
            message = raw_message.decode("utf-8")
        else:
            message = raw_message
        payload = json.loads(message)
        if not isinstance(payload, Mapping):
            raise ValueError("Binance public signal payload must be an object")
        selected = str(source or "").strip().upper()
        with self._lock:
            if selected == "BINANCE_SPOT":
                self._spot_sequence += 1
                spot_sequence = self._spot_sequence
            else:
                spot_sequence = None
        tick = parse_binance_btc_public_tick(
            selected,
            payload,
            received_at_ms=(
                int(time.time_ns() // 1_000_000)
                if received_at_ms is None
                else int(received_at_ms)
            ),
            spot_sequence=spot_sequence,
        )
        with self._lock:
            prior = self._ticks.get(selected)
            if prior is not None and (
                tick.sequence <= prior.sequence
                or tick.event_time_ms < prior.event_time_ms
            ):
                raise ValueError("Binance public signal sequence regressed")
            self._ticks[selected] = tick
            self._connected[selected] = True
            self._faults[selected] = ""
        return tick

    async def _feed(
        self,
        source: str,
        url: str,
        stop: asyncio.Event,
    ) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "Binance public signal requires websockets"
            ) from exc
        delay_seconds = 0.5
        while not stop.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    open_timeout=10,
                    max_size=64 * 1024,
                    max_queue=64,
                ) as websocket:
                    delay_seconds = 0.5
                    while not stop.is_set():
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=self.message_timeout_seconds,
                        )
                        self._handle_message(source, message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._note_disconnected(
                    source,
                    f"{exc.__class__.__name__}",
                )
                try:
                    await asyncio.wait_for(
                        stop.wait(),
                        timeout=delay_seconds,
                    )
                except TimeoutError:
                    delay_seconds = min(10.0, delay_seconds * 2)
        self._note_disconnected(source, "stopped")

    async def run(self, stop: asyncio.Event) -> None:
        tasks = (
            asyncio.create_task(
                self._feed(
                    "BINANCE_SPOT",
                    BINANCE_SPOT_BTC_TICKER_URL,
                    stop,
                )
            ),
            asyncio.create_task(
                self._feed(
                    "BINANCE_USD_M_FUTURES",
                    BINANCE_FUTURES_BTC_BOOK_TICKER_URL,
                    stop,
                )
            ),
        )
        try:
            await stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def evaluate(
        self,
        *,
        proposal: PolymarketAutonomousOpenProposal,
        observed_at_ms: int,
    ) -> PolymarketExternalSignalDecision:
        if not isinstance(proposal, PolymarketAutonomousOpenProposal):
            return self._abstain("proposal_type_mismatch")
        with self._lock:
            spot = self._ticks.get("BINANCE_SPOT")
            futures = self._ticks.get("BINANCE_USD_M_FUTURES")
            connected = dict(self._connected)
        if spot is None or not connected["BINANCE_SPOT"]:
            return self._abstain("binance_spot_unavailable")
        if futures is None or not connected["BINANCE_USD_M_FUTURES"]:
            return self._abstain("binance_futures_unavailable")
        return self.monitor.evaluate(
            spot=spot,
            futures=futures,
            observed_at_ms=int(observed_at_ms),
        )

    def snapshot(
        self,
        *,
        observed_at_ms: int | None = None,
    ) -> BinanceBtcSignalSnapshot:
        now = (
            int(time.time_ns() // 1_000_000)
            if observed_at_ms is None
            else int(observed_at_ms)
        )
        with self._lock:
            spot = self._ticks.get("BINANCE_SPOT")
            futures = self._ticks.get("BINANCE_USD_M_FUTURES")
            connected = dict(self._connected)
            faults = dict(self._faults)
        return BinanceBtcSignalSnapshot(
            spot_connected=connected["BINANCE_SPOT"],
            futures_connected=connected["BINANCE_USD_M_FUTURES"],
            spot_age_ms=(
                None if spot is None else max(0, now - spot.received_at_ms)
            ),
            futures_age_ms=(
                None if futures is None else max(0, now - futures.received_at_ms)
            ),
            spot_fault=faults["BINANCE_SPOT"],
            futures_fault=faults["BINANCE_USD_M_FUTURES"],
            credentials_used=False,
            execution_authority=False,
        )


__all__ = [
    "BINANCE_FUTURES_BTC_BOOK_TICKER_URL",
    "BINANCE_SPOT_BTC_TICKER_URL",
    "BinanceBtcPublicSignalProvider",
    "BinanceBtcSignalSnapshot",
    "parse_binance_btc_public_tick",
]
