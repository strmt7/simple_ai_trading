"""Independent runtime authority and authenticated Polymarket user stream."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
import json
from threading import RLock
import time
from typing import Callable, Mapping, Sequence

from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderLedger,
    PolymarketLiveOrderRecord,
    PolymarketReconciliation,
    PolymarketRemoteFill,
)
from .polymarket_live_v2 import PolymarketLiveCredentials


POLYMARKET_USER_STREAM_URL = (
    "wss://ws-subscriptions-clob.polymarket.com/ws/user"
)


@dataclass(frozen=True, slots=True)
class PolymarketRuntimeSnapshot:
    stream_connected: bool
    stream_age_ms: int | None
    reconciliation_age_ms: int | None
    reconciliation_can_open: bool
    reconciliation_can_close: bool
    soft_fault: str
    hard_faults: tuple[str, ...]
    stopped: bool


class PolymarketLiveRuntimeGuard:
    """Fail-closed liveness state independent from model and execution loops."""

    def __init__(
        self,
        *,
        maximum_stream_age_ms: int = 15_000,
        maximum_reconciliation_age_ms: int = 30_000,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.maximum_stream_age_ms = int(maximum_stream_age_ms)
        self.maximum_reconciliation_age_ms = int(
            maximum_reconciliation_age_ms
        )
        if not 1_000 <= self.maximum_stream_age_ms <= 120_000:
            raise ValueError("maximum_stream_age_ms must lie in [1000, 120000]")
        if not 1_000 <= self.maximum_reconciliation_age_ms <= 300_000:
            raise ValueError(
                "maximum_reconciliation_age_ms must lie in [1000, 300000]"
            )
        self._monotonic_ns = monotonic_ns
        self._lock = RLock()
        self._stream_connected = False
        self._last_stream_ns: int | None = None
        self._last_reconciliation_ns: int | None = None
        self._reconciliation_can_open = False
        self._reconciliation_can_close = False
        self._soft_fault = "runtime_booting"
        self._hard_faults: set[str] = set()
        self._stopped = False

    def note_stream_liveness(self) -> None:
        with self._lock:
            self._stream_connected = True
            self._last_stream_ns = self._monotonic_ns()
            if self._soft_fault in {
                "runtime_booting",
                "stream_disconnected",
                "stream_timeout",
            }:
                self._soft_fault = ""

    def note_stream_disconnected(self, failure_code: str) -> None:
        code = str(failure_code or "").strip() or "stream_disconnected"
        with self._lock:
            self._stream_connected = False
            self._soft_fault = code

    def note_hard_fault(self, failure_code: str) -> None:
        code = str(failure_code or "").strip()
        if not code:
            raise ValueError("hard runtime fault code cannot be empty")
        with self._lock:
            self._hard_faults.add(code)

    def note_reconciliation(self, result: PolymarketReconciliation) -> None:
        with self._lock:
            self._last_reconciliation_ns = self._monotonic_ns()
            self._reconciliation_can_open = result.can_open
            self._reconciliation_can_close = result.can_close
            if not result.ok:
                self._soft_fault = "reconciliation:" + ",".join(result.errors)
            elif self._stream_connected:
                self._soft_fault = ""

    def note_reconciliation_failure(self, failure_code: str) -> None:
        code = str(failure_code or "").strip() or "unknown"
        with self._lock:
            self._last_reconciliation_ns = None
            self._reconciliation_can_open = False
            self._reconciliation_can_close = False
            self._soft_fault = f"reconciliation_failure:{code}"

    def mark_stopped(self) -> None:
        with self._lock:
            self._stopped = True
            self._stream_connected = False
            self._soft_fault = "runtime_stopped"

    def snapshot(self) -> PolymarketRuntimeSnapshot:
        now = self._monotonic_ns()
        with self._lock:
            stream_age = (
                None
                if self._last_stream_ns is None
                else max(0, (now - self._last_stream_ns) // 1_000_000)
            )
            reconciliation_age = (
                None
                if self._last_reconciliation_ns is None
                else max(
                    0,
                    (now - self._last_reconciliation_ns) // 1_000_000,
                )
            )
            return PolymarketRuntimeSnapshot(
                stream_connected=self._stream_connected,
                stream_age_ms=stream_age,
                reconciliation_age_ms=reconciliation_age,
                reconciliation_can_open=self._reconciliation_can_open,
                reconciliation_can_close=self._reconciliation_can_close,
                soft_fault=self._soft_fault,
                hard_faults=tuple(sorted(self._hard_faults)),
                stopped=self._stopped,
            )

    def assert_submission_allowed(self, *, closing_only: bool) -> None:
        state = self.snapshot()
        if state.stopped:
            raise PolymarketLiveBlocked("Polymarket runtime is stopped")
        if state.hard_faults:
            raise PolymarketLiveBlocked(
                f"Polymarket runtime has hard faults: {state.hard_faults}"
            )
        if (
            state.reconciliation_age_ms is None
            or state.reconciliation_age_ms > self.maximum_reconciliation_age_ms
        ):
            raise PolymarketLiveBlocked("Polymarket reconciliation is stale")
        if closing_only:
            if not state.reconciliation_can_close:
                raise PolymarketLiveBlocked(
                    "Polymarket reconciliation does not permit closing"
                )
            return
        if (
            not state.stream_connected
            or state.stream_age_ms is None
            or state.stream_age_ms > self.maximum_stream_age_ms
        ):
            raise PolymarketLiveBlocked("Polymarket user stream is stale")
        if not state.reconciliation_can_open:
            raise PolymarketLiveBlocked(
                "Polymarket reconciliation does not permit new exposure"
            )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _observed_at_ms(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return int(time.time() * 1_000)
    parsed = int(text)
    return parsed if parsed >= 10_000_000_000 else parsed * 1_000


class PolymarketUserStreamConsumer:
    """Apply only exact bot-owned user events to the durable ledger."""

    def __init__(
        self,
        ledger: PolymarketLiveOrderLedger,
        runtime_guard: PolymarketLiveRuntimeGuard,
    ) -> None:
        self.ledger = ledger
        self.runtime_guard = runtime_guard

    def _owned_records(self) -> dict[str, PolymarketLiveOrderRecord]:
        return {
            record.expected_order_id: record for record in self.ledger.records()
        }

    def _handle_order(self, payload: Mapping[str, object]) -> None:
        order_id = str(payload.get("id") or "").strip().lower()
        records = self._owned_records()
        if order_id not in records:
            self.runtime_guard.note_hard_fault("foreign_order_stream_event")
            return
        record = records[order_id]
        market_id = str(payload.get("market") or "").strip().lower()
        token_id = str(payload.get("asset_id") or "").strip()
        side = str(payload.get("side") or "").strip().upper()
        original_quantity = Decimal(str(payload.get("original_size")))
        matched_quantity = Decimal(str(payload.get("size_matched") or "0"))
        if (
            market_id != record.intent.market_id
            or token_id != record.intent.token_id
            or side != record.intent.side
            or original_quantity != record.intent.quantity
            or matched_quantity < 0
            or matched_quantity > original_quantity
        ):
            self.runtime_guard.note_hard_fault("owned_order_stream_identity_mismatch")
            return
        event_type = str(payload.get("type") or "").strip().upper()
        if record.state not in {
            "prepared",
            "submitting",
            "unknown",
            "live",
            "partial",
            "matched_pending",
            "cancel_pending",
            "cancel_unknown",
        }:
            return
        if event_type == "CANCELLATION":
            next_state = "cancelled"
        elif event_type in {"PLACEMENT", "UPDATE"}:
            next_state = "partial" if matched_quantity > 0 else "live"
        else:
            raise ValueError("unknown Polymarket order stream event")
        self.ledger.transition(
            record.intent.intent_id,
            expected_states=(record.state,),
            state=next_state,
            observed_at_ms=_observed_at_ms(payload.get("timestamp")),
            remote_status=event_type,
            matched_quantity=matched_quantity,
        )

    def _fill(
        self,
        payload: Mapping[str, object],
        *,
        order_id: str,
        token_id: str,
        side: str,
        quantity: object,
        price: object,
    ) -> PolymarketRemoteFill:
        status = str(payload.get("status") or "").strip().upper()
        if status.startswith("TRADE_STATUS_"):
            status = status.removeprefix("TRADE_STATUS_")
        return PolymarketRemoteFill(
            trade_id=str(payload.get("id") or ""),
            order_id=order_id,
            market_id=str(payload.get("market") or ""),
            token_id=token_id,
            side=side,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            status=status,
            observed_at_ms=_observed_at_ms(
                payload.get("last_update")
                or payload.get("timestamp")
                or payload.get("matchtime")
            ),
        )

    def _handle_trade(self, payload: Mapping[str, object]) -> None:
        owned = set(self._owned_records())
        fills: list[PolymarketRemoteFill] = []
        taker_order_id = str(payload.get("taker_order_id") or "").strip().lower()
        if taker_order_id in owned:
            fills.append(
                self._fill(
                    payload,
                    order_id=taker_order_id,
                    token_id=str(payload.get("asset_id") or ""),
                    side=str(payload.get("side") or ""),
                    quantity=payload.get("size"),
                    price=payload.get("price"),
                )
            )
        maker_orders = payload.get("maker_orders") or []
        if not isinstance(maker_orders, list):
            raise ValueError("Polymarket maker order stream payload is invalid")
        taker_side = str(payload.get("side") or "").strip().upper()
        if taker_side not in {"BUY", "SELL"}:
            raise ValueError("Polymarket trade stream side is invalid")
        for raw in maker_orders:
            maker = _mapping(raw, name="maker order stream event")
            maker_id = str(maker.get("order_id") or "").strip().lower()
            if maker_id not in owned:
                continue
            maker_side = str(maker.get("side") or "").strip().upper()
            if not maker_side:
                maker_side = "SELL" if taker_side == "BUY" else "BUY"
            fills.append(
                self._fill(
                    payload,
                    order_id=maker_id,
                    token_id=str(maker.get("asset_id") or ""),
                    side=maker_side,
                    quantity=maker.get("matched_amount"),
                    price=maker.get("price"),
                )
            )
        if not fills:
            self.runtime_guard.note_hard_fault("foreign_trade_stream_event")
            return
        for fill in fills:
            self.ledger.record_fill(fill)

    def handle(self, raw_message: str | bytes) -> int:
        if isinstance(raw_message, bytes):
            message = raw_message.decode("utf-8")
        else:
            message = raw_message
        if message.strip().upper() == "PONG":
            self.runtime_guard.note_stream_liveness()
            return 0
        payload = json.loads(message)
        events: Sequence[object] = payload if isinstance(payload, list) else (payload,)
        handled = 0
        for raw in events:
            event = _mapping(raw, name="user stream event")
            event_type = str(event.get("event_type") or "").strip().lower()
            wire_type = str(event.get("type") or "").strip().lower()
            if event_type == "order" or wire_type in {
                "placement",
                "update",
                "cancellation",
            }:
                self._handle_order(event)
            elif event_type == "trade" or wire_type == "trade":
                self._handle_trade(event)
            else:
                raise ValueError("unknown Polymarket user stream event")
            handled += 1
        self.runtime_guard.note_stream_liveness()
        return handled


class PolymarketAuthenticatedUserStream:
    """Reconnectable user stream with bounded messages and application PINGs."""

    def __init__(
        self,
        credentials: PolymarketLiveCredentials,
        consumer: PolymarketUserStreamConsumer,
        *,
        markets: Sequence[str] = (),
        message_timeout_seconds: float = 15.0,
    ) -> None:
        self.credentials = credentials
        self.consumer = consumer
        self.markets = tuple(dict.fromkeys(str(value).lower() for value in markets))
        self.message_timeout_seconds = max(
            11.0,
            min(60.0, float(message_timeout_seconds)),
        )

    def _subscription(self) -> str:
        return json.dumps(
            {
                "auth": {
                    "apiKey": self.credentials.api_key,
                    "secret": self.credentials.api_secret,
                    "passphrase": self.credentials.api_passphrase,
                },
                "markets": list(self.markets),
                "type": "user",
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    async def _connected(self, websocket: object, stop: asyncio.Event) -> None:
        async def ping_loop() -> None:
            while not stop.is_set():
                await websocket.send("PING")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=10.0)
                except TimeoutError:
                    continue

        async def receive_loop() -> None:
            while not stop.is_set():
                message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=self.message_timeout_seconds,
                )
                self.consumer.handle(message)

        ping = asyncio.create_task(ping_loop())
        receive = asyncio.create_task(receive_loop())
        stopped = asyncio.create_task(stop.wait())
        tasks = {ping, receive, stopped}
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is not stopped:
                    task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self, stop: asyncio.Event) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Polymarket live stream requires websockets") from exc
        delay_seconds = 0.5
        while not stop.is_set():
            try:
                async with websockets.connect(
                    POLYMARKET_USER_STREAM_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    open_timeout=10,
                    max_size=2 * 1024 * 1024,
                    max_queue=256,
                ) as websocket:
                    await websocket.send(self._subscription())
                    self.consumer.runtime_guard.note_stream_liveness()
                    delay_seconds = 0.5
                    await self._connected(websocket, stop)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.consumer.runtime_guard.note_stream_disconnected(
                    f"stream_disconnected:{exc.__class__.__name__}"
                )
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay_seconds)
                except TimeoutError:
                    delay_seconds = min(10.0, delay_seconds * 2)
        self.consumer.runtime_guard.note_stream_disconnected("stream_stopped")


class PolymarketReconciliationService:
    """Run blocking authenticated reconciliation off the event loop."""

    def __init__(
        self,
        coordinator: PolymarketLiveCoordinator,
        runtime_guard: PolymarketLiveRuntimeGuard,
        *,
        interval_seconds: float = 5.0,
    ) -> None:
        self.coordinator = coordinator
        self.runtime_guard = runtime_guard
        self.interval_seconds = max(1.0, min(60.0, float(interval_seconds)))

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.coordinator.reconcile)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.runtime_guard.note_reconciliation_failure(
                    exc.__class__.__name__
                )
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self.interval_seconds,
                )
            except TimeoutError:
                continue


__all__ = [
    "POLYMARKET_USER_STREAM_URL",
    "PolymarketAuthenticatedUserStream",
    "PolymarketLiveRuntimeGuard",
    "PolymarketReconciliationService",
    "PolymarketRuntimeSnapshot",
    "PolymarketUserStreamConsumer",
]
