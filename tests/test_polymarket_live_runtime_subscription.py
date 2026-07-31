from __future__ import annotations

import asyncio
import json

import pytest

from simple_ai_trading.polymarket_live_runtime import (
    PolymarketAuthenticatedUserStream,
    PolymarketLiveRuntimeGuard,
)
from simple_ai_trading.polymarket_live_v2 import PolymarketLiveCredentials


CONDITION_A = "0x" + "1" * 64
CONDITION_B = "0x" + "2" * 64


class _Consumer:
    def __init__(self, runtime_guard: PolymarketLiveRuntimeGuard | None = None) -> None:
        self.runtime_guard = runtime_guard or PolymarketLiveRuntimeGuard()

    def handle(self, message: str) -> int:
        if message == "PONG":
            self.runtime_guard.note_stream_liveness()
        return 0


class _WebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._never = asyncio.Event()

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        await self._never.wait()
        raise AssertionError("unreachable")


class _Connection:
    def __init__(self, websocket: _WebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> _WebSocket:
        return self.websocket

    async def __aexit__(self, *_args: object) -> None:
        return None


def _credentials() -> PolymarketLiveCredentials:
    return PolymarketLiveCredentials(
        private_key="0x" + "1" * 64,
        api_key="api-key-test",
        api_secret="api-secret-test",
        api_passphrase="passphrase-test",
        funder_address="0x" + "2" * 40,
        signature_type=1,
    )


def test_user_stream_dynamic_market_updates_are_exact() -> None:
    async def scenario() -> None:
        stream = PolymarketAuthenticatedUserStream(
            _credentials(),
            _Consumer(),  # type: ignore[arg-type]
            markets=(CONDITION_A,),
        )
        websocket = _WebSocket()
        stop = asyncio.Event()
        connected = asyncio.create_task(
            stream._connected(
                websocket,
                stop,
                active_markets=(CONDITION_A,),
            )
        )
        assert await stream.subscribe_markets((CONDITION_B, CONDITION_B)) == (
            CONDITION_B,
        )
        for _ in range(20):
            await asyncio.sleep(0)
            updates = [
                json.loads(message)
                for message in websocket.sent
                if message.startswith("{")
            ]
            if updates:
                break
        assert updates == [
            {
                "markets": [CONDITION_B],
                "operation": "subscribe",
            }
        ]
        assert await stream.subscribe_markets((CONDITION_B,)) == ()
        assert await stream.unsubscribe_markets((CONDITION_A,)) == (CONDITION_A,)
        for _ in range(20):
            await asyncio.sleep(0)
            updates = [
                json.loads(message)
                for message in websocket.sent
                if message.startswith("{")
            ]
            if len(updates) == 2:
                break
        assert updates[-1] == {
            "markets": [CONDITION_A],
            "operation": "unsubscribe",
        }
        assert stream.markets == (CONDITION_B,)
        stop.set()
        await connected

    asyncio.run(scenario())


def test_user_stream_reconnect_snapshot_tracks_current_markets() -> None:
    async def scenario() -> None:
        stream = PolymarketAuthenticatedUserStream(
            _credentials(),
            _Consumer(),  # type: ignore[arg-type]
        )
        await stream.subscribe_markets((CONDITION_B, CONDITION_A))
        payload = json.loads(stream._subscription(await stream._market_snapshot()))

        assert payload["markets"] == [CONDITION_A, CONDITION_B]
        assert payload["type"] == "user"

    asyncio.run(scenario())


def test_user_stream_is_not_authoritative_before_first_server_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import websockets

        runtime_guard = PolymarketLiveRuntimeGuard()
        consumer = _Consumer(runtime_guard)
        websocket = _WebSocket()
        monkeypatch.setattr(
            websockets,
            "connect",
            lambda *_args, **_kwargs: _Connection(websocket),
        )
        stream = PolymarketAuthenticatedUserStream(
            _credentials(),
            consumer,  # type: ignore[arg-type]
            markets=(CONDITION_A,),
        )
        stop = asyncio.Event()
        running = asyncio.create_task(stream.run(stop))
        for _ in range(20):
            await asyncio.sleep(0)
            if websocket.sent:
                break

        assert json.loads(websocket.sent[0])["type"] == "user"
        assert runtime_guard.snapshot().stream_connected is False

        stop.set()
        await running

    asyncio.run(scenario())


def test_user_stream_rejects_invalid_dynamic_condition() -> None:
    async def scenario() -> None:
        stream = PolymarketAuthenticatedUserStream(
            _credentials(),
            _Consumer(),  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="condition ID"):
            await stream.subscribe_markets(("btc-updown-5m",))
        with pytest.raises(ValueError, match="empty"):
            await stream.unsubscribe_markets(())

    asyncio.run(scenario())
