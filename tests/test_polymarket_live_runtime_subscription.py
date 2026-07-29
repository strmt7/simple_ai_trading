from __future__ import annotations

import asyncio
import json

import pytest

from simple_ai_trading.polymarket_live_runtime import (
    PolymarketAuthenticatedUserStream,
)
from simple_ai_trading.polymarket_live_v2 import PolymarketLiveCredentials


CONDITION_A = "0x" + "1" * 64
CONDITION_B = "0x" + "2" * 64


class _Consumer:
    def handle(self, _message: str) -> int:
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
