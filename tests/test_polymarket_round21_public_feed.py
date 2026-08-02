from __future__ import annotations

import asyncio
import itertools
import json
from unittest.mock import Mock

import pytest

from simple_ai_trading.polymarket_recorder import (
    CLOB_MARKET_WEBSOCKET,
    POLYMARKET_RTDS_WEBSOCKET,
)
from simple_ai_trading.polymarket_round21_live_features import (
    Round21LiveFeatureCoordinator,
)
from simple_ai_trading.polymarket_round21_public_feed import (
    ROUND21_CLOB_MARKET_WEBSOCKET,
    ROUND21_POLYMARKET_RTDS_WEBSOCKET,
    Round21PolymarketPublicFeed,
)

from polymarket_round21_support import round21_replay_condition


MARKET = round21_replay_condition().market


class _Socket:
    def __init__(self, messages: tuple[object, ...]) -> None:
        self.messages = iter(messages)
        self.sent = []
        self.closed = asyncio.Event()

    async def send(self, message: str) -> None:
        self.sent.append(message)

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
        self.calls = []
        self.clob_sockets = [_Socket(("PONG",)), _Socket(("PONG",))]
        self.clob = list(self.clob_sockets)
        self.rtds = _Socket(("PING",))

    def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if url == ROUND21_POLYMARKET_RTDS_WEBSOCKET:
            return _Context(self.rtds)
        return _Context(self.clob.pop(0))


def _coordinator() -> Mock:
    coordinator = Mock(spec=Round21LiveFeatureCoordinator)
    coordinator.market = MARKET
    return coordinator


def _feed(*, connector=None, coordinator=None) -> Round21PolymarketPublicFeed:
    ticks = itertools.count(1)
    return Round21PolymarketPublicFeed(
        market=MARKET,
        coordinator=coordinator or _coordinator(),
        connector=connector or _Connector(),
        wall_clock_ms=lambda: MARKET.event_start_ms + next(ticks),
        monotonic_ns=lambda: next(ticks),
        connection_id_factory=lambda lane: f"{lane}:connection",
    )


def test_public_endpoints_match_recorder_and_have_no_binance_surface() -> None:
    assert ROUND21_CLOB_MARKET_WEBSOCKET == CLOB_MARKET_WEBSOCKET
    assert ROUND21_POLYMARKET_RTDS_WEBSOCKET == POLYMARKET_RTDS_WEBSOCKET
    feed = _feed()
    assert feed.credentials_used is False
    assert feed.account_connected is False
    assert feed.binance_connected is False
    assert not hasattr(feed, "binance_client")


def test_feed_connects_redundant_clob_and_chainlink_without_persistence() -> None:
    async def run() -> tuple[_Connector, Mock, Round21PolymarketPublicFeed]:
        connector = _Connector()
        coordinator = _coordinator()
        feed = _feed(connector=connector, coordinator=coordinator)
        stop = asyncio.Event()
        task = asyncio.create_task(feed.run(stop))
        for _ in range(100):
            if feed.health().processed_message_count >= 3:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        return connector, coordinator, feed

    connector, coordinator, feed = asyncio.run(run())

    health = feed.health()
    assert health.running is False
    assert health.received_message_count == 3
    assert health.processed_message_count == 3
    assert health.gap_count == 0
    assert health.clob_message_count == 2
    assert health.chainlink_message_count == 1
    assert health.queue_high_watermark <= 3
    assert coordinator.ingest_clob_receipt.call_count == 2
    assert coordinator.ingest_chainlink_record.call_count == 1
    assert coordinator.record_gap.call_count == 0
    assert len(connector.calls) == 3
    clob_subscriptions = [
        json.loads(socket.sent[0]) for socket in connector.clob_sockets
    ]
    assert sum(url == ROUND21_CLOB_MARKET_WEBSOCKET for url, _ in connector.calls) == 2
    assert clob_subscriptions == [
        {
            "assets_ids": sorted(MARKET.token_ids),
            "custom_feature_enabled": True,
            "type": "market",
        },
        {
            "assets_ids": sorted(MARKET.token_ids),
            "custom_feature_enabled": True,
            "type": "market",
        },
    ]
    rtds_subscription = json.loads(connector.rtds.sent[0])
    assert rtds_subscription["subscriptions"] == [
        {
            "filters": json.dumps(
                {"symbol": "BTC/USD"},
                separators=(",", ":"),
                sort_keys=True,
            ),
            "topic": "crypto_prices_chainlink",
            "type": "update",
        }
    ]


class _FailingSocket(_Socket):
    async def recv(self) -> object:
        raise ConnectionError("offline")


class _FailingConnector(_Connector):
    def __init__(self) -> None:
        super().__init__()
        self.clob_sockets = [_FailingSocket(()), _Socket(())]
        self.clob = list(self.clob_sockets)


def test_network_failure_is_a_gap_and_does_not_touch_execution() -> None:
    async def run() -> tuple[Mock, Round21PolymarketPublicFeed]:
        connector = _FailingConnector()
        coordinator = _coordinator()
        stop = asyncio.Event()

        def record_gap(_gap) -> None:
            stop.set()

        coordinator.record_gap.side_effect = record_gap
        feed = _feed(connector=connector, coordinator=coordinator)
        await asyncio.wait_for(feed.run(stop), timeout=2.0)
        return coordinator, feed

    coordinator, feed = asyncio.run(run())

    assert coordinator.record_gap.call_count >= 1
    gap = coordinator.record_gap.call_args.args[0]
    assert gap.stream == "clob_market"
    assert gap.reason == "ConnectionError"
    assert feed.health().gap_count >= 1
    assert feed.binance_connected is False


def test_queue_saturation_fails_loudly() -> None:
    feed = _feed()
    output = asyncio.Queue(maxsize=1)
    first = feed._gap(
        stream="clob_market",
        connection_id="clob-a:connection",
        last_sequence_number=0,
        error=ConnectionError(),
    )
    second = feed._gap(
        stream="clob_market",
        connection_id="clob-b:connection",
        last_sequence_number=0,
        error=ConnectionError(),
    )
    feed._enqueue(output, first)
    with pytest.raises(MemoryError, match="queue saturated"):
        feed._enqueue(output, second)


def test_chainlink_rejects_schema_drift_before_coordinator_ingress() -> None:
    Round21PolymarketPublicFeed._validate_chainlink_text(
        json.dumps(
            {
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "timestamp": MARKET.event_start_ms,
                "payload": {
                    "symbol": "btc/usd",
                    "timestamp": MARKET.event_start_ms,
                    "value": 60_000,
                },
            }
        ),
        received_at_ms=MARKET.event_start_ms,
    )
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        Round21PolymarketPublicFeed._validate_chainlink_text(
            '{"topic":"x","topic":"y"}',
            received_at_ms=MARKET.event_start_ms,
        )


def test_feed_rejects_invalid_connection_identity_and_restart() -> None:
    invalid = Round21PolymarketPublicFeed(
        market=MARKET,
        coordinator=_coordinator(),
        connection_id_factory=lambda _lane: "contains space",
    )
    with pytest.raises(ValueError, match="connection identity"):
        invalid._connection_id("clob-a")

    async def run_twice() -> None:
        feed = _feed()
        stop = asyncio.Event()
        stop.set()
        await feed.run(stop)
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            await feed.run(stop)

    asyncio.run(run_twice())
    with pytest.raises(ValueError, match="envelope identity"):
        Round21PolymarketPublicFeed._validate_chainlink_text(
            json.dumps(
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "timestamp": MARKET.event_start_ms,
                    "payload": {
                        "symbol": "btcusdt",
                        "timestamp": MARKET.event_start_ms,
                        "value": 60_000,
                    },
                }
            ),
            received_at_ms=MARKET.event_start_ms,
        )
