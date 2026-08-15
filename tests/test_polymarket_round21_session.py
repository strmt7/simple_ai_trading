from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from simple_ai_trading.polymarket import PolymarketPublicClient
from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket_round21_binance_feed import (
    Round21BinancePublicFeedHealth,
    Round21BinancePublicSidecar,
)
from simple_ai_trading.polymarket_round21_live_features import (
    Round21PublicSourceGap,
)
from simple_ai_trading.polymarket_round21_prospective import (
    Round21ProspectiveScorer,
)
from simple_ai_trading.polymarket_round21_public_feed import (
    Round21PolymarketPublicFeed,
    Round21PublicFeedHealth,
)
from simple_ai_trading.polymarket_round21_session import (
    Round21RollingPublicDataService,
)

from polymarket_round21_support import round21_replay_condition


MARKET = round21_replay_condition().market


def _next_market():
    return replace(
        MARKET,
        condition_id="0x" + "d" * 64,
        market_id="next-market",
        slug="btc-updown-5m-next",
        event_start_ms=MARKET.event_start_ms + 300_000,
        end_ms=MARKET.end_ms + 300_000,
        up_token_id="8" * 32,
        down_token_id="9" * 32,
    )


def _feed_health(*, running: bool = False) -> Round21PublicFeedHealth:
    return Round21PublicFeedHealth(
        running=running,
        received_message_count=0,
        processed_message_count=0,
        gap_count=0,
        clob_message_count=0,
        chainlink_message_count=0,
        queue_capacity=20_000,
        queue_size=0,
        queue_high_watermark=0,
        started_at_ms=MARKET.event_start_ms,
        last_receipt_at_ms=0,
        last_gap_sha256="",
    )


class _FeedFactory:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.feeds: list[Mock] = []

    def __call__(self, *, market, coordinator, queue_capacity):
        assert queue_capacity == 20_000
        feed = Mock(spec=Round21PolymarketPublicFeed)
        feed.market = market
        feed.coordinator = coordinator
        feed.health.return_value = _feed_health()

        async def run(stop: asyncio.Event) -> None:
            if self.fail:
                raise RuntimeError("feed failed")
            await stop.wait()

        feed.run.side_effect = run
        self.feeds.append(feed)
        return feed


def _service(
    *,
    client,
    clock,
    factory,
    layer="core",
    sidecar_factory=None,
):
    scorer = Mock(spec=Round21ProspectiveScorer)
    scorer.population_layer = layer
    scorer.tcn_training_backend_kind = "cpu"
    scorer.tcn_training_backend_device = "cpu"
    scorer.tcn_runtime_backend_kind = "cpu"
    scorer.tcn_runtime_backend_device = "cpu"
    scorer.tcn_backend_substituted = False
    scorer.tcn_accelerator_fallback = False
    return Round21RollingPublicDataService(
        public_client=client,
        scorer=scorer,
        discovery_interval_seconds=0.25,
        feed_factory=factory,
        binance_sidecar_factory=sidecar_factory,
        wall_clock_ms=clock,
    )


async def _wait_for(predicate, *, attempts: int = 200) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


def test_session_activates_and_returns_fail_closed_causal_state() -> None:
    async def exercise():
        now = MARKET.event_start_ms + 60_000
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.return_value = (MARKET,)
        factory = _FeedFactory()
        service = _service(client=client, clock=lambda: now, factory=factory)
        assert service.evaluate(MARKET, observed_at_ms=now) is None
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        await _wait_for(lambda: service.health().market_activation_count == 1)
        assert service.current_market() == MARKET

        result = service.evaluate(MARKET, observed_at_ms=now)
        assert result is not None
        assert result.status == "abstain"
        assert result.live_trading_authority is False
        stop.set()
        await task
        return service, factory

    service, factory = asyncio.run(exercise())
    health = service.health()
    assert service.current_market() is None
    assert health.running is False
    assert health.discovery_healthy is True
    assert health.discovery_count >= 1
    assert health.market_activation_count == 1
    assert health.market_rollover_count == 0
    assert service.public_client.discover_five_minute_markets.call_count == 1
    assert len(factory.feeds) == 1
    assert not any(
        (
            service.credentials_used,
            service.account_connected,
            service.binance_connected,
            service.trading_authority,
        )
    )


def test_session_rolls_market_without_reusing_feed_or_coordinator() -> None:
    async def exercise():
        second = _next_market()
        current = {"now": MARKET.event_start_ms + 60_000}
        client = Mock(spec=PolymarketPublicClient)

        def discover(*, now_ms, **_kwargs):
            return (MARKET,) if now_ms < second.event_start_ms else (second,)

        client.discover_five_minute_markets.side_effect = discover
        factory = _FeedFactory()
        service = _service(
            client=client,
            clock=lambda: current["now"],
            factory=factory,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        await _wait_for(lambda: service.health().market_activation_count == 1)
        first_coordinator = factory.feeds[0].coordinator
        current["now"] = second.event_start_ms + 1_000
        await _wait_for(lambda: service.health().market_activation_count == 2)
        assert factory.feeds[1].coordinator is not first_coordinator
        assert service.evaluate(MARKET, observed_at_ms=current["now"]) is None
        assert service.evaluate(second, observed_at_ms=current["now"]) is not None
        stop.set()
        await task
        return service, factory

    service, factory = asyncio.run(exercise())
    assert service.health().market_rollover_count == 1
    assert len(factory.feeds) == 2
    assert all(feed.run.await_count == 1 for feed in factory.feeds)


def test_session_retries_discovery_but_blocks_decisions_while_unhealthy() -> None:
    async def exercise():
        now = MARKET.event_start_ms + 60_000
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.side_effect = (
            ConnectionError("offline"),
            (MARKET,),
        )
        factory = _FeedFactory()
        service = _service(client=client, clock=lambda: now, factory=factory)
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        await _wait_for(lambda: service.health().discovery_failure_count == 1)
        assert service.health().discovery_healthy is False
        assert service.evaluate(MARKET, observed_at_ms=now) is None
        await _wait_for(lambda: service.health().market_activation_count == 1)
        assert service.health().discovery_healthy is True
        stop.set()
        await task
        return service

    service = asyncio.run(exercise())
    assert service.health().last_discovery_error == ""
    assert service.health().discovery_count >= 1


def test_session_propagates_a_public_feed_processing_failure() -> None:
    async def exercise():
        now = MARKET.event_start_ms + 60_000
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.return_value = (MARKET,)
        service = _service(
            client=client,
            clock=lambda: now,
            factory=_FeedFactory(fail=True),
        )
        with pytest.raises(RuntimeError, match="feed failed"):
            await service.run(asyncio.Event())
        assert service.health().running is False

    asyncio.run(exercise())


def test_session_rejects_restart_and_out_of_scope_market() -> None:
    async def exercise():
        now = MARKET.event_start_ms + 60_000
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.return_value = (MARKET,)
        service = _service(
            client=client,
            clock=lambda: now,
            factory=_FeedFactory(),
        )
        stop = asyncio.Event()
        stop.set()
        await service.run(stop)
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            await service.run(stop)

    asyncio.run(exercise())


def test_optional_binance_sidecar_is_selected_by_sealed_population_layer() -> None:
    class SidecarFactory:
        def __init__(self) -> None:
            self.sidecar = Mock(spec=Round21BinancePublicSidecar)
            self.sidecar.health.return_value = Round21BinancePublicFeedHealth(
                running=False,
                enabled_markets=("spot",),
                received_message_count=0,
                processed_message_count=0,
                gap_count=0,
                spot_message_count=0,
                usdm_message_count=0,
                reconnect_counts={"spot": 0, "usdm": 0},
                queue_capacity=20_000,
                queue_size=0,
                queue_high_watermark=0,
                started_at_ms=MARKET.event_start_ms,
                last_receipt_at_ms=0,
                last_gap_sha256="",
            )
            self.record_consumer = None
            self.gap_consumer = None

        def __call__(
            self,
            *,
            markets,
            record_consumer,
            gap_consumer,
            queue_capacity,
        ):
            assert markets == ("spot",)
            assert queue_capacity == 20_000
            self.record_consumer = record_consumer
            self.gap_consumer = gap_consumer

            async def run(stop: asyncio.Event) -> None:
                await stop.wait()

            self.sidecar.run.side_effect = run
            return self.sidecar

    async def exercise():
        now = MARKET.event_start_ms + 60_000
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.return_value = (MARKET,)
        feed_factory = _FeedFactory()
        sidecar_factory = SidecarFactory()
        service = _service(
            client=client,
            clock=lambda: now,
            factory=feed_factory,
            layer="core_spot",
            sidecar_factory=sidecar_factory,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(service.run(stop))
        await _wait_for(lambda: service.health().market_activation_count == 1)
        assert sidecar_factory.record_consumer is not None
        record = CaptureFrameRecord(
            stream="binance_spot",
            connection_id="binance:spot:connection",
            sequence_number=1,
            received_wall_ms=now,
            received_monotonic_ns=1,
            raw_text=json.dumps(
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
            ),
        )
        sidecar_factory.record_consumer(record)
        coordinator = feed_factory.feeds[0].coordinator
        assert coordinator._optional_connections == {"spot": "binance:spot:connection"}
        assert sidecar_factory.gap_consumer is not None
        sidecar_factory.gap_consumer(
            Round21PublicSourceGap(
                stream="binance_spot",
                connection_id="binance:spot:connection",
                observed_wall_ms=now + 1,
                observed_monotonic_ns=2,
                last_sequence_number=1,
                reason="disconnect",
            )
        )
        assert coordinator._optional_connections == {}
        stop.set()
        await task
        return service, sidecar_factory

    service, sidecar_factory = asyncio.run(exercise())
    health = service.health()
    assert health.optional_binance_feed is not None
    assert health.optional_binance_feed.binance_execution_connected is False
    assert sidecar_factory.sidecar.run.await_count == 1


def test_core_population_never_constructs_binance_sidecar() -> None:
    sidecar_calls = 0

    def forbidden_sidecar(**_kwargs):
        nonlocal sidecar_calls
        sidecar_calls += 1
        raise AssertionError("core Polymarket data requested a Binance sidecar")

    async def exercise() -> Round21RollingPublicDataService:
        now = MARKET.event_start_ms + 60_000
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.return_value = (MARKET,)
        service = _service(
            client=client,
            clock=lambda: now,
            factory=_FeedFactory(),
            layer="core",
            sidecar_factory=forbidden_sidecar,
        )
        stop = asyncio.Event()
        stop.set()
        await service.run(stop)
        return service

    service = asyncio.run(exercise())

    assert sidecar_calls == 0
    health = service.health()
    assert health.optional_binance_feed is None
    assert health.model_tcn_training_backend_kind == "cpu"
    assert health.model_tcn_runtime_backend_kind == "cpu"
    assert health.model_tcn_backend_substituted is False
    assert health.model_tcn_accelerator_fallback is False


def test_optional_binance_sidecar_processing_failure_stops_session() -> None:
    sidecar = Mock(spec=Round21BinancePublicSidecar)
    sidecar.health.return_value = Round21BinancePublicFeedHealth(
        running=False,
        enabled_markets=("spot",),
        received_message_count=0,
        processed_message_count=0,
        gap_count=0,
        spot_message_count=0,
        usdm_message_count=0,
        reconnect_counts={"spot": 0, "usdm": 0},
        queue_capacity=20_000,
        queue_size=0,
        queue_high_watermark=0,
        started_at_ms=MARKET.event_start_ms,
        last_receipt_at_ms=0,
        last_gap_sha256="",
    )

    async def failed_run(_stop: asyncio.Event) -> None:
        raise RuntimeError("sidecar processing failed")

    sidecar.run.side_effect = failed_run

    async def exercise() -> None:
        client = Mock(spec=PolymarketPublicClient)
        client.discover_five_minute_markets.return_value = (MARKET,)
        service = _service(
            client=client,
            clock=lambda: MARKET.event_start_ms + 60_000,
            factory=_FeedFactory(),
            layer="core_spot",
            sidecar_factory=lambda **_kwargs: sidecar,
        )
        with pytest.raises(ExceptionGroup) as raised:
            await service.run(asyncio.Event())
        assert any(
            isinstance(error, RuntimeError)
            and str(error) == "sidecar processing failed"
            for error in raised.value.exceptions
        )
        assert service.health().running is False

    asyncio.run(exercise())
