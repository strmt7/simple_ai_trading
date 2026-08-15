"""Market-rolling public-data session for independent Round 21 decisions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any

from .polymarket import PolymarketFiveMinuteMarket, PolymarketPublicClient
from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_round21_binance_feed import (
    Round21BinancePublicFeedHealth,
    Round21BinancePublicSidecar,
)
from .polymarket_round21_live_features import (
    Round21CoordinatedPrediction,
    Round21LiveFeatureCoordinator,
    Round21PublicSourceGap,
)
from .polymarket_round21_prospective import Round21ProspectiveScorer
from .polymarket_round21_public_feed import (
    Round21PolymarketPublicFeed,
    Round21PublicFeedHealth,
)


POLYMARKET_ROUND21_ROLLING_SESSION_SCHEMA_VERSION = (
    "polymarket-round21-rolling-public-session-v1"
)
_DECISION_CADENCE_MS = 250


@dataclass(frozen=True, slots=True)
class Round21RollingSessionHealth:
    running: bool
    active_condition_id: str
    discovery_healthy: bool
    discovery_count: int
    discovery_failure_count: int
    market_activation_count: int
    market_rollover_count: int
    last_discovery_error: str
    active_feed: Round21PublicFeedHealth | None
    optional_binance_feed: Round21BinancePublicFeedHealth | None
    model_tcn_training_backend_kind: str
    model_tcn_training_backend_device: str
    model_tcn_runtime_backend_kind: str
    model_tcn_runtime_backend_device: str
    model_tcn_backend_substituted: bool
    model_tcn_accelerator_fallback: bool
    credentials_used: bool = False
    account_connected: bool = False
    binance_connected: bool = False
    trading_authority: bool = False


class Round21RollingPublicDataService:
    """Roll public CLOB/Chainlink state across exact BTC five-minute markets."""

    credentials_used = False
    account_connected = False
    binance_connected = False
    trading_authority = False

    def __init__(
        self,
        *,
        public_client: PolymarketPublicClient,
        scorer: Round21ProspectiveScorer,
        discovery_interval_seconds: float = 1.0,
        queue_capacity: int = 20_000,
        feed_factory: Callable[..., Round21PolymarketPublicFeed] | None = None,
        binance_sidecar_factory: Callable[..., Round21BinancePublicSidecar]
        | None = None,
        wall_clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(public_client, PolymarketPublicClient):
            raise TypeError("Round 21 rolling-session public client differs")
        if not isinstance(scorer, Round21ProspectiveScorer):
            raise TypeError("Round 21 rolling-session scorer differs")
        interval = float(discovery_interval_seconds)
        capacity = int(queue_capacity)
        if not 0.25 <= interval <= 30:
            raise ValueError("Round 21 discovery interval is invalid")
        if not 1_000 <= capacity <= 100_000:
            raise ValueError("Round 21 rolling-session queue capacity is invalid")
        if feed_factory is not None and not callable(feed_factory):
            raise TypeError("Round 21 rolling-session feed factory is invalid")
        if binance_sidecar_factory is not None and not callable(
            binance_sidecar_factory
        ):
            raise TypeError("Round 21 Binance sidecar factory is invalid")
        if wall_clock_ms is not None and not callable(wall_clock_ms):
            raise TypeError("Round 21 rolling-session wall clock is invalid")
        if not callable(monotonic_ns):
            raise TypeError("Round 21 rolling-session monotonic clock is invalid")
        self.public_client = public_client
        self.scorer = scorer
        self.discovery_interval_seconds = interval
        self.queue_capacity = capacity
        self._feed_factory = feed_factory or Round21PolymarketPublicFeed
        self._binance_sidecar_factory = (
            binance_sidecar_factory or Round21BinancePublicSidecar
        )
        if scorer.population_layer == "core":
            self._binance_markets: tuple[str, ...] = ()
        elif scorer.population_layer == "core_spot":
            self._binance_markets = ("spot",)
        elif scorer.population_layer == "core_spot_usdm":
            self._binance_markets = ("spot", "usdm")
        else:
            raise ValueError("Round 21 rolling-session population layer differs")
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1_000))
        self._monotonic_ns = monotonic_ns
        self._lock = Lock()
        self._running = False
        self._started = False
        self._discovery_healthy = False
        self._discovery_count = 0
        self._discovery_failures = 0
        self._activations = 0
        self._rollovers = 0
        self._last_discovery_error = ""
        self._last_activated_condition_id = ""
        self._active_market: PolymarketFiveMinuteMarket | None = None
        self._active_coordinator: Round21LiveFeatureCoordinator | None = None
        self._active_feed: Round21PolymarketPublicFeed | None = None
        self._last_feed_health: Round21PublicFeedHealth | None = None
        self._binance_sidecar: Round21BinancePublicSidecar | None = None
        self._last_binance_health: Round21BinancePublicFeedHealth | None = None

    def _record_discovery_failure(self, error: BaseException) -> None:
        with self._lock:
            self._discovery_healthy = False
            self._discovery_failures += 1
            self._last_discovery_error = error.__class__.__name__

    def _record_discovery_success(self) -> None:
        with self._lock:
            self._discovery_healthy = True
            self._discovery_count += 1
            self._last_discovery_error = ""

    def health(self) -> Round21RollingSessionHealth:
        with self._lock:
            feed = self._active_feed
            active_health = self._last_feed_health if feed is None else feed.health()
            binance = self._binance_sidecar
            binance_health = (
                self._last_binance_health if binance is None else binance.health()
            )
            return Round21RollingSessionHealth(
                running=self._running,
                active_condition_id=(
                    ""
                    if self._active_market is None
                    else self._active_market.condition_id
                ),
                discovery_healthy=self._discovery_healthy,
                discovery_count=self._discovery_count,
                discovery_failure_count=self._discovery_failures,
                market_activation_count=self._activations,
                market_rollover_count=self._rollovers,
                last_discovery_error=self._last_discovery_error,
                active_feed=active_health,
                optional_binance_feed=binance_health,
                model_tcn_training_backend_kind=(self.scorer.tcn_training_backend_kind),
                model_tcn_training_backend_device=(
                    self.scorer.tcn_training_backend_device
                ),
                model_tcn_runtime_backend_kind=self.scorer.tcn_runtime_backend_kind,
                model_tcn_runtime_backend_device=(
                    self.scorer.tcn_runtime_backend_device
                ),
                model_tcn_backend_substituted=(self.scorer.tcn_backend_substituted),
                model_tcn_accelerator_fallback=(self.scorer.tcn_accelerator_fallback),
            )

    def current_market(self) -> PolymarketFiveMinuteMarket | None:
        """Return the immutable active market without another Gamma request."""

        with self._lock:
            return self._active_market

    def _ingest_optional_binance_record(self, record: CaptureFrameRecord) -> None:
        with self._lock:
            coordinator = self._active_coordinator
        if coordinator is not None:
            coordinator.ingest_optional_binance_record(record)

    def _record_optional_binance_gap(self, gap: Round21PublicSourceGap) -> None:
        with self._lock:
            coordinator = self._active_coordinator
        if coordinator is not None:
            coordinator.record_gap(gap)

    def _build_binance_sidecar(self) -> Round21BinancePublicSidecar | None:
        if not self._binance_markets:
            return None
        sidecar = self._binance_sidecar_factory(
            markets=self._binance_markets,
            record_consumer=self._ingest_optional_binance_record,
            gap_consumer=self._record_optional_binance_gap,
            queue_capacity=self.queue_capacity,
        )
        if not isinstance(sidecar, Round21BinancePublicSidecar):
            raise TypeError("Round 21 rolling-session Binance sidecar differs")
        return sidecar

    async def _discover(self, observed_at_ms: int) -> PolymarketFiveMinuteMarket:
        markets = await asyncio.to_thread(
            self.public_client.discover_five_minute_markets,
            now_ms=observed_at_ms,
            include_next=True,
            require_all_assets=True,
            assets=("BTC",),
        )
        if any(
            market.asset != "BTC" or market.horizon_minutes != 5 for market in markets
        ):
            raise ValueError("Round 21 discovery returned an out-of-scope market")
        active = tuple(
            market
            for market in markets
            if market.event_start_ms <= observed_at_ms < market.end_ms
        )
        if len(active) != 1:
            raise ValueError("Round 21 discovery requires one current BTC market")
        return active[0]

    def _build_feed(
        self,
        market: PolymarketFiveMinuteMarket,
    ) -> tuple[Round21LiveFeatureCoordinator, Round21PolymarketPublicFeed]:
        coordinator = Round21LiveFeatureCoordinator(
            market=market,
            scorer=self.scorer,
        )
        feed = self._feed_factory(
            market=market,
            coordinator=coordinator,
            queue_capacity=self.queue_capacity,
        )
        if not isinstance(feed, Round21PolymarketPublicFeed):
            raise TypeError("Round 21 rolling-session feed differs")
        return coordinator, feed

    def _activate(
        self,
        market: PolymarketFiveMinuteMarket,
        coordinator: Round21LiveFeatureCoordinator,
        feed: Round21PolymarketPublicFeed,
    ) -> None:
        with self._lock:
            self._active_market = market
            self._active_coordinator = coordinator
            self._active_feed = feed
            self._activations += 1
            if (
                self._last_activated_condition_id
                and self._last_activated_condition_id != market.condition_id
            ):
                self._rollovers += 1
            self._last_activated_condition_id = market.condition_id

    def _deactivate(self, feed: Round21PolymarketPublicFeed) -> None:
        health = feed.health()
        with self._lock:
            self._last_feed_health = health
            if self._active_feed is feed:
                self._active_market = None
                self._active_coordinator = None
                self._active_feed = None

    async def _stop_feed(
        self,
        feed: Round21PolymarketPublicFeed | None,
        feed_stop: asyncio.Event | None,
        feed_task: asyncio.Task[None] | None,
    ) -> None:
        if feed is None or feed_stop is None or feed_task is None:
            return
        feed_stop.set()
        try:
            await feed_task
        finally:
            self._deactivate(feed)

    async def _wait_iteration(
        self,
        stop: asyncio.Event,
        feed_task: asyncio.Task[None] | None,
    ) -> None:
        stopping = asyncio.create_task(stop.wait())
        interval = asyncio.create_task(asyncio.sleep(self.discovery_interval_seconds))
        tasks: set[asyncio.Task[Any]] = {stopping, interval}
        if feed_task is not None:
            tasks.add(feed_task)
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if feed_task is not None and feed_task in done:
                await feed_task
                if not stop.is_set():
                    raise RuntimeError("Round 21 public feed returned unexpectedly")
        finally:
            for task in (stopping, interval):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stopping, interval, return_exceptions=True)

    async def _run_polymarket(self, stop: asyncio.Event) -> None:
        feed: Round21PolymarketPublicFeed | None = None
        feed_stop: asyncio.Event | None = None
        feed_task: asyncio.Task[None] | None = None
        try:
            while not stop.is_set():
                observed = int(self._wall_clock_ms())
                if feed is not None and observed < feed.market.end_ms:
                    await self._wait_iteration(stop, feed_task)
                    continue
                try:
                    market = await self._discover(observed)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._record_discovery_failure(exc)
                    await self._wait_iteration(stop, feed_task)
                    continue
                self._record_discovery_success()
                if feed is None or feed.market.condition_id != market.condition_id:
                    await self._stop_feed(feed, feed_stop, feed_task)
                    coordinator, feed = self._build_feed(market)
                    feed_stop = asyncio.Event()
                    self._activate(market, coordinator, feed)
                    feed_task = asyncio.create_task(feed.run(feed_stop))
                await self._wait_iteration(stop, feed_task)
        finally:
            await self._stop_feed(feed, feed_stop, feed_task)

    async def run(self, stop: asyncio.Event) -> None:
        """Run independent public feeds until Stop; local faults propagate."""

        if not isinstance(stop, asyncio.Event):
            raise TypeError("Round 21 rolling-session Stop type differs")
        with self._lock:
            if self._running:
                raise RuntimeError("Round 21 rolling session is already running")
            if self._started:
                raise RuntimeError("Round 21 rolling session cannot be restarted")
            self._running = True
            self._started = True
        sidecar = self._build_binance_sidecar()
        with self._lock:
            self._binance_sidecar = sidecar
        try:
            if sidecar is None:
                await self._run_polymarket(stop)
            else:
                async with asyncio.TaskGroup() as group:
                    group.create_task(self._run_polymarket(stop))
                    group.create_task(sidecar.run(stop))
        finally:
            with self._lock:
                if sidecar is not None:
                    self._last_binance_health = sidecar.health()
                self._binance_sidecar = None
                self._running = False

    def evaluate(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        observed_at_ms: int,
    ) -> Round21CoordinatedPrediction | None:
        """Return one target-free score or None while public state is unsafe."""

        if not isinstance(market, PolymarketFiveMinuteMarket):
            raise TypeError("Round 21 rolling-session decision market differs")
        observed = int(observed_at_ms)
        with self._lock:
            active = self._active_market
            coordinator = self._active_coordinator
            healthy = self._discovery_healthy
        if (
            not healthy
            or active is None
            or coordinator is None
            or active.condition_id != market.condition_id
            or market.asset != "BTC"
            or market.horizon_minutes != 5
            or not market.event_start_ms <= observed < market.end_ms
        ):
            return None
        decision = (
            market.event_start_ms
            + ((observed - market.event_start_ms) // _DECISION_CADENCE_MS)
            * _DECISION_CADENCE_MS
        )
        result = coordinator.evaluate(
            decision_time_ms=decision,
            observed_at_ms=observed,
            observed_monotonic_ns=int(self._monotonic_ns()),
        )
        with self._lock:
            if (
                not self._discovery_healthy
                or self._active_coordinator is not coordinator
                or self._active_market is None
                or self._active_market.condition_id != market.condition_id
            ):
                return None
        return result


credentials_used = False
account_connected = False
binance_connected = False
trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_ROLLING_SESSION_SCHEMA_VERSION",
    "Round21RollingPublicDataService",
    "Round21RollingSessionHealth",
]
