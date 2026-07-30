"""Independent asynchronous supervisor for promoted BTC Polymarket trading."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Callable, Protocol

from .polymarket import PolymarketFiveMinuteMarket, PolymarketPublicClient
from .polymarket_autonomous import (
    PolymarketAutonomousOpenProposal,
    PolymarketAutonomousOpenResult,
    submit_promoted_open,
)
from .polymarket_external_signal import PolymarketExternalSignalDecision
from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderLedger,
)
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion
from .polymarket_live_runtime import (
    PolymarketAuthenticatedUserStream,
    PolymarketLiveRuntimeGuard,
    PolymarketReconciliationService,
)
from .polymarket_live_settlement import PolymarketSettlementService
from .polymarket_live_stop import stop_owned_polymarket_exposure


class PolymarketAutonomousDecisionProvider(Protocol):
    """A promoted model boundary with public market data and no order authority."""

    def decide(
        self,
        *,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        observed_at_ms: int,
    ) -> "PolymarketAutonomousDecision": ...


class PolymarketExternalSignalProvider(Protocol):
    """Optional read-only evidence; implementations cannot access order methods."""

    trading_authority: bool
    credentials_used: bool
    execution_connected: bool

    def evaluate(
        self,
        *,
        proposal: PolymarketAutonomousOpenProposal,
        observed_at_ms: int,
    ) -> PolymarketExternalSignalDecision: ...


class PolymarketDecisionDataService(Protocol):
    """Independent predictor-data loop with no trading authority."""

    trading_authority: bool

    async def run(self, stop: asyncio.Event) -> None: ...


class PolymarketDurableControlService(Protocol):
    """Persistent single-writer and Stop supervision with no order authority."""

    trading_authority: bool

    async def run(
        self,
        stop: asyncio.Event,
        *,
        request_stop: Callable[[], None],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousDecision:
    proposals: tuple[PolymarketAutonomousOpenProposal, ...] = ()
    close_owned_exposure: bool = False
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        proposals = tuple(self.proposals)
        if any(
            not isinstance(item, PolymarketAutonomousOpenProposal) for item in proposals
        ):
            raise TypeError("decision proposals must be Polymarket proposals")
        hashes = tuple(item.proposal_sha256 for item in proposals)
        if len(set(hashes)) != len(hashes):
            raise ValueError("decision contains duplicate Polymarket proposals")
        if self.close_owned_exposure and proposals:
            raise ValueError(
                "decision cannot open and close Polymarket exposure together"
            )
        reasons = tuple(str(item or "").strip() for item in self.reasons)
        if any(not item or len(item) > 160 for item in reasons):
            raise ValueError("decision reason is invalid")
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousRuntimeSnapshot:
    venue: str
    symbol: str
    market_variant: str
    horizon_minutes: int
    paused: bool
    stop_requested: bool
    stop_completed: bool
    discovered_market_ids: tuple[str, ...]
    subscribed_market_ids: tuple[str, ...]
    decisions: int
    submitted_opens: int
    blocked_opens: int
    requested_closes: int
    completed_closes: int
    last_fault: str
    external_signal_enabled: bool
    binance_execution_connected: bool


class PolymarketAutonomousSupervisor:
    """Coordinate independent safety loops and a promotion-gated model loop."""

    def __init__(
        self,
        *,
        public_client: PolymarketPublicClient,
        coordinator: PolymarketLiveCoordinator,
        ledger: PolymarketLiveOrderLedger,
        runtime_guard: PolymarketLiveRuntimeGuard,
        user_stream: PolymarketAuthenticatedUserStream,
        reconciliation: PolymarketReconciliationService,
        promotion: VerifiedPolymarketLivePromotion,
        decision_provider: PolymarketAutonomousDecisionProvider,
        settlement: PolymarketSettlementService | None = None,
        decision_data_service: PolymarketDecisionDataService | None = None,
        durable_control_service: PolymarketDurableControlService | None = None,
        external_signal_provider: PolymarketExternalSignalProvider | None = None,
        decision_interval_seconds: float = 1.0,
        decision_timeout_seconds: float = 3.0,
        forced_exit_seconds: int = 20,
        stop_timeout_seconds: float = 20.0,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(promotion, VerifiedPolymarketLivePromotion):
            raise PolymarketLiveBlocked(
                "autonomous Polymarket runtime requires verified promotion evidence"
            )
        interval = float(decision_interval_seconds)
        timeout = float(decision_timeout_seconds)
        forced_exit = int(forced_exit_seconds)
        stop_timeout = float(stop_timeout_seconds)
        if not 0.25 <= interval <= 30:
            raise ValueError("decision interval must lie in [0.25, 30] seconds")
        if not 0.1 <= timeout <= 30:
            raise ValueError("decision timeout must lie in [0.1, 30] seconds")
        if not 10 <= forced_exit <= 240:
            raise ValueError("forced exit must lie in [10, 240] seconds")
        if not 1 <= stop_timeout <= 300:
            raise ValueError("stop timeout must lie in [1, 300] seconds")
        if coordinator.ledger is not ledger:
            raise ValueError("coordinator and supervisor ledgers differ")
        if coordinator.runtime_authority is not runtime_guard:
            raise ValueError("coordinator and supervisor runtime guards differ")
        if reconciliation.coordinator is not coordinator:
            raise ValueError("reconciliation and supervisor coordinators differ")
        if reconciliation.runtime_guard is not runtime_guard:
            raise ValueError("reconciliation and supervisor runtime guards differ")
        if user_stream.consumer.ledger is not ledger:
            raise ValueError("user stream and supervisor ledgers differ")
        if user_stream.consumer.runtime_guard is not runtime_guard:
            raise ValueError("user stream and supervisor runtime guards differ")
        if decision_data_service is not None:
            if getattr(decision_data_service, "trading_authority", None) is not False:
                raise PolymarketLiveBlocked(
                    "Polymarket predictor-data service must have no trading authority"
                )
            if not callable(getattr(decision_data_service, "run", None)):
                raise TypeError("predictor-data service must expose an async run loop")
        if durable_control_service is not None:
            if getattr(durable_control_service, "trading_authority", None) is not False:
                raise PolymarketLiveBlocked(
                    "Polymarket durable control service must have no trading authority"
                )
            if not callable(getattr(durable_control_service, "run", None)):
                raise TypeError("durable control service must expose an async run loop")
        if external_signal_provider is not None and any(
            getattr(external_signal_provider, name, None) is not False
            for name in (
                "trading_authority",
                "credentials_used",
                "execution_connected",
            )
        ):
            raise PolymarketLiveBlocked(
                "external Polymarket signal provider must be public and read-only"
            )
        self.public_client = public_client
        self.coordinator = coordinator
        self.ledger = ledger
        self.runtime_guard = runtime_guard
        self.user_stream = user_stream
        self.reconciliation = reconciliation
        self.promotion = promotion
        self.decision_provider = decision_provider
        self.settlement = settlement
        self.decision_data_service = decision_data_service
        self.durable_control_service = durable_control_service
        self.external_signal_provider = external_signal_provider
        self.decision_interval_seconds = interval
        self.decision_timeout_seconds = timeout
        self.forced_exit_seconds = forced_exit
        self.stop_timeout_seconds = stop_timeout
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self._shutdown_requested = asyncio.Event()
        self._paused = False
        self._stop_completed = False
        self._markets: tuple[PolymarketFiveMinuteMarket, ...] = ()
        self._decisions = 0
        self._submitted_opens = 0
        self._blocked_opens = 0
        self._requested_closes = 0
        self._completed_closes = 0
        self._last_fault = ""
        self._consumed_proposals: set[str] = set()
        self._pending_decision: asyncio.Task[PolymarketAutonomousDecision] | None = None
        self._pending_signal: (
            tuple[
                str,
                asyncio.Task[PolymarketExternalSignalDecision],
            ]
            | None
        ) = None

    def pause(self) -> None:
        """Block model decisions while reconciliation and close loops continue."""

        self._paused = True

    def resume(self) -> None:
        if self._shutdown_requested.is_set():
            raise PolymarketLiveBlocked("Polymarket Stop is already requested")
        self._paused = False

    def request_stop(self) -> None:
        """Block new entries and keep supervising until owned exposure is closed."""

        self._paused = True
        self._shutdown_requested.set()

    def snapshot(self) -> PolymarketAutonomousRuntimeSnapshot:
        return PolymarketAutonomousRuntimeSnapshot(
            venue="polymarket",
            symbol="BTC",
            market_variant=self.promotion.promotion.market_variant,
            horizon_minutes=(
                5 if self.promotion.promotion.market_variant == "fiveminute" else 15
            ),
            paused=self._paused,
            stop_requested=self._shutdown_requested.is_set(),
            stop_completed=self._stop_completed,
            discovered_market_ids=tuple(
                market.condition_id for market in self._markets
            ),
            subscribed_market_ids=self.user_stream.markets,
            decisions=self._decisions,
            submitted_opens=self._submitted_opens,
            blocked_opens=self._blocked_opens,
            requested_closes=self._requested_closes,
            completed_closes=self._completed_closes,
            last_fault=self._last_fault,
            external_signal_enabled=self.external_signal_provider is not None,
            binance_execution_connected=False,
        )

    def _owned_market_ids(self) -> set[str]:
        markets = {item.market_id for item in self.ledger.owned_inventory()}
        markets.update(
            record.intent.market_id
            for record in self.ledger.records()
            if record.blocks_new_exposure
        )
        return markets

    async def _discover_and_subscribe(
        self,
        *,
        observed_at_ms: int,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        market_variant = self.promotion.promotion.market_variant
        if market_variant == "fiveminute":
            markets = await asyncio.to_thread(
                self.public_client.discover_five_minute_markets,
                now_ms=observed_at_ms,
                include_next=True,
                require_all_assets=True,
                assets=("BTC",),
            )
            expected_horizon = 5
        elif market_variant == "fifteenminute":
            markets = await asyncio.to_thread(
                self.public_client.discover_fifteen_minute_markets,
                now_ms=observed_at_ms,
                include_next=True,
                require_market=True,
            )
            expected_horizon = 15
        else:
            raise PolymarketLiveBlocked(
                "Polymarket promotion market variant is unsupported"
            )
        if any(
            market.asset != "BTC" or market.horizon_minutes != expected_horizon
            for market in markets
        ):
            raise PolymarketLiveBlocked(
                "Polymarket discovery returned an out-of-scope market"
            )
        active = tuple(
            market
            for market in markets
            if market.event_start_ms <= observed_at_ms < market.end_ms
        )
        if len(active) != 1:
            raise PolymarketLiveBlocked(
                "exactly one current BTC promoted-horizon market is required"
            )
        self._markets = markets
        desired = {market.condition_id for market in markets}
        desired.update(self._owned_market_ids())
        current = set(self.user_stream.markets)
        added = tuple(sorted(desired - current))
        removed = tuple(sorted(current - desired))
        if added:
            await self.user_stream.subscribe_markets(added)
        if removed:
            await self.user_stream.unsubscribe_markets(removed)
        return active

    def _assert_proposal_market(
        self,
        proposal: PolymarketAutonomousOpenProposal,
        market: PolymarketFiveMinuteMarket,
    ) -> None:
        token = market.up_token_id if proposal.outcome == "Up" else market.down_token_id
        if (
            proposal.market_id != market.condition_id
            or proposal.token_id != token
            or proposal.event_start_time_ms != market.event_start_ms
            or proposal.event_end_time_ms != market.end_ms
        ):
            raise PolymarketLiveBlocked(
                "model proposal differs from discovered Polymarket identity"
            )

    def _market_has_owned_exposure(self, market_id: str) -> bool:
        return market_id in self._owned_market_ids()

    async def _decision(
        self,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        *,
        observed_at_ms: int,
    ) -> PolymarketAutonomousDecision | None:
        if self._pending_decision is None:
            self._pending_decision = asyncio.create_task(
                asyncio.to_thread(
                    self.decision_provider.decide,
                    markets=markets,
                    observed_at_ms=observed_at_ms,
                )
            )
        try:
            decision = await asyncio.wait_for(
                asyncio.shield(self._pending_decision),
                timeout=self.decision_timeout_seconds,
            )
        except TimeoutError:
            self._last_fault = "decision_timeout"
            return None
        finally:
            if self._pending_decision is not None and self._pending_decision.done():
                self._pending_decision = None
        if not isinstance(decision, PolymarketAutonomousDecision):
            raise TypeError("decision provider returned an invalid decision")
        self._decisions += 1
        return decision

    async def _external_signal(
        self,
        proposal: PolymarketAutonomousOpenProposal,
        *,
        observed_at_ms: int,
    ) -> PolymarketExternalSignalDecision | None:
        if self.external_signal_provider is None:
            return None
        pending = self._pending_signal
        if pending is not None and pending[0] != proposal.proposal_sha256:
            if not pending[1].done():
                raise PolymarketLiveBlocked(
                    "external BTC price discovery is still evaluating"
                )
            self._pending_signal = None
        if self._pending_signal is None:
            self._pending_signal = (
                proposal.proposal_sha256,
                asyncio.create_task(
                    asyncio.to_thread(
                        self.external_signal_provider.evaluate,
                        proposal=proposal,
                        observed_at_ms=observed_at_ms,
                    )
                ),
            )
        try:
            signal = await asyncio.wait_for(
                asyncio.shield(self._pending_signal[1]),
                timeout=self.decision_timeout_seconds,
            )
        except TimeoutError as exc:
            self._last_fault = "external_signal_failure:TimeoutError"
            raise PolymarketLiveBlocked(
                "external BTC price discovery is unavailable"
            ) from exc
        except Exception as exc:
            self._last_fault = f"external_signal_failure:{exc.__class__.__name__}"
            raise PolymarketLiveBlocked(
                "external BTC price discovery is unavailable"
            ) from exc
        finally:
            if self._pending_signal is not None and self._pending_signal[1].done():
                self._pending_signal = None
        if not isinstance(signal, PolymarketExternalSignalDecision):
            raise PolymarketLiveBlocked(
                "external BTC price discovery returned an invalid decision"
            )
        return signal

    async def _close_owned(self) -> bool:
        self._requested_closes += 1
        payload, status = await asyncio.to_thread(
            stop_owned_polymarket_exposure,
            coordinator=self.coordinator,
            ledger=self.ledger,
            timeout_seconds=self.stop_timeout_seconds,
        )
        completed = status == 0 and payload["completed"] is True
        if completed:
            self._completed_closes += 1
            self._last_fault = ""
        else:
            self._last_fault = "owned_exposure_close_incomplete:" + str(
                payload["reason"]
            )
        return completed

    async def _apply_decision(
        self,
        decision: PolymarketAutonomousDecision,
        market: PolymarketFiveMinuteMarket,
        *,
        observed_at_ms: int,
    ) -> None:
        if self._shutdown_requested.is_set():
            return
        if decision.close_owned_exposure and self._owned_market_ids():
            await self._close_owned()
        if self._paused:
            return
        for proposal in decision.proposals:
            if self._shutdown_requested.is_set() or self._paused:
                return
            if proposal.proposal_sha256 in self._consumed_proposals:
                continue
            try:
                self._assert_proposal_market(proposal, market)
                if self._market_has_owned_exposure(proposal.market_id):
                    raise PolymarketLiveBlocked(
                        "bot already owns exposure in this Polymarket event"
                    )
                signal = await self._external_signal(
                    proposal,
                    observed_at_ms=observed_at_ms,
                )
                if self._shutdown_requested.is_set() or self._paused:
                    return
                result = await asyncio.to_thread(
                    submit_promoted_open,
                    proposal,
                    self.promotion,
                    self.coordinator,
                    external_signal=signal,
                    clock_ms=self._clock_ms,
                )
            except PolymarketLiveBlocked as exc:
                self._blocked_opens += 1
                self._last_fault = str(exc)
                continue
            if not isinstance(result, PolymarketAutonomousOpenResult):
                raise TypeError("autonomous open returned an invalid result")
            self._consumed_proposals.add(proposal.proposal_sha256)
            self._submitted_opens += 1
            self._last_fault = ""

    async def _model_loop(self, services_stop: asyncio.Event) -> None:
        while not services_stop.is_set():
            now = self._clock_ms()
            try:
                active = await self._discover_and_subscribe(observed_at_ms=now)
                market = active[0]
                owned = self._owned_market_ids()
                entry_window_closed = (
                    market.end_ms - now <= self.forced_exit_seconds * 1_000
                )
                if self._shutdown_requested.is_set() or entry_window_closed:
                    if owned and await self._close_owned():
                        self._stop_completed = self._shutdown_requested.is_set()
                    if (
                        self._shutdown_requested.is_set()
                        and not self._owned_market_ids()
                    ):
                        self._stop_completed = True
                        services_stop.set()
                        break
                elif not self._paused:
                    decision = await self._decision(active, observed_at_ms=now)
                    if decision is not None:
                        await self._apply_decision(
                            decision,
                            market,
                            observed_at_ms=now,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_fault = f"supervisor_iteration:{exc.__class__.__name__}"
            try:
                await asyncio.wait_for(
                    services_stop.wait(),
                    timeout=self.decision_interval_seconds,
                )
            except TimeoutError:
                continue

    async def run(self, *, duration_seconds: float = 0) -> None:
        """Run until Stop closes owned exposure or the optional duration elapses."""

        duration = float(duration_seconds)
        if not 0 <= duration <= 31_536_000:
            raise ValueError("duration_seconds must lie in [0, 31536000]")
        initial = await asyncio.to_thread(self.coordinator.preflight)
        if not initial.can_close:
            raise PolymarketLiveBlocked(
                f"autonomous supervision preflight failed: {initial.errors}"
            )
        await self._discover_and_subscribe(observed_at_ms=self._clock_ms())
        services_stop = asyncio.Event()
        tasks: dict[str, asyncio.Task[None]] = {
            "authenticated_user_stream": asyncio.create_task(
                self.user_stream.run(services_stop)
            ),
            "ownership_reconciliation": asyncio.create_task(
                self.reconciliation.run(services_stop)
            ),
        }
        if self.settlement is not None:
            tasks["settlement"] = asyncio.create_task(
                self.settlement.run(services_stop)
            )
        if self.decision_data_service is not None:
            tasks["predictor_market_data"] = asyncio.create_task(
                self.decision_data_service.run(services_stop)
            )
        if self.durable_control_service is not None:
            tasks["durable_runtime_control"] = asyncio.create_task(
                self.durable_control_service.run(
                    services_stop,
                    request_stop=self.request_stop,
                )
            )
        external_run = getattr(self.external_signal_provider, "run", None)
        if callable(external_run):
            tasks["external_public_signal"] = asyncio.create_task(
                external_run(services_stop)
            )
        timer: asyncio.Task[None] | None = None
        if duration:

            async def request_stop_after_duration() -> None:
                await asyncio.sleep(duration)
                self.request_stop()

            timer = asyncio.create_task(request_stop_after_duration())
        critical_error: RuntimeError | None = None
        critical_fault = ""
        try:
            # Give safety services one scheduling turn before model decisions.
            await asyncio.sleep(0)
            early = {name: task for name, task in tasks.items() if task.done()}
            if early:
                name, task = next(iter(early.items()))
                exception = task.exception() if not task.cancelled() else None
                detail = (
                    exception.__class__.__name__
                    if exception is not None
                    else "returned"
                )
                critical_fault = f"critical_service_exit:{name}:{detail}"
                self._last_fault = critical_fault
                self.request_stop()
                critical_error = RuntimeError(critical_fault)
            else:
                tasks["model"] = asyncio.create_task(self._model_loop(services_stop))
                done, _ = await asyncio.wait(
                    set(tasks.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not self._shutdown_requested.is_set():
                    by_task = {task: name for name, task in tasks.items()}
                    failed_task = next(iter(done))
                    name = by_task[failed_task]
                    exception = (
                        failed_task.exception() if not failed_task.cancelled() else None
                    )
                    detail = (
                        exception.__class__.__name__
                        if exception is not None
                        else "returned"
                    )
                    critical_fault = f"critical_service_exit:{name}:{detail}"
                    self._last_fault = critical_fault
                    self.request_stop()
                    critical_error = RuntimeError(critical_fault)
        finally:
            self.request_stop()
            if self._owned_market_ids():
                await self._close_owned()
            services_stop.set()
            if timer is not None:
                timer.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            if timer is not None:
                await asyncio.gather(timer, return_exceptions=True)
            self._stop_completed = not self._owned_market_ids()
            self.runtime_guard.mark_stopped()
            if critical_fault:
                self._last_fault = critical_fault
        if critical_error is not None:
            raise critical_error


__all__ = [
    "PolymarketAutonomousDecision",
    "PolymarketAutonomousDecisionProvider",
    "PolymarketAutonomousRuntimeSnapshot",
    "PolymarketAutonomousSupervisor",
    "PolymarketDecisionDataService",
    "PolymarketDurableControlService",
    "PolymarketExternalSignalProvider",
]
