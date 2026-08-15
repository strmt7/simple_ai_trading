"""Independent asynchronous supervisor for promoted BTC Polymarket trading."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import re
import time
from typing import Callable, Protocol

from .polymarket import PolymarketFiveMinuteMarket, PolymarketPublicClient
from .polymarket_autonomous import (
    PolymarketAutonomousLockProposal,
    PolymarketAutonomousLockResult,
    PolymarketAutonomousOpenProposal,
    PolymarketAutonomousOpenResult,
    PolymarketAutonomousReduceProposal,
    PolymarketAutonomousReduceResult,
    submit_promoted_lock,
    submit_promoted_open,
    submit_promoted_reduction,
)
from .polymarket_live import (
    PolymarketConditionAccounting,
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderLedger,
)
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion
from .polymarket_live_qualification import (
    VerifiedPolymarketLifecycleQualification,
)
from .polymarket_live_risk import PolymarketLiveRiskService, PolymarketLiveRiskState
from .polymarket_live_runtime import (
    PolymarketAuthenticatedUserStream,
    PolymarketLiveRuntimeGuard,
    PolymarketReconciliationService,
)
from .polymarket_live_settlement import PolymarketSettlementService
from .polymarket_live_stop import stop_owned_polymarket_exposure


_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")


class PolymarketAutonomousDecisionProvider(Protocol):
    """A promoted model boundary with public market data and no order authority."""

    def decide(
        self,
        *,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        observed_at_ms: int,
        portfolio: "PolymarketAutonomousPortfolio",
    ) -> "PolymarketAutonomousDecision": ...


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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousPortfolioLot:
    parent_intent_id: str
    market_id: str
    token_id: str
    outcome: str
    quantity: Decimal

    def __post_init__(self) -> None:
        parent = str(self.parent_intent_id or "").strip()
        market = str(self.market_id or "").strip().lower()
        token = str(self.token_id or "").strip()
        outcome = str(self.outcome or "").strip().title()
        quantity = Decimal(str(self.quantity))
        if (
            not parent
            or len(parent) > 128
            or _CONDITION_ID.fullmatch(market) is None
            or _TOKEN_ID.fullmatch(token) is None
            or outcome not in {"Up", "Down"}
            or not quantity.is_finite()
            or quantity <= 0
        ):
            raise ValueError("Polymarket autonomous portfolio lot is invalid")
        object.__setattr__(self, "parent_intent_id", parent)
        object.__setattr__(self, "market_id", market)
        object.__setattr__(self, "token_id", token)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "quantity", quantity)


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousPortfolio:
    condition_id: str
    lots: tuple[PolymarketAutonomousPortfolioLot, ...]
    accounting: PolymarketConditionAccounting
    risk_state: PolymarketLiveRiskState
    portfolio_sha256: str = ""

    def __post_init__(self) -> None:
        condition = str(self.condition_id or "").strip().lower()
        lots = tuple(
            sorted(
                self.lots,
                key=lambda item: (item.outcome, item.parent_intent_id),
            )
        )
        if (
            self.accounting.condition_id != condition
            or self.risk_state.condition_id != condition
            or any(
                not isinstance(item, PolymarketAutonomousPortfolioLot)
                or item.market_id != condition
                for item in lots
            )
            or len({item.parent_intent_id for item in lots}) != len(lots)
            or sum(
                (item.quantity for item in lots if item.outcome == "Up"),
                start=Decimal("0"),
            )
            != self.accounting.up_quantity
            or sum(
                (item.quantity for item in lots if item.outcome == "Down"),
                start=Decimal("0"),
            )
            != self.accounting.down_quantity
        ):
            raise ValueError("Polymarket autonomous portfolio differs")
        body = {
            "schema_version": "polymarket-autonomous-portfolio-v1",
            "condition_id": condition,
            "lots": [
                {
                    "parent_intent_id": item.parent_intent_id,
                    "market_id": item.market_id,
                    "token_id": item.token_id,
                    "outcome": item.outcome,
                    "quantity": format(item.quantity, "f"),
                }
                for item in lots
            ],
            "accounting": {
                "gross_buy_cost_quote": format(
                    self.accounting.gross_buy_cost_quote,
                    "f",
                ),
                "gross_sell_proceeds_quote": format(
                    self.accounting.gross_sell_proceeds_quote,
                    "f",
                ),
                "confirmed_redemption_payout_quote": format(
                    self.accounting.confirmed_redemption_payout_quote,
                    "f",
                ),
                "up_quantity": format(self.accounting.up_quantity, "f"),
                "down_quantity": format(self.accounting.down_quantity, "f"),
                "up_cost_basis_quote": format(
                    self.accounting.up_cost_basis_quote,
                    "f",
                ),
                "down_cost_basis_quote": format(
                    self.accounting.down_cost_basis_quote,
                    "f",
                ),
                "confirmed_fill_count": self.accounting.confirmed_fill_count,
            },
            "risk_state": self.risk_state.body(),
            "risk_state_sha256": self.risk_state.risk_state_sha256,
            "credentials_used": False,
            "account_connected": False,
            "binance_execution_connected": False,
        }
        actual = _canonical_sha256(body)
        claimed = str(self.portfolio_sha256 or "").strip().lower()
        if claimed and claimed != actual:
            raise ValueError("Polymarket autonomous portfolio hash differs")
        object.__setattr__(self, "condition_id", condition)
        object.__setattr__(self, "lots", lots)
        object.__setattr__(self, "portfolio_sha256", actual)


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousDecision:
    proposals: tuple[PolymarketAutonomousOpenProposal, ...] = ()
    locks: tuple[PolymarketAutonomousLockProposal, ...] = ()
    reductions: tuple[PolymarketAutonomousReduceProposal, ...] = ()
    close_owned_exposure: bool = False
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        proposals = tuple(self.proposals)
        if any(
            not isinstance(item, PolymarketAutonomousOpenProposal) for item in proposals
        ):
            raise TypeError("decision proposals must be Polymarket proposals")
        hashes = tuple(item.proposal_sha256 for item in proposals)
        locks = tuple(self.locks)
        if any(
            not isinstance(item, PolymarketAutonomousLockProposal) for item in locks
        ):
            raise TypeError("decision locks must be Polymarket proposals")
        reductions = tuple(self.reductions)
        if any(
            not isinstance(item, PolymarketAutonomousReduceProposal)
            for item in reductions
        ):
            raise TypeError("decision reductions must be Polymarket proposals")
        hashes += tuple(item.proposal_sha256 for item in locks)
        hashes += tuple(item.proposal_sha256 for item in reductions)
        if len(set(hashes)) != len(hashes):
            raise ValueError("decision contains duplicate Polymarket proposals")
        action_groups = sum(bool(group) for group in (proposals, locks, reductions))
        if action_groups > 1:
            raise ValueError("decision cannot mix Polymarket transition types")
        if self.close_owned_exposure and (proposals or locks or reductions):
            raise ValueError(
                "decision cannot open and close Polymarket exposure together"
            )
        reasons = tuple(str(item or "").strip() for item in self.reasons)
        if any(not item or len(item) > 160 for item in reasons):
            raise ValueError("decision reason is invalid")
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "locks", locks)
        object.__setattr__(self, "reductions", reductions)
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
    binance_execution_connected: bool
    submitted_reductions: int = 0
    blocked_reductions: int = 0
    submitted_locks: int = 0
    blocked_locks: int = 0
    risk_state_sha256: str = ""
    daily_realized_pnl_quote: str = "0"
    drawdown_capital_fraction: str = "0"
    cooldown_until_ms: int = 0
    entry_allowed: bool = False
    wallet_capital_support_quote: str = "0"
    wallet_capital_gate_passed: bool = False


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
        lifecycle_qualification: VerifiedPolymarketLifecycleQualification,
        decision_provider: PolymarketAutonomousDecisionProvider,
        risk_capital_quote: Decimal,
        risk_level: str,
        settlement: PolymarketSettlementService | None = None,
        decision_data_service: PolymarketDecisionDataService | None = None,
        durable_control_service: PolymarketDurableControlService | None = None,
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
        if not isinstance(
            lifecycle_qualification,
            VerifiedPolymarketLifecycleQualification,
        ):
            raise PolymarketLiveBlocked(
                "autonomous Polymarket runtime requires verified authenticated "
                "lifecycle evidence"
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
        self.public_client = public_client
        self.coordinator = coordinator
        self.ledger = ledger
        self.runtime_guard = runtime_guard
        self.user_stream = user_stream
        self.reconciliation = reconciliation
        self.promotion = promotion
        self.lifecycle_qualification = lifecycle_qualification
        self.decision_provider = decision_provider
        self.live_risk = PolymarketLiveRiskService(
            ledger,
            risk_capital_quote=risk_capital_quote,
            risk_profile=risk_level,
        )
        self.settlement = settlement
        self.decision_data_service = decision_data_service
        self.durable_control_service = durable_control_service
        self.decision_interval_seconds = interval
        self.decision_timeout_seconds = timeout
        self.forced_exit_seconds = forced_exit
        self.stop_timeout_seconds = stop_timeout
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self._shutdown_requested = asyncio.Event()
        self._paused = False
        self._stop_completed = False
        self._markets: tuple[PolymarketFiveMinuteMarket, ...] = ()
        self._discovered_active_condition_id = ""
        self._decisions = 0
        self._submitted_opens = 0
        self._blocked_opens = 0
        self._submitted_reductions = 0
        self._blocked_reductions = 0
        self._submitted_locks = 0
        self._blocked_locks = 0
        self._requested_closes = 0
        self._completed_closes = 0
        self._last_fault = ""
        self._consumed_proposals: set[str] = set()
        self._pending_decision: asyncio.Task[PolymarketAutonomousDecision] | None = None
        self._last_risk_state: PolymarketLiveRiskState | None = None
        self._wallet_capital_support_quote = Decimal("0")
        self._wallet_capital_gate_passed = False

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
        risk_state = self._last_risk_state
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
            submitted_reductions=self._submitted_reductions,
            blocked_reductions=self._blocked_reductions,
            submitted_locks=self._submitted_locks,
            blocked_locks=self._blocked_locks,
            requested_closes=self._requested_closes,
            completed_closes=self._completed_closes,
            last_fault=self._last_fault,
            binance_execution_connected=False,
            risk_state_sha256=(
                "" if risk_state is None else risk_state.risk_state_sha256
            ),
            daily_realized_pnl_quote=(
                "0"
                if risk_state is None
                else format(risk_state.daily_realized_pnl_quote, "f")
            ),
            drawdown_capital_fraction=(
                "0"
                if risk_state is None
                else format(risk_state.drawdown_capital_fraction, "f")
            ),
            cooldown_until_ms=(
                0 if risk_state is None else risk_state.cooldown_until_ms
            ),
            entry_allowed=(
                False
                if risk_state is None
                else risk_state.entry_allowed and self._wallet_capital_gate_passed
            ),
            wallet_capital_support_quote=format(
                self._wallet_capital_support_quote,
                "f",
            ),
            wallet_capital_gate_passed=self._wallet_capital_gate_passed,
        )

    def _owned_market_ids(self) -> set[str]:
        markets = {item.market_id for item in self.ledger.owned_inventory()}
        markets.update(
            record.intent.market_id
            for record in self.ledger.records()
            if record.blocks_new_exposure
        )
        return markets

    def _portfolio(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        observed_at_ms: int,
    ) -> PolymarketAutonomousPortfolio:
        if self.ledger.open_owned_order_ids():
            raise PolymarketLiveBlocked(
                "Polymarket portfolio has an active or unknown transition"
            )
        records = {record.intent.intent_id: record for record in self.ledger.records()}
        lots: list[PolymarketAutonomousPortfolioLot] = []
        for lot in self.ledger.owned_lots():
            if lot.market_id != market.condition_id:
                continue
            parent = records.get(lot.parent_intent_id)
            if (
                parent is None
                or parent.intent.side != "BUY"
                or parent.intent.market_id != market.condition_id
                or parent.intent.token_id != lot.token_id
                or lot.provisional
                or lot.reserved_close_quantity
            ):
                raise PolymarketLiveBlocked(
                    "Polymarket portfolio contains an unsafe owned lot"
                )
            lots.append(
                PolymarketAutonomousPortfolioLot(
                    parent_intent_id=lot.parent_intent_id,
                    market_id=lot.market_id,
                    token_id=lot.token_id,
                    outcome=parent.intent.outcome,
                    quantity=lot.quantity,
                )
            )
        risk_capture = self.live_risk.capture(
            market.condition_id,
            observed_at_ms=observed_at_ms,
        )
        self._last_risk_state = risk_capture.state
        return PolymarketAutonomousPortfolio(
            condition_id=market.condition_id,
            lots=tuple(lots),
            accounting=risk_capture.condition_accounting,
            risk_state=risk_capture.state,
        )

    async def _discover_and_subscribe(
        self,
        *,
        observed_at_ms: int,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        async def synchronize_subscriptions(
            markets: tuple[PolymarketFiveMinuteMarket, ...],
        ) -> None:
            desired = {market.condition_id for market in markets}
            desired.update(self._owned_market_ids())
            current = set(self.user_stream.markets)
            added = tuple(sorted(desired - current))
            removed = tuple(sorted(current - desired))
            if added:
                await self.user_stream.subscribe_markets(added)
            if removed:
                await self.user_stream.unsubscribe_markets(removed)

        market_variant = self.promotion.promotion.market_variant
        cached = tuple(
            market
            for market in self._markets
            if market.event_start_ms <= observed_at_ms < market.end_ms
        )
        if (
            len(cached) == 1
            and cached[0].condition_id == self._discovered_active_condition_id
        ):
            await synchronize_subscriptions(self._markets)
            return cached
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
        self._discovered_active_condition_id = active[0].condition_id
        await synchronize_subscriptions(markets)
        return active

    def _assert_proposal_market(
        self,
        proposal: PolymarketAutonomousOpenProposal
        | PolymarketAutonomousLockProposal
        | PolymarketAutonomousReduceProposal,
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
            portfolio = self._portfolio(
                markets[0],
                observed_at_ms=observed_at_ms,
            )
            self._pending_decision = asyncio.create_task(
                asyncio.to_thread(
                    self.decision_provider.decide,
                    markets=markets,
                    observed_at_ms=observed_at_ms,
                    portfolio=portfolio,
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

    async def _close_owned(self) -> bool:
        self._requested_closes += 1
        try:
            payload, status = await asyncio.to_thread(
                stop_owned_polymarket_exposure,
                coordinator=self.coordinator,
                ledger=self.ledger,
                timeout_seconds=self.stop_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_fault = f"owned_exposure_close_failure:{exc.__class__.__name__}"
            return False
        completed = status == 0 and payload["completed"] is True
        if completed:
            self._completed_closes += 1
            self._last_fault = ""
        else:
            self._last_fault = "owned_exposure_close_incomplete:" + str(
                payload["reason"]
            )
        return completed

    async def _drain_owned_exposure(self) -> None:
        """Stay alive in close-only recovery until the owned ledger is flat."""

        retry_delay = max(0.25, self.decision_interval_seconds)
        while self._owned_market_ids():
            await self._close_owned()
            if self._owned_market_ids():
                await asyncio.sleep(retry_delay)
        self._stop_completed = True

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
        for proposal in decision.locks:
            if self._shutdown_requested.is_set() or self._paused:
                return
            if proposal.proposal_sha256 in self._consumed_proposals:
                continue
            try:
                self._assert_proposal_market(proposal, market)
                if not self._market_has_owned_exposure(proposal.market_id):
                    raise PolymarketLiveBlocked(
                        "bot has no owned exposure to lock in this Polymarket event"
                    )
                result = await asyncio.to_thread(
                    submit_promoted_lock,
                    proposal,
                    self.promotion,
                    self.coordinator,
                    lifecycle_qualification=self.lifecycle_qualification,
                    clock_ms=self._clock_ms,
                )
            except PolymarketLiveBlocked as exc:
                self._blocked_locks += 1
                self._last_fault = str(exc)
                continue
            if not isinstance(result, PolymarketAutonomousLockResult):
                raise TypeError("autonomous lock returned an invalid result")
            self._consumed_proposals.add(proposal.proposal_sha256)
            self._submitted_locks += 1
            self._last_fault = ""
        for proposal in decision.reductions:
            if self._shutdown_requested.is_set() or self._paused:
                return
            if proposal.proposal_sha256 in self._consumed_proposals:
                continue
            try:
                self._assert_proposal_market(proposal, market)
                if not self._market_has_owned_exposure(proposal.market_id):
                    raise PolymarketLiveBlocked(
                        "bot has no owned exposure to reduce in this Polymarket event"
                    )
                result = await asyncio.to_thread(
                    submit_promoted_reduction,
                    proposal,
                    self.promotion,
                    self.coordinator,
                    lifecycle_qualification=self.lifecycle_qualification,
                    clock_ms=self._clock_ms,
                )
            except PolymarketLiveBlocked as exc:
                self._blocked_reductions += 1
                self._last_fault = str(exc)
                continue
            if not isinstance(result, PolymarketAutonomousReduceResult):
                raise TypeError("autonomous reduction returned an invalid result")
            self._consumed_proposals.add(proposal.proposal_sha256)
            self._submitted_reductions += 1
            self._last_fault = ""
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
                fresh_portfolio = self._portfolio(
                    market,
                    observed_at_ms=observed_at_ms,
                )
                if (
                    not self._wallet_capital_gate_passed
                    or not fresh_portfolio.risk_state.entry_allowed
                    or proposal.risk_state_sha256
                    != fresh_portfolio.risk_state.risk_state_sha256
                    or proposal.maximum_projected_inventory_downside_quote
                    > fresh_portfolio.risk_state.maximum_current_condition_downside_quote
                ):
                    raise PolymarketLiveBlocked(
                        "Polymarket open proposal differs from fresh live risk state"
                    )
                result = await asyncio.to_thread(
                    submit_promoted_open,
                    proposal,
                    self.promotion,
                    self.coordinator,
                    lifecycle_qualification=self.lifecycle_qualification,
                    external_signal=None,
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
        if self._shutdown_requested.is_set():
            if self._owned_market_ids():
                await self._drain_owned_exposure()
            self._stop_completed = not self._owned_market_ids()
            self.runtime_guard.mark_stopped()
            return
        try:
            initial = await asyncio.to_thread(self.coordinator.preflight)
            if not initial.can_close:
                raise PolymarketLiveBlocked(
                    f"autonomous supervision preflight failed: {initial.errors}"
                )
            collateral_balance = await asyncio.to_thread(
                self.coordinator.venue.collateral_balance
            )
            collateral = Decimal(str(collateral_balance))
            if not collateral.is_finite() or collateral < 0:
                raise PolymarketLiveBlocked(
                    "Polymarket venue returned an invalid collateral balance"
                )
            owned_cost_basis = Decimal("0")
            if not self.ledger.open_owned_order_ids():
                for market_id in self._owned_market_ids():
                    accounting = self.ledger.condition_accounting(market_id)
                    owned_cost_basis += (
                        accounting.up_cost_basis_quote
                        + accounting.down_cost_basis_quote
                    )
            self._wallet_capital_support_quote = collateral + owned_cost_basis
            self._wallet_capital_gate_passed = (
                self.live_risk.risk_capital_quote <= self._wallet_capital_support_quote
            )
            await self._discover_and_subscribe(observed_at_ms=self._clock_ms())
        except Exception as exc:
            self._last_fault = f"startup_failure:{exc.__class__.__name__}"
            if self._owned_market_ids():
                self.request_stop()
                await self._drain_owned_exposure()
            self.runtime_guard.mark_stopped()
            raise
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
                    pause_runtime=self.pause,
                    resume_runtime=self.resume,
                )
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
                await self._drain_owned_exposure()
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
    "PolymarketAutonomousPortfolio",
    "PolymarketAutonomousPortfolioLot",
    "PolymarketAutonomousRuntimeSnapshot",
    "PolymarketAutonomousSupervisor",
    "PolymarketDecisionDataService",
    "PolymarketDurableControlService",
]
