from __future__ import annotations

import ast
import asyncio
from decimal import Decimal
import inspect
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from polymarket_live_support import build_polymarket_live_promotion_fixture
import simple_ai_trading.polymarket_autonomous_runtime as runtime_module
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFifteenMinuteMarket,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_autonomous import (
    PolymarketAutonomousLockProposal,
    PolymarketAutonomousOpenProposal,
    PolymarketAutonomousReduceProposal,
)
from simple_ai_trading.polymarket_autonomous_runtime import (
    PolymarketAutonomousDecision,
    PolymarketAutonomousSupervisor,
)
from simple_ai_trading.polymarket_live import (
    PolymarketLedgerRevision,
    PolymarketLiveBlocked,
    PolymarketLiveOrderLedger,
)
from simple_ai_trading.polymarket_live_risk import PolymarketLiveRiskState
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
    load_polymarket_live_promotion,
)
from simple_ai_trading.polymarket_live_qualification import (
    VerifiedPolymarketLifecycleQualification,
)


EVENT_START_MS = 1_800_000_000_000
EVENT_END_MS = EVENT_START_MS + 300_000
NOW_MS = EVENT_START_MS + 120_000
MARKET_ID = "0x" + "1" * 64
NEXT_MARKET_ID = "0x" + "2" * 64
OWNED_MARKET_ID = "0x" + "3" * 64
TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _promotion(
    root: Path,
    *,
    market_variant: str = "fiveminute",
) -> VerifiedPolymarketLivePromotion:
    payload = build_polymarket_live_promotion_fixture(
        root,
        now_ms=NOW_MS,
        market_variant=market_variant,
        created_at_ms=NOW_MS - 86_400_000,
    )
    promotion_path = root / "promotion.json"
    promotion_path.write_text(_canonical(payload), encoding="ascii")
    return load_polymarket_live_promotion(
        promotion_path,
        evidence_root=root,
        require_live_authority=True,
        observed_at_ms=NOW_MS,
    )


def _market(
    *,
    condition_id: str = MARKET_ID,
    start_ms: int = EVENT_START_MS,
    horizon_minutes: int = 5,
) -> PolymarketFiveMinuteMarket:
    market_type = (
        PolymarketFiveMinuteMarket
        if horizon_minutes == 5
        else PolymarketFifteenMinuteMarket
    )
    return market_type(
        asset="BTC",
        market_id="123",
        condition_id=condition_id,
        slug=(
            f"btc-updown-5m-{start_ms // 1000}"
            if horizon_minutes == 5
            else f"btc-updown-15m-{start_ms // 1000}"
        ),
        question="Bitcoin Up or Down",
        event_start_ms=start_ms,
        end_ms=start_ms + horizon_minutes * 60_000,
        up_token_id=TOKEN_ID,
        down_token_id=DOWN_TOKEN_ID,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.25"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("10000"),
        volume_quote=Decimal("100000"),
        resolution_source="https://www.binance.com/en/price/bitcoin",
        gamma_payload_sha256="4" * 64,
        gamma_payload_json="{}",
    )


def _proposal(
    promotion: VerifiedPolymarketLivePromotion,
) -> PolymarketAutonomousOpenProposal:
    horizon_minutes = 5 if promotion.promotion.market_variant == "fiveminute" else 15
    return PolymarketAutonomousOpenProposal(
        proposal_id="runtime-test-proposal",
        input_sha256="5" * 64,
        model_artifact_sha256=promotion.promotion.model_artifact.sha256,
        promotion_sha256=promotion.promotion.promotion_sha256,
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        market_variant=promotion.promotion.market_variant,
        outcome="Up",
        selected_outcome_probability=Decimal("0.7"),
        requested_quantity=Decimal("5"),
        risk_state_sha256=_empty_risk_state().risk_state_sha256,
        maximum_projected_inventory_downside_quote=Decimal("10"),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_START_MS + horizon_minutes * 60_000,
        decision_time_ms=NOW_MS - 100,
        expires_at_ms=NOW_MS + 900,
    )


def _empty_risk_state() -> PolymarketLiveRiskState:
    return PolymarketLiveRiskState(
        condition_id=MARKET_ID,
        risk_profile="conservative",
        risk_capital_quote=Decimal("10000"),
        observed_at_ms=NOW_MS,
        utc_day_index=NOW_MS // 86_400_000,
        ledger_revision=PolymarketLedgerRevision(0, "0" * 64),
        realized_event_count=0,
        realized_condition_count=0,
        daily_realized_pnl_quote=Decimal("0"),
        lifetime_realized_pnl_quote=Decimal("0"),
        settled_equity_quote=Decimal("10000"),
        settled_peak_equity_quote=Decimal("10000"),
        drawdown_capital_fraction=Decimal("0"),
        consecutive_losing_conditions=0,
        cooldown_until_ms=0,
        cooldown_active=False,
        current_condition_inventory_downside_quote=Decimal("0"),
        other_condition_inventory_downside_quote=Decimal("0"),
        total_inventory_downside_quote=Decimal("0"),
        maximum_current_condition_downside_quote=Decimal("10"),
        entry_allowed=True,
        entry_block_reasons=(),
    )


def _reduction(
    promotion: VerifiedPolymarketLivePromotion,
) -> PolymarketAutonomousReduceProposal:
    return PolymarketAutonomousReduceProposal(
        proposal_id="runtime-reduction-proposal",
        input_sha256="6" * 64,
        model_artifact_sha256=promotion.promotion.model_artifact.sha256,
        promotion_sha256=promotion.promotion.promotion_sha256,
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        market_variant=promotion.promotion.market_variant,
        outcome="Up",
        parent_intent_id="poly-open-parent-0001",
        maximum_outcome_probability=Decimal("0.6"),
        requested_quantity=Decimal("5"),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_END_MS,
        decision_time_ms=NOW_MS - 100,
        expires_at_ms=NOW_MS + 900,
    )


def _lock(
    promotion: VerifiedPolymarketLivePromotion,
) -> PolymarketAutonomousLockProposal:
    return PolymarketAutonomousLockProposal(
        proposal_id="runtime-lock-proposal",
        input_sha256="7" * 64,
        model_artifact_sha256=promotion.promotion.model_artifact.sha256,
        promotion_sha256=promotion.promotion.promotion_sha256,
        market_id=MARKET_ID,
        token_id=DOWN_TOKEN_ID,
        symbol="BTC",
        market_variant=promotion.promotion.market_variant,
        outcome="Down",
        owned_outcome="Up",
        requested_quantity=Decimal("5"),
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_END_MS,
        decision_time_ms=NOW_MS - 100,
        expires_at_ms=NOW_MS + 900,
    )


class _Stream:
    def __init__(self, ledger: object, guard: object) -> None:
        self.consumer = SimpleNamespace(ledger=ledger, runtime_guard=guard)
        self._markets: set[str] = set()

    @property
    def markets(self) -> tuple[str, ...]:
        return tuple(sorted(self._markets))

    async def subscribe_markets(self, markets: tuple[str, ...]) -> tuple[str, ...]:
        changed = tuple(sorted(set(markets) - self._markets))
        self._markets.update(markets)
        return changed

    async def unsubscribe_markets(self, markets: tuple[str, ...]) -> tuple[str, ...]:
        changed = tuple(sorted(set(markets) & self._markets))
        self._markets.difference_update(markets)
        return changed

    async def run(self, stop: asyncio.Event) -> None:
        await stop.wait()


def _supervisor(
    tmp_path: Path,
    *,
    provider: object,
    decision_data: object | None = None,
    markets: tuple[PolymarketFiveMinuteMarket, ...] | None = None,
    market_variant: str = "fiveminute",
    wallet_balance: Decimal = Decimal("10000"),
) -> PolymarketAutonomousSupervisor:
    promotion = _promotion(
        tmp_path / "promotion",
        market_variant=market_variant,
    )
    ledger = PolymarketLiveOrderLedger(tmp_path / "ledger.sqlite3")
    guard = SimpleNamespace(mark_stopped=lambda: None)
    coordinator = SimpleNamespace(
        ledger=ledger,
        runtime_authority=guard,
        venue=SimpleNamespace(collateral_balance=lambda: wallet_balance),
        preflight=lambda: SimpleNamespace(can_close=True, errors=()),
    )
    reconciliation = SimpleNamespace(
        coordinator=coordinator,
        runtime_guard=guard,
        run=lambda stop: stop.wait(),
    )
    stream = _Stream(ledger, guard)
    selected = markets or (
        _market(),
        _market(
            condition_id=NEXT_MARKET_ID,
            start_ms=EVENT_END_MS,
        ),
    )
    client = SimpleNamespace(
        discover_five_minute_markets=lambda **_kwargs: selected,
        discover_fifteen_minute_markets=lambda **_kwargs: selected,
    )
    return PolymarketAutonomousSupervisor(
        public_client=client,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
        ledger=ledger,
        runtime_guard=guard,  # type: ignore[arg-type]
        user_stream=stream,  # type: ignore[arg-type]
        reconciliation=reconciliation,  # type: ignore[arg-type]
        promotion=promotion,
        lifecycle_qualification=Mock(spec=VerifiedPolymarketLifecycleQualification),
        decision_provider=provider,  # type: ignore[arg-type]
        risk_capital_quote=Decimal("10000"),
        risk_level="conservative",
        decision_data_service=decision_data,  # type: ignore[arg-type]
        decision_interval_seconds=0.25,
        decision_timeout_seconds=0.1,
        clock_ms=lambda: NOW_MS,
    )


def test_runtime_imports_no_binance_execution_module() -> None:
    tree = ast.parse(inspect.getsource(runtime_module))
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imports.extend(
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert all("binance" not in name.lower() for name in imports)


def test_decision_rejects_duplicate_proposals(tmp_path: Path) -> None:
    proposal = _proposal(_promotion(tmp_path))
    reduction = _reduction(_promotion(tmp_path / "reduce"))
    with pytest.raises(ValueError, match="duplicate"):
        PolymarketAutonomousDecision(proposals=(proposal, proposal))
    with pytest.raises(ValueError, match="open and close"):
        PolymarketAutonomousDecision(
            proposals=(proposal,),
            close_owned_exposure=True,
        )
    with pytest.raises(ValueError, match="mix"):
        PolymarketAutonomousDecision(
            proposals=(proposal,),
            reductions=(reduction,),
        )


def test_discovery_subscribes_current_next_and_owned_market(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        supervisor = _supervisor(
            tmp_path,
            provider=SimpleNamespace(decide=lambda **_kwargs: None),
        )
        supervisor._owned_market_ids = lambda: {  # type: ignore[method-assign]
            OWNED_MARKET_ID
        }

        active = await supervisor._discover_and_subscribe(
            observed_at_ms=NOW_MS,
        )

        assert tuple(market.condition_id for market in active) == (MARKET_ID,)
        assert supervisor.snapshot().subscribed_market_ids == (
            MARKET_ID,
            NEXT_MARKET_ID,
            OWNED_MARKET_ID,
        )
        assert supervisor.snapshot().binance_execution_connected is False

    asyncio.run(run())


def test_discovery_reuses_current_identity_but_refreshes_at_rollover(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    markets = (
        _market(),
        _market(condition_id=NEXT_MARKET_ID, start_ms=EVENT_END_MS),
    )
    discover = Mock(return_value=markets)
    supervisor.public_client.discover_five_minute_markets = discover

    async def run() -> None:
        first = await supervisor._discover_and_subscribe(observed_at_ms=NOW_MS)
        cached = await supervisor._discover_and_subscribe(observed_at_ms=NOW_MS + 1_000)
        rolled = await supervisor._discover_and_subscribe(
            observed_at_ms=EVENT_END_MS + 1
        )

        assert first == cached == (markets[0],)
        assert rolled == (markets[1],)

    asyncio.run(run())
    assert discover.call_count == 2


def test_verified_fifteen_minute_promotion_selects_only_fifteen_minute_discovery(
    tmp_path: Path,
) -> None:
    current = _market(horizon_minutes=15)
    following = _market(
        condition_id=NEXT_MARKET_ID,
        start_ms=EVENT_START_MS + 900_000,
        horizon_minutes=15,
    )
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
        markets=(current, following),
        market_variant="fifteenminute",
    )

    async def run() -> None:
        active = await supervisor._discover_and_subscribe(
            observed_at_ms=NOW_MS,
        )
        assert active == (current,)

    asyncio.run(run())

    snapshot = supervisor.snapshot()
    assert snapshot.market_variant == "fifteenminute"
    assert snapshot.horizon_minutes == 15
    assert snapshot.discovered_market_ids == (MARKET_ID, NEXT_MARKET_ID)


def test_pause_or_stop_prevents_a_late_model_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion = _promotion(tmp_path / "proposal")
    proposal = _proposal(promotion)
    supervisor = _supervisor(
        tmp_path / "runtime",
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    submitted = False

    def submit(*_args: object, **_kwargs: object) -> object:
        nonlocal submitted
        submitted = True
        return object()

    monkeypatch.setattr(runtime_module, "submit_promoted_open", submit)

    async def run() -> None:
        supervisor.pause()
        await supervisor._apply_decision(
            PolymarketAutonomousDecision(proposals=(proposal,)),
            _market(),
            observed_at_ms=NOW_MS,
        )
        supervisor.resume()
        supervisor.request_stop()
        await supervisor._apply_decision(
            PolymarketAutonomousDecision(proposals=(proposal,)),
            _market(),
            observed_at_ms=NOW_MS,
        )

    asyncio.run(run())

    assert submitted is False


def test_pause_still_allows_a_model_requested_close(tmp_path: Path) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    closed = False
    supervisor._owned_market_ids = lambda: {MARKET_ID}  # type: ignore[method-assign]

    async def close_owned() -> bool:
        nonlocal closed
        closed = True
        return True

    supervisor._close_owned = close_owned  # type: ignore[method-assign]
    supervisor.pause()
    asyncio.run(
        supervisor._apply_decision(
            PolymarketAutonomousDecision(close_owned_exposure=True),
            _market(),
            observed_at_ms=NOW_MS,
        )
    )

    assert closed is True


def test_parent_bound_reduction_is_dispatched_without_close_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion = _promotion(tmp_path / "proposal")
    reduction = _reduction(promotion)
    supervisor = _supervisor(
        tmp_path / "runtime",
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    supervisor._owned_market_ids = lambda: {MARKET_ID}  # type: ignore[method-assign]
    submitted: list[PolymarketAutonomousReduceProposal] = []

    class Result:
        pass

    def submit(
        proposal: PolymarketAutonomousReduceProposal,
        *_args: object,
        **_kwargs: object,
    ) -> Result:
        submitted.append(proposal)
        return Result()

    monkeypatch.setattr(runtime_module, "PolymarketAutonomousReduceResult", Result)
    monkeypatch.setattr(runtime_module, "submit_promoted_reduction", submit)

    asyncio.run(
        supervisor._apply_decision(
            PolymarketAutonomousDecision(reductions=(reduction,)),
            _market(),
            observed_at_ms=NOW_MS,
        )
    )

    assert submitted == [reduction]
    assert supervisor.snapshot().submitted_reductions == 1
    assert supervisor.snapshot().requested_closes == 0


def test_risk_reducing_lock_is_dispatched_without_binance_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion = _promotion(tmp_path / "proposal")
    lock = _lock(promotion)
    supervisor = _supervisor(
        tmp_path / "runtime",
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    supervisor._owned_market_ids = lambda: {MARKET_ID}  # type: ignore[method-assign]
    submitted: list[PolymarketAutonomousLockProposal] = []

    class Result:
        pass

    def submit(
        proposal: PolymarketAutonomousLockProposal,
        *_args: object,
        **_kwargs: object,
    ) -> Result:
        submitted.append(proposal)
        return Result()

    monkeypatch.setattr(runtime_module, "PolymarketAutonomousLockResult", Result)
    monkeypatch.setattr(runtime_module, "submit_promoted_lock", submit)

    asyncio.run(
        supervisor._apply_decision(
            PolymarketAutonomousDecision(locks=(lock,)),
            _market(),
            observed_at_ms=NOW_MS,
        )
    )

    snapshot = supervisor.snapshot()
    assert submitted == [lock]
    assert snapshot.submitted_locks == 1
    assert snapshot.binance_execution_connected is False


def test_timed_out_model_is_not_started_twice(tmp_path: Path) -> None:
    calls = 0
    release = Event()

    class Provider:
        def decide(self, **_kwargs: object) -> PolymarketAutonomousDecision:
            nonlocal calls
            calls += 1
            release.wait(timeout=2)
            return PolymarketAutonomousDecision()

    async def run() -> tuple[object, object]:
        supervisor = _supervisor(tmp_path, provider=Provider())
        first = await supervisor._decision((_market(),), observed_at_ms=NOW_MS)
        second = await supervisor._decision((_market(),), observed_at_ms=NOW_MS)
        release.set()
        await asyncio.sleep(0.05)
        return first, second

    first, second = asyncio.run(run())

    assert first is None
    assert second is None
    assert calls == 1


def test_stop_retries_until_owned_exposure_is_closed(tmp_path: Path) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    owned = {MARKET_ID}
    attempts = 0
    supervisor._owned_market_ids = lambda: set(owned)  # type: ignore[method-assign]

    async def close_owned() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            owned.clear()
            return True
        return False

    supervisor._close_owned = close_owned  # type: ignore[method-assign]
    supervisor.request_stop()

    async def run() -> None:
        services_stop = asyncio.Event()
        await asyncio.wait_for(
            supervisor._model_loop(services_stop),
            timeout=2,
        )
        assert services_stop.is_set()

    asyncio.run(run())

    assert attempts == 2
    assert supervisor.snapshot().stop_completed is True


def test_close_failure_is_retained_as_retriable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )

    def fail_close(**_kwargs: object) -> object:
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(
        runtime_module,
        "stop_owned_polymarket_exposure",
        fail_close,
    )

    assert asyncio.run(supervisor._close_owned()) is False
    assert supervisor.snapshot().last_fault == (
        "owned_exposure_close_failure:ConnectionError"
    )


def test_critical_service_failure_drains_owned_exposure_before_exit(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    owned = {MARKET_ID}
    attempts = 0
    supervisor._owned_market_ids = lambda: set(owned)  # type: ignore[method-assign]

    async def close_owned() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            owned.clear()
            return True
        return False

    async def failed_stream(_stop: asyncio.Event) -> None:
        raise ConnectionError("authenticated stream failed")

    supervisor._close_owned = close_owned  # type: ignore[method-assign]
    supervisor.user_stream.run = failed_stream  # type: ignore[method-assign]

    with pytest.raises(
        RuntimeError,
        match=("critical_service_exit:authenticated_user_stream:ConnectionError"),
    ):
        asyncio.run(supervisor.run())

    assert attempts == 2
    assert supervisor.snapshot().stop_completed is True


def test_startup_network_failure_drains_owned_exposure_before_exit(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
    )
    owned = {MARKET_ID}
    attempts = 0
    supervisor._owned_market_ids = lambda: set(owned)  # type: ignore[method-assign]
    supervisor.coordinator.preflight = Mock(  # type: ignore[method-assign]
        side_effect=ConnectionError("startup network unavailable")
    )

    async def close_owned() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            owned.clear()
            return True
        return False

    supervisor._close_owned = close_owned  # type: ignore[method-assign]

    with pytest.raises(ConnectionError, match="startup network unavailable"):
        asyncio.run(supervisor.run())

    assert attempts == 2
    assert supervisor.snapshot().stop_completed is True
    assert supervisor.snapshot().stop_requested is True
    assert supervisor.snapshot().last_fault == "startup_failure:ConnectionError"


def test_forced_exit_window_never_opens_new_exposure(tmp_path: Path) -> None:
    calls = 0

    class Provider:
        def decide(self, **_kwargs: object) -> PolymarketAutonomousDecision:
            nonlocal calls
            calls += 1
            return PolymarketAutonomousDecision()

    supervisor = _supervisor(tmp_path, provider=Provider())
    supervisor._clock_ms = lambda: EVENT_END_MS - 10_000

    async def run() -> None:
        services_stop = asyncio.Event()
        task = asyncio.create_task(supervisor._model_loop(services_stop))
        await asyncio.sleep(0.05)
        services_stop.set()
        await task

    asyncio.run(run())

    assert calls == 0


def test_run_with_pre_requested_stop_never_invokes_model(tmp_path: Path) -> None:
    calls = 0

    class Provider:
        def decide(self, **_kwargs: object) -> PolymarketAutonomousDecision:
            nonlocal calls
            calls += 1
            return PolymarketAutonomousDecision()

    supervisor = _supervisor(tmp_path, provider=Provider())
    supervisor.request_stop()
    asyncio.run(supervisor.run())

    assert calls == 0
    assert supervisor.snapshot().stop_completed is True
    assert supervisor.snapshot().stop_requested is True
    assert supervisor.snapshot().wallet_capital_gate_passed is False
    assert supervisor.snapshot().wallet_capital_support_quote == "0"


def test_wallet_capital_gate_fails_closed_without_blocking_supervision(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
        wallet_balance=Decimal("9999.99"),
    )
    asyncio.run(supervisor.run(duration_seconds=0.01))

    snapshot = supervisor.snapshot()
    assert snapshot.stop_completed is True
    assert snapshot.wallet_capital_gate_passed is False
    assert snapshot.wallet_capital_support_quote == "9999.99"
    assert snapshot.entry_allowed is False


def test_run_supervises_non_authoritative_predictor_data_service(
    tmp_path: Path,
) -> None:
    started = False
    stopped = False

    class PredictorData:
        trading_authority = False

        async def run(self, stop: asyncio.Event) -> None:
            nonlocal started, stopped
            started = True
            await stop.wait()
            stopped = True

    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
        decision_data=PredictorData(),
    )
    asyncio.run(supervisor.run(duration_seconds=0.01))

    assert started is True
    assert stopped is True
    assert supervisor.snapshot().binance_execution_connected is False


def test_predictor_data_service_with_authority_is_rejected(
    tmp_path: Path,
) -> None:
    class PredictorData:
        trading_authority = True

        async def run(self, stop: asyncio.Event) -> None:
            await stop.wait()

    with pytest.raises(PolymarketLiveBlocked, match="no trading authority"):
        _supervisor(
            tmp_path,
            provider=SimpleNamespace(decide=lambda **_kwargs: None),
            decision_data=PredictorData(),
        )


def test_unexpected_predictor_data_exit_stops_before_model_decision(
    tmp_path: Path,
) -> None:
    calls = 0

    class Provider:
        def decide(self, **_kwargs: object) -> PolymarketAutonomousDecision:
            nonlocal calls
            calls += 1
            return PolymarketAutonomousDecision()

    class PredictorData:
        trading_authority = False

        async def run(self, _stop: asyncio.Event) -> None:
            raise ConnectionError("predictor feed failed")

    supervisor = _supervisor(
        tmp_path,
        provider=Provider(),
        decision_data=PredictorData(),
    )

    with pytest.raises(
        RuntimeError,
        match="critical_service_exit:predictor_market_data:ConnectionError",
    ):
        asyncio.run(supervisor.run())

    assert calls == 0
    assert supervisor.snapshot().stop_requested is True
    assert supervisor.snapshot().stop_completed is True


def test_unexpected_safety_service_exit_stops_before_model_decision(
    tmp_path: Path,
) -> None:
    calls = 0

    class Provider:
        def decide(self, **_kwargs: object) -> PolymarketAutonomousDecision:
            nonlocal calls
            calls += 1
            return PolymarketAutonomousDecision()

    supervisor = _supervisor(tmp_path, provider=Provider())

    async def failed_stream(_stop: asyncio.Event) -> None:
        raise ConnectionError("authenticated stream failed")

    supervisor.user_stream.run = failed_stream  # type: ignore[method-assign]

    with pytest.raises(
        RuntimeError,
        match=("critical_service_exit:authenticated_user_stream:ConnectionError"),
    ):
        asyncio.run(supervisor.run())

    assert calls == 0
    assert supervisor.snapshot().stop_requested is True
    assert supervisor.snapshot().stop_completed is True
    assert supervisor.snapshot().last_fault == (
        "critical_service_exit:authenticated_user_stream:ConnectionError"
    )
