from __future__ import annotations

import ast
import asyncio
from decimal import Decimal
import hashlib
import inspect
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_autonomous_runtime as runtime_module
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFifteenMinuteMarket,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_autonomous import (
    PolymarketAutonomousOpenProposal,
    PolymarketAutonomousOpenResult,
)
from simple_ai_trading.polymarket_autonomous_runtime import (
    PolymarketAutonomousDecision,
    PolymarketAutonomousSupervisor,
)
from simple_ai_trading.polymarket_external_signal import (
    PolymarketBtcReferenceFeatures,
    PolymarketExternalSignalDecision,
)
from simple_ai_trading.polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveOrderLedger,
)
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
    load_polymarket_live_promotion,
)


EVENT_START_MS = 1_800_000_000_000
EVENT_END_MS = EVENT_START_MS + 300_000
NOW_MS = EVENT_START_MS + 120_000
MARKET_ID = "0x" + "1" * 64
NEXT_MARKET_ID = "0x" + "2" * 64
OWNED_MARKET_ID = "0x" + "3" * 64
TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40
GATES = {
    "prospective_untouched_test": True,
    "source_integrity": True,
    "causal_feature_replay": True,
    "proper_scoring_uplift": True,
    "after_cost_edge": True,
    "uncertainty_lower_bound": True,
    "drawdown_limit": True,
    "latency_stress": True,
    "displayed_depth_stress": True,
    "authenticated_order_lifecycle": True,
    "settlement_and_redemption": True,
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _promotion(
    root: Path,
    *,
    market_variant: str = "fiveminute",
) -> VerifiedPolymarketLivePromotion:
    root.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name in ("model", "evaluation", "implementation"):
        path = root / f"{name}.json"
        path.write_text(_canonical({name: "frozen"}), encoding="ascii")
        evidence[name] = {"path": path.name, "sha256": _file_sha(path)}
    body: dict[str, object] = {
        "schema_version": "polymarket-live-promotion-v1",
        "promotion_id": "a" * 64,
        "created_at_ms": NOW_MS - 86_400_000,
        "expires_at_ms": NOW_MS + 86_400_000,
        "source_commit": "b" * 40,
        "venue": "polymarket",
        "protocol_version": 2,
        "asset": "BTC",
        "market_variant": market_variant,
        "environment": "live",
        "bot_id": "simple-ai-trading-polymarket-btc",
        "model_artifact": evidence["model"],
        "evaluation_report": evidence["evaluation"],
        "implementation_manifest": evidence["implementation"],
        "gates": dict(GATES),
        "policy": {
            "minimum_expected_edge_quote_per_share": "0.02",
            "maximum_prediction_age_ms": 1_000,
            "minimum_remaining_seconds": 30,
        },
        "authority": {"paper": True, "live": True},
    }
    payload = {**body, "promotion_sha256": _sha(body)}
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
    horizon_minutes = (
        5 if promotion.promotion.market_variant == "fiveminute" else 15
    )
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
        event_start_time_ms=EVENT_START_MS,
        event_end_time_ms=EVENT_START_MS + horizon_minutes * 60_000,
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
    external: object | None = None,
    decision_data: object | None = None,
    markets: tuple[PolymarketFiveMinuteMarket, ...] | None = None,
    market_variant: str = "fiveminute",
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
        decision_provider=provider,  # type: ignore[arg-type]
        decision_data_service=decision_data,  # type: ignore[arg-type]
        external_signal_provider=external,  # type: ignore[arg-type]
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
    with pytest.raises(ValueError, match="duplicate"):
        PolymarketAutonomousDecision(proposals=(proposal, proposal))
    with pytest.raises(ValueError, match="open and close"):
        PolymarketAutonomousDecision(
            proposals=(proposal,),
            close_owned_exposure=True,
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


def test_external_failure_vetoes_without_calling_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion = _promotion(tmp_path / "proposal")
    proposal = _proposal(promotion)

    class External:
        trading_authority = False
        credentials_used = False
        execution_connected = False

        def evaluate(self, **_kwargs: object) -> PolymarketExternalSignalDecision:
            raise RuntimeError("public feed unavailable")

    supervisor = _supervisor(
        tmp_path / "runtime",
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
        external=External(),
    )
    submitted = False

    def submit(*_args: object, **_kwargs: object) -> object:
        nonlocal submitted
        submitted = True
        return object()

    monkeypatch.setattr(runtime_module, "submit_promoted_open", submit)
    asyncio.run(
        supervisor._apply_decision(
            PolymarketAutonomousDecision(proposals=(proposal,)),
            _market(),
            observed_at_ms=NOW_MS,
        )
    )

    assert submitted is False
    assert supervisor.snapshot().blocked_opens == 1
    assert "unavailable" in supervisor.snapshot().last_fault


def test_optional_external_signal_can_only_reduce_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    promotion = _promotion(tmp_path / "proposal")
    proposal = _proposal(promotion)
    signal = PolymarketExternalSignalDecision(
        action="reduce",
        maximum_size_multiplier=Decimal("0.5"),
        reasons=("wide_reference_spread",),
        features=PolymarketBtcReferenceFeatures(
            observed_at_ms=NOW_MS,
            spot_mid=Decimal("100"),
            futures_mid=Decimal("100.01"),
            spot_spread_bps=Decimal("1"),
            futures_spread_bps=Decimal("1"),
            futures_basis_bps=Decimal("1"),
            spot_log_return=0.0,
            futures_log_return=0.0,
            event_time_skew_ms=0,
            receive_time_skew_ms=0,
        ),
    )
    external = SimpleNamespace(
        trading_authority=False,
        credentials_used=False,
        execution_connected=False,
        evaluate=lambda **_kwargs: signal,
    )
    supervisor = _supervisor(
        tmp_path / "runtime",
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
        external=external,
    )
    received: list[PolymarketExternalSignalDecision | None] = []

    def submit(
        *_args: object,
        external_signal: PolymarketExternalSignalDecision | None,
        **_kwargs: object,
    ) -> PolymarketAutonomousOpenResult:
        received.append(external_signal)
        return PolymarketAutonomousOpenResult(
            record=SimpleNamespace(),  # type: ignore[arg-type]
            quote=SimpleNamespace(),  # type: ignore[arg-type]
            effective_quantity=Decimal("2.5"),
            worst_notional_quote=Decimal("1"),
            worst_fee_quote=Decimal("0.01"),
            net_edge_quote_per_share=Decimal("0.1"),
        )

    monkeypatch.setattr(runtime_module, "submit_promoted_open", submit)
    asyncio.run(
        supervisor._apply_decision(
            PolymarketAutonomousDecision(proposals=(proposal,)),
            _market(),
            observed_at_ms=NOW_MS,
        )
    )

    assert received == [signal]
    assert supervisor.snapshot().submitted_opens == 1
    assert supervisor.snapshot().external_signal_enabled is True


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


def test_run_starts_optional_public_signal_service(tmp_path: Path) -> None:
    started = False

    class External:
        trading_authority = False
        credentials_used = False
        execution_connected = False

        def evaluate(self, **_kwargs: object) -> PolymarketExternalSignalDecision:
            raise AssertionError("no proposal should be evaluated")

        async def run(self, stop: asyncio.Event) -> None:
            nonlocal started
            started = True
            await stop.wait()

    supervisor = _supervisor(
        tmp_path,
        provider=SimpleNamespace(decide=lambda **_kwargs: None),
        external=External(),
    )
    supervisor.request_stop()
    asyncio.run(supervisor.run())

    assert started is True


def test_runtime_rejects_external_provider_with_exchange_authority(
    tmp_path: Path,
) -> None:
    external = SimpleNamespace(
        trading_authority=True,
        credentials_used=False,
        execution_connected=False,
        evaluate=lambda **_kwargs: None,
    )

    with pytest.raises(PolymarketLiveBlocked, match="public and read-only"):
        _supervisor(
            tmp_path,
            provider=SimpleNamespace(decide=lambda **_kwargs: None),
            external=external,
        )


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
    supervisor.request_stop()
    asyncio.run(supervisor.run())

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
        match=(
            "critical_service_exit:"
            "authenticated_user_stream:ConnectionError"
        ),
    ):
        asyncio.run(supervisor.run())

    assert calls == 0
    assert supervisor.snapshot().stop_requested is True
    assert supervisor.snapshot().stop_completed is True
    assert supervisor.snapshot().last_fault == (
        "critical_service_exit:"
        "authenticated_user_stream:ConnectionError"
    )
