from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import time
from typing import Callable, Sequence

import pytest

from simple_ai_trading.polymarket_live import (
    PolymarketCancelResult,
    PolymarketFundingPreflight,
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveError,
    PolymarketLiveOrderIntent,
    PolymarketLiveOrderLedger,
    PolymarketLiveRiskLimits,
    PolymarketLiveUnknownState,
    PolymarketPreparedOrder,
    PolymarketReconciliation,
    PolymarketRemoteFill,
    PolymarketRemoteOrder,
    PolymarketRemotePosition,
    PolymarketSubmission,
    PolymarketVenuePreflight,
    PolymarketVenueRejected,
)
from simple_ai_trading.polymarket_live_runtime import (
    PolymarketLiveRuntimeGuard,
    PolymarketUserStreamConsumer,
)
from simple_ai_trading.polymarket_live_v2 import PolymarketLiveCredentials


NOW_MS = int(time.time() * 1_000)
MARKET_ID = "0x" + "1" * 64
TOKEN_ID = "1" * 40
WALLET = "0x" + "a" * 40
RISK_LIMITS = PolymarketLiveRiskLimits(
    maximum_order_quote=Decimal("10"),
    maximum_token_quantity=Decimal("20"),
    maximum_intent_age_ms=30_000,
)


def _order_id(seed: int) -> str:
    return "0x" + f"{seed:064x}"


def _intent(
    seed: int = 1,
    *,
    quantity: str = "1",
    side: str = "BUY",
    closing_only: bool = False,
) -> PolymarketLiveOrderIntent:
    return PolymarketLiveOrderIntent(
        intent_id=f"intent-{seed:04d}",
        bot_id="polymarket-live-bot",
        market_id=MARKET_ID,
        token_id=TOKEN_ID,
        symbol="BTC",
        outcome="Up",
        side=side,
        order_type="FAK",
        limit_price=Decimal("0.51"),
        quantity=Decimal(quantity),
        fee_reserve_quote=Decimal("0.10"),
        created_at_ms=NOW_MS,
        expires_at_ms=NOW_MS + 120_000,
        parent_intent_id="intent-0001" if closing_only else "",
        closing_only=closing_only,
    )


def _remote_order(
    intent: PolymarketLiveOrderIntent,
    order_id: str,
    *,
    matched: str = "0",
) -> PolymarketRemoteOrder:
    return PolymarketRemoteOrder(
        order_id=order_id,
        market_id=intent.market_id,
        token_id=intent.token_id,
        side=intent.side,
        status="LIVE",
        original_quantity=intent.quantity,
        matched_quantity=Decimal(matched),
    )


def _fill(
    intent: PolymarketLiveOrderIntent,
    order_id: str,
    *,
    seed: int = 1,
    quantity: str = "1",
    price: str = "0.51",
    status: str = "CONFIRMED",
) -> PolymarketRemoteFill:
    return PolymarketRemoteFill(
        trade_id=f"trade-{seed:04d}",
        order_id=order_id,
        market_id=intent.market_id,
        token_id=intent.token_id,
        side=intent.side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        status=status,
        observed_at_ms=NOW_MS + seed,
    )


class FakeVenue:
    def __init__(self) -> None:
        self.open: tuple[PolymarketRemoteOrder, ...] = ()
        self.remote_positions: tuple[PolymarketRemotePosition, ...] = ()
        self.fills: tuple[PolymarketRemoteFill, ...] = ()
        self.geoblocked = False
        self.closed_only = False
        self.protocol_version = 2
        self.clock_skew_ms = 0
        self.submit_calls = 0
        self.cancel_calls: list[tuple[str, ...]] = []
        self.submit_error: Exception | None = None
        self.submission_factory: (
            Callable[[PolymarketPreparedOrder], PolymarketSubmission] | None
        ) = None
        self.before_submit: Callable[[PolymarketPreparedOrder], None] | None = None
        self.cancel_result: PolymarketCancelResult | None = None
        self.cancel_error: Exception | None = None
        self.funding_balance = Decimal("100")
        self.funding_allowance = Decimal("100")

    def preflight(self) -> PolymarketVenuePreflight:
        return PolymarketVenuePreflight(
            protocol_version=self.protocol_version,
            server_time_ms=NOW_MS,
            observed_at_ms=NOW_MS + self.clock_skew_ms,
            geoblocked=self.geoblocked,
            country="US",
            region="NY",
            closed_only=self.closed_only,
            wallet_address=WALLET,
            open_orders=self.open,
            positions=self.remote_positions,
        )

    def prepare_order(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        tick_size: Decimal,
        neg_risk: bool,
    ) -> PolymarketPreparedOrder:
        del tick_size, neg_risk
        seed = int(intent.intent_id.rsplit("-", 1)[-1])
        return PolymarketPreparedOrder(
            intent=intent,
            expected_order_id=_order_id(seed),
            metadata=intent.metadata,
            opaque_signed_order={
                "signature": "signed-order-must-never-enter-the-ledger",
            },
        )

    def submit_order(self, prepared: PolymarketPreparedOrder) -> PolymarketSubmission:
        self.submit_calls += 1
        if self.before_submit is not None:
            self.before_submit(prepared)
        if self.submit_error is not None:
            raise self.submit_error
        if self.submission_factory is not None:
            return self.submission_factory(prepared)
        return PolymarketSubmission(
            accepted=True,
            order_id=prepared.expected_order_id,
            status="live",
        )

    def open_orders(self) -> tuple[PolymarketRemoteOrder, ...]:
        return self.open

    def fills_for_orders(
        self,
        order_ids: Sequence[str],
        *,
        market_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteFill, ...]:
        del market_ids
        requested = set(order_ids)
        return tuple(fill for fill in self.fills if fill.order_id in requested)

    def positions(self) -> tuple[PolymarketRemotePosition, ...]:
        return self.remote_positions

    def funding(
        self,
        intent: PolymarketLiveOrderIntent,
        *,
        neg_risk: bool,
    ) -> PolymarketFundingPreflight:
        del neg_risk
        return PolymarketFundingPreflight(
            asset_type="COLLATERAL" if intent.side == "BUY" else "CONDITIONAL",
            token_id="" if intent.side == "BUY" else intent.token_id,
            available_balance=self.funding_balance,
            available_allowance=self.funding_allowance,
        )

    def cancel_orders(self, order_ids: Sequence[str]) -> PolymarketCancelResult:
        requested = tuple(order_ids)
        self.cancel_calls.append(requested)
        if self.cancel_error is not None:
            raise self.cancel_error
        return self.cancel_result or PolymarketCancelResult(requested, ())


def _seed_live_order(
    ledger: PolymarketLiveOrderLedger,
    venue: FakeVenue,
    intent: PolymarketLiveOrderIntent,
) -> PolymarketPreparedOrder:
    prepared = venue.prepare_order(intent, tick_size=Decimal("0.01"), neg_risk=False)
    ledger.reserve(prepared, observed_at_ms=NOW_MS)
    ledger.transition(
        intent.intent_id,
        expected_states=("prepared",),
        state="live",
        observed_at_ms=NOW_MS + 1,
        remote_status="LIVE",
    )
    return prepared


@pytest.mark.parametrize("order_type", ["GTC", "IOC", ""])
def test_live_intent_rejects_unbounded_order_types(order_type: str) -> None:
    with pytest.raises(ValueError, match="FAK, FOK, or GTD"):
        replace(_intent(), order_type=order_type)


def test_live_credentials_never_render_secrets() -> None:
    credentials = PolymarketLiveCredentials(
        private_key="0x" + "1" * 64,
        api_key="api-key-secret",
        api_secret="api-secret-value",
        api_passphrase="passphrase-secret",
        funder_address=WALLET,
        signature_type=0,
    )

    rendered = repr(credentials)

    assert "api-key-secret" not in rendered
    assert "api-secret-value" not in rendered
    assert "passphrase-secret" not in rendered
    assert "0x" + "1" * 64 not in rendered


def test_submission_reserves_expected_hash_before_network_and_stores_no_signature(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)
    observed_states: list[str] = []

    def before_submit(prepared: PolymarketPreparedOrder) -> None:
        record = ledger.record(prepared.intent.intent_id)
        observed_states.append(record.state)
        assert record.expected_order_id == prepared.expected_order_id

    venue.before_submit = before_submit
    record = coordinator.submit(_intent(), tick_size=Decimal("0.01"), neg_risk=False)

    assert record.state == "live"
    assert observed_states == ["submitting"]
    database = (tmp_path / "live.sqlite3").read_bytes()
    assert b"signed-order-must-never-enter-the-ledger" not in database


def test_ambiguous_submission_is_never_retried_and_blocks_exposure(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    venue.submit_error = TimeoutError("transport ambiguity")
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="no retry"):
        coordinator.submit(_intent(), tick_size=Decimal("0.01"), neg_risk=False)

    assert venue.submit_calls == 1
    assert ledger.record("intent-0001").state == "unknown"
    assert coordinator.preflight().can_open is False


@pytest.mark.parametrize("transport_error", [False, True])
def test_authenticated_stream_can_resolve_http_submission_race(
    tmp_path: Path,
    transport_error: bool,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / f"race-{transport_error}.sqlite3")
    venue = FakeVenue()
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    def before_submit(prepared: PolymarketPreparedOrder) -> None:
        ledger.transition(
            prepared.intent.intent_id,
            expected_states=("submitting",),
            state="live",
            observed_at_ms=int(time.time() * 1_000),
            remote_status="PLACEMENT",
        )

    venue.before_submit = before_submit
    if transport_error:
        venue.submit_error = TimeoutError("response lost after placement")

    record = coordinator.submit(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )

    assert record.state == "live"
    assert record.remote_status == "PLACEMENT"
    assert venue.submit_calls == 1


def test_proven_venue_rejection_is_terminal(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    venue.submit_error = PolymarketVenueRejected("proven rejection")
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    record = coordinator.submit(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )

    assert record.state == "rejected"
    assert venue.submit_calls == 1


def test_foreign_order_and_position_block_without_being_touched(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    venue.open = (_remote_order(intent, _order_id(99)),)
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("2"),
            redeemable=False,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    gate = coordinator.preflight()

    assert gate.can_open is False
    assert gate.can_close is False
    assert gate.foreign_order_ids == (_order_id(99),)
    assert gate.foreign_position_token_ids == (TOKEN_ID,)
    assert venue.cancel_calls == []


def test_cancellation_targets_only_exact_owned_open_order_ids(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (
        _remote_order(intent, prepared.expected_order_id),
        _remote_order(intent, _order_id(99)),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    result = coordinator.cancel_owned_open_orders()

    assert result.cancelled_order_ids == (prepared.expected_order_id,)
    assert venue.cancel_calls == [(prepared.expected_order_id,)]
    assert ledger.record(intent.intent_id).state == "cancelled"


def test_missing_owned_order_during_cancel_is_recorded_as_unknown(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    first = _intent(1)
    second = _intent(2)
    first_prepared = _seed_live_order(ledger, venue, first)
    _seed_live_order(ledger, venue, second)
    venue.open = (_remote_order(first, first_prepared.expected_order_id),)
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="absent"):
        coordinator.cancel_owned_open_orders()

    assert venue.cancel_calls == [(first_prepared.expected_order_id,)]
    assert ledger.record(first.intent_id).state == "cancelled"
    assert ledger.record(second.intent_id).state == "cancel_unknown"


def test_reconciliation_never_associates_fill_by_token_only(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    first = _intent(1)
    second = _intent(2)
    first_prepared = _seed_live_order(ledger, venue, first)
    _seed_live_order(ledger, venue, second)
    venue.fills = (_fill(first, first_prepared.expected_order_id),)
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            redeemable=False,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    gate = coordinator.reconcile()

    assert ledger.record(first.intent_id).state == "filled"
    assert ledger.record(first.intent_id).matched_quantity == Decimal("1")
    assert ledger.record(second.intent_id).state == "unknown"
    assert "unknown_order_state" in gate.errors


def test_fill_economics_are_immutable_and_cumulative_quantity_is_capped(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    intent = _intent(quantity="1")
    prepared = _seed_live_order(ledger, venue, intent)
    first = _fill(
        intent,
        prepared.expected_order_id,
        quantity="0.75",
        status="MINED",
    )
    ledger.record_fill(first)

    with pytest.raises(PolymarketLiveBlocked, match="economics differ"):
        ledger.record_fill(replace(first, price=Decimal("0.52"), status="CONFIRMED"))
    with pytest.raises(PolymarketLiveBlocked, match="cumulative fills"):
        ledger.record_fill(
            _fill(
                intent,
                prepared.expected_order_id,
                seed=2,
                quantity="0.5",
                status="MATCHED",
            )
        )

    evidence = ledger.order_fill_evidence(prepared.expected_order_id)
    assert evidence.quantity == Decimal("0.75")
    assert evidence.all_active_fills_confirmed is False


def test_order_and_fill_snapshot_corruption_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "live.sqlite3"
    ledger = PolymarketLiveOrderLedger(path)
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    ledger.record_fill(_fill(intent, prepared.expected_order_id))
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE polymarket_live_fills SET quantity = '9' WHERE order_id = ?",
            [prepared.expected_order_id],
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PolymarketLiveError, match="fill snapshot hash"):
        ledger.owned_inventory()

    second_path = tmp_path / "orders.sqlite3"
    second = PolymarketLiveOrderLedger(second_path)
    second_prepared = _seed_live_order(second, venue, _intent(2))
    connection = sqlite3.connect(second_path)
    try:
        connection.execute(
            "UPDATE polymarket_live_orders SET state = 'filled' "
            "WHERE expected_order_id = ?",
            [second_prepared.expected_order_id],
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PolymarketLiveError, match="order snapshot hash"):
        second.records()


@pytest.mark.parametrize(
    ("attribute", "value", "expected_error"),
    [
        ("geoblocked", True, "geoblocked"),
        ("closed_only", True, "closed_only"),
        ("protocol_version", 1, "unsupported_clob_protocol"),
        ("clock_skew_ms", 6_000, "clock_skew"),
    ],
)
def test_preflight_fails_closed_on_venue_safety_gates(
    tmp_path: Path,
    attribute: str,
    value: object,
    expected_error: str,
) -> None:
    venue = FakeVenue()
    setattr(venue, attribute, value)
    coordinator = PolymarketLiveCoordinator(
        venue,
        PolymarketLiveOrderLedger(tmp_path / "live.sqlite3"),
        risk_limits=RISK_LIMITS,
    )

    gate = coordinator.preflight()

    assert gate.can_open is False
    assert expected_error in gate.errors


def test_live_execution_requires_a_dedicated_wallet(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dedicated wallet"):
        PolymarketLiveCoordinator(
            FakeVenue(),
            PolymarketLiveOrderLedger(tmp_path / "live.sqlite3"),
            risk_limits=RISK_LIMITS,
            require_dedicated_wallet=False,
        )


def test_hard_risk_limits_block_stale_and_oversized_open_intents(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    coordinator = PolymarketLiveCoordinator(
        FakeVenue(),
        ledger,
        risk_limits=RISK_LIMITS,
    )
    stale = replace(
        _intent(),
        created_at_ms=int(time.time() * 1_000) - 31_000,
        expires_at_ms=int(time.time() * 1_000) + 30_000,
    )

    with pytest.raises(PolymarketLiveBlocked, match="execution TTL"):
        coordinator.submit(stale, tick_size=Decimal("0.01"), neg_risk=False)
    with pytest.raises(PolymarketLiveBlocked, match="quote ceiling"):
        coordinator.submit(
            _intent(quantity="20"),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    assert ledger.records() == ()


@pytest.mark.parametrize("field", ["funding_balance", "funding_allowance"])
def test_order_requires_sufficient_exact_asset_funding(
    tmp_path: Path,
    field: str,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / f"{field}.sqlite3")
    venue = FakeVenue()
    setattr(venue, field, Decimal("0.60"))
    coordinator = PolymarketLiveCoordinator(
        venue,
        ledger,
        risk_limits=RISK_LIMITS,
    )

    with pytest.raises(PolymarketLiveBlocked, match="balance or exchange allowance"):
        coordinator.submit(
            _intent(),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    assert venue.submit_calls == 0
    assert ledger.records() == ()


def test_sell_cannot_be_created_without_bot_owned_parent() -> None:
    with pytest.raises(ValueError, match="must close bot-owned"):
        _intent(side="SELL")


def test_closing_order_requires_matching_confirmed_owned_inventory(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    coordinator = PolymarketLiveCoordinator(
        venue,
        ledger,
        risk_limits=RISK_LIMITS,
    )

    with pytest.raises(PolymarketLiveBlocked, match="parent is not bot-owned"):
        coordinator.submit(
            _intent(2, side="SELL", closing_only=True),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    parent = _intent(1)
    prepared = _seed_live_order(ledger, venue, parent)
    ledger.record_fill(_fill(parent, prepared.expected_order_id))
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            redeemable=False,
        ),
    )
    coordinator.reconcile()

    close = coordinator.submit(
        _intent(2, side="SELL", closing_only=True),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )

    assert close.intent.parent_intent_id == parent.intent_id
    assert close.intent.closing_only is True
    assert close.state == "live"


def test_runtime_guard_requires_fresh_stream_for_open_but_not_owned_close() -> None:
    clock = [0]
    guard = PolymarketLiveRuntimeGuard(
        maximum_stream_age_ms=15_000,
        maximum_reconciliation_age_ms=30_000,
        monotonic_ns=lambda: clock[0],
    )
    clean = PolymarketReconciliation(
        ok=True,
        can_open=True,
        can_close=True,
        foreign_order_ids=(),
        foreign_position_token_ids=(),
        missing_position_token_ids=(),
        blocking_intent_ids=(),
        errors=(),
    )
    guard.note_reconciliation(clean)

    with pytest.raises(PolymarketLiveBlocked, match="user stream"):
        guard.assert_submission_allowed(closing_only=False)
    guard.assert_submission_allowed(closing_only=True)

    guard.note_stream_liveness()
    guard.assert_submission_allowed(closing_only=False)
    clock[0] = 16_000_000_000

    with pytest.raises(PolymarketLiveBlocked, match="user stream"):
        guard.assert_submission_allowed(closing_only=False)
    guard.assert_submission_allowed(closing_only=True)


def test_runtime_hard_fault_blocks_open_and_close() -> None:
    guard = PolymarketLiveRuntimeGuard()
    guard.note_stream_liveness()
    guard.note_reconciliation(
        PolymarketReconciliation(
            ok=True,
            can_open=True,
            can_close=True,
            foreign_order_ids=(),
            foreign_position_token_ids=(),
            missing_position_token_ids=(),
            blocking_intent_ids=(),
            errors=(),
        )
    )
    guard.note_hard_fault("foreign_order_stream_event")

    with pytest.raises(PolymarketLiveBlocked, match="hard faults"):
        guard.assert_submission_allowed(closing_only=False)
    with pytest.raises(PolymarketLiveBlocked, match="hard faults"):
        guard.assert_submission_allowed(closing_only=True)


def test_coordinator_runtime_authority_blocks_network_submission_until_live(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    guard = PolymarketLiveRuntimeGuard()
    coordinator = PolymarketLiveCoordinator(
        venue,
        ledger,
        risk_limits=RISK_LIMITS,
        runtime_authority=guard,
    )

    with pytest.raises(PolymarketLiveBlocked, match="user stream"):
        coordinator.submit(
            _intent(),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )
    assert venue.submit_calls == 0
    guard.note_stream_liveness()

    record = coordinator.submit(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )

    assert record.state == "live"
    assert venue.submit_calls == 1


def test_user_stream_applies_only_exact_owned_order_and_fill_events(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    guard = PolymarketLiveRuntimeGuard()
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    timestamp = str(int(time.time()))
    order_event = {
        "event_type": "order",
        "type": "UPDATE",
        "id": prepared.expected_order_id,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "side": "BUY",
        "original_size": "1",
        "size_matched": "0.25",
        "timestamp": timestamp,
    }
    trade_event = {
        "event_type": "trade",
        "type": "TRADE",
        "id": "stream-trade-0001",
        "taker_order_id": prepared.expected_order_id,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "side": "BUY",
        "size": "0.25",
        "price": "0.50",
        "status": "CONFIRMED",
        "last_update": timestamp,
        "maker_orders": [],
    }

    assert consumer.handle(json.dumps(order_event)) == 1
    assert consumer.handle(json.dumps(trade_event)) == 1

    record = ledger.record(intent.intent_id)
    evidence = ledger.order_fill_evidence(prepared.expected_order_id)
    assert record.state == "partial"
    assert record.matched_quantity == Decimal("0.25")
    assert evidence.quantity == Decimal("0.25")
    assert evidence.all_active_fills_confirmed is True
    assert guard.snapshot().hard_faults == ()


def test_foreign_user_stream_event_latches_without_touching_ledger(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "live.sqlite3")
    guard = PolymarketLiveRuntimeGuard()
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    foreign = {
        "event_type": "order",
        "type": "PLACEMENT",
        "id": _order_id(99),
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "side": "BUY",
        "original_size": "1",
        "size_matched": "0",
        "timestamp": str(int(time.time())),
    }

    assert consumer.handle(json.dumps(foreign)) == 1

    assert ledger.records() == ()
    assert guard.snapshot().hard_faults == ("foreign_order_stream_event",)
