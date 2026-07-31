from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Event, Thread
import time
from typing import Callable, Sequence

import pytest

from simple_ai_trading.polymarket_live import (
    PolymarketCancelResult,
    PolymarketCloseQuote,
    PolymarketFundingPreflight,
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveError,
    PolymarketLiveOrderIntent,
    PolymarketLiveOrderLedger,
    PolymarketLiveRiskLimits,
    PolymarketLiveUnknownState,
    PolymarketOwnedInventory,
    PolymarketOwnedLot,
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
from simple_ai_trading.polymarket_runtime_control import (
    PolymarketRuntimeControl,
    PolymarketRuntimeLeaseInterlock,
)


NOW_MS = int(time.time() * 1_000)
MARKET_ID = "0x" + "1" * 64
TOKEN_ID = "1" * 40
WALLET = "0x" + "a" * 40
RISK_LIMITS = PolymarketLiveRiskLimits(
    maximum_order_quote=Decimal("10"),
    maximum_token_quantity=Decimal("20"),
    maximum_total_at_risk_quote=Decimal("10"),
    maximum_active_markets=1,
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
        maker_address=WALLET,
        side=intent.side,
        order_type=intent.order_type,
        price=intent.limit_price,
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
        role="TAKER",
        reported_fee_rate_bps=0,
        fee_rate=Decimal("0"),
        fee_exponent=1,
        fee_quote=Decimal("0"),
        fee_schedule_sha256="a" * 64,
        transaction_hash="0x" + "d" * 64,
    )


class FakeVenue:
    def __init__(self) -> None:
        self.wallet_address = WALLET
        self.open: tuple[PolymarketRemoteOrder, ...] = ()
        self.remote_positions: tuple[PolymarketRemotePosition, ...] = ()
        self.fills: tuple[PolymarketRemoteFill, ...] = ()
        self.exact_orders: tuple[PolymarketRemoteOrder, ...] = ()
        self.exact_order_calls: list[tuple[str, ...]] = []
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
        self.before_open_orders: Callable[[], None] | None = None
        self.before_cancel: Callable[[tuple[str, ...]], None] | None = None
        self.cancel_result: PolymarketCancelResult | None = None
        self.cancel_error: Exception | None = None
        self.funding_balance = Decimal("100")
        self.funding_allowance = Decimal("100")
        self.close_quotes: list[PolymarketCloseQuote] = []
        self.close_quote_calls: list[tuple[str, str, Decimal, int]] = []

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
        suffix = intent.intent_id.rsplit("-", 1)[-1]
        order_id = (
            _order_id(int(suffix))
            if suffix.isdigit()
            else "0x" + hashlib.sha256(intent.intent_id.encode("ascii")).hexdigest()
        )
        return PolymarketPreparedOrder(
            intent=intent,
            expected_order_id=order_id,
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
        if self.before_open_orders is not None:
            self.before_open_orders()
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

    def orders_by_id(
        self,
        order_ids: Sequence[str],
    ) -> tuple[PolymarketRemoteOrder, ...]:
        requested = tuple(order_ids)
        self.exact_order_calls.append(requested)
        requested_set = set(requested)
        return tuple(
            order for order in self.exact_orders if order.order_id in requested_set
        )

    def close_quote(
        self,
        *,
        market_id: str,
        token_id: str,
        quantity: Decimal,
        maximum_book_age_ms: int,
    ) -> PolymarketCloseQuote:
        self.close_quote_calls.append(
            (market_id, token_id, quantity, maximum_book_age_ms)
        )
        if self.close_quotes:
            return self.close_quotes.pop(0)
        return PolymarketCloseQuote(
            market_id=market_id,
            token_id=token_id,
            quantity=quantity,
            limit_price=Decimal("0.49"),
            average_price=Decimal("0.49"),
            fee_quote=Decimal("0"),
            net_quote=Decimal("0.49") * quantity,
            fee_rate=Decimal("0"),
            fee_exponent=1,
            tick_size=Decimal("0.01"),
            minimum_order_size=Decimal("0.1"),
            neg_risk=False,
            source_time_ms=int(time.time() * 1_000),
            observed_at_ms=int(time.time() * 1_000),
            book_payload_sha256="a" * 64,
        )

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
        if self.before_cancel is not None:
            self.before_cancel(requested)
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


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"limit_price": Decimal("0.495")}, "venue tick"),
        ({"quantity": Decimal("4.9")}, "venue minimum"),
        ({"source_time_ms": 0}, "chronology"),
        ({"book_payload_sha256": "bad"}, "payload hash"),
    ],
)
def test_close_quote_value_object_rejects_invalid_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "market_id": MARKET_ID,
        "token_id": TOKEN_ID,
        "quantity": Decimal("5"),
        "limit_price": Decimal("0.49"),
        "average_price": Decimal("0.49"),
        "fee_quote": Decimal("0"),
        "net_quote": Decimal("2.45"),
        "fee_rate": Decimal("0"),
        "fee_exponent": 1,
        "tick_size": Decimal("0.01"),
        "minimum_order_size": Decimal("5"),
        "neg_risk": False,
        "source_time_ms": NOW_MS,
        "observed_at_ms": NOW_MS + 1,
        "book_payload_sha256": "a" * 64,
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        PolymarketCloseQuote(**values)  # type: ignore[arg-type]


def test_owned_lot_cannot_reserve_more_than_it_owns() -> None:
    with pytest.raises(ValueError, match="exceeds lot"):
        PolymarketOwnedLot(
            parent_intent_id="intent-0001",
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            reserved_close_quantity=Decimal("1.1"),
            provisional=False,
        )


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


def test_fill_price_cannot_violate_signed_limit_economics(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "fill-limit.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)

    with pytest.raises(PolymarketLiveBlocked, match="signed limit"):
        ledger.record_fill(
            _fill(
                intent,
                prepared.expected_order_id,
                price="0.52",
            )
        )

    assert ledger.order_fill_evidence(prepared.expected_order_id).quantity == 0


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


def test_submission_response_order_id_mismatch_is_unknown(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "submission-id-mismatch.sqlite3")
    venue = FakeVenue()
    venue.submission_factory = lambda _prepared: PolymarketSubmission(
        accepted=True,
        order_id=_order_id(999),
        status="live",
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="order ID differs"):
        coordinator.submit(
            _intent(),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    record = ledger.record(_intent().intent_id)
    assert record.state == "unknown"
    assert record.failure_code == "venue_order_id_mismatch"


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


def test_cancelled_partial_order_waits_for_exact_fill_evidence(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "partial-cancel.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (
        _remote_order(
            intent,
            prepared.expected_order_id,
            matched="0.4",
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="lack terminal"):
        coordinator.cancel_owned_open_orders()

    record = ledger.record(intent.intent_id)
    assert record.state == "matched_pending"
    assert record.matched_quantity == Decimal("0.4")
    assert record.failure_code == "cancelled_order_awaiting_exact_fill_evidence"


def test_cancelled_partial_order_can_close_after_exact_confirmed_fill(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "confirmed-partial-cancel.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    ledger.record_fill(
        _fill(
            intent,
            prepared.expected_order_id,
            quantity="0.4",
            status="CONFIRMED",
        )
    )
    venue.open = (
        _remote_order(
            intent,
            prepared.expected_order_id,
            matched="0.4",
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    result = coordinator.cancel_owned_open_orders()

    assert result.cancelled_order_ids == (prepared.expected_order_id,)
    record = ledger.record(intent.intent_id)
    assert record.state == "filled"
    assert record.matched_quantity == Decimal("0.4")


@pytest.mark.parametrize(
    "result",
    [
        PolymarketCancelResult((_order_id(1), _order_id(2)), ()),
        PolymarketCancelResult((), (_order_id(1), _order_id(2))),
    ],
)
def test_cancel_result_rejects_duplicate_ids(
    result: PolymarketCancelResult,
) -> None:
    values = (result.cancelled_order_ids[0],) * 2 if result.cancelled_order_ids else ()
    failures = (result.failed_order_ids[0],) * 2 if result.failed_order_ids else ()
    with pytest.raises(ValueError, match="duplicates"):
        PolymarketCancelResult(values, failures)


def test_venue_cancel_failure_is_unknown_and_blocks_normally_returning(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-failed.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    venue.cancel_result = PolymarketCancelResult(
        (),
        (prepared.expected_order_id,),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="lack terminal"):
        coordinator.cancel_owned_open_orders()

    assert ledger.record(intent.intent_id).state == "cancel_unknown"


def test_incomplete_or_unrequested_cancel_response_fails_closed(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-response.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    venue.cancel_result = PolymarketCancelResult((), ())
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="incomplete"):
        coordinator.cancel_owned_open_orders()

    assert ledger.record(intent.intent_id).state == "cancel_unknown"

    second_ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-unrequested.sqlite3")
    second_venue = FakeVenue()
    second_intent = _intent(2)
    second_prepared = _seed_live_order(
        second_ledger,
        second_venue,
        second_intent,
    )
    second_venue.open = (
        _remote_order(second_intent, second_prepared.expected_order_id),
    )
    second_venue.cancel_result = PolymarketCancelResult((_order_id(99),), ())
    second_coordinator = PolymarketLiveCoordinator(
        second_venue,
        second_ledger,
        risk_limits=RISK_LIMITS,
    )

    with pytest.raises(PolymarketLiveBlocked, match="unrequested"):
        second_coordinator.cancel_owned_open_orders()


def test_pending_cancel_is_not_blindly_retried(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-pending.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    ledger.transition(
        intent.intent_id,
        expected_states=("live",),
        state="cancel_pending",
        observed_at_ms=NOW_MS + 2,
        remote_status="LIVE",
    )
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="terminal"):
        coordinator.cancel_owned_open_orders()

    assert venue.cancel_calls == []


def test_cancel_transport_error_never_retries(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-timeout.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    venue.cancel_error = TimeoutError("response lost")
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveUnknownState, match="outcome is unknown"):
        coordinator.cancel_owned_open_orders()

    assert venue.cancel_calls == [(prepared.expected_order_id,)]
    assert ledger.record(intent.intent_id).state == "cancel_unknown"


@pytest.mark.parametrize("remote_present", [False, True])
def test_cancel_snapshot_compare_and_swap_conflict_stays_fail_closed(
    tmp_path: Path,
    remote_present: bool,
) -> None:
    ledger = PolymarketLiveOrderLedger(
        tmp_path / f"cancel-cas-{remote_present}.sqlite3"
    )
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (
        (_remote_order(intent, prepared.expected_order_id),) if remote_present else ()
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    def before_open_orders() -> None:
        venue.before_open_orders = None
        ledger.transition(
            intent.intent_id,
            expected_states=("live",),
            state="partial",
            observed_at_ms=int(time.time() * 1_000),
            remote_status="UPDATE",
            matched_quantity=Decimal("0.5"),
        )

    venue.before_open_orders = before_open_orders

    with pytest.raises(PolymarketLiveUnknownState, match="terminal"):
        coordinator.cancel_owned_open_orders()

    assert venue.cancel_calls == []
    assert ledger.record(intent.intent_id).state == "partial"


@pytest.mark.parametrize("response_lost", [False, True])
def test_cancel_stream_race_after_submit_never_masks_uncertainty(
    tmp_path: Path,
    response_lost: bool,
) -> None:
    ledger = PolymarketLiveOrderLedger(
        tmp_path / f"cancel-post-race-{response_lost}.sqlite3"
    )
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    def before_cancel(order_ids: tuple[str, ...]) -> None:
        assert order_ids == (prepared.expected_order_id,)
        ledger.transition(
            intent.intent_id,
            expected_states=("cancel_pending",),
            state="cancelled" if response_lost else "live",
            observed_at_ms=int(time.time() * 1_000),
            remote_status="CANCELLATION" if response_lost else "UPDATE",
        )

    venue.before_cancel = before_cancel
    if response_lost:
        venue.cancel_error = TimeoutError("response lost")

    with pytest.raises(PolymarketLiveUnknownState):
        coordinator.cancel_owned_open_orders()

    assert venue.cancel_calls == [(prepared.expected_order_id,)]
    assert ledger.record(intent.intent_id).state == (
        "cancelled" if response_lost else "live"
    )


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


def test_authenticated_stream_can_close_before_cancel_compare_and_swap(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-before-cas.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    def before_open_orders() -> None:
        venue.before_open_orders = None
        ledger.transition(
            intent.intent_id,
            expected_states=("live",),
            state="cancelled",
            observed_at_ms=int(time.time() * 1_000),
            remote_status="CANCELLATION",
        )

    venue.before_open_orders = before_open_orders

    result = coordinator.cancel_owned_open_orders()

    assert result == PolymarketCancelResult((), ())
    assert venue.cancel_calls == []
    assert ledger.record(intent.intent_id).state == "cancelled"


def test_authenticated_stream_can_close_while_cancel_response_is_in_flight(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "cancel-in-flight.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.open = (_remote_order(intent, prepared.expected_order_id),)
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    def before_cancel(order_ids: tuple[str, ...]) -> None:
        assert order_ids == (prepared.expected_order_id,)
        ledger.transition(
            intent.intent_id,
            expected_states=("cancel_pending",),
            state="cancelled",
            observed_at_ms=int(time.time() * 1_000),
            remote_status="CANCELLATION",
        )

    venue.before_cancel = before_cancel

    result = coordinator.cancel_owned_open_orders()

    assert result.cancelled_order_ids == (prepared.expected_order_id,)
    assert venue.cancel_calls == [(prepared.expected_order_id,)]
    assert ledger.record(intent.intent_id).state == "cancelled"


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


@pytest.mark.parametrize(
    ("remote_status", "expected_state"),
    [
        ("ORDER_STATUS_CANCELED", "cancelled"),
        ("ORDER_STATUS_CANCELED_MARKET_RESOLVED", "expired"),
        ("ORDER_STATUS_INVALID", "rejected"),
    ],
)
def test_exact_terminal_order_proves_zero_fill_without_open_snapshot(
    tmp_path: Path,
    remote_status: str,
    expected_state: str,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / f"{expected_state}.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.exact_orders = (
        replace(
            _remote_order(intent, prepared.expected_order_id),
            status=remote_status,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    gate = coordinator.reconcile()

    assert venue.exact_order_calls == [(prepared.expected_order_id,)]
    assert ledger.record(intent.intent_id).state == expected_state
    assert "unknown_order_state" not in gate.errors


def test_terminal_partial_match_waits_for_exact_confirmed_fill(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "terminal-partial.sqlite3")
    venue = FakeVenue()
    intent = _intent(quantity="1")
    prepared = _seed_live_order(ledger, venue, intent)
    venue.exact_orders = (
        replace(
            _remote_order(intent, prepared.expected_order_id, matched="0.4"),
            status="ORDER_STATUS_CANCELED",
        ),
    )
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("0.4"),
            redeemable=False,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    first_gate = coordinator.reconcile()

    first = ledger.record(intent.intent_id)
    assert first.state == "matched_pending"
    assert first.matched_quantity == Decimal("0.4")
    assert first.failure_code == "terminal_order_awaiting_exact_fill_evidence"
    assert "unknown_order_state" not in first_gate.errors
    assert "provisional_fill_state" not in first_gate.errors

    venue.fills = (
        _fill(
            intent,
            prepared.expected_order_id,
            quantity="0.4",
            status="MATCHED",
        ),
    )
    second_gate = coordinator.reconcile()
    assert ledger.record(intent.intent_id).state == "matched_pending"
    assert "provisional_fill_state" in second_gate.errors

    venue.fills = (
        _fill(
            intent,
            prepared.expected_order_id,
            quantity="0.4",
            status="CONFIRMED",
        ),
    )
    final_gate = coordinator.reconcile()

    final = ledger.record(intent.intent_id)
    assert final.state == "filled"
    assert final.matched_quantity == Decimal("0.4")
    assert final.failure_code == ""
    assert final_gate.ok is True


def test_unsupported_exact_order_status_stays_unknown(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "unsupported-status.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.exact_orders = (
        replace(
            _remote_order(intent, prepared.expected_order_id),
            status="ORDER_STATUS_UNDOCUMENTED",
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    gate = coordinator.reconcile()

    record = ledger.record(intent.intent_id)
    assert record.state == "unknown"
    assert record.failure_code == "unsupported_remote_order_status"
    assert "unknown_order_state" in gate.errors


def test_unrequested_exact_order_response_is_rejected(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "foreign-exact.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    _seed_live_order(ledger, venue, intent)
    venue.exact_orders = (_remote_order(intent, _order_id(999)),)
    venue.orders_by_id = lambda order_ids: venue.exact_orders  # type: ignore[method-assign]
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveBlocked, match="unrequested"):
        coordinator.reconcile()


def test_duplicate_exact_order_response_is_rejected(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "duplicate-exact.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    exact = _remote_order(intent, prepared.expected_order_id)
    venue.orders_by_id = lambda order_ids: (exact, exact)  # type: ignore[method-assign]
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveBlocked, match="duplicate"):
        coordinator.reconcile()


@pytest.mark.parametrize(
    ("remote_factory", "fill", "failure_code"),
    [
        (
            lambda intent, order_id: replace(
                _remote_order(intent, order_id),
                token_id="2" * 40,
            ),
            None,
            "remote_order_identity_mismatch",
        ),
        (
            lambda intent, order_id: replace(
                _remote_order(intent, order_id),
                maker_address="0x" + "b" * 40,
            ),
            None,
            "remote_order_identity_mismatch",
        ),
        (
            lambda intent, order_id: replace(
                _remote_order(intent, order_id),
                order_type="FOK",
            ),
            None,
            "remote_order_identity_mismatch",
        ),
        (
            lambda intent, order_id: replace(
                _remote_order(intent, order_id),
                price=Decimal("0.52"),
            ),
            None,
            "remote_order_identity_mismatch",
        ),
        (
            lambda intent, order_id: _remote_order(intent, order_id),
            ("0.2", "CONFIRMED"),
            "remote_order_fill_quantity_mismatch",
        ),
        (
            lambda intent, order_id: replace(
                _remote_order(intent, order_id, matched="0.2"),
                status="ORDER_STATUS_INVALID",
            ),
            None,
            "invalid_order_has_fill_evidence",
        ),
        (
            lambda intent, order_id: replace(
                _remote_order(intent, order_id),
                status="ORDER_STATUS_MATCHED",
            ),
            None,
            "terminal_order_fill_quantity_mismatch",
        ),
    ],
)
def test_exact_order_contradictions_remain_unknown(
    tmp_path: Path,
    remote_factory: Callable[
        [PolymarketLiveOrderIntent, str],
        PolymarketRemoteOrder,
    ],
    fill: tuple[str, str] | None,
    failure_code: str,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / f"{failure_code}.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    if fill is not None:
        ledger.record_fill(
            _fill(
                intent,
                prepared.expected_order_id,
                quantity=fill[0],
                status=fill[1],
            )
        )
        venue.remote_positions = (
            PolymarketRemotePosition(
                market_id=MARKET_ID,
                token_id=TOKEN_ID,
                quantity=Decimal(fill[0]),
                redeemable=False,
            ),
        )
    venue.exact_orders = (remote_factory(intent, prepared.expected_order_id),)
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    gate = coordinator.reconcile()

    record = ledger.record(intent.intent_id)
    assert record.state == "unknown"
    assert record.failure_code == failure_code
    assert "unknown_order_state" in gate.errors


def test_cancel_uses_exact_live_proof_but_never_queries_foreign_ids(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "exact-live-cancel.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.exact_orders = (
        replace(
            _remote_order(intent, prepared.expected_order_id),
            status="ORDER_STATUS_LIVE",
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    result = coordinator.cancel_owned_open_orders()

    assert venue.exact_order_calls == [(prepared.expected_order_id,)]
    assert venue.cancel_calls == [(prepared.expected_order_id,)]
    assert result.cancelled_order_ids == (prepared.expected_order_id,)
    assert ledger.record(intent.intent_id).state == "cancelled"


def test_cancel_rejects_duplicate_exact_order_evidence(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "duplicate-cancel-exact.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    exact = _remote_order(intent, prepared.expected_order_id)
    venue.orders_by_id = lambda order_ids: (exact, exact)  # type: ignore[method-assign]
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveBlocked, match="duplicate"):
        coordinator.cancel_owned_open_orders()

    assert venue.cancel_calls == []


def test_cancel_accepts_exact_zero_fill_terminal_proof_without_retry(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "exact-terminal-cancel.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    venue.exact_orders = (
        replace(
            _remote_order(intent, prepared.expected_order_id),
            status="ORDER_STATUS_CANCELED",
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    result = coordinator.cancel_owned_open_orders()

    assert result == PolymarketCancelResult((), ())
    assert venue.cancel_calls == []
    assert ledger.record(intent.intent_id).state == "cancelled"


def test_unresolved_redemption_blocks_reconciled_exposure(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "redemption-gate.sqlite3")
    ledger.reserve_redemption(
        MARKET_ID,
        (
            PolymarketOwnedInventory(
                market_id=MARKET_ID,
                token_id=TOKEN_ID,
                quantity=Decimal("1"),
                provisional=False,
            ),
        ),
        observed_at_ms=NOW_MS,
    )
    venue = FakeVenue()
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            redeemable=True,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    gate = coordinator.preflight()

    assert gate.can_open is False
    assert "unknown_redemption_state" in gate.errors


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


def test_stream_fill_accounting_upgrades_once_and_blocks_open_until_verified(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "fill-accounting.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    verified = _fill(intent, prepared.expected_order_id)
    stream_only = replace(
        verified,
        fee_rate=None,
        fee_exponent=None,
        fee_quote=None,
        fee_schedule_sha256="",
    )
    ledger.record_fill(stream_only)
    ledger.transition(
        intent.intent_id,
        expected_states=("live",),
        state="filled",
        observed_at_ms=NOW_MS + 10,
        remote_status="CONFIRMED",
        matched_quantity=Decimal("1"),
    )
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            redeemable=False,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    blocked = coordinator.reconcile()

    assert ledger.unverified_fill_accounting_count() == 1
    assert blocked.can_open is False
    assert blocked.can_close is True
    assert "unverified_fill_accounting" in blocked.errors

    ledger.record_fill(verified)
    admitted = coordinator.reconcile()

    assert ledger.unverified_fill_accounting_count() == 0
    assert "unverified_fill_accounting" not in admitted.errors
    with pytest.raises(PolymarketLiveBlocked, match="fee accounting differs"):
        ledger.record_fill(
            replace(
                verified,
                fee_rate=Decimal("0.07"),
                fee_quote=Decimal("0.01"),
                fee_schedule_sha256="b" * 64,
            )
        )


def test_v1_fill_ledger_migrates_without_inventing_fee_accounting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-live.sqlite3"
    ledger = PolymarketLiveOrderLedger(path)
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    ledger.record_fill(_fill(intent, prepared.expected_order_id))

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            """
            CREATE TABLE polymarket_live_fills_v1 (
                trade_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT NOT NULL,
                status TEXT NOT NULL,
                observed_at_ms INTEGER NOT NULL,
                fill_sha256 TEXT NOT NULL,
                PRIMARY KEY (trade_id, order_id)
            )
            """
        )
        for row in connection.execute(
            "SELECT * FROM polymarket_live_fills"
        ).fetchall():
            payload = {
                "trade_id": str(row["trade_id"]),
                "order_id": str(row["order_id"]),
                "market_id": str(row["market_id"]),
                "token_id": str(row["token_id"]),
                "side": str(row["side"]),
                "quantity": str(row["quantity"]),
                "price": str(row["price"]),
                "status": str(row["status"]),
                "observed_at_ms": int(row["observed_at_ms"]),
            }
            payload_json = json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            connection.execute(
                """
                INSERT INTO polymarket_live_fills_v1
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    *payload.values(),
                    hashlib.sha256(payload_json.encode("ascii")).hexdigest(),
                ],
            )
        connection.execute("DROP TABLE polymarket_live_fills")
        connection.execute(
            "ALTER TABLE polymarket_live_fills_v1 "
            "RENAME TO polymarket_live_fills"
        )
        connection.execute(
            """
            UPDATE polymarket_live_metadata SET value = 'polymarket-live-ledger-v1'
            WHERE key = 'schema_version'
            """
        )
        connection.commit()
    finally:
        connection.close()

    migrated = PolymarketLiveOrderLedger(path)

    assert migrated.unverified_fill_accounting_count() == 1
    assert migrated.order_fill_evidence(
        prepared.expected_order_id
    ).all_active_fills_confirmed
    connection = sqlite3.connect(path)
    try:
        version = connection.execute(
            """
            SELECT value FROM polymarket_live_metadata
            WHERE key = 'schema_version'
            """
        ).fetchone()
        row = connection.execute(
            """
            SELECT role, accounting_state, fee_rate, fee_exponent, fee_quote
            FROM polymarket_live_fills
            """
        ).fetchone()
    finally:
        connection.close()
    assert version == ("polymarket-live-ledger-v3",)
    assert row == ("UNKNOWN", "UNKNOWN", "", 0, "")


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


def test_aggregate_capital_at_risk_blocks_repeated_small_orders(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "aggregate-risk.sqlite3")
    venue = FakeVenue()
    limits = PolymarketLiveRiskLimits(
        maximum_order_quote=Decimal("1"),
        maximum_token_quantity=Decimal("20"),
        maximum_total_at_risk_quote=Decimal("1"),
        maximum_active_markets=1,
        maximum_intent_age_ms=30_000,
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=limits)
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

    with pytest.raises(PolymarketLiveBlocked, match="capital-at-risk"):
        coordinator.submit(
            _intent(2),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    assert venue.submit_calls == 0


def test_active_market_ceiling_blocks_cross_contract_accumulation(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "active-markets.sqlite3")
    venue = FakeVenue()
    coordinator = PolymarketLiveCoordinator(
        venue,
        ledger,
        risk_limits=RISK_LIMITS,
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
    other_market = replace(
        _intent(2),
        market_id="0x" + "2" * 64,
        token_id="2" * 40,
    )

    with pytest.raises(PolymarketLiveBlocked, match="active-market ceiling"):
        coordinator.submit(
            other_market,
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    assert venue.submit_calls == 0


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


def test_owned_lots_bind_close_capacity_to_exact_parent(tmp_path: Path) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "parent-lots.sqlite3")
    venue = FakeVenue()
    first = _intent(1, quantity="0.4")
    second = _intent(2, quantity="1")
    first_prepared = _seed_live_order(ledger, venue, first)
    second_prepared = _seed_live_order(ledger, venue, second)
    ledger.record_fill(
        _fill(first, first_prepared.expected_order_id, quantity="0.4", seed=1)
    )
    ledger.record_fill(
        _fill(second, second_prepared.expected_order_id, quantity="1", seed=2)
    )
    for intent in (first, second):
        ledger.transition(
            intent.intent_id,
            expected_states=("live",),
            state="filled",
            observed_at_ms=NOW_MS + 10,
            remote_status="MATCHED",
            matched_quantity=intent.quantity,
        )
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1.4"),
            redeemable=False,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)
    close = replace(
        _intent(3, quantity="0.5", side="SELL", closing_only=True),
        parent_intent_id=first.intent_id,
    )

    with pytest.raises(PolymarketLiveBlocked, match="unreserved bot-owned lot"):
        coordinator.submit(close, tick_size=Decimal("0.01"), neg_risk=False)

    assert ledger.owned_lots() == (
        PolymarketOwnedLot(
            parent_intent_id=first.intent_id,
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("0.4"),
            reserved_close_quantity=Decimal("0"),
            provisional=False,
        ),
        PolymarketOwnedLot(
            parent_intent_id=second.intent_id,
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            reserved_close_quantity=Decimal("0"),
            provisional=False,
        ),
    )


def test_open_close_reserves_parent_lot_and_cancellation_releases_it(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "close-reservation.sqlite3")
    venue = FakeVenue()
    parent = _intent(1)
    prepared = _seed_live_order(ledger, venue, parent)
    ledger.record_fill(_fill(parent, prepared.expected_order_id))
    ledger.transition(
        parent.intent_id,
        expected_states=("live",),
        state="filled",
        observed_at_ms=NOW_MS + 10,
        remote_status="MATCHED",
        matched_quantity=Decimal("1"),
    )
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("1"),
            redeemable=False,
        ),
    )
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)
    first_close = replace(
        _intent(2, quantity="0.6", side="SELL", closing_only=True),
        parent_intent_id=parent.intent_id,
    )

    submitted = coordinator.submit(
        first_close,
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )
    lot = ledger.owned_lots()[0]
    assert lot.reserved_close_quantity == Decimal("0.6")
    assert lot.available_quantity == Decimal("0.4")

    second_close = replace(
        _intent(3, quantity="0.5", side="SELL", closing_only=True),
        parent_intent_id=parent.intent_id,
    )
    with pytest.raises(PolymarketLiveBlocked, match="unreserved bot-owned lot"):
        coordinator.submit(
            second_close,
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )

    ledger.transition(
        submitted.intent.intent_id,
        expected_states=("live",),
        state="cancelled",
        observed_at_ms=NOW_MS + 20,
        remote_status="CANCELED",
    )
    assert ledger.owned_lots()[0].available_quantity == Decimal("1")


def test_partial_close_fill_is_provisional_and_reserves_only_remainder(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "partial-close-lot.sqlite3")
    venue = FakeVenue()
    parent = _intent(1)
    parent_prepared = _seed_live_order(ledger, venue, parent)
    ledger.record_fill(_fill(parent, parent_prepared.expected_order_id))
    ledger.transition(
        parent.intent_id,
        expected_states=("live",),
        state="filled",
        observed_at_ms=NOW_MS + 10,
        remote_status="MATCHED",
        matched_quantity=Decimal("1"),
    )
    close = replace(
        _intent(2, quantity="0.6", side="SELL", closing_only=True),
        parent_intent_id=parent.intent_id,
    )
    close_prepared = _seed_live_order(ledger, venue, close)
    ledger.record_fill(
        _fill(
            close,
            close_prepared.expected_order_id,
            quantity="0.2",
            seed=2,
            status="MATCHED",
        )
    )

    provisional = ledger.owned_lots()[0]
    assert provisional.quantity == Decimal("0.8")
    assert provisional.reserved_close_quantity == Decimal("0.4")
    assert provisional.provisional is True
    assert provisional.available_quantity == Decimal("0")

    ledger.record_fill(
        _fill(
            close,
            close_prepared.expected_order_id,
            quantity="0.2",
            seed=2,
            status="CONFIRMED",
        )
    )
    confirmed = ledger.owned_lots()[0]
    assert confirmed.quantity == Decimal("0.8")
    assert confirmed.reserved_close_quantity == Decimal("0.4")
    assert confirmed.provisional is False
    assert confirmed.available_quantity == Decimal("0.4")


def test_owned_close_uses_fresh_polymarket_quote_for_exact_parent_lot(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "owned-close.sqlite3")
    venue = FakeVenue()
    parent = _intent(1, quantity="5")
    prepared = _seed_live_order(ledger, venue, parent)
    ledger.record_fill(_fill(parent, prepared.expected_order_id, quantity="5"))
    ledger.transition(
        parent.intent_id,
        expected_states=("live",),
        state="filled",
        observed_at_ms=NOW_MS + 10,
        remote_status="MATCHED",
        matched_quantity=Decimal("5"),
    )
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            redeemable=False,
        ),
    )
    observed = int(time.time() * 1_000)
    venue.close_quotes = [
        PolymarketCloseQuote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            limit_price=Decimal("0.47"),
            average_price=Decimal("0.47"),
            fee_quote=Decimal("0.08719"),
            net_quote=Decimal("2.26281"),
            fee_rate=Decimal("0.07"),
            fee_exponent=1,
            tick_size=Decimal("0.01"),
            minimum_order_size=Decimal("5"),
            neg_risk=True,
            source_time_ms=observed - 20,
            observed_at_ms=observed,
            book_payload_sha256="b" * 64,
        )
    ]
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    closes = coordinator.submit_owned_close_orders(maximum_book_age_ms=500)

    assert venue.close_quote_calls == [(MARKET_ID, TOKEN_ID, Decimal("5"), 500)]
    assert len(closes) == 1
    close = closes[0]
    assert close.intent.side == "SELL"
    assert close.intent.closing_only is True
    assert close.intent.parent_intent_id == parent.intent_id
    assert close.intent.quantity == Decimal("5")
    assert close.intent.limit_price == Decimal("0.47")
    assert close.intent.fee_reserve_quote == Decimal("0.08719")
    assert close.intent.order_type == "FAK"
    assert close.state == "live"


def test_owned_close_rejects_stale_or_mismatched_quote_without_signing(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "bad-close-quote.sqlite3")
    venue = FakeVenue()
    parent = _intent(1, quantity="5")
    prepared = _seed_live_order(ledger, venue, parent)
    ledger.record_fill(_fill(parent, prepared.expected_order_id, quantity="5"))
    ledger.transition(
        parent.intent_id,
        expected_states=("live",),
        state="filled",
        observed_at_ms=NOW_MS + 10,
        remote_status="MATCHED",
        matched_quantity=Decimal("5"),
    )
    venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            redeemable=False,
        ),
    )
    observed = int(time.time() * 1_000)
    venue.close_quotes = [
        PolymarketCloseQuote(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            limit_price=Decimal("0.47"),
            average_price=Decimal("0.47"),
            fee_quote=Decimal("0"),
            net_quote=Decimal("2.35"),
            fee_rate=Decimal("0"),
            fee_exponent=1,
            tick_size=Decimal("0.01"),
            minimum_order_size=Decimal("5"),
            neg_risk=False,
            source_time_ms=observed - 2_000,
            observed_at_ms=observed,
            book_payload_sha256="c" * 64,
        )
    ]
    coordinator = PolymarketLiveCoordinator(venue, ledger, risk_limits=RISK_LIMITS)

    with pytest.raises(PolymarketLiveBlocked, match="freshness"):
        coordinator.submit_owned_close_orders(maximum_book_age_ms=500)

    assert venue.submit_calls == 0


def test_owned_close_validates_bounds_gate_prior_orders_and_provisional_lots(
    tmp_path: Path,
) -> None:
    empty = PolymarketLiveCoordinator(
        FakeVenue(),
        PolymarketLiveOrderLedger(tmp_path / "empty-close.sqlite3"),
        risk_limits=RISK_LIMITS,
    )
    with pytest.raises(ValueError, match="maximum_book_age_ms"):
        empty.submit_owned_close_orders(maximum_book_age_ms=99)

    foreign_venue = FakeVenue()
    foreign_venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id="2" * 40,
            quantity=Decimal("1"),
            redeemable=False,
        ),
    )
    foreign = PolymarketLiveCoordinator(
        foreign_venue,
        PolymarketLiveOrderLedger(tmp_path / "foreign-close.sqlite3"),
        risk_limits=RISK_LIMITS,
    )
    with pytest.raises(PolymarketLiveBlocked, match="foreign_positions"):
        foreign.submit_owned_close_orders()

    open_ledger = PolymarketLiveOrderLedger(tmp_path / "open-close.sqlite3")
    open_venue = FakeVenue()
    open_intent = _intent()
    open_prepared = _seed_live_order(open_ledger, open_venue, open_intent)
    open_venue.open = (_remote_order(open_intent, open_prepared.expected_order_id),)
    open_coordinator = PolymarketLiveCoordinator(
        open_venue,
        open_ledger,
        risk_limits=RISK_LIMITS,
    )
    with pytest.raises(PolymarketLiveBlocked, match="prior bot orders"):
        open_coordinator.submit_owned_close_orders()

    provisional_ledger = PolymarketLiveOrderLedger(
        tmp_path / "provisional-close.sqlite3"
    )
    provisional_venue = FakeVenue()
    parent = _intent(1, quantity="5")
    prepared = _seed_live_order(provisional_ledger, provisional_venue, parent)
    provisional_ledger.record_fill(
        _fill(
            parent,
            prepared.expected_order_id,
            quantity="5",
            status="MATCHED",
        )
    )
    provisional_ledger.transition(
        parent.intent_id,
        expected_states=("live",),
        state="filled",
        observed_at_ms=NOW_MS + 10,
        remote_status="MATCHED",
        matched_quantity=Decimal("5"),
    )
    provisional_venue.remote_positions = (
        PolymarketRemotePosition(
            market_id=MARKET_ID,
            token_id=TOKEN_ID,
            quantity=Decimal("5"),
            redeemable=False,
        ),
    )
    provisional = PolymarketLiveCoordinator(
        provisional_venue,
        provisional_ledger,
        risk_limits=RISK_LIMITS,
    )
    with pytest.raises(PolymarketLiveBlocked, match="confirmed unreserved"):
        provisional.submit_owned_close_orders()


def test_reserving_same_intent_id_with_different_order_hash_is_blocked(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "intent-rebind.sqlite3")
    venue = FakeVenue()
    prepared = venue.prepare_order(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )
    ledger.reserve(prepared, observed_at_ms=NOW_MS)

    with pytest.raises(PolymarketLiveBlocked, match="bound differently"):
        ledger.reserve(
            replace(prepared, expected_order_id=_order_id(999)),
            observed_at_ms=NOW_MS + 1,
        )


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


def test_durable_stop_is_ordered_after_final_network_dispatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live.sqlite3"
    ledger = PolymarketLiveOrderLedger(path)
    venue = FakeVenue()
    control = PolymarketRuntimeControl(path)
    lease = control.acquire(owner_process_id=101)
    guard = PolymarketLiveRuntimeGuard(
        opening_interlock=PolymarketRuntimeLeaseInterlock(control, lease)
    )
    coordinator = PolymarketLiveCoordinator(
        venue,
        ledger,
        risk_limits=RISK_LIMITS,
        runtime_authority=guard,
    )
    guard.note_stream_liveness()
    coordinator.preflight()
    stop_started = Event()
    stop_finished = Event()
    stopper: Thread | None = None

    def request_stop() -> None:
        stop_started.set()
        control.request_stop(reason="operator_stop")
        stop_finished.set()

    def before_submit(_prepared: PolymarketPreparedOrder) -> None:
        nonlocal stopper
        stopper = Thread(target=request_stop)
        stopper.start()
        assert stop_started.wait(timeout=5)
        assert not stop_finished.wait(timeout=0.1)

    venue.before_submit = before_submit
    record = coordinator.submit(
        _intent(),
        tick_size=Decimal("0.01"),
        neg_risk=False,
    )
    assert stopper is not None
    stopper.join(timeout=5)

    assert record.state == "live"
    assert not stopper.is_alive()
    assert stop_finished.is_set()
    assert control.snapshot().state == "stop_requested"
    calls = venue.submit_calls
    with pytest.raises(PolymarketLiveBlocked, match="does not permit"):
        coordinator.submit(
            _intent(seed=2),
            tick_size=Decimal("0.01"),
            neg_risk=False,
        )
    assert venue.submit_calls == calls


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
        "trader_side": "TAKER",
        "fee_rate_bps": "0",
        "transaction_hash": "0x" + "d" * 64,
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
    assert ledger.unverified_fill_accounting_count() == 1
    assert guard.snapshot().hard_faults == ()


def test_user_stream_cancellation_with_unresolved_match_remains_blocking(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "matched-cancel.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    guard = PolymarketLiveRuntimeGuard()
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    cancellation = {
        "event_type": "order",
        "type": "CANCELLATION",
        "id": prepared.expected_order_id,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "side": "BUY",
        "original_size": "1",
        "size_matched": "0.25",
        "timestamp": str(int(time.time())),
    }

    assert consumer.handle(json.dumps(cancellation)) == 1

    record = ledger.record(intent.intent_id)
    assert record.state == "matched_pending"
    assert record.blocks_new_exposure is True
    assert record.matched_quantity == Decimal("0.25")
    assert guard.snapshot().hard_faults == ()


def test_user_stream_cancellation_closes_only_after_exact_confirmed_fill(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "confirmed-cancel.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    ledger.record_fill(
        _fill(
            intent,
            prepared.expected_order_id,
            quantity="0.25",
            status="CONFIRMED",
        )
    )
    guard = PolymarketLiveRuntimeGuard()
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    cancellation = {
        "event_type": "order",
        "type": "CANCELLATION",
        "id": prepared.expected_order_id,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "side": "BUY",
        "original_size": "1",
        "size_matched": "0.25",
        "timestamp": str(int(time.time())),
    }

    assert consumer.handle(json.dumps(cancellation)) == 1

    record = ledger.record(intent.intent_id)
    assert record.state == "filled"
    assert record.matched_quantity == Decimal("0.25")
    assert guard.snapshot().hard_faults == ()


def test_user_stream_never_regresses_owned_matched_quantity(
    tmp_path: Path,
) -> None:
    ledger = PolymarketLiveOrderLedger(tmp_path / "regressed-match.sqlite3")
    venue = FakeVenue()
    intent = _intent()
    prepared = _seed_live_order(ledger, venue, intent)
    ledger.transition(
        intent.intent_id,
        expected_states=("live",),
        state="partial",
        observed_at_ms=NOW_MS + 2,
        remote_status="UPDATE",
        matched_quantity=Decimal("0.50"),
    )
    guard = PolymarketLiveRuntimeGuard()
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    regressed = {
        "event_type": "order",
        "type": "UPDATE",
        "id": prepared.expected_order_id,
        "market": MARKET_ID,
        "asset_id": TOKEN_ID,
        "side": "BUY",
        "original_size": "1",
        "size_matched": "0.25",
        "timestamp": str(int(time.time())),
    }

    assert consumer.handle(json.dumps(regressed)) == 1

    record = ledger.record(intent.intent_id)
    assert record.state == "partial"
    assert record.matched_quantity == Decimal("0.50")
    assert guard.snapshot().hard_faults == (
        "owned_order_stream_matched_quantity_regressed",
    )


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
