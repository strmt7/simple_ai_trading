from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest

from simple_ai_trading.paper_execution import (
    AggressiveTradePrint,
    BinancePaperExecutionAdapter,
    BinanceBpsFeeModel,
    BookLevel,
    PAPER_EXECUTION_SCHEMA_VERSION,
    PaperBookSnapshot,
    PaperOrderIntent,
    PaperOrderJournal,
    PaperOrderTransition,
    PassiveQueueState,
    PolymarketFeeModel,
    PolymarketPaperExecutionAdapter,
    apply_passive_trade_print,
    assert_shared_venue_execution_contract,
    binance_book_ticker_snapshot,
    paper_intent_id,
    simulate_aggressive_order,
)


SOURCE_SHA = "a" * 64


def _intent(**overrides: object) -> PaperOrderIntent:
    values: dict[str, object] = {
        "intent_id": "paper-order-1",
        "venue": "polymarket",
        "market_id": "condition-1",
        "asset_id": "token-1",
        "symbol": "BTC",
        "outcome": "Up",
        "side": "BUY",
        "order_type": "FAK",
        "limit_price": Decimal("0.55"),
        "quantity": Decimal("10"),
        "created_at_ms": 1_000,
        "expires_at_ms": 10_000,
    }
    values.update(overrides)
    return PaperOrderIntent(**values)


def _transition(
    event_id: str, state: str, at_ms: int, **overrides: object
) -> PaperOrderTransition:
    values: dict[str, object] = {
        "event_id": event_id,
        "state": state,
        "occurred_at_ms": at_ms,
    }
    values.update(overrides)
    return PaperOrderTransition(**values)


def _book(
    *, asks: tuple[BookLevel, ...], bids: tuple[BookLevel, ...] = ()
) -> PaperBookSnapshot:
    return PaperBookSnapshot(
        venue="polymarket",
        market_id="condition-1",
        asset_id="token-1",
        bids=bids,
        asks=asks,
        source_time_ms=1_600,
        received_wall_ms=1_700,
        received_monotonic_ns=1,
        source_payload_sha256=SOURCE_SHA,
    )


def test_journal_is_idempotent_hash_chained_and_reconciliation_gated() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    intent = _intent()

    first = journal.record_intent(intent)
    journal.record_intent(intent)
    submitted = _transition("event-submit", "SUBMITTED", 1_100)
    journal.transition(intent.intent_id, submitted)
    unknown = _transition("event-unknown", "UNKNOWN", 1_200, reason="timeout")
    journal.transition(intent.intent_id, unknown)

    with pytest.raises(ValueError, match="only reconciliation"):
        journal.transition(
            intent.intent_id,
            _transition("event-ack-invalid", "ACKNOWLEDGED", 1_300),
        )

    journal.transition(
        intent.intent_id,
        _transition(
            "event-ack-reconciled",
            "ACKNOWLEDGED",
            1_300,
            source="reconciliation",
        ),
    )
    replayed = journal.transition(intent.intent_id, submitted)

    assert first.state == "INTENT"
    assert replayed.state == "SUBMITTED"
    assert journal.current(intent.intent_id).state == "ACKNOWLEDGED"
    assert journal.integrity_errors() == ()
    assert (
        connection.execute("SELECT count(*) FROM paper_order_event").fetchone()[0] == 4
    )


def test_journal_rejects_mutated_idempotency_key_and_overfill() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    intent = _intent()
    journal.record_intent(intent)
    journal.transition(intent.intent_id, _transition("submit", "SUBMITTED", 1_100))

    with pytest.raises(ValueError, match="different immutable payload"):
        journal.transition(
            intent.intent_id, _transition("submit", "ACKNOWLEDGED", 1_100)
        )
    with pytest.raises(ValueError, match="exceeds order quantity"):
        journal.transition(
            intent.intent_id,
            _transition(
                "overfill",
                "FILLED",
                1_200,
                cumulative_filled_quantity=Decimal("11"),
                average_fill_price=Decimal("0.5"),
            ),
        )


def test_shared_aggressive_simulator_walks_depth_and_keeps_close_remainder_visible() -> (
    None
):
    intent = _intent(side="SELL", order_type="FAK", limit_price=Decimal("0.40"))
    book = _book(
        asks=(),
        bids=(
            BookLevel(Decimal("0.50"), Decimal("4")),
            BookLevel(Decimal("0.45"), Decimal("3")),
        ),
    )
    fee_model = PolymarketFeeModel(
        enabled=True,
        rate=Decimal("0.07"),
        exponent=1,
        taker_only=True,
    )

    result = simulate_aggressive_order(
        intent,
        book,
        execution_time_ms=1_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
        fee=fee_model,
        owned_quantity=Decimal("10"),
        closing_position=True,
    )

    assert result.state == "CLOSE_PENDING"
    assert result.filled_quantity == Decimal("7")
    assert result.remaining_quantity == Decimal("3")
    assert result.average_fill_price == Decimal("0.4785714285714285714285714286")
    assert result.fee_quote == Decimal("0.12198")


def test_shared_aggressive_simulator_rejects_external_inventory_and_zero_latency() -> (
    None
):
    intent = _intent(side="SELL", limit_price=Decimal("0.40"))
    book = _book(asks=(), bids=(BookLevel(Decimal("0.50"), Decimal("20")),))
    fee_model = BinanceBpsFeeModel(Decimal("2"), Decimal("5"))

    external = simulate_aggressive_order(
        intent,
        book,
        execution_time_ms=1_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
        fee=fee_model,
        owned_quantity=Decimal("0"),
        closing_position=True,
    )
    zero_latency = simulate_aggressive_order(
        intent,
        book,
        execution_time_ms=1_800,
        submission_latency_ms=0,
        maximum_book_age_ms=500,
        fee=fee_model,
    )

    assert external.state == "REJECTED"
    assert external.reason == "order_exceeds_bot_owned_inventory"
    assert zero_latency.state == "REJECTED"
    assert zero_latency.reason == "zero_or_negative_latency_prohibited"


def test_passive_queue_ignores_arrival_wrong_side_and_wrong_price() -> None:
    state = PassiveQueueState(
        intent_id="passive-1",
        asset_id="token-1",
        side="BUY",
        price=Decimal("0.50"),
        queue_ahead_quantity=Decimal("5"),
        remaining_quantity=Decimal("3"),
        filled_quantity=Decimal("0"),
        activated_at_ms=1_000,
        expires_at_ms=5_000,
    )

    for occurred, side, price in (
        (1_000, "SELL", "0.50"),
        (1_100, "BUY", "0.50"),
        (1_100, "SELL", "0.51"),
    ):
        state, filled = apply_passive_trade_print(
            state,
            AggressiveTradePrint(
                asset_id="token-1",
                side=side,
                price=Decimal(price),
                quantity=Decimal("100"),
                occurred_at_ms=occurred,
                source_payload_sha256=SOURCE_SHA,
            ),
        )
        assert filled == 0

    state, first_fill = apply_passive_trade_print(
        state,
        AggressiveTradePrint(
            asset_id="token-1",
            side="SELL",
            price=Decimal("0.50"),
            quantity=Decimal("6"),
            occurred_at_ms=1_200,
            source_payload_sha256=SOURCE_SHA,
        ),
    )
    state, second_fill = apply_passive_trade_print(
        state,
        AggressiveTradePrint(
            asset_id="token-1",
            side="SELL",
            price=Decimal("0.50"),
            quantity=Decimal("2"),
            occurred_at_ms=1_300,
            source_payload_sha256=SOURCE_SHA,
        ),
    )

    assert first_fill == Decimal("1")
    assert second_fill == Decimal("2")
    assert state.queue_ahead_quantity == 0
    assert state.remaining_quantity == 0


def test_venue_adapters_must_publish_the_shared_contract_identity() -> None:
    class Adapter:
        execution_schema_version = PAPER_EXECUTION_SCHEMA_VERSION

    assert_shared_venue_execution_contract((Adapter(), Adapter()))
    with pytest.raises(ValueError, match="shared paper execution"):
        assert_shared_venue_execution_contract((object(),))


def test_shared_adapter_derives_inventory_and_keeps_partial_close_blocking() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    adapter = BinancePaperExecutionAdapter(
        journal,
        market_type="futures",
        maker_fee_bps=Decimal("10"),
        taker_fee_bps=Decimal("10"),
    )
    open_id = paper_intent_id("binance-futures", "position-1", "open")
    opening = PaperOrderIntent(
        intent_id=open_id,
        venue="binance-futures",
        market_id="BTCUSDT",
        asset_id="BTCUSDT",
        symbol="BTCUSDT",
        outcome="LONG",
        side="BUY",
        order_type="FAK",
        limit_price=Decimal("101"),
        quantity=Decimal("2"),
        created_at_ms=1_000,
        expires_at_ms=5_000,
    )
    open_book = binance_book_ticker_snapshot(
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99",
            "bidQty": "4",
            "askPrice": "100",
            "askQty": "2",
        },
        market_type="futures",
        symbol="BTCUSDT",
        received_wall_ms=1_700,
        received_monotonic_ns=10,
    )
    opened = adapter.execute_aggressive(
        opening,
        open_book,
        execution_time_ms=1_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
    )

    first_close = PaperOrderIntent(
        intent_id=paper_intent_id("binance-futures", "position-1", "close", attempt=1),
        venue="binance-futures",
        market_id="BTCUSDT",
        asset_id="BTCUSDT",
        symbol="BTCUSDT",
        outcome="LONG",
        side="SELL",
        order_type="FAK",
        limit_price=Decimal("98"),
        quantity=Decimal("2"),
        created_at_ms=2_000,
        expires_at_ms=5_000,
        parent_inventory_id=open_id,
    )
    partial_book = binance_book_ticker_snapshot(
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99",
            "bidQty": "1",
            "askPrice": "100",
            "askQty": "5",
        },
        market_type="futures",
        symbol="BTCUSDT",
        received_wall_ms=2_700,
        received_monotonic_ns=20,
    )
    partial = adapter.execute_aggressive(
        first_close,
        partial_book,
        execution_time_ms=2_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
        closing_position=True,
    )
    partial_report = journal.reconcile("binance-futures")

    assert opened.state == "FILLED"
    assert partial.state == "CLOSE_PENDING"
    assert journal.current(first_close.intent_id).terminal is False
    assert partial_report.can_open is False
    assert partial_report.can_close is True
    assert partial_report.inventory[0].remaining_quantity == Decimal("1")
    assert partial_report.blocking_intent_ids == (first_close.intent_id,)

    second_close = PaperOrderIntent(
        **{
            **first_close.__dict__,
            "intent_id": paper_intent_id(
                "binance-futures", "position-1", "close", attempt=2
            ),
            "quantity": Decimal("1"),
            "created_at_ms": 3_000,
            "expires_at_ms": 6_000,
        }
    )
    final_book = binance_book_ticker_snapshot(
        {
            "symbol": "BTCUSDT",
            "bidPrice": "98.5",
            "bidQty": "1",
            "askPrice": "99",
            "askQty": "5",
            "E": 3_700,
        },
        market_type="futures",
        symbol="BTCUSDT",
        received_wall_ms=3_700,
        received_monotonic_ns=30,
    )
    final = adapter.execute_aggressive(
        second_close,
        final_book,
        execution_time_ms=3_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
        closing_position=True,
    )
    final_report = journal.reconcile("binance-futures")

    assert final.state == "FILLED"
    assert final_report.ok is True
    assert final_report.inventory[0].remaining_quantity == 0
    assert final_report.blocking_intent_ids == ()
    assert final_report.can_open is True
    assert journal.integrity_errors() == ()


def test_adapters_reject_external_inventory_naked_shorts_and_terminal_replay() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    binance = BinancePaperExecutionAdapter(
        journal,
        market_type="spot",
        maker_fee_bps=Decimal("10"),
        taker_fee_bps=Decimal("10"),
    )
    book = binance_book_ticker_snapshot(
        {
            "s": "BTCUSDT",
            "bidPrice": "99",
            "bidQty": "1",
            "askPrice": "100",
            "askQty": "1",
        },
        market_type="spot",
        symbol="BTCUSDT",
        received_wall_ms=1_700,
        received_monotonic_ns=1,
    )
    naked_short = PaperOrderIntent(
        intent_id="spot-short",
        venue="binance-spot",
        market_id="BTCUSDT",
        asset_id="BTCUSDT",
        symbol="BTCUSDT",
        outcome="SHORT",
        side="SELL",
        order_type="FAK",
        limit_price=Decimal("99"),
        quantity=Decimal("1"),
        created_at_ms=1_000,
        expires_at_ms=5_000,
    )
    with pytest.raises(ValueError, match="naked short"):
        binance.execute_aggressive(
            naked_short,
            book,
            execution_time_ms=1_800,
            submission_latency_ms=750,
            maximum_book_age_ms=500,
        )
    with pytest.raises(KeyError, match="unknown paper intent"):
        journal.current("spot-short")

    polymarket = PolymarketPaperExecutionAdapter(
        journal,
        fee=PolymarketFeeModel(True, Decimal("0.07"), 1, True),
    )
    polymarket_sell = _intent(intent_id="poly-short", side="SELL")
    with pytest.raises(ValueError, match="naked token shorts"):
        polymarket.execute_aggressive(
            polymarket_sell,
            _book(asks=(), bids=(BookLevel(Decimal("0.5"), Decimal("10")),)),
            execution_time_ms=1_800,
            submission_latency_ms=750,
            maximum_book_age_ms=500,
        )

    opening = PaperOrderIntent(
        **{
            **naked_short.__dict__,
            "intent_id": "spot-open",
            "outcome": "LONG",
            "side": "BUY",
            "limit_price": Decimal("101"),
        }
    )
    binance.execute_aggressive(
        opening,
        book,
        execution_time_ms=1_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
    )
    with pytest.raises(ValueError, match="already active or resolved"):
        binance.execute_aggressive(
            opening,
            book,
            execution_time_ms=1_800,
            submission_latency_ms=750,
            maximum_book_age_ms=500,
        )
    assert journal.reconcile().inventory[0].remaining_quantity == Decimal("1")


def test_close_parent_is_verified_before_close_intent_is_recorded() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    adapter = BinancePaperExecutionAdapter(
        journal,
        market_type="futures",
        maker_fee_bps=Decimal("0"),
        taker_fee_bps=Decimal("0"),
    )
    close = PaperOrderIntent(
        intent_id="external-close",
        venue="binance-futures",
        market_id="BTCUSDT",
        asset_id="BTCUSDT",
        symbol="BTCUSDT",
        outcome="LONG",
        side="SELL",
        order_type="FAK",
        limit_price=Decimal("90"),
        quantity=Decimal("1"),
        created_at_ms=1_000,
        expires_at_ms=5_000,
        parent_inventory_id="not-owned",
    )
    book = binance_book_ticker_snapshot(
        {
            "symbol": "BTCUSDT",
            "bidPrice": "99",
            "bidQty": "1",
            "askPrice": "100",
            "askQty": "1",
        },
        market_type="futures",
        symbol="BTCUSDT",
        received_wall_ms=1_700,
        received_monotonic_ns=1,
    )

    with pytest.raises(KeyError, match="unknown paper intent"):
        adapter.execute_aggressive(
            close,
            book,
            execution_time_ms=1_800,
            submission_latency_ms=750,
            maximum_book_age_ms=500,
            closing_position=True,
        )
    assert (
        connection.execute("SELECT count(*) FROM paper_order_intent").fetchone()[0] == 0
    )


def test_official_settlement_closes_exact_remaining_inventory_without_fake_order() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    adapter = PolymarketPaperExecutionAdapter(
        journal,
        fee=PolymarketFeeModel(True, Decimal("0.07"), 1, True),
    )
    opening = _intent(
        intent_id="polymarket-opening",
        side="BUY",
        quantity=Decimal("5"),
        limit_price=Decimal("0.55"),
    )
    book = _book(
        bids=(BookLevel(Decimal("0.49"), Decimal("10")),),
        asks=(BookLevel(Decimal("0.50"), Decimal("10")),),
    )
    result = adapter.execute_aggressive(
        opening,
        book,
        execution_time_ms=1_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
    )
    assert result.state == "FILLED"

    settlement = journal.record_settlement(
        settlement_id="official-resolution-1",
        opening_intent_id=opening.intent_id,
        quantity=Decimal("5"),
        payout_per_unit=Decimal("1"),
        fee_quote=Decimal("0"),
        occurred_at_ms=6_000,
        source_event_id="market-resolved-event",
        source_payload_sha256="a" * 64,
    )
    report = journal.reconcile("polymarket")

    assert settlement.payout_per_unit == 1
    assert report.inventory[0].remaining_quantity == 0
    assert report.can_open is True
    assert journal.integrity_errors() == ()
    with pytest.raises(ValueError, match="no remaining quantity"):
        journal.record_settlement(
            settlement_id="official-resolution-2",
            opening_intent_id=opening.intent_id,
            quantity=Decimal("5"),
            payout_per_unit=Decimal("1"),
            fee_quote=Decimal("0"),
            occurred_at_ms=6_001,
            source_event_id="market-resolved-event-2",
            source_payload_sha256="b" * 64,
        )


def test_settlement_inventory_proof_detects_late_order_history_mutation() -> None:
    connection = duckdb.connect(":memory:")
    journal = PaperOrderJournal(connection)
    adapter = PolymarketPaperExecutionAdapter(
        journal,
        fee=PolymarketFeeModel(False, Decimal("0"), 1, True),
    )
    opening = _intent(
        intent_id="settled-opening",
        side="BUY",
        quantity=Decimal("5"),
        limit_price=Decimal("0.55"),
    )
    adapter.execute_aggressive(
        opening,
        _book(asks=(BookLevel(Decimal("0.50"), Decimal("5")),), bids=()),
        execution_time_ms=1_800,
        submission_latency_ms=750,
        maximum_book_age_ms=500,
    )
    journal.record_settlement(
        settlement_id="official-resolution",
        opening_intent_id=opening.intent_id,
        quantity=Decimal("5"),
        payout_per_unit=Decimal("0"),
        fee_quote=Decimal("0"),
        occurred_at_ms=6_000,
        source_event_id="resolution-event",
        source_payload_sha256="c" * 64,
    )
    journal.record_intent(
        PaperOrderIntent(
            intent_id="late-child",
            venue="polymarket",
            market_id=opening.market_id,
            asset_id=opening.asset_id,
            symbol=opening.symbol,
            outcome=opening.outcome,
            side="SELL",
            order_type="FAK",
            limit_price=Decimal("0.40"),
            quantity=Decimal("1"),
            created_at_ms=6_100,
            expires_at_ms=6_200,
            parent_inventory_id=opening.intent_id,
        )
    )

    errors = journal.integrity_errors()
    assert any(
        error.startswith("settlement_inventory_proof_mismatch:") for error in errors
    )
