from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest

from simple_ai_trading.paper_execution import (
    AggressiveTradePrint,
    BinanceBpsFeeModel,
    BookLevel,
    PAPER_EXECUTION_SCHEMA_VERSION,
    PaperBookSnapshot,
    PaperOrderIntent,
    PaperOrderJournal,
    PaperOrderTransition,
    PassiveQueueState,
    PolymarketFeeModel,
    apply_passive_trade_print,
    assert_shared_venue_execution_contract,
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
