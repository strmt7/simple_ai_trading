"""Recorded paper cash must retain partial losses and chronological lot costs."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import duckdb
import pytest

from simple_ai_trading.paper_execution import (
    AggressiveTradePrint,
    BinancePaperExecutionAdapter,
    BookLevel,
    PaperBookSnapshot,
    PaperOrderIntent,
    PaperOrderJournal,
    PaperOrderTransition,
    PassiveQueueState,
    apply_passive_trade_print,
)


def _intent(**overrides):
    values = dict(
        intent_id="opening",
        venue="binance-spot",
        market_id="BTCUSDT",
        asset_id="BTC",
        symbol="BTCUSDT",
        outcome="",
        side="BUY",
        order_type="GTC",
        limit_price=Decimal("100"),
        quantity=Decimal("10"),
        created_at_ms=1000,
        expires_at_ms=20000,
    )
    values.update(overrides)
    return PaperOrderIntent(**values)


def _submit(journal, intent):
    journal.record_intent(intent)
    journal.transition(
        intent.intent_id,
        PaperOrderTransition(
            intent.intent_id + "-submit",
            "SUBMITTED",
            intent.created_at_ms + 1,
            source_event_id="synthetic-submit",
            source_payload_sha256="a" * 64,
        ),
    )


def _fill(journal, intent, state, time, quantity, price, fee="0"):
    return journal.transition(
        intent.intent_id,
        PaperOrderTransition(
            f"{intent.intent_id}-{time}",
            state,
            time,
            Decimal(quantity),
            Decimal(price),
            Decimal(fee),
            source_event_id=f"synthetic-{time}",
            source_payload_sha256="b" * 64,
        ),
    )


def _partial(journal, **overrides):
    intent = _intent(**overrides)
    _submit(journal, intent)
    _fill(journal, intent, "PARTIAL", 2000, "4", str(intent.limit_price), ".12")
    return intent


def _close(
    journal, parent, *, quantity="4", price="98", time=4000, fee=".2352", name="close"
):
    close = replace(
        parent,
        intent_id=name,
        side="SELL",
        order_type="FAK",
        quantity=Decimal(quantity),
        limit_price=Decimal(price),
        created_at_ms=time - 100,
        parent_inventory_id=parent.intent_id,
    )
    _submit(journal, close)
    _fill(journal, close, "FILLED", time, quantity, price, fee)
    return close


def _cancel(journal, intent, time):
    current = journal.current(intent.intent_id)
    journal.transition(
        intent.intent_id,
        PaperOrderTransition(
            intent.intent_id + "-cancel",
            "CANCELLED",
            time,
            current.cumulative_filled_quantity,
            current.average_fill_price,
            current.cumulative_fee_quote,
        ),
    )


def _rewrite_latest(journal, intent, **changes):
    rows = journal.connection.execute(
        "SELECT event_id, sequence_number, previous_event_sha256 FROM paper_order_event "
        "WHERE intent_id=? ORDER BY sequence_number DESC LIMIT 1",
        [intent.intent_id],
    ).fetchone()
    current = journal.current(intent.intent_id)
    transition = PaperOrderTransition(
        rows[0],
        current.state,
        current.occurred_at_ms,
        current.cumulative_filled_quantity,
        current.average_fill_price,
        current.cumulative_fee_quote,
        source_event_id="synthetic-rewrite",
        source_payload_sha256="b" * 64,
    )
    transition = replace(transition, **changes)
    payload = journal._transition_payload(
        intent.intent_id, rows[1], rows[2], transition
    )
    journal.connection.execute(
        "UPDATE paper_order_event SET state=?, cumulative_filled_quantity=?, "
        "average_fill_price=?, cumulative_fee_quote=?, source_event_id=?, "
        "source_payload_sha256=?, event_sha256=? WHERE event_id=?",
        [
            payload[name]
            for name in (
                "state",
                "cumulative_filled_quantity",
                "average_fill_price",
                "cumulative_fee_quote",
                "source_event_id",
                "source_payload_sha256",
                "event_sha256",
                "event_id",
            )
        ],
    )


def test_virtual_partial_cash_reaches_owned_close_before_complete_fill_and_restart(
    tmp_path,
):
    path = str(tmp_path / "paper.duckdb")
    connection = duckdb.connect(path)
    journal = PaperOrderJournal(connection)
    opening = _intent()
    _submit(journal, opening)
    marker = PassiveQueueState(
        opening.intent_id,
        "BTC",
        "BUY",
        Decimal(100),
        Decimal(100),
        Decimal(10),
        Decimal(0),
        1100,
        20000,
    )
    marker, quantity = apply_passive_trade_print(
        marker,
        AggressiveTradePrint(
            "BTC",
            "SELL",
            Decimal(100),
            Decimal(104),
            2000,
            "b" * 64,
        ),
    )
    assert quantity == 4 and marker.remaining_quantity == 6
    _fill(journal, opening, "PARTIAL", 2000, "4", "100", ".12")
    first = journal.inventory_cash(opening.intent_id)
    assert first.requires_inventory_protection and first.first_fill_time_ms == 2000
    assert first.recorded_cash_change == Fraction("-400.12")
    assert first.remaining_entry_cost == Fraction("400.12")
    assert first.blocking_intent_ids == (opening.intent_id,)
    close = replace(
        opening,
        intent_id="close",
        side="SELL",
        order_type="FAK",
        quantity=Decimal(4),
        limit_price=Decimal(98),
        created_at_ms=3000,
        parent_inventory_id=opening.intent_id,
    )
    adapter = BinancePaperExecutionAdapter(
        journal,
        market_type="spot",
        maker_fee_bps=Decimal(3),
        taker_fee_bps=Decimal(6),
    )
    book = PaperBookSnapshot(
        "binance-spot",
        "BTCUSDT",
        "BTC",
        (BookLevel(Decimal(98), Decimal(4)),),
        (),
        3900,
        3900,
        1,
        "c" * 64,
    )
    result = adapter.execute_aggressive(
        close,
        book,
        execution_time_ms=4000,
        submission_latency_ms=100,
        maximum_book_age_ms=1000,
        closing_position=True,
    )
    assert result.state == "FILLED" and journal.current("opening").state == "PARTIAL"
    cash = journal.inventory_cash(opening.intent_id)
    assert cash.realized_pnl == cash.recorded_cash_change == Fraction("-8.3552")
    assert cash.remaining_quantity == cash.remaining_entry_cost == 0
    assert not cash.requires_inventory_protection
    assert cash.blocking_intent_ids == (opening.intent_id,)  # Six may still fill later.
    _cancel(journal, opening, 4001)
    final = journal.inventory_cash(opening.intent_id)
    assert not final.blocking_intent_ids and not final.native_cash_qualified
    assert not final.profitability_claim
    connection.close()
    with duckdb.connect(path) as restored:
        assert PaperOrderJournal(restored).inventory_cash(opening.intent_id) == final


def test_fifo_close_cost_does_not_use_later_opening_price_or_fee():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _intent(quantity=Decimal(2), limit_price=Decimal(200))
        _submit(journal, opening)
        _fill(journal, opening, "PARTIAL", 2000, "1", "100", "1")
        _close(journal, opening, quantity="1", price="110", time=3000, fee="2")
        _fill(journal, opening, "FILLED", 4000, "2", "150", "3")
        with localcontext() as context:
            context.prec = 3
            cash = journal.inventory_cash(opening.intent_id)
        assert cash.realized_pnl == 7
        assert cash.allocated_entry_cost == 101 and cash.remaining_entry_cost == 202
        assert cash.recorded_cash_change == -195 and cash.remaining_quantity == 1
        assert (
            cash.acquisition_cash_and_fees
            == cash.allocated_entry_cost + cash.remaining_entry_cost
        )
        assert cash.requires_inventory_protection


def test_multiple_partial_disposals_conserve_cost_and_fee_allocation():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _partial(journal)
        _close(
            journal, opening, quantity="1", price="101", time=3000, fee=".1", name="c1"
        )
        _close(
            journal, opening, quantity="2", price="99", time=4000, fee=".2", name="c2"
        )
        cash = journal.inventory_cash(opening.intent_id)
        assert cash.closed_quantity == 3 and cash.remaining_quantity == 1
        assert cash.allocated_entry_cost == Fraction("300.09")
        assert cash.remaining_entry_cost == Fraction("100.03")
        assert cash.realized_pnl == Fraction("-1.39")


@pytest.mark.parametrize("payout", ["0", "1"])
def test_token_sale_and_settlement_keep_distinct_quantity_cash(payout):
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _intent(
            venue="polymarket",
            market_id="condition",
            asset_id="token",
            symbol="BTC",
            outcome="YES",
            limit_price=Decimal(".5"),
        )
        _submit(journal, opening)
        _fill(journal, opening, "PARTIAL", 2000, "4", ".5", ".002")
        _cancel(journal, opening, 2001)
        _close(journal, opening, quantity="2", price=".6", time=3000, fee=".001")
        journal.record_settlement(
            settlement_id="synthetic-resolution",
            opening_intent_id=opening.intent_id,
            quantity="2",
            payout_per_unit=payout,
            fee_quote=".003",
            occurred_at_ms=4000,
            source_event_id="synthetic-resolution",
            source_payload_sha256="d" * 64,
        )
        cash = journal.inventory_cash(opening.intent_id)
        assert cash.closed_quantity == 4 and cash.remaining_quantity == 0
        assert cash.realized_pnl == Fraction("1.196") + 2 * Fraction(payout) - Fraction(
            "2.002"
        )
        assert (
            cash.settlement_sha256
            == journal.settlement(opening.intent_id).settlement_sha256
        )
        assert not cash.native_cash_qualified


@pytest.mark.parametrize("same_time", [False, True])
def test_close_before_or_at_opening_fill_is_not_causal(same_time):
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _intent()
        _submit(journal, opening)
        _close(journal, opening, time=2000, fee="0")
        _fill(journal, opening, "PARTIAL", 2000 if same_time else 3000, "4", "100")
        assert journal.reconcile().ok  # Final quantities alone miss causal ordering.
        with pytest.raises(ValueError, match="causal ordering|precedes"):
            journal.inventory_cash(opening.intent_id)


@pytest.mark.parametrize(
    "changes",
    [
        {"cumulative_filled_quantity": Decimal("10.0000000000001"), "state": "FILLED"},
        {"average_fill_price": Decimal("101")},
        {"average_fill_price": Decimal("0")},
        {"source_event_id": ""},
        {"source_payload_sha256": ""},
        {"cumulative_fee_quote": Decimal("-1")},
    ],
)
def test_recomputed_hash_does_not_qualify_invalid_cash_semantics(changes):
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _partial(journal)
        _rewrite_latest(journal, opening, **changes)
        assert journal.reconcile().ok
        with pytest.raises(ValueError):
            journal.inventory_cash(opening.intent_id)
        assert connection.execute("SELECT 1").fetchone() == (1,)


def test_raw_journal_tamper_rejects_before_cash_projection():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _partial(journal)
        connection.execute(
            "UPDATE paper_order_event SET cumulative_fee_quote='0' WHERE state='PARTIAL'"
        )
        with pytest.raises(ValueError, match="reconciliation"):
            journal.inventory_cash(opening.intent_id)


def test_small_overclose_inside_existing_tolerance_is_not_free_inventory():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _partial(journal)
        _close(journal, opening, quantity="4.0000000000001", fee="0")
        assert journal.reconcile().ok
        with pytest.raises(ValueError, match="exceeds owned"):
            journal.inventory_cash(opening.intent_id)


def test_increasing_quantity_with_regressing_cumulative_cash_rejects():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _partial(journal)
        _fill(journal, opening, "PARTIAL", 3000, "6", "60", ".12")
        assert journal.reconcile().ok
        with pytest.raises(ValueError, match="positive cash"):
            journal.inventory_cash(opening.intent_id)


def test_close_outcome_mismatch_is_not_the_same_token_inventory():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _partial(journal)
        close = replace(
            opening,
            intent_id="close",
            side="SELL",
            outcome="other",
            quantity=Decimal(4),
            parent_inventory_id=opening.intent_id,
        )
        _submit(journal, close)
        _fill(journal, close, "FILLED", 3000, "4", "100", "0")
        assert journal.reconcile().ok
        with pytest.raises(ValueError, match="differs from"):
            journal.inventory_cash(opening.intent_id)


@pytest.mark.parametrize(
    "changes", [{"side": "SELL"}, {"venue": "binance-futures"}, {"venue": "other"}]
)
def test_unimplemented_cash_products_are_not_silent_zero(changes):
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _intent(**changes)
        journal.record_intent(opening)
        with pytest.raises(ValueError, match="supports long"):
            journal.inventory_cash(opening.intent_id)


def test_unfilled_known_journal_is_zero_and_unknown_intent_is_not():
    with duckdb.connect(":memory:") as connection:
        journal = PaperOrderJournal(connection)
        opening = _intent()
        journal.record_intent(opening)
        cash = journal.inventory_cash(opening.intent_id)
        assert (
            cash.opened_quantity == cash.recorded_cash_change == cash.realized_pnl == 0
        )
        assert (
            cash.first_fill_time_ms is None and not cash.requires_inventory_protection
        )
        assert cash.blocking_intent_ids == (opening.intent_id,)
        with pytest.raises(KeyError):
            journal.inventory_cash("missing")
        assert connection.execute("SELECT 1").fetchone() == (1,)
