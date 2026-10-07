"""FIFO cash reconstruction from supplied paper events, not native account cash.

Rational arithmetic preserves recorded quantities, averages and quote fees. An
upstream rounded cumulative average cannot recover exact per-fill cash. Quote
currency, fee-asset conversions, venue origin and liquidation costs are not proven.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

from .paper_execution import PaperOrderSnapshot, PaperOrderTransition

if TYPE_CHECKING:
    from .paper_execution import PaperOrderIntent, PaperOrderJournal


@dataclass(frozen=True)
class PaperInventoryCash:
    opening_intent_id: str
    venue: str
    asset_id: str
    symbol: str
    opened_quantity: Fraction
    closed_quantity: Fraction
    remaining_quantity: Fraction
    acquisition_cash_and_fees: Fraction
    disposal_cash_after_fees: Fraction
    allocated_entry_cost: Fraction
    remaining_entry_cost: Fraction
    first_fill_time_ms: int | None
    blocking_intent_ids: tuple[str, ...]
    current_event_sha256_by_intent: tuple[tuple[str, str], ...]
    settlement_sha256: str | None
    native_cash_qualified: bool = False
    profitability_claim: bool = False

    @property
    def recorded_cash_change(self) -> Fraction:
        return self.disposal_cash_after_fees - self.acquisition_cash_and_fees

    @property
    def realized_pnl(self) -> Fraction:
        return self.disposal_cash_after_fees - self.allocated_entry_cost

    @property
    def requires_inventory_protection(self) -> bool:
        """Inventory needs protection from the first partial fill, not full fill."""
        return self.remaining_quantity > 0


@dataclass(frozen=True)
class _CashEvent:
    occurred_at_ms: int
    opening: bool
    quantity: Fraction
    cash: Fraction
    fee: Fraction


def _recorded_events(
    journal: PaperOrderJournal, intent: PaperOrderIntent
) -> list[_CashEvent]:
    rows = journal.connection.execute(
        """
        SELECT event_id, state, occurred_at_ms, cumulative_filled_quantity,
               average_fill_price, cumulative_fee_quote, reason, source,
               source_event_id, source_payload_sha256, event_sha256, sequence_number
        FROM paper_order_event WHERE intent_id = ? ORDER BY sequence_number
        """,
        [intent.intent_id],
    ).fetchall()
    events = []
    previous = None
    old_quantity = old_cash = old_fee = Fraction(0)
    for row in rows:
        snapshot = PaperOrderSnapshot(
            intent.intent_id,
            str(row[1]),
            int(row[11]),
            int(row[2]),
            Decimal(row[3]),
            Decimal(row[4]),
            Decimal(row[5]),
            str(row[10]),
        )
        quantity = Fraction(snapshot.cumulative_filled_quantity)
        cash = quantity * Fraction(snapshot.average_fill_price)
        fee = Fraction(snapshot.cumulative_fee_quote)
        if previous is None:
            if (
                snapshot.state != "INTENT"
                or snapshot.sequence_number != 1
                or snapshot.occurred_at_ms != intent.created_at_ms
                or quantity != 0
                or snapshot.average_fill_price != 0
                or cash != 0
                or fee != 0
            ):
                raise ValueError("paper cash initial event is inconsistent")
        else:
            transition = PaperOrderTransition(
                str(row[0]),
                snapshot.state,
                snapshot.occurred_at_ms,
                snapshot.cumulative_filled_quantity,
                snapshot.average_fill_price,
                snapshot.cumulative_fee_quote,
                str(row[6]),
                str(row[7]),
                str(row[8]),
                str(row[9]),
            ).validated()
            journal._validate_transition(
                previous=previous, transition=transition, order_quantity=intent.quantity
            )
        delta = quantity - old_quantity
        if delta < 0 or quantity > Fraction(intent.quantity) or fee < old_fee:
            raise ValueError("paper cash quantities or fees regressed or overfilled")
        if delta == 0:
            if cash != old_cash or fee != old_fee:
                raise ValueError("paper cash changed without a quantity increment")
        else:
            if (
                cash <= old_cash
                or not row[8]
                or not row[9]
                or row[7] not in {"execution", "reconciliation", "simulator"}
            ):
                raise ValueError(
                    "paper cash increment lacks positive cash or source binding"
                )
            incremental_price = (cash - old_cash) / delta
            if (
                incremental_price > Fraction(intent.limit_price)
                if intent.side == "BUY"
                else incremental_price < Fraction(intent.limit_price)
            ):
                raise ValueError("paper cash increment violates the order limit")
            events.append(
                _CashEvent(
                    snapshot.occurred_at_ms,
                    not intent.parent_inventory_id,
                    delta,
                    cash - old_cash,
                    fee - old_fee,
                )
            )
        previous = snapshot
        old_quantity, old_cash, old_fee = quantity, cash, fee
    return events


def _derive(journal: PaperOrderJournal, opening_id: str) -> PaperInventoryCash:
    report = journal.reconcile()
    if not report.ok:
        raise ValueError(
            "paper cash requires intact ownership and journal reconciliation"
        )
    parent = journal.intent(opening_id)
    if (
        parent.parent_inventory_id
        or parent.side != "BUY"
        or parent.venue not in {"binance-spot", "polymarket"}
    ):
        raise ValueError(
            "paper cash supports long Spot/token inventory, not shorts or futures"
        )
    ids = journal.connection.execute(
        "SELECT intent_id FROM paper_order_intent WHERE parent_inventory_id = ? ORDER BY intent_id",
        [parent.intent_id],
    ).fetchall()
    related = [parent] + [journal.intent(str(row[0])) for row in ids]
    events = []
    bindings = []
    for intent in related:
        if intent != parent and (
            intent.side != "SELL"
            or intent.outcome != parent.outcome
            or any(
                getattr(intent, name) != getattr(parent, name)
                for name in ("venue", "market_id", "asset_id", "symbol")
            )
        ):
            raise ValueError(
                "paper cash close differs from its owned opening inventory"
            )
        events.extend(_recorded_events(journal, intent))
        bindings.append(
            (intent.intent_id, journal.current(intent.intent_id).event_sha256)
        )
    has_settlement = journal.connection.execute(
        "SELECT count(*) FROM paper_inventory_settlement WHERE opening_intent_id = ?",
        [parent.intent_id],
    ).fetchone()[0]
    settlement_sha = None
    if has_settlement:
        if parent.venue != "polymarket":
            raise ValueError("paper Spot cash does not admit token-settlement proceeds")
        settlement = journal.settlement(parent.intent_id)
        events.append(
            _CashEvent(
                settlement.occurred_at_ms,
                False,
                Fraction(settlement.quantity),
                Fraction(settlement.quantity) * Fraction(settlement.payout_per_unit),
                Fraction(settlement.fee_quote),
            )
        )
        settlement_sha = settlement.settlement_sha256
    events.sort(key=lambda event: event.occurred_at_ms)
    if len({event.occurred_at_ms for event in events}) != len(events):
        raise ValueError("paper cash equal-time inventory events lack causal ordering")
    lots: deque[tuple[Fraction, Fraction]] = deque()
    opened = closed = acquisition = proceeds = allocated = Fraction(0)
    first_fill = None
    for event in events:
        if event.opening:
            cost = event.cash + event.fee
            lots.append((event.quantity, cost))
            opened += event.quantity
            acquisition += cost
            if first_fill is None:
                first_fill = event.occurred_at_ms
        else:
            quantity = event.quantity
            if quantity > opened - closed:
                raise ValueError(
                    "paper cash disposal precedes or exceeds owned filled inventory"
                )
            while quantity:
                lot_quantity, lot_cost = lots.popleft()
                used = min(quantity, lot_quantity)
                cost = lot_cost * used / lot_quantity
                allocated += cost
                if used < lot_quantity:
                    lots.appendleft((lot_quantity - used, lot_cost - cost))
                quantity -= used
            closed += event.quantity
            proceeds += event.cash - event.fee
    remaining_cost = sum((cost for _, cost in lots), Fraction(0))
    assert acquisition == allocated + remaining_cost
    return PaperInventoryCash(
        parent.intent_id,
        parent.venue,
        parent.asset_id,
        parent.symbol,
        opened,
        closed,
        opened - closed,
        acquisition,
        proceeds,
        allocated,
        remaining_cost,
        first_fill,
        tuple(item for item in report.blocking_intent_ids if item in dict(bindings)),
        tuple(sorted(bindings)),
        settlement_sha,
    )


def derive_paper_inventory_cash(
    journal: PaperOrderJournal, opening_intent_id: str
) -> PaperInventoryCash:
    """Read one consistent journal snapshot; must be outside an existing transaction."""
    journal.connection.execute("BEGIN TRANSACTION")
    try:
        return _derive(journal, opening_intent_id)
    finally:
        journal.connection.execute("ROLLBACK")
