"""Durable Binance paper broker backed by the shared execution contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import math
from pathlib import Path
import time
from typing import Mapping

import duckdb

from .api import BinanceClient
from .paper_execution import (
    BinancePaperExecutionAdapter,
    PaperBookSnapshot,
    PaperExecutionResult,
    PaperOrderIntent,
    PaperOrderJournal,
    PaperReconciliationReport,
    binance_book_ticker_snapshot,
    paper_intent_id,
)
from .positions import BOT_OWNER, ClosedTrade, OpenPosition, PositionsStore


@dataclass(frozen=True)
class PaperPositionReconciliation:
    venue: str
    journal: PaperReconciliationReport
    position_errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.journal.ok and not self.position_errors

    @property
    def can_open(self) -> bool:
        return self.ok and self.journal.can_open

    @property
    def can_close(self) -> bool:
        return self.ok and self.journal.can_close

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        payload["can_open"] = self.can_open
        payload["can_close"] = self.can_close
        return payload


@dataclass(frozen=True)
class _ObservedBook:
    book: PaperBookSnapshot
    created_at_ms: int
    execution_time_ms: int
    submission_latency_ms: int


class BinancePaperBroker:
    """Execute Binance paper orders only against a freshly observed real BBO."""

    def __init__(
        self,
        database: str | Path,
        client: BinanceClient,
        *,
        market_type: str,
        taker_fee_bps: float,
        maximum_book_age_ms: int = 2_000,
        order_ttl_ms: int = 30_000,
    ) -> None:
        normalized_market = str(market_type or "").strip().lower()
        if normalized_market not in {"spot", "futures"}:
            raise ValueError("Binance paper market_type must be spot or futures")
        fee_bps = Decimal(str(taker_fee_bps))
        if not fee_bps.is_finite() or fee_bps < 0:
            raise ValueError("Binance paper taker fee must be non-negative")
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.market_type = normalized_market
        self.venue = f"binance-{normalized_market}"
        self.maximum_book_age_ms = int(maximum_book_age_ms)
        self.order_ttl_ms = int(order_ttl_ms)
        if self.maximum_book_age_ms < 0 or self.maximum_book_age_ms > 60_000:
            raise ValueError("maximum_book_age_ms must lie in [0, 60000]")
        if self.order_ttl_ms < 1_000 or self.order_ttl_ms > 300_000:
            raise ValueError("order_ttl_ms must lie in [1000, 300000]")
        self.connection = duckdb.connect(str(self.database))
        self.connection.execute("SET memory_limit='512MB'")
        self.connection.execute("SET threads=2")
        self.connection.execute("SET TimeZone='UTC'")
        self.connection.execute("SET preserve_insertion_order=false")
        self._closed = False
        self.journal = PaperOrderJournal(self.connection)
        self.adapter = BinancePaperExecutionAdapter(
            self.journal,
            market_type=normalized_market,
            maker_fee_bps=fee_bps,
            taker_fee_bps=fee_bps,
        )

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self) -> "BinancePaperBroker":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _capture_book(self, symbol: str) -> _ObservedBook:
        created_wall_ms = time.time_ns() // 1_000_000
        started_monotonic_ns = time.monotonic_ns()
        payload = self.client.get_book_ticker(symbol)
        received_monotonic_ns = time.monotonic_ns()
        received_wall_ms = time.time_ns() // 1_000_000
        if not isinstance(payload, Mapping):
            raise ValueError("Binance book ticker response must be an object")
        elapsed_ns = max(1, received_monotonic_ns - started_monotonic_ns)
        latency_ms = max(1, math.ceil(elapsed_ns / 1_000_000))
        execution_time_ms = max(received_wall_ms, created_wall_ms + latency_ms)
        book = binance_book_ticker_snapshot(
            payload,
            market_type=self.market_type,
            symbol=symbol,
            received_wall_ms=received_wall_ms,
            received_monotonic_ns=received_monotonic_ns,
        )
        return _ObservedBook(
            book=book,
            created_at_ms=created_wall_ms,
            execution_time_ms=execution_time_ms,
            submission_latency_ms=latency_ms,
        )

    def open_position(
        self,
        position: OpenPosition,
    ) -> tuple[OpenPosition | None, PaperExecutionResult]:
        if not position.dry_run or position.owner != BOT_OWNER:
            raise ValueError("Binance paper open requires bot-owned paper position")
        if str(position.market_type).strip().lower() != self.market_type:
            raise ValueError("paper position market_type does not match the broker")
        if str(position.side).strip().upper() not in {"LONG", "SHORT"}:
            raise ValueError("paper position side must be LONG or SHORT")
        preflight = self.journal.reconcile(self.venue)
        if not preflight.can_open:
            raise ValueError(f"paper reconciliation blocks open: {asdict(preflight)}")
        observed = self._capture_book(position.symbol)
        side = "BUY" if str(position.side).upper() == "LONG" else "SELL"
        levels = observed.book.asks if side == "BUY" else observed.book.bids
        if not levels:
            raise ValueError("Binance BBO has no executable level")
        intent_id = paper_intent_id(self.venue, position.id, "open")
        intent = PaperOrderIntent(
            intent_id=intent_id,
            venue=self.venue,
            market_id=position.symbol,
            asset_id=position.symbol,
            symbol=position.symbol,
            outcome=str(position.side).upper(),
            side=side,
            order_type="FAK",
            limit_price=levels[0].price,
            quantity=Decimal(str(position.qty)),
            created_at_ms=observed.created_at_ms,
            expires_at_ms=observed.created_at_ms + self.order_ttl_ms,
        )
        result = self.adapter.execute_aggressive(
            intent,
            observed.book,
            execution_time_ms=observed.execution_time_ms,
            submission_latency_ms=observed.submission_latency_ms,
            maximum_book_age_ms=self.maximum_book_age_ms,
        )
        if result.filled_quantity <= 0:
            return None, result
        status = "PARTIALLY_FILLED" if result.remaining_quantity > 0 else result.state
        opened = replace(
            position,
            qty=float(result.filled_quantity),
            entry_price=float(result.average_fill_price),
            opened_at_ms=observed.execution_time_ms,
            notional=float(result.filled_quantity * result.average_fill_price),
            exchange_status=status,
            paper_open_intent_id=intent_id,
            entry_fees=float(result.fee_quote),
        )
        return opened, result

    def _next_close_attempt(self, opening_intent_id: str) -> int:
        row = self.connection.execute(
            """
            SELECT count(*) FROM paper_order_intent
            WHERE parent_inventory_id = ?
            """,
            [opening_intent_id],
        ).fetchone()
        return int(row[0]) + 1

    def close_position(
        self,
        position: OpenPosition,
        *,
        reason: str,
    ) -> tuple[ClosedTrade | None, PaperExecutionResult]:
        if not position.dry_run or position.owner != BOT_OWNER:
            raise ValueError("Binance paper close requires bot-owned paper position")
        if str(position.market_type).strip().lower() != self.market_type:
            raise ValueError("paper position market_type does not match the broker")
        if str(position.side).strip().upper() not in {"LONG", "SHORT"}:
            raise ValueError("paper position side must be LONG or SHORT")
        opening_intent_id = str(position.paper_open_intent_id or "").strip()
        if not opening_intent_id:
            raise ValueError("paper position has no opening intent proof")
        reconciliation = self.journal.reconcile(self.venue)
        if not reconciliation.can_close:
            raise ValueError(
                f"paper reconciliation blocks close: {asdict(reconciliation)}"
            )
        owned = next(
            (
                item
                for item in reconciliation.inventory
                if item.opening_intent_id == opening_intent_id
            ),
            None,
        )
        if owned is None:
            raise ValueError("paper position has no remaining bot-owned inventory")
        position_quantity = Decimal(str(position.qty))
        tolerance = max(Decimal("0.000000000001"), position_quantity * Decimal("1e-8"))
        if abs(position_quantity - owned.remaining_quantity) > tolerance:
            raise ValueError("paper position quantity does not match owned inventory")
        observed = self._capture_book(position.symbol)
        side = "SELL" if str(position.side).upper() == "LONG" else "BUY"
        levels = observed.book.bids if side == "SELL" else observed.book.asks
        if not levels:
            raise ValueError("Binance BBO has no executable close level")
        attempt = self._next_close_attempt(opening_intent_id)
        close_intent_id = paper_intent_id(
            self.venue,
            position.id,
            "close",
            attempt=attempt,
        )
        intent = PaperOrderIntent(
            intent_id=close_intent_id,
            venue=self.venue,
            market_id=position.symbol,
            asset_id=position.symbol,
            symbol=position.symbol,
            outcome=str(position.side).upper(),
            side=side,
            order_type="FAK",
            limit_price=levels[0].price,
            quantity=Decimal(str(position.qty)),
            created_at_ms=observed.created_at_ms,
            expires_at_ms=observed.created_at_ms + self.order_ttl_ms,
            parent_inventory_id=opening_intent_id,
        )
        result = self.adapter.execute_aggressive(
            intent,
            observed.book,
            execution_time_ms=observed.execution_time_ms,
            submission_latency_ms=observed.submission_latency_ms,
            maximum_book_age_ms=self.maximum_book_age_ms,
            closing_position=True,
        )
        if result.filled_quantity <= 0:
            return None, result
        quantity = float(result.filled_quantity)
        exit_price = float(result.average_fill_price)
        allocation = min(1.0, quantity / max(float(position.qty), 1e-18))
        entry_fee = max(0.0, float(position.entry_fees)) * allocation
        exit_fee = float(result.fee_quote)
        if str(position.side).upper() == "LONG":
            gross_pnl = (exit_price - float(position.entry_price)) * quantity
        else:
            gross_pnl = (float(position.entry_price) - exit_price) * quantity
        fees = entry_fee + exit_fee
        realized = gross_pnl - fees
        entry_notional = float(position.entry_price) * quantity
        trade = ClosedTrade(
            id=position.id,
            symbol=position.symbol,
            market_type=position.market_type,
            side=position.side,
            qty=quantity,
            entry_price=float(position.entry_price),
            exit_price=exit_price,
            leverage=float(position.leverage),
            opened_at_ms=int(position.opened_at_ms),
            closed_at_ms=observed.execution_time_ms,
            realized_pnl=realized,
            realized_pnl_pct=(realized / entry_notional) if entry_notional > 0 else 0.0,
            fees=fees,
            reason=str(reason),
            strategy_profile=position.strategy_profile,
            objective=position.objective,
            dry_run=True,
            owner=position.owner,
            open_client_order_id=position.open_client_order_id,
            exchange_status=result.state,
            paper_open_intent_id=opening_intent_id,
            paper_close_intent_id=close_intent_id,
            ai_review_mode=position.ai_review_mode,
            ai_review_case_id=position.ai_review_case_id,
            ai_review_status=position.ai_review_status,
        )
        return trade, result

    def reconcile_positions(self, store: PositionsStore) -> PaperPositionReconciliation:
        return self.reconcile_positions_data(
            store.load_open(),
            ledger_errors=store.open_integrity_errors(),
        )

    def reconcile_positions_data(
        self,
        positions: list[OpenPosition],
        *,
        ledger_errors: tuple[str, ...],
    ) -> PaperPositionReconciliation:
        journal_report = self.journal.reconcile(self.venue)
        errors = list(ledger_errors)
        for position in positions:
            position_market = str(position.market_type).strip().lower()
            if not position.dry_run:
                errors.append(f"paper_mode_live_position_present:{position.id}")
            elif position_market != self.market_type:
                errors.append(
                    f"paper_position_market_mismatch:{position.id}:{position_market}"
                )
        relevant = [
            position
            for position in positions
            if position.dry_run
            and str(position.market_type).lower() == self.market_type
        ]
        inventory = {
            item.opening_intent_id: item
            for item in journal_report.inventory
            if item.remaining_quantity > 0
        }
        position_intents: set[str] = set()
        for position in relevant:
            intent_id = str(position.paper_open_intent_id or "").strip()
            if position.owner != BOT_OWNER:
                errors.append(f"paper_position_owner_unverified:{position.id}")
                continue
            if not intent_id:
                errors.append(f"paper_position_intent_missing:{position.id}")
                continue
            if intent_id in position_intents:
                errors.append(f"paper_position_intent_duplicated:{intent_id}")
                continue
            position_intents.add(intent_id)
            owned = inventory.get(intent_id)
            if owned is None:
                errors.append(f"paper_position_without_inventory:{position.id}")
                continue
            if owned.symbol != position.symbol:
                errors.append(f"paper_position_symbol_mismatch:{position.id}")
            quantity = Decimal(str(position.qty))
            tolerance = max(Decimal("0.000000000001"), quantity * Decimal("1e-8"))
            if abs(quantity - owned.remaining_quantity) > tolerance:
                errors.append(f"paper_position_quantity_mismatch:{position.id}")
        for intent_id in sorted(set(inventory) - position_intents):
            errors.append(f"paper_inventory_without_position:{intent_id}")
        return PaperPositionReconciliation(
            venue=self.venue,
            journal=journal_report,
            position_errors=tuple(sorted(set(errors))),
        )


__all__ = ["BinancePaperBroker", "PaperPositionReconciliation"]
