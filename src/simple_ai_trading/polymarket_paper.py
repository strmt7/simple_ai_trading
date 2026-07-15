"""Durable Polymarket paper broker over prospective replay evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from .paper_execution import (
    PaperExecutionResult,
    PaperOrderIntent,
    PaperOrderJournal,
    PaperReconciliationReport,
    PolymarketPaperExecutionAdapter,
    paper_intent_id,
)
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketRecordedBook,
    PolymarketResolutionEvidence,
)
from .polymarket_resolution import load_official_resolutions


POLYMARKET_PAPER_CONTEXT_SCHEMA_VERSION = "polymarket-paper-context-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


@dataclass(frozen=True)
class PolymarketPaperPosition:
    opening_intent_id: str
    run_id: str
    market_id: str
    asset: str
    token_id: str
    outcome: str
    opened_quantity: Decimal
    remaining_quantity: Decimal
    average_entry_price: Decimal
    remaining_entry_fee_quote: Decimal
    opened_at_ms: int
    decision_event_id: str
    execution_event_id: str


@dataclass(frozen=True)
class PolymarketPaperClose:
    closing_intent_id: str
    opening_intent_id: str
    quantity: Decimal
    average_entry_price: Decimal
    average_exit_price: Decimal
    entry_fee_quote: Decimal
    exit_fee_quote: Decimal
    realized_pnl_quote: Decimal
    closed_at_ms: int


@dataclass(frozen=True)
class PolymarketPaperSettlement:
    settlement_id: str
    opening_intent_id: str
    quantity: Decimal
    payout_per_unit: Decimal
    gross_payout_quote: Decimal
    entry_cost_quote: Decimal
    entry_fee_quote: Decimal
    realized_pnl_quote: Decimal
    resolved_at_ms: int
    resolution_event_id: str


@dataclass(frozen=True)
class PolymarketPaperStopReport:
    status: str
    close_fill_count: int
    settlement_count: int
    remaining_opening_intent_ids: tuple[str, ...]
    blocking_intent_ids: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def stopped(self) -> bool:
        return self.status == "STOPPED"

    def asdict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "stopped": self.stopped,
            "close_fill_count": self.close_fill_count,
            "settlement_count": self.settlement_count,
            "remaining_opening_intent_ids": self.remaining_opening_intent_ids,
            "blocking_intent_ids": self.blocking_intent_ids,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class PolymarketPaperReconciliation:
    venue: str
    journal: PaperReconciliationReport
    evidence_errors: tuple[str, ...]
    context_errors: tuple[str, ...]

    @property
    def position_errors(self) -> tuple[str, ...]:
        return (*self.evidence_errors, *self.context_errors)

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
        return {
            "venue": self.venue,
            "journal": {
                "venue": self.journal.venue,
                "tracked_inventory_count": len(self.journal.inventory),
                "open_inventory_count": sum(
                    item.remaining_quantity > 0
                    for item in self.journal.inventory
                ),
                "blocking_intent_ids": self.journal.blocking_intent_ids,
                "integrity_errors": self.journal.integrity_errors,
                "ownership_errors": self.journal.ownership_errors,
            },
            "evidence_errors": self.evidence_errors,
            "context_errors": self.context_errors,
            "ok": self.ok,
            "can_open": self.can_open,
            "can_close": self.can_close,
        }


@dataclass(frozen=True)
class _ExecutionSelection:
    book: PolymarketRecordedBook | None
    execution_time_ms: int
    snapshot_sha256: str
    event_id: str


class PolymarketPaperBroker:
    """Execute paper outcome-token orders against closed prospective evidence."""

    venue = "polymarket"

    def __init__(
        self,
        database: str | Path,
        *,
        run_id: str | None = None,
        maximum_book_age_ms: int = 2_000,
        order_ttl_ms: int = 30_000,
        allow_segmented_gaps: bool = False,
        memory_limit: str = "1GB",
        threads: int = 2,
    ) -> None:
        self.store = PolymarketEvidenceStore(
            database,
            memory_limit=memory_limit,
            threads=threads,
        )
        self.store.connect()
        self._closed = False
        try:
            self.allow_segmented_gaps = bool(allow_segmented_gaps)
            self.replay = PolymarketEvidenceReplay.load(
                self.store,
                run_id=run_id,
                allow_segmented_gaps=self.allow_segmented_gaps,
            )
            self._replay_cache = {self.replay.run_id: self.replay}
            if self.store.paper_journal is None:
                raise RuntimeError("shared paper journal is unavailable")
            self.journal: PaperOrderJournal = self.store.paper_journal
            self.maximum_book_age_ms = int(maximum_book_age_ms)
            self.order_ttl_ms = int(order_ttl_ms)
            if self.maximum_book_age_ms < 0 or self.maximum_book_age_ms > 60_000:
                raise ValueError("maximum_book_age_ms must lie in [0, 60000]")
            if self.order_ttl_ms < 1_000 or self.order_ttl_ms > 300_000:
                raise ValueError("order_ttl_ms must lie in [1000, 300000]")
            self._init_context_schema()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    def __enter__(self) -> "PolymarketPaperBroker":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _init_context_schema(self) -> None:
        self.store.connect().execute(
            """
            CREATE TABLE IF NOT EXISTS polymarket_paper_order_context (
                intent_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                token_id VARCHAR NOT NULL,
                decision_event_id VARCHAR NOT NULL,
                execution_event_id VARCHAR NOT NULL,
                decision_snapshot_sha256 VARCHAR NOT NULL,
                execution_snapshot_sha256 VARCHAR NOT NULL,
                requested_latency_ms INTEGER NOT NULL,
                effective_latency_ms INTEGER NOT NULL,
                context_json VARCHAR NOT NULL,
                context_sha256 VARCHAR NOT NULL
            );
            """
        )

    @staticmethod
    def _validate_decision(
        decision: PolymarketRecordedBook,
        *,
        outcome: str,
    ) -> tuple[PolymarketFiveMinuteMarket, str]:
        normalized_outcome = str(outcome or "").strip().title()
        if normalized_outcome not in {"Up", "Down"}:
            raise ValueError("Polymarket outcome must be Up or Down")
        market = decision.market
        expected_token = (
            market.up_token_id if normalized_outcome == "Up" else market.down_token_id
        )
        if decision.token_id != expected_token or decision.outcome != normalized_outcome:
            raise ValueError("decision book does not match the requested outcome")
        if not (
            market.event_start_ms <= decision.received_wall_ms < market.end_ms
        ):
            raise ValueError("decision book lies outside the five-minute market window")
        return market, normalized_outcome

    @staticmethod
    def _validate_limit(
        value: object,
        *,
        tick_size: Decimal,
        name: str,
    ) -> Decimal:
        price = _decimal(value, name=name, positive=True)
        if price >= 1:
            raise ValueError(f"{name} must be below one")
        if price % tick_size != 0:
            raise ValueError(f"{name} is not aligned to the decision-time tick")
        return price

    def _select_execution(
        self,
        decision: PolymarketRecordedBook,
        *,
        latency_ms: int,
    ) -> _ExecutionSelection:
        execution = self.replay.first_book_after_latency(
            decision,
            latency_ms=latency_ms,
        )
        minimum_time = decision.received_wall_ms + int(latency_ms)
        if execution is None:
            return _ExecutionSelection(
                book=None,
                execution_time_ms=minimum_time,
                snapshot_sha256=decision.snapshot.source_payload_sha256,
                event_id="",
            )
        return _ExecutionSelection(
            book=execution,
            execution_time_ms=max(minimum_time, execution.received_wall_ms),
            snapshot_sha256=execution.snapshot.source_payload_sha256,
            event_id=execution.event_id,
        )

    def _latest_consumed_monotonic_ns(self) -> int:
        rows = self.store.connect().execute(
            """
            SELECT token_id, decision_event_id, execution_event_id
            FROM polymarket_paper_order_context
            WHERE run_id = ?
            """,
            [self.replay.run_id],
        ).fetchall()
        latest_monotonic_ns = -1
        for token_id, decision_event_id, execution_event_id in rows:
            for event_id in (str(decision_event_id), str(execution_event_id)):
                if not event_id:
                    continue
                try:
                    recorded = self.replay.book_for_event(event_id, str(token_id))
                except KeyError as exc:
                    raise ValueError(
                        "existing paper context is absent from replay evidence"
                    ) from exc
                latest_monotonic_ns = max(
                    latest_monotonic_ns,
                    recorded.received_monotonic_ns,
                )
        return latest_monotonic_ns

    def _validate_replay_chronology(
        self,
        decision: PolymarketRecordedBook,
    ) -> None:
        latest_monotonic_ns = self._latest_consumed_monotonic_ns()
        if decision.received_monotonic_ns <= latest_monotonic_ns:
            raise ValueError(
                "paper decision must follow every previously consumed replay state"
            )

    def _execute(
        self,
        intent: PaperOrderIntent,
        *,
        market: PolymarketFiveMinuteMarket,
        decision: PolymarketRecordedBook,
        latency_ms: int,
        closing_position: bool,
    ) -> tuple[PaperExecutionResult, _ExecutionSelection]:
        selection = self._select_execution(decision, latency_ms=latency_ms)
        snapshot = (
            selection.book.snapshot
            if selection.book is not None
            else replace(decision.snapshot, connected=False, gap_free=False)
        )
        if selection.book is not None and intent.limit_price % selection.book.tick_size:
            raise ValueError("order limit is not aligned to the execution-time tick")
        adapter = PolymarketPaperExecutionAdapter(
            self.journal,
            fee=market.fee_schedule.fee_model(),
        )
        result = adapter.execute_aggressive(
            intent,
            snapshot,
            execution_time_ms=selection.execution_time_ms,
            submission_latency_ms=int(latency_ms),
            maximum_book_age_ms=self.maximum_book_age_ms,
            closing_position=closing_position,
        )
        self._record_context(
            intent,
            decision=decision,
            selection=selection,
            latency_ms=int(latency_ms),
        )
        return result, selection

    def open_position(
        self,
        *,
        position_id: str,
        decision: PolymarketRecordedBook,
        outcome: str,
        quantity: object,
        maximum_price: object,
        submission_latency_ms: int,
    ) -> tuple[PolymarketPaperPosition | None, PaperExecutionResult]:
        reconciliation = self.reconcile()
        if not reconciliation.can_open:
            raise ValueError(
                f"Polymarket paper reconciliation blocks open: {reconciliation.asdict()}"
            )
        market, normalized_outcome = self._validate_decision(
            decision,
            outcome=outcome,
        )
        self._validate_replay_chronology(decision)
        requested_quantity = _decimal(quantity, name="quantity", positive=True)
        if requested_quantity < market.minimum_order_size:
            raise ValueError("quantity is below the market minimum order size")
        limit = self._validate_limit(
            maximum_price,
            tick_size=decision.tick_size,
            name="maximum_price",
        )
        latency = int(submission_latency_ms)
        if latency <= 0 or latency > 60_000:
            raise ValueError("submission_latency_ms must lie in [1, 60000]")
        created_at = decision.received_wall_ms
        expires_at = min(market.end_ms, created_at + self.order_ttl_ms)
        if expires_at <= created_at:
            raise ValueError("paper order has no valid lifetime before market end")
        intent_id = paper_intent_id(self.venue, position_id, "open")
        intent = PaperOrderIntent(
            intent_id=intent_id,
            venue=self.venue,
            market_id=market.condition_id,
            asset_id=decision.token_id,
            symbol=market.asset,
            outcome=normalized_outcome,
            side="BUY",
            order_type="FAK",
            limit_price=limit,
            quantity=requested_quantity,
            created_at_ms=created_at,
            expires_at_ms=expires_at,
        )
        result, _selection = self._execute(
            intent,
            market=market,
            decision=decision,
            latency_ms=latency,
            closing_position=False,
        )
        if result.filled_quantity <= 0:
            return None, result
        return self._position(intent_id), result

    def close_position(
        self,
        *,
        opening_intent_id: str,
        decision: PolymarketRecordedBook,
        quantity: object | None = None,
        minimum_price: object,
        submission_latency_ms: int,
    ) -> tuple[PolymarketPaperClose | None, PaperExecutionResult]:
        reconciliation = self.reconcile()
        if not reconciliation.can_close:
            raise ValueError(
                f"Polymarket paper reconciliation blocks close: {reconciliation.asdict()}"
            )
        owned = next(
            (
                item
                for item in reconciliation.journal.inventory
                if item.opening_intent_id == opening_intent_id
                and item.remaining_quantity > 0
            ),
            None,
        )
        if owned is None:
            raise ValueError("opening intent has no remaining bot-owned inventory")
        parent = self.journal.intent(opening_intent_id)
        market = self._market(parent.market_id)
        if decision.token_id != parent.asset_id or decision.market != market:
            raise ValueError("close decision does not match bot-owned inventory")
        self._validate_replay_chronology(decision)
        requested_quantity = (
            owned.remaining_quantity
            if quantity is None
            else _decimal(quantity, name="quantity", positive=True)
        )
        if requested_quantity > owned.remaining_quantity:
            raise ValueError("close quantity exceeds bot-owned inventory")
        if requested_quantity < market.minimum_order_size:
            raise ValueError("close quantity is below the market minimum order size")
        limit = self._validate_limit(
            minimum_price,
            tick_size=decision.tick_size,
            name="minimum_price",
        )
        latency = int(submission_latency_ms)
        if latency <= 0 or latency > 60_000:
            raise ValueError("submission_latency_ms must lie in [1, 60000]")
        created_at = decision.received_wall_ms
        expires_at = min(market.end_ms, created_at + self.order_ttl_ms)
        if expires_at <= created_at:
            raise ValueError("paper close has no valid lifetime before market end")
        attempt = self._next_close_attempt(opening_intent_id)
        intent_id = paper_intent_id(
            self.venue,
            opening_intent_id,
            "close",
            attempt=attempt,
        )
        intent = PaperOrderIntent(
            intent_id=intent_id,
            venue=self.venue,
            market_id=parent.market_id,
            asset_id=parent.asset_id,
            symbol=parent.symbol,
            outcome=parent.outcome,
            side="SELL",
            order_type="FAK",
            limit_price=limit,
            quantity=requested_quantity,
            created_at_ms=created_at,
            expires_at_ms=expires_at,
            parent_inventory_id=opening_intent_id,
        )
        result, selection = self._execute(
            intent,
            market=market,
            decision=decision,
            latency_ms=latency,
            closing_position=True,
        )
        if result.filled_quantity <= 0:
            return None, result
        opening = self.journal.current(opening_intent_id)
        entry_fee = opening.cumulative_fee_quote * (
            result.filled_quantity / opening.cumulative_filled_quantity
        )
        entry_cost = opening.average_fill_price * result.filled_quantity
        exit_proceeds = result.average_fill_price * result.filled_quantity
        realized = exit_proceeds - entry_cost - entry_fee - result.fee_quote
        return (
            PolymarketPaperClose(
                closing_intent_id=intent_id,
                opening_intent_id=opening_intent_id,
                quantity=result.filled_quantity,
                average_entry_price=opening.average_fill_price,
                average_exit_price=result.average_fill_price,
                entry_fee_quote=entry_fee,
                exit_fee_quote=result.fee_quote,
                realized_pnl_quote=realized,
                closed_at_ms=selection.execution_time_ms,
            ),
            result,
        )

    def settle_position(
        self,
        *,
        opening_intent_id: str,
        resolution: PolymarketResolutionEvidence,
    ) -> PolymarketPaperSettlement:
        reconciliation = self.reconcile()
        if not reconciliation.can_close:
            raise ValueError(
                "Polymarket paper reconciliation blocks settlement: "
                f"{reconciliation.asdict()}"
            )
        owned = next(
            (
                item
                for item in reconciliation.journal.inventory
                if item.opening_intent_id == opening_intent_id
                and item.remaining_quantity > 0
            ),
            None,
        )
        if owned is None:
            raise ValueError("opening intent has no remaining bot-owned inventory")
        official = next(
            (
                item
                for item in self.replay.resolutions
                if item.event_id == resolution.event_id
            ),
            None,
        )
        if official is None or official != resolution:
            raise ValueError("resolution is not immutable evidence from this replay")
        parent = self.journal.intent(opening_intent_id)
        if resolution.condition_id != parent.market_id:
            raise ValueError("resolution market does not match bot-owned inventory")
        payout = Decimal("1") if resolution.winning_asset_id == parent.asset_id else Decimal("0")
        settlement_id = "paper-polymarket-settlement-" + _canonical_sha256(
            {
                "opening_intent_id": opening_intent_id,
                "resolution_event_id": resolution.event_id,
            }
        )[:32]
        settlement = self.journal.record_settlement(
            settlement_id=settlement_id,
            opening_intent_id=opening_intent_id,
            quantity=owned.remaining_quantity,
            payout_per_unit=payout,
            fee_quote=Decimal("0"),
            occurred_at_ms=resolution.resolved_at_ms,
            source_event_id=resolution.event_id,
            source_payload_sha256=resolution.event_sha256,
        )
        opening = self.journal.current(opening_intent_id)
        entry_fee = opening.cumulative_fee_quote * (
            settlement.quantity / opening.cumulative_filled_quantity
        )
        entry_cost = opening.average_fill_price * settlement.quantity
        gross_payout = settlement.payout_per_unit * settlement.quantity
        return PolymarketPaperSettlement(
            settlement_id=settlement.settlement_id,
            opening_intent_id=opening_intent_id,
            quantity=settlement.quantity,
            payout_per_unit=settlement.payout_per_unit,
            gross_payout_quote=gross_payout,
            entry_cost_quote=entry_cost,
            entry_fee_quote=entry_fee,
            realized_pnl_quote=gross_payout - entry_cost - entry_fee,
            resolved_at_ms=settlement.occurred_at_ms,
            resolution_event_id=resolution.event_id,
        )

    def _market(self, condition_id: str) -> PolymarketFiveMinuteMarket:
        for market in self.replay.markets:
            if market.condition_id == condition_id:
                return market
        raise ValueError("paper intent market is absent from replay evidence")

    def _replay_for_run(self, run_id: str) -> PolymarketEvidenceReplay:
        selected = str(run_id or "").strip()
        replay = self._replay_cache.get(selected)
        if replay is None:
            replay = PolymarketEvidenceReplay.load(
                self.store,
                run_id=selected,
                allow_segmented_gaps=self.allow_segmented_gaps,
            )
            self._replay_cache[selected] = replay
        return replay

    def _next_close_attempt(self, opening_intent_id: str) -> int:
        row = self.store.connect().execute(
            """
            SELECT count(*) FROM paper_order_intent
            WHERE parent_inventory_id = ?
            """,
            [opening_intent_id],
        ).fetchone()
        return int(row[0]) + 1

    def _record_context(
        self,
        intent: PaperOrderIntent,
        *,
        decision: PolymarketRecordedBook,
        selection: _ExecutionSelection,
        latency_ms: int,
    ) -> None:
        effective_latency = max(0, selection.execution_time_ms - intent.created_at_ms)
        payload = {
            "schema_version": POLYMARKET_PAPER_CONTEXT_SCHEMA_VERSION,
            "intent_id": intent.intent_id,
            "run_id": self.replay.run_id,
            "token_id": intent.asset_id,
            "decision_event_id": decision.event_id,
            "execution_event_id": selection.event_id,
            "decision_snapshot_sha256": decision.snapshot.source_payload_sha256,
            "execution_snapshot_sha256": selection.snapshot_sha256,
            "requested_latency_ms": int(latency_ms),
            "effective_latency_ms": effective_latency,
        }
        payload_json = _canonical_json(payload)
        payload_sha = hashlib.sha256(payload_json.encode("ascii")).hexdigest()
        existing = self.store.connect().execute(
            """
            SELECT context_sha256 FROM polymarket_paper_order_context
            WHERE intent_id = ?
            """,
            [intent.intent_id],
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_sha:
                raise ValueError("paper intent context already exists with other evidence")
            return
        self.store.connect().execute(
            """
            INSERT INTO polymarket_paper_order_context VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                intent.intent_id,
                POLYMARKET_PAPER_CONTEXT_SCHEMA_VERSION,
                self.replay.run_id,
                intent.asset_id,
                decision.event_id,
                selection.event_id,
                decision.snapshot.source_payload_sha256,
                selection.snapshot_sha256,
                int(latency_ms),
                effective_latency,
                payload_json,
                payload_sha,
            ],
        )

    def _context_errors(
        self,
        journal_report: PaperReconciliationReport,
    ) -> tuple[str, ...]:
        connection = self.store.connect()
        intent_rows = connection.execute(
            """
            SELECT intent_id, asset_id FROM paper_order_intent
            WHERE venue = 'polymarket' ORDER BY intent_id
            """
        ).fetchall()
        context_rows = connection.execute(
            """
            SELECT intent_id, schema_version, run_id, token_id,
                   decision_event_id, execution_event_id,
                   decision_snapshot_sha256, execution_snapshot_sha256,
                   requested_latency_ms, effective_latency_ms,
                   context_json, context_sha256
            FROM polymarket_paper_order_context ORDER BY intent_id
            """
        ).fetchall()
        intent_assets = {str(row[0]): str(row[1]) for row in intent_rows}
        contexts = {str(row[0]): row for row in context_rows}
        errors: list[str] = []
        missing = sorted(set(intent_assets) - set(contexts))
        extra = sorted(set(contexts) - set(intent_assets))
        errors.extend(f"paper_context_missing:{intent_id}" for intent_id in missing)
        errors.extend(f"paper_context_without_intent:{intent_id}" for intent_id in extra)
        active_ids = {
            item.opening_intent_id
            for item in journal_report.inventory
            if item.remaining_quantity > 0
        } | set(journal_report.blocking_intent_ids)
        for intent_id, row in contexts.items():
            payload = {
                "schema_version": str(row[1]),
                "intent_id": intent_id,
                "run_id": str(row[2]),
                "token_id": str(row[3]),
                "decision_event_id": str(row[4]),
                "execution_event_id": str(row[5]),
                "decision_snapshot_sha256": str(row[6]),
                "execution_snapshot_sha256": str(row[7]),
                "requested_latency_ms": int(row[8]),
                "effective_latency_ms": int(row[9]),
            }
            payload_json = _canonical_json(payload)
            if str(row[1]) != POLYMARKET_PAPER_CONTEXT_SCHEMA_VERSION:
                errors.append(f"paper_context_schema_mismatch:{intent_id}")
            if payload_json != str(row[10]):
                errors.append(f"paper_context_payload_mismatch:{intent_id}")
            if hashlib.sha256(payload_json.encode("ascii")).hexdigest() != str(row[11]):
                errors.append(f"paper_context_hash_mismatch:{intent_id}")
            if intent_assets.get(intent_id) != str(row[3]):
                errors.append(f"paper_context_token_mismatch:{intent_id}")
            if intent_id in active_ids and str(row[2]) != self.replay.run_id:
                errors.append(f"active_paper_context_run_mismatch:{intent_id}")
            if str(row[2]) == self.replay.run_id:
                try:
                    decision = self.replay.book_for_event(str(row[4]), str(row[3]))
                except KeyError:
                    errors.append(f"paper_context_decision_missing:{intent_id}")
                else:
                    if decision.snapshot.source_payload_sha256 != str(row[6]):
                        errors.append(f"paper_context_decision_hash_mismatch:{intent_id}")
                execution_event = str(row[5])
                if execution_event:
                    try:
                        execution = self.replay.book_for_event(
                            execution_event,
                            str(row[3]),
                        )
                    except KeyError:
                        errors.append(f"paper_context_execution_missing:{intent_id}")
                    else:
                        if execution.snapshot.source_payload_sha256 != str(row[7]):
                            errors.append(
                                f"paper_context_execution_hash_mismatch:{intent_id}"
                            )
        settlements = connection.execute(
            """
            SELECT settlement_id, opening_intent_id, payout_per_unit,
                   source_event_id, source_payload_sha256
            FROM paper_inventory_settlement
            WHERE venue = 'polymarket' ORDER BY settlement_id
            """
        ).fetchall()
        for settlement_id, opening_id, payout, source_event_id, source_sha in settlements:
            context = contexts.get(str(opening_id))
            if context is None:
                errors.append(f"paper_settlement_context_missing:{settlement_id}")
                continue
            settlement_run = str(context[2])
            try:
                settlement_replay = self._replay_for_run(settlement_run)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(
                    f"paper_settlement_run_invalid:{settlement_id}:"
                    f"{exc.__class__.__name__}"
                )
                continue
            resolution_by_event = {
                item.event_id: item for item in settlement_replay.resolutions
            }
            resolution = resolution_by_event.get(str(source_event_id))
            if resolution is None:
                errors.append(f"paper_settlement_resolution_missing:{settlement_id}")
                continue
            if resolution.event_sha256 != str(source_sha):
                errors.append(f"paper_settlement_resolution_hash_mismatch:{settlement_id}")
            try:
                parent = self.journal.intent(str(opening_id))
            except KeyError:
                errors.append(f"paper_settlement_parent_missing:{settlement_id}")
                continue
            expected_payout = (
                Decimal("1")
                if resolution.winning_asset_id == parent.asset_id
                else Decimal("0")
            )
            if resolution.condition_id != parent.market_id:
                errors.append(f"paper_settlement_market_mismatch:{settlement_id}")
            if Decimal(str(payout)) != expected_payout:
                errors.append(f"paper_settlement_payout_mismatch:{settlement_id}")
        return tuple(sorted(set(errors)))

    def reconcile(self) -> PolymarketPaperReconciliation:
        journal_report = self.journal.reconcile(self.venue)
        evidence_errors = list(self.store.integrity_errors(self.replay.run_id))
        run = self.store.connect().execute(
            """
            SELECT status FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [self.replay.run_id],
        ).fetchone()
        run_status = "" if run is None else str(run[0])
        status_allowed = run_status == "complete" or (
            self.allow_segmented_gaps and run_status == "degraded"
        )
        if not status_allowed:
            evidence_errors.append("paper_replay_run_not_complete")
        try:
            PolymarketEvidenceReplay.validate_stream_gaps(
                self.store,
                self.replay.run_id,
                allow_segmented_gaps=self.allow_segmented_gaps,
            )
        except ValueError as exc:
            evidence_errors.append(
                f"paper_replay_gap_invalid:{exc.__class__.__name__}:{exc}"
            )
        try:
            stored_resolutions = load_official_resolutions(
                self.store,
                run_id=self.replay.run_id,
            )
            replay_by_condition = {
                item.condition_id: item
                for item in self.replay.resolutions
            }
            expected_external_ids = {
                item.event_id
                for item in self.replay.resolutions
                if item.source == "clob_gamma_crosscheck"
            }
            stored_ids = {item.resolution_id for item in stored_resolutions}
            source_mismatch = any(
                item.condition_id not in replay_by_condition
                or replay_by_condition[item.condition_id].winning_asset_id
                != item.winning_asset_id
                for item in stored_resolutions
            )
            if not expected_external_ids.issubset(stored_ids) or source_mismatch:
                evidence_errors.append("paper_replay_resolution_set_mismatch")
        except ValueError as exc:
            evidence_errors.append(
                f"paper_replay_resolution_invalid:{exc.__class__.__name__}:{exc}"
            )
        return PolymarketPaperReconciliation(
            venue=self.venue,
            journal=journal_report,
            evidence_errors=tuple(sorted(set(evidence_errors))),
            context_errors=self._context_errors(journal_report),
        )

    def positions(self) -> tuple[PolymarketPaperPosition, ...]:
        reconciliation = self.reconcile()
        if not reconciliation.ok:
            raise ValueError(
                f"Polymarket paper positions are not reconcilable: {reconciliation.asdict()}"
            )
        output: list[PolymarketPaperPosition] = []
        for inventory in reconciliation.journal.inventory:
            if inventory.remaining_quantity <= 0:
                continue
            output.append(self._position(inventory.opening_intent_id))
        return tuple(output)

    def stop_all_positions(
        self,
        *,
        submission_latency_ms: int,
    ) -> PolymarketPaperStopReport:
        """Close or officially settle every provably bot-owned paper position."""

        latency = int(submission_latency_ms)
        if latency <= 0 or latency > 60_000:
            raise ValueError("submission_latency_ms must lie in [1, 60000]")
        errors: list[str] = []
        close_fills = 0
        settlements = 0
        initial = self.reconcile()
        if not initial.can_close:
            return PolymarketPaperStopReport(
                status="STOPPING",
                close_fill_count=0,
                settlement_count=0,
                remaining_opening_intent_ids=tuple(
                    item.opening_intent_id
                    for item in initial.journal.inventory
                    if item.remaining_quantity > 0
                ),
                blocking_intent_ids=initial.journal.blocking_intent_ids,
                errors=tuple(
                    sorted(
                        set(
                            (
                                *initial.position_errors,
                                *initial.journal.integrity_errors,
                                *initial.journal.ownership_errors,
                            )
                        )
                    )
                ),
            )

        resolutions = {
            item.condition_id: item
            for item in self.replay.resolutions
            if item.source == "clob_gamma_crosscheck"
        }
        for position in self.positions():
            resolution = resolutions.get(position.market_id)
            if resolution is None:
                continue
            try:
                self.settle_position(
                    opening_intent_id=position.opening_intent_id,
                    resolution=resolution,
                )
            except Exception as exc:  # noqa: BLE001 - stop must preserve all failures
                errors.append(
                    f"settlement:{position.opening_intent_id}:"
                    f"{exc.__class__.__name__}:{exc}"
                )
            else:
                settlements += 1

        max_attempts = len(self.replay.books) + 1
        for _attempt in range(max_attempts):
            reconciliation = self.reconcile()
            remaining = [
                item
                for item in reconciliation.journal.inventory
                if item.remaining_quantity > 0
            ]
            if not remaining or not reconciliation.can_close:
                break
            latest = self._latest_consumed_monotonic_ns()
            candidates: list[tuple[int, str, PolymarketRecordedBook]] = []
            for inventory in remaining:
                for book in self.replay.books:
                    if (
                        book.token_id != inventory.asset_id
                        or book.received_monotonic_ns <= latest
                        or self.replay.first_book_after_latency(
                            book,
                            latency_ms=latency,
                        )
                        is None
                    ):
                        continue
                    candidates.append(
                        (
                            book.received_monotonic_ns,
                            inventory.opening_intent_id,
                            book,
                        )
                    )
                    break
            if not candidates:
                break
            _monotonic_ns, opening_id, decision = min(candidates)
            market = decision.market
            try:
                closed, result = self.close_position(
                    opening_intent_id=opening_id,
                    decision=decision,
                    minimum_price=market.tick_size,
                    submission_latency_ms=latency,
                )
            except Exception as exc:  # noqa: BLE001 - stop must preserve all failures
                errors.append(
                    f"close:{opening_id}:{exc.__class__.__name__}:{exc}"
                )
                break
            if closed is None or result.filled_quantity <= 0:
                errors.append(
                    f"close:{opening_id}:unfilled:{result.state}:{result.reason}"
                )
                break
            close_fills += 1

        final = self.reconcile()
        remaining_ids = tuple(
            item.opening_intent_id
            for item in final.journal.inventory
            if item.remaining_quantity > 0
        )
        blockers = final.journal.blocking_intent_ids
        if not final.ok:
            errors.extend(final.position_errors)
            errors.extend(final.journal.integrity_errors)
            errors.extend(final.journal.ownership_errors)
        status = (
            "STOPPED"
            if final.ok and not remaining_ids and not blockers and not errors
            else "STOPPING"
        )
        return PolymarketPaperStopReport(
            status=status,
            close_fill_count=close_fills,
            settlement_count=settlements,
            remaining_opening_intent_ids=remaining_ids,
            blocking_intent_ids=blockers,
            errors=tuple(sorted(set(errors))),
        )

    def _position(self, opening_intent_id: str) -> PolymarketPaperPosition:
        report = self.journal.reconcile(self.venue)
        inventory = next(
            (
                item
                for item in report.inventory
                if item.opening_intent_id == opening_intent_id
            ),
            None,
        )
        if inventory is None or inventory.remaining_quantity <= 0:
            raise ValueError("opening intent has no remaining inventory")
        intent = self.journal.intent(opening_intent_id)
        current = self.journal.current(opening_intent_id)
        context = self.store.connect().execute(
            """
            SELECT run_id, decision_event_id, execution_event_id
            FROM polymarket_paper_order_context WHERE intent_id = ?
            """,
            [opening_intent_id],
        ).fetchone()
        if context is None:
            raise ValueError("opening intent context is missing")
        market = self._market(intent.market_id)
        remaining_fee = current.cumulative_fee_quote * (
            inventory.remaining_quantity / inventory.opened_quantity
        )
        return PolymarketPaperPosition(
            opening_intent_id=opening_intent_id,
            run_id=str(context[0]),
            market_id=intent.market_id,
            asset=market.asset,
            token_id=intent.asset_id,
            outcome=intent.outcome,
            opened_quantity=inventory.opened_quantity,
            remaining_quantity=inventory.remaining_quantity,
            average_entry_price=current.average_fill_price,
            remaining_entry_fee_quote=remaining_fee,
            opened_at_ms=current.occurred_at_ms,
            decision_event_id=str(context[1]),
            execution_event_id=str(context[2]),
        )


__all__ = [
    "POLYMARKET_PAPER_CONTEXT_SCHEMA_VERSION",
    "PolymarketPaperBroker",
    "PolymarketPaperClose",
    "PolymarketPaperPosition",
    "PolymarketPaperReconciliation",
    "PolymarketPaperSettlement",
    "PolymarketPaperStopReport",
]
