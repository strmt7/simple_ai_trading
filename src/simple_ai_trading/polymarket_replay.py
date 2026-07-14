"""Strict reconstruction of prospective Polymarket CLOB evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Mapping

from .paper_execution import BookLevel, PaperBookSnapshot
from .polymarket import (
    PolymarketFiveMinuteMarket,
    parse_polymarket_five_minute_market,
    validate_clob_order_book,
)
from .polymarket_recorder import PolymarketEvidenceStore


_KNOWN_NO_BOOK_CHANGE_EVENTS = frozenset(
    {"last_trade_price", "new_market", "market_resolved"}
)
_BOOK_DEPTH_EVENTS = frozenset({"book", "price_change"})
_BEST_CORROBORATION_MAX_SOURCE_SKEW_MS = 1_000
_BEST_CORROBORATION_MAX_ARRIVAL_NS = 2_000_000_000
_CAUSAL_REORDER_MAX_SOURCE_SKEW_MS = 1_000
_CAUSAL_REORDER_MAX_ARRIVAL_NS = 2_000_000_000
_CAUSALLY_ORDERED_EVENTS = _BOOK_DEPTH_EVENTS | frozenset(
    {"best_bid_ask", "tick_size_change", "market_resolved"}
)
POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION = "polymarket-replay-diagnostics-v1"


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


def _decimal(
    value: object,
    *,
    name: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return parsed


def _timestamp(value: object, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer timestamp") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class PolymarketRecordedBook:
    run_id: str
    event_id: str
    event_type: str
    connection_id: str
    sequence_number: int
    sub_index: int
    market: PolymarketFiveMinuteMarket
    outcome: str
    tick_size: Decimal
    snapshot: PaperBookSnapshot

    @property
    def token_id(self) -> str:
        return self.snapshot.asset_id

    @property
    def received_wall_ms(self) -> int:
        return self.snapshot.received_wall_ms

    @property
    def received_monotonic_ns(self) -> int:
        return self.snapshot.received_monotonic_ns


@dataclass(frozen=True)
class PolymarketResolutionEvidence:
    run_id: str
    event_id: str
    condition_id: str
    winning_asset_id: str
    winning_outcome: str
    resolved_at_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    event_sha256: str


@dataclass(frozen=True)
class PolymarketReplayDiagnostics:
    schema_version: str
    total_event_count: int
    causally_ordered_event_count: int
    late_event_count: int
    maximum_source_regression_ms: int
    maximum_late_arrival_delay_ns: int
    deferred_event_count: int
    maximum_availability_delay_ns: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _BookState:
    bids: dict[Decimal, Decimal]
    asks: dict[Decimal, Decimal]
    source_time_ms: int
    provenance_sha256: str
    tick_size: Decimal
    book_hash: str


@dataclass
class _PendingBookBatch:
    condition_id: str
    source_time_ms: int
    rows: list[_EventRow]


@dataclass(frozen=True)
class _EventRow:
    event_id: str
    event_type: str
    condition_id: str
    asset_id: str
    event: Mapping[str, object]
    event_sha256: str
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    available_wall_ms: int
    available_monotonic_ns: int
    sub_index: int


class PolymarketEvidenceReplay:
    """Validated run metadata and deterministic level-2 book states."""

    def __init__(
        self,
        *,
        run_id: str,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        books: tuple[PolymarketRecordedBook, ...],
        resolutions: tuple[PolymarketResolutionEvidence, ...],
        diagnostics: PolymarketReplayDiagnostics,
    ) -> None:
        self.run_id = run_id
        self.markets = markets
        self.books = books
        self.resolutions = resolutions
        self.diagnostics = diagnostics
        self._books_by_token: dict[str, tuple[PolymarketRecordedBook, ...]] = {}
        for token in sorted({book.token_id for book in books}):
            self._books_by_token[token] = tuple(
                book for book in books if book.token_id == token
            )
        self._book_by_event_token = {
            (book.event_id, book.token_id): book for book in books
        }
        if len(self._book_by_event_token) != len(books):
            raise ValueError("Polymarket replay book event identities are duplicated")

    @classmethod
    def load(
        cls,
        store: PolymarketEvidenceStore,
        *,
        run_id: str | None = None,
    ) -> "PolymarketEvidenceReplay":
        connection = store.connect()
        selected = str(run_id or "").strip()
        if not selected:
            row = connection.execute(
                """
                SELECT run_id FROM polymarket_recorder_run
                WHERE status = 'complete'
                ORDER BY ended_at_ms DESC, run_id DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                raise ValueError("no complete Polymarket recorder run is available")
            selected = str(row[0])
        run = connection.execute(
            """
            SELECT status FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if run is None:
            raise ValueError(f"unknown Polymarket recorder run: {selected}")
        if str(run[0]) != "complete":
            raise ValueError("Polymarket replay requires a complete gap-free run")
        integrity = store.integrity_errors(selected)
        if integrity:
            raise ValueError(
                "Polymarket replay evidence failed integrity: " + "; ".join(integrity)
            )
        gap_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?",
                [selected],
            ).fetchone()[0]
        )
        if gap_count:
            raise ValueError("Polymarket replay refuses runs with stream gaps")

        markets = cls._load_markets(store, selected)
        events = cls._load_events(store, selected)
        events, diagnostics = cls._causal_event_order(events, markets)
        books, resolutions = cls._reconstruct(selected, markets, events)
        if not books:
            raise ValueError("Polymarket replay contains no validated book states")
        return cls(
            run_id=selected,
            markets=markets,
            books=books,
            resolutions=resolutions,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _load_markets(
        store: PolymarketEvidenceStore,
        run_id: str,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        rows = store.connect().execute(
            """
            SELECT condition_id, gamma_payload_json
            FROM polymarket_market_snapshot
            WHERE run_id = ? ORDER BY event_start_ms, asset
            """,
            [run_id],
        ).fetchall()
        markets: list[PolymarketFiveMinuteMarket] = []
        for condition_id, payload_json in rows:
            try:
                payload = json.loads(str(payload_json))
            except json.JSONDecodeError as exc:
                raise ValueError("stored Gamma payload is invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError("stored Gamma payload must be an object")
            market = parse_polymarket_five_minute_market(payload)
            if market.condition_id != str(condition_id):
                raise ValueError("stored market identity differs from Gamma evidence")
            markets.append(market)
        if not markets:
            raise ValueError("Polymarket replay contains no market metadata")
        if len({market.condition_id for market in markets}) != len(markets):
            raise ValueError("Polymarket replay market identities are duplicated")
        return tuple(markets)

    @staticmethod
    def _load_events(
        store: PolymarketEvidenceStore,
        run_id: str,
    ) -> tuple[_EventRow, ...]:
        rows = store.connect().execute(
            """
            SELECT e.event_id, e.event_type, e.condition_id, e.asset_id,
                   e.event_json, e.event_sha256, r.connection_id,
                   r.sequence_number, r.received_wall_ms,
                   r.received_monotonic_ns, e.sub_index
            FROM polymarket_public_event AS e
            JOIN polymarket_raw_message AS r ON r.message_id = e.message_id
            WHERE e.run_id = ?
              AND e.stream IN ('clob_market', 'clob_rest_book')
            ORDER BY r.received_monotonic_ns, r.received_wall_ms,
                     r.connection_id, r.sequence_number, e.sub_index
            """,
            [run_id],
        ).fetchall()
        events: list[_EventRow] = []
        for row in rows:
            try:
                payload = json.loads(str(row[4]))
            except json.JSONDecodeError as exc:
                raise ValueError("stored CLOB event is invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError("stored CLOB event must be an object")
            events.append(
                _EventRow(
                    event_id=str(row[0]),
                    event_type=str(row[1]),
                    condition_id=str(row[2]),
                    asset_id=str(row[3]),
                    event=dict(payload),
                    event_sha256=str(row[5]),
                    connection_id=str(row[6]),
                    sequence_number=int(row[7]),
                    received_wall_ms=int(row[8]),
                    received_monotonic_ns=int(row[9]),
                    available_wall_ms=int(row[8]),
                    available_monotonic_ns=int(row[9]),
                    sub_index=int(row[10]),
                )
            )
        return tuple(events)

    @classmethod
    def _causal_event_order(
        cls,
        events: tuple[_EventRow, ...],
        markets: tuple[PolymarketFiveMinuteMarket, ...],
    ) -> tuple[tuple[_EventRow, ...], PolymarketReplayDiagnostics]:
        market_by_condition = {market.condition_id: market for market in markets}
        by_condition: dict[str, list[tuple[int, _EventRow]]] = {}
        passthrough: list[_EventRow] = []
        for row in events:
            condition = str(row.event.get("market") or row.condition_id).lower()
            if (
                row.event_type not in _CAUSALLY_ORDERED_EVENTS
                or condition not in market_by_condition
            ):
                passthrough.append(row)
                continue
            source_time = _timestamp(
                row.event.get("timestamp"), name=f"{row.event_type} timestamp"
            )
            by_condition.setdefault(condition, []).append((source_time, row))

        ordered: list[_EventRow] = []
        late_event_count = 0
        maximum_source_regression_ms = 0
        maximum_late_arrival_delay_ns = 0
        deferred_event_count = 0
        maximum_availability_delay_ns = 0
        for condition in sorted(by_condition):
            arrivals = by_condition[condition]
            maximum_source_time = -1
            maximum_source_row: _EventRow | None = None
            for source_time, row in arrivals:
                if source_time < maximum_source_time:
                    if maximum_source_row is None:
                        raise TypeError("causal source watermark is unavailable")
                    source_skew = maximum_source_time - source_time
                    arrival_delay = (
                        row.received_monotonic_ns
                        - maximum_source_row.received_monotonic_ns
                    )
                    late_event_count += 1
                    maximum_source_regression_ms = max(
                        maximum_source_regression_ms, source_skew
                    )
                    maximum_late_arrival_delay_ns = max(
                        maximum_late_arrival_delay_ns, arrival_delay
                    )
                    if (
                        source_skew > _CAUSAL_REORDER_MAX_SOURCE_SKEW_MS
                        or not 0
                        <= arrival_delay
                        <= _CAUSAL_REORDER_MAX_ARRIVAL_NS
                    ):
                        raise ValueError(
                            "CLOB event exceeded the bounded causal reorder window"
                        )
                elif source_time >= maximum_source_time:
                    maximum_source_time = max(maximum_source_time, source_time)
                    maximum_source_row = row

            available_monotonic_ns = 0
            available_wall_ms = 0
            for _source_time, row in sorted(
                arrivals,
                key=lambda item: (item[0], cls._event_arrival_key(item[1])),
            ):
                if row.received_monotonic_ns > available_monotonic_ns:
                    available_monotonic_ns = row.received_monotonic_ns
                    available_wall_ms = row.received_wall_ms
                else:
                    available_monotonic_ns += 1
                    available_wall_ms = max(available_wall_ms, row.received_wall_ms)
                availability_delay = (
                    available_monotonic_ns - row.received_monotonic_ns
                )
                if availability_delay > 0:
                    deferred_event_count += 1
                    maximum_availability_delay_ns = max(
                        maximum_availability_delay_ns, availability_delay
                    )
                ordered.append(
                    replace(
                        row,
                        available_wall_ms=available_wall_ms,
                        available_monotonic_ns=available_monotonic_ns,
                    )
                )
        ordered.extend(passthrough)
        diagnostics = PolymarketReplayDiagnostics(
            schema_version=POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION,
            total_event_count=len(events),
            causally_ordered_event_count=sum(len(rows) for rows in by_condition.values()),
            late_event_count=late_event_count,
            maximum_source_regression_ms=maximum_source_regression_ms,
            maximum_late_arrival_delay_ns=maximum_late_arrival_delay_ns,
            deferred_event_count=deferred_event_count,
            maximum_availability_delay_ns=maximum_availability_delay_ns,
        )
        return tuple(ordered), diagnostics

    @classmethod
    def _reconstruct(
        cls,
        run_id: str,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        events: tuple[_EventRow, ...],
    ) -> tuple[
        tuple[PolymarketRecordedBook, ...],
        tuple[PolymarketResolutionEvidence, ...],
    ]:
        market_by_condition = {market.condition_id: market for market in markets}
        market_by_token = {
            token: market for market in markets for token in market.token_ids
        }
        if len(market_by_token) != sum(len(market.token_ids) for market in markets):
            raise ValueError("Polymarket replay token identities are duplicated")
        state: dict[str, _BookState] = {}
        books: list[PolymarketRecordedBook] = []
        resolutions: list[PolymarketResolutionEvidence] = []
        pending: dict[str, _PendingBookBatch] = {}
        pending_best: dict[str, list[_EventRow]] = {}

        def flush(condition_id: str) -> None:
            batch = pending.pop(condition_id, None)
            if batch is not None:
                books.extend(
                    cls._flush_book_batch(
                        run_id,
                        batch,
                        market_by_condition,
                        market_by_token,
                        state,
                        pending_best,
                    )
                )

        for row in events:
            event_type = row.event_type
            if event_type in _BOOK_DEPTH_EVENTS:
                condition = str(
                    row.event.get("market") or row.condition_id
                ).lower()
                if market_by_condition.get(condition) is None:
                    raise ValueError(
                        f"{event_type} event references an unknown market"
                    )
                if row.condition_id and row.condition_id.lower() != condition:
                    raise ValueError(
                        f"{event_type} event condition identity drifted"
                    )
                source_time = _timestamp(
                    row.event.get("timestamp"),
                    name=f"{event_type} timestamp",
                )
                batch = pending.get(condition)
                if batch is not None and source_time < batch.source_time_ms:
                    raise ValueError("CLOB book-state source time regressed")
                if batch is None or source_time > batch.source_time_ms:
                    flush(condition)
                    pending[condition] = _PendingBookBatch(
                        condition_id=condition,
                        source_time_ms=source_time,
                        rows=[row],
                    )
                else:
                    batch.rows.append(row)
            elif event_type == "best_bid_ask":
                condition = str(
                    row.event.get("market") or row.condition_id
                ).lower()
                flush(condition)
                cls._observe_best_bid_ask(
                    row,
                    market_by_condition,
                    market_by_token,
                    state,
                    pending_best,
                )
            elif event_type == "tick_size_change":
                condition = str(
                    row.event.get("market") or row.condition_id
                ).lower()
                flush(condition)
                cls._apply_tick_size_change(row, market_by_token, state)
            elif event_type == "market_resolved":
                condition = str(
                    row.event.get("market") or row.condition_id
                ).lower()
                flush(condition)
                resolutions.append(
                    cls._resolution(run_id, row, market_by_condition)
                )
            elif event_type not in _KNOWN_NO_BOOK_CHANGE_EVENTS:
                raise ValueError(f"unsupported CLOB replay event type: {event_type}")
        for condition in tuple(pending):
            flush(condition)
        unresolved_best = sum(len(rows) for rows in pending_best.values())
        if unresolved_best:
            raise ValueError(
                "best_bid_ask evidence was not corroborated by a subsequent "
                "depth transition"
            )
        books.sort(
            key=lambda item: (
                item.received_monotonic_ns,
                item.received_wall_ms,
                item.connection_id,
                item.sequence_number,
                item.sub_index,
                item.token_id,
            )
        )
        return tuple(books), tuple(resolutions)

    @classmethod
    def _flush_book_batch(
        cls,
        run_id: str,
        batch: _PendingBookBatch,
        market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
        market_by_token: Mapping[str, PolymarketFiveMinuteMarket],
        state: dict[str, _BookState],
        pending_best: dict[str, list[_EventRow]],
    ) -> tuple[PolymarketRecordedBook, ...]:
        market = market_by_condition.get(batch.condition_id)
        if market is None:
            raise ValueError("book-state batch references an unknown market")
        operations: dict[
            str,
            list[tuple[str, _EventRow, Mapping[str, object] | PaperBookSnapshot]],
        ] = {}
        for row in batch.rows:
            if row.event_type == "book":
                token = str(row.event.get("asset_id") or row.asset_id)
                if market_by_token.get(token) != market:
                    raise ValueError("book event references an unknown market or token")
                snapshot = validate_clob_order_book(
                    market,
                    token,
                    row.event,
                    received_wall_ms=row.received_wall_ms,
                    received_monotonic_ns=row.received_monotonic_ns,
                )
                if snapshot.source_time_ms != batch.source_time_ms:
                    raise ValueError("full book timestamp differs from its atomic batch")
                operations.setdefault(token, []).append(("book", row, snapshot))
            elif row.event_type == "price_change":
                changes = row.event.get("price_changes")
                if not isinstance(changes, list) or not changes:
                    raise ValueError(
                        "price_change event is missing market or level updates"
                    )
                for change in changes:
                    if not isinstance(change, Mapping):
                        raise ValueError("price_change level is malformed")
                    token = str(change.get("asset_id") or "")
                    if market_by_token.get(token) != market:
                        raise ValueError(
                            "price_change references an unknown market or token"
                        )
                    operations.setdefault(token, []).append(
                        ("change", row, dict(change))
                    )

        output: list[PolymarketRecordedBook] = []
        for token in sorted(operations):
            current = state.get(token)
            previous_provenance = current.provenance_sha256 if current else ""
            depth_changed = False
            pending_hash = ""
            pending_changes: list[tuple[_EventRow, Mapping[str, object]]] = []
            relevant_rows: list[_EventRow] = []

            def flush_changes() -> None:
                nonlocal current, depth_changed, pending_hash, pending_changes
                if not pending_changes:
                    return
                if current is None:
                    raise ValueError(
                        "price_change arrived without a proven token baseline"
                    )
                if batch.source_time_ms < current.source_time_ms:
                    raise ValueError("price_change source time regressed")
                reported_checksum: tuple[Decimal, Decimal] | None = None
                for change_row, change in pending_changes:
                    side = str(change.get("side") or "").upper()
                    if side not in {"BUY", "SELL"}:
                        raise ValueError("price_change side must be BUY or SELL")
                    price = _decimal(
                        change.get("price"),
                        name="price_change price",
                        minimum=Decimal("0.0001"),
                        maximum=Decimal("0.9999"),
                    )
                    size = _decimal(
                        change.get("size"),
                        name="price_change size",
                        minimum=Decimal("0"),
                    )
                    if price % current.tick_size != 0:
                        raise ValueError(
                            "price_change price is not aligned to the active tick"
                        )
                    levels = current.bids if side == "BUY" else current.asks
                    if size == 0:
                        levels.pop(price, None)
                    else:
                        levels[price] = size
                    reported_checksum = (
                        _decimal(
                            change.get("best_bid"),
                            name="price_change best_bid",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                        _decimal(
                            change.get("best_ask"),
                            name="price_change best_ask",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                    )
                    relevant_rows.append(change_row)
                expected = (
                    max(current.bids, default=Decimal("0")),
                    min(current.asks, default=Decimal("1")),
                )
                if reported_checksum != expected:
                    raise ValueError(
                        "price_change best bid/ask checksum disagrees with "
                        "atomic depth"
                    )
                current.source_time_ms = batch.source_time_ms
                current.book_hash = pending_hash
                depth_changed = True
                pending_hash = ""
                pending_changes = []

            for operation, row, payload in operations.get(token, ()):
                if operation == "change":
                    if not isinstance(payload, Mapping):
                        raise TypeError("internal price_change payload is malformed")
                    book_hash = str(payload.get("hash") or "").strip()
                    if not book_hash:
                        raise ValueError("price_change is missing its order-book hash")
                    pending_hash = book_hash
                    pending_changes.append((row, payload))
                    continue

                flush_changes()
                if not isinstance(payload, PaperBookSnapshot):
                    raise TypeError("internal full-book payload is malformed")
                book_hash = str(row.event.get("hash") or "").strip()
                if not book_hash:
                    raise ValueError("full book is missing its order-book hash")
                active_tick = current.tick_size if current else market.tick_size
                if any(
                    level.price % active_tick != 0
                    for level in (*payload.bids, *payload.asks)
                ):
                    raise ValueError("full book contains a price off the active tick")
                payload_bids = {
                    level.price: level.quantity for level in payload.bids
                }
                payload_asks = {
                    level.price: level.quantity for level in payload.asks
                }
                if current is not None and payload.source_time_ms < current.source_time_ms:
                    raise ValueError("full book source time regressed")
                if (
                    current is not None
                    and current.book_hash == book_hash
                    and current.bids == payload_bids
                    and current.asks == payload_asks
                ):
                    current.source_time_ms = payload.source_time_ms
                else:
                    # A trade can change depth without a price_change event. The
                    # official full book is therefore an authoritative resync,
                    # even when its opaque exchange hash equals the last delta.
                    current = _BookState(
                        bids=payload_bids,
                        asks=payload_asks,
                        source_time_ms=payload.source_time_ms,
                        provenance_sha256=(
                            current.provenance_sha256
                            if current is not None
                            else payload.source_payload_sha256
                        ),
                        tick_size=active_tick,
                        book_hash=book_hash,
                    )
                relevant_rows.append(row)
                depth_changed = True
            flush_changes()

            if current is None:
                raise ValueError("depth transition has no proven token baseline")
            relevant_rows.extend(
                cls._consume_pending_best(
                    token,
                    batch,
                    current,
                    relevant_rows,
                    pending_best,
                )
            )
            if not depth_changed:
                continue

            state[token] = current
            unique_rows = {row.event_id: row for row in relevant_rows}
            evidence_rows = sorted(
                unique_rows.values(), key=cls._event_arrival_key
            )
            current.provenance_sha256 = _canonical_sha256(
                {
                    "schema_version": "polymarket-atomic-book-v1",
                    "previous": previous_provenance,
                    "condition_id": batch.condition_id,
                    "token_id": token,
                    "source_time_ms": batch.source_time_ms,
                    "official_book_hash": current.book_hash,
                    "tick_size": format(current.tick_size, "f"),
                    "events": [
                        {
                            "event_id": item.event_id,
                            "event_sha256": item.event_sha256,
                            "event_type": item.event_type,
                        }
                        for item in evidence_rows
                    ],
                    "bids": [
                        [format(price, "f"), format(current.bids[price], "f")]
                        for price in sorted(current.bids, reverse=True)
                    ],
                    "asks": [
                        [format(price, "f"), format(current.asks[price], "f")]
                        for price in sorted(current.asks)
                    ],
                }
            )
            selected_row = evidence_rows[-1]
            output.append(
                cls._recorded_book(
                    run_id,
                    selected_row,
                    market,
                    token,
                    current,
                )
            )
        return tuple(output)

    @classmethod
    def _observe_best_bid_ask(
        cls,
        row: _EventRow,
        market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
        market_by_token: Mapping[str, PolymarketFiveMinuteMarket],
        state: Mapping[str, _BookState],
        pending_best: dict[str, list[_EventRow]],
    ) -> None:
        condition = str(row.event.get("market") or row.condition_id).lower()
        token = str(row.event.get("asset_id") or row.asset_id)
        market = market_by_condition.get(condition)
        if market is None or market_by_token.get(token) != market:
            raise ValueError("best_bid_ask references an unknown market or token")
        source_time = _timestamp(
            row.event.get("timestamp"), name="best_bid_ask timestamp"
        )
        observed = cls._best_bid_ask_values(row)
        current = state.get(token)
        if current is not None:
            if (
                current.source_time_ms - source_time
                > _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
            ):
                raise ValueError("best_bid_ask source time regressed")
            expected = (
                max(current.bids, default=Decimal("0")),
                min(current.asks, default=Decimal("1")),
            )
            if observed == expected:
                return
        if pending_best.get(token):
            raise ValueError(
                "best_bid_ask advanced before its prior observation was corroborated"
            )
        pending_best[token] = [row]

    @classmethod
    def _consume_pending_best(
        cls,
        token: str,
        batch: _PendingBookBatch,
        current: _BookState,
        depth_rows: list[_EventRow],
        pending_best: dict[str, list[_EventRow]],
    ) -> tuple[_EventRow, ...]:
        rows = pending_best.get(token, [])
        if not rows:
            return ()
        if not depth_rows:
            raise TypeError("atomic depth transition has no evidence rows")
        completion = max(depth_rows, key=cls._event_arrival_key)
        expected = (
            max(current.bids, default=Decimal("0")),
            min(current.asks, default=Decimal("1")),
        )
        for row in rows:
            source_time = _timestamp(
                row.event.get("timestamp"), name="best_bid_ask timestamp"
            )
            if (
                abs(batch.source_time_ms - source_time)
                > _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
            ):
                raise ValueError(
                    "best_bid_ask was not temporally adjacent to its depth transition"
                )
            arrival_delta = (
                completion.received_monotonic_ns - row.received_monotonic_ns
            )
            if not 0 <= arrival_delta <= _BEST_CORROBORATION_MAX_ARRIVAL_NS:
                raise ValueError(
                    "best_bid_ask was not arrival-adjacent to its depth transition"
                )
            if cls._best_bid_ask_values(row) != expected:
                raise ValueError(
                    "best_bid_ask event disagrees with reconstructed atomic depth"
                )
        pending_best.pop(token, None)
        return tuple(rows)

    @staticmethod
    def _best_bid_ask_values(row: _EventRow) -> tuple[Decimal, Decimal]:
        best_bid = _decimal(
            row.event.get("best_bid"),
            name="best_bid",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        best_ask = _decimal(
            row.event.get("best_ask"),
            name="best_ask",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if best_bid > best_ask:
            raise ValueError("best_bid_ask event is crossed")
        if "spread" in row.event:
            spread = _decimal(
                row.event.get("spread"),
                name="best_bid_ask spread",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
            if spread != best_ask - best_bid:
                raise ValueError("best_bid_ask spread is inconsistent")
        return best_bid, best_ask

    @staticmethod
    def _event_arrival_key(row: _EventRow) -> tuple[int, int, str, int, int]:
        return (
            row.received_monotonic_ns,
            row.received_wall_ms,
            row.connection_id,
            row.sequence_number,
            row.sub_index,
        )

    @staticmethod
    def _apply_tick_size_change(
        row: _EventRow,
        market_by_token: Mapping[str, PolymarketFiveMinuteMarket],
        state: dict[str, _BookState],
    ) -> None:
        token = str(row.event.get("asset_id") or row.asset_id)
        if token not in market_by_token or token not in state:
            raise ValueError("tick_size_change arrived without a proven token baseline")
        old_tick = _decimal(
            row.event.get("old_tick_size"),
            name="old tick size",
            minimum=Decimal("0.0001"),
            maximum=Decimal("0.1"),
        )
        new_tick = _decimal(
            row.event.get("new_tick_size"),
            name="new tick size",
            minimum=Decimal("0.0001"),
            maximum=Decimal("0.1"),
        )
        if old_tick != state[token].tick_size:
            raise ValueError("tick_size_change old value disagrees with replay state")
        source_time = _timestamp(
            row.event.get("timestamp"), name="tick_size_change timestamp"
        )
        current = state[token]
        if source_time < current.source_time_ms:
            raise ValueError("tick_size_change source time regressed")
        if any(price % new_tick != 0 for price in (*current.bids, *current.asks)):
            raise ValueError("existing book is not aligned to the new tick size")
        current.tick_size = new_tick
        current.source_time_ms = source_time
        current.provenance_sha256 = _canonical_sha256(
            {
                "schema_version": "polymarket-tick-transition-v1",
                "previous": current.provenance_sha256,
                "event_id": row.event_id,
                "event_sha256": row.event_sha256,
                "token_id": token,
                "old_tick_size": format(old_tick, "f"),
                "new_tick_size": format(new_tick, "f"),
                "source_time_ms": source_time,
            }
        )

    @staticmethod
    def _resolution(
        run_id: str,
        row: _EventRow,
        market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
    ) -> PolymarketResolutionEvidence:
        condition = str(row.event.get("market") or row.condition_id).lower()
        market = market_by_condition.get(condition)
        assets = row.event.get("assets_ids")
        outcomes = row.event.get("outcomes")
        winner = str(row.event.get("winning_asset_id") or "")
        winning_outcome = str(row.event.get("winning_outcome") or "")
        if market is None:
            raise ValueError("market_resolved references an unknown market")
        if assets != list(market.token_ids) or outcomes != ["Up", "Down"]:
            raise ValueError("market_resolved token/outcome mapping drifted")
        expected_outcome = (
            "Up" if winner == market.up_token_id else "Down"
            if winner == market.down_token_id
            else ""
        )
        if not expected_outcome or winning_outcome != expected_outcome:
            raise ValueError("market_resolved winner is inconsistent")
        resolved_at = _timestamp(
            row.event.get("timestamp"), name="market_resolved timestamp"
        )
        if resolved_at < market.end_ms:
            raise ValueError("market resolved before its documented end")
        return PolymarketResolutionEvidence(
            run_id=run_id,
            event_id=row.event_id,
            condition_id=condition,
            winning_asset_id=winner,
            winning_outcome=winning_outcome,
            resolved_at_ms=resolved_at,
            received_wall_ms=row.available_wall_ms,
            received_monotonic_ns=row.available_monotonic_ns,
            event_sha256=row.event_sha256,
        )

    @staticmethod
    def _recorded_book(
        run_id: str,
        row: _EventRow,
        market: PolymarketFiveMinuteMarket,
        token: str,
        state: _BookState,
    ) -> PolymarketRecordedBook:
        outcome = "Up" if token == market.up_token_id else "Down"
        snapshot = PaperBookSnapshot(
            venue="polymarket",
            market_id=market.condition_id,
            asset_id=token,
            bids=tuple(
                BookLevel(price, state.bids[price])
                for price in sorted(state.bids, reverse=True)
            ),
            asks=tuple(
                BookLevel(price, state.asks[price]) for price in sorted(state.asks)
            ),
            source_time_ms=state.source_time_ms,
            received_wall_ms=row.available_wall_ms,
            received_monotonic_ns=row.available_monotonic_ns,
            source_payload_sha256=state.provenance_sha256,
        ).validated()
        return PolymarketRecordedBook(
            run_id=run_id,
            event_id=row.event_id,
            event_type=row.event_type,
            connection_id=row.connection_id,
            sequence_number=row.sequence_number,
            sub_index=row.sub_index,
            market=market,
            outcome=outcome,
            tick_size=state.tick_size,
            snapshot=snapshot,
        )

    def book_for_event(
        self,
        event_id: str,
        token_id: str,
    ) -> PolymarketRecordedBook:
        try:
            return self._book_by_event_token[(str(event_id), str(token_id))]
        except KeyError as exc:
            raise KeyError("unknown replay book event/token") from exc

    def first_book_after_latency(
        self,
        decision: PolymarketRecordedBook,
        *,
        latency_ms: int,
    ) -> PolymarketRecordedBook | None:
        if decision.run_id != self.run_id:
            raise ValueError("decision book belongs to a different recorder run")
        if self.book_for_event(decision.event_id, decision.token_id) != decision:
            raise ValueError("decision book does not match immutable replay evidence")
        latency = int(latency_ms)
        if latency <= 0 or latency > 60_000:
            raise ValueError("latency_ms must lie in [1, 60000]")
        target = decision.received_monotonic_ns + latency * 1_000_000
        for candidate in self._books_by_token.get(decision.token_id, ()):
            if candidate.received_monotonic_ns < target:
                continue
            if candidate.received_wall_ms >= decision.market.end_ms:
                return None
            return candidate
        return None


__all__ = [
    "POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION",
    "PolymarketEvidenceReplay",
    "PolymarketRecordedBook",
    "PolymarketReplayDiagnostics",
    "PolymarketResolutionEvidence",
]
