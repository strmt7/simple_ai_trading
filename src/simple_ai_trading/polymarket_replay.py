"""Strict reconstruction of prospective Polymarket CLOB evidence."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Mapping, Sequence

from .paper_execution import BookLevel, PaperBookSnapshot
from .polymarket import (
    POLYMARKET_TAKER_ORDER_DELAY_MS,
    PolymarketFiveMinuteMarket,
    parse_clob_general_order_delay_seconds,
    parse_polymarket_five_minute_market,
    validate_clob_order_book,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_resolution import load_official_resolutions


_KNOWN_NO_BOOK_CHANGE_EVENTS = frozenset(
    {"last_trade_price", "new_market", "market_resolved"}
)
_BOOK_DEPTH_EVENTS = frozenset({"book", "price_change"})
_BEST_CORROBORATION_MAX_SOURCE_SKEW_MS = 1_000
_BEST_CORROBORATION_MAX_ARRIVAL_NS = 2_000_000_000
_CAUSAL_REORDER_MAX_SOURCE_SKEW_MS = 1_000
_CAUSAL_REORDER_MAX_ARRIVAL_NS = 2_000_000_000
_REPLAY_FETCH_SIZE = 4_096
_CAUSALLY_ORDERED_EVENTS = _BOOK_DEPTH_EVENTS | frozenset(
    {"best_bid_ask", "tick_size_change", "market_resolved"}
)
POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION = "polymarket-replay-diagnostics-v2"
_NAMED_LIVE_LANE_STREAMS = {
    "clob": "clob_market",
    "binance:combined:btc-eth-sol": "binance_spot",
    "rtds:binance:btc": "polymarket_rtds",
    "rtds:binance:eth": "polymarket_rtds",
    "rtds:binance:sol": "polymarket_rtds",
    "rtds:chainlink:btc": "polymarket_rtds",
    "rtds:chainlink:eth": "polymarket_rtds",
    "rtds:chainlink:sol": "polymarket_rtds",
}


def _named_live_lane(stream: str, connection_id: str) -> str | None:
    lane, separator, instance_id = connection_id.rpartition(":")
    expected_stream = _NAMED_LIVE_LANE_STREAMS.get(lane)
    if expected_stream is None:
        return None
    if (
        separator != ":"
        or len(instance_id) != 32
        or any(character not in "0123456789abcdef" for character in instance_id)
    ):
        raise ValueError("named stream connection has an invalid instance identity")
    if stream != expected_stream:
        raise ValueError("named stream connection is bound to the wrong stream")
    return lane


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
    segment_id: str
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
class PolymarketMarketExecutionEvidence:
    """Integrity-bound venue parameters observed for one market."""

    run_id: str
    condition_id: str
    observed_wall_ms: int
    observed_monotonic_ns: int
    maker_base_fee: int
    taker_base_fee: int
    taker_order_delay_enabled: bool
    general_order_delay_seconds: int
    minimum_order_age_seconds: int
    clob_info_sha256: str
    up_fee_rate_sha256: str
    down_fee_rate_sha256: str
    snapshot_sha256: str

    @property
    def taker_order_delay_ms(self) -> int:
        return POLYMARKET_TAKER_ORDER_DELAY_MS if self.taker_order_delay_enabled else 0

    def asdict(self) -> dict[str, object]:
        item = self.validated()
        return {
            "run_id": item.run_id,
            "condition_id": item.condition_id,
            "observed_wall_ms": item.observed_wall_ms,
            "observed_monotonic_ns": item.observed_monotonic_ns,
            "maker_base_fee": item.maker_base_fee,
            "taker_base_fee": item.taker_base_fee,
            "taker_order_delay_enabled": item.taker_order_delay_enabled,
            "taker_order_delay_ms": item.taker_order_delay_ms,
            "general_order_delay_seconds": item.general_order_delay_seconds,
            "minimum_order_age_seconds": item.minimum_order_age_seconds,
            "clob_info_sha256": item.clob_info_sha256,
            "up_fee_rate_sha256": item.up_fee_rate_sha256,
            "down_fee_rate_sha256": item.down_fee_rate_sha256,
            "snapshot_sha256": item.snapshot_sha256,
        }

    def validated(self) -> "PolymarketMarketExecutionEvidence":
        if self.general_order_delay_seconds != 0:
            raise ValueError(
                "Round 9 does not model a nonzero CLOB general order delay"
            )
        if (
            not self.run_id
            or not self.condition_id
            or self.observed_wall_ms < 0
            or self.observed_monotonic_ns < 0
            or self.maker_base_fee < 0
            or self.taker_base_fee < 0
            or not isinstance(self.taker_order_delay_enabled, bool)
            or self.general_order_delay_seconds < 0
            or self.minimum_order_age_seconds < 0
            or len(self.clob_info_sha256) != 64
            or len(self.up_fee_rate_sha256) != 64
            or len(self.down_fee_rate_sha256) != 64
            or len(self.snapshot_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for digest in (
                    self.clob_info_sha256,
                    self.up_fee_rate_sha256,
                    self.down_fee_rate_sha256,
                    self.snapshot_sha256,
                )
                for character in digest
            )
        ):
            raise ValueError("Polymarket market execution evidence is invalid")
        return self


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
    source: str


@dataclass(frozen=True)
class PolymarketReplayDiagnostics:
    schema_version: str
    continuity_mode: str
    stream_gap_count: int
    clob_connection_segment_count: int
    state_reset_count: int
    discarded_uncorroborated_best_count: int
    book_sample_interval_ms: int
    book_state_transition_count: int
    materialized_book_count: int
    suppressed_book_count: int
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


@dataclass(slots=True)
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


@dataclass(slots=True)
class _CausalReplayMetrics:
    total_event_count: int = 0
    causally_ordered_event_count: int = 0
    late_event_count: int = 0
    maximum_source_regression_ms: int = 0
    maximum_late_arrival_delay_ns: int = 0
    deferred_event_count: int = 0
    maximum_availability_delay_ns: int = 0


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
        market_execution_evidence: tuple[PolymarketMarketExecutionEvidence, ...] = (),
    ) -> None:
        self.run_id = run_id
        self.markets = markets
        self.books = books
        self.resolutions = resolutions
        self.diagnostics = diagnostics
        validated_execution_evidence = tuple(
            item.validated() for item in market_execution_evidence
        )
        if any(item.run_id != run_id for item in validated_execution_evidence):
            raise ValueError("market execution evidence belongs to another replay run")
        if len(
            {
                (
                    item.condition_id,
                    item.observed_monotonic_ns,
                    item.snapshot_sha256,
                )
                for item in validated_execution_evidence
            }
        ) != len(validated_execution_evidence):
            raise ValueError("market execution evidence identities are duplicated")
        self.market_execution_evidence = validated_execution_evidence
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
        allow_segmented_gaps: bool = False,
        book_sample_interval_ms: int = 0,
        condition_ids: Sequence[str] | None = None,
    ) -> "PolymarketEvidenceReplay":
        sample_interval_ms = int(book_sample_interval_ms)
        if sample_interval_ms < 0 or sample_interval_ms > 5_000:
            raise ValueError("book_sample_interval_ms must lie in [0, 5000]")
        selected_conditions: tuple[str, ...] | None = None
        if condition_ids is not None:
            normalized = tuple(
                sorted({str(value or "").strip().lower() for value in condition_ids})
            )
            if not normalized or any(not value for value in normalized):
                raise ValueError("condition_ids must contain unique non-empty values")
            selected_conditions = normalized
        connection = store.connect()
        selected = str(run_id or "").strip()
        if not selected:
            latest_query = (
                """
                SELECT run_id FROM polymarket_recorder_run
                WHERE status IN ('complete', 'degraded')
                ORDER BY ended_at_ms DESC, run_id DESC LIMIT 1
                """
                if allow_segmented_gaps
                else """
                SELECT run_id FROM polymarket_recorder_run
                WHERE status = 'complete'
                ORDER BY ended_at_ms DESC, run_id DESC LIMIT 1
                """
            )
            row = connection.execute(latest_query).fetchone()
            if row is None:
                qualifier = "finished" if allow_segmented_gaps else "complete"
                raise ValueError(f"no {qualifier} Polymarket recorder run is available")
            selected = str(row[0])
        run = connection.execute(
            """
            SELECT status, error FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if run is None:
            raise ValueError(f"unknown Polymarket recorder run: {selected}")
        run_status = str(run[0])
        if run_status != "complete" and not allow_segmented_gaps:
            raise ValueError("Polymarket replay requires a complete gap-free run")
        if run_status not in {"complete", "degraded"}:
            raise ValueError("Polymarket replay requires a finished valid run")
        if str(run[1] or "").strip():
            raise ValueError("Polymarket replay refuses runs with recorder errors")
        integrity = store.integrity_errors(selected)
        if integrity:
            raise ValueError(
                "Polymarket replay evidence failed integrity: " + "; ".join(integrity)
            )
        gap_count = cls.validate_stream_gaps(
            store,
            selected,
            allow_segmented_gaps=allow_segmented_gaps,
        )

        markets = cls._load_markets(
            store,
            selected,
            condition_ids=selected_conditions,
        )
        market_execution_evidence = cls._load_market_execution_evidence(
            store,
            selected,
            condition_ids=selected_conditions,
        )
        if {item.condition_id for item in market_execution_evidence} != {
            market.condition_id for market in markets
        }:
            raise ValueError("Polymarket execution evidence does not cover every market")
        continuity_mode = "segmented" if gap_count else "strict"
        causal_metrics = _CausalReplayMetrics()
        events = cls._iter_causal_events(
            store,
            selected,
            markets,
            causal_metrics,
        )
        (
            books,
            resolutions,
            segment_count,
            reset_count,
            discarded_best_count,
            transition_count,
        ) = cls._reconstruct(
            selected,
            markets,
            events,
            book_sample_interval_ms=sample_interval_ms,
        )
        resolutions = cls._merge_external_resolutions(
            store,
            selected,
            resolutions,
            condition_ids=selected_conditions,
        )
        diagnostics = PolymarketReplayDiagnostics(
            schema_version=POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION,
            continuity_mode=continuity_mode,
            stream_gap_count=int(gap_count),
            clob_connection_segment_count=segment_count,
            state_reset_count=reset_count,
            discarded_uncorroborated_best_count=discarded_best_count,
            book_sample_interval_ms=sample_interval_ms,
            book_state_transition_count=transition_count,
            materialized_book_count=len(books),
            suppressed_book_count=transition_count - len(books),
            total_event_count=causal_metrics.total_event_count,
            causally_ordered_event_count=(causal_metrics.causally_ordered_event_count),
            late_event_count=causal_metrics.late_event_count,
            maximum_source_regression_ms=(causal_metrics.maximum_source_regression_ms),
            maximum_late_arrival_delay_ns=(
                causal_metrics.maximum_late_arrival_delay_ns
            ),
            deferred_event_count=causal_metrics.deferred_event_count,
            maximum_availability_delay_ns=(
                causal_metrics.maximum_availability_delay_ns
            ),
        )
        if not books:
            raise ValueError("Polymarket replay contains no validated book states")
        return cls(
            run_id=selected,
            markets=markets,
            books=books,
            resolutions=resolutions,
            diagnostics=diagnostics,
            market_execution_evidence=market_execution_evidence,
        )

    @staticmethod
    def validate_stream_gaps(
        store: PolymarketEvidenceStore,
        run_id: str,
        *,
        allow_segmented_gaps: bool,
    ) -> int:
        connection = store.connect()
        rows = connection.execute(
            """
            SELECT stream, connection_id, opened_at_ms, last_sequence_number
            FROM polymarket_stream_gap
            WHERE run_id = ? ORDER BY opened_at_ms, gap_id
            """,
            [run_id],
        ).fetchall()
        if rows and not allow_segmented_gaps:
            raise ValueError("Polymarket replay refuses runs with stream gaps")
        segmentable_streams = {"clob_market", "binance_spot", "polymarket_rtds"}
        seen_connections: set[tuple[str, str]] = set()
        for stream, connection_id, opened_at_ms, last_sequence_number in rows:
            normalized_stream = str(stream)
            normalized_connection = str(connection_id)
            if normalized_stream not in segmentable_streams:
                raise ValueError(
                    "segmented Polymarket replay found a non-segmentable stream gap"
                )
            _named_live_lane(normalized_stream, normalized_connection)
            connection_key = (normalized_stream, normalized_connection)
            if connection_key in seen_connections:
                raise ValueError("stream connection has duplicate gap evidence")
            seen_connections.add(connection_key)
            evidence = connection.execute(
                """
                SELECT count(*), max(sequence_number), max(received_wall_ms),
                       sum(CASE WHEN received_wall_ms > ? THEN 1 ELSE 0 END)
                FROM polymarket_raw_message
                WHERE run_id = ? AND stream = ? AND connection_id = ?
                """,
                [
                    int(opened_at_ms),
                    run_id,
                    normalized_stream,
                    normalized_connection,
                ],
            ).fetchone()
            evidence_count = 0 if evidence is None else int(evidence[0] or 0)
            expected_last_sequence = int(last_sequence_number)
            if expected_last_sequence == 0:
                if evidence_count != 0:
                    raise ValueError(
                        "zero-sequence stream gap has matching message evidence"
                    )
                continue
            if evidence_count < 1:
                raise ValueError("stream gap has no matching connection evidence")
            if int(evidence[1]) != expected_last_sequence:
                raise ValueError("stream gap does not close the final sequence")
            if int(evidence[2]) > int(opened_at_ms) or int(evidence[3] or 0):
                raise ValueError("stream gap precedes messages on its connection")

        connection_rows = connection.execute(
            """
            SELECT stream, connection_id,
                   min(received_monotonic_ns), max(received_monotonic_ns),
                   min(sequence_number)
            FROM polymarket_raw_message
            WHERE run_id = ?
              AND stream IN ('clob_market', 'binance_spot', 'polymarket_rtds')
            GROUP BY stream, connection_id
            """,
            [run_id],
        ).fetchall()
        named_lanes: dict[str, list[tuple[int, int, str, str]]] = {}
        for stream, connection_id, first_ns, last_ns, first_sequence in connection_rows:
            normalized_stream = str(stream)
            normalized_connection = str(connection_id)
            lane = _named_live_lane(normalized_stream, normalized_connection)
            if lane is None:
                continue
            if int(first_sequence) != 1:
                raise ValueError("named stream connection does not begin at sequence one")
            named_lanes.setdefault(lane, []).append(
                (
                    int(first_ns),
                    int(last_ns),
                    normalized_stream,
                    normalized_connection,
                )
            )
        for lane_connections in named_lanes.values():
            lane_connections.sort(key=lambda item: (item[0], item[3]))
            for current, following in zip(
                lane_connections,
                lane_connections[1:],
                strict=False,
            ):
                if current[1] > following[0]:
                    raise ValueError("named stream connection segments overlap")
                if (current[2], current[3]) not in seen_connections:
                    raise ValueError(
                        "named stream connection transition has no gap evidence"
                    )
        return len(rows)

    @staticmethod
    def _merge_external_resolutions(
        store: PolymarketEvidenceStore,
        run_id: str,
        recorded: tuple[PolymarketResolutionEvidence, ...],
        *,
        condition_ids: Sequence[str] | None = None,
    ) -> tuple[PolymarketResolutionEvidence, ...]:
        allowed = None if condition_ids is None else set(condition_ids)
        recorded_by_condition: dict[str, PolymarketResolutionEvidence] = {}
        for item in recorded:
            if allowed is not None and item.condition_id not in allowed:
                continue
            if item.condition_id in recorded_by_condition:
                raise ValueError("Polymarket replay has duplicate resolution events")
            recorded_by_condition[item.condition_id] = item
        official_by_condition: dict[str, PolymarketResolutionEvidence] = {}
        for item in load_official_resolutions(store, run_id=run_id):
            if allowed is not None and item.condition_id not in allowed:
                continue
            external = PolymarketResolutionEvidence(
                run_id=item.run_id,
                event_id=item.resolution_id,
                condition_id=item.condition_id,
                winning_asset_id=item.winning_asset_id,
                winning_outcome=item.winning_outcome,
                resolved_at_ms=item.observed_wall_ms,
                received_wall_ms=item.observed_wall_ms,
                received_monotonic_ns=0,
                event_sha256=item.evidence_sha256,
                source="clob_gamma_crosscheck",
            )
            observed = recorded_by_condition.get(item.condition_id)
            if observed is not None:
                if (
                    observed.winning_asset_id != external.winning_asset_id
                    or observed.winning_outcome != external.winning_outcome
                ):
                    raise ValueError(
                        "recorded and finalized Polymarket resolutions disagree"
                    )
            if item.condition_id in official_by_condition:
                raise ValueError("Polymarket replay has duplicate official resolutions")
            official_by_condition[item.condition_id] = external
        return tuple(
            sorted(
                official_by_condition.values(),
                key=lambda item: (
                    item.received_wall_ms,
                    item.condition_id,
                    item.event_id,
                ),
            )
        )

    @staticmethod
    def load_markets(
        store: PolymarketEvidenceStore,
        *,
        run_id: str,
        condition_ids: Sequence[str] | None = None,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        """Load validated immutable market metadata without reconstructing books."""

        selected_conditions: tuple[str, ...] | None = None
        if condition_ids is not None:
            normalized = tuple(
                sorted({str(value or "").strip().lower() for value in condition_ids})
            )
            if not normalized or any(not value for value in normalized):
                raise ValueError("condition_ids must contain unique non-empty values")
            selected_conditions = normalized
        return PolymarketEvidenceReplay._load_markets(
            store,
            str(run_id),
            condition_ids=selected_conditions,
        )

    @staticmethod
    def _load_markets(
        store: PolymarketEvidenceStore,
        run_id: str,
        *,
        condition_ids: Sequence[str] | None = None,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        parameters: list[object] = [run_id]
        condition_filter = ""
        if condition_ids is not None:
            placeholders = ", ".join("?" for _ in condition_ids)
            condition_filter = f" AND condition_id IN ({placeholders})"
            parameters.extend(condition_ids)
        rows = (
            store.connect()
            .execute(
                f"""
            SELECT condition_id, gamma_payload_json
            FROM polymarket_market_snapshot
            WHERE run_id = ?{condition_filter}
            ORDER BY event_start_ms, asset
            """,
                parameters,
            )
            .fetchall()
        )
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
    def _load_market_execution_evidence(
        store: PolymarketEvidenceStore,
        run_id: str,
        *,
        condition_ids: Sequence[str] | None = None,
    ) -> tuple[PolymarketMarketExecutionEvidence, ...]:
        parameters: list[object] = [run_id]
        condition_filter = ""
        if condition_ids is not None:
            placeholders = ", ".join("?" for _ in condition_ids)
            condition_filter = f" AND condition_id IN ({placeholders})"
            parameters.extend(condition_ids)
        rows = (
            store.connect()
            .execute(
                f"""
                SELECT condition_id, observed_wall_ms, observed_monotonic_ns,
                       maker_base_fee, taker_base_fee,
                       taker_order_delay_enabled, minimum_order_age_seconds,
                       clob_info_json, clob_info_sha256, up_fee_rate_sha256,
                       down_fee_rate_sha256, snapshot_sha256
                FROM polymarket_market_snapshot
                WHERE run_id = ?{condition_filter}
                ORDER BY observed_monotonic_ns, condition_id, snapshot_id
                """,
                parameters,
            )
            .fetchall()
        )
        evidence_items: list[PolymarketMarketExecutionEvidence] = []
        for row in rows:
            try:
                clob_info = json.loads(str(row[7]))
            except json.JSONDecodeError as exc:
                raise ValueError("stored CLOB market info is invalid JSON") from exc
            if not isinstance(clob_info, Mapping):
                raise ValueError("stored CLOB market info must be an object")
            general_delay_seconds = parse_clob_general_order_delay_seconds(clob_info)
            if bool(row[5]) != (clob_info.get("itode") is True):
                raise ValueError("stored CLOB taker-delay flag is inconsistent")
            evidence_items.append(
                PolymarketMarketExecutionEvidence(
                    run_id=run_id,
                    condition_id=str(row[0]),
                    observed_wall_ms=int(row[1]),
                    observed_monotonic_ns=int(row[2]),
                    maker_base_fee=int(row[3]),
                    taker_base_fee=int(row[4]),
                    taker_order_delay_enabled=bool(row[5]),
                    general_order_delay_seconds=general_delay_seconds,
                    minimum_order_age_seconds=int(row[6]),
                    clob_info_sha256=str(row[8]),
                    up_fee_rate_sha256=str(row[9]),
                    down_fee_rate_sha256=str(row[10]),
                    snapshot_sha256=str(row[11]),
                ).validated()
            )
        evidence = tuple(evidence_items)
        if not evidence:
            raise ValueError("Polymarket replay contains no market execution evidence")
        return evidence

    @classmethod
    def _iter_causal_events(
        cls,
        store: PolymarketEvidenceStore,
        run_id: str,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        metrics: _CausalReplayMetrics,
    ) -> Iterator[_EventRow]:
        market_by_condition = {market.condition_id: market for market in markets}
        conditions = tuple(sorted(market_by_condition))
        source_watermarks: dict[tuple[str, str], tuple[int, int]] = {}
        availability: dict[str, tuple[int, int]] = {}
        raw_group: list[tuple[object, ...]] = []
        group_monotonic_ns: int | None = None

        def release_group(
            grouped: list[tuple[object, ...]],
        ) -> tuple[_EventRow, ...]:
            prepared: list[tuple[_EventRow, str, int | None]] = []
            reorder_slots: dict[tuple[str, str, int], list[int]] = {}
            reorder_epoch: dict[tuple[str, str], int] = {}
            release_wall_by_condition: dict[str, int] = {}
            for raw_row in grouped:
                payload = raw_row[4]
                if not isinstance(payload, Mapping):
                    raise ValueError("stored CLOB event must be an object")
                row = _EventRow(
                    event_id=str(raw_row[0]),
                    event_type=str(raw_row[1]),
                    condition_id=str(raw_row[2]),
                    asset_id=str(raw_row[3]),
                    event=dict(payload),
                    event_sha256=str(raw_row[5]),
                    connection_id=str(raw_row[6]),
                    sequence_number=int(raw_row[7]),
                    received_wall_ms=int(raw_row[8]),
                    received_monotonic_ns=int(raw_row[9]),
                    available_wall_ms=int(raw_row[8]),
                    available_monotonic_ns=int(raw_row[9]),
                    sub_index=int(raw_row[10]),
                )
                metrics.total_event_count += 1
                condition = str(row.event.get("market") or row.condition_id).lower()
                if (
                    row.event_type not in _CAUSALLY_ORDERED_EVENTS
                    or condition not in market_by_condition
                ):
                    prepared.append((row, condition, None))
                    continue

                metrics.causally_ordered_event_count += 1
                source_time = _timestamp(
                    row.event.get("timestamp"),
                    name=f"{row.event_type} timestamp",
                )
                segment_key = (condition, row.connection_id)
                maximum_source_time, maximum_source_arrival_ns = source_watermarks.get(
                    segment_key, (-1, 0)
                )
                if source_time < maximum_source_time:
                    source_skew = maximum_source_time - source_time
                    arrival_delay = (
                        row.received_monotonic_ns - maximum_source_arrival_ns
                    )
                    metrics.late_event_count += 1
                    metrics.maximum_source_regression_ms = max(
                        metrics.maximum_source_regression_ms,
                        source_skew,
                    )
                    metrics.maximum_late_arrival_delay_ns = max(
                        metrics.maximum_late_arrival_delay_ns,
                        arrival_delay,
                    )
                    if (
                        source_skew > _CAUSAL_REORDER_MAX_SOURCE_SKEW_MS
                        or not 0 <= arrival_delay <= _CAUSAL_REORDER_MAX_ARRIVAL_NS
                    ):
                        raise ValueError(
                            "CLOB event exceeded the bounded causal reorder window"
                        )
                else:
                    source_watermarks[segment_key] = (
                        source_time,
                        row.received_monotonic_ns,
                    )
                index = len(prepared)
                if row.event_type in {"price_change", "best_bid_ask"}:
                    epoch = reorder_epoch.get(segment_key, 0)
                    reorder_slots.setdefault((*segment_key, epoch), []).append(index)
                else:
                    reorder_epoch[segment_key] = reorder_epoch.get(segment_key, 0) + 1
                release_wall_by_condition[condition] = max(
                    release_wall_by_condition.get(condition, 0),
                    row.received_wall_ms,
                )
                prepared.append((row, condition, source_time))

            for indices in reorder_slots.values():
                ordered = sorted(
                    (prepared[index] for index in indices),
                    key=lambda item: (
                        int(item[2]) if item[2] is not None else -1,
                        cls._event_arrival_key(item[0]),
                    ),
                )
                for index, item in zip(indices, ordered, strict=True):
                    prepared[index] = item

            output: list[_EventRow] = []
            for row, condition, source_time in prepared:
                if source_time is None:
                    output.append(row)
                    continue
                release_wall_ms = release_wall_by_condition[condition]
                wall_delay_ns = (
                    max(
                        0,
                        release_wall_ms - row.received_wall_ms,
                    )
                    * 1_000_000
                )
                available_monotonic_ns, available_wall_ms = availability.get(
                    condition,
                    (0, 0),
                )
                causal_monotonic_ns = row.received_monotonic_ns + wall_delay_ns
                if causal_monotonic_ns > available_monotonic_ns:
                    available_monotonic_ns = causal_monotonic_ns
                    available_wall_ms = max(
                        release_wall_ms,
                        row.received_wall_ms,
                    )
                else:
                    available_monotonic_ns += 1
                    available_wall_ms = max(
                        available_wall_ms,
                        release_wall_ms,
                        row.received_wall_ms,
                    )
                availability[condition] = (
                    available_monotonic_ns,
                    available_wall_ms,
                )
                availability_delay = available_monotonic_ns - row.received_monotonic_ns
                if availability_delay > 0:
                    metrics.deferred_event_count += 1
                    metrics.maximum_availability_delay_ns = max(
                        metrics.maximum_availability_delay_ns,
                        availability_delay,
                    )
                row.available_wall_ms = available_wall_ms
                row.available_monotonic_ns = available_monotonic_ns
                output.append(row)
            return tuple(output)

        for decoded in store.iter_public_events(
            run_id,
            streams=("clob_market", "clob_rest_book"),
            condition_ids=conditions,
            verified_source=True,
        ):
            raw_row = (
                decoded.event_id,
                decoded.event_type,
                decoded.condition_id,
                decoded.asset_id,
                decoded.event,
                decoded.event_sha256,
                decoded.connection_id,
                decoded.sequence_number,
                decoded.received_wall_ms,
                decoded.received_monotonic_ns,
                decoded.sub_index,
            )
            received_monotonic_ns = decoded.received_monotonic_ns
            if (
                group_monotonic_ns is not None
                and received_monotonic_ns != group_monotonic_ns
            ):
                yield from release_group(raw_group)
                raw_group = []
            group_monotonic_ns = received_monotonic_ns
            raw_group.append(raw_row)
        if raw_group:
            yield from release_group(raw_group)

    @classmethod
    def _reconstruct(
        cls,
        run_id: str,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        events: Iterable[_EventRow],
        *,
        book_sample_interval_ms: int,
    ) -> tuple[
        tuple[PolymarketRecordedBook, ...],
        tuple[PolymarketResolutionEvidence, ...],
        int,
        int,
        int,
        int,
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
        active_connection: dict[str, str] = {}
        segments: set[tuple[str, str]] = set()
        state_reset_count = 0
        discarded_best_count = 0
        last_materialized_ns: dict[str, int] = {}
        transition_count = 0

        def flush(condition_id: str) -> None:
            nonlocal transition_count
            batch = pending.pop(condition_id, None)
            if batch is not None:
                materialized, transitions = cls._flush_book_batch(
                    run_id,
                    batch,
                    market_by_condition,
                    market_by_token,
                    state,
                    pending_best,
                    last_materialized_ns,
                    book_sample_interval_ms=book_sample_interval_ms,
                )
                books.extend(materialized)
                transition_count += transitions

        for row in events:
            event_type = row.event_type
            condition = str(row.event.get("market") or row.condition_id).lower()
            if event_type in _BOOK_DEPTH_EVENTS | {
                "best_bid_ask",
                "tick_size_change",
            }:
                market = market_by_condition.get(condition)
                if market is None:
                    raise ValueError(f"{event_type} event references an unknown market")
                prior_connection = active_connection.get(condition)
                if prior_connection != row.connection_id:
                    flush(condition)
                    if prior_connection is not None:
                        state_reset_count += 1
                    for token in market.token_ids:
                        state.pop(token, None)
                        last_materialized_ns.pop(token, None)
                        discarded_best_count += len(pending_best.pop(token, ()))
                    active_connection[condition] = row.connection_id
                    segments.add((condition, row.connection_id))
            if event_type in _BOOK_DEPTH_EVENTS:
                if row.condition_id and row.condition_id.lower() != condition:
                    raise ValueError(f"{event_type} event condition identity drifted")
                source_time = _timestamp(
                    row.event.get("timestamp"),
                    name=f"{event_type} timestamp",
                )
                batch = pending.get(condition)
                if batch is None or source_time != batch.source_time_ms:
                    flush(condition)
                    pending[condition] = _PendingBookBatch(
                        condition_id=condition,
                        source_time_ms=source_time,
                        rows=[row],
                    )
                else:
                    batch.rows.append(row)
            elif event_type == "best_bid_ask":
                source_time = _timestamp(
                    row.event.get("timestamp"), name="best_bid_ask timestamp"
                )
                batch = pending.get(condition)
                if batch is not None and source_time == batch.source_time_ms:
                    batch.rows.append(row)
                else:
                    if batch is not None and (
                        any(item.event_type == "book" for item in batch.rows)
                        or cls._delta_batch_has_complete_top(batch, state)
                    ):
                        flush(condition)
                    cls._observe_best_bid_ask(
                        row,
                        market_by_condition,
                        market_by_token,
                        state,
                        pending_best,
                    )
            elif event_type == "tick_size_change":
                flush(condition)
                cls._apply_tick_size_change(row, market_by_token, state)
            elif event_type == "market_resolved":
                flush(condition)
                resolutions.append(cls._resolution(run_id, row, market_by_condition))
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
        return (
            tuple(books),
            tuple(resolutions),
            len(segments),
            state_reset_count,
            discarded_best_count,
            transition_count,
        )

    @staticmethod
    def _delta_batch_has_complete_top(
        batch: _PendingBookBatch,
        state: Mapping[str, _BookState],
    ) -> bool:
        simulated: dict[
            str,
            tuple[dict[Decimal, Decimal], dict[Decimal, Decimal]],
        ] = {}
        reported: dict[str, tuple[Decimal, Decimal]] = {}
        changed = False
        for row in batch.rows:
            if row.event_type != "price_change":
                continue
            changes = row.event.get("price_changes")
            if not isinstance(changes, list) or not changes:
                return False
            for change in changes:
                if not isinstance(change, Mapping):
                    return False
                token = str(change.get("asset_id") or "")
                current = state.get(token)
                if current is None:
                    return False
                books = simulated.setdefault(
                    token,
                    (dict(current.bids), dict(current.asks)),
                )
                side = str(change.get("side") or "").upper()
                if side not in {"BUY", "SELL"}:
                    return False
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
                levels = books[0] if side == "BUY" else books[1]
                if size == 0:
                    levels.pop(price, None)
                else:
                    levels[price] = size
                reported[token] = (
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
                changed = True
        if not changed:
            return False
        return all(
            reported[token]
            == (
                max(bids, default=Decimal("0")),
                min(asks, default=Decimal("1")),
            )
            for token, (bids, asks) in simulated.items()
        )

    @classmethod
    def _flush_book_batch(
        cls,
        run_id: str,
        batch: _PendingBookBatch,
        market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
        market_by_token: Mapping[str, PolymarketFiveMinuteMarket],
        state: dict[str, _BookState],
        pending_best: dict[str, list[_EventRow]],
        last_materialized_ns: dict[str, int],
        *,
        book_sample_interval_ms: int,
    ) -> tuple[tuple[PolymarketRecordedBook, ...], int]:
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
                    raise ValueError(
                        "full book timestamp differs from its atomic batch"
                    )
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
            elif row.event_type == "best_bid_ask":
                token = str(row.event.get("asset_id") or row.asset_id)
                if market_by_token.get(token) != market:
                    raise ValueError(
                        "best_bid_ask references an unknown market or token"
                    )
                if (
                    _timestamp(
                        row.event.get("timestamp"), name="best_bid_ask timestamp"
                    )
                    != batch.source_time_ms
                ):
                    raise ValueError(
                        "best_bid_ask timestamp differs from its atomic batch"
                    )
                cls._best_bid_ask_values(row)
                operations.setdefault(token, []).append(("best", row, dict(row.event)))

        output: list[PolymarketRecordedBook] = []
        transition_count = 0
        for token in sorted(operations):
            current = state.get(token)
            previous_provenance = current.provenance_sha256 if current else ""
            depth_changed = False
            pending_hash = ""
            pending_checksum: tuple[Decimal, Decimal] | None = None
            pending_changes: list[tuple[_EventRow, Mapping[str, object]]] = []
            relevant_rows: list[_EventRow] = []

            def flush_changes() -> None:
                nonlocal current, depth_changed, pending_hash, pending_checksum
                nonlocal pending_changes
                if not pending_changes:
                    return
                if current is None:
                    event_ids = [item.event_id for item, _change in pending_changes]
                    raise ValueError(
                        "price_change arrived without a proven token baseline: "
                        f"condition={batch.condition_id} token={token} "
                        f"source_time_ms={batch.source_time_ms} "
                        f"events={event_ids}"
                    )
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
                    observed_checksum = (
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
                    if reported_checksum is None:
                        reported_checksum = observed_checksum
                    elif reported_checksum != observed_checksum:
                        raise ValueError(
                            "same-hash price_change fragments report different "
                            "best bid/ask checksums"
                        )
                    relevant_rows.append(change_row)
                expected = (
                    max(current.bids, default=Decimal("0")),
                    min(current.asks, default=Decimal("1")),
                )
                if reported_checksum != expected:
                    raise ValueError(
                        "price_change best bid/ask checksum disagrees with "
                        "atomic depth: "
                        f"condition={batch.condition_id} token={token} "
                        f"source_time_ms={batch.source_time_ms} "
                        f"reported={reported_checksum} expected={expected} "
                        f"events={[item.event_id for item, _change in pending_changes]}"
                    )
                current.source_time_ms = batch.source_time_ms
                current.book_hash = pending_hash
                depth_changed = True
                relevant_rows.extend(
                    cls._consume_pending_best(
                        token,
                        batch,
                        current,
                        [item for item, _change in pending_changes],
                        pending_best,
                    )
                )
                pending_hash = ""
                pending_checksum = None
                pending_changes = []

            for operation, row, payload in operations.get(token, ()):
                if operation == "change":
                    if not isinstance(payload, Mapping):
                        raise TypeError("internal price_change payload is malformed")
                    book_hash = str(payload.get("hash") or "").strip()
                    if not book_hash:
                        raise ValueError("price_change is missing its order-book hash")
                    change_checksum = (
                        _decimal(
                            payload.get("best_bid"),
                            name="price_change best_bid",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                        _decimal(
                            payload.get("best_ask"),
                            name="price_change best_ask",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                    )
                    same_message = bool(
                        pending_changes
                        and pending_changes[-1][0].event_id == row.event_id
                    )
                    same_fragment = bool(
                        pending_changes
                        and pending_hash == book_hash
                        and pending_checksum == change_checksum
                    )
                    if pending_changes and not same_message and not same_fragment:
                        flush_changes()
                    if pending_checksum is None:
                        pending_checksum = change_checksum
                    elif pending_checksum != change_checksum:
                        raise ValueError(
                            "one price_change message reports multiple best "
                            "bid/ask checksums for the same token"
                        )
                    pending_hash = book_hash
                    pending_changes.append((row, payload))
                    continue

                if operation == "best":
                    local_state = dict(state)
                    if current is not None:
                        local_state[token] = current
                    cls._observe_best_bid_ask(
                        row,
                        market_by_condition,
                        market_by_token,
                        local_state,
                        pending_best,
                    )
                    continue

                if not isinstance(payload, PaperBookSnapshot):
                    raise TypeError("internal full-book payload is malformed")
                book_hash = str(row.event.get("hash") or "").strip()
                if not book_hash:
                    raise ValueError("full book is missing its order-book hash")
                if pending_changes and pending_hash == book_hash:
                    last_change = pending_changes[-1][1]
                    side = str(last_change.get("side") or "").upper()
                    price = _decimal(
                        last_change.get("price"),
                        name="corroborated price_change price",
                        minimum=Decimal("0.0001"),
                        maximum=Decimal("0.9999"),
                    )
                    size = _decimal(
                        last_change.get("size"),
                        name="corroborated price_change size",
                        minimum=Decimal("0"),
                    )
                    levels = payload.bids if side == "BUY" else payload.asks
                    if side not in {"BUY", "SELL"}:
                        raise ValueError("price_change side must be BUY or SELL")
                    level_size = next(
                        (level.quantity for level in levels if level.price == price),
                        Decimal("0"),
                    )
                    expected_best = (
                        max(
                            (level.price for level in payload.bids),
                            default=Decimal("0"),
                        ),
                        min(
                            (level.price for level in payload.asks),
                            default=Decimal("1"),
                        ),
                    )
                    reported_best = (
                        _decimal(
                            last_change.get("best_bid"),
                            name="corroborated price_change best_bid",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                        _decimal(
                            last_change.get("best_ask"),
                            name="corroborated price_change best_ask",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                    )
                    if level_size != size or reported_best != expected_best:
                        raise ValueError(
                            "full book does not corroborate its same-hash price_change"
                        )
                    relevant_rows.extend(
                        change_row for change_row, _change in pending_changes
                    )
                    pending_changes = []
                    pending_hash = ""
                    pending_checksum = None
                else:
                    flush_changes()
                active_tick = current.tick_size if current else market.tick_size
                if any(
                    level.price % active_tick != 0
                    for level in (*payload.bids, *payload.asks)
                ):
                    raise ValueError("full book contains a price off the active tick")
                payload_bids = {level.price: level.quantity for level in payload.bids}
                payload_asks = {level.price: level.quantity for level in payload.asks}
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
                relevant_rows.extend(
                    cls._consume_pending_best(
                        token,
                        batch,
                        current,
                        [row],
                        pending_best,
                    )
                )
            flush_changes()

            if current is None:
                raise ValueError("depth transition has no proven token baseline")
            if not depth_changed:
                continue

            state[token] = current
            transition_count += 1
            unique_rows = {row.event_id: row for row in relevant_rows}
            evidence_rows = sorted(unique_rows.values(), key=cls._event_arrival_key)
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
            previous_materialized_ns = last_materialized_ns.get(token)
            if (
                previous_materialized_ns is not None
                and selected_row.available_monotonic_ns - previous_materialized_ns
                < book_sample_interval_ms * 1_000_000
            ):
                continue
            output.append(
                cls._recorded_book(
                    run_id,
                    selected_row,
                    market,
                    token,
                    current,
                )
            )
            last_materialized_ns[token] = selected_row.available_monotonic_ns
        return tuple(output), transition_count

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
        existing = pending_best.get(token)
        if existing:
            first = existing[0]
            first_source_time = _timestamp(
                first.event.get("timestamp"), name="best_bid_ask timestamp"
            )
            same_source_batch = source_time == first_source_time
            repeated_observation = (
                first_source_time <= source_time
                and source_time - first_source_time
                <= _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
                and observed == cls._best_bid_ask_values(first)
            )
            if (same_source_batch or repeated_observation) and abs(
                row.received_monotonic_ns - first.received_monotonic_ns
            ) <= _BEST_CORROBORATION_MAX_ARRIVAL_NS:
                existing.append(row)
                return
            raise ValueError(
                "best_bid_ask advanced before its prior observation was "
                "corroborated: "
                f"condition={condition} token={token} "
                f"pending_event={first.event_id} "
                f"pending_source_time_ms={first_source_time} "
                f"pending_values={cls._best_bid_ask_values(first)} "
                f"current_event={row.event_id} "
                f"current_source_time_ms={source_time} "
                f"current_values={observed}"
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
        consumed: list[_EventRow] = []
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
            arrival_delta = completion.received_monotonic_ns - row.received_monotonic_ns
            if abs(arrival_delta) > _BEST_CORROBORATION_MAX_ARRIVAL_NS:
                raise ValueError(
                    "best_bid_ask was not arrival-adjacent to its depth transition"
                )
            if cls._best_bid_ask_values(row) != expected:
                break
            consumed.append(row)
        if not consumed:
            return ()
        del rows[: len(consumed)]
        if not rows:
            pending_best.pop(token, None)
        return tuple(consumed)

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
        if old_tick == new_tick:
            raise ValueError("tick_size_change does not change the tick size")
        source_time = _timestamp(
            row.event.get("timestamp"), name="tick_size_change timestamp"
        )
        current = state[token]
        if old_tick != current.tick_size:
            if current.tick_size != new_tick or current.source_time_ms != source_time:
                raise ValueError(
                    "tick_size_change old value disagrees with replay state: "
                    f"token={token} event_id={row.event_id} "
                    f"reported_old={old_tick} active={current.tick_size} "
                    f"reported_new={new_tick} source_time_ms={source_time} "
                    f"state_source_time_ms={current.source_time_ms}"
                )
            current.provenance_sha256 = _canonical_sha256(
                {
                    "schema_version": "polymarket-duplicate-tick-transition-v1",
                    "previous": current.provenance_sha256,
                    "event_id": row.event_id,
                    "event_sha256": row.event_sha256,
                    "token_id": token,
                    "reported_old_tick_size": format(old_tick, "f"),
                    "active_tick_size": format(current.tick_size, "f"),
                    "source_time_ms": source_time,
                }
            )
            return
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
        if assets != list(market.token_ids):
            raise ValueError("market_resolved token/outcome mapping drifted")
        if outcomes is not None and outcomes != ["Up", "Down"]:
            raise ValueError("market_resolved token/outcome mapping drifted")
        optional_identity = {
            "id": market.market_id,
            "question": market.question,
            "slug": market.slug,
        }
        for field, expected in optional_identity.items():
            observed = row.event.get(field)
            if observed is not None and str(observed) != expected:
                raise ValueError(f"market_resolved {field} identity drifted")
        expected_outcome = (
            "Up"
            if winner == market.up_token_id
            else "Down"
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
            source="clob_websocket",
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
            segment_id=_canonical_sha256(
                {
                    "schema_version": "polymarket-continuity-segment-v1",
                    "run_id": run_id,
                    "condition_id": market.condition_id,
                    "connection_id": row.connection_id,
                }
            ),
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
        maximum_observation_delay_ms: int,
    ) -> PolymarketRecordedBook | None:
        if decision.run_id != self.run_id:
            raise ValueError("decision book belongs to a different recorder run")
        if self.book_for_event(decision.event_id, decision.token_id) != decision:
            raise ValueError("decision book does not match immutable replay evidence")
        latency = int(latency_ms)
        if latency <= 0 or latency > 60_000:
            raise ValueError("latency_ms must lie in [1, 60000]")
        maximum_delay = int(maximum_observation_delay_ms)
        if maximum_delay < 0 or maximum_delay > 60_000:
            raise ValueError("maximum_observation_delay_ms must lie in [0, 60000]")
        target = decision.received_monotonic_ns + latency * 1_000_000
        deadline = target + maximum_delay * 1_000_000
        for candidate in self._books_by_token.get(decision.token_id, ()):
            if candidate.received_monotonic_ns < target:
                continue
            if candidate.received_monotonic_ns > deadline:
                return None
            if candidate.segment_id != decision.segment_id:
                return None
            if candidate.received_wall_ms >= decision.market.end_ms:
                return None
            return candidate
        return None


__all__ = [
    "POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION",
    "PolymarketEvidenceReplay",
    "PolymarketMarketExecutionEvidence",
    "PolymarketRecordedBook",
    "PolymarketReplayDiagnostics",
    "PolymarketResolutionEvidence",
]
