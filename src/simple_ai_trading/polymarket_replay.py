"""Strict reconstruction of prospective Polymarket CLOB evidence."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
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
_RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS = 250
_RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS = 250_000_000
_DUPLICATE_TICK_TRANSITION_MAX_SOURCE_SKEW_MS = 1
_CAUSAL_REORDER_MAX_SOURCE_SKEW_MS = 1_000
_CAUSAL_REORDER_MAX_ARRIVAL_NS = 2_000_000_000
_REPLAY_FETCH_SIZE = 4_096
_REPLAY_EVENT_SCOPE_SCHEMA_VERSION = "polymarket-replay-event-scope-v1"
_REPLAY_MATERIALIZATION_POLICY_SCHEMA_VERSION = (
    "polymarket-replay-materialization-policy-v1"
)
_CAUSALLY_ORDERED_EVENTS = _BOOK_DEPTH_EVENTS | frozenset(
    {"best_bid_ask", "tick_size_change", "market_resolved"}
)
POLYMARKET_REPLAY_DIAGNOSTICS_SCHEMA_VERSION = "polymarket-replay-diagnostics-v3"
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
    event_scope_mode: str = "complete"
    event_scope_sha256: str = ""
    excluded_after_event_scope_count: int = 0
    materialized_minimum_depth_levels: int = 0
    materialized_depth_quantity_cap: bool = False
    materialization_policy_sha256: str = ""

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
    last_full_book_row: _EventRow | None = None
    last_tick_size_change_row: _EventRow | None = None


@dataclass(frozen=True, slots=True)
class _AppliedPriceChangeBatch:
    source_time_ms: int
    changes: tuple[tuple[_EventRow, Mapping[str, object]], ...]
    book_hash: str
    checksum: tuple[Decimal, Decimal]


@dataclass
class _PendingBookBatch:
    condition_id: str
    source_time_ms: int
    rows: list[_EventRow]
    correction_evidence_by_token: dict[str, tuple[_EventRow, ...]]


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
    following_receive_group_rows: tuple[_EventRow, ...] = ()
    following_bounded_receive_group_rows: tuple[_EventRow, ...] = ()
    following_scope_boundary_rows: tuple[_EventRow, ...] = ()
    preceding_receive_group_best_rows: tuple[_EventRow, ...] = ()


@dataclass(slots=True)
class _CausalReplayMetrics:
    total_event_count: int = 0
    causally_ordered_event_count: int = 0
    late_event_count: int = 0
    maximum_source_regression_ms: int = 0
    maximum_late_arrival_delay_ns: int = 0
    deferred_event_count: int = 0
    maximum_availability_delay_ns: int = 0
    excluded_after_event_scope_count: int = 0


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

    def with_book_sample_interval(
        self,
        book_sample_interval_ms: int,
    ) -> "PolymarketEvidenceReplay":
        """Derive the exact sampled view from one full-resolution reconstruction."""

        if isinstance(book_sample_interval_ms, bool):
            raise ValueError("book_sample_interval_ms must lie in [0, 5000]")
        interval_ms = int(book_sample_interval_ms)
        if interval_ms < 0 or interval_ms > 5_000:
            raise ValueError("book_sample_interval_ms must lie in [0, 5000]")
        current_interval_ms = int(self.diagnostics.book_sample_interval_ms)
        if interval_ms == current_interval_ms:
            return self
        if current_interval_ms != 0:
            raise ValueError(
                "Polymarket replay can only derive a sampled view from full resolution"
            )
        if self.diagnostics.materialized_book_count != len(self.books):
            raise ValueError("Polymarket full-resolution replay diagnostics disagree")

        selected: list[PolymarketRecordedBook] = []
        last_selected_ns: dict[str, int] = {}
        minimum_delta_ns = interval_ms * 1_000_000
        for book in self.books:
            previous = last_selected_ns.get(book.token_id)
            if (
                previous is not None
                and book.received_monotonic_ns - previous < minimum_delta_ns
            ):
                continue
            selected.append(book)
            last_selected_ns[book.token_id] = book.received_monotonic_ns
        diagnostics = replace(
            self.diagnostics,
            book_sample_interval_ms=interval_ms,
            materialized_book_count=len(selected),
            suppressed_book_count=(
                self.diagnostics.book_state_transition_count - len(selected)
            ),
        )
        return PolymarketEvidenceReplay(
            run_id=self.run_id,
            markets=self.markets,
            books=tuple(selected),
            resolutions=self.resolutions,
            diagnostics=diagnostics,
            market_execution_evidence=self.market_execution_evidence,
        )

    @classmethod
    def load(
        cls,
        store: PolymarketEvidenceStore,
        *,
        run_id: str | None = None,
        allow_segmented_gaps: bool = False,
        book_sample_interval_ms: int = 0,
        condition_ids: Sequence[str] | None = None,
        continuity_report_sha256: str = "",
        maximum_received_wall_ms_by_condition: Mapping[str, int] | None = None,
        materialized_minimum_depth_levels: int = 0,
        cap_materialized_depth_to_minimum_order_size: bool = False,
    ) -> "PolymarketEvidenceReplay":
        sample_interval_ms = int(book_sample_interval_ms)
        if sample_interval_ms < 0 or sample_interval_ms > 5_000:
            raise ValueError("book_sample_interval_ms must lie in [0, 5000]")
        if isinstance(materialized_minimum_depth_levels, bool):
            raise ValueError("materialized_minimum_depth_levels must lie in [0, 100]")
        minimum_depth_levels = int(materialized_minimum_depth_levels)
        if minimum_depth_levels < 0 or minimum_depth_levels > 100:
            raise ValueError("materialized_minimum_depth_levels must lie in [0, 100]")
        if not isinstance(cap_materialized_depth_to_minimum_order_size, bool):
            raise ValueError("materialized depth quantity cap must be boolean")
        materialization_policy_payload = {
            "schema_version": _REPLAY_MATERIALIZATION_POLICY_SCHEMA_VERSION,
            "minimum_depth_levels": minimum_depth_levels,
            "quantity_cap": (
                "market_minimum_order_size"
                if cap_materialized_depth_to_minimum_order_size
                else "complete_depth"
            ),
        }
        materialization_policy_sha256 = _canonical_sha256(
            materialization_policy_payload
        )
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
        integrity = store.resume_integrity_errors(selected)
        if integrity:
            raise ValueError(
                "Polymarket replay evidence failed integrity: " + "; ".join(integrity)
            )
        continuity_sha256 = str(continuity_report_sha256 or "").strip().lower()
        if continuity_sha256:
            if (
                not allow_segmented_gaps
                or selected_conditions is None
                or len(continuity_sha256) != 64
                or any(value not in "0123456789abcdef" for value in continuity_sha256)
            ):
                raise ValueError("Polymarket replay continuity proof is invalid")
            from .polymarket_continuity import (
                evaluate_polymarket_continuity_eligibility,
            )

            continuity = evaluate_polymarket_continuity_eligibility(
                store,
                run_id=selected,
            )
            if continuity.report_sha256 != continuity_sha256 or not set(
                selected_conditions
            ).issubset(continuity.eligible_condition_ids):
                raise ValueError("Polymarket replay continuity proof differs")
            gap_count = int(
                connection.execute(
                    "SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?",
                    [selected],
                ).fetchone()[0]
            )
        else:
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
        event_scope: dict[str, int] | None = None
        if maximum_received_wall_ms_by_condition is not None:
            event_scope = {}
            for (
                raw_condition,
                raw_cutoff,
            ) in maximum_received_wall_ms_by_condition.items():
                condition = str(raw_condition or "").strip().lower()
                if (
                    not condition
                    or condition in event_scope
                    or isinstance(raw_cutoff, bool)
                ):
                    raise ValueError("Polymarket replay event scope is invalid")
                try:
                    cutoff = int(raw_cutoff)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "Polymarket replay event scope is invalid"
                    ) from exc
                if cutoff < 0:
                    raise ValueError("Polymarket replay event scope is invalid")
                event_scope[condition] = cutoff
            if set(event_scope) != {market.condition_id for market in markets}:
                raise ValueError(
                    "Polymarket replay event scope must cover every selected market"
                )
        event_scope_payload: dict[str, object] = {
            "schema_version": _REPLAY_EVENT_SCOPE_SCHEMA_VERSION,
            "mode": (
                "complete"
                if event_scope is None
                else "condition_received_wall_upper_bound"
            ),
            "condition_maximum_received_wall_ms": (
                {} if event_scope is None else dict(sorted(event_scope.items()))
            ),
        }
        event_scope_sha256 = _canonical_sha256(event_scope_payload)
        market_execution_evidence = cls._load_market_execution_evidence(
            store,
            selected,
            condition_ids=selected_conditions,
        )
        if {item.condition_id for item in market_execution_evidence} != {
            market.condition_id for market in markets
        }:
            raise ValueError(
                "Polymarket execution evidence does not cover every market"
            )
        continuity_mode = "segmented" if gap_count else "strict"
        causal_metrics = _CausalReplayMetrics()
        events = cls._iter_causal_events(
            store,
            selected,
            markets,
            causal_metrics,
            maximum_received_wall_ms_by_condition=event_scope,
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
            materialized_minimum_depth_levels=minimum_depth_levels,
            cap_materialized_depth_to_minimum_order_size=(
                cap_materialized_depth_to_minimum_order_size
            ),
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
            event_scope_mode=str(event_scope_payload["mode"]),
            event_scope_sha256=event_scope_sha256,
            excluded_after_event_scope_count=(
                causal_metrics.excluded_after_event_scope_count
            ),
            materialized_minimum_depth_levels=minimum_depth_levels,
            materialized_depth_quantity_cap=(
                cap_materialized_depth_to_minimum_order_size
            ),
            materialization_policy_sha256=materialization_policy_sha256,
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
        lane_summaries = store.raw_message_lane_summaries(
            run_id,
            streams=tuple(sorted(segmentable_streams)),
        )
        summaries_by_connection = {
            (summary.stream, summary.connection_id): summary
            for summary in lane_summaries
        }
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
            evidence = summaries_by_connection.get(connection_key)
            evidence_count = 0 if evidence is None else evidence.message_count
            expected_last_sequence = int(last_sequence_number)
            if expected_last_sequence == 0:
                if evidence_count != 0:
                    raise ValueError(
                        "zero-sequence stream gap has matching message evidence"
                    )
                continue
            if evidence_count < 1:
                raise ValueError("stream gap has no matching connection evidence")
            assert evidence is not None
            if evidence.maximum_sequence_number != expected_last_sequence:
                raise ValueError("stream gap does not close the final sequence")
            if evidence.last_received_wall_ms > int(opened_at_ms):
                raise ValueError("stream gap precedes messages on its connection")

        named_lanes: dict[str, list[tuple[int, int, str, str]]] = {}
        for summary in lane_summaries:
            normalized_stream = summary.stream
            normalized_connection = summary.connection_id
            lane = _named_live_lane(normalized_stream, normalized_connection)
            if lane is None:
                continue
            if summary.minimum_sequence_number != 1:
                raise ValueError(
                    "named stream connection does not begin at sequence one"
                )
            named_lanes.setdefault(lane, []).append(
                (
                    summary.first_received_monotonic_ns,
                    summary.last_received_monotonic_ns,
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
            prior = recorded_by_condition.get(item.condition_id)
            if prior is not None:
                if (
                    prior.winning_asset_id != item.winning_asset_id
                    or prior.winning_outcome != item.winning_outcome
                ):
                    raise ValueError(
                        "Polymarket replay has conflicting resolution events"
                    )
                if (
                    item.received_monotonic_ns,
                    item.received_wall_ms,
                    item.event_id,
                ) >= (
                    prior.received_monotonic_ns,
                    prior.received_wall_ms,
                    prior.event_id,
                ):
                    continue
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
        *,
        maximum_received_wall_ms_by_condition: Mapping[str, int] | None = None,
    ) -> Iterator[_EventRow]:
        market_by_condition = {market.condition_id: market for market in markets}
        conditions = tuple(sorted(market_by_condition))
        source_watermarks: dict[tuple[str, str], tuple[int, int]] = {}
        availability: dict[str, tuple[int, int]] = {}
        raw_group: list[tuple[object, ...]] = []
        group_monotonic_ns: int | None = None
        released_group: tuple[_EventRow, ...] | None = None

        def release_group(
            grouped: list[tuple[object, ...]],
        ) -> tuple[_EventRow, ...]:
            prepared: list[tuple[_EventRow, str, int | None]] = []
            excluded_scope_rows: list[tuple[_EventRow, str]] = []
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
                cutoff = (
                    None
                    if maximum_received_wall_ms_by_condition is None
                    else maximum_received_wall_ms_by_condition.get(condition)
                )
                if (
                    cutoff is not None
                    and row.event_type != "market_resolved"
                    and row.received_wall_ms > cutoff
                ):
                    metrics.excluded_after_event_scope_count += 1
                    excluded_scope_rows.append((row, condition))
                    continue
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

            for index, (row, condition, _source_time) in enumerate(prepared):
                if row.event_type != "price_change":
                    continue
                row.preceding_receive_group_best_rows = tuple(
                    candidate
                    for candidate, candidate_condition, _candidate_source in prepared[
                        :index
                    ]
                    if candidate.connection_id == row.connection_id
                    and candidate_condition == condition
                    and candidate.event_type == "best_bid_ask"
                )

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
            for row in output:
                if row.event_type != "best_bid_ask":
                    continue
                condition = str(row.event.get("market") or row.condition_id).lower()
                source_time = _timestamp(
                    row.event.get("timestamp"),
                    name="scope-boundary best_bid_ask timestamp",
                )
                row.following_scope_boundary_rows = tuple(
                    sorted(
                        (
                            candidate
                            for candidate, candidate_condition in excluded_scope_rows
                            if candidate_condition == condition
                            and candidate.connection_id == row.connection_id
                            and candidate.received_monotonic_ns
                            == row.received_monotonic_ns
                            and candidate.event_type in _BOOK_DEPTH_EVENTS
                            and cls._event_arrival_key(candidate)
                            > cls._event_arrival_key(row)
                            and _timestamp(
                                candidate.event.get("timestamp"),
                                name=("scope-boundary book-depth event timestamp"),
                            )
                            == source_time
                        ),
                        key=cls._event_arrival_key,
                    )
                )
            for index, row in enumerate(output):
                if row.event_type != "price_change":
                    continue
                condition = str(row.event.get("market") or row.condition_id).lower()
                ordered_preceding = tuple(
                    candidate
                    for candidate in output[:index]
                    if candidate.connection_id == row.connection_id
                    and str(
                        candidate.event.get("market") or candidate.condition_id
                    ).lower()
                    == condition
                    and candidate.event_type == "best_bid_ask"
                )
                preceding_by_id = {
                    candidate.event_id: candidate
                    for candidate in (
                        *row.preceding_receive_group_best_rows,
                        *ordered_preceding,
                    )
                }
                row.preceding_receive_group_best_rows = tuple(
                    sorted(preceding_by_id.values(), key=cls._event_arrival_key)
                )
                following = tuple(
                    candidate
                    for candidate in output[index + 1 :]
                    if candidate.connection_id == row.connection_id
                    and str(
                        candidate.event.get("market") or candidate.condition_id
                    ).lower()
                    == condition
                    and candidate.event_type in _BOOK_DEPTH_EVENTS
                )
                if any(candidate.event_type == "book" for candidate in following):
                    row.following_receive_group_rows = following
            return tuple(output)

        def attach_bounded_following_group(
            previous: tuple[_EventRow, ...],
            following: tuple[_EventRow, ...],
        ) -> None:
            for row in previous:
                if row.event_type != "price_change":
                    continue
                condition = str(row.event.get("market") or row.condition_id).lower()
                candidates = tuple(
                    candidate
                    for candidate in following
                    if candidate.connection_id == row.connection_id
                    and str(
                        candidate.event.get("market") or candidate.condition_id
                    ).lower()
                    == condition
                    and candidate.event_type in _BOOK_DEPTH_EVENTS
                    and 0
                    < candidate.received_monotonic_ns - row.received_monotonic_ns
                    <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
                    and 0
                    <= candidate.received_wall_ms - row.received_wall_ms
                    <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
                )
                if any(candidate.event_type == "book" for candidate in candidates):
                    row.following_bounded_receive_group_rows = candidates

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
                current_group = release_group(raw_group)
                if released_group is not None:
                    attach_bounded_following_group(released_group, current_group)
                    yield from released_group
                released_group = current_group
                raw_group = []
            group_monotonic_ns = received_monotonic_ns
            raw_group.append(raw_row)
        if raw_group:
            current_group = release_group(raw_group)
            if released_group is not None:
                attach_bounded_following_group(released_group, current_group)
                yield from released_group
            released_group = current_group
        if released_group is not None:
            yield from released_group

    @classmethod
    def _reconstruct(
        cls,
        run_id: str,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        events: Iterable[_EventRow],
        *,
        book_sample_interval_ms: int,
        materialized_minimum_depth_levels: int,
        cap_materialized_depth_to_minimum_order_size: bool,
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
        delta_history: dict[str, list[_AppliedPriceChangeBatch]] = {}
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
                    delta_history,
                    last_materialized_ns,
                    book_sample_interval_ms=book_sample_interval_ms,
                    materialized_minimum_depth_levels=(
                        materialized_minimum_depth_levels
                    ),
                    cap_materialized_depth_to_minimum_order_size=(
                        cap_materialized_depth_to_minimum_order_size
                    ),
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
                        delta_history.pop(token, None)
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
                    correction_evidence: dict[str, tuple[_EventRow, ...]] | None = None
                    if batch is not None and event_type == "price_change":
                        correction_evidence = (
                            cls._cross_timestamp_idempotent_correction_evidence(
                                batch,
                                row,
                                state,
                                pending_best,
                                market_by_token,
                            )
                        )
                        if (
                            correction_evidence is None
                            and len(batch.rows) > 1
                            and not batch.correction_evidence_by_token
                            and all(
                                item.event_type in {"price_change", "best_bid_ask"}
                                for item in batch.rows
                            )
                        ):
                            matches: list[
                                tuple[
                                    int,
                                    _PendingBookBatch,
                                    dict[str, tuple[_EventRow, ...]],
                                    list[_EventRow],
                                ]
                            ] = []
                            for index, candidate in enumerate(batch.rows):
                                candidate_batch = _PendingBookBatch(
                                    condition_id=batch.condition_id,
                                    source_time_ms=batch.source_time_ms,
                                    rows=[candidate],
                                    correction_evidence_by_token={},
                                )
                                candidate_evidence = (
                                    cls._cross_timestamp_idempotent_correction_evidence(
                                        candidate_batch,
                                        row,
                                        state,
                                        pending_best,
                                        market_by_token,
                                        consume_stale_best=False,
                                    )
                                )
                                if candidate_evidence is not None:
                                    evidence_ids = {
                                        evidence_row.event_id
                                        for evidence_rows in candidate_evidence.values()
                                        for evidence_row in evidence_rows
                                    }
                                    residual_candidates = [
                                        item
                                        for residual_index, item in enumerate(
                                            batch.rows
                                        )
                                        if residual_index != index
                                        and item.event_id not in evidence_ids
                                    ]
                                    if all(
                                        item.event_type == "price_change"
                                        and item.connection_id
                                        == candidate.connection_id
                                        and 0
                                        <= candidate.received_monotonic_ns
                                        - item.received_monotonic_ns
                                        <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
                                        and 0
                                        <= candidate.received_wall_ms
                                        - item.received_wall_ms
                                        <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
                                        and cls._event_arrival_key(item)
                                        < cls._event_arrival_key(candidate)
                                        for item in residual_candidates
                                    ):
                                        matches.append(
                                            (
                                                index,
                                                candidate_batch,
                                                candidate_evidence,
                                                residual_candidates,
                                            )
                                        )
                            if len(matches) > 1:
                                raise ValueError(
                                    "cross-timestamp correction matches multiple "
                                    "pending price-change rows"
                                )
                            if matches:
                                (
                                    _candidate_index,
                                    candidate_batch,
                                    provisional,
                                    residual_rows,
                                ) = matches[0]
                                if residual_rows:
                                    pending[condition] = _PendingBookBatch(
                                        condition_id=batch.condition_id,
                                        source_time_ms=batch.source_time_ms,
                                        rows=residual_rows,
                                        correction_evidence_by_token={},
                                    )
                                    flush(condition)
                                else:
                                    pending.pop(condition, None)
                                correction_evidence = (
                                    cls._cross_timestamp_idempotent_correction_evidence(
                                        candidate_batch,
                                        row,
                                        state,
                                        pending_best,
                                        market_by_token,
                                    )
                                )
                                if correction_evidence is None:
                                    raise ValueError(
                                        "cross-timestamp correction proof changed "
                                        "after replaying its independent batch prefix"
                                    )
                                provisional_ids = {
                                    token: tuple(
                                        item.event_id for item in evidence_rows
                                    )
                                    for token, evidence_rows in provisional.items()
                                }
                                corrected_ids = {
                                    token: tuple(
                                        item.event_id for item in evidence_rows
                                    )
                                    for token, evidence_rows in correction_evidence.items()
                                }
                                if corrected_ids != provisional_ids:
                                    raise ValueError(
                                        "cross-timestamp correction evidence changed "
                                        "after replaying its independent batch prefix"
                                    )
                    if correction_evidence is None:
                        flush(condition)
                    else:
                        pending.pop(condition, None)
                    pending[condition] = _PendingBookBatch(
                        condition_id=condition,
                        source_time_ms=source_time,
                        rows=[row],
                        correction_evidence_by_token=(correction_evidence or {}),
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
                delta_history.pop(str(row.event.get("asset_id") or row.asset_id), None)
            elif event_type == "market_resolved":
                flush(condition)
                resolutions.append(cls._resolution(run_id, row, market_by_condition))
            elif event_type not in _KNOWN_NO_BOOK_CHANGE_EVENTS:
                raise ValueError(f"unsupported CLOB replay event type: {event_type}")
        for condition in tuple(pending):
            flush(condition)
        discarded_best_count += cls._discard_scope_boundary_pending_best(
            pending_best,
            state,
        )
        unresolved_best = sum(len(rows) for rows in pending_best.values())
        if unresolved_best:
            unresolved_details = []
            for token in sorted(pending_best):
                rows = pending_best[token]
                if not rows:
                    continue
                market = market_by_token.get(token)
                unresolved_details.append(
                    {
                        "condition": market.condition_id if market is not None else "",
                        "token": token,
                        "events": [
                            {
                                "event_id": row.event_id,
                                "source_time_ms": _timestamp(
                                    row.event.get("timestamp"),
                                    name="unresolved best_bid_ask timestamp",
                                ),
                                "best": cls._best_bid_ask_values(row),
                            }
                            for row in rows[:3]
                        ],
                        "additional_event_count": max(0, len(rows) - 3),
                    }
                )
                if len(unresolved_details) == 5:
                    break
            raise ValueError(
                "best_bid_ask evidence was not corroborated by a subsequent "
                f"depth transition: total={unresolved_best} "
                f"details={unresolved_details}"
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
    def _discard_scope_boundary_pending_best(
        cls,
        pending_best: dict[str, list[_EventRow]],
        state: Mapping[str, _BookState],
    ) -> int:
        """Retire BBO metadata proven only by an excluded same-group delta."""

        grouped: dict[
            tuple[str, str, int, int, tuple[str, ...]],
            list[tuple[str, _EventRow]],
        ] = {}
        for token, rows in pending_best.items():
            if len(rows) != 1:
                continue
            row = rows[0]
            evidence = row.following_scope_boundary_rows
            if not evidence or any(
                item.event_type != "price_change" for item in evidence
            ):
                continue
            condition = str(row.event.get("market") or row.condition_id).lower()
            source_time = _timestamp(
                row.event.get("timestamp"),
                name="scope-boundary best_bid_ask timestamp",
            )
            key = (
                condition,
                row.connection_id,
                row.received_monotonic_ns,
                source_time,
                tuple(item.event_id for item in evidence),
            )
            grouped.setdefault(key, []).append((token, row))

        discarded = 0
        for key, observations in grouped.items():
            condition, _connection, _receive_group, source_time, _event_ids = key
            evidence = observations[0][1].following_scope_boundary_rows
            batch = _PendingBookBatch(
                condition_id=condition,
                source_time_ms=source_time,
                rows=list(evidence),
                correction_evidence_by_token={},
            )
            if not cls._delta_batch_has_complete_top(batch, state):
                continue
            reported: dict[str, tuple[Decimal, Decimal]] = {}
            for evidence_row in evidence:
                changes = evidence_row.event.get("price_changes")
                if not isinstance(changes, list):
                    reported.clear()
                    break
                for change in changes:
                    if not isinstance(change, Mapping):
                        reported.clear()
                        break
                    token = str(change.get("asset_id") or "")
                    reported[token] = (
                        _decimal(
                            change.get("best_bid"),
                            name="scope-boundary price_change best_bid",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                        _decimal(
                            change.get("best_ask"),
                            name="scope-boundary price_change best_ask",
                            minimum=Decimal("0"),
                            maximum=Decimal("1"),
                        ),
                    )
                if not reported:
                    break
            if not reported or any(
                cls._best_bid_ask_values(row) != reported.get(token)
                for token, row in observations
            ):
                continue
            for token, row in observations:
                if pending_best.get(token) != [row]:
                    raise TypeError("scope-boundary BBO evidence ordering drifted")
                pending_best.pop(token)
                discarded += 1
        return discarded

    @classmethod
    def _corrected_idempotent_duplicate_evidence(
        cls,
        token: str,
        batch: _PendingBookBatch,
        current: _BookState | None,
        pending_row: _EventRow,
        pending_change: Mapping[str, object],
        pending_checksum: tuple[Decimal, Decimal],
        replacement_row: _EventRow,
        replacement_change: Mapping[str, object],
        replacement_checksum: tuple[Decimal, Decimal],
        pending_best: Mapping[str, list[_EventRow]],
    ) -> tuple[_EventRow, ...]:
        """Recognize one narrowly bounded exchange-side checksum correction."""

        if (
            current is None
            or pending_row.connection_id != replacement_row.connection_id
            or pending_row.received_monotonic_ns
            != replacement_row.received_monotonic_ns
            or cls._event_arrival_key(pending_row)
            >= cls._event_arrival_key(replacement_row)
            or pending_checksum == replacement_checksum
            or pending_checksum[0] > pending_checksum[1]
            or replacement_checksum[0] > replacement_checksum[1]
        ):
            return ()

        def mutation(
            change: Mapping[str, object],
        ) -> tuple[str, str, Decimal, Decimal, str]:
            return (
                str(change.get("asset_id") or ""),
                str(change.get("side") or "").upper(),
                _decimal(
                    change.get("price"),
                    name="corrected price_change price",
                    minimum=Decimal("0.0001"),
                    maximum=Decimal("0.9999"),
                ),
                _decimal(
                    change.get("size"),
                    name="corrected price_change size",
                    minimum=Decimal("0"),
                ),
                str(change.get("hash") or "").strip(),
            )

        pending_mutation = mutation(pending_change)
        replacement_mutation = mutation(replacement_change)
        if (
            pending_mutation != replacement_mutation
            or pending_mutation[0] != token
            or pending_mutation[1] not in {"BUY", "SELL"}
            or not pending_mutation[4]
        ):
            return ()
        _asset_id, side, price, size, _book_hash = pending_mutation
        levels = current.bids if side == "BUY" else current.asks
        if (size == 0 and price in levels) or (size != 0 and levels.get(price) != size):
            return ()
        expected = (
            max(current.bids, default=Decimal("0")),
            min(current.asks, default=Decimal("1")),
        )
        if pending_checksum == expected or replacement_checksum != expected:
            return ()

        best_rows = pending_best.get(token, [])
        if not best_rows:
            return ()
        stale_best = best_rows[0]
        receive_group_delta_ns = (
            pending_row.received_monotonic_ns - stale_best.received_monotonic_ns
        )
        same_receive_group = receive_group_delta_ns == 0
        bounded_later_receive_group = (
            0 < receive_group_delta_ns <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
            and 0
            <= pending_row.received_wall_ms - stale_best.received_wall_ms
            <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
        )
        if (
            _timestamp(
                stale_best.event.get("timestamp"),
                name="corrected best_bid_ask timestamp",
            )
            != batch.source_time_ms
            or stale_best.connection_id != pending_row.connection_id
            or not (same_receive_group or bounded_later_receive_group)
            or cls._event_arrival_key(stale_best) > cls._event_arrival_key(pending_row)
            or cls._best_bid_ask_values(stale_best) != pending_checksum
        ):
            return ()
        corrected_best = next(
            (
                row
                for row in best_rows[1:]
                if _timestamp(
                    row.event.get("timestamp"),
                    name="corrected best_bid_ask timestamp",
                )
                == batch.source_time_ms
                and row.connection_id == replacement_row.connection_id
                and row.received_monotonic_ns == replacement_row.received_monotonic_ns
                and cls._event_arrival_key(pending_row)
                < cls._event_arrival_key(row)
                < cls._event_arrival_key(replacement_row)
                and cls._best_bid_ask_values(row) == replacement_checksum
            ),
            None,
        )
        if corrected_best is None:
            removed_ephemeral_top = size == 0 and (
                (
                    side == "BUY"
                    and price == pending_checksum[0]
                    and pending_checksum[0] > expected[0]
                    and pending_checksum[1] == expected[1]
                )
                or (
                    side == "SELL"
                    and price == pending_checksum[1]
                    and pending_checksum[1] < expected[1]
                    and pending_checksum[0] == expected[0]
                )
            )
            if not removed_ephemeral_top:
                return ()
            return pending_row, stale_best
        return pending_row, stale_best, corrected_best

    @classmethod
    def _cross_timestamp_idempotent_correction_evidence(
        cls,
        batch: _PendingBookBatch,
        replacement_row: _EventRow,
        state: Mapping[str, _BookState],
        pending_best: dict[str, list[_EventRow]],
        market_by_token: Mapping[str, PolymarketFiveMinuteMarket],
        *,
        consume_stale_best: bool = True,
    ) -> dict[str, tuple[_EventRow, ...]] | None:
        """Bind one receive-group correction without granting stale depth authority."""

        if (
            len(batch.rows) != 1
            or batch.rows[0].event_type != "price_change"
            or replacement_row.event_type != "price_change"
        ):
            return None
        pending_row = batch.rows[0]
        replacement_source_time_ms = _timestamp(
            replacement_row.event.get("timestamp"),
            name="corrected price_change timestamp",
        )
        receive_group_delta_ns = (
            replacement_row.received_monotonic_ns - pending_row.received_monotonic_ns
        )
        same_receive_group = receive_group_delta_ns == 0
        bounded_later_receive_group = (
            0 < receive_group_delta_ns <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
            and 0
            <= replacement_row.received_wall_ms - pending_row.received_wall_ms
            <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
        )
        if (
            replacement_source_time_ms <= batch.source_time_ms
            or replacement_source_time_ms - batch.source_time_ms
            > _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
            or pending_row.connection_id != replacement_row.connection_id
            or not (same_receive_group or bounded_later_receive_group)
            or cls._event_arrival_key(pending_row)
            >= cls._event_arrival_key(replacement_row)
        ):
            return None
        pending_changes = pending_row.event.get("price_changes")
        replacement_changes = replacement_row.event.get("price_changes")
        if (
            not isinstance(pending_changes, list)
            or not pending_changes
            or not isinstance(replacement_changes, list)
            or not replacement_changes
        ):
            return None

        def changes_by_token(
            values: list[object],
        ) -> dict[str, Mapping[str, object]] | None:
            output: dict[str, Mapping[str, object]] = {}
            for value in values:
                if not isinstance(value, Mapping):
                    return None
                token = str(value.get("asset_id") or "")
                if not token or token in output or token not in market_by_token:
                    return None
                output[token] = value
            return output

        pending_by_token = changes_by_token(pending_changes)
        replacement_by_token = changes_by_token(replacement_changes)
        if (
            pending_by_token is None
            or replacement_by_token is None
            or set(pending_by_token) != set(replacement_by_token)
        ):
            return None

        evidence: dict[str, tuple[_EventRow, ...]] = {}
        stale_best_rows: dict[str, _EventRow] = {}
        pending_stale_best_tokens: set[str] = set()
        for token in sorted(pending_by_token):
            current = state.get(token)
            pending_change = pending_by_token[token]
            replacement_change = replacement_by_token[token]
            if current is None:
                return None

            def mutation(
                change: Mapping[str, object],
            ) -> tuple[str, str, Decimal, Decimal]:
                return (
                    str(change.get("asset_id") or ""),
                    str(change.get("side") or "").upper(),
                    _decimal(
                        change.get("price"),
                        name="corrected price_change price",
                        minimum=Decimal("0.0001"),
                        maximum=Decimal("0.9999"),
                    ),
                    _decimal(
                        change.get("size"),
                        name="corrected price_change size",
                        minimum=Decimal("0"),
                    ),
                )

            pending_mutation = mutation(pending_change)
            replacement_mutation = mutation(replacement_change)
            if (
                pending_mutation != replacement_mutation
                or pending_mutation[0] != token
                or pending_mutation[1] not in {"BUY", "SELL"}
                or not str(pending_change.get("hash") or "").strip()
                or not str(replacement_change.get("hash") or "").strip()
            ):
                return None
            _asset_id, side, price, size = pending_mutation
            if price % current.tick_size != 0:
                return None
            simulated_bids = dict(current.bids)
            simulated_asks = dict(current.asks)
            levels = simulated_bids if side == "BUY" else simulated_asks
            if size == 0:
                levels.pop(price, None)
            else:
                levels[price] = size
            expected = (
                max(simulated_bids, default=Decimal("0")),
                min(simulated_asks, default=Decimal("1")),
            )

            def checksum(
                change: Mapping[str, object],
            ) -> tuple[Decimal, Decimal]:
                return (
                    _decimal(
                        change.get("best_bid"),
                        name="corrected price_change best_bid",
                        minimum=Decimal("0"),
                        maximum=Decimal("1"),
                    ),
                    _decimal(
                        change.get("best_ask"),
                        name="corrected price_change best_ask",
                        minimum=Decimal("0"),
                        maximum=Decimal("1"),
                    ),
                )

            pending_checksum = checksum(pending_change)
            replacement_checksum = checksum(replacement_change)
            if (
                pending_checksum == expected
                or replacement_checksum != expected
                or pending_checksum[0] > pending_checksum[1]
                or replacement_checksum[0] > replacement_checksum[1]
            ):
                return None

            best_rows = pending_best.get(token, [])
            stale_candidates_by_id = {
                row.event_id: row
                for row in (
                    *best_rows,
                    *pending_row.preceding_receive_group_best_rows,
                )
                if str(row.event.get("asset_id") or row.asset_id) == token
            }
            stale_best = next(
                (
                    row
                    for row in sorted(
                        stale_candidates_by_id.values(),
                        key=cls._event_arrival_key,
                        reverse=True,
                    )
                    if batch.source_time_ms
                    <= _timestamp(
                        row.event.get("timestamp"),
                        name="stale corrected best_bid_ask timestamp",
                    )
                    <= replacement_source_time_ms
                    and row.connection_id == pending_row.connection_id
                    and row.received_monotonic_ns == pending_row.received_monotonic_ns
                    and cls._event_arrival_key(row)
                    < cls._event_arrival_key(pending_row)
                    and cls._best_bid_ask_values(row) == pending_checksum
                ),
                None,
            )
            if stale_best is None:
                return None
            stale_is_pending = bool(
                best_rows and best_rows[0].event_id == stale_best.event_id
            )
            if best_rows and not stale_is_pending:
                return None
            corrected_candidates_by_id = {
                row.event_id: row
                for row in (
                    *(best_rows[1:] if stale_is_pending else best_rows),
                    *replacement_row.preceding_receive_group_best_rows,
                )
                if str(row.event.get("asset_id") or row.asset_id) == token
            }
            corrected_best = next(
                (
                    row
                    for row in sorted(
                        corrected_candidates_by_id.values(),
                        key=cls._event_arrival_key,
                    )
                    if replacement_source_time_ms
                    <= _timestamp(
                        row.event.get("timestamp"),
                        name="replacement corrected best_bid_ask timestamp",
                    )
                    <= replacement_source_time_ms
                    + _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
                    and row.connection_id == replacement_row.connection_id
                    and row.received_monotonic_ns
                    == replacement_row.received_monotonic_ns
                    and cls._event_arrival_key(pending_row)
                    < cls._event_arrival_key(row)
                    < cls._event_arrival_key(replacement_row)
                    and cls._best_bid_ask_values(row) == replacement_checksum
                ),
                None,
            )
            if corrected_best is None:
                return None
            if not (
                cls._event_arrival_key(stale_best)
                < cls._event_arrival_key(pending_row)
                < cls._event_arrival_key(corrected_best)
                < cls._event_arrival_key(replacement_row)
            ):
                return None
            stale_best_rows[token] = stale_best
            if stale_is_pending:
                pending_stale_best_tokens.add(token)
            evidence[token] = (stale_best, pending_row, corrected_best)

        if consume_stale_best:
            for token in pending_stale_best_tokens:
                stale_best = stale_best_rows[token]
                best_rows = pending_best[token]
                if best_rows[0].event_id != stale_best.event_id:
                    raise TypeError("corrected best_bid_ask evidence ordering drifted")
                del best_rows[0]
                if not best_rows:
                    pending_best.pop(token)
        return evidence

    @classmethod
    def _next_group_full_book_crossing_evidence(
        cls,
        token: str,
        batch: _PendingBookBatch,
        market: PolymarketFiveMinuteMarket,
        current: _BookState,
        pending_changes: list[tuple[_EventRow, Mapping[str, object]]],
        reported_checksum: tuple[Decimal, Decimal],
        expected_checksum: tuple[Decimal, Decimal],
    ) -> tuple[_EventRow, ...]:
        """Repair one stale opposite top proven by BBO and the next full book."""

        if (
            not pending_changes
            or expected_checksum[0] < expected_checksum[1]
            or reported_checksum[0] >= reported_checksum[1]
        ):
            return ()
        initiating: list[
            tuple[_EventRow, Mapping[str, object], str, Decimal, Decimal]
        ] = []
        parsed_changes: list[tuple[_EventRow, str, Decimal, Decimal]] = []
        for row, change in pending_changes:
            candidate_side = str(change.get("side") or "").upper()
            candidate_price = _decimal(
                change.get("price"),
                name="crossing correction price",
                minimum=Decimal("0.0001"),
                maximum=Decimal("0.9999"),
            )
            candidate_size = _decimal(
                change.get("size"),
                name="crossing correction size",
                minimum=Decimal("0"),
            )
            if candidate_side not in {"BUY", "SELL"}:
                return ()
            parsed_changes.append(
                (row, candidate_side, candidate_price, candidate_size)
            )
            if candidate_size > 0 and (
                (
                    candidate_side == "BUY"
                    and candidate_price == expected_checksum[0]
                    and candidate_price == reported_checksum[0]
                )
                or (
                    candidate_side == "SELL"
                    and candidate_price == expected_checksum[1]
                    and candidate_price == reported_checksum[1]
                )
            ):
                initiating.append(
                    (
                        row,
                        change,
                        candidate_side,
                        candidate_price,
                        candidate_size,
                    )
                )
        if len(initiating) != 1:
            return ()
        change_row, _change, side, price, size = initiating[0]
        if side == "BUY":
            stale_prices = tuple(
                level for level in current.asks if level < reported_checksum[1]
            )
            opposite_side = "SELL"
        else:
            stale_prices = tuple(
                level for level in current.bids if level > reported_checksum[0]
            )
            opposite_side = "BUY"
        if not stale_prices:
            return ()
        for row, candidate_side, candidate_price, candidate_size in parsed_changes:
            if (
                row == change_row
                and candidate_side == side
                and candidate_price == price
            ):
                continue
            bounded_opposite_removal = (
                candidate_side == opposite_side
                and candidate_size == 0
                and (
                    (side == "BUY" and candidate_price < reported_checksum[1])
                    or (side == "SELL" and candidate_price > reported_checksum[0])
                )
            )
            if not bounded_opposite_removal:
                return ()

        preceding_best = next(
            (
                row
                for row in reversed(change_row.preceding_receive_group_best_rows)
                if str(row.event.get("asset_id") or row.asset_id) == token
                and row.connection_id == change_row.connection_id
                and row.received_monotonic_ns == change_row.received_monotonic_ns
                and cls._event_arrival_key(row) < cls._event_arrival_key(change_row)
                and batch.source_time_ms
                <= _timestamp(
                    row.event.get("timestamp"),
                    name="crossing best_bid_ask timestamp",
                )
                <= batch.source_time_ms + _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
                and cls._best_bid_ask_values(row) == reported_checksum
            ),
            None,
        )
        if preceding_best is None:
            return ()

        following_rows = sorted(
            change_row.following_bounded_receive_group_rows,
            key=cls._event_arrival_key,
        )
        full_book_row = next(
            (
                row
                for row in following_rows
                if row.event_type == "book"
                and str(row.event.get("asset_id") or row.asset_id) == token
            ),
            None,
        )
        if full_book_row is None:
            return ()
        full_book = validate_clob_order_book(
            market,
            token,
            full_book_row.event,
            received_wall_ms=full_book_row.received_wall_ms,
            received_monotonic_ns=full_book_row.received_monotonic_ns,
        )
        if (
            full_book.source_time_ms < batch.source_time_ms
            or full_book.source_time_ms - batch.source_time_ms
            > _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
        ):
            return ()
        full_bids = {level.price: level.quantity for level in full_book.bids}
        full_asks = {level.price: level.quantity for level in full_book.asks}
        full_checksum = (
            max(full_bids, default=Decimal("0")),
            min(full_asks, default=Decimal("1")),
        )
        if full_checksum != reported_checksum:
            return ()
        full_opposite = full_asks if side == "BUY" else full_bids
        if any(level in full_opposite for level in stale_prices):
            return ()

        corroborating_changes: list[
            tuple[_EventRow, Mapping[str, object], str, Decimal, Decimal]
        ] = []
        for row in following_rows:
            if cls._event_arrival_key(row) >= cls._event_arrival_key(full_book_row):
                break
            if row.event_type != "price_change":
                continue
            changes = row.event.get("price_changes")
            if not isinstance(changes, list):
                return ()
            for candidate in changes:
                if (
                    not isinstance(candidate, Mapping)
                    or str(candidate.get("asset_id") or "") != token
                ):
                    continue
                candidate_side = str(candidate.get("side") or "").upper()
                if candidate_side not in {"BUY", "SELL"}:
                    return ()
                candidate_price = _decimal(
                    candidate.get("price"),
                    name="crossing continuation price",
                    minimum=Decimal("0.0001"),
                    maximum=Decimal("0.9999"),
                )
                if candidate_side == opposite_side and candidate_price in stale_prices:
                    return ()
                candidate_size = _decimal(
                    candidate.get("size"),
                    name="crossing corroboration size",
                    minimum=Decimal("0"),
                )
                corroborating_changes.append(
                    (
                        row,
                        candidate,
                        candidate_side,
                        candidate_price,
                        candidate_size,
                    )
                )
        full_book_hash = str(full_book_row.event.get("hash") or "").strip()
        if not full_book_hash:
            return ()
        latest_by_level: dict[
            tuple[str, Decimal],
            tuple[_EventRow, Mapping[str, object], Decimal],
        ] = {}
        for (
            row,
            candidate,
            candidate_side,
            candidate_price,
            candidate_size,
        ) in corroborating_changes:
            candidate_checksum = (
                _decimal(
                    candidate.get("best_bid"),
                    name="crossing corroboration best_bid",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                ),
                _decimal(
                    candidate.get("best_ask"),
                    name="crossing corroboration best_ask",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                ),
            )
            if (
                candidate_checksum != reported_checksum
                or _timestamp(
                    row.event.get("timestamp"),
                    name="crossing corroboration timestamp",
                )
                != full_book.source_time_ms
                or str(candidate.get("hash") or "").strip() != full_book_hash
            ):
                return ()
            latest_by_level[(candidate_side, candidate_price)] = (
                row,
                candidate,
                candidate_size,
            )
        for (candidate_side, candidate_price), (
            _row,
            _candidate,
            candidate_size,
        ) in latest_by_level.items():
            levels = full_bids if candidate_side == "BUY" else full_asks
            if (candidate_size == 0 and candidate_price in levels) or (
                candidate_size > 0 and levels.get(candidate_price) != candidate_size
            ):
                return ()

        full_same_side = full_bids if side == "BUY" else full_asks
        continuation = latest_by_level.get((side, price))
        if continuation is None:
            if full_same_side.get(price) != size:
                return ()
        else:
            _continuation_row, _continuation_change, continuation_size = continuation
            if continuation_size == 0 or full_same_side.get(price) != continuation_size:
                return ()

        repaired_levels = current.asks if side == "BUY" else current.bids
        for level in stale_prices:
            repaired_levels.pop(level, None)
        repaired_checksum = (
            max(current.bids, default=Decimal("0")),
            min(current.asks, default=Decimal("1")),
        )
        if repaired_checksum != reported_checksum:
            raise ValueError("next-group full-book crossing repair did not converge")
        corroborating_rows = [item[0] for item in corroborating_changes]
        return tuple((preceding_best, *corroborating_rows, full_book_row))

    @classmethod
    def _prefix_stale_full_book_correction_evidence(
        cls,
        token: str,
        batch: _PendingBookBatch,
        market: PolymarketFiveMinuteMarket,
        current: _BookState,
        pending_changes: list[tuple[_EventRow, Mapping[str, object]]],
        reported_checksum: tuple[Decimal, Decimal],
        expected_checksum: tuple[Decimal, Decimal],
        *,
        checksum_matched_proper_prefix: bool,
        depth_top_unchanged: bool,
        pending_best: dict[str, list[_EventRow]],
    ) -> tuple[_EventRow, ...]:
        """Accept one stale fragment checksum only when a full book proves the chain."""

        if (
            not (checksum_matched_proper_prefix or depth_top_unchanged)
            or not pending_changes
            or reported_checksum == expected_checksum
            or reported_checksum[0] > reported_checksum[1]
            or expected_checksum[0] > expected_checksum[1]
        ):
            return ()
        first_change = pending_changes[0][0]
        last_change = pending_changes[-1][0]
        if any(
            row.connection_id != first_change.connection_id
            or row.received_monotonic_ns != first_change.received_monotonic_ns
            for row, _change in pending_changes
        ):
            return ()

        pending_best_rows = pending_best.get(token, [])
        stale_best: list[_EventRow] = []
        for row in pending_best_rows:
            source_time = _timestamp(
                row.event.get("timestamp"), name="stale best_bid_ask timestamp"
            )
            if (
                source_time != batch.source_time_ms
                or row.connection_id != first_change.connection_id
                or row.received_monotonic_ns != first_change.received_monotonic_ns
                or cls._event_arrival_key(row) > cls._event_arrival_key(first_change)
                or cls._best_bid_ask_values(row) != reported_checksum
            ):
                break
            stale_best.append(row)
        stale_best_is_pending = bool(stale_best)
        if not stale_best and pending_best_rows:
            return ()
        if not stale_best:
            for row in reversed(first_change.preceding_receive_group_best_rows):
                row_token = str(row.event.get("asset_id") or row.asset_id)
                source_time = _timestamp(
                    row.event.get("timestamp"),
                    name="preceding stale best_bid_ask timestamp",
                )
                if (
                    row_token == token
                    and source_time <= batch.source_time_ms
                    and batch.source_time_ms - source_time
                    <= _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
                    and row.connection_id == first_change.connection_id
                    and row.received_monotonic_ns == first_change.received_monotonic_ns
                    and cls._event_arrival_key(row)
                    < cls._event_arrival_key(first_change)
                    and cls._best_bid_ask_values(row) == reported_checksum
                ):
                    stale_best.append(row)
                    break
        if not stale_best:
            return ()

        simulated_bids = dict(current.bids)
        simulated_asks = dict(current.asks)
        corroborating_rows: list[_EventRow] = []
        for row in last_change.following_receive_group_rows:
            if (
                row.connection_id != last_change.connection_id
                or row.received_monotonic_ns != last_change.received_monotonic_ns
                or cls._event_arrival_key(row) <= cls._event_arrival_key(last_change)
            ):
                return ()
            source_time = _timestamp(
                row.event.get("timestamp"),
                name=f"corroborating {row.event_type} timestamp",
            )
            if (
                source_time < batch.source_time_ms
                or source_time - batch.source_time_ms
                > _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
            ):
                return ()
            if row.event_type == "price_change":
                changes = row.event.get("price_changes")
                if not isinstance(changes, list) or not changes:
                    return ()
                changed_token = False
                for change in changes:
                    if not isinstance(change, Mapping):
                        return ()
                    if str(change.get("asset_id") or "") != token:
                        continue
                    side = str(change.get("side") or "").upper()
                    if (
                        side not in {"BUY", "SELL"}
                        or not str(change.get("hash") or "").strip()
                    ):
                        return ()
                    price = _decimal(
                        change.get("price"),
                        name="corroborating price_change price",
                        minimum=Decimal("0.0001"),
                        maximum=Decimal("0.9999"),
                    )
                    size = _decimal(
                        change.get("size"),
                        name="corroborating price_change size",
                        minimum=Decimal("0"),
                    )
                    if price % current.tick_size != 0:
                        return ()
                    levels = simulated_bids if side == "BUY" else simulated_asks
                    if size == 0:
                        levels.pop(price, None)
                    else:
                        levels[price] = size
                    changed_token = True
                if changed_token:
                    corroborating_rows.append(row)
                continue

            if row.event_type != "book":
                continue
            row_token = str(row.event.get("asset_id") or row.asset_id)
            if row_token != token:
                continue
            if not str(row.event.get("hash") or "").strip():
                return ()
            snapshot = validate_clob_order_book(
                market,
                token,
                row.event,
                received_wall_ms=row.received_wall_ms,
                received_monotonic_ns=row.received_monotonic_ns,
            )
            reported_tick = row.event.get("tick_size")
            if (
                reported_tick is not None
                and _decimal(
                    reported_tick,
                    name="corroborating full-book tick size",
                    minimum=Decimal("0.0001"),
                    maximum=Decimal("0.1"),
                )
                != current.tick_size
            ):
                return ()
            snapshot_bids = {level.price: level.quantity for level in snapshot.bids}
            snapshot_asks = {level.price: level.quantity for level in snapshot.asks}
            if snapshot_bids != simulated_bids or snapshot_asks != simulated_asks:
                return ()
            if stale_best_is_pending:
                best_rows = pending_best[token]
                if best_rows[: len(stale_best)] != stale_best:
                    raise TypeError("stale best_bid_ask evidence ordering drifted")
                del best_rows[: len(stale_best)]
                if not best_rows:
                    pending_best.pop(token)
            return tuple((*stale_best, *corroborating_rows, row))
        return ()

    @classmethod
    def _recent_full_book_stale_checksum_evidence(
        cls,
        token: str,
        batch: _PendingBookBatch,
        current: _BookState,
        pending_changes: list[tuple[_EventRow, Mapping[str, object]]],
        reported_checksum: tuple[Decimal, Decimal],
        expected_checksum: tuple[Decimal, Decimal],
        *,
        depth_top_unchanged: bool,
    ) -> tuple[_EventRow, ...]:
        """Bound one-tick lagging BBO metadata to a recent authoritative book."""

        full_book = current.last_full_book_row
        if (
            not depth_top_unchanged
            or full_book is None
            or not pending_changes
            or reported_checksum == expected_checksum
            or reported_checksum[0] > reported_checksum[1]
            or expected_checksum[0] > expected_checksum[1]
        ):
            return ()
        first_change = pending_changes[0][0]
        full_book_token = str(full_book.event.get("asset_id") or full_book.asset_id)
        full_book_source_time = _timestamp(
            full_book.event.get("timestamp"), name="recent full-book timestamp"
        )
        arrival_age_ns = (
            first_change.received_monotonic_ns - full_book.received_monotonic_ns
        )
        source_age_ms = batch.source_time_ms - full_book_source_time
        if (
            full_book_token != token
            or full_book.connection_id != first_change.connection_id
            or not 0 <= source_age_ms <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
            or not 0 <= arrival_age_ns <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
        ):
            return ()

        reported_bid, reported_ask = reported_checksum
        expected_bid, expected_ask = expected_checksum
        bid_differs = reported_bid != expected_bid
        ask_differs = reported_ask != expected_ask
        if not (bid_differs ^ ask_differs):
            return ()
        if bid_differs and (
            reported_bid != expected_bid + current.tick_size
            or reported_bid in current.bids
        ):
            return ()
        if ask_differs and (
            reported_ask != expected_ask - current.tick_size
            or reported_ask in current.asks
        ):
            return ()
        return (full_book,)

    @classmethod
    def _flush_book_batch(
        cls,
        run_id: str,
        batch: _PendingBookBatch,
        market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
        market_by_token: Mapping[str, PolymarketFiveMinuteMarket],
        state: dict[str, _BookState],
        pending_best: dict[str, list[_EventRow]],
        delta_history: dict[str, list[_AppliedPriceChangeBatch]],
        last_materialized_ns: dict[str, int],
        *,
        book_sample_interval_ms: int,
        materialized_minimum_depth_levels: int,
        cap_materialized_depth_to_minimum_order_size: bool,
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

        if not set(batch.correction_evidence_by_token).issubset(operations):
            raise ValueError("cross-timestamp correction evidence lost its token")

        output: list[PolymarketRecordedBook] = []
        transition_count = 0
        for token in sorted(operations):
            current = state.get(token)
            previous_provenance = current.provenance_sha256 if current else ""
            depth_changed = False
            pending_hash = ""
            pending_checksum: tuple[Decimal, Decimal] | None = None
            pending_changes: list[tuple[_EventRow, Mapping[str, object]]] = []
            relevant_rows = list(batch.correction_evidence_by_token.get(token, ()))

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
                starting_checksum = (
                    max(current.bids, default=Decimal("0")),
                    min(current.asks, default=Decimal("1")),
                )
                checksum_matched_proper_prefix = False
                for change_index, (change_row, change) in enumerate(pending_changes):
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
                    intermediate_checksum = (
                        max(current.bids, default=Decimal("0")),
                        min(current.asks, default=Decimal("1")),
                    )
                    if (
                        change_index < len(pending_changes) - 1
                        and observed_checksum == intermediate_checksum
                    ):
                        checksum_matched_proper_prefix = True
                    relevant_rows.append(change_row)
                expected = (
                    max(current.bids, default=Decimal("0")),
                    min(current.asks, default=Decimal("1")),
                )
                if reported_checksum != expected:
                    correction_evidence = cls._next_group_full_book_crossing_evidence(
                        token,
                        batch,
                        market,
                        current,
                        pending_changes,
                        reported_checksum,
                        expected,
                    )
                    if correction_evidence:
                        expected = (
                            max(current.bids, default=Decimal("0")),
                            min(current.asks, default=Decimal("1")),
                        )
                    if not correction_evidence:
                        correction_evidence = (
                            cls._prefix_stale_full_book_correction_evidence(
                                token,
                                batch,
                                market,
                                current,
                                pending_changes,
                                reported_checksum,
                                expected,
                                checksum_matched_proper_prefix=(
                                    checksum_matched_proper_prefix
                                ),
                                depth_top_unchanged=starting_checksum == expected,
                                pending_best=pending_best,
                            )
                        )
                    if not correction_evidence:
                        correction_evidence = (
                            cls._recent_full_book_stale_checksum_evidence(
                                token,
                                batch,
                                current,
                                pending_changes,
                                reported_checksum,
                                expected,
                                depth_top_unchanged=starting_checksum == expected,
                            )
                        )
                    if not correction_evidence:
                        raise ValueError(
                            "price_change best bid/ask checksum disagrees with "
                            "atomic depth: "
                            f"condition={batch.condition_id} token={token} "
                            f"source_time_ms={batch.source_time_ms} "
                            f"reported={reported_checksum} expected={expected} "
                            "events="
                            f"{[item.event_id for item, _change in pending_changes]}"
                        )
                    relevant_rows.extend(correction_evidence)
                    reported_checksum = expected
                history = delta_history.setdefault(token, [])
                history.append(
                    _AppliedPriceChangeBatch(
                        source_time_ms=batch.source_time_ms,
                        changes=tuple(
                            (change_row, dict(change))
                            for change_row, change in pending_changes
                        ),
                        book_hash=pending_hash,
                        checksum=reported_checksum,
                    )
                )
                latest_source_time_ms = max(
                    current.source_time_ms,
                    *(item.source_time_ms for item in history),
                )
                retained_after_ms = (
                    latest_source_time_ms - _CAUSAL_REORDER_MAX_SOURCE_SKEW_MS
                )
                history[:] = [
                    item for item in history if item.source_time_ms >= retained_after_ms
                ]
                current.source_time_ms = batch.source_time_ms
                current.book_hash = pending_hash
                depth_changed = True
                relevant_rows.extend(
                    cls._consume_corrected_pending_best(
                        token,
                        batch,
                        starting_checksum,
                        current,
                        [item for item, _change in pending_changes],
                        pending_best,
                    )
                )
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
                    correction_evidence: tuple[_EventRow, _EventRow] | tuple[()] = ()
                    if (
                        len(pending_changes) == 1
                        and not same_message
                        and pending_hash == book_hash
                        and pending_checksum is not None
                    ):
                        correction_evidence = (
                            cls._corrected_idempotent_duplicate_evidence(
                                token,
                                batch,
                                current,
                                pending_changes[0][0],
                                pending_changes[0][1],
                                pending_checksum,
                                row,
                                payload,
                                change_checksum,
                                pending_best,
                            )
                        )
                    if correction_evidence:
                        relevant_rows.extend(correction_evidence)
                        stale_rows = pending_best[token]
                        if stale_rows[0] != correction_evidence[1]:
                            raise TypeError(
                                "corrected price_change evidence ordering drifted"
                            )
                        del stale_rows[0]
                        if not stale_rows:
                            pending_best.pop(token)
                        pending_changes = []
                        pending_hash = ""
                        pending_checksum = None
                    elif pending_changes and not same_message and not same_fragment:
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
                        if current is None:
                            raise ValueError(
                                "full book does not corroborate its same-hash "
                                "price_change"
                            )
                        flush_changes()
                    else:
                        relevant_rows.extend(
                            change_row for change_row, _change in pending_changes
                        )
                        pending_changes = []
                        pending_hash = ""
                        pending_checksum = None
                else:
                    flush_changes()
                reported_tick = row.event.get("tick_size")
                if reported_tick is None:
                    if current is None:
                        raise ValueError(
                            "full book does not prove its same-segment active tick: "
                            f"condition={batch.condition_id} token={token} "
                            f"source_time_ms={batch.source_time_ms} "
                            f"event_id={row.event_id}"
                        )
                    active_tick = current.tick_size
                else:
                    active_tick = _decimal(
                        reported_tick,
                        name="full book tick size",
                        minimum=Decimal("0.0001"),
                        maximum=Decimal("0.1"),
                    )
                    if Decimal("1") % active_tick != 0:
                        raise ValueError(
                            "full book tick size does not divide one exactly"
                        )
                    if current is not None and active_tick != current.tick_size:
                        raise ValueError(
                            "full book tick size disagrees with same-segment replay "
                            "state: "
                            f"condition={batch.condition_id} token={token} "
                            f"event_id={row.event_id} "
                            f"reported={format(active_tick, 'f')} "
                            f"active={format(current.tick_size, 'f')}"
                        )
                off_tick_prices = sorted(
                    {
                        level.price
                        for level in (*payload.bids, *payload.asks)
                        if level.price % active_tick != 0
                    }
                )
                if off_tick_prices:
                    preview = ",".join(
                        format(price, "f") for price in off_tick_prices[:8]
                    )
                    raise ValueError(
                        "full book contains a price off the active tick: "
                        f"condition={batch.condition_id} token={token} "
                        f"source_time_ms={batch.source_time_ms} "
                        f"event_id={row.event_id} "
                        f"active_tick={format(active_tick, 'f')} "
                        f"off_tick_count={len(off_tick_prices)} "
                        f"off_tick_prices={preview}"
                    )
                payload_bids = {level.price: level.quantity for level in payload.bids}
                payload_asks = {level.price: level.quantity for level in payload.asks}
                if (
                    current is not None
                    and payload.source_time_ms < current.source_time_ms
                ):
                    history = delta_history.get(token, [])
                    newer_batches = [
                        item
                        for item in history
                        if item.source_time_ms > payload.source_time_ms
                    ]
                    if (
                        not newer_batches
                        or max(item.source_time_ms for item in newer_batches)
                        < current.source_time_ms
                    ):
                        # This snapshot cannot safely supersede a newer state and
                        # there is not enough retained evidence to fast-forward it.
                        continue
                    newer_batches.sort(
                        key=lambda item: (
                            item.source_time_ms,
                            max(
                                cls._event_arrival_key(change_row)
                                for change_row, _change in item.changes
                            ),
                        )
                    )
                    rebased = _BookState(
                        bids=payload_bids,
                        asks=payload_asks,
                        source_time_ms=payload.source_time_ms,
                        provenance_sha256=current.provenance_sha256,
                        tick_size=active_tick,
                        book_hash=book_hash,
                        last_full_book_row=row,
                        last_tick_size_change_row=(current.last_tick_size_change_row),
                    )
                    replay_rows: list[_EventRow] = []
                    for applied in newer_batches:
                        for change_row, change in applied.changes:
                            side = str(change.get("side") or "").upper()
                            if side not in {"BUY", "SELL"}:
                                raise ValueError(
                                    "retained price_change side must be BUY or SELL"
                                )
                            price = _decimal(
                                change.get("price"),
                                name="retained price_change price",
                                minimum=Decimal("0.0001"),
                                maximum=Decimal("0.9999"),
                            )
                            size = _decimal(
                                change.get("size"),
                                name="retained price_change size",
                                minimum=Decimal("0"),
                            )
                            if price % active_tick != 0:
                                raise ValueError(
                                    "retained price_change is off the active tick"
                                )
                            levels = rebased.bids if side == "BUY" else rebased.asks
                            if size == 0:
                                levels.pop(price, None)
                            else:
                                levels[price] = size
                            replay_rows.append(change_row)
                        expected = (
                            max(rebased.bids, default=Decimal("0")),
                            min(rebased.asks, default=Decimal("1")),
                        )
                        if expected != applied.checksum:
                            raise ValueError(
                                "stale full book cannot be causally fast-forwarded: "
                                f"condition={batch.condition_id} token={token} "
                                f"snapshot_source_time_ms={payload.source_time_ms} "
                                f"delta_source_time_ms={applied.source_time_ms} "
                                f"reported={applied.checksum} expected={expected}"
                            )
                        rebased.source_time_ms = applied.source_time_ms
                        rebased.book_hash = applied.book_hash
                    if rebased.source_time_ms != current.source_time_ms:
                        raise ValueError(
                            "stale full-book fast-forward did not recover current time"
                        )
                    current = rebased
                    delta_history[token] = newer_batches
                    relevant_rows.append(row)
                    relevant_rows.extend(replay_rows)
                    depth_changed = True
                    continue
                if (
                    current is not None
                    and current.book_hash == book_hash
                    and current.bids == payload_bids
                    and current.asks == payload_asks
                ):
                    current.source_time_ms = payload.source_time_ms
                    current.last_full_book_row = row
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
                        last_full_book_row=row,
                        last_tick_size_change_row=(
                            current.last_tick_size_change_row
                            if current is not None
                            else None
                        ),
                    )
                delta_history[token] = [
                    item
                    for item in delta_history.get(token, [])
                    if item.source_time_ms > payload.source_time_ms
                ]
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
                    "source_time_ms": current.source_time_ms,
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
            batch_event_ids = {row.event_id for row in batch.rows}
            if selected_row.event_id not in batch_event_ids:
                identity_rows = [
                    row
                    for operation, row, _payload in operations[token]
                    if operation in {"book", "change"}
                ]
                if not identity_rows:
                    raise TypeError("depth transition lost its identity evidence")
                identity_row = max(identity_rows, key=cls._event_arrival_key)
                availability_row = max(
                    evidence_rows,
                    key=lambda item: (
                        item.available_monotonic_ns,
                        item.available_wall_ms,
                        cls._event_arrival_key(item),
                    ),
                )
                selected_row = replace(
                    identity_row,
                    available_wall_ms=availability_row.available_wall_ms,
                    available_monotonic_ns=availability_row.available_monotonic_ns,
                )
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
                    materialized_minimum_depth_levels=(
                        materialized_minimum_depth_levels
                    ),
                    cap_materialized_depth_to_minimum_order_size=(
                        cap_materialized_depth_to_minimum_order_size
                    ),
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
        existing = pending_best.get(token)
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
                if existing:
                    first = existing[0]
                    first_source_time = _timestamp(
                        first.event.get("timestamp"),
                        name="best_bid_ask timestamp",
                    )
                    receive_group_delta_ns = (
                        row.received_monotonic_ns - first.received_monotonic_ns
                    )
                    same_receive_group = receive_group_delta_ns == 0
                    bounded_later_receive_group = (
                        0
                        < receive_group_delta_ns
                        <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
                        and 0
                        <= row.received_wall_ms - first.received_wall_ms
                        <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
                    )
                    if (
                        first_source_time <= source_time
                        and source_time - first_source_time
                        <= _BEST_CORROBORATION_MAX_SOURCE_SKEW_MS
                        and first.connection_id == row.connection_id
                        and (same_receive_group or bounded_later_receive_group)
                        and cls._event_arrival_key(first) < cls._event_arrival_key(row)
                        and cls._best_bid_ask_values(first) != observed
                    ):
                        existing.append(row)
                return
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
                f"proven_values={expected if current is not None else None} "
                f"current_event={row.event_id} "
                f"current_source_time_ms={source_time} "
                f"current_values={observed}"
            )
        pending_best[token] = [row]

    @classmethod
    def _consume_corrected_pending_best(
        cls,
        token: str,
        batch: _PendingBookBatch,
        starting_checksum: tuple[Decimal, Decimal],
        current: _BookState,
        depth_rows: list[_EventRow],
        pending_best: dict[str, list[_EventRow]],
    ) -> tuple[_EventRow, ...]:
        """Consume a stale/corrected BBO pair proven by one atomic transition."""

        rows = pending_best.get(token, [])
        if len(rows) < 2 or not depth_rows:
            return ()
        stale, corrected = rows[:2]
        first_depth = min(depth_rows, key=cls._event_arrival_key)
        completion = max(depth_rows, key=cls._event_arrival_key)
        stale_source_time = _timestamp(
            stale.event.get("timestamp"), name="stale best_bid_ask timestamp"
        )
        corrected_source_time = _timestamp(
            corrected.event.get("timestamp"), name="corrected best_bid_ask timestamp"
        )
        expected = (
            max(current.bids, default=Decimal("0")),
            min(current.asks, default=Decimal("1")),
        )
        stale_values = cls._best_bid_ask_values(stale)
        corrected_values = cls._best_bid_ask_values(corrected)
        top_transition = (
            starting_checksum != expected and stale_values == starting_checksum
        )
        metadata_correction = starting_checksum == expected and stale_values != expected
        stale_arrival_delta_ns = (
            corrected.received_monotonic_ns - stale.received_monotonic_ns
        )
        stale_wall_delta_ms = corrected.received_wall_ms - stale.received_wall_ms
        stale_is_bounded_prior_group = (
            0 < stale_arrival_delta_ns <= _RECENT_FULL_BOOK_STALE_BBO_MAX_ARRIVAL_NS
            and 0 <= stale_wall_delta_ms <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
        )
        if (
            not (top_transition or metadata_correction)
            or corrected_source_time != batch.source_time_ms
            or not (
                stale_source_time <= corrected_source_time
                and corrected_source_time - stale_source_time
                <= _RECENT_FULL_BOOK_STALE_BBO_MAX_AGE_MS
            )
            or any(
                row.connection_id != completion.connection_id
                or row.received_monotonic_ns != completion.received_monotonic_ns
                for row in depth_rows
            )
            or stale.connection_id != completion.connection_id
            or corrected.connection_id != completion.connection_id
            or corrected.received_monotonic_ns != completion.received_monotonic_ns
            or not (
                stale.received_monotonic_ns == completion.received_monotonic_ns
                or stale_is_bounded_prior_group
            )
            or not (
                cls._event_arrival_key(stale)
                < cls._event_arrival_key(corrected)
                < cls._event_arrival_key(first_depth)
            )
            or corrected_values != expected
        ):
            return ()
        del rows[:2]
        if not rows:
            pending_best.pop(token)
        return stale, corrected

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
                    "best_bid_ask was not temporally adjacent to its depth "
                    "transition: "
                    f"condition={batch.condition_id} token={token} "
                    f"pending_event={row.event_id} "
                    f"pending_source_time_ms={source_time} "
                    f"depth_source_time_ms={batch.source_time_ms} "
                    f"source_delta_ms={batch.source_time_ms - source_time}"
                )
            arrival_delta = completion.received_monotonic_ns - row.received_monotonic_ns
            if abs(arrival_delta) > _BEST_CORROBORATION_MAX_ARRIVAL_NS:
                raise ValueError(
                    "best_bid_ask was not arrival-adjacent to its depth transition: "
                    f"condition={batch.condition_id} token={token} "
                    f"pending_event={row.event_id} "
                    f"depth_event={completion.event_id} "
                    f"arrival_delta_ns={arrival_delta}"
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

    @classmethod
    def _apply_tick_size_change(
        cls,
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
            prior = current.last_tick_size_change_row
            prior_source_time = (
                -1
                if prior is None
                else _timestamp(
                    prior.event.get("timestamp"),
                    name="prior tick_size_change timestamp",
                )
            )
            prior_old_tick = (
                None
                if prior is None
                else _decimal(
                    prior.event.get("old_tick_size"),
                    name="prior old tick size",
                    minimum=Decimal("0.0001"),
                    maximum=Decimal("0.1"),
                )
            )
            prior_new_tick = (
                None
                if prior is None
                else _decimal(
                    prior.event.get("new_tick_size"),
                    name="prior new tick size",
                    minimum=Decimal("0.0001"),
                    maximum=Decimal("0.1"),
                )
            )
            exact_source_duplicate = source_time == prior_source_time
            bounded_receive_group_duplicate = bool(
                prior is not None
                and source_time > prior_source_time
                and source_time - prior_source_time
                <= _DUPLICATE_TICK_TRANSITION_MAX_SOURCE_SKEW_MS
                and row.received_monotonic_ns == prior.received_monotonic_ns
            )
            if (
                current.tick_size != new_tick
                or prior is None
                or prior_old_tick != old_tick
                or prior_new_tick != new_tick
                or prior.connection_id != row.connection_id
                or cls._event_arrival_key(prior) >= cls._event_arrival_key(row)
                or current.source_time_ms != prior_source_time
                or not (exact_source_duplicate or bounded_receive_group_duplicate)
            ):
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
                    "prior_event_id": prior.event_id,
                    "prior_event_sha256": prior.event_sha256,
                    "prior_source_time_ms": prior_source_time,
                    "source_time_ms": source_time,
                }
            )
            current.source_time_ms = source_time
            current.last_tick_size_change_row = row
            return
        if any(price % new_tick != 0 for price in (*current.bids, *current.asks)):
            raise ValueError("existing book is not aligned to the new tick size")
        current.tick_size = new_tick
        current.source_time_ms = source_time
        current.last_tick_size_change_row = row
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
        *,
        materialized_minimum_depth_levels: int,
        cap_materialized_depth_to_minimum_order_size: bool,
    ) -> PolymarketRecordedBook:
        outcome = "Up" if token == market.up_token_id else "Down"

        def retained_levels(
            levels: Mapping[Decimal, Decimal],
            *,
            reverse: bool,
        ) -> tuple[BookLevel, ...]:
            retained: list[BookLevel] = []
            cumulative = Decimal("0")
            for price in sorted(levels, reverse=reverse):
                quantity = levels[price]
                retained.append(BookLevel(price, quantity))
                cumulative += quantity
                if (
                    cap_materialized_depth_to_minimum_order_size
                    and len(retained) >= materialized_minimum_depth_levels
                    and cumulative >= market.minimum_order_size
                ):
                    break
            return tuple(retained)

        snapshot = PaperBookSnapshot(
            venue="polymarket",
            market_id=market.condition_id,
            asset_id=token,
            bids=retained_levels(state.bids, reverse=True),
            asks=retained_levels(state.asks, reverse=False),
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
