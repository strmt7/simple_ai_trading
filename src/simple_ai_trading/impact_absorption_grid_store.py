"""Transactional builder and auditor for the Round 73 causal feature grid."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import struct
import time

import duckdb
import numpy as np

from .impact_absorption import (
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
    pre_event_level_band,
    validate_combined_stream_name,
)

from .impact_absorption_corpus import (
    ROUND73_CORPUS_RUN_TABLE,
    Round73CorpusManifestAudit,
    audit_round73_corpus_manifest,
)
from .impact_absorption_grid import (
    ROUND73_GRID_BANDS,
    ROUND73_GRID_CONTRACT_SHA256,
    ROUND73_GRID_FEATURE_NAMES,
    ROUND73_GRID_FEATURE_NAMES_SHA256,
    ROUND73_GRID_SCHEMA_VERSION,
    ROUND73_GRID_STEP_NS,
    ROUND73_GRID_WARMUP_NS,
    Round73CausalGridAccumulator,
    Round73L2State,
    Round73MarkState,
    Round73OpenInterestState,
    round73_grid_invalid_reasons,
)
from .impact_absorption_store import (
    IMPACT_AGGREGATE_TRADE_TABLE,
    IMPACT_BOOK_TICKER_TABLE,
    IMPACT_CAPTURE_CONTRACT_SHA256,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_DEPTH_BAND_FLOW_TABLE,
    IMPACT_DEPTH_UPDATE_TABLE,
    IMPACT_EVENT_LINK_TABLE,
    IMPACT_L2_STATE_TABLE,
    IMPACT_LIQUIDATION_SNAPSHOT_TABLE,
    IMPACT_MARK_PRICE_TABLE,
    IMPACT_REST_EVENT_TABLE,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    ImpactAbsorptionStore,
    iter_impact_capture_v9_records,
    load_impact_capture_v9_preflight,
)


ROUND73_GRID_ANCHOR_TABLE = "impact_feature_anchor_v3"
ROUND73_GRID_VECTOR_TABLE = "impact_feature_vector_v3"
ROUND73_GRID_MANIFEST_TABLE = "impact_feature_run_manifest_v3"
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FETCH_BATCH_SIZE = 8_192

CorpusAuditFunction = Callable[..., Round73CorpusManifestAudit]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _strict_json_object(raw_text: str, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key is forbidden in {label}: {key}")
            output[key] = value
        return output

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
    )


def _validated_run_id(value: str) -> str:
    selected = str(value).strip().lower()
    if _RUN_ID.fullmatch(selected) is None:
        raise ValueError("Round 73 grid run ID must be 32 lowercase hex characters")
    return selected


@dataclass(frozen=True)
class _CausalClockTimeline:
    received_monotonic_ns: tuple[int, ...]
    clock_offset_ns: tuple[int, ...]

    def offset_strictly_before(self, receipt_monotonic_ns: int, fallback: int) -> int:
        index = bisect_left(self.received_monotonic_ns, int(receipt_monotonic_ns)) - 1
        return int(fallback) if index < 0 else self.clock_offset_ns[index]


def _vector_sha256(
    run_id: str, symbol: str, anchor_index: int, values: tuple[float, ...]
) -> str:
    identity = f"{run_id}:{symbol}:{int(anchor_index)}:".encode("ascii")
    payload = struct.pack(f"<{len(values)}d", *values)
    return hashlib.sha256(identity + payload).hexdigest()


def _anchor_rows_sha256(rows: list[tuple[object, ...]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item[1]), int(item[2]))):
        identity = [
            str(row[0]),
            str(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
            bool(row[6]),
            int(row[7]),
            str(row[8]),
            float(row[9]),
            float(row[10]),
            None if row[11] is None else float(row[11]),
            None if row[12] is None else float(row[12]),
            int(row[13]),
            float(row[14]),
        ]
        digest.update(_canonical_json(identity).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _vector_rows_sha256(rows: list[tuple[object, ...]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item[1]), int(item[2]))):
        identity = [str(row[0]), str(row[1]), int(row[2]), str(row[4])]
        digest.update(_canonical_json(identity).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Round73GridBuildReport:
    run_id: str
    source_corpus_manifest_sha256: str
    anchor_count: int
    valid_anchor_count: int
    vector_count: int
    first_anchor_wall_ns: int
    last_anchor_wall_ns: int
    per_symbol: dict[str, dict[str, int]]
    build_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = ROUND73_GRID_SCHEMA_VERSION
        payload["contract_sha256"] = ROUND73_GRID_CONTRACT_SHA256
        payload["feature_names_sha256"] = ROUND73_GRID_FEATURE_NAMES_SHA256
        payload["feature_count"] = len(ROUND73_GRID_FEATURE_NAMES)
        payload["target_constructed"] = False
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


@dataclass(frozen=True)
class Round73GridBuildAudit:
    run_id: str
    passed: bool
    errors: tuple[str, ...]
    anchor_count: int
    valid_anchor_count: int
    vector_count: int
    build_manifest_sha256: str
    source_corpus_manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "round-073-grid-build-audit-v1"
        payload["contract_sha256"] = ROUND73_GRID_CONTRACT_SHA256
        payload["errors"] = list(self.errors)
        payload["target_constructed"] = False
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


def _create_grid_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_GRID_ANCHOR_TABLE} (
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            anchor_monotonic_ns UBIGINT NOT NULL,
            anchor_wall_ns UBIGINT NOT NULL,
            source_max_received_monotonic_ns UBIGINT NOT NULL,
            valid BOOLEAN NOT NULL,
            invalid_reason_mask UBIGINT NOT NULL,
            invalid_reasons_json VARCHAR NOT NULL,
            signed_aggressive_quote_1s DOUBLE NOT NULL,
            absolute_aggressive_quote_1s DOUBLE NOT NULL,
            trailing_median_absolute_aggressive_quote_60s DOUBLE,
            shock_ratio DOUBLE,
            shock_direction TINYINT NOT NULL,
            shock_direction_taker_share DOUBLE NOT NULL,
            PRIMARY KEY (run_id, symbol, anchor_index)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_GRID_VECTOR_TABLE} (
            run_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            anchor_index UINTEGER NOT NULL,
            feature_values DOUBLE[] NOT NULL,
            vector_sha256 VARCHAR NOT NULL CHECK (length(vector_sha256) = 64),
            PRIMARY KEY (run_id, symbol, anchor_index)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_GRID_MANIFEST_TABLE} (
            run_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            source_corpus_manifest_sha256 VARCHAR NOT NULL,
            feature_names_json VARCHAR NOT NULL,
            feature_names_sha256 VARCHAR NOT NULL,
            build_manifest_json VARCHAR NOT NULL,
            build_manifest_sha256 VARCHAR NOT NULL,
            anchor_count UINTEGER NOT NULL,
            valid_anchor_count UINTEGER NOT NULL,
            vector_count UINTEGER NOT NULL,
            first_anchor_wall_ns UBIGINT NOT NULL,
            last_anchor_wall_ns UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(source_corpus_manifest_sha256) = 64),
            CHECK (length(feature_names_sha256) = 64),
            CHECK (length(build_manifest_sha256) = 64)
        )
        """
    )


def _assert_grid_table_shapes(connection: duckdb.DuckDBPyConnection) -> None:
    expected = {
        ROUND73_GRID_ANCHOR_TABLE: (
            "run_id",
            "symbol",
            "anchor_index",
            "anchor_monotonic_ns",
            "anchor_wall_ns",
            "source_max_received_monotonic_ns",
            "valid",
            "invalid_reason_mask",
            "invalid_reasons_json",
            "signed_aggressive_quote_1s",
            "absolute_aggressive_quote_1s",
            "trailing_median_absolute_aggressive_quote_60s",
            "shock_ratio",
            "shock_direction",
            "shock_direction_taker_share",
        ),
        ROUND73_GRID_VECTOR_TABLE: (
            "run_id",
            "symbol",
            "anchor_index",
            "feature_values",
            "vector_sha256",
        ),
        ROUND73_GRID_MANIFEST_TABLE: (
            "run_id",
            "schema_version",
            "contract_sha256",
            "source_corpus_manifest_sha256",
            "feature_names_json",
            "feature_names_sha256",
            "build_manifest_json",
            "build_manifest_sha256",
            "anchor_count",
            "valid_anchor_count",
            "vector_count",
            "first_anchor_wall_ns",
            "last_anchor_wall_ns",
            "recorded_at_wall_ns",
        ),
    }
    for table, columns in expected.items():
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        )
        if observed != columns:
            raise RuntimeError(f"Round 73 grid table schema differs: {table}")


def _insert_rows_columnar(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[tuple[object, ...]],
) -> None:
    if not rows:
        return
    views: list[str] = []
    projections: list[str] = []
    try:
        for index, column in enumerate(zip(*rows, strict=True)):
            values = tuple(column)
            view = f"_round73_{table}_{index}"
            if all(isinstance(value, bool) for value in values):
                array = np.asarray(values, dtype=np.bool_)
                projection = f"{view}.column0"
            elif all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                array = np.asarray(values, dtype=np.int64)
                projection = f"{view}.column0"
            elif all(
                value is None or isinstance(value, (int, float)) for value in values
            ):
                array = np.asarray(
                    [np.nan if value is None else value for value in values],
                    dtype=np.float64,
                )
                projection = (
                    f"CASE WHEN isnan({view}.column0) THEN NULL ELSE {view}.column0 END"
                )
            elif all(isinstance(value, str) for value in values):
                array = np.asarray(values, dtype=np.str_)
                projection = f"{view}.column0"
            elif all(isinstance(value, (list, tuple)) for value in values):
                widths = {len(value) for value in values}
                if len(widths) != 1 or not widths or min(widths) < 1:
                    raise ValueError("Round 73 nested insert width differs")
                array = np.asarray(values, dtype=np.float64).T
                projection = (
                    "list_value("
                    + ", ".join(
                        f"{view}.column{nested_index}"
                        for nested_index in range(array.shape[0])
                    )
                    + ")"
                )
            else:
                raise TypeError("Round 73 grid insert column type is unsupported")
            connection.register(view, array)
            views.append(view)
            projections.append(projection)
        sources = " POSITIONAL JOIN ".join(views)
        connection.execute(
            f"INSERT INTO {table} SELECT {', '.join(projections)} FROM {sources}"
        )
    finally:
        for view in views:
            connection.unregister(view)


def _cursor_rows(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: list[object],
) -> Iterator[tuple[object, ...]]:
    cursor = connection.cursor()
    cursor.execute(query, parameters)
    try:
        while rows := cursor.fetchmany(_FETCH_BATCH_SIZE):
            yield from (tuple(row) for row in rows)
    finally:
        cursor.close()


def _event_sources(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    symbol: str,
    coverage_start_monotonic_ns: int,
    coverage_end_monotonic_ns: int,
) -> list[tuple[str, Iterator[tuple[object, ...]]]]:
    key = "e.received_monotonic_ns, e.frame_index, e.message_index"
    common = [
        run_id,
        symbol,
        int(coverage_start_monotonic_ns),
        int(coverage_end_monotonic_ns),
    ]
    interval = "AND e.received_monotonic_ns >= ? AND e.received_monotonic_ns < ?"
    band_columns = ", ".join(
        f"f.{side}_{action}_{band}"
        for side in ("bid", "ask")
        for action in ("added_quote", "removed_quote")
        for band in ROUND73_GRID_BANDS
    )
    return [
        (
            "bbo",
            _cursor_rows(
                connection,
                f"""
                SELECT e.received_monotonic_ns, e.frame_index, e.message_index,
                       e.received_wall_ns, e.event_time_ms,
                       b.bid, b.bid_qty, b.ask, b.ask_qty
                FROM {IMPACT_EVENT_LINK_TABLE} e
                JOIN {IMPACT_BOOK_TICKER_TABLE} b USING (run_id, frame_index, message_index)
                WHERE e.run_id = ? AND e.symbol = ? AND e.event_type = 'bookTicker'
                  {interval}
                ORDER BY {key}
                """,
                common,
            ),
        ),
        (
            "depth",
            _cursor_rows(
                connection,
                f"""
                SELECT e.received_monotonic_ns, e.frame_index, e.message_index,
                       e.received_wall_ns, e.event_time_ms, d.stale,
                       l.bid_prices, l.bid_quantities, l.ask_prices, l.ask_quantities,
                       l.bid_depth_quote_5, l.ask_depth_quote_5,
                       l.bid_depth_quote_10, l.ask_depth_quote_10,
                       l.bid_depth_quote_20, l.ask_depth_quote_20,
                       l.imbalance_5, l.imbalance_10, l.imbalance_20,
                       {band_columns}
                FROM {IMPACT_EVENT_LINK_TABLE} e
                JOIN {IMPACT_DEPTH_UPDATE_TABLE} d USING (run_id, frame_index, message_index)
                LEFT JOIN {IMPACT_L2_STATE_TABLE} l USING (run_id, frame_index, message_index)
                JOIN {IMPACT_DEPTH_BAND_FLOW_TABLE} f USING (run_id, frame_index, message_index)
                WHERE e.run_id = ? AND e.symbol = ? AND e.event_type = 'depthUpdate'
                  {interval}
                ORDER BY {key}
                """,
                common,
            ),
        ),
        (
            "trade",
            _cursor_rows(
                connection,
                f"""
                SELECT e.received_monotonic_ns, e.frame_index, e.message_index,
                       t.price, t.normalized_qty, t.buyer_is_maker
                FROM {IMPACT_EVENT_LINK_TABLE} e
                JOIN {IMPACT_AGGREGATE_TRADE_TABLE} t USING (run_id, frame_index, message_index)
                WHERE e.run_id = ? AND e.symbol = ? AND e.event_type = 'aggTrade'
                  {interval}
                ORDER BY {key}
                """,
                common,
            ),
        ),
        (
            "mark",
            _cursor_rows(
                connection,
                f"""
                SELECT e.received_monotonic_ns, e.frame_index, e.message_index,
                       m.mark_price, m.index_price, m.funding_rate,
                       m.next_funding_time_ms
                FROM {IMPACT_EVENT_LINK_TABLE} e
                JOIN {IMPACT_MARK_PRICE_TABLE} m USING (run_id, frame_index, message_index)
                WHERE e.run_id = ? AND e.symbol = ?
                  AND e.event_type = 'markPriceUpdate'
                  {interval}
                ORDER BY {key}
                """,
                common,
            ),
        ),
        (
            "open_interest",
            _cursor_rows(
                connection,
                f"""
                SELECT e.received_monotonic_ns, e.frame_index, e.message_index,
                       r.open_interest
                FROM {IMPACT_EVENT_LINK_TABLE} e
                JOIN {IMPACT_REST_EVENT_TABLE} r USING (run_id, frame_index, message_index)
                WHERE e.run_id = ? AND e.symbol = ?
                  AND e.event_type = 'openInterest'
                  {interval}
                ORDER BY {key}
                """,
                common,
            ),
        ),
        (
            "liquidation",
            _cursor_rows(
                connection,
                f"""
                SELECT e.received_monotonic_ns, e.frame_index, e.message_index,
                       q.average_price, q.price, q.last_filled_qty
                FROM {IMPACT_EVENT_LINK_TABLE} e
                JOIN {IMPACT_LIQUIDATION_SNAPSHOT_TABLE} q
                  USING (run_id, frame_index, message_index)
                WHERE e.run_id = ? AND e.symbol = ? AND e.event_type = 'forceOrder'
                  {interval}
                ORDER BY {key}
                """,
                common,
            ),
        ),
    ]


def _causal_clock_timeline(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> _CausalClockTimeline:
    rows = connection.execute(
        f"""
        SELECT e.received_monotonic_ns, e.received_wall_ns,
               r.request_started_monotonic_ns, r.request_started_wall_ns,
               r.exchange_time_ms
        FROM {IMPACT_EVENT_LINK_TABLE} e
        JOIN {IMPACT_REST_EVENT_TABLE} r
          USING (run_id, frame_index, message_index)
        WHERE e.run_id = ? AND e.event_type = 'serverTime'
        ORDER BY e.received_monotonic_ns, e.frame_index, e.message_index
        """,
        [run_id],
    ).fetchall()
    receipt_times: list[int] = []
    offsets: list[int] = []
    best_rtt: int | None = None
    best_offset = 0
    for row in rows:
        received_monotonic_ns = int(row[0])
        received_wall_ns = int(row[1])
        request_started_monotonic_ns = int(row[2])
        request_started_wall_ns = int(row[3])
        rtt_ns = received_monotonic_ns - request_started_monotonic_ns
        if rtt_ns < 0 or received_wall_ns < request_started_wall_ns:
            raise ValueError("Round 73 server-time probe clocks are reversed")
        if best_rtt is None or rtt_ns < best_rtt:
            best_rtt = rtt_ns
            midpoint_wall_ns = (request_started_wall_ns + received_wall_ns) // 2
            best_offset = int(row[4]) * 1_000_000 - midpoint_wall_ns
        receipt_times.append(received_monotonic_ns)
        offsets.append(best_offset)
    return _CausalClockTimeline(tuple(receipt_times), tuple(offsets))


def _v9_rest_context(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> dict[tuple[int, int], dict[str, object]]:
    rows = connection.execute(
        f"""
        SELECT frame_index, message_index, event_type, symbol,
               request_started_wall_ns, request_started_monotonic_ns,
               exchange_time_ms, open_interest
        FROM {IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE}
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    result = {
        (int(row[0]), int(row[1])): {
            "event_type": str(row[2]),
            "symbol": str(row[3]),
            "request_started_wall_ns": int(row[4]),
            "request_started_monotonic_ns": int(row[5]),
            "exchange_time_ms": None if row[6] is None else int(row[6]),
            "open_interest": None if row[7] is None else float(row[7]),
        }
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("Round 73 v9 REST context contains duplicate keys")
    return result


def _v9_causal_clock_timeline(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    rest_context: Mapping[tuple[int, int], Mapping[str, object]],
) -> _CausalClockTimeline:
    receipt_times: list[int] = []
    offsets: list[int] = []
    best_rtt: int | None = None
    best_offset = 0
    for frame_index, message_index, record in iter_impact_capture_v9_records(
        connection,
        run_id=run_id,
    ):
        context = rest_context.get((frame_index, message_index))
        if context is None or context.get("event_type") != "serverTime":
            continue
        exchange_time_ms = context.get("exchange_time_ms")
        if exchange_time_ms is None:
            raise ValueError("Round 73 v9 server-time context is incomplete")
        request_started_monotonic_ns = int(context["request_started_monotonic_ns"])
        request_started_wall_ns = int(context["request_started_wall_ns"])
        rtt_ns = record.received_monotonic_ns - request_started_monotonic_ns
        if rtt_ns < 0 or record.received_wall_ns < request_started_wall_ns:
            raise ValueError("Round 73 v9 server-time probe clocks are reversed")
        if best_rtt is None or rtt_ns < best_rtt:
            best_rtt = rtt_ns
            midpoint_wall_ns = (request_started_wall_ns + record.received_wall_ns) // 2
            best_offset = int(exchange_time_ms) * 1_000_000 - midpoint_wall_ns
        receipt_times.append(record.received_monotonic_ns)
        offsets.append(best_offset)
    return _CausalClockTimeline(tuple(receipt_times), tuple(offsets))


def _v9_depth_band_flow(pre_state, changes) -> dict[str, dict[str, dict[str, float]]]:
    flow = {
        side: {
            band: {"added_quote": 0.0, "removed_quote": 0.0}
            for band in ROUND73_GRID_BANDS
        }
        for side in ("bid", "ask")
    }
    for change in changes:
        band = pre_event_level_band(pre_state, change)
        bucket = flow[change.side][band]
        bucket["added_quote"] += change.added_quote
        bucket["removed_quote"] += change.removed_quote
    return flow


def _merged_events(
    sources: list[tuple[str, Iterator[tuple[object, ...]]]],
) -> Iterator[tuple[str, tuple[object, ...]]]:
    heap: list[
        tuple[int, int, int, int, str, tuple[object, ...], Iterator[tuple[object, ...]]]
    ] = []
    for rank, (kind, iterator) in enumerate(sources):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (int(row[0]), int(row[1]), int(row[2]), rank, kind, row, iterator),
        )
    while heap:
        _timestamp, _frame, _message, rank, kind, row, iterator = heapq.heappop(heap)
        yield kind, row
        try:
            following = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                int(following[0]),
                int(following[1]),
                int(following[2]),
                rank,
                kind,
                following,
                iterator,
            ),
        )


def _corrected_latency_ms(
    *,
    received_wall_ns: int,
    event_time_ms: int,
    clock_offset_ns: int,
) -> float:
    return (
        int(received_wall_ns) + int(clock_offset_ns) - int(event_time_ms) * 1_000_000
    ) / 1_000_000.0


def _process_event(
    accumulator: Round73CausalGridAccumulator,
    kind: str,
    row: tuple[object, ...],
    *,
    clock_timeline: _CausalClockTimeline,
    fallback_clock_offset_ns: int,
) -> None:
    timestamp = int(row[0])
    clock_offset_ns = clock_timeline.offset_strictly_before(
        timestamp,
        fallback_clock_offset_ns,
    )
    if kind == "bbo":
        accumulator.observe_bbo(
            received_monotonic_ns=timestamp,
            bid=float(row[5]),
            bid_qty=float(row[6]),
            ask=float(row[7]),
            ask_qty=float(row[8]),
            corrected_event_latency_ms=_corrected_latency_ms(
                received_wall_ns=int(row[3]),
                event_time_ms=int(row[4]),
                clock_offset_ns=clock_offset_ns,
            ),
        )
        return
    if kind == "depth":
        if bool(row[5]):
            return
        if any(value is None for value in row[6:19]):
            raise ValueError("Round 73 synchronized depth state is missing")
        state = Round73L2State(
            received_monotonic_ns=timestamp,
            bid_prices=tuple(float(value) for value in row[6]),
            bid_quantities=tuple(float(value) for value in row[7]),
            ask_prices=tuple(float(value) for value in row[8]),
            ask_quantities=tuple(float(value) for value in row[9]),
            bid_depth_quote_5=float(row[10]),
            ask_depth_quote_5=float(row[11]),
            bid_depth_quote_10=float(row[12]),
            ask_depth_quote_10=float(row[13]),
            bid_depth_quote_20=float(row[14]),
            ask_depth_quote_20=float(row[15]),
            imbalance_5=float(row[16]),
            imbalance_10=float(row[17]),
            imbalance_20=float(row[18]),
            corrected_event_latency_ms=_corrected_latency_ms(
                received_wall_ns=int(row[3]),
                event_time_ms=int(row[4]),
                clock_offset_ns=clock_offset_ns,
            ),
        )
        values = tuple(float(value) for value in row[19:35])
        index = 0
        flow: dict[str, dict[str, dict[str, float]]] = {}
        for side in ("bid", "ask"):
            flow[side] = {band: {} for band in ROUND73_GRID_BANDS}
            for action in ("added_quote", "removed_quote"):
                for band in ROUND73_GRID_BANDS:
                    flow[side][band][action] = values[index]
                    index += 1
        accumulator.observe_l2(state=state, depth_band_flow=flow)
        return
    if kind == "trade":
        accumulator.observe_trade(
            received_monotonic_ns=timestamp,
            price=float(row[3]),
            quantity=float(row[4]),
            buyer_is_maker=bool(row[5]),
        )
        return
    if kind == "mark":
        accumulator.observe_mark(
            Round73MarkState(
                received_monotonic_ns=timestamp,
                mark_price=float(row[3]),
                index_price=float(row[4]),
                funding_rate=float(row[5]),
                next_funding_time_ms=int(row[6]),
            )
        )
        return
    if kind == "open_interest":
        accumulator.observe_open_interest(
            Round73OpenInterestState(
                received_monotonic_ns=timestamp,
                open_interest=float(row[3]),
            )
        )
        return
    if kind == "liquidation":
        average_price = float(row[3])
        price = average_price if average_price > 0 else float(row[4])
        accumulator.observe_liquidation(
            received_monotonic_ns=timestamp,
            price=price,
            last_filled_quantity=float(row[5]),
        )
        return
    raise ValueError(f"unsupported Round 73 grid event kind: {kind}")


def _symbol_grid_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    symbol: str,
    clock_timeline: _CausalClockTimeline,
    fallback_clock_offset_ns: int,
    run_started_wall_ns: int,
    run_started_monotonic_ns: int,
    coverage_start_wall_ns: int,
    coverage_end_wall_ns: int,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], dict[str, int]]:
    coverage_start_mono = run_started_monotonic_ns + (
        coverage_start_wall_ns - run_started_wall_ns
    )
    coverage_end_mono = run_started_monotonic_ns + (
        coverage_end_wall_ns - run_started_wall_ns
    )
    first_anchor = (
        coverage_start_mono // ROUND73_GRID_STEP_NS + 1
    ) * ROUND73_GRID_STEP_NS
    first_persisted_anchor = first_anchor + ROUND73_GRID_WARMUP_NS
    accumulator = Round73CausalGridAccumulator(symbol)
    anchor_rows: list[tuple[object, ...]] = []
    vector_rows: list[tuple[object, ...]] = []
    anchor = first_anchor
    anchor_index = 0

    def emit_anchor() -> None:
        nonlocal anchor_index
        anchor_wall_ns = run_started_wall_ns + (anchor - run_started_monotonic_ns)
        result = accumulator.emit(
            anchor_monotonic_ns=anchor,
            anchor_wall_ns=anchor_wall_ns,
        )
        if anchor < first_persisted_anchor:
            return
        anchor_rows.append(
            (
                run_id,
                symbol,
                anchor_index,
                result.anchor_monotonic_ns,
                result.anchor_wall_ns,
                result.source_max_received_monotonic_ns,
                result.valid,
                result.invalid_reason_mask,
                _canonical_json(list(result.invalid_reasons)),
                result.signed_aggressive_quote_1s,
                result.absolute_aggressive_quote_1s,
                result.trailing_median_absolute_aggressive_quote_60s,
                result.shock_ratio,
                result.shock_direction,
                result.shock_direction_taker_share,
            )
        )
        if result.feature_values is not None:
            vector_rows.append(
                (
                    run_id,
                    symbol,
                    anchor_index,
                    list(result.feature_values),
                    _vector_sha256(
                        run_id,
                        symbol,
                        anchor_index,
                        result.feature_values,
                    ),
                )
            )
        anchor_index += 1

    for kind, row in _merged_events(
        _event_sources(
            connection,
            run_id=run_id,
            symbol=symbol,
            coverage_start_monotonic_ns=coverage_start_mono,
            coverage_end_monotonic_ns=coverage_end_mono,
        )
    ):
        event_mono = int(row[0])
        while anchor <= coverage_end_mono and anchor <= event_mono:
            emit_anchor()
            anchor += ROUND73_GRID_STEP_NS
        if event_mono < coverage_end_mono:
            _process_event(
                accumulator,
                kind,
                row,
                clock_timeline=clock_timeline,
                fallback_clock_offset_ns=fallback_clock_offset_ns,
            )
    while anchor <= coverage_end_mono:
        emit_anchor()
        anchor += ROUND73_GRID_STEP_NS
    return (
        anchor_rows,
        vector_rows,
        {
            "anchors": len(anchor_rows),
            "valid_anchors": len(vector_rows),
            "invalid_anchors": len(anchor_rows) - len(vector_rows),
        },
    )


def _v9_grid_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    segments: list[tuple[object, ...]],
    run_started_wall_ns: int,
    run_started_monotonic_ns: int,
    coverage_start_wall_ns: int,
    coverage_end_wall_ns: int,
) -> tuple[
    list[tuple[object, ...]],
    list[tuple[object, ...]],
    dict[str, dict[str, int]],
]:
    preflight = load_impact_capture_v9_preflight(connection, run_id=run_id)
    if coverage_start_wall_ns < preflight.ready_wall_ns:
        raise ValueError("Round 73 v9 grid coverage precedes feature-ready marker")
    coverage_start_mono = run_started_monotonic_ns + (
        coverage_start_wall_ns - run_started_wall_ns
    )
    coverage_end_mono = run_started_monotonic_ns + (
        coverage_end_wall_ns - run_started_wall_ns
    )
    first_anchor = (
        coverage_start_mono // ROUND73_GRID_STEP_NS + 1
    ) * ROUND73_GRID_STEP_NS
    first_persisted_anchor = first_anchor + ROUND73_GRID_WARMUP_NS
    accumulators = {
        str(symbol): Round73CausalGridAccumulator(str(symbol))
        for symbol, _status, _clock_offset, _tick_size in segments
    }
    books = {
        str(symbol): SynchronizedDepthBook(str(symbol), float(tick_size))
        for symbol, _status, _clock_offset, tick_size in segments
    }
    snapshot_records = dict(preflight.snapshot_records)
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        books[symbol].initialize(
            _strict_json_object(
                snapshot_records[symbol].raw_text,
                "v9 preloaded depth snapshot",
            )
        )
    fallback_offsets = {
        str(symbol): int(clock_offset)
        for symbol, _status, clock_offset, _tick_size in segments
    }
    rest_context = _v9_rest_context(connection, run_id=run_id)
    clock_timeline = _v9_causal_clock_timeline(
        connection,
        run_id=run_id,
        rest_context=rest_context,
    )
    anchor_rows: list[tuple[object, ...]] = []
    vector_rows: list[tuple[object, ...]] = []
    per_symbol = {
        symbol: {"anchors": 0, "valid_anchors": 0, "invalid_anchors": 0}
        for symbol in accumulators
    }
    anchor_indices = {symbol: 0 for symbol in accumulators}
    anchor = first_anchor
    observed_snapshot_symbols: set[str] = set()

    def emit_anchor() -> None:
        for symbol in IMPACT_CAPTURE_SYMBOLS:
            accumulator = accumulators[symbol]
            anchor_wall_ns = run_started_wall_ns + (anchor - run_started_monotonic_ns)
            result = accumulator.emit(
                anchor_monotonic_ns=anchor,
                anchor_wall_ns=anchor_wall_ns,
            )
            if anchor < first_persisted_anchor:
                continue
            anchor_index = anchor_indices[symbol]
            anchor_rows.append(
                (
                    run_id,
                    symbol,
                    anchor_index,
                    result.anchor_monotonic_ns,
                    result.anchor_wall_ns,
                    result.source_max_received_monotonic_ns,
                    result.valid,
                    result.invalid_reason_mask,
                    _canonical_json(list(result.invalid_reasons)),
                    result.signed_aggressive_quote_1s,
                    result.absolute_aggressive_quote_1s,
                    result.trailing_median_absolute_aggressive_quote_60s,
                    result.shock_ratio,
                    result.shock_direction,
                    result.shock_direction_taker_share,
                )
            )
            per_symbol[symbol]["anchors"] += 1
            if result.feature_values is not None:
                vector_rows.append(
                    (
                        run_id,
                        symbol,
                        anchor_index,
                        list(result.feature_values),
                        _vector_sha256(
                            run_id,
                            symbol,
                            anchor_index,
                            result.feature_values,
                        ),
                    )
                )
                per_symbol[symbol]["valid_anchors"] += 1
            else:
                per_symbol[symbol]["invalid_anchors"] += 1
            anchor_indices[symbol] += 1

    for frame_index, message_index, record in iter_impact_capture_v9_records(
        connection,
        run_id=run_id,
    ):
        event_mono = int(record.received_monotonic_ns)
        if event_mono >= coverage_start_mono:
            while anchor <= coverage_end_mono and anchor <= event_mono:
                emit_anchor()
                anchor += ROUND73_GRID_STEP_NS
        if event_mono >= coverage_end_mono:
            break
        key = (frame_index, message_index)
        context = rest_context.get(key)
        if record.stream == "binance_futures_rest":
            if context is None:
                raise ValueError("Round 73 v9 grid REST context is missing")
            event_type = str(context["event_type"])
            symbol = str(context["symbol"])
            if event_type == "depthSnapshot":
                if symbol not in books:
                    raise ValueError("Round 73 v9 grid snapshot symbol differs")
                if record != snapshot_records[symbol]:
                    raise ValueError("Round 73 v9 grid preloaded snapshot differs")
                observed_snapshot_symbols.add(symbol)
            elif (
                event_type == "openInterest"
                and record.received_wall_ns >= coverage_start_wall_ns
            ):
                value = context.get("open_interest")
                if symbol not in accumulators or value is None:
                    raise ValueError("Round 73 v9 grid open interest is incomplete")
                accumulators[symbol].observe_open_interest(
                    Round73OpenInterestState(
                        received_monotonic_ns=event_mono,
                        open_interest=float(value),
                    )
                )
            continue
        if ImpactAbsorptionStore._v9_websocket_event_type(record) == "rejectedWire":
            continue
        root = _strict_json_object(record.raw_text, "v9 WebSocket event")
        payload = root.get("data")
        if not isinstance(payload, Mapping):
            raise ValueError("Round 73 v9 grid WebSocket payload is missing")
        event_type = str(payload.get("e", ""))
        symbol_value = payload.get("s")
        if event_type == "forceOrder" and isinstance(payload.get("o"), Mapping):
            symbol_value = payload["o"].get("s")
        symbol = str(symbol_value).strip().upper()
        if symbol not in accumulators:
            raise ValueError("Round 73 v9 grid symbol differs")
        validate_combined_stream_name(
            str(root.get("stream", "")),
            event_type=event_type,
            symbol=symbol,
        )
        if event_type == "depthUpdate":
            book = books[symbol]
            feature_eligible = record.received_wall_ns >= coverage_start_wall_ns
            pre_state = book.state(20) if feature_eligible else None
            event = book.apply(payload, receive_time_ns=event_mono)
            if event.stale or not feature_eligible:
                continue
            if pre_state is None:
                raise RuntimeError("Round 73 v9 eligible pre-event L2 is missing")
            rebuilt = book.state(20)
            accumulators[symbol].observe_l2(
                state=Round73L2State(
                    received_monotonic_ns=event_mono,
                    bid_prices=tuple(price for price, _qty in rebuilt.bid_levels),
                    bid_quantities=tuple(qty for _price, qty in rebuilt.bid_levels),
                    ask_prices=tuple(price for price, _qty in rebuilt.ask_levels),
                    ask_quantities=tuple(qty for _price, qty in rebuilt.ask_levels),
                    bid_depth_quote_5=rebuilt.bid_depth_quote_5,
                    ask_depth_quote_5=rebuilt.ask_depth_quote_5,
                    bid_depth_quote_10=rebuilt.bid_depth_quote_10,
                    ask_depth_quote_10=rebuilt.ask_depth_quote_10,
                    bid_depth_quote_20=rebuilt.bid_depth_quote_20,
                    ask_depth_quote_20=rebuilt.ask_depth_quote_20,
                    imbalance_5=rebuilt.imbalance_5,
                    imbalance_10=rebuilt.imbalance_10,
                    imbalance_20=rebuilt.imbalance_20,
                    corrected_event_latency_ms=_corrected_latency_ms(
                        received_wall_ns=record.received_wall_ns,
                        event_time_ms=event.event_time_ms,
                        clock_offset_ns=clock_timeline.offset_strictly_before(
                            event_mono,
                            fallback_offsets[symbol],
                        ),
                    ),
                ),
                depth_band_flow=_v9_depth_band_flow(pre_state, event.changes),
            )
            continue
        if record.received_wall_ns < coverage_start_wall_ns:
            continue
        if event_type == "bookTicker":
            event = parse_book_ticker(
                payload,
                symbol=symbol,
                receive_time_ns=event_mono,
            )
            accumulators[symbol].observe_bbo(
                received_monotonic_ns=event_mono,
                bid=event.bid,
                bid_qty=event.bid_qty,
                ask=event.ask,
                ask_qty=event.ask_qty,
                corrected_event_latency_ms=_corrected_latency_ms(
                    received_wall_ns=record.received_wall_ns,
                    event_time_ms=event.event_time_ms,
                    clock_offset_ns=clock_timeline.offset_strictly_before(
                        event_mono,
                        fallback_offsets[symbol],
                    ),
                ),
            )
        elif event_type == "aggTrade":
            event = parse_aggregate_trade(
                payload,
                symbol=symbol,
                receive_time_ns=event_mono,
            )
            accumulators[symbol].observe_trade(
                received_monotonic_ns=event_mono,
                price=event.price,
                quantity=event.normalized_qty,
                buyer_is_maker=event.buyer_is_maker,
            )
        elif event_type == "markPriceUpdate":
            event = parse_mark_price(
                payload,
                symbol=symbol,
                receive_time_ns=event_mono,
            )
            accumulators[symbol].observe_mark(
                Round73MarkState(
                    received_monotonic_ns=event_mono,
                    mark_price=event.mark_price,
                    index_price=event.index_price,
                    funding_rate=event.funding_rate,
                    next_funding_time_ms=event.next_funding_time_ms,
                )
            )
        elif event_type == "forceOrder":
            event = parse_liquidation_snapshot(
                payload,
                symbol=symbol,
                receive_time_ns=event_mono,
            )
            price = event.average_price if event.average_price > 0 else event.price
            accumulators[symbol].observe_liquidation(
                received_monotonic_ns=event_mono,
                price=price,
                last_filled_quantity=event.last_filled_qty,
            )
    if tuple(sorted(observed_snapshot_symbols)) != IMPACT_CAPTURE_SYMBOLS:
        raise ValueError("Round 73 v9 grid snapshot replay is incomplete")
    while anchor <= coverage_end_mono:
        emit_anchor()
        anchor += ROUND73_GRID_STEP_NS
    return anchor_rows, vector_rows, per_symbol


def _build_identity(
    *,
    run_id: str,
    source_manifest_sha256: str,
    anchor_rows: list[tuple[object, ...]],
    vector_rows: list[tuple[object, ...]],
    per_symbol: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    return {
        "schema_version": ROUND73_GRID_SCHEMA_VERSION,
        "contract_sha256": ROUND73_GRID_CONTRACT_SHA256,
        "run_id": run_id,
        "source_corpus_manifest_sha256": source_manifest_sha256,
        "feature_names_sha256": ROUND73_GRID_FEATURE_NAMES_SHA256,
        "feature_count": len(ROUND73_GRID_FEATURE_NAMES),
        "anchor_count": len(anchor_rows),
        "valid_anchor_count": len(vector_rows),
        "vector_count": len(vector_rows),
        "anchor_rows_sha256": _anchor_rows_sha256(anchor_rows),
        "vector_rows_sha256": _vector_rows_sha256(vector_rows),
        "first_anchor_wall_ns": min(int(row[4]) for row in anchor_rows),
        "last_anchor_wall_ns": max(int(row[4]) for row in anchor_rows),
        "per_symbol": {symbol: dict(values) for symbol, values in per_symbol.items()},
        "source_max_receipt_precedes_every_anchor": all(
            int(row[5]) < int(row[3]) for row in anchor_rows
        ),
        "crypto_formal_daily_close": False,
        "target_constructed": False,
        "model_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
    }


def _report_from_identity(
    identity: Mapping[str, object],
    manifest_sha256: str,
) -> Round73GridBuildReport:
    return Round73GridBuildReport(
        run_id=str(identity["run_id"]),
        source_corpus_manifest_sha256=str(identity["source_corpus_manifest_sha256"]),
        anchor_count=int(identity["anchor_count"]),
        valid_anchor_count=int(identity["valid_anchor_count"]),
        vector_count=int(identity["vector_count"]),
        first_anchor_wall_ns=int(identity["first_anchor_wall_ns"]),
        last_anchor_wall_ns=int(identity["last_anchor_wall_ns"]),
        per_symbol={
            str(symbol): {str(key): int(value) for key, value in values.items()}
            for symbol, values in dict(identity["per_symbol"]).items()
        },
        build_manifest_sha256=manifest_sha256,
    )


def build_round73_causal_grid(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    corpus_audit_function: CorpusAuditFunction = audit_round73_corpus_manifest,
) -> Round73GridBuildReport:
    """Build and atomically publish one admitted run's target-free feature grid."""

    selected = _validated_run_id(run_id)
    existing: tuple[object, ...] | None = None
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        if _table_exists(connection, ROUND73_GRID_MANIFEST_TABLE):
            existing = connection.execute(
                f"SELECT build_manifest_json, build_manifest_sha256 "
                f"FROM {ROUND73_GRID_MANIFEST_TABLE} WHERE run_id = ?",
                [selected],
            ).fetchone()
    if existing is not None:
        audit = audit_round73_causal_grid(
            database,
            run_id=selected,
            memory_limit=memory_limit,
            threads=threads,
            corpus_audit_function=corpus_audit_function,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing grid build audit failed")
        identity = _strict_json_object(str(existing[0]), "grid build manifest")
        return _report_from_identity(identity, str(existing[1]))
    source_audit = corpus_audit_function(
        database,
        run_id=selected,
        memory_limit=memory_limit,
        threads=threads,
    )
    if not source_audit.passed:
        raise ValueError("Round 73 grid source corpus manifest audit failed")
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        source = connection.execute(
            f"""
            SELECT c.manifest_sha256, c.coverage_start_wall_ns,
                   c.coverage_end_wall_ns, r.schema_version,
                   r.capture_contract_sha256, r.started_wall_ns,
                   r.started_monotonic_ns, c.feature_ready_wall_ns
            FROM {ROUND73_CORPUS_RUN_TABLE} c
            JOIN impact_capture_run r ON r.run_id = c.run_id
            WHERE c.run_id = ?
            """,
            [selected],
        ).fetchone()
        if source is None:
            raise ValueError("Round 73 grid source corpus manifest is missing")
        source_manifest_sha256 = str(source[0])
        source_schema_version = str(source[3])
        admissible_capture_contracts = {
            IMPACT_CAPTURE_SCHEMA_VERSION: IMPACT_CAPTURE_CONTRACT_SHA256,
            IMPACT_CAPTURE_V9_SCHEMA_VERSION: IMPACT_CAPTURE_V9_CONTRACT_SHA256,
        }
        if (
            source_manifest_sha256 != source_audit.manifest_sha256
            or _SHA256.fullmatch(source_manifest_sha256) is None
            or admissible_capture_contracts.get(source_schema_version) != str(source[4])
        ):
            raise ValueError("Round 73 grid source identity differs")
        coverage_start_wall_ns = int(source[1])
        coverage_end_wall_ns = int(source[2])
        feature_ready_wall_ns = int(source[7])
        if coverage_start_wall_ns < feature_ready_wall_ns:
            raise ValueError("Round 73 grid source precedes feature-ready marker")
        run_started_wall_ns = int(source[5])
        run_started_monotonic_ns = int(source[6])
        segments = connection.execute(
            """
            SELECT symbol, status, clock_offset_ns, tick_size
            FROM impact_capture_segment WHERE run_id = ? ORDER BY symbol
            """,
            [selected],
        ).fetchall()
        if tuple(str(row[0]) for row in segments) != IMPACT_CAPTURE_SYMBOLS or any(
            str(row[1]) != "valid" for row in segments
        ):
            raise ValueError("Round 73 grid symbol segments are invalid")
        anchor_rows: list[tuple[object, ...]] = []
        vector_rows: list[tuple[object, ...]] = []
        per_symbol: dict[str, dict[str, int]] = {}
        if source_schema_version == IMPACT_CAPTURE_V9_SCHEMA_VERSION:
            anchor_rows, vector_rows, per_symbol = _v9_grid_rows(
                connection,
                run_id=selected,
                segments=segments,
                run_started_wall_ns=run_started_wall_ns,
                run_started_monotonic_ns=run_started_monotonic_ns,
                coverage_start_wall_ns=coverage_start_wall_ns,
                coverage_end_wall_ns=coverage_end_wall_ns,
            )
        else:
            clock_timeline = _causal_clock_timeline(connection, run_id=selected)
            for symbol, _status, clock_offset_ns, _tick_size in segments:
                symbol_anchors, symbol_vectors, diagnostic = _symbol_grid_rows(
                    connection,
                    run_id=selected,
                    symbol=str(symbol),
                    clock_timeline=clock_timeline,
                    fallback_clock_offset_ns=int(clock_offset_ns),
                    run_started_wall_ns=run_started_wall_ns,
                    run_started_monotonic_ns=run_started_monotonic_ns,
                    coverage_start_wall_ns=coverage_start_wall_ns,
                    coverage_end_wall_ns=coverage_end_wall_ns,
                )
                anchor_rows.extend(symbol_anchors)
                vector_rows.extend(symbol_vectors)
                per_symbol[str(symbol)] = diagnostic
    anchor_counts = {values["anchors"] for values in per_symbol.values()}
    if len(anchor_counts) != 1 or not anchor_counts or min(anchor_counts) < 1:
        raise ValueError("Round 73 grid symbols do not share a nonempty anchor grid")
    identity = _build_identity(
        run_id=selected,
        source_manifest_sha256=source_manifest_sha256,
        anchor_rows=anchor_rows,
        vector_rows=vector_rows,
        per_symbol=per_symbol,
    )
    if identity["source_max_receipt_precedes_every_anchor"] is not True:
        raise ValueError("Round 73 grid contains a future receipt")
    feature_names_text = _canonical_json(list(ROUND73_GRID_FEATURE_NAMES))
    if _sha256_text(feature_names_text) != ROUND73_GRID_FEATURE_NAMES_SHA256:
        raise RuntimeError("Round 73 grid feature-name hash differs")
    manifest_text = _canonical_json(identity)
    manifest_sha256 = _sha256_text(manifest_text)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_grid_tables(connection)
        _assert_grid_table_shapes(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            concurrent = connection.execute(
                f"SELECT build_manifest_sha256 FROM {ROUND73_GRID_MANIFEST_TABLE} "
                "WHERE run_id = ?",
                [selected],
            ).fetchone()
            if concurrent is not None:
                if str(concurrent[0]) != manifest_sha256:
                    raise ValueError("Round 73 concurrent grid build differs")
            else:
                _insert_rows_columnar(
                    connection,
                    ROUND73_GRID_ANCHOR_TABLE,
                    anchor_rows,
                )
                _insert_rows_columnar(
                    connection,
                    ROUND73_GRID_VECTOR_TABLE,
                    vector_rows,
                )
                connection.execute(
                    f"INSERT INTO {ROUND73_GRID_MANIFEST_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected,
                        ROUND73_GRID_SCHEMA_VERSION,
                        ROUND73_GRID_CONTRACT_SHA256,
                        source_manifest_sha256,
                        feature_names_text,
                        ROUND73_GRID_FEATURE_NAMES_SHA256,
                        manifest_text,
                        manifest_sha256,
                        int(identity["anchor_count"]),
                        int(identity["valid_anchor_count"]),
                        int(identity["vector_count"]),
                        int(identity["first_anchor_wall_ns"]),
                        int(identity["last_anchor_wall_ns"]),
                        time.time_ns(),
                    ],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return _report_from_identity(identity, manifest_sha256)


def audit_round73_causal_grid(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
    corpus_audit_function: CorpusAuditFunction = audit_round73_corpus_manifest,
) -> Round73GridBuildAudit:
    """Reconcile one target-free grid with its source manifest and row hashes."""

    selected = _validated_run_id(run_id)
    errors: list[str] = []
    anchor_count = 0
    valid_anchor_count = 0
    vector_count = 0
    manifest_sha256 = ""
    source_manifest_sha256 = ""
    try:
        source_audit = corpus_audit_function(
            database,
            run_id=selected,
            memory_limit=memory_limit,
            threads=threads,
        )
        if not source_audit.passed:
            errors.append("source_corpus_manifest_audit_failed")
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"source:{type(exc).__name__}:{exc}")
        source_audit = None
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        if not _table_exists(connection, ROUND73_GRID_MANIFEST_TABLE):
            raise ValueError("Round 73 grid manifest table was not found")
        if not all(
            _table_exists(connection, table)
            for table in (
                ROUND73_GRID_ANCHOR_TABLE,
                ROUND73_GRID_VECTOR_TABLE,
                ROUND73_GRID_MANIFEST_TABLE,
            )
        ):
            raise ValueError("Round 73 grid table set is incomplete")
        _assert_grid_table_shapes(connection)
        row = connection.execute(
            f"""
            SELECT schema_version, contract_sha256,
                   source_corpus_manifest_sha256, feature_names_json,
                   feature_names_sha256, build_manifest_json,
                   build_manifest_sha256, anchor_count,
                   valid_anchor_count, vector_count,
                   first_anchor_wall_ns, last_anchor_wall_ns
            FROM {ROUND73_GRID_MANIFEST_TABLE} WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if row is None:
            raise ValueError("Round 73 grid build manifest was not found")
        source_manifest_sha256 = str(row[2])
        manifest_sha256 = str(row[6])
        try:
            feature_names_text = str(row[3])
            manifest_text = str(row[5])
            if (
                str(row[0]) != ROUND73_GRID_SCHEMA_VERSION
                or str(row[1]) != ROUND73_GRID_CONTRACT_SHA256
                or source_audit is None
                or source_manifest_sha256 != source_audit.manifest_sha256
                or _sha256_text(feature_names_text) != str(row[4])
                or str(row[4]) != ROUND73_GRID_FEATURE_NAMES_SHA256
                or tuple(json.loads(feature_names_text)) != ROUND73_GRID_FEATURE_NAMES
                or _sha256_text(manifest_text) != manifest_sha256
            ):
                raise ValueError("grid build manifest identity differs")
            identity = _strict_json_object(manifest_text, "grid build manifest")
            if (
                identity.get("schema_version") != ROUND73_GRID_SCHEMA_VERSION
                or identity.get("contract_sha256") != ROUND73_GRID_CONTRACT_SHA256
                or identity.get("run_id") != selected
                or identity.get("source_corpus_manifest_sha256")
                != source_manifest_sha256
                or identity.get("feature_names_sha256")
                != ROUND73_GRID_FEATURE_NAMES_SHA256
                or identity.get("feature_count") != len(ROUND73_GRID_FEATURE_NAMES)
                or identity.get("source_max_receipt_precedes_every_anchor") is not True
                or identity.get("crypto_formal_daily_close") is not False
                or identity.get("target_constructed") is not False
                or identity.get("model_evaluated") is not False
                or identity.get("profitability_claim") is not False
                or identity.get("trading_authority") is not False
            ):
                raise ValueError("grid build manifest fields differ")
            anchors = connection.execute(
                f"""
                SELECT run_id, symbol, anchor_index, anchor_monotonic_ns,
                       anchor_wall_ns, source_max_received_monotonic_ns, valid,
                       invalid_reason_mask, invalid_reasons_json,
                       signed_aggressive_quote_1s,
                       absolute_aggressive_quote_1s,
                       trailing_median_absolute_aggressive_quote_60s,
                       shock_ratio, shock_direction,
                       shock_direction_taker_share
                FROM {ROUND73_GRID_ANCHOR_TABLE}
                WHERE run_id = ? ORDER BY symbol, anchor_index
                """,
                [selected],
            ).fetchall()
            vectors = connection.execute(
                f"""
                SELECT run_id, symbol, anchor_index, feature_values, vector_sha256
                FROM {ROUND73_GRID_VECTOR_TABLE}
                WHERE run_id = ? ORDER BY symbol, anchor_index
                """,
                [selected],
            ).fetchall()
            anchor_count = len(anchors)
            valid_anchor_count = sum(bool(item[6]) for item in anchors)
            vector_count = len(vectors)
            if any(int(item[5]) >= int(item[3]) for item in anchors):
                raise ValueError("grid anchor uses a future receipt")
            for item in anchors:
                invalid_mask = int(item[7])
                invalid_reasons = json.loads(str(item[8]))
                optional_values = (item[11], item[12])
                signed_quote = float(item[9])
                absolute_quote = float(item[10])
                expected_direction = (
                    1 if signed_quote > 0 else -1 if signed_quote < 0 else 0
                )
                if (
                    bool(item[6]) != (invalid_mask == 0)
                    or not isinstance(invalid_reasons, list)
                    or tuple(invalid_reasons)
                    != round73_grid_invalid_reasons(invalid_mask)
                    or not all(math.isfinite(float(value)) for value in item[9:11])
                    or absolute_quote < 0
                    or abs(signed_quote)
                    > absolute_quote + max(1e-9, absolute_quote * 1e-12)
                    or any(
                        value is not None and not math.isfinite(float(value))
                        for value in optional_values
                    )
                    or any(
                        value is not None and float(value) < 0
                        for value in optional_values
                    )
                    or int(item[13]) != expected_direction
                    or not math.isfinite(float(item[14]))
                    or not 0 <= float(item[14]) <= 1
                ):
                    raise ValueError("grid anchor validity fields differ")
            grouped: dict[str, list[tuple[int, int, int, int]]] = {
                symbol: [] for symbol in IMPACT_CAPTURE_SYMBOLS
            }
            valid_keys = set()
            for item in anchors:
                symbol = str(item[1])
                anchor_index = int(item[2])
                grouped.setdefault(symbol, []).append(
                    (anchor_index, int(item[3]), int(item[4]), int(item[5]))
                )
                if bool(item[6]):
                    valid_keys.add((symbol, anchor_index))
            if set(grouped) != set(IMPACT_CAPTURE_SYMBOLS):
                raise ValueError("grid contains an unsupported symbol")
            reference_grid: tuple[tuple[int, int, int], ...] | None = None
            for entries in grouped.values():
                if not entries or [item[0] for item in entries] != list(
                    range(len(entries))
                ):
                    raise ValueError("grid anchor indices are not contiguous")
                if any(
                    following[1] - previous[1] != ROUND73_GRID_STEP_NS
                    or following[2] - previous[2] != ROUND73_GRID_STEP_NS
                    or following[3] < previous[3]
                    for previous, following in zip(entries, entries[1:], strict=False)
                ):
                    raise ValueError(
                        "grid anchor clocks or source receipts are not monotone"
                    )
                observed_grid = tuple((item[0], item[1], item[2]) for item in entries)
                if reference_grid is None:
                    reference_grid = observed_grid
                elif observed_grid != reference_grid:
                    raise ValueError("grid symbols do not share identical anchors")
            vector_keys = set()
            for _run_id, symbol, anchor_index, raw_values, claimed_sha256 in vectors:
                values = tuple(float(value) for value in raw_values)
                key = (str(symbol), int(anchor_index))
                vector_keys.add(key)
                if (
                    len(values) != len(ROUND73_GRID_FEATURE_NAMES)
                    or not all(math.isfinite(value) for value in values)
                    or _vector_sha256(selected, key[0], key[1], values)
                    != str(claimed_sha256)
                ):
                    raise ValueError("grid feature vector hash or width differs")
            if vector_keys != valid_keys:
                raise ValueError("grid valid anchors and vectors differ")
            if identity.get("anchor_rows_sha256") != _anchor_rows_sha256(
                anchors
            ) or identity.get("vector_rows_sha256") != _vector_rows_sha256(vectors):
                raise ValueError("grid persisted-row aggregate hash differs")
            if (
                anchor_count != int(row[7])
                or valid_anchor_count != int(row[8])
                or vector_count != int(row[9])
                or anchor_count != identity.get("anchor_count")
                or valid_anchor_count != identity.get("valid_anchor_count")
                or vector_count != identity.get("vector_count")
            ):
                raise ValueError("grid aggregate counts differ")
            observed_per_symbol = {
                symbol: {
                    "anchors": len(entries),
                    "valid_anchors": sum(key[0] == symbol for key in valid_keys),
                    "invalid_anchors": len(entries)
                    - sum(key[0] == symbol for key in valid_keys),
                }
                for symbol, entries in grouped.items()
            }
            if identity.get("per_symbol") != observed_per_symbol:
                raise ValueError("grid per-symbol counts differ")
            wall_bounds = connection.execute(
                f"SELECT min(anchor_wall_ns), max(anchor_wall_ns) "
                f"FROM {ROUND73_GRID_ANCHOR_TABLE} WHERE run_id = ?",
                [selected],
            ).fetchone()
            if (
                int(wall_bounds[0]) != int(row[10])
                or int(wall_bounds[1]) != int(row[11])
                or int(row[10]) != identity.get("first_anchor_wall_ns")
                or int(row[11]) != identity.get("last_anchor_wall_ns")
            ):
                raise ValueError("grid wall-clock bounds differ")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"grid:{type(exc).__name__}:{exc}")
    return Round73GridBuildAudit(
        run_id=selected,
        passed=not errors,
        errors=tuple(errors),
        anchor_count=anchor_count,
        valid_anchor_count=valid_anchor_count,
        vector_count=vector_count,
        build_manifest_sha256=manifest_sha256,
        source_corpus_manifest_sha256=source_manifest_sha256,
    )


__all__ = [
    "ROUND73_GRID_ANCHOR_TABLE",
    "ROUND73_GRID_MANIFEST_TABLE",
    "ROUND73_GRID_VECTOR_TABLE",
    "Round73GridBuildAudit",
    "Round73GridBuildReport",
    "audit_round73_causal_grid",
    "build_round73_causal_grid",
]
