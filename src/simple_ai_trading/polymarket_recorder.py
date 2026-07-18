"""Prospective public-data recorder for Polymarket BTC/ETH/SOL five-minute paper trading."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import struct
import time
from typing import Mapping, Protocol, Sequence
import uuid

import duckdb
import zstandard
from websockets.asyncio.client import connect

from .paper_execution import PaperOrderJournal
from .polymarket import (
    POLYMARKET_MARKET_SCHEMA_VERSION,
    PolymarketFiveMinuteMarket,
    PolymarketPublicClient,
    validate_clob_market_info,
    validate_clob_order_book,
)
from .polymarket_capture_frame import (
    CAPTURE_FRAME_FORMAT,
    CAPTURE_FRAME_MAGIC,
    CaptureFrameRecord,
    LocatedCaptureFrameRecord,
    capture_frame_record_size,
    decode_capture_frame,
    encode_capture_frame,
)


POLYMARKET_EVIDENCE_SCHEMA_VERSION = "polymarket-public-evidence-v1"
POLYMARKET_RECORDER_SCHEMA_VERSION = "polymarket-public-recorder-v1"
POLYMARKET_RECORDER_PROGRESS_SCHEMA_VERSION = "polymarket-recorder-progress-v1"
POLYMARKET_STORAGE_SCHEMA_VERSION = "polymarket-evidence-storage-v4"
POLYMARKET_CAPTURE_MANIFEST_SCHEMA_VERSION = "polymarket-capture-manifest-v1"
POLYMARKET_TERMINAL_RECOVERY_SCHEMA_VERSION = (
    "polymarket-terminal-audit-recovery-v1"
)
_RAW_RECONSTRUCTED_STORAGE_SCHEMA_VERSION = "polymarket-evidence-storage-v3"
_INDEXED_COMPACT_STORAGE_SCHEMA_VERSION = "polymarket-evidence-storage-v2"
_LEGACY_STORAGE_SCHEMA_VERSION = "polymarket-public-evidence-v1"
_COMPACT_STORAGE_SCHEMA_VERSIONS = frozenset(
    {
        _INDEXED_COMPACT_STORAGE_SCHEMA_VERSION,
        _RAW_RECONSTRUCTED_STORAGE_SCHEMA_VERSION,
    }
)
_CHUNK_STORAGE_SCHEMA_VERSIONS = frozenset(
    {*_COMPACT_STORAGE_SCHEMA_VERSIONS, POLYMARKET_STORAGE_SCHEMA_VERSION}
)
CLOB_MARKET_WEBSOCKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_RTDS_WEBSOCKET = "wss://ws-live-data.polymarket.com"
BINANCE_SPOT_WEBSOCKET = (
    "wss://stream.binance.com:9443/stream?streams="
    "btcusdt@bookTicker/btcusdt@trade/"
    "ethusdt@bookTicker/ethusdt@trade/"
    "solusdt@bookTicker/solusdt@trade"
)
_STREAMS = frozenset(
    {"clob_market", "polymarket_rtds", "binance_spot", "clob_rest_book"}
)
_MAX_RAW_MESSAGE_BYTES = 8 * 1024 * 1024
_MAX_RAW_CHUNK_BYTES = 64 * 1024 * 1024
_CLOB_MAX_MESSAGE_BYTES = 512 * 1024
_SIMPLE_STREAM_MAX_MESSAGE_BYTES = 64 * 1024
_CLOB_MAX_QUEUE_FRAMES = 2_048
_SIMPLE_STREAM_MAX_QUEUE_FRAMES = 1_024
_OUTPUT_PUT_TIMEOUT_SECONDS = 5.0
_STREAM_INACTIVITY_SECONDS = 30.0
_STABLE_CONNECTION_SECONDS = 30.0
_RAW_CHUNK_FRAME_FORMAT = "length-prefixed-utf8-v1"
_RAW_CHUNK_CODEC = "zstd"
_RAW_CHUNK_COMPRESSION_LEVEL = 1
_DUCKDB_MEMORY_LIMIT = re.compile(r"[1-9][0-9]*(?:KB|MB|GB|TB)", re.IGNORECASE)
_RAW_CHUNK_MESSAGE_LIMIT = 1_024
_CONDITION_CACHE_SCHEMA_VERSION = "polymarket-condition-message-cache-v1"
_CONDITION_CACHE_FRAME_MESSAGE_LIMIT = 512
_CONDITION_CACHE_MAX_FRAME_BYTES = 16 * 1024 * 1024
_MAX_AI_CACHE_LATENCY_SECONDS = 600.0
_WRITER_BATCH_SIZE = 8_192
_WRITER_COALESCE_SECONDS = 0.5
_WRITER_MIN_DRAIN_SECONDS = 60.0
_WRITER_MAX_DRAIN_SECONDS = 600.0
_WRITER_STALL_SECONDS = 30.0
_WRITER_ASSUMED_MINIMUM_MESSAGES_PER_SECOND = 100.0
_INTEGRITY_FETCH_SIZE = 4_096
_CAPTURE_AUDIT_CHUNK_PAGE_SIZE = 32
_TERMINAL_AUDIT_RESOURCE_EXHAUSTED_PREFIX = (
    "finish_run:OutOfMemoryException:Out of Memory Error:"
)
_TERMINAL_AUDIT_RECOVERY_REASON = "terminal_integrity_audit_resource_exhausted"

_LEGACY_RAW_MESSAGE_INSERT_SQL = """
    INSERT INTO polymarket_raw_message (
        message_id, run_id, schema_version, stream, connection_id,
        sequence_number, received_wall_ms, received_monotonic_ns,
        raw_payload_sha256, raw_text, parse_status, parse_error
    )
    SELECT unnest(?), unnest(?), unnest(?), unnest(?), unnest(?), unnest(?),
           unnest(?), unnest(?), unnest(?), unnest(?), unnest(?), unnest(?)
"""
_COMPACT_RAW_MESSAGE_INSERT_SQL = """
    INSERT INTO polymarket_raw_message (
        message_id, run_id, schema_version, stream, connection_id,
        sequence_number, received_wall_ms, received_monotonic_ns,
        raw_payload_sha256, raw_text, parse_status, parse_error,
        storage_chunk_id, chunk_message_index, raw_offset, raw_size,
        normalized_event_count
    )
    SELECT unnest(?), unnest(?), unnest(?), unnest(?), unnest(?), unnest(?),
           unnest(?), unnest(?), unnest(?), unnest(?), unnest(?), unnest(?),
           unnest(?), unnest(?), unnest(?), unnest(?), unnest(?)
"""
_PUBLIC_EVENT_INSERT_SQL = """
    INSERT INTO polymarket_public_event (
        event_id, run_id, message_id, sub_index, stream, event_type, symbol,
        condition_id, asset_id, source_time_ms, publisher_time_ms, event_json,
        event_sha256
    )
    SELECT unnest(?), unnest(?), unnest(?), unnest(?), unnest(?), unnest(?),
           unnest(?), unnest(?), unnest(?), unnest(?), unnest(?), unnest(?),
           unnest(?)
"""


class _TextSender(Protocol):
    async def send(self, message: str) -> object: ...


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _columnar_batch(
    rows: Sequence[tuple[object, ...]],
    *,
    width: int,
) -> tuple[list[object], ...]:
    if not rows or any(len(row) != width for row in rows):
        raise ValueError("DuckDB evidence batch has an invalid row width")
    return tuple(list(column) for column in zip(*rows, strict=True))


def _query_batches(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: Sequence[object],
) -> Iterator[list[tuple[object, ...]]]:
    cursor = connection.execute(query, parameters)
    while rows := cursor.fetchmany(_INTEGRITY_FETCH_SIZE):
        yield rows


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _compact_message_manifest_hash(row: Sequence[object]) -> str:
    if len(row) != 17:
        raise ValueError("compact raw-message manifest row has an invalid width")
    return _canonical_sha256(
        {
            "message_id": str(row[0]),
            "run_id": str(row[1]),
            "schema_version": str(row[2]),
            "stream": str(row[3]),
            "connection_id": str(row[4]),
            "sequence_number": int(row[5]),
            "received_wall_ms": int(row[6]),
            "received_monotonic_ns": int(row[7]),
            "raw_payload_sha256": str(row[8]),
            "parse_status": str(row[10]),
            "parse_error": str(row[11]),
            "chunk_message_index": int(row[13]),
            "raw_offset": int(row[14]),
            "raw_size": int(row[15]),
            "normalized_event_count": int(row[16]),
        }
    )


def _capture_message_manifest_hash(
    *,
    run_id: str,
    message_id: str,
    raw_payload_sha256: str,
    located: LocatedCaptureFrameRecord,
) -> str:
    message = located.record
    return _canonical_sha256(
        {
            "run_id": run_id,
            "schema_version": POLYMARKET_STORAGE_SCHEMA_VERSION,
            "message_id": message_id,
            "message_index": located.message_index,
            "stream": message.stream,
            "connection_id": message.connection_id,
            "sequence_number": message.sequence_number,
            "received_wall_ms": message.received_wall_ms,
            "received_monotonic_ns": message.received_monotonic_ns,
            "raw_payload_sha256": raw_payload_sha256,
            "raw_offset": located.raw_offset,
            "raw_size": located.raw_size,
        }
    )


def _public_event_manifest_hash(
    *,
    event_id: object,
    run_id: object,
    message_id: object,
    sub_index: object,
    stream: object,
    event_type: object,
    symbol: object,
    condition_id: object,
    asset_id: object,
    source_time_ms: object,
    publisher_time_ms: object,
    event_sha256: object,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "polymarket-public-event-manifest-v1",
            "event_id": str(event_id),
            "run_id": str(run_id),
            "message_id": str(message_id),
            "sub_index": int(sub_index),
            "stream": str(stream),
            "event_type": str(event_type),
            "symbol": str(symbol),
            "condition_id": str(condition_id),
            "asset_id": str(asset_id),
            "source_time_ms": None if source_time_ms is None else int(source_time_ms),
            "publisher_time_ms": (
                None if publisher_time_ms is None else int(publisher_time_ms)
            ),
            "event_sha256": str(event_sha256),
        }
    )


def _wall_ms() -> int:
    return time.time_ns() // 1_000_000


def _monotonic_ns() -> int:
    return time.monotonic_ns()


def _capture_order_key(message: RawStreamMessage) -> tuple[int, int, str, int, str]:
    return (
        int(message.received_monotonic_ns),
        int(message.received_wall_ms),
        str(message.connection_id),
        int(message.sequence_number),
        str(message.stream),
    )


def _capture_receipt_key(message: RawStreamMessage) -> tuple[int, int]:
    """Return the causal receipt clocks without arbitrary tie breakers."""

    return (
        int(message.received_monotonic_ns),
        int(message.received_wall_ms),
    )


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _strict_json_loads(value: str) -> object:
    def reject_constant(constant: str) -> object:
        raise ValueError(f"non-finite JSON constant: {constant}")

    return json.loads(value, parse_constant=reject_constant)


def _validated_canonical_payload(
    name: str,
    payload_json: str,
    claimed_sha256: str,
) -> object:
    parsed = _strict_json_loads(str(payload_json))
    canonical = _canonical_json(parsed)
    if canonical != payload_json:
        raise ValueError(f"{name} payload is not canonical JSON")
    actual_sha256 = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    if not hmac.compare_digest(actual_sha256, str(claimed_sha256)):
        raise ValueError(f"{name} payload hash mismatch")
    return parsed


def _stream_message_events(
    raw_text: str,
) -> tuple[str, str, tuple[Mapping[str, object], ...]]:
    try:
        parsed = _strict_json_loads(raw_text)
        candidates = parsed if isinstance(parsed, list) else [parsed]
        if not all(isinstance(item, Mapping) for item in candidates):
            raise ValueError("JSON stream message contains a non-object event")
        return (
            "ok",
            "",
            tuple(item for item in candidates if isinstance(item, Mapping)),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        if raw_text.strip() in {"", "PING", "PONG"}:
            return "control", "", ()
        return "invalid", f"{exc.__class__.__name__}:{exc}", ()


@dataclass(frozen=True)
class RawStreamMessage:
    stream: str
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    raw_text: str

    def validated(self) -> "RawStreamMessage":
        stream = str(self.stream or "").strip()
        if stream not in _STREAMS:
            raise ValueError(f"unsupported public stream: {stream}")
        connection_id = str(self.connection_id or "").strip()
        if not connection_id or len(connection_id) > 160:
            raise ValueError("connection_id is invalid")
        sequence = int(self.sequence_number)
        wall = int(self.received_wall_ms)
        monotonic = int(self.received_monotonic_ns)
        if min(sequence, wall, monotonic) < 0:
            raise ValueError("stream sequence and timestamps must be non-negative")
        raw_text = str(self.raw_text)
        if len(raw_text.encode("utf-8")) > _MAX_RAW_MESSAGE_BYTES:
            raise ValueError("public stream message exceeded the bounded size")
        return RawStreamMessage(
            stream=stream,
            connection_id=connection_id,
            sequence_number=sequence,
            received_wall_ms=wall,
            received_monotonic_ns=monotonic,
            raw_text=raw_text,
        )


@dataclass(frozen=True)
class DecodedPublicEvent:
    event_id: str
    run_id: str
    message_id: str
    sub_index: int
    stream: str
    event_type: str
    symbol: str
    condition_id: str
    asset_id: str
    source_time_ms: int | None
    publisher_time_ms: int | None
    event_sha256: str
    event: Mapping[str, object]
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int


@dataclass(frozen=True)
class StreamGap:
    stream: str
    connection_id: str
    opened_at_ms: int
    reason: str
    last_sequence_number: int

    def validated(self) -> "StreamGap":
        stream = str(self.stream or "").strip()
        if stream not in _STREAMS:
            raise ValueError(f"unsupported public stream: {stream}")
        connection_id = str(self.connection_id or "").strip()
        if not connection_id or len(connection_id) > 160:
            raise ValueError("connection_id is invalid")
        opened_at_ms = int(self.opened_at_ms)
        sequence = int(self.last_sequence_number)
        if min(opened_at_ms, sequence) < 0:
            raise ValueError("gap timestamp and sequence must be non-negative")
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("gap reason is required")
        return StreamGap(
            stream=stream,
            connection_id=connection_id,
            opened_at_ms=opened_at_ms,
            reason=reason[:2_000],
            last_sequence_number=sequence,
        )


@dataclass(frozen=True)
class MarketEvidence:
    market: PolymarketFiveMinuteMarket
    observed_wall_ms: int
    observed_monotonic_ns: int
    clob_info_json: str
    clob_info_sha256: str
    up_fee_rate_json: str
    up_fee_rate_sha256: str
    down_fee_rate_json: str
    down_fee_rate_sha256: str
    maker_base_fee: int
    taker_base_fee: int
    taker_order_delay_enabled: bool
    minimum_order_age_seconds: int


@dataclass(frozen=True)
class RecorderReport:
    schema_version: str
    run_id: str
    status: str
    database: str
    started_at_ms: int
    ended_at_ms: int
    duration_seconds: float
    market_snapshot_count: int
    raw_message_count: int
    normalized_event_count: int
    stream_gap_count: int
    stream_counts: dict[str, int]
    assets: tuple[str, ...]
    conditions: tuple[str, ...]
    integrity_errors: tuple[str, ...]
    errors: tuple[str, ...]
    evidence_manifest_sha256: str
    report_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalAuditRecoveryReport:
    schema_version: str
    run_id: str
    recovered_at_ms: int
    recovery_reason: str
    prior_report_sha256: str
    recovered_report_sha256: str
    pre_recovery_evidence_fingerprint: str
    raw_message_count: int
    normalized_event_count: int
    memory_limit: str
    database_threads: int
    labels_consulted: bool
    outcomes_consulted: bool
    model_scores_consulted: bool
    profitability_claim: bool
    trading_authority: bool
    training_authority: bool
    recovery_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RawMessageLaneSummary:
    stream: str
    connection_id: str
    message_count: int
    minimum_sequence_number: int
    maximum_sequence_number: int
    first_received_wall_ms: int
    last_received_wall_ms: int
    first_received_monotonic_ns: int
    last_received_monotonic_ns: int


@dataclass
class _RawMessageLaneAccumulator:
    message_count: int
    minimum_sequence_number: int
    maximum_sequence_number: int
    first_received_wall_ms: int
    last_received_wall_ms: int
    first_received_monotonic_ns: int
    last_received_monotonic_ns: int

    @classmethod
    def from_message(cls, message: RawStreamMessage) -> _RawMessageLaneAccumulator:
        return cls(
            message_count=1,
            minimum_sequence_number=message.sequence_number,
            maximum_sequence_number=message.sequence_number,
            first_received_wall_ms=message.received_wall_ms,
            last_received_wall_ms=message.received_wall_ms,
            first_received_monotonic_ns=message.received_monotonic_ns,
            last_received_monotonic_ns=message.received_monotonic_ns,
        )

    def update(self, message: RawStreamMessage) -> None:
        self.message_count += 1
        self.minimum_sequence_number = min(
            self.minimum_sequence_number,
            message.sequence_number,
        )
        self.maximum_sequence_number = max(
            self.maximum_sequence_number,
            message.sequence_number,
        )
        self.first_received_wall_ms = min(
            self.first_received_wall_ms,
            message.received_wall_ms,
        )
        self.last_received_wall_ms = max(
            self.last_received_wall_ms,
            message.received_wall_ms,
        )
        self.first_received_monotonic_ns = min(
            self.first_received_monotonic_ns,
            message.received_monotonic_ns,
        )
        self.last_received_monotonic_ns = max(
            self.last_received_monotonic_ns,
            message.received_monotonic_ns,
        )

    def freeze(self, stream: str, connection_id: str) -> RawMessageLaneSummary:
        return RawMessageLaneSummary(
            stream=stream,
            connection_id=connection_id,
            message_count=self.message_count,
            minimum_sequence_number=self.minimum_sequence_number,
            maximum_sequence_number=self.maximum_sequence_number,
            first_received_wall_ms=self.first_received_wall_ms,
            last_received_wall_ms=self.last_received_wall_ms,
            first_received_monotonic_ns=self.first_received_monotonic_ns,
            last_received_monotonic_ns=self.last_received_monotonic_ns,
        )


def _freeze_lane_summaries(
    accumulators: Mapping[tuple[str, str], _RawMessageLaneAccumulator],
) -> tuple[RawMessageLaneSummary, ...]:
    return tuple(accumulators[key].freeze(*key) for key in sorted(accumulators))


@dataclass(frozen=True)
class _StoredCaptureMessage:
    message_id: str
    raw_payload_sha256: str
    storage_chunk_id: str
    raw_offset: int
    raw_size: int
    message: RawStreamMessage

    def cache_metadata(self, normalized_event_count: int) -> tuple[object, ...]:
        return (
            self.message_id,
            self.message.stream,
            self.message.connection_id,
            self.message.sequence_number,
            self.message.received_wall_ms,
            self.message.received_monotonic_ns,
            self.raw_payload_sha256,
            self.storage_chunk_id,
            self.raw_offset,
            self.raw_size,
            int(normalized_event_count),
        )


@dataclass(frozen=True)
class _EvidenceAuditSummary:
    raw_message_count: int
    normalized_event_count: int
    stream_counts: dict[str, int]
    out_of_window_message_count: int = 0
    lane_summaries: tuple[RawMessageLaneSummary, ...] = ()


class PolymarketEvidenceStore:
    """Single-writer compressed DuckDB store for public evidence and paper state."""

    def __init__(
        self,
        path: str | Path,
        *,
        memory_limit: str = "1GB",
        threads: int = 2,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.memory_limit = str(memory_limit).upper()
        if not _DUCKDB_MEMORY_LIMIT.fullmatch(self.memory_limit):
            raise ValueError("memory_limit must be a positive KB, MB, GB, or TB value")
        self.threads = int(threads)
        if self.threads < 1 or self.threads > 8:
            raise ValueError("threads must lie in [1, 8]")
        if not isinstance(read_only, bool):
            raise ValueError("read_only must be a boolean")
        self.read_only = read_only
        self.connection: duckdb.DuckDBPyConnection | None = None
        self.payload_connection: duckdb.DuckDBPyConnection | None = None
        self.paper_journal: PaperOrderJournal | None = None
        # Terminal recorder tables are immutable through this API. Cache only
        # their full row-hash audit; mutable paper-journal checks always rerun.
        self._terminal_evidence_integrity_cache: dict[
            str, tuple[str, tuple[str, ...]]
        ] = {}
        self._evidence_audit_summary: dict[str, _EvidenceAuditSummary] = {}
        self._storage_schema_version_by_run: dict[str, str] = {}
        self._next_chunk_index_by_run: dict[str, int] = {}
        self._last_sequence_by_lane: dict[tuple[str, str, str], int] = {}
        self._last_capture_order_key_by_run: dict[str, tuple[int, int]] = {}
        self._sequence_state_initialized_runs: set[str] = set()
        self._frame_cache_id = ""
        self._frame_cache = b""
        self._validated_condition_cache_runs: dict[str, str] = {}
        self._decompressor = zstandard.ZstdDecompressor(
            max_window_size=_MAX_RAW_CHUNK_BYTES // 1024
        )

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self.connection is None:
            if self.read_only:
                if not self.path.is_file():
                    raise ValueError(
                        "read-only Polymarket evidence database does not exist"
                    )
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(
                str(self.path),
                read_only=self.read_only,
            )
            self.connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self.connection.execute(f"SET threads={self.threads}")
            self.connection.execute("SET TimeZone='UTC'")
            self.connection.execute("SET preserve_insertion_order=false")
            if not self.read_only:
                self.connection.execute("PRAGMA enable_checkpoint_on_shutdown")
                self._init_schema()
                self.paper_journal = PaperOrderJournal(self.connection)
        return self.connection

    def close(self) -> None:
        if self.payload_connection is not None:
            self.payload_connection.close()
            self.payload_connection = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.paper_journal = None
            self._terminal_evidence_integrity_cache.clear()
            self._evidence_audit_summary.clear()
            self._storage_schema_version_by_run.clear()
            self._next_chunk_index_by_run.clear()
            self._last_sequence_by_lane.clear()
            self._last_capture_order_key_by_run.clear()
            self._sequence_state_initialized_runs.clear()
            self._frame_cache_id = ""
            self._frame_cache = b""
            self._validated_condition_cache_runs.clear()

    def _payload_connection(self) -> duckdb.DuckDBPyConnection:
        if self.payload_connection is None:
            self.payload_connection = duckdb.connect(
                str(self.path),
                read_only=self.read_only,
            )
            self.payload_connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self.payload_connection.execute(f"SET threads={self.threads}")
            self.payload_connection.execute("SET TimeZone='UTC'")
            self.payload_connection.execute("SET preserve_insertion_order=false")
        return self.payload_connection

    def __enter__(self) -> "PolymarketEvidenceStore":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _init_schema(self) -> None:
        connection = self.connection
        if connection is None:
            raise RuntimeError("Polymarket evidence connection is unavailable")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS polymarket_recorder_run (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                storage_schema_version VARCHAR NOT NULL DEFAULT
                    'polymarket-public-evidence-v1',
                status VARCHAR NOT NULL,
                started_at_ms BIGINT NOT NULL,
                ended_at_ms BIGINT,
                report_json VARCHAR NOT NULL,
                report_sha256 VARCHAR NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS polymarket_terminal_audit_recovery (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                recovered_at_ms BIGINT NOT NULL,
                recovery_reason VARCHAR NOT NULL,
                prior_report_json VARCHAR NOT NULL,
                prior_report_sha256 VARCHAR NOT NULL,
                prior_error VARCHAR NOT NULL,
                recovered_report_sha256 VARCHAR NOT NULL,
                recovery_json VARCHAR NOT NULL,
                recovery_sha256 VARCHAR NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS polymarket_market_snapshot (
                snapshot_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL,
                observed_wall_ms BIGINT NOT NULL,
                observed_monotonic_ns UBIGINT NOT NULL,
                asset VARCHAR NOT NULL,
                market_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                slug VARCHAR NOT NULL,
                question VARCHAR NOT NULL,
                event_start_ms BIGINT NOT NULL,
                end_ms BIGINT NOT NULL,
                up_token_id VARCHAR NOT NULL,
                down_token_id VARCHAR NOT NULL,
                tick_size VARCHAR NOT NULL,
                minimum_order_size VARCHAR NOT NULL,
                fees_enabled BOOLEAN NOT NULL,
                fee_rate VARCHAR NOT NULL,
                fee_exponent INTEGER NOT NULL,
                fee_taker_only BOOLEAN NOT NULL,
                fee_rebate_rate VARCHAR NOT NULL,
                liquidity_quote VARCHAR NOT NULL,
                volume_quote VARCHAR NOT NULL,
                resolution_source VARCHAR NOT NULL,
                gamma_payload_json VARCHAR NOT NULL,
                gamma_payload_sha256 VARCHAR NOT NULL,
                clob_info_json VARCHAR NOT NULL,
                clob_info_sha256 VARCHAR NOT NULL,
                up_fee_rate_json VARCHAR NOT NULL,
                up_fee_rate_sha256 VARCHAR NOT NULL,
                down_fee_rate_json VARCHAR NOT NULL,
                down_fee_rate_sha256 VARCHAR NOT NULL,
                maker_base_fee INTEGER NOT NULL,
                taker_base_fee INTEGER NOT NULL,
                taker_order_delay_enabled BOOLEAN NOT NULL,
                minimum_order_age_seconds INTEGER NOT NULL,
                snapshot_payload_json VARCHAR NOT NULL,
                snapshot_sha256 VARCHAR NOT NULL,
                UNIQUE(run_id, condition_id)
            );

            CREATE TABLE IF NOT EXISTS polymarket_raw_message (
                message_id VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                sequence_number UBIGINT NOT NULL,
                received_wall_ms BIGINT NOT NULL,
                received_monotonic_ns UBIGINT NOT NULL,
                raw_payload_sha256 VARCHAR NOT NULL,
                raw_text VARCHAR NOT NULL,
                parse_status VARCHAR NOT NULL,
                parse_error VARCHAR NOT NULL,
                storage_chunk_id VARCHAR,
                chunk_message_index UINTEGER,
                raw_offset UBIGINT,
                raw_size UINTEGER,
                normalized_event_count UINTEGER
            );

            CREATE TABLE IF NOT EXISTS polymarket_raw_chunk (
                chunk_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL,
                chunk_index UBIGINT NOT NULL,
                frame_format VARCHAR NOT NULL,
                codec VARCHAR NOT NULL,
                compression_level UTINYINT NOT NULL,
                message_count UINTEGER NOT NULL,
                first_message_id VARCHAR NOT NULL,
                last_message_id VARCHAR NOT NULL,
                message_manifest_xor VARCHAR NOT NULL,
                uncompressed_bytes UBIGINT NOT NULL,
                uncompressed_sha256 VARCHAR NOT NULL,
                compressed_bytes UBIGINT NOT NULL,
                compressed_sha256 VARCHAR NOT NULL,
                compressed_payload BLOB NOT NULL,
                stream_counts_json VARCHAR DEFAULT '{}',
                UNIQUE(run_id, chunk_index),
                CHECK(message_count BETWEEN 1 AND 1024),
                CHECK(uncompressed_bytes BETWEEN 1 AND 67108864),
                CHECK(compressed_bytes >= 1)
            );

            CREATE TABLE IF NOT EXISTS polymarket_public_event (
                event_id VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                message_id VARCHAR NOT NULL,
                sub_index UINTEGER NOT NULL,
                stream VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                asset_id VARCHAR NOT NULL,
                source_time_ms BIGINT,
                publisher_time_ms BIGINT,
                event_json VARCHAR NOT NULL,
                event_sha256 VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS polymarket_condition_message_frame (
                run_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                schema_version VARCHAR NOT NULL,
                previous_frame_sha256 VARCHAR NOT NULL,
                message_count UINTEGER NOT NULL,
                first_received_monotonic_ns UBIGINT NOT NULL,
                last_received_monotonic_ns UBIGINT NOT NULL,
                uncompressed_bytes UINTEGER NOT NULL,
                uncompressed_sha256 VARCHAR NOT NULL,
                compressed_bytes UINTEGER NOT NULL,
                compressed_sha256 VARCHAR NOT NULL,
                compressed_payload BLOB NOT NULL,
                frame_sha256 VARCHAR NOT NULL,
                PRIMARY KEY(run_id, condition_id, frame_index)
            );

            CREATE TABLE IF NOT EXISTS polymarket_condition_message_manifest (
                run_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL,
                source_run_report_sha256 VARCHAR NOT NULL,
                frame_count UINTEGER NOT NULL,
                message_count UBIGINT NOT NULL,
                first_received_monotonic_ns UBIGINT NOT NULL,
                last_received_monotonic_ns UBIGINT NOT NULL,
                last_frame_sha256 VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL,
                PRIMARY KEY(run_id, condition_id)
            );

            CREATE TABLE IF NOT EXISTS polymarket_condition_cache_build (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                source_run_report_sha256 VARCHAR NOT NULL,
                state VARCHAR NOT NULL,
                condition_count UINTEGER NOT NULL,
                frame_count UBIGINT NOT NULL,
                message_count UBIGINT NOT NULL,
                report_json VARCHAR NOT NULL,
                report_sha256 VARCHAR NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS polymarket_stream_gap (
                gap_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                opened_at_ms BIGINT NOT NULL,
                reason VARCHAR NOT NULL,
                last_sequence_number UBIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS polymarket_resolution_evidence (
                resolution_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                market_id VARCHAR NOT NULL,
                asset VARCHAR NOT NULL,
                observed_wall_ms BIGINT NOT NULL,
                observed_monotonic_ns UBIGINT NOT NULL,
                winning_asset_id VARCHAR NOT NULL,
                winning_outcome VARCHAR NOT NULL,
                clob_payload_json VARCHAR NOT NULL,
                clob_payload_sha256 VARCHAR NOT NULL,
                gamma_payload_json VARCHAR NOT NULL,
                gamma_payload_sha256 VARCHAR NOT NULL,
                evidence_payload_json VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL,
                UNIQUE(run_id, condition_id)
            );

            CREATE TABLE IF NOT EXISTS polymarket_ai_veto_cache_v2 (
                cache_key_sha256 VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                identity_json VARCHAR NOT NULL,
                response_json VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL,
                latency_seconds DOUBLE NOT NULL,
                created_at_ms BIGINT NOT NULL,
                CHECK(latency_seconds >= 0.0 AND latency_seconds <= 600.0)
            );
            """
        )
        connection.execute(
            """
            ALTER TABLE polymarket_recorder_run ADD COLUMN IF NOT EXISTS
                storage_schema_version VARCHAR DEFAULT
                    'polymarket-public-evidence-v1';
            ALTER TABLE polymarket_raw_message ADD COLUMN IF NOT EXISTS
                storage_chunk_id VARCHAR;
            ALTER TABLE polymarket_raw_message ADD COLUMN IF NOT EXISTS
                chunk_message_index UINTEGER;
            ALTER TABLE polymarket_raw_message ADD COLUMN IF NOT EXISTS
                raw_offset UBIGINT;
            ALTER TABLE polymarket_raw_message ADD COLUMN IF NOT EXISTS
                raw_size UINTEGER;
            ALTER TABLE polymarket_raw_message ADD COLUMN IF NOT EXISTS
                normalized_event_count UINTEGER;
            ALTER TABLE polymarket_raw_chunk ADD COLUMN IF NOT EXISTS
                stream_counts_json VARCHAR DEFAULT '{}';
            UPDATE polymarket_recorder_run
            SET storage_schema_version = 'polymarket-public-evidence-v1'
            WHERE storage_schema_version IS NULL OR storage_schema_version = '';
            """
        )

    def get_polymarket_ai_veto_cache(
        self,
        cache_key_sha256: str,
    ) -> Mapping[str, object] | None:
        key = str(cache_key_sha256 or "").strip().lower()
        if len(key) != 64 or any(value not in "0123456789abcdef" for value in key):
            raise ValueError("Polymarket AI cache key is invalid")
        row = (
            self.connect()
            .execute(
                """
            SELECT schema_version, identity_json, response_json,
                   evidence_sha256, latency_seconds
            FROM polymarket_ai_veto_cache_v2
            WHERE cache_key_sha256 = ?
            """,
                [key],
            )
            .fetchone()
        )
        if row is None:
            return None
        identity = _strict_json_loads(str(row[1]))
        if (
            not isinstance(identity, Mapping)
            or str(identity.get("schema_version") or "") != str(row[0])
            or _canonical_json(identity) != str(row[1])
            or not hmac.compare_digest(_canonical_sha256(identity), key)
        ):
            raise ValueError("Polymarket AI cache identity is corrupt")
        response_json = str(row[2])
        response = _strict_json_loads(response_json)
        if _canonical_json(response) != response_json:
            raise ValueError("Polymarket AI cache response is not canonical")
        latency = float(row[4])
        if (
            not math.isfinite(latency)
            or not 0.0 <= latency <= _MAX_AI_CACHE_LATENCY_SECONDS
        ):
            raise ValueError("Polymarket AI cache latency is corrupt")
        evidence_sha256 = _canonical_sha256(
            {
                "latency_seconds": latency,
                "response_payload": response,
            }
        )
        if not hmac.compare_digest(evidence_sha256, str(row[3])):
            raise ValueError("Polymarket AI cache payload hash mismatch")
        return {
            "identity": dict(identity),
            "response_payload": response,
            "evidence_sha256": evidence_sha256,
            "latency_seconds": latency,
        }

    def put_polymarket_ai_veto_cache(
        self,
        cache_key_sha256: str,
        *,
        identity: Mapping[str, object],
        response_payload: object,
        latency_seconds: float,
    ) -> None:
        if self.read_only:
            raise ValueError("read-only Polymarket evidence cannot store AI cache rows")
        key = str(cache_key_sha256 or "").strip().lower()
        identity_payload = dict(identity)
        identity_json = _canonical_json(identity_payload)
        schema_version = str(identity_payload.get("schema_version") or "").strip()
        latency = float(latency_seconds)
        if (
            len(key) != 64
            or any(value not in "0123456789abcdef" for value in key)
            or not schema_version
            or not hmac.compare_digest(_canonical_sha256(identity_payload), key)
            or not math.isfinite(latency)
            or not 0.0 <= latency <= _MAX_AI_CACHE_LATENCY_SECONDS
        ):
            raise ValueError("Polymarket AI cache entry is invalid")
        response_json = _canonical_json(response_payload)
        evidence_sha256 = _canonical_sha256(
            {
                "latency_seconds": latency,
                "response_payload": response_payload,
            }
        )
        self.connect().execute(
            """
            INSERT INTO polymarket_ai_veto_cache_v2 (
                cache_key_sha256, schema_version, identity_json, response_json,
                evidence_sha256, latency_seconds, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cache_key_sha256) DO NOTHING
            """,
            [
                key,
                schema_version,
                identity_json,
                response_json,
                evidence_sha256,
                latency,
                _wall_ms(),
            ],
        )

    def start_run(self, run_id: str, started_at_ms: int) -> None:
        connection = self.connect()
        connection.execute(
            """
            INSERT INTO polymarket_recorder_run (
                run_id, schema_version, storage_schema_version, status,
                started_at_ms, ended_at_ms, report_json, report_sha256, error
            ) VALUES (?, ?, ?, 'running', ?, NULL, '', '', '')
            """,
            [
                run_id,
                POLYMARKET_RECORDER_SCHEMA_VERSION,
                POLYMARKET_STORAGE_SCHEMA_VERSION,
                int(started_at_ms),
            ],
        )
        self._storage_schema_version_by_run[run_id] = POLYMARKET_STORAGE_SCHEMA_VERSION

    def _require_unindexed_compact_hot_tables(self) -> None:
        connection = self.connect()
        indexed_hot_tables = {
            str(table_name)
            for (table_name,) in connection.execute(
                """
                SELECT DISTINCT table_name
                FROM duckdb_constraints()
                WHERE constraint_type = 'UNIQUE'
                  AND table_name IN (
                      'polymarket_raw_message', 'polymarket_public_event'
                  )
                UNION
                SELECT DISTINCT table_name
                FROM duckdb_indexes()
                WHERE is_unique
                  AND table_name IN (
                      'polymarket_raw_message', 'polymarket_public_event'
                  )
                """
            ).fetchall()
        }
        if indexed_hot_tables:
            tables = ",".join(sorted(indexed_hot_tables))
            raise ValueError(
                "legacy compact recorder writes require a database without "
                f"incremental hot-path uniqueness indexes: {tables}"
            )

    def _require_running_run(self, run_id: str) -> None:
        row = (
            self.connect()
            .execute(
                "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
                [run_id],
            )
            .fetchone()
        )
        if row is None:
            raise ValueError(f"unknown Polymarket recorder run: {run_id}")
        if str(row[0]) != "running":
            raise ValueError("terminal Polymarket recorder evidence is immutable")

    def _storage_schema_version(self, run_id: str) -> str:
        cached = self._storage_schema_version_by_run.get(run_id)
        if cached is not None:
            return cached
        connection = self.connect()
        has_column = bool(
            connection.execute(
                """
                SELECT count(*) FROM duckdb_columns()
                WHERE table_name = 'polymarket_recorder_run'
                  AND column_name = 'storage_schema_version'
                """
            ).fetchone()[0]
        )
        if not has_column:
            row = connection.execute(
                "SELECT run_id FROM polymarket_recorder_run WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown Polymarket recorder run: {run_id}")
            self._storage_schema_version_by_run[run_id] = _LEGACY_STORAGE_SCHEMA_VERSION
            return _LEGACY_STORAGE_SCHEMA_VERSION
        row = connection.execute(
            """
            SELECT storage_schema_version FROM polymarket_recorder_run
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown Polymarket recorder run: {run_id}")
        version = str(row[0] or _LEGACY_STORAGE_SCHEMA_VERSION)
        if version not in {
            _LEGACY_STORAGE_SCHEMA_VERSION,
            *_COMPACT_STORAGE_SCHEMA_VERSIONS,
            POLYMARKET_STORAGE_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported Polymarket storage schema: {version}")
        self._storage_schema_version_by_run[run_id] = version
        return version

    def _terminal_recovery_row(self, run_id: str) -> tuple[object, ...] | None:
        connection = self.connect()
        table_exists = bool(
            connection.execute(
                """
                SELECT count(*) FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name = 'polymarket_terminal_audit_recovery'
                """
            ).fetchone()[0]
        )
        if not table_exists:
            return None
        return connection.execute(
            """
            SELECT run_id, schema_version, recovered_at_ms, recovery_reason,
                   prior_report_json, prior_report_sha256, prior_error,
                   recovered_report_sha256, recovery_json, recovery_sha256
            FROM polymarket_terminal_audit_recovery WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

    def _terminal_evidence_fingerprint(self, run_id: str) -> str:
        row = (
            self.connect()
            .execute(
                """
            SELECT r.schema_version, r.status, r.started_at_ms, r.ended_at_ms,
                   r.report_json, r.report_sha256, r.error,
                   (SELECT count(*) FROM polymarket_market_snapshot WHERE run_id = ?),
                   (SELECT count(*) FROM polymarket_raw_message WHERE run_id = ?),
                   (SELECT count(*) FROM polymarket_public_event WHERE run_id = ?),
                   (SELECT count(*) FROM polymarket_raw_chunk WHERE run_id = ?),
                   (SELECT coalesce(sum(message_count), 0)
                    FROM polymarket_raw_chunk WHERE run_id = ?),
                   (SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?)
            FROM polymarket_recorder_run AS r WHERE r.run_id = ?
            """,
                [run_id, run_id, run_id, run_id, run_id, run_id, run_id],
            )
            .fetchone()
        )
        recovery_row = self._terminal_recovery_row(run_id)
        file_state: list[dict[str, object]] = []
        if self.read_only:
            for label, path in (
                ("database", self.path),
                ("write_ahead_log", Path(f"{self.path}.wal")),
            ):
                try:
                    stat = path.stat()
                    file_state.append(
                        {
                            "label": label,
                            "exists": True,
                            "size": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                        }
                    )
                except FileNotFoundError:
                    file_state.append({"label": label, "exists": False})
                except OSError as exc:
                    file_state.append(
                        {
                            "label": label,
                            "exists": None,
                            "stat_error": exc.__class__.__name__,
                        }
                    )
        return _canonical_sha256(
            {
                "run_id": run_id,
                "capture_manifest_sha256": (
                    self._capture_manifest_sha256(run_id)
                    if row is not None
                    and self._storage_schema_version(run_id)
                    == POLYMARKET_STORAGE_SCHEMA_VERSION
                    else ""
                ),
                "database_file_state": file_state,
                "terminal_metadata_and_counts": None
                if row is None
                else [None if value is None else str(value) for value in row],
                "terminal_recovery": None
                if recovery_row is None
                else [
                    None if value is None else str(value) for value in recovery_row
                ],
            }
        )

    def _terminal_recovery_integrity_errors(
        self,
        run_id: str,
        current_report: Mapping[str, object] | None,
    ) -> tuple[str, ...]:
        row = self._terminal_recovery_row(run_id)
        if row is None:
            return ()
        errors: list[str] = []

        def invalid(detail: str) -> None:
            errors.append(f"terminal_audit_recovery_invalid:{run_id}:{detail}")

        (
            stored_run_id,
            schema_version,
            recovered_at_ms,
            recovery_reason,
            prior_report_json,
            prior_report_sha256,
            prior_error,
            recovered_report_sha256,
            recovery_json,
            recovery_sha256,
        ) = row
        recovery_payload: Mapping[str, object] | None = None
        try:
            parsed_recovery = _strict_json_loads(str(recovery_json))
            if not isinstance(parsed_recovery, Mapping):
                raise ValueError("recovery report is not an object")
            recovery_payload = parsed_recovery
            if _canonical_json(recovery_payload) != str(recovery_json):
                invalid("recovery_not_canonical")
            if set(recovery_payload) != set(
                TerminalAuditRecoveryReport.__dataclass_fields__
            ):
                invalid("recovery_fields")
            unhashed_recovery = dict(recovery_payload)
            embedded_recovery_sha = str(
                unhashed_recovery.pop("recovery_sha256", "")
            )
            actual_recovery_sha = _canonical_sha256(unhashed_recovery)
            if not hmac.compare_digest(actual_recovery_sha, str(recovery_sha256)):
                invalid("recovery_hash")
            if not hmac.compare_digest(
                embedded_recovery_sha,
                str(recovery_sha256),
            ):
                invalid("recovery_embedded_hash")
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
            invalid(f"recovery_json:{exc.__class__.__name__}")

        prior_report: Mapping[str, object] | None = None
        try:
            parsed_prior = _strict_json_loads(str(prior_report_json))
            if not isinstance(parsed_prior, Mapping):
                raise ValueError("prior report is not an object")
            prior_report = parsed_prior
            if _canonical_json(prior_report) != str(prior_report_json):
                invalid("prior_report_not_canonical")
            if set(prior_report) != set(RecorderReport.__dataclass_fields__):
                invalid("prior_report_fields")
            unhashed_prior = dict(prior_report)
            embedded_prior_sha = str(unhashed_prior.pop("report_sha256", ""))
            actual_prior_sha = _canonical_sha256(unhashed_prior)
            if not hmac.compare_digest(actual_prior_sha, str(prior_report_sha256)):
                invalid("prior_report_hash")
            if not hmac.compare_digest(embedded_prior_sha, str(prior_report_sha256)):
                invalid("prior_report_embedded_hash")
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
            invalid(f"prior_report_json:{exc.__class__.__name__}")

        if str(stored_run_id) != run_id:
            invalid("stored_run_id")
        if str(schema_version) != POLYMARKET_TERMINAL_RECOVERY_SCHEMA_VERSION:
            invalid("schema_version")
        if str(recovery_reason) != _TERMINAL_AUDIT_RECOVERY_REASON:
            invalid("recovery_reason")
        if not re.fullmatch(r"[0-9a-f]{64}", str(recovery_sha256)):
            invalid("recovery_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(prior_report_sha256)):
            invalid("prior_report_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(recovered_report_sha256)):
            invalid("recovered_report_sha256")
        if hmac.compare_digest(
            str(prior_report_sha256),
            str(recovered_report_sha256),
        ):
            invalid("report_hashes_equal")
        if prior_report is not None:
            if (
                str(prior_report.get("schema_version") or "")
                != POLYMARKET_RECORDER_SCHEMA_VERSION
                or str(prior_report.get("run_id") or "") != run_id
                or str(prior_report.get("status") or "") != "failed"
                or prior_report.get("integrity_errors")
                != ["terminal_integrity_audit_incomplete"]
                or prior_report.get("errors") != [str(prior_error)]
                or prior_report.get("normalized_event_count") != 0
                or prior_report.get("evidence_manifest_sha256") != ""
                or not str(prior_error).startswith(
                    _TERMINAL_AUDIT_RESOURCE_EXHAUSTED_PREFIX
                )
            ):
                invalid("prior_report_contract")
        if recovery_payload is not None:
            expected_scalars = {
                "schema_version": str(schema_version),
                "run_id": run_id,
                "recovered_at_ms": int(recovered_at_ms),
                "recovery_reason": str(recovery_reason),
                "prior_report_sha256": str(prior_report_sha256),
                "recovered_report_sha256": str(recovered_report_sha256),
                "recovery_sha256": str(recovery_sha256),
            }
            if any(
                recovery_payload.get(key) != value
                for key, value in expected_scalars.items()
            ):
                invalid("stored_fields_disagree")
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                str(recovery_payload.get("pre_recovery_evidence_fingerprint") or ""),
            ):
                invalid("pre_recovery_fingerprint")
            memory_limit = str(recovery_payload.get("memory_limit") or "")
            database_threads = recovery_payload.get("database_threads")
            if (
                not _DUCKDB_MEMORY_LIMIT.fullmatch(memory_limit)
                or memory_limit != memory_limit.upper()
                or type(database_threads) is not int
                or not 1 <= database_threads <= 8
            ):
                invalid("database_resources")
            for field in (
                "labels_consulted",
                "outcomes_consulted",
                "model_scores_consulted",
                "profitability_claim",
                "trading_authority",
                "training_authority",
            ):
                if recovery_payload.get(field) is not False:
                    invalid(field)
            for field in ("raw_message_count", "normalized_event_count"):
                value = recovery_payload.get(field)
                if type(value) is not int or value < 0:
                    invalid(field)
            if (
                prior_report is not None
                and recovery_payload.get("raw_message_count")
                != prior_report.get("raw_message_count")
            ):
                invalid("prior_report_count")
        if current_report is None:
            invalid("current_report_missing")
        else:
            current_report_sha256 = str(current_report.get("report_sha256") or "")
            if (
                not hmac.compare_digest(
                    current_report_sha256,
                    str(recovered_report_sha256),
                )
                or str(current_report.get("status") or "")
                not in {"complete", "degraded"}
                or current_report.get("errors") != []
                or current_report.get("integrity_errors") != []
            ):
                invalid("current_report_contract")
            if recovery_payload is not None and any(
                recovery_payload.get(field) != current_report.get(field)
                for field in ("raw_message_count", "normalized_event_count")
            ):
                invalid("current_report_counts")
            ended_at_ms = current_report.get("ended_at_ms")
            if (
                type(ended_at_ms) is not int
                or int(recovered_at_ms) < ended_at_ms
            ):
                invalid("recovery_time")
        return tuple(errors)

    def _capture_manifest_sha256(self, run_id: str) -> str:
        if self._storage_schema_version(run_id) != POLYMARKET_STORAGE_SCHEMA_VERSION:
            return ""
        previous = _canonical_sha256(
            {
                "schema_version": POLYMARKET_CAPTURE_MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
        cursor = self.connect().execute(
            """
            SELECT chunk_id, schema_version, chunk_index, frame_format, codec,
                   compression_level, message_count, first_message_id,
                   last_message_id, message_manifest_xor, uncompressed_bytes,
                   uncompressed_sha256, compressed_bytes, compressed_sha256,
                   stream_counts_json
            FROM polymarket_raw_chunk
            WHERE run_id = ? ORDER BY chunk_index
            """,
            [run_id],
        )
        for rows in iter(lambda: cursor.fetchmany(4_096), []):
            for row in rows:
                previous = _canonical_sha256(
                    {
                        "schema_version": POLYMARKET_CAPTURE_MANIFEST_SCHEMA_VERSION,
                        "run_id": run_id,
                        "previous_sha256": previous,
                        "chunk": {
                            "chunk_id": str(row[0]),
                            "storage_schema_version": str(row[1]),
                            "chunk_index": int(row[2]),
                            "frame_format": str(row[3]),
                            "codec": str(row[4]),
                            "compression_level": int(row[5]),
                            "message_count": int(row[6]),
                            "first_message_id": str(row[7]),
                            "last_message_id": str(row[8]),
                            "message_manifest_xor": str(row[9]),
                            "uncompressed_bytes": int(row[10]),
                            "uncompressed_sha256": str(row[11]),
                            "compressed_bytes": int(row[12]),
                            "compressed_sha256": str(row[13]),
                            "stream_counts_json": str(row[14]),
                        },
                    }
                )
        return previous

    def record_market_evidence(self, run_id: str, evidence: MarketEvidence) -> str:
        self._require_running_run(run_id)
        market = evidence.market
        _validated_canonical_payload(
            "Gamma market", market.gamma_payload_json, market.gamma_payload_sha256
        )
        _validated_canonical_payload(
            "CLOB market", evidence.clob_info_json, evidence.clob_info_sha256
        )
        _validated_canonical_payload(
            "Up fee-rate", evidence.up_fee_rate_json, evidence.up_fee_rate_sha256
        )
        _validated_canonical_payload(
            "Down fee-rate", evidence.down_fee_rate_json, evidence.down_fee_rate_sha256
        )
        if (
            min(
                int(evidence.observed_wall_ms),
                int(evidence.observed_monotonic_ns),
                int(evidence.maker_base_fee),
                int(evidence.taker_base_fee),
                int(evidence.minimum_order_age_seconds),
            )
            < 0
        ):
            raise ValueError(
                "market evidence timestamps and fee parameters must be non-negative"
            )
        normalized = market.asdict()
        identity = {
            "run_id": run_id,
            "observed_wall_ms": int(evidence.observed_wall_ms),
            "market": normalized,
            "clob_info_sha256": evidence.clob_info_sha256,
            "up_fee_rate_sha256": evidence.up_fee_rate_sha256,
            "down_fee_rate_sha256": evidence.down_fee_rate_sha256,
        }
        snapshot_id = _canonical_sha256(identity)
        snapshot_payload = {
            **identity,
            "observed_monotonic_ns": int(evidence.observed_monotonic_ns),
            "maker_base_fee": int(evidence.maker_base_fee),
            "taker_base_fee": int(evidence.taker_base_fee),
            "taker_order_delay_enabled": bool(evidence.taker_order_delay_enabled),
            "minimum_order_age_seconds": int(evidence.minimum_order_age_seconds),
        }
        snapshot_payload_json = _canonical_json(snapshot_payload)
        snapshot_sha = _canonical_sha256(snapshot_payload)
        self.connect().execute(
            """
            INSERT INTO polymarket_market_snapshot (
                snapshot_id, run_id, schema_version, observed_wall_ms,
                observed_monotonic_ns, asset, market_id, condition_id, slug,
                question, event_start_ms, end_ms, up_token_id, down_token_id,
                tick_size, minimum_order_size, fees_enabled, fee_rate,
                fee_exponent, fee_taker_only, fee_rebate_rate, liquidity_quote,
                volume_quote, resolution_source, gamma_payload_json,
                gamma_payload_sha256, clob_info_json, clob_info_sha256,
                up_fee_rate_json, up_fee_rate_sha256, down_fee_rate_json,
                down_fee_rate_sha256, maker_base_fee, taker_base_fee,
                taker_order_delay_enabled, minimum_order_age_seconds,
                snapshot_payload_json, snapshot_sha256
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                snapshot_id,
                run_id,
                POLYMARKET_EVIDENCE_SCHEMA_VERSION,
                evidence.observed_wall_ms,
                evidence.observed_monotonic_ns,
                market.asset,
                market.market_id,
                market.condition_id,
                market.slug,
                market.question,
                market.event_start_ms,
                market.end_ms,
                market.up_token_id,
                market.down_token_id,
                format(market.tick_size, "f"),
                format(market.minimum_order_size, "f"),
                market.fee_schedule.enabled,
                format(market.fee_schedule.rate, "f"),
                market.fee_schedule.exponent,
                market.fee_schedule.taker_only,
                format(market.fee_schedule.rebate_rate, "f"),
                format(market.liquidity_quote, "f"),
                format(market.volume_quote, "f"),
                market.resolution_source,
                market.gamma_payload_json,
                market.gamma_payload_sha256,
                evidence.clob_info_json,
                evidence.clob_info_sha256,
                evidence.up_fee_rate_json,
                evidence.up_fee_rate_sha256,
                evidence.down_fee_rate_json,
                evidence.down_fee_rate_sha256,
                evidence.maker_base_fee,
                evidence.taker_base_fee,
                evidence.taker_order_delay_enabled,
                evidence.minimum_order_age_seconds,
                snapshot_payload_json,
                snapshot_sha,
            ],
        )
        return snapshot_id

    def append_messages(
        self, run_id: str, messages: Sequence[RawStreamMessage]
    ) -> None:
        if not messages:
            return
        self._require_running_run(run_id)
        validated = tuple(message.validated() for message in messages)
        self._initialize_sequence_state(run_id)
        next_sequences: dict[tuple[str, str, str], int] = {}
        for message in validated:
            lane = (run_id, message.stream, message.connection_id)
            previous = next_sequences.get(
                lane,
                self._last_sequence_by_lane.get(lane),
            )
            if previous is not None and message.sequence_number <= previous:
                raise ValueError(
                    "stream sequence must increase strictly within each connection: "
                    f"{message.stream}:{message.connection_id}:"
                    f"{message.sequence_number}<={previous}"
                )
            next_sequences[lane] = message.sequence_number
        storage_schema_version = self._storage_schema_version(run_id)
        if storage_schema_version == _LEGACY_STORAGE_SCHEMA_VERSION:
            for start in range(0, len(validated), _RAW_CHUNK_MESSAGE_LIMIT):
                self._append_legacy_message_batch(
                    run_id,
                    validated[start : start + _RAW_CHUNK_MESSAGE_LIMIT],
                )
            self._last_sequence_by_lane.update(next_sequences)
            return
        if storage_schema_version == POLYMARKET_STORAGE_SCHEMA_VERSION:
            ordered_messages = tuple(sorted(validated, key=_capture_order_key))
            previous_order_key = self._last_capture_order_key_by_run.get(run_id)
            if (
                previous_order_key is not None
                and _capture_receipt_key(ordered_messages[0]) < previous_order_key
            ):
                raise ValueError(
                    "capture batch predates already persisted receipt order"
                )
            chunks: list[tuple[RawStreamMessage, ...]] = []
            pending: list[RawStreamMessage] = []
            pending_bytes = len(CAPTURE_FRAME_MAGIC)
            for message in ordered_messages:
                framed_size = capture_frame_record_size(
                    CaptureFrameRecord(**message.__dict__)
                )
                if pending and (
                    len(pending) >= _RAW_CHUNK_MESSAGE_LIMIT
                    or pending_bytes + framed_size > _MAX_RAW_CHUNK_BYTES
                ):
                    chunks.append(tuple(pending))
                    pending = []
                    pending_bytes = len(CAPTURE_FRAME_MAGIC)
                pending.append(message)
                pending_bytes += framed_size
            if pending:
                chunks.append(tuple(pending))
            connection = self.connect()
            cached_chunk_index_present = run_id in self._next_chunk_index_by_run
            cached_chunk_index = self._next_chunk_index_by_run.get(run_id)
            connection.execute("BEGIN TRANSACTION")
            try:
                for chunk in chunks:
                    self._append_capture_message_chunk(run_id, chunk, connection)
                connection.execute("COMMIT")
                self._last_sequence_by_lane.update(next_sequences)
                self._last_capture_order_key_by_run[run_id] = _capture_receipt_key(
                    ordered_messages[-1]
                )
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.TransactionException:
                    pass
                if cached_chunk_index_present:
                    assert cached_chunk_index is not None
                    self._next_chunk_index_by_run[run_id] = cached_chunk_index
                else:
                    self._next_chunk_index_by_run.pop(run_id, None)
                raise
            return
        if storage_schema_version not in _COMPACT_STORAGE_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported compact Polymarket storage schema: {storage_schema_version}"
            )
        self._require_unindexed_compact_hot_tables()
        chunks: list[tuple[RawStreamMessage, ...]] = []
        pending: list[RawStreamMessage] = []
        pending_bytes = 0
        for message in validated:
            framed_size = 4 + len(message.raw_text.encode("utf-8"))
            if pending and (
                len(pending) >= _RAW_CHUNK_MESSAGE_LIMIT
                or pending_bytes + framed_size > _MAX_RAW_CHUNK_BYTES
            ):
                chunks.append(tuple(pending))
                pending = []
                pending_bytes = 0
            pending.append(message)
            pending_bytes += framed_size
        if pending:
            chunks.append(tuple(pending))
        connection = self.connect()
        cached_chunk_index_present = run_id in self._next_chunk_index_by_run
        cached_chunk_index = self._next_chunk_index_by_run.get(run_id)
        connection.execute("BEGIN TRANSACTION")
        try:
            for chunk in chunks:
                self._append_compact_message_chunk(
                    run_id,
                    chunk,
                    connection,
                    storage_schema_version=storage_schema_version,
                )
            connection.execute("COMMIT")
            self._last_sequence_by_lane.update(next_sequences)
        except Exception:
            try:
                try:
                    connection.execute("ROLLBACK")
                except duckdb.TransactionException:
                    # DuckDB can end the transaction while reporting a failed
                    # COMMIT (for example, an unavailable WAL). Preserve the
                    # original commit error instead of masking it with a
                    # second "no transaction is active" exception.
                    pass
            finally:
                if cached_chunk_index_present:
                    assert cached_chunk_index is not None
                    self._next_chunk_index_by_run[run_id] = cached_chunk_index
                else:
                    self._next_chunk_index_by_run.pop(run_id, None)
            raise

    def _initialize_sequence_state(self, run_id: str) -> None:
        if run_id in self._sequence_state_initialized_runs:
            return
        if self._storage_schema_version(run_id) == POLYMARKET_STORAGE_SCHEMA_VERSION:
            latest: dict[tuple[str, str, str], int] = {}
            last_order_key: tuple[int, int] | None = None
            for stored in self._iter_capture_messages(run_id):
                message = stored.message
                order_key = _capture_receipt_key(message)
                if last_order_key is not None and order_key < last_order_key:
                    raise ValueError("stored capture receipt order is not monotonic")
                last_order_key = order_key
                lane = (run_id, message.stream, message.connection_id)
                latest[lane] = max(latest.get(lane, -1), message.sequence_number)
            self._last_sequence_by_lane.update(latest)
            if last_order_key is not None:
                self._last_capture_order_key_by_run[run_id] = last_order_key
            self._sequence_state_initialized_runs.add(run_id)
            return
        rows = (
            self.connect()
            .execute(
                """
            SELECT stream, connection_id, max(sequence_number)
            FROM polymarket_raw_message
            WHERE run_id = ?
            GROUP BY stream, connection_id
            """,
                [run_id],
            )
            .fetchall()
        )
        self._last_sequence_by_lane.update(
            {
                (run_id, str(stream), str(connection_id)): int(sequence_number)
                for stream, connection_id, sequence_number in rows
            }
        )
        self._sequence_state_initialized_runs.add(run_id)

    def _append_legacy_message_batch(
        self,
        run_id: str,
        messages: Sequence[RawStreamMessage],
    ) -> None:
        raw_rows: list[tuple[object, ...]] = []
        event_rows: list[tuple[object, ...]] = []
        for message in messages:
            raw_row, normalized_rows = self._message_rows(
                run_id,
                message,
                inline_payload=True,
            )
            raw_rows.append(raw_row)
            event_rows.extend(normalized_rows)
        connection = self.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                _LEGACY_RAW_MESSAGE_INSERT_SQL,
                _columnar_batch(raw_rows, width=12),
            )
            if event_rows:
                connection.execute(
                    _PUBLIC_EVENT_INSERT_SQL,
                    _columnar_batch(event_rows, width=13),
                )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except duckdb.TransactionException:
                pass
            raise

    def _append_compact_message_chunk(
        self,
        run_id: str,
        messages: Sequence[RawStreamMessage],
        connection: duckdb.DuckDBPyConnection,
        *,
        storage_schema_version: str,
    ) -> None:
        if not messages or len(messages) > _RAW_CHUNK_MESSAGE_LIMIT:
            raise ValueError("compact evidence chunk has an invalid message count")
        frame = bytearray()
        prepared: list[
            tuple[
                tuple[object, ...],
                tuple[tuple[object, ...], ...],
                int,
                int,
            ]
        ] = []
        event_rows: list[tuple[object, ...]] = []
        for message in messages:
            raw_bytes = message.raw_text.encode("utf-8")
            raw_offset = len(frame) + 4
            frame.extend(struct.pack("<I", len(raw_bytes)))
            frame.extend(raw_bytes)
            raw_row, normalized_rows = self._message_rows(
                run_id,
                message,
                inline_payload=False,
            )
            prepared.append((raw_row, normalized_rows, raw_offset, len(raw_bytes)))
            event_rows.extend(normalized_rows)
        if not 1 <= len(frame) <= _MAX_RAW_CHUNK_BYTES:
            raise ValueError("compact evidence chunk exceeded its bounded size")
        chunk_index = self._next_chunk_index_by_run.get(run_id)
        if chunk_index is None:
            chunk_index = int(
                connection.execute(
                    """
                    SELECT coalesce(max(chunk_index), -1) + 1
                    FROM polymarket_raw_chunk WHERE run_id = ?
                    """,
                    [run_id],
                ).fetchone()[0]
            )
        uncompressed = bytes(frame)
        uncompressed_sha = hashlib.sha256(uncompressed).hexdigest()
        compressed = zstandard.ZstdCompressor(
            level=_RAW_CHUNK_COMPRESSION_LEVEL,
            write_checksum=True,
            write_content_size=True,
            threads=0,
        ).compress(uncompressed)
        compressed_sha = hashlib.sha256(compressed).hexdigest()
        raw_rows = [
            raw_row
            + (
                "",
                message_index,
                raw_offset,
                raw_size,
                len(normalized_rows),
            )
            for message_index, (
                raw_row,
                normalized_rows,
                raw_offset,
                raw_size,
            ) in enumerate(prepared)
        ]
        manifest_xor = 0
        for raw_row in raw_rows:
            manifest_xor ^= int(_compact_message_manifest_hash(raw_row), 16)
        manifest_xor_hex = f"{manifest_xor:064x}"
        chunk_id = _canonical_sha256(
            {
                "run_id": run_id,
                "schema_version": storage_schema_version,
                "chunk_index": chunk_index,
                "message_count": len(prepared),
                "first_message_id": str(prepared[0][0][0]),
                "last_message_id": str(prepared[-1][0][0]),
                "message_manifest_xor": manifest_xor_hex,
                "uncompressed_sha256": uncompressed_sha,
            }
        )
        raw_rows = [row[:12] + (chunk_id,) + row[13:] for row in raw_rows]
        connection.execute(
            """
            INSERT INTO polymarket_raw_chunk (
                chunk_id, run_id, schema_version, chunk_index, frame_format,
                codec, compression_level, message_count, first_message_id,
                last_message_id, message_manifest_xor, uncompressed_bytes,
                uncompressed_sha256, compressed_bytes, compressed_sha256,
                compressed_payload
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                chunk_id,
                run_id,
                storage_schema_version,
                chunk_index,
                _RAW_CHUNK_FRAME_FORMAT,
                _RAW_CHUNK_CODEC,
                _RAW_CHUNK_COMPRESSION_LEVEL,
                len(prepared),
                str(prepared[0][0][0]),
                str(prepared[-1][0][0]),
                manifest_xor_hex,
                len(uncompressed),
                uncompressed_sha,
                len(compressed),
                compressed_sha,
                compressed,
            ],
        )
        connection.execute(
            _COMPACT_RAW_MESSAGE_INSERT_SQL,
            _columnar_batch(raw_rows, width=17),
        )
        if event_rows:
            connection.execute(
                _PUBLIC_EVENT_INSERT_SQL,
                _columnar_batch(event_rows, width=13),
            )
        self._next_chunk_index_by_run[run_id] = chunk_index + 1

    def _append_capture_message_chunk(
        self,
        run_id: str,
        messages: Sequence[RawStreamMessage],
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        if not messages or len(messages) > _RAW_CHUNK_MESSAGE_LIMIT:
            raise ValueError("capture evidence chunk has an invalid message count")
        records = tuple(CaptureFrameRecord(**message.__dict__) for message in messages)
        uncompressed, located = encode_capture_frame(records)
        message_ids: list[str] = []
        manifest_xor = 0
        stream_counts: dict[str, int] = {}
        for item in located:
            raw_sha = hashlib.sha256(item.record.raw_text.encode("utf-8")).hexdigest()
            message_id = _canonical_sha256(
                {
                    "run_id": run_id,
                    "stream": item.record.stream,
                    "connection_id": item.record.connection_id,
                    "sequence_number": item.record.sequence_number,
                    "raw_payload_sha256": raw_sha,
                }
            )
            message_ids.append(message_id)
            manifest_xor ^= int(
                _capture_message_manifest_hash(
                    run_id=run_id,
                    message_id=message_id,
                    raw_payload_sha256=raw_sha,
                    located=item,
                ),
                16,
            )
            stream_counts[item.record.stream] = (
                stream_counts.get(item.record.stream, 0) + 1
            )
        chunk_index = self._next_chunk_index_by_run.get(run_id)
        if chunk_index is None:
            chunk_index = int(
                connection.execute(
                    """
                    SELECT coalesce(max(chunk_index), -1) + 1
                    FROM polymarket_raw_chunk WHERE run_id = ?
                    """,
                    [run_id],
                ).fetchone()[0]
            )
        uncompressed_sha = hashlib.sha256(uncompressed).hexdigest()
        compressed = zstandard.ZstdCompressor(
            level=_RAW_CHUNK_COMPRESSION_LEVEL,
            write_checksum=True,
            write_content_size=True,
            threads=0,
        ).compress(uncompressed)
        compressed_sha = hashlib.sha256(compressed).hexdigest()
        manifest_xor_hex = f"{manifest_xor:064x}"
        chunk_id = _canonical_sha256(
            {
                "run_id": run_id,
                "schema_version": POLYMARKET_STORAGE_SCHEMA_VERSION,
                "chunk_index": chunk_index,
                "message_count": len(located),
                "first_message_id": message_ids[0],
                "last_message_id": message_ids[-1],
                "message_manifest_xor": manifest_xor_hex,
                "uncompressed_sha256": uncompressed_sha,
            }
        )
        connection.execute(
            """
            INSERT INTO polymarket_raw_chunk (
                chunk_id, run_id, schema_version, chunk_index, frame_format,
                codec, compression_level, message_count, first_message_id,
                last_message_id, message_manifest_xor, uncompressed_bytes,
                uncompressed_sha256, compressed_bytes, compressed_sha256,
                compressed_payload, stream_counts_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                chunk_id,
                run_id,
                POLYMARKET_STORAGE_SCHEMA_VERSION,
                chunk_index,
                CAPTURE_FRAME_FORMAT,
                _RAW_CHUNK_CODEC,
                _RAW_CHUNK_COMPRESSION_LEVEL,
                len(located),
                message_ids[0],
                message_ids[-1],
                manifest_xor_hex,
                len(uncompressed),
                uncompressed_sha,
                len(compressed),
                compressed_sha,
                compressed,
                _canonical_json(dict(sorted(stream_counts.items()))),
            ],
        )
        self._next_chunk_index_by_run[run_id] = chunk_index + 1

    @staticmethod
    def _message_rows(
        run_id: str,
        message: RawStreamMessage,
        *,
        inline_payload: bool,
    ) -> tuple[tuple[object, ...], tuple[tuple[object, ...], ...]]:
        raw_bytes = message.raw_text.encode("utf-8")
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        message_id = _canonical_sha256(
            {
                "run_id": run_id,
                "stream": message.stream,
                "connection_id": message.connection_id,
                "sequence_number": message.sequence_number,
                "raw_payload_sha256": raw_sha,
            }
        )
        parse_status, parse_error, events = _stream_message_events(message.raw_text)
        raw_row = (
            message_id,
            run_id,
            POLYMARKET_EVIDENCE_SCHEMA_VERSION,
            message.stream,
            message.connection_id,
            message.sequence_number,
            message.received_wall_ms,
            message.received_monotonic_ns,
            raw_sha,
            message.raw_text if inline_payload else "",
            parse_status,
            parse_error,
        )
        event_rows: list[tuple[object, ...]] = []
        for sub_index, event in enumerate(events):
            event_json = _canonical_json(dict(event))
            event_sha = hashlib.sha256(event_json.encode("ascii")).hexdigest()
            event_id = _canonical_sha256(
                {
                    "message_id": message_id,
                    "sub_index": sub_index,
                    "event_sha256": event_sha,
                }
            )
            normalized = _event_index(message.stream, event)
            event_rows.append(
                (
                    event_id,
                    run_id,
                    message_id,
                    sub_index,
                    message.stream,
                    normalized["event_type"],
                    normalized["symbol"],
                    normalized["condition_id"],
                    normalized["asset_id"],
                    normalized["source_time_ms"],
                    normalized["publisher_time_ms"],
                    event_json if inline_payload else "",
                    event_sha,
                )
            )
        return raw_row, tuple(event_rows)

    def _decode_raw_chunk_payload(
        self,
        run_id: str,
        row: Sequence[object],
    ) -> bytes:
        if len(row) != 17:
            raise ValueError("compressed evidence chunk row has an invalid width")
        (
            claimed_chunk_id,
            chunk_run,
            schema_version,
            chunk_index,
            frame_format,
            codec,
            compression_level,
            message_count,
            first_message_id,
            last_message_id,
            message_manifest_xor,
            uncompressed_bytes,
            uncompressed_sha256,
            compressed_bytes,
            compressed_sha256,
            compressed_payload,
            _stream_counts_json,
        ) = row
        expected_chunk_id = _canonical_sha256(
            {
                "run_id": str(chunk_run),
                "schema_version": str(schema_version),
                "chunk_index": int(chunk_index),
                "message_count": int(message_count),
                "first_message_id": str(first_message_id),
                "last_message_id": str(last_message_id),
                "message_manifest_xor": str(message_manifest_xor),
                "uncompressed_sha256": str(uncompressed_sha256),
            }
        )
        expected_storage_schema = self._storage_schema_version(run_id)
        expected_frame_format = (
            CAPTURE_FRAME_FORMAT
            if str(schema_version) == POLYMARKET_STORAGE_SCHEMA_VERSION
            else _RAW_CHUNK_FRAME_FORMAT
        )
        if (
            str(chunk_run) != run_id
            or str(schema_version) != expected_storage_schema
            or str(schema_version) not in _CHUNK_STORAGE_SCHEMA_VERSIONS
            or str(frame_format) != expected_frame_format
            or str(codec) != _RAW_CHUNK_CODEC
            or int(compression_level) != _RAW_CHUNK_COMPRESSION_LEVEL
            or not 1 <= int(message_count) <= _RAW_CHUNK_MESSAGE_LIMIT
            or not 1 <= int(uncompressed_bytes) <= _MAX_RAW_CHUNK_BYTES
            or int(compressed_bytes) < 1
            or not hmac.compare_digest(str(claimed_chunk_id), expected_chunk_id)
        ):
            raise ValueError("compact raw-message chunk metadata is invalid")
        compressed = bytes(compressed_payload)
        if len(compressed) != int(compressed_bytes) or hashlib.sha256(
            compressed
        ).hexdigest() != str(compressed_sha256):
            raise ValueError("compact raw-message chunk payload hash mismatch")
        try:
            content_size = zstandard.frame_content_size(compressed)
            if int(content_size) != int(uncompressed_bytes):
                raise ValueError("compact raw-message frame size header drifted")
            frame = self._decompressor.decompress(
                compressed,
                max_output_size=int(uncompressed_bytes),
                allow_extra_data=False,
            )
        except (MemoryError, zstandard.ZstdError) as exc:
            raise ValueError("compact raw-message chunk decompression failed") from exc
        if len(frame) != int(uncompressed_bytes) or hashlib.sha256(
            frame
        ).hexdigest() != str(uncompressed_sha256):
            raise ValueError("compact raw-message frame hash mismatch")
        return frame

    def _load_compact_frame(self, run_id: str, chunk_id: str) -> bytes:
        if chunk_id == self._frame_cache_id:
            return self._frame_cache
        stream_counts_projection = (
            "stream_counts_json"
            if self._storage_schema_version(run_id) == POLYMARKET_STORAGE_SCHEMA_VERSION
            else "'{}' AS stream_counts_json"
        )
        row = (
            self._payload_connection()
            .execute(
                f"""
            SELECT chunk_id, run_id, schema_version, chunk_index, frame_format,
                   codec, compression_level, message_count, first_message_id,
                   last_message_id, message_manifest_xor, uncompressed_bytes,
                   uncompressed_sha256, compressed_bytes, compressed_sha256,
                   compressed_payload, {stream_counts_projection}
            FROM polymarket_raw_chunk
            WHERE run_id = ? AND chunk_id = ?
            """,
                [run_id, chunk_id],
            )
            .fetchone()
        )
        if row is None:
            raise ValueError("compact raw message references a missing chunk")
        frame = self._decode_raw_chunk_payload(run_id, row)
        self._frame_cache_id = chunk_id
        self._frame_cache = frame
        return frame

    def _decode_capture_chunk_row(
        self,
        run_id: str,
        row: Sequence[object],
    ) -> tuple[_StoredCaptureMessage, ...]:
        if self._storage_schema_version(run_id) != POLYMARKET_STORAGE_SCHEMA_VERSION:
            raise ValueError("capture-frame decoding requires storage v4")
        frame = self._decode_raw_chunk_payload(run_id, row)
        chunk_id = str(row[0])
        expected_message_count = int(row[7])
        try:
            located = decode_capture_frame(
                frame,
                expected_message_count=expected_message_count,
            )
        except ValueError as exc:
            raise ValueError("capture evidence frame cannot be decoded") from exc
        stored: list[_StoredCaptureMessage] = []
        manifest_xor = 0
        stream_counts: dict[str, int] = {}
        for item in located:
            raw_sha = hashlib.sha256(item.record.raw_text.encode("utf-8")).hexdigest()
            message_id = _canonical_sha256(
                {
                    "run_id": run_id,
                    "stream": item.record.stream,
                    "connection_id": item.record.connection_id,
                    "sequence_number": item.record.sequence_number,
                    "raw_payload_sha256": raw_sha,
                }
            )
            manifest_xor ^= int(
                _capture_message_manifest_hash(
                    run_id=run_id,
                    message_id=message_id,
                    raw_payload_sha256=raw_sha,
                    located=item,
                ),
                16,
            )
            stream_counts[item.record.stream] = (
                stream_counts.get(item.record.stream, 0) + 1
            )
            stored.append(
                _StoredCaptureMessage(
                    message_id=message_id,
                    raw_payload_sha256=raw_sha,
                    storage_chunk_id=chunk_id,
                    raw_offset=item.raw_offset,
                    raw_size=item.raw_size,
                    message=RawStreamMessage(**item.record.__dict__).validated(),
                )
            )
        try:
            claimed_stream_counts = _strict_json_loads(str(row[16]))
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            raise ValueError("capture evidence stream counts are invalid") from exc
        if (
            not isinstance(claimed_stream_counts, Mapping)
            or _canonical_json(claimed_stream_counts) != str(row[16])
            or dict(claimed_stream_counts) != dict(sorted(stream_counts.items()))
            or str(row[8]) != stored[0].message_id
            or str(row[9]) != stored[-1].message_id
            or str(row[10]) != f"{manifest_xor:064x}"
        ):
            raise ValueError("capture evidence frame manifest differs")
        return tuple(stored)

    def _iter_capture_messages(
        self,
        run_id: str,
        *,
        streams: tuple[str, ...] | None = None,
    ) -> Iterator[_StoredCaptureMessage]:
        if self._storage_schema_version(run_id) != POLYMARKET_STORAGE_SCHEMA_VERSION:
            raise ValueError("capture-frame iteration requires storage v4")
        metadata_cursor = self.connect().execute(
            """
            SELECT chunk_id, run_id, schema_version, chunk_index,
                   frame_format, codec, compression_level, message_count,
                   first_message_id, last_message_id, message_manifest_xor,
                   uncompressed_bytes, uncompressed_sha256, compressed_bytes,
                   compressed_sha256, stream_counts_json
            FROM polymarket_raw_chunk
            WHERE run_id = ? ORDER BY chunk_index
            """,
            [run_id],
        )
        expected_chunk_index = 0
        while metadata_rows := metadata_cursor.fetchmany(
            _CAPTURE_AUDIT_CHUNK_PAGE_SIZE
        ):
            chunk_ids = [str(row[0]) for row in metadata_rows]
            placeholders = ",".join("?" for _value in chunk_ids)
            payload_rows = self._payload_connection().execute(
                f"""
                SELECT chunk_id, compressed_payload
                FROM polymarket_raw_chunk
                WHERE run_id = ? AND chunk_id IN ({placeholders})
                """,
                [run_id, *chunk_ids],
            ).fetchall()
            payloads = {str(chunk_id): payload for chunk_id, payload in payload_rows}
            if len(payloads) != len(metadata_rows):
                raise ValueError("capture evidence chunk payload set differs")
            for metadata in metadata_rows:
                if int(metadata[3]) != expected_chunk_index:
                    raise ValueError("capture evidence chunk sequence differs")
                chunk_id = str(metadata[0])
                row = (
                    *metadata[:15],
                    payloads[chunk_id],
                    metadata[15],
                )
                expected_chunk_index += 1
                for stored in self._decode_capture_chunk_row(run_id, row):
                    if streams is None or stored.message.stream in streams:
                        yield stored

    def _capture_frame_fast_counts(self, run_id: str) -> tuple[int, dict[str, int]]:
        raw_message_count = 0
        stream_counts: dict[str, int] = {}
        rows = (
            self.connect()
            .execute(
                """
            SELECT message_count, stream_counts_json
            FROM polymarket_raw_chunk
            WHERE run_id = ? ORDER BY chunk_index
            """,
                [run_id],
            )
            .fetchall()
        )
        for message_count, stream_counts_json in rows:
            count = int(message_count)
            parsed = _strict_json_loads(str(stream_counts_json))
            if (
                count < 1
                or not isinstance(parsed, Mapping)
                or _canonical_json(parsed) != str(stream_counts_json)
            ):
                raise ValueError("capture-frame count metadata is invalid")
            chunk_stream_count = 0
            for stream, value in parsed.items():
                normalized_stream = str(stream)
                normalized_count = int(value)
                if normalized_stream not in _STREAMS or normalized_count < 1:
                    raise ValueError("capture-frame stream count is invalid")
                stream_counts[normalized_stream] = (
                    stream_counts.get(normalized_stream, 0) + normalized_count
                )
                chunk_stream_count += normalized_count
            if chunk_stream_count != count:
                raise ValueError("capture-frame stream total differs")
            raw_message_count += count
        return raw_message_count, dict(sorted(stream_counts.items()))

    def raw_message_lane_summaries(
        self,
        run_id: str,
        *,
        streams: Sequence[str] | None = None,
    ) -> tuple[RawMessageLaneSummary, ...]:
        """Return exact per-connection receipt bounds without exposing storage layout."""

        selected_streams: tuple[str, ...] | None = None
        if streams is not None:
            selected_streams = tuple(sorted({str(value) for value in streams}))
            if not selected_streams or any(
                value not in _STREAMS for value in selected_streams
            ):
                raise ValueError("raw-message stream filter is invalid")
        storage_version = self._storage_schema_version(run_id)
        if storage_version == POLYMARKET_STORAGE_SCHEMA_VERSION:
            audit = self._evidence_audit_summary.get(run_id)
            if audit is None:
                accumulators: dict[tuple[str, str], _RawMessageLaneAccumulator] = {}
                for stored in self._iter_capture_messages(run_id):
                    message = stored.message
                    key = (message.stream, message.connection_id)
                    accumulator = accumulators.get(key)
                    if accumulator is None:
                        accumulators[key] = _RawMessageLaneAccumulator.from_message(
                            message
                        )
                    else:
                        accumulator.update(message)
                summaries = _freeze_lane_summaries(accumulators)
            else:
                summaries = audit.lane_summaries
            if selected_streams is None:
                return summaries
            return tuple(
                summary for summary in summaries if summary.stream in selected_streams
            )
        parameters: list[object] = [run_id]
        stream_filter = ""
        if selected_streams is not None:
            placeholders = ", ".join("?" for _ in selected_streams)
            stream_filter = f" AND stream IN ({placeholders})"
            parameters.extend(selected_streams)
        rows = (
            self.connect()
            .execute(
                f"""
            SELECT stream, connection_id, count(*), min(sequence_number),
                   max(sequence_number), min(received_wall_ms),
                   max(received_wall_ms), min(received_monotonic_ns),
                   max(received_monotonic_ns)
            FROM polymarket_raw_message
            WHERE run_id = ?{stream_filter}
            GROUP BY stream, connection_id
            ORDER BY stream, connection_id
            """,
                parameters,
            )
            .fetchall()
        )
        return tuple(
            RawMessageLaneSummary(
                stream=str(row[0]),
                connection_id=str(row[1]),
                message_count=int(row[2]),
                minimum_sequence_number=int(row[3]),
                maximum_sequence_number=int(row[4]),
                first_received_wall_ms=int(row[5]),
                last_received_wall_ms=int(row[6]),
                first_received_monotonic_ns=int(row[7]),
                last_received_monotonic_ns=int(row[8]),
            )
            for row in rows
        )

    def _decode_compact_raw_text(
        self,
        *,
        run_id: str,
        chunk_id: object,
        raw_offset: object,
        raw_size: object,
        raw_payload_sha256: object,
        verify_payload_hash: bool = True,
    ) -> str:
        normalized_chunk_id = str(chunk_id or "")
        offset = int(raw_offset)
        size = int(raw_size)
        if (
            not normalized_chunk_id
            or offset < 4
            or not 0 <= size <= _MAX_RAW_MESSAGE_BYTES
        ):
            raise ValueError("compact raw-message location is invalid")
        frame = self._load_compact_frame(run_id, normalized_chunk_id)
        end = offset + size
        if end > len(frame) or struct.unpack_from("<I", frame, offset - 4)[0] != size:
            raise ValueError("compact raw-message frame boundary is invalid")
        raw_bytes = frame[offset:end]
        if verify_payload_hash and hashlib.sha256(raw_bytes).hexdigest() != str(
            raw_payload_sha256
        ):
            raise ValueError("compact raw-message payload hash mismatch")
        try:
            return raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("compact raw-message payload is not UTF-8") from exc

    @staticmethod
    def _condition_cache_manifest_payload(
        *,
        run_id: str,
        condition_id: str,
        source_run_report_sha256: str,
        frame_count: int,
        message_count: int,
        first_received_monotonic_ns: int,
        last_received_monotonic_ns: int,
        last_frame_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
            "run_id": run_id,
            "condition_id": condition_id,
            "source_run_report_sha256": source_run_report_sha256,
            "frame_count": frame_count,
            "message_count": message_count,
            "first_received_monotonic_ns": first_received_monotonic_ns,
            "last_received_monotonic_ns": last_received_monotonic_ns,
            "last_frame_sha256": last_frame_sha256,
        }

    def _validate_condition_message_cache(
        self,
        run_id: str,
        source_run_report_sha256: str,
    ) -> dict[str, object]:
        connection = self.connect()
        build = connection.execute(
            """
            SELECT schema_version, source_run_report_sha256, state,
                   condition_count, frame_count, message_count,
                   report_json, report_sha256, error
            FROM polymarket_condition_cache_build WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if build is None or str(build[2]) != "complete":
            raise ValueError("Polymarket condition cache is not complete")
        if (
            str(build[0]) != _CONDITION_CACHE_SCHEMA_VERSION
            or str(build[1]) != source_run_report_sha256
            or str(build[8])
        ):
            raise ValueError("Polymarket condition cache build identity is invalid")
        try:
            report = _strict_json_loads(str(build[6]))
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            raise ValueError("Polymarket condition cache report is invalid") from exc
        if (
            not isinstance(report, Mapping)
            or _canonical_json(report) != str(build[6])
            or _canonical_sha256(report) != str(build[7])
        ):
            raise ValueError("Polymarket condition cache report hash differs")
        manifest_rows = connection.execute(
            """
            SELECT condition_id, schema_version, source_run_report_sha256,
                   frame_count, message_count, first_received_monotonic_ns,
                   last_received_monotonic_ns, last_frame_sha256,
                   manifest_sha256
            FROM polymarket_condition_message_manifest
            WHERE run_id = ? ORDER BY condition_id
            """,
            [run_id],
        ).fetchall()
        if not manifest_rows:
            raise ValueError("Polymarket condition cache has no manifests")
        manifests: list[dict[str, str]] = []
        manifest_frame_count = 0
        manifest_message_count = 0
        for row in manifest_rows:
            frame_count = int(row[3])
            message_count = int(row[4])
            first_received_monotonic_ns = int(row[5])
            last_received_monotonic_ns = int(row[6])
            last_frame_sha256 = str(row[7])
            payload = self._condition_cache_manifest_payload(
                run_id=run_id,
                condition_id=str(row[0]),
                source_run_report_sha256=str(row[2]),
                frame_count=frame_count,
                message_count=message_count,
                first_received_monotonic_ns=first_received_monotonic_ns,
                last_received_monotonic_ns=last_received_monotonic_ns,
                last_frame_sha256=last_frame_sha256,
            )
            if (
                str(row[1]) != _CONDITION_CACHE_SCHEMA_VERSION
                or str(row[2]) != source_run_report_sha256
                or _canonical_sha256(payload) != str(row[8])
            ):
                raise ValueError("Polymarket condition cache manifest differs")
            if frame_count == 0:
                if (
                    message_count != 0
                    or first_received_monotonic_ns != 0
                    or last_received_monotonic_ns != 0
                    or last_frame_sha256
                ):
                    raise ValueError(
                        "Polymarket empty condition cache manifest differs"
                    )
            elif (
                message_count <= 0
                or first_received_monotonic_ns < 0
                or last_received_monotonic_ns < first_received_monotonic_ns
                or not re.fullmatch(r"[0-9a-f]{64}", last_frame_sha256)
            ):
                raise ValueError("Polymarket condition cache manifest values differ")
            manifest_frame_count += frame_count
            manifest_message_count += message_count
            manifests.append(
                {
                    "condition_id": str(row[0]),
                    "manifest_sha256": str(row[8]),
                }
            )
        aggregate = connection.execute(
            """
            SELECT count(*), coalesce(sum(message_count), 0)
            FROM polymarket_condition_message_frame WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if manifest_frame_count != int(aggregate[0]) or manifest_message_count != int(
            aggregate[1]
        ):
            raise ValueError("Polymarket condition cache frame totals differ")
        expected = {
            "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
            "run_id": run_id,
            "source_run_report_sha256": source_run_report_sha256,
            "condition_count": len(manifests),
            "frame_count": int(aggregate[0]),
            "message_count": int(aggregate[1]),
            "manifests": manifests,
        }
        if (
            expected != dict(report)
            or int(build[3]) != len(manifests)
            or int(build[4]) != int(aggregate[0])
            or int(build[5]) != int(aggregate[1])
        ):
            raise ValueError("Polymarket condition cache summary differs")
        self._validated_condition_cache_runs[run_id] = str(build[7])
        return expected

    def _prune_condition_message_cache(
        self,
        run_id: str,
        source_run_report_sha256: str,
        condition_ids: tuple[str, ...],
    ) -> dict[str, object]:
        """Atomically narrow a verified cache without re-reading capture frames."""

        if self.read_only:
            raise ValueError("read-only Polymarket condition cache cannot be pruned")
        connection = self.connect()
        placeholders = ", ".join("?" for _ in condition_ids)
        parameters: list[object] = [run_id, *condition_ids]
        manifest_rows = connection.execute(
            f"""
            SELECT condition_id, manifest_sha256, frame_count, message_count
            FROM polymarket_condition_message_manifest
            WHERE run_id = ? AND condition_id IN ({placeholders})
            ORDER BY condition_id
            """,
            parameters,
        ).fetchall()
        if tuple(str(row[0]) for row in manifest_rows) != condition_ids:
            raise ValueError("Polymarket condition cache cannot cover the selection")
        frame_count = sum(int(row[2]) for row in manifest_rows)
        message_count = sum(int(row[3]) for row in manifest_rows)
        report = {
            "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
            "run_id": run_id,
            "source_run_report_sha256": source_run_report_sha256,
            "condition_count": len(condition_ids),
            "frame_count": frame_count,
            "message_count": message_count,
            "manifests": [
                {
                    "condition_id": str(row[0]),
                    "manifest_sha256": str(row[1]),
                }
                for row in manifest_rows
            ],
        }
        report_json = _canonical_json(report)
        report_sha256 = _canonical_sha256(report)
        self._validated_condition_cache_runs.pop(run_id, None)
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                f"""
                DELETE FROM polymarket_condition_message_frame
                WHERE run_id = ? AND condition_id NOT IN ({placeholders})
                """,
                parameters,
            )
            connection.execute(
                f"""
                DELETE FROM polymarket_condition_message_manifest
                WHERE run_id = ? AND condition_id NOT IN ({placeholders})
                """,
                parameters,
            )
            updated = connection.execute(
                """
                UPDATE polymarket_condition_cache_build
                SET condition_count = ?, frame_count = ?, message_count = ?,
                    report_json = ?, report_sha256 = ?, error = ''
                WHERE run_id = ? AND state = 'complete'
                  AND source_run_report_sha256 = ?
                RETURNING run_id
                """,
                [
                    len(condition_ids),
                    frame_count,
                    message_count,
                    report_json,
                    report_sha256,
                    run_id,
                    source_run_report_sha256,
                ],
            ).fetchall()
            if updated != [(run_id,)]:
                raise ValueError("Polymarket condition cache build changed while pruning")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        return self._validate_condition_message_cache(
            run_id,
            source_run_report_sha256,
        )

    def ensure_condition_message_cache(
        self,
        run_id: str,
        *,
        condition_ids: Sequence[str] | None = None,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> dict[str, object]:
        """Build an exact, compressed condition lookup from audited chunks."""

        selected = str(run_id or "").strip()
        cached_audit = self._terminal_evidence_integrity_cache.get(selected)
        if (
            not selected
            or self._storage_schema_version(selected)
            not in {
                _RAW_RECONSTRUCTED_STORAGE_SCHEMA_VERSION,
                POLYMARKET_STORAGE_SCHEMA_VERSION,
            }
            or cached_audit is None
            or cached_audit[0] != self._terminal_evidence_fingerprint(selected)
            or cached_audit[1]
        ):
            raise ValueError(
                "Polymarket condition cache requires a clean current chunk audit"
            )
        connection = self.connect()
        run = connection.execute(
            """
            SELECT status, error, report_sha256 FROM polymarket_recorder_run
            WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if (
            run is None
            or str(run[0]) not in {"complete", "degraded"}
            or str(run[1] or "")
            or not re.fullmatch(r"[0-9a-f]{64}", str(run[2]))
        ):
            raise ValueError("Polymarket condition cache source run is not terminal")
        source_report_sha256 = str(run[2])
        market_conditions = tuple(
            str(row[0]).strip().lower()
            for row in connection.execute(
                """
                SELECT condition_id FROM polymarket_market_snapshot
                WHERE run_id = ? ORDER BY condition_id
                """,
                [selected],
            ).fetchall()
        )
        if (
            not market_conditions
            or any(not condition_id for condition_id in market_conditions)
            or len(set(market_conditions)) != len(market_conditions)
        ):
            raise ValueError("Polymarket condition cache market coverage is invalid")
        requested_conditions = (
            market_conditions
            if condition_ids is None
            else tuple(
                sorted(
                    {
                        str(condition_id or "").strip().lower()
                        for condition_id in condition_ids
                    }
                )
            )
        )
        if (
            not requested_conditions
            or "" in requested_conditions
            or not set(requested_conditions).issubset(market_conditions)
        ):
            raise ValueError("Polymarket condition cache selection is invalid")
        existing = connection.execute(
            """
            SELECT state, source_run_report_sha256
            FROM polymarket_condition_cache_build WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if existing is not None and str(existing[0]) == "complete":
            if str(existing[1]) != source_report_sha256:
                raise ValueError("Polymarket condition cache source identity differs")
            report = self._validate_condition_message_cache(
                selected,
                source_report_sha256,
            )
            cached_conditions = tuple(
                str(item["condition_id"])
                for item in report["manifests"]
                if isinstance(item, Mapping)
            )
            if set(requested_conditions).issubset(cached_conditions):
                if cached_conditions == requested_conditions or self.read_only:
                    return report
                if progress is not None:
                    progress(
                        "condition-cache-prune",
                        {
                            "cached_condition_count": len(cached_conditions),
                            "selected_condition_count": len(requested_conditions),
                        },
                    )
                return self._prune_condition_message_cache(
                    selected,
                    source_report_sha256,
                    requested_conditions,
                )
        if self.read_only:
            raise ValueError(
                "read-only Polymarket store has no complete condition cache"
            )

        self._validated_condition_cache_runs.pop(selected, None)
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "DELETE FROM polymarket_condition_message_frame WHERE run_id = ?",
                [selected],
            )
            connection.execute(
                "DELETE FROM polymarket_condition_message_manifest WHERE run_id = ?",
                [selected],
            )
            connection.execute(
                "DELETE FROM polymarket_condition_cache_build WHERE run_id = ?",
                [selected],
            )
            connection.execute(
                """
                INSERT INTO polymarket_condition_cache_build VALUES (
                    ?, ?, ?, 'started', 0, 0, 0, '', '', ''
                )
                """,
                [selected, _CONDITION_CACHE_SCHEMA_VERSION, source_report_sha256],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

        started = time.monotonic()
        last_progress = started
        scanned_messages = 0
        cached_message_references = 0
        buffers: dict[str, list[tuple[object, ...]]] = {}
        frame_counts: dict[str, int] = {
            condition_id: 0 for condition_id in requested_conditions
        }
        message_counts: dict[str, int] = {
            condition_id: 0 for condition_id in requested_conditions
        }
        allowed_conditions = frozenset(requested_conditions)
        first_received: dict[str, int] = {}
        last_received: dict[str, int] = {}
        last_frame_sha256: dict[str, str] = {}
        pending_frames: list[tuple[object, ...]] = []
        compressor = zstandard.ZstdCompressor(level=_RAW_CHUNK_COMPRESSION_LEVEL)

        def mark_failed(exc: BaseException) -> None:
            self._validated_condition_cache_runs.pop(selected, None)
            try:
                connection.execute(
                    """
                    UPDATE polymarket_condition_cache_build
                    SET state = 'failed', error = ?
                    WHERE run_id = ? AND state = 'started'
                    """,
                    [f"{type(exc).__name__}:{exc}"[:2_000], selected],
                )
            except Exception:
                return

        writer: duckdb.DuckDBPyConnection | None = None
        try:
            writer = duckdb.connect(str(self.path))
            writer.execute(f"SET memory_limit='{self.memory_limit}'")
            writer.execute(f"SET threads={self.threads}")
            writer.execute("SET TimeZone='UTC'")
            writer.execute("SET preserve_insertion_order=false")
        except Exception as exc:
            if writer is not None:
                writer.close()
            mark_failed(exc)
            raise
        if writer is None:
            raise RuntimeError("Polymarket condition cache writer did not initialize")
        cache_writer = writer

        def notify(*, force: bool = False) -> None:
            nonlocal last_progress
            if progress is None:
                return
            now = time.monotonic()
            if not force and now - last_progress < 30.0:
                return
            last_progress = now
            try:
                progress(
                    "condition-cache",
                    {
                        "elapsed_seconds": round(now - started, 3),
                        "scanned_message_count": scanned_messages,
                        "cached_message_reference_count": cached_message_references,
                        "condition_count": len(frame_counts),
                        "frame_count": sum(frame_counts.values()),
                    },
                )
            except Exception:
                return

        def flush_pending_frames() -> None:
            if not pending_frames:
                return
            cache_writer.execute("BEGIN TRANSACTION")
            try:
                cache_writer.executemany(
                    """
                    INSERT INTO polymarket_condition_message_frame VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    pending_frames,
                )
                cache_writer.execute("COMMIT")
            except Exception:
                cache_writer.execute("ROLLBACK")
                raise
            pending_frames.clear()

        def flush_condition(condition_id: str) -> None:
            rows = buffers.get(condition_id)
            if not rows:
                return
            frame_index = frame_counts.get(condition_id, 0)
            payload = {
                "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
                "condition_id": condition_id,
                "frame_index": frame_index,
                "rows": [list(row) for row in rows],
            }
            encoded = _canonical_json(payload).encode("ascii")
            if not 1 <= len(encoded) <= _CONDITION_CACHE_MAX_FRAME_BYTES:
                raise ValueError("Polymarket condition cache frame is oversized")
            compressed = compressor.compress(encoded)
            uncompressed_sha256 = hashlib.sha256(encoded).hexdigest()
            compressed_sha256 = hashlib.sha256(compressed).hexdigest()
            previous = last_frame_sha256.get(condition_id, "")
            frame_identity = {
                "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
                "run_id": selected,
                "condition_id": condition_id,
                "frame_index": frame_index,
                "previous_frame_sha256": previous,
                "message_count": len(rows),
                "first_received_monotonic_ns": int(rows[0][5]),
                "last_received_monotonic_ns": int(rows[-1][5]),
                "uncompressed_bytes": len(encoded),
                "uncompressed_sha256": uncompressed_sha256,
                "compressed_bytes": len(compressed),
                "compressed_sha256": compressed_sha256,
            }
            frame_sha256 = _canonical_sha256(frame_identity)
            pending_frames.append(
                (
                    selected,
                    condition_id,
                    frame_index,
                    _CONDITION_CACHE_SCHEMA_VERSION,
                    previous,
                    len(rows),
                    int(rows[0][5]),
                    int(rows[-1][5]),
                    len(encoded),
                    uncompressed_sha256,
                    len(compressed),
                    compressed_sha256,
                    compressed,
                    frame_sha256,
                )
            )
            frame_counts[condition_id] = frame_index + 1
            message_counts[condition_id] = message_counts.get(condition_id, 0) + len(
                rows
            )
            first_received.setdefault(condition_id, int(rows[0][5]))
            last_received[condition_id] = int(rows[-1][5])
            last_frame_sha256[condition_id] = frame_sha256
            rows.clear()
            if len(pending_frames) >= 64:
                flush_pending_frames()

        def cache_source_message(
            cache_row: tuple[object, ...],
            candidates: Sequence[Mapping[str, object]],
        ) -> None:
            nonlocal cached_message_references
            conditions = {
                str(_event_index(str(cache_row[1]), item)["condition_id"])
                for item in candidates
            }
            conditions.discard("")
            conditions.intersection_update(allowed_conditions)
            if not conditions:
                return
            for condition_id in sorted(conditions):
                buffer = buffers.setdefault(condition_id, [])
                buffer.append(cache_row)
                cached_message_references += 1
                if len(buffer) >= _CONDITION_CACHE_FRAME_MESSAGE_LIMIT:
                    flush_condition(condition_id)

        notify(force=True)
        try:
            if (
                self._storage_schema_version(selected)
                == POLYMARKET_STORAGE_SCHEMA_VERSION
            ):
                for stored in self._iter_capture_messages(
                    selected,
                    streams=("clob_market", "clob_rest_book"),
                ):
                    scanned_messages += 1
                    parse_status, _parse_error, candidates = _stream_message_events(
                        stored.message.raw_text
                    )
                    if parse_status == "invalid":
                        raise ValueError(
                            "Polymarket condition cache source payload is invalid"
                        )
                    if candidates:
                        cache_source_message(
                            stored.cache_metadata(len(candidates)),
                            candidates,
                        )
                    if scanned_messages % _INTEGRITY_FETCH_SIZE == 0:
                        notify()
            else:
                cursor = connection.execute(
                    """
                    SELECT message_id, stream, connection_id, sequence_number,
                           received_wall_ms, received_monotonic_ns,
                           raw_payload_sha256, raw_text, parse_status, parse_error,
                           storage_chunk_id, raw_offset, raw_size,
                           normalized_event_count
                    FROM polymarket_raw_message
                    WHERE run_id = ?
                      AND stream IN ('clob_market', 'clob_rest_book')
                      AND normalized_event_count > 0
                    ORDER BY received_monotonic_ns, received_wall_ms,
                             connection_id, sequence_number
                    """,
                    [selected],
                )
                for batch in iter(lambda: cursor.fetchmany(_INTEGRITY_FETCH_SIZE), []):
                    for row in batch:
                        scanned_messages += 1
                        event_count = int(row[13])
                        if (
                            event_count <= 0
                            or str(row[7])
                            or str(row[8]) != "ok"
                            or str(row[9])
                        ):
                            raise ValueError(
                                "Polymarket condition cache source metadata is invalid"
                            )
                        raw_text = self._decode_compact_raw_text(
                            run_id=selected,
                            chunk_id=row[10],
                            raw_offset=row[11],
                            raw_size=row[12],
                            raw_payload_sha256=row[6],
                            verify_payload_hash=False,
                        )
                        parse_status, _parse_error, candidates = _stream_message_events(
                            raw_text
                        )
                        if parse_status != "ok" or len(candidates) != event_count:
                            raise ValueError(
                                "Polymarket condition cache normalized-event count differs"
                            )
                        cache_source_message(
                            (
                                str(row[0]),
                                str(row[1]),
                                str(row[2]),
                                int(row[3]),
                                int(row[4]),
                                int(row[5]),
                                str(row[6]),
                                str(row[10]),
                                int(row[11]),
                                int(row[12]),
                                event_count,
                            ),
                            candidates,
                        )
                        if scanned_messages % _INTEGRITY_FETCH_SIZE == 0:
                            notify()
            for condition_id in sorted(buffers):
                flush_condition(condition_id)
            flush_pending_frames()

            manifest_rows: list[tuple[object, ...]] = []
            manifest_links: list[dict[str, str]] = []
            for condition_id in sorted(frame_counts):
                payload = self._condition_cache_manifest_payload(
                    run_id=selected,
                    condition_id=condition_id,
                    source_run_report_sha256=source_report_sha256,
                    frame_count=frame_counts[condition_id],
                    message_count=message_counts[condition_id],
                    first_received_monotonic_ns=first_received.get(condition_id, 0),
                    last_received_monotonic_ns=last_received.get(condition_id, 0),
                    last_frame_sha256=last_frame_sha256.get(condition_id, ""),
                )
                manifest_sha256 = _canonical_sha256(payload)
                manifest_rows.append(
                    (
                        selected,
                        condition_id,
                        _CONDITION_CACHE_SCHEMA_VERSION,
                        source_report_sha256,
                        frame_counts[condition_id],
                        message_counts[condition_id],
                        first_received.get(condition_id, 0),
                        last_received.get(condition_id, 0),
                        last_frame_sha256.get(condition_id, ""),
                        manifest_sha256,
                    )
                )
                manifest_links.append(
                    {
                        "condition_id": condition_id,
                        "manifest_sha256": manifest_sha256,
                    }
                )
            report = {
                "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
                "run_id": selected,
                "source_run_report_sha256": source_report_sha256,
                "condition_count": len(manifest_rows),
                "frame_count": sum(frame_counts.values()),
                "message_count": sum(message_counts.values()),
                "manifests": manifest_links,
            }
            report_json = _canonical_json(report)
            report_sha256 = _canonical_sha256(report)
            cache_writer.execute("BEGIN TRANSACTION")
            try:
                cache_writer.executemany(
                    """
                    INSERT INTO polymarket_condition_message_manifest VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    manifest_rows,
                )
                cache_writer.execute(
                    """
                    UPDATE polymarket_condition_cache_build
                    SET state = 'complete', condition_count = ?, frame_count = ?,
                        message_count = ?, report_json = ?, report_sha256 = ?,
                        error = ''
                    WHERE run_id = ? AND state = 'started'
                    """,
                    [
                        len(manifest_rows),
                        sum(frame_counts.values()),
                        sum(message_counts.values()),
                        report_json,
                        report_sha256,
                        selected,
                    ],
                )
                cache_writer.execute("COMMIT")
            except Exception:
                cache_writer.execute("ROLLBACK")
                raise
        except Exception as exc:
            mark_failed(exc)
            raise
        finally:
            cache_writer.close()
        notify(force=True)
        return self._validate_condition_message_cache(
            selected,
            source_report_sha256,
        )

    def _decode_verified_chunk_message(
        self,
        run_id: str,
        metadata: Sequence[object],
        *,
        selected_conditions: tuple[str, ...] | None,
        candidates: Sequence[Mapping[str, object]] | None = None,
    ) -> Iterator[DecodedPublicEvent]:
        if len(metadata) != 11:
            raise ValueError("verified chunk-message metadata width is invalid")
        (
            message_id,
            stream,
            connection_id,
            sequence_number,
            received_wall_ms,
            received_monotonic_ns,
            raw_payload_sha256,
            chunk_id,
            raw_offset,
            raw_size,
            normalized_event_count,
        ) = metadata
        event_count = int(normalized_event_count)
        if event_count <= 0:
            return
        selected_events = candidates
        if selected_events is None:
            raw_text = self._decode_compact_raw_text(
                run_id=run_id,
                chunk_id=chunk_id,
                raw_offset=raw_offset,
                raw_size=raw_size,
                raw_payload_sha256=raw_payload_sha256,
                verify_payload_hash=False,
            )
            parse_status, _parse_error, parsed_events = _stream_message_events(raw_text)
            if parse_status != "ok":
                raise ValueError("verified chunk message cannot be reconstructed")
            selected_events = parsed_events
        if len(selected_events) != event_count:
            raise ValueError("verified chunk normalized-event count drifted")
        normalized_message_id = str(message_id)
        normalized_stream = str(stream)
        for sub_index, raw_event in enumerate(selected_events):
            if not isinstance(raw_event, Mapping):
                raise ValueError("verified v3 event is not an object")
            event = dict(raw_event)
            normalized = _event_index(normalized_stream, event)
            condition_id = str(normalized["condition_id"])
            if (
                selected_conditions is not None
                and condition_id not in selected_conditions
            ):
                continue
            event_json = _canonical_json(event)
            event_sha256 = hashlib.sha256(event_json.encode("ascii")).hexdigest()
            event_id = _canonical_sha256(
                {
                    "message_id": normalized_message_id,
                    "sub_index": sub_index,
                    "event_sha256": event_sha256,
                }
            )
            yield DecodedPublicEvent(
                event_id=event_id,
                run_id=run_id,
                message_id=normalized_message_id,
                sub_index=sub_index,
                stream=normalized_stream,
                event_type=str(normalized["event_type"]),
                symbol=str(normalized["symbol"]),
                condition_id=condition_id,
                asset_id=str(normalized["asset_id"]),
                source_time_ms=(
                    None
                    if normalized["source_time_ms"] is None
                    else int(normalized["source_time_ms"])
                ),
                publisher_time_ms=(
                    None
                    if normalized["publisher_time_ms"] is None
                    else int(normalized["publisher_time_ms"])
                ),
                event_sha256=event_sha256,
                event=event,
                connection_id=str(connection_id),
                sequence_number=int(sequence_number),
                received_wall_ms=int(received_wall_ms),
                received_monotonic_ns=int(received_monotonic_ns),
            )

    def _condition_message_cache_available(
        self,
        run_id: str,
        selected_conditions: tuple[str, ...],
    ) -> bool:
        connection = self.connect()
        build = connection.execute(
            """
            SELECT state, source_run_report_sha256, report_sha256
            FROM polymarket_condition_cache_build WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if build is None:
            return False
        if str(build[0]) != "complete":
            raise ValueError("Polymarket condition cache build is incomplete")
        if self._validated_condition_cache_runs.get(run_id) != str(build[2]):
            self._validate_condition_message_cache(run_id, str(build[1]))
        placeholders = ", ".join("?" for _ in selected_conditions)
        count = connection.execute(
            f"""
            SELECT count(*) FROM polymarket_condition_message_manifest
            WHERE run_id = ? AND condition_id IN ({placeholders})
            """,
            [run_id, *selected_conditions],
        ).fetchone()[0]
        if int(count) != len(selected_conditions):
            return False
        return True

    def _iter_verified_condition_cache_events(
        self,
        run_id: str,
        *,
        selected_streams: tuple[str, ...] | None,
        selected_conditions: tuple[str, ...],
        ordered: bool,
    ) -> Iterator[DecodedPublicEvent]:
        """Replay selected conditions from integrity-checked compressed references."""

        placeholders = ", ".join("?" for _ in selected_conditions)
        parameters: list[object] = [run_id, *selected_conditions]
        connection = self.connect()
        manifest_rows = connection.execute(
            f"""
            SELECT condition_id, source_run_report_sha256, frame_count,
                   message_count, first_received_monotonic_ns,
                   last_received_monotonic_ns, last_frame_sha256,
                   manifest_sha256
            FROM polymarket_condition_message_manifest
            WHERE run_id = ? AND condition_id IN ({placeholders})
            ORDER BY condition_id
            """,
            parameters,
        ).fetchall()
        manifests = {str(row[0]): row for row in manifest_rows}
        if set(manifests) != set(selected_conditions):
            raise ValueError("Polymarket condition cache manifest selection differs")
        frame_rows = connection.execute(
            f"""
            SELECT condition_id, frame_index, schema_version,
                   previous_frame_sha256, message_count,
                   first_received_monotonic_ns, last_received_monotonic_ns,
                   uncompressed_bytes, uncompressed_sha256,
                   compressed_bytes, compressed_sha256, compressed_payload,
                   frame_sha256
            FROM polymarket_condition_message_frame
            WHERE run_id = ? AND condition_id IN ({placeholders})
            ORDER BY condition_id, frame_index
            """,
            parameters,
        ).fetchall()
        metadata_by_message: dict[str, tuple[object, ...]] = {}
        conditions_by_message: dict[str, set[str]] = {}
        observed: dict[str, dict[str, object]] = {}
        for row in frame_rows:
            condition_id = str(row[0])
            state = observed.setdefault(
                condition_id,
                {
                    "frame_count": 0,
                    "message_count": 0,
                    "first_received_monotonic_ns": None,
                    "last_received_monotonic_ns": None,
                    "last_frame_sha256": "",
                    "last_order_key": None,
                },
            )
            frame_index = int(row[1])
            compressed = bytes(row[11])
            if (
                str(row[2]) != _CONDITION_CACHE_SCHEMA_VERSION
                or frame_index != int(state["frame_count"])
                or str(row[3]) != str(state["last_frame_sha256"])
                or not 1 <= int(row[7]) <= _CONDITION_CACHE_MAX_FRAME_BYTES
                or int(row[9]) <= 0
                or len(compressed) != int(row[9])
                or hashlib.sha256(compressed).hexdigest() != str(row[10])
            ):
                raise ValueError("Polymarket condition cache frame identity differs")
            try:
                encoded = self._decompressor.decompress(
                    compressed,
                    max_output_size=_CONDITION_CACHE_MAX_FRAME_BYTES,
                )
            except (MemoryError, zstandard.ZstdError) as exc:
                raise ValueError(
                    "Polymarket condition cache frame cannot be decoded"
                ) from exc
            if len(encoded) != int(row[7]) or hashlib.sha256(
                encoded
            ).hexdigest() != str(row[8]):
                raise ValueError("Polymarket condition cache frame payload differs")
            try:
                payload = _strict_json_loads(encoded.decode("ascii"))
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    "Polymarket condition cache frame JSON is invalid"
                ) from exc
            if (
                not isinstance(payload, Mapping)
                or _canonical_json(payload).encode("ascii") != encoded
                or payload.get("schema_version") != _CONDITION_CACHE_SCHEMA_VERSION
                or payload.get("condition_id") != condition_id
                or payload.get("frame_index") != frame_index
                or not isinstance(payload.get("rows"), list)
                or len(payload["rows"]) != int(row[4])
            ):
                raise ValueError("Polymarket condition cache frame contract differs")
            decoded_rows: list[tuple[object, ...]] = []
            for item in payload["rows"]:
                if not isinstance(item, list) or len(item) != 11:
                    raise ValueError(
                        "Polymarket condition cache message row is invalid"
                    )
                try:
                    metadata = (
                        str(item[0]),
                        str(item[1]),
                        str(item[2]),
                        int(item[3]),
                        int(item[4]),
                        int(item[5]),
                        str(item[6]),
                        str(item[7]),
                        int(item[8]),
                        int(item[9]),
                        int(item[10]),
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "Polymarket condition cache message row is invalid"
                    ) from exc
                if (
                    not metadata[0]
                    or metadata[1] not in {"clob_market", "clob_rest_book"}
                    or not metadata[2]
                    or not re.fullmatch(r"[0-9a-f]{64}", str(metadata[6]))
                    or not re.fullmatch(r"[0-9a-f]{64}", str(metadata[7]))
                    or any(int(metadata[index]) < 0 for index in (3, 4, 5, 8))
                    or int(metadata[9]) <= 0
                    or int(metadata[10]) <= 0
                ):
                    raise ValueError("Polymarket condition cache message values differ")
                decoded_rows.append(metadata)
                existing = metadata_by_message.get(str(metadata[0]))
                if existing is not None and existing != metadata:
                    raise ValueError(
                        "Polymarket condition cache duplicated a message differently"
                    )
                metadata_by_message[str(metadata[0])] = metadata
                conditions_by_message.setdefault(str(metadata[0]), set()).add(
                    condition_id
                )
            order_keys = [
                (
                    int(metadata[5]),
                    int(metadata[4]),
                    str(metadata[2]),
                    int(metadata[3]),
                    str(metadata[0]),
                )
                for metadata in decoded_rows
            ]
            if (
                not decoded_rows
                or int(decoded_rows[0][5]) != int(row[5])
                or int(decoded_rows[-1][5]) != int(row[6])
                or int(row[5]) > int(row[6])
                or any(
                    previous > current
                    for previous, current in zip(order_keys, order_keys[1:])
                )
                or (
                    state["last_order_key"] is not None
                    and state["last_order_key"] > order_keys[0]
                )
            ):
                raise ValueError("Polymarket condition cache frame order differs")
            frame_identity = {
                "schema_version": _CONDITION_CACHE_SCHEMA_VERSION,
                "run_id": run_id,
                "condition_id": condition_id,
                "frame_index": frame_index,
                "previous_frame_sha256": str(row[3]),
                "message_count": int(row[4]),
                "first_received_monotonic_ns": int(row[5]),
                "last_received_monotonic_ns": int(row[6]),
                "uncompressed_bytes": int(row[7]),
                "uncompressed_sha256": str(row[8]),
                "compressed_bytes": int(row[9]),
                "compressed_sha256": str(row[10]),
            }
            if _canonical_sha256(frame_identity) != str(row[12]):
                raise ValueError("Polymarket condition cache frame hash differs")
            state["frame_count"] = int(state["frame_count"]) + 1
            state["message_count"] = int(state["message_count"]) + len(decoded_rows)
            if state["first_received_monotonic_ns"] is None:
                state["first_received_monotonic_ns"] = int(row[5])
            state["last_received_monotonic_ns"] = int(row[6])
            state["last_frame_sha256"] = str(row[12])
            state["last_order_key"] = order_keys[-1]
        for condition_id in selected_conditions:
            manifest = manifests[condition_id]
            state = observed.get(condition_id)
            if state is None:
                if (
                    int(manifest[2]) != 0
                    or int(manifest[3]) != 0
                    or int(manifest[4]) != 0
                    or int(manifest[5]) != 0
                    or str(manifest[6])
                ):
                    raise ValueError(
                        "Polymarket condition cache contains no selected frames"
                    )
                state = {
                    "frame_count": 0,
                    "message_count": 0,
                    "first_received_monotonic_ns": 0,
                    "last_received_monotonic_ns": 0,
                    "last_frame_sha256": "",
                }
            payload = self._condition_cache_manifest_payload(
                run_id=run_id,
                condition_id=condition_id,
                source_run_report_sha256=str(manifest[1]),
                frame_count=int(manifest[2]),
                message_count=int(manifest[3]),
                first_received_monotonic_ns=int(manifest[4]),
                last_received_monotonic_ns=int(manifest[5]),
                last_frame_sha256=str(manifest[6]),
            )
            if (
                int(state["frame_count"]) != int(manifest[2])
                or int(state["message_count"]) != int(manifest[3])
                or int(state["first_received_monotonic_ns"]) != int(manifest[4])
                or int(state["last_received_monotonic_ns"]) != int(manifest[5])
                or str(state["last_frame_sha256"]) != str(manifest[6])
                or _canonical_sha256(payload) != str(manifest[7])
            ):
                raise ValueError("Polymarket condition cache manifest replay differs")
        metadata_rows = list(metadata_by_message.values())
        if ordered:
            metadata_rows.sort(
                key=lambda item: (
                    int(item[5]),
                    int(item[4]),
                    str(item[2]),
                    int(item[3]),
                    str(item[0]),
                )
            )
        for metadata in metadata_rows:
            decoded = tuple(
                self._decode_verified_chunk_message(
                    run_id,
                    metadata,
                    selected_conditions=selected_conditions,
                )
            )
            decoded_conditions = {event.condition_id for event in decoded}
            if not conditions_by_message[str(metadata[0])].issubset(decoded_conditions):
                raise ValueError("Polymarket condition cache reference differs")
            if selected_streams is None or str(metadata[1]) in selected_streams:
                yield from decoded

    def _iter_verified_v3_raw_events(
        self,
        run_id: str,
        *,
        selected_streams: tuple[str, ...] | None,
        selected_conditions: tuple[str, ...] | None,
        ordered: bool,
    ) -> Iterator[DecodedPublicEvent]:
        """Replay audited v3 source chunks without joining the redundant event index."""

        if (
            self._storage_schema_version(run_id)
            != _RAW_RECONSTRUCTED_STORAGE_SCHEMA_VERSION
        ):
            raise ValueError("raw-only public-event replay requires storage v3")
        filters = ["run_id = ?"]
        parameters: list[object] = [run_id]
        if selected_streams is not None:
            placeholders = ", ".join("?" for _ in selected_streams)
            filters.append(f"stream IN ({placeholders})")
            parameters.extend(selected_streams)
        order_clause = ""
        if ordered:
            order_clause = """
                ORDER BY received_monotonic_ns, received_wall_ms,
                         connection_id, sequence_number
            """
        cursor = self.connect().execute(
            f"""
            SELECT message_id, stream, connection_id, sequence_number,
                   received_wall_ms, received_monotonic_ns,
                   raw_payload_sha256, raw_text, parse_status, parse_error,
                   storage_chunk_id, raw_offset, raw_size,
                   normalized_event_count
            FROM polymarket_raw_message
            WHERE {" AND ".join(filters)}
            {order_clause}
            """,
            parameters,
        )
        for rows in iter(lambda: cursor.fetchmany(_INTEGRITY_FETCH_SIZE), []):
            for row in rows:
                event_count = int(row[13])
                if event_count == 0:
                    continue
                if event_count < 0 or str(row[7]) or str(row[8]) != "ok" or str(row[9]):
                    raise ValueError("verified v3 raw-message metadata is invalid")
                metadata = (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[10],
                    row[11],
                    row[12],
                    row[13],
                )
                yield from self._decode_verified_chunk_message(
                    run_id,
                    metadata,
                    selected_conditions=selected_conditions,
                )

    def _iter_capture_frame_events(
        self,
        run_id: str,
        *,
        selected_streams: tuple[str, ...] | None,
        selected_conditions: tuple[str, ...] | None,
        ordered: bool,
    ) -> Iterator[DecodedPublicEvent]:
        last_order_key: tuple[int, int] | None = None
        for stored in self._iter_capture_messages(run_id, streams=selected_streams):
            message = stored.message
            order_key = _capture_receipt_key(message)
            if ordered and last_order_key is not None and order_key < last_order_key:
                raise ValueError("capture-frame receive order is not monotonic")
            last_order_key = order_key
            parse_status, parse_error, candidates = _stream_message_events(
                message.raw_text
            )
            if parse_status == "invalid":
                raise ValueError(
                    "capture-frame stream message is invalid: "
                    f"{stored.message_id}:{parse_error}"
                )
            if not candidates:
                continue
            yield from self._decode_verified_chunk_message(
                run_id,
                stored.cache_metadata(len(candidates)),
                selected_conditions=selected_conditions,
                candidates=candidates,
            )

    def iter_public_events(
        self,
        run_id: str,
        *,
        streams: Sequence[str] | None = None,
        condition_ids: Sequence[str] | None = None,
        ordered: bool = True,
        verified_source: bool = False,
    ) -> Iterator[DecodedPublicEvent]:
        """Yield source-reconstructed events in deterministic receive order."""

        if not isinstance(ordered, bool) or not isinstance(verified_source, bool):
            raise ValueError("public-event control flags must be boolean")
        if verified_source:
            cached = self._terminal_evidence_integrity_cache.get(run_id)
            if (
                cached is None
                or cached[0] != self._terminal_evidence_fingerprint(run_id)
                or cached[1]
            ):
                raise ValueError(
                    "verified public-event iteration requires a clean current "
                    "terminal integrity audit"
                )
        selected_streams: tuple[str, ...] | None = None
        if streams is not None:
            selected_streams = tuple(sorted({str(value) for value in streams}))
            if not selected_streams or any(
                value not in _STREAMS for value in selected_streams
            ):
                raise ValueError("public-event stream filter is invalid")
        selected_conditions: tuple[str, ...] | None = None
        if condition_ids is not None:
            selected_conditions = tuple(
                sorted({str(value or "").strip().lower() for value in condition_ids})
            )
            if not selected_conditions or any(
                not value for value in selected_conditions
            ):
                raise ValueError("public-event condition filter is invalid")
        storage_version = self._storage_schema_version(run_id)
        if (
            storage_version
            in {
                _RAW_RECONSTRUCTED_STORAGE_SCHEMA_VERSION,
                POLYMARKET_STORAGE_SCHEMA_VERSION,
            }
            and verified_source
            and selected_conditions is not None
            and self._condition_message_cache_available(run_id, selected_conditions)
        ):
            yield from self._iter_verified_condition_cache_events(
                run_id,
                selected_streams=selected_streams,
                selected_conditions=selected_conditions,
                ordered=ordered,
            )
            return
        if storage_version == POLYMARKET_STORAGE_SCHEMA_VERSION:
            yield from self._iter_capture_frame_events(
                run_id,
                selected_streams=selected_streams,
                selected_conditions=selected_conditions,
                ordered=ordered,
            )
            return
        if (
            storage_version == _RAW_RECONSTRUCTED_STORAGE_SCHEMA_VERSION
            and verified_source
            and selected_conditions is None
        ):
            yield from self._iter_verified_v3_raw_events(
                run_id,
                selected_streams=selected_streams,
                selected_conditions=None,
                ordered=ordered,
            )
            return
        compact = storage_version in _COMPACT_STORAGE_SCHEMA_VERSIONS
        compact_columns = (
            ", r.storage_chunk_id, r.raw_offset, r.raw_size" if compact else ""
        )
        wide_select = f"""
            SELECT e.event_id, e.run_id, e.message_id, e.sub_index,
                   e.stream, e.event_type, e.symbol, e.condition_id,
                   e.asset_id, e.source_time_ms, e.publisher_time_ms,
                   e.event_json, e.event_sha256, r.connection_id,
                   r.sequence_number, r.received_wall_ms,
                   r.received_monotonic_ns, r.raw_payload_sha256,
                   r.raw_text{compact_columns}
            FROM polymarket_public_event AS e
            JOIN polymarket_raw_message AS r
              ON r.run_id = e.run_id AND r.message_id = e.message_id
        """

        def ordered_row_batches() -> Iterator[list[tuple[object, ...]]]:
            filters = ["e.run_id = ?"]
            parameters: list[object] = [run_id]
            if selected_streams is not None:
                stream_placeholders = ", ".join("?" for _ in selected_streams)
                filters.append(f"e.stream IN ({stream_placeholders})")
                parameters.extend(selected_streams)
            if selected_conditions is not None:
                condition_placeholders = ", ".join("?" for _ in selected_conditions)
                filters.append(f"e.condition_id IN ({condition_placeholders})")
                parameters.extend(selected_conditions)
            order_clause = ""
            if ordered:
                order_clause = """
                    ORDER BY r.received_monotonic_ns, r.received_wall_ms,
                             r.connection_id, r.sequence_number, e.sub_index,
                             e.event_id
                """
            cursor = self.connect().execute(
                wide_select + f" WHERE {' AND '.join(filters)} {order_clause}",
                parameters,
            )
            yield from iter(lambda: cursor.fetchmany(_INTEGRITY_FETCH_SIZE), [])

        cached_message_id = ""
        cached_events: tuple[Mapping[str, object], ...] = ()
        for rows in ordered_row_batches():
            for row in rows:
                (
                    event_id,
                    event_run,
                    message_id,
                    sub_index,
                    stream,
                    event_type,
                    symbol,
                    condition_id,
                    asset_id,
                    source_time_ms,
                    publisher_time_ms,
                    inline_event_json,
                    event_sha256,
                    connection_id,
                    sequence_number,
                    received_wall_ms,
                    received_monotonic_ns,
                    raw_payload_sha256,
                    inline_raw_text,
                    *compact_location,
                ) = row
                normalized_message_id = str(message_id)
                if normalized_message_id != cached_message_id:
                    if compact:
                        if str(inline_raw_text) or str(inline_event_json):
                            raise ValueError(
                                "compact evidence contains an unexpected inline payload"
                            )
                        raw_text = self._decode_compact_raw_text(
                            run_id=run_id,
                            chunk_id=compact_location[0],
                            raw_offset=compact_location[1],
                            raw_size=compact_location[2],
                            raw_payload_sha256=raw_payload_sha256,
                            verify_payload_hash=not verified_source,
                        )
                    else:
                        raw_text = str(inline_raw_text)
                        if hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != str(
                            raw_payload_sha256
                        ):
                            raise ValueError("legacy raw-message payload hash mismatch")
                    try:
                        parsed = _strict_json_loads(raw_text)
                    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
                        raise ValueError(
                            "normalized event references an invalid raw message"
                        ) from exc
                    candidates = parsed if isinstance(parsed, list) else [parsed]
                    if not all(isinstance(item, Mapping) for item in candidates):
                        raise ValueError(
                            "normalized event raw message contains a non-object"
                        )
                    cached_events = tuple(
                        item for item in candidates if isinstance(item, Mapping)
                    )
                    cached_message_id = normalized_message_id
                index = int(sub_index)
                if not 0 <= index < len(cached_events):
                    raise ValueError("normalized event sub-index is out of bounds")
                event = dict(cached_events[index])
                if verified_source:
                    yield DecodedPublicEvent(
                        event_id=str(event_id),
                        run_id=run_id,
                        message_id=normalized_message_id,
                        sub_index=index,
                        stream=str(stream),
                        event_type=str(event_type),
                        symbol=str(symbol),
                        condition_id=str(condition_id),
                        asset_id=str(asset_id),
                        source_time_ms=(
                            None if source_time_ms is None else int(source_time_ms)
                        ),
                        publisher_time_ms=(
                            None
                            if publisher_time_ms is None
                            else int(publisher_time_ms)
                        ),
                        event_sha256=str(event_sha256),
                        event=event,
                        connection_id=str(connection_id),
                        sequence_number=int(sequence_number),
                        received_wall_ms=int(received_wall_ms),
                        received_monotonic_ns=int(received_monotonic_ns),
                    )
                    continue
                canonical = _canonical_json(event)
                actual_event_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
                expected_event_id = _canonical_sha256(
                    {
                        "message_id": normalized_message_id,
                        "sub_index": index,
                        "event_sha256": actual_event_sha,
                    }
                )
                normalized = _event_index(str(stream), event)
                stored_index = {
                    "event_type": str(event_type),
                    "symbol": str(symbol),
                    "condition_id": str(condition_id),
                    "asset_id": str(asset_id),
                    "source_time_ms": source_time_ms,
                    "publisher_time_ms": publisher_time_ms,
                }

                def index_mismatches(candidate: Mapping[str, object]) -> list[str]:
                    return [
                        name
                        for name, stored_value in stored_index.items()
                        if candidate[name] != stored_value
                    ]

                metadata_mismatches = index_mismatches(normalized)
                if metadata_mismatches and not compact:
                    legacy = _event_index(
                        str(stream),
                        event,
                        canonicalize_chainlink_topic=False,
                    )
                    legacy_mismatches = index_mismatches(legacy)
                    if not legacy_mismatches:
                        metadata_mismatches = []
                mismatches = [
                    *([] if str(event_run) == run_id else ["run_id"]),
                    *(
                        []
                        if actual_event_sha == str(event_sha256)
                        else ["event_sha256"]
                    ),
                    *(
                        []
                        if hmac.compare_digest(str(event_id), expected_event_id)
                        else ["event_id"]
                    ),
                    *(
                        []
                        if not compact or not str(inline_event_json)
                        else ["compact_event_json"]
                    ),
                    *(
                        []
                        if compact or canonical == str(inline_event_json)
                        else ["event_json"]
                    ),
                    *metadata_mismatches,
                ]
                if mismatches:
                    raise ValueError(
                        "normalized event disagrees with its raw source:"
                        f"{event_id}:{','.join(mismatches)}"
                    )
                yield DecodedPublicEvent(
                    event_id=str(event_id),
                    run_id=run_id,
                    message_id=normalized_message_id,
                    sub_index=index,
                    stream=str(stream),
                    event_type=str(event_type),
                    symbol=str(symbol),
                    condition_id=str(condition_id),
                    asset_id=str(asset_id),
                    source_time_ms=(
                        None if source_time_ms is None else int(source_time_ms)
                    ),
                    publisher_time_ms=(
                        None if publisher_time_ms is None else int(publisher_time_ms)
                    ),
                    event_sha256=str(event_sha256),
                    event=event,
                    connection_id=str(connection_id),
                    sequence_number=int(sequence_number),
                    received_wall_ms=int(received_wall_ms),
                    received_monotonic_ns=int(received_monotonic_ns),
                )

    def record_gap(self, run_id: str, gap: StreamGap) -> str:
        self._require_running_run(run_id)
        gap = gap.validated()
        payload = {
            "run_id": run_id,
            "stream": gap.stream,
            "connection_id": gap.connection_id,
            "opened_at_ms": int(gap.opened_at_ms),
            "reason": str(gap.reason),
            "last_sequence_number": int(gap.last_sequence_number),
        }
        gap_id = _canonical_sha256(payload)
        self.connect().execute(
            "INSERT INTO polymarket_stream_gap VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                gap_id,
                run_id,
                gap.stream,
                gap.connection_id,
                gap.opened_at_ms,
                gap.reason,
                gap.last_sequence_number,
            ],
        )
        return gap_id

    def finish_run(
        self,
        run_id: str,
        *,
        started_at_ms: int,
        ended_at_ms: int,
        database: str,
        errors: Sequence[str],
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        progress_interval_seconds: int = 30,
        _verified_recovery_fingerprint: str | None = None,
    ) -> RecorderReport:
        if self.read_only:
            raise ValueError("read-only Polymarket evidence cannot finish a run")
        connection = self.connect()
        run_row = connection.execute(
            """
            SELECT status, started_at_ms, ended_at_ms
            FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if run_row is None:
            raise ValueError(f"unknown Polymarket recorder run: {run_id}")
        current_status, recorded_started_at_ms, recorded_ended_at_ms = run_row
        recovery_mode = _verified_recovery_fingerprint is not None
        if int(started_at_ms) != int(recorded_started_at_ms):
            raise ValueError("Polymarket recorder start time differs")
        if int(ended_at_ms) < int(started_at_ms):
            raise ValueError("Polymarket recorder end time predates its start")
        if recovery_mode:
            if (
                str(current_status) != "failed"
                or recorded_ended_at_ms is None
                or int(recorded_ended_at_ms) != int(ended_at_ms)
                or tuple(errors)
            ):
                raise ValueError("terminal audit recovery preconditions changed")
            if not hmac.compare_digest(
                self._terminal_evidence_fingerprint(run_id),
                str(_verified_recovery_fingerprint),
            ):
                raise ValueError("terminal audit recovery evidence changed")
        elif str(current_status) != "running" or recorded_ended_at_ms is not None:
            raise ValueError("terminal Polymarket recorder evidence is immutable")
        market_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_market_snapshot WHERE run_id = ?",
                [run_id],
            ).fetchone()[0]
        )
        storage_schema_version = self._storage_schema_version(run_id)
        integrity: tuple[str, ...] = ()
        if storage_schema_version == POLYMARKET_STORAGE_SCHEMA_VERSION:
            if not recovery_mode:
                integrity = self.integrity_errors(
                    run_id,
                    progress=progress,
                    progress_interval_seconds=progress_interval_seconds,
                )
            audit_summary = self._evidence_audit_summary.get(run_id)
            if audit_summary is None:
                integrity = (
                    *integrity,
                    f"capture_frame_audit_summary_missing:{run_id}",
                )
                raw_count = 0
                event_count = 0
                stream_counts: dict[str, int] = {}
            else:
                raw_count = audit_summary.raw_message_count
                event_count = audit_summary.normalized_event_count
                stream_counts = dict(audit_summary.stream_counts)
        else:
            raw_count = int(
                connection.execute(
                    "SELECT count(*) FROM polymarket_raw_message WHERE run_id = ?",
                    [run_id],
                ).fetchone()[0]
            )
            event_count = int(
                connection.execute(
                    "SELECT count(*) FROM polymarket_public_event WHERE run_id = ?",
                    [run_id],
                ).fetchone()[0]
            )
            stream_counts = {
                str(stream): int(count)
                for stream, count in connection.execute(
                    """
                    SELECT stream, count(*) FROM polymarket_raw_message
                    WHERE run_id = ? GROUP BY stream ORDER BY stream
                    """,
                    [run_id],
                ).fetchall()
            }
        gap_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?",
                [run_id],
            ).fetchone()[0]
        )
        conditions = tuple(
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT condition_id FROM polymarket_market_snapshot
                WHERE run_id = ? ORDER BY condition_id
                """,
                [run_id],
            ).fetchall()
        )
        assets = tuple(
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT asset FROM polymarket_market_snapshot
                WHERE run_id = ? ORDER BY asset
                """,
                [run_id],
            ).fetchall()
        )
        if storage_schema_version != POLYMARKET_STORAGE_SCHEMA_VERSION:
            integrity = self.integrity_errors(
                run_id,
                progress=progress,
                progress_interval_seconds=progress_interval_seconds,
            )
        required_streams = {"clob_market", "polymarket_rtds", "binance_spot"}
        coverage_errors: list[str] = []
        missing_streams = sorted(required_streams - set(stream_counts))
        if missing_streams:
            coverage_errors.append(f"missing_streams:{','.join(missing_streams)}")
        if assets != ("BTC", "ETH", "SOL"):
            coverage_errors.append(f"asset_coverage:{','.join(assets)}")
        if market_count < 3:
            coverage_errors.append(f"insufficient_market_snapshots:{market_count}")
        if raw_count < 1:
            coverage_errors.append("no_public_messages")
        run_errors = tuple(str(error) for error in errors) + tuple(coverage_errors)
        if run_errors or integrity:
            status = "failed"
        elif gap_count:
            status = "degraded"
        else:
            status = "complete"
        evidence_manifest_sha256 = (
            self._capture_manifest_sha256(run_id)
            if storage_schema_version == POLYMARKET_STORAGE_SCHEMA_VERSION
            else ""
        )
        payload: dict[str, object] = {
            "schema_version": POLYMARKET_RECORDER_SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "database": database,
            "started_at_ms": int(started_at_ms),
            "ended_at_ms": int(ended_at_ms),
            "duration_seconds": max(0.0, (ended_at_ms - started_at_ms) / 1_000.0),
            "market_snapshot_count": market_count,
            "raw_message_count": raw_count,
            "normalized_event_count": event_count,
            "stream_gap_count": gap_count,
            "stream_counts": stream_counts,
            "assets": assets,
            "conditions": conditions,
            "integrity_errors": integrity,
            "errors": run_errors,
            "evidence_manifest_sha256": evidence_manifest_sha256,
        }
        report_sha = _canonical_sha256(payload)
        payload["report_sha256"] = report_sha
        report = RecorderReport(**payload)
        report_json = _canonical_json(report.asdict())
        connection.execute(
            """
            UPDATE polymarket_recorder_run
            SET status = ?, ended_at_ms = ?, report_json = ?, report_sha256 = ?, error = ?
            WHERE run_id = ?
            """,
            [
                status,
                ended_at_ms,
                report_json,
                report_sha,
                "; ".join(run_errors),
                run_id,
            ],
        )
        return report

    def fail_run(
        self,
        run_id: str,
        *,
        started_at_ms: int,
        ended_at_ms: int,
        database: str,
        errors: Sequence[str],
    ) -> RecorderReport:
        """Terminalize an interrupted run without requiring the full evidence audit."""

        if self.read_only:
            raise ValueError("read-only Polymarket evidence cannot fail a run")
        self._require_running_run(run_id)
        connection = self.connect()
        terminal_errors = [str(error) for error in errors]

        def query_rows(label: str, query: str) -> list[tuple[object, ...]]:
            try:
                return connection.execute(query, [run_id]).fetchall()
            except Exception as exc:
                terminal_errors.append(
                    f"terminal_summary_{label}:{exc.__class__.__name__}:{exc}"
                )
                return []

        def query_count(label: str, table: str) -> int:
            rows = query_rows(
                label,
                f"SELECT count(*) FROM {table} WHERE run_id = ?",
            )
            return int(rows[0][0]) if rows else 0

        market_count = query_count("markets", "polymarket_market_snapshot")
        gap_count = query_count("gaps", "polymarket_stream_gap")
        if self._storage_schema_version(run_id) == POLYMARKET_STORAGE_SCHEMA_VERSION:
            try:
                raw_count, stream_counts = self._capture_frame_fast_counts(run_id)
            except (TypeError, ValueError, OverflowError) as exc:
                terminal_errors.append(
                    f"terminal_summary_frames:{exc.__class__.__name__}:{exc}"
                )
                raw_count = 0
                stream_counts = {}
            event_count = 0
        else:
            raw_count = query_count("messages", "polymarket_raw_message")
            event_count = query_count("events", "polymarket_public_event")
            stream_counts = {
                str(stream): int(count)
                for stream, count in query_rows(
                    "streams",
                    """
                    SELECT stream, count(*) FROM polymarket_raw_message
                    WHERE run_id = ? GROUP BY stream ORDER BY stream
                    """,
                )
            }
        conditions = tuple(
            str(row[0])
            for row in query_rows(
                "conditions",
                """
                SELECT DISTINCT condition_id FROM polymarket_market_snapshot
                WHERE run_id = ? ORDER BY condition_id
                """,
            )
        )
        assets = tuple(
            str(row[0])
            for row in query_rows(
                "assets",
                """
                SELECT DISTINCT asset FROM polymarket_market_snapshot
                WHERE run_id = ? ORDER BY asset
                """,
            )
        )
        integrity = ("terminal_integrity_audit_incomplete",)
        payload: dict[str, object] = {
            "schema_version": POLYMARKET_RECORDER_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "failed",
            "database": database,
            "started_at_ms": int(started_at_ms),
            "ended_at_ms": int(ended_at_ms),
            "duration_seconds": max(0.0, (ended_at_ms - started_at_ms) / 1_000.0),
            "market_snapshot_count": market_count,
            "raw_message_count": raw_count,
            "normalized_event_count": event_count,
            "stream_gap_count": gap_count,
            "stream_counts": stream_counts,
            "assets": assets,
            "conditions": conditions,
            "integrity_errors": integrity,
            "errors": tuple(terminal_errors),
            "evidence_manifest_sha256": "",
        }
        report_sha = _canonical_sha256(payload)
        payload["report_sha256"] = report_sha
        report = RecorderReport(**payload)
        connection.execute(
            """
            UPDATE polymarket_recorder_run
            SET status = 'failed', ended_at_ms = ?, report_json = ?,
                report_sha256 = ?, error = ?
            WHERE run_id = ?
            """,
            [
                int(ended_at_ms),
                _canonical_json(report.asdict()),
                report_sha,
                "; ".join(terminal_errors),
                run_id,
            ],
        )
        return report

    def recover_terminal_audit_if_resource_exhausted(
        self,
        run_id: str,
        *,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        progress_interval_seconds: int = 30,
    ) -> TerminalAuditRecoveryReport | None:
        """Recover one exact storage-v4 terminal-audit OOM without hiding evidence."""

        if self.read_only:
            raise ValueError("read-only Polymarket evidence cannot recover an audit")
        connection = self.connect()
        run_row = connection.execute(
            """
            SELECT status, storage_schema_version, started_at_ms, ended_at_ms,
                   report_json, report_sha256, error
            FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if run_row is None:
            raise ValueError(f"unknown Polymarket recorder run: {run_id}")
        (
            status,
            storage_schema_version,
            started_at_ms,
            ended_at_ms,
            prior_report_json,
            prior_report_sha256,
            prior_error,
        ) = run_row
        recovery_row = self._terminal_recovery_row(run_id)
        if str(status) != "failed":
            return None
        if recovery_row is not None:
            raise ValueError("failed Polymarket run already has an audit recovery")
        normalized_error = str(prior_error or "")
        if (
            str(storage_schema_version) != POLYMARKET_STORAGE_SCHEMA_VERSION
            or not normalized_error.startswith(
                _TERMINAL_AUDIT_RESOURCE_EXHAUSTED_PREFIX
            )
        ):
            return None
        if ended_at_ms is None:
            raise ValueError("terminal audit recovery requires an ended run")

        try:
            parsed_prior_report = _strict_json_loads(str(prior_report_json))
            if not isinstance(parsed_prior_report, Mapping):
                raise ValueError("prior recorder report is not an object")
            prior_report = parsed_prior_report
            if _canonical_json(prior_report) != str(prior_report_json):
                raise ValueError("prior recorder report is not canonical")
            if set(prior_report) != set(RecorderReport.__dataclass_fields__):
                raise ValueError("prior recorder report fields differ")
            unhashed_prior = dict(prior_report)
            embedded_prior_sha256 = str(unhashed_prior.pop("report_sha256", ""))
            actual_prior_sha256 = _canonical_sha256(unhashed_prior)
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
            raise ValueError("terminal audit recovery prior report is invalid") from exc
        if (
            not hmac.compare_digest(actual_prior_sha256, str(prior_report_sha256))
            or not hmac.compare_digest(
                embedded_prior_sha256,
                str(prior_report_sha256),
            )
            or str(prior_report.get("schema_version") or "")
            != POLYMARKET_RECORDER_SCHEMA_VERSION
            or str(prior_report.get("run_id") or "") != run_id
            or str(prior_report.get("status") or "") != "failed"
            or prior_report.get("started_at_ms") != int(started_at_ms)
            or prior_report.get("ended_at_ms") != int(ended_at_ms)
            or prior_report.get("integrity_errors")
            != ["terminal_integrity_audit_incomplete"]
            or prior_report.get("errors") != [normalized_error]
            or prior_report.get("normalized_event_count") != 0
            or prior_report.get("evidence_manifest_sha256") != ""
        ):
            raise ValueError("terminal audit recovery prior report contract differs")
        prior_raw_message_count = prior_report.get("raw_message_count")
        if type(prior_raw_message_count) is not int or prior_raw_message_count < 1:
            raise ValueError("terminal audit recovery prior count is invalid")

        def notify(phase: str, payload: Mapping[str, object]) -> None:
            if progress is None:
                return
            try:
                progress(phase, payload)
            except Exception:
                return

        fast_raw_count, fast_stream_counts = self._capture_frame_fast_counts(run_id)
        if fast_raw_count != prior_raw_message_count:
            raise ValueError("terminal audit recovery fast count differs")
        pre_recovery_fingerprint = self._terminal_evidence_fingerprint(run_id)
        notify(
            "terminal-audit-recovery-started",
            {
                "raw_message_count": fast_raw_count,
                "memory_limit": self.memory_limit,
                "database_threads": self.threads,
            },
        )
        integrity = self.integrity_errors(
            run_id,
            progress=progress,
            progress_interval_seconds=progress_interval_seconds,
            _skip_terminal_report_validation=True,
        )
        if integrity:
            raise ValueError(
                "terminal audit recovery evidence failed: " + "; ".join(integrity)
            )
        audit_summary = self._evidence_audit_summary.get(run_id)
        if (
            audit_summary is None
            or audit_summary.raw_message_count != fast_raw_count
            or audit_summary.normalized_event_count < 1
            or audit_summary.stream_counts != fast_stream_counts
        ):
            raise ValueError("terminal audit recovery summary differs")
        if not hmac.compare_digest(
            self._terminal_evidence_fingerprint(run_id),
            pre_recovery_fingerprint,
        ):
            raise ValueError("terminal audit recovery evidence changed during audit")

        recovered_at_ms = int(time.time() * 1_000)
        connection.execute("BEGIN TRANSACTION")
        try:
            recovered_report = self.finish_run(
                run_id,
                started_at_ms=int(started_at_ms),
                ended_at_ms=int(ended_at_ms),
                database=str(prior_report.get("database") or ""),
                errors=(),
                _verified_recovery_fingerprint=pre_recovery_fingerprint,
            )
            if (
                recovered_report.status not in {"complete", "degraded"}
                or recovered_report.errors
                or recovered_report.integrity_errors
                or recovered_report.raw_message_count != fast_raw_count
                or recovered_report.normalized_event_count
                != audit_summary.normalized_event_count
            ):
                raise ValueError("terminal audit recovery produced an invalid report")
            recovery_payload: dict[str, object] = {
                "schema_version": POLYMARKET_TERMINAL_RECOVERY_SCHEMA_VERSION,
                "run_id": run_id,
                "recovered_at_ms": recovered_at_ms,
                "recovery_reason": _TERMINAL_AUDIT_RECOVERY_REASON,
                "prior_report_sha256": str(prior_report_sha256),
                "recovered_report_sha256": recovered_report.report_sha256,
                "pre_recovery_evidence_fingerprint": pre_recovery_fingerprint,
                "raw_message_count": fast_raw_count,
                "normalized_event_count": audit_summary.normalized_event_count,
                "memory_limit": self.memory_limit,
                "database_threads": self.threads,
                "labels_consulted": False,
                "outcomes_consulted": False,
                "model_scores_consulted": False,
                "profitability_claim": False,
                "trading_authority": False,
                "training_authority": False,
            }
            recovery_sha256 = _canonical_sha256(recovery_payload)
            recovery_payload["recovery_sha256"] = recovery_sha256
            recovery = TerminalAuditRecoveryReport(**recovery_payload)
            recovery_json = _canonical_json(recovery.asdict())
            connection.execute(
                """
                INSERT INTO polymarket_terminal_audit_recovery VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    run_id,
                    POLYMARKET_TERMINAL_RECOVERY_SCHEMA_VERSION,
                    recovered_at_ms,
                    _TERMINAL_AUDIT_RECOVERY_REASON,
                    str(prior_report_json),
                    str(prior_report_sha256),
                    normalized_error,
                    recovered_report.report_sha256,
                    recovery_json,
                    recovery_sha256,
                ],
            )
            parsed_recovered_report = _strict_json_loads(
                _canonical_json(recovered_report.asdict())
            )
            if not isinstance(parsed_recovered_report, Mapping):
                raise ValueError("recovered recorder report is not an object")
            chain_errors = self._terminal_recovery_integrity_errors(
                run_id,
                parsed_recovered_report,
            )
            if chain_errors:
                raise ValueError("; ".join(chain_errors))
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
            raise
        terminal_fingerprint = self._terminal_evidence_fingerprint(run_id)
        self._terminal_evidence_integrity_cache[run_id] = (
            terminal_fingerprint,
            (),
        )
        notify(
            "terminal-audit-recovery-complete",
            {
                "raw_message_count": recovery.raw_message_count,
                "normalized_event_count": recovery.normalized_event_count,
                "recovery_sha256": recovery.recovery_sha256,
            },
        )
        return recovery

    def resume_integrity_errors(
        self,
        run_id: str,
        *,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        progress_interval_seconds: int = 30,
    ) -> tuple[str, ...]:
        """Attest a deeply audited recovery without decompressing it again."""

        selected = str(run_id or "").strip()
        if not selected:
            raise ValueError("Polymarket resume audit requires a run ID")
        if (
            self._storage_schema_version(selected) != POLYMARKET_STORAGE_SCHEMA_VERSION
            or self._terminal_recovery_row(selected) is None
        ):
            return self.integrity_errors(
                selected,
                progress=progress,
                progress_interval_seconds=progress_interval_seconds,
            )
        cached = self._terminal_evidence_integrity_cache.get(selected)
        if cached is not None and cached[0] == self._terminal_evidence_fingerprint(
            selected
        ):
            errors = list(cached[1])
            if self.paper_journal is not None:
                errors.extend(self.paper_journal.integrity_errors())
            return tuple(errors)

        interval = max(1, int(progress_interval_seconds))
        started = time.monotonic()
        last_progress = started
        verified_chunk_count = 0
        verified_compressed_bytes = 0

        def notify(phase: str, *, force: bool = False) -> None:
            nonlocal last_progress
            if progress is None:
                return
            now = time.monotonic()
            if not force and now - last_progress < interval:
                return
            last_progress = now
            try:
                progress(
                    phase,
                    {
                        "audit_elapsed_seconds": round(now - started, 3),
                        "verified_chunk_count": verified_chunk_count,
                        "verified_compressed_bytes": verified_compressed_bytes,
                    },
                )
            except Exception:
                return

        connection = self.connect()
        errors: list[str] = []
        notify("resume-integrity-started", force=True)
        fingerprint_before = self._terminal_evidence_fingerprint(selected)
        row = connection.execute(
            """
            SELECT schema_version, storage_schema_version, status, started_at_ms,
                   ended_at_ms, report_json, report_sha256, error
            FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        parsed_report: Mapping[str, object] | None = None
        if row is None:
            errors.append(f"missing_recorder_run:{selected}")
        else:
            (
                run_schema,
                storage_schema,
                run_status,
                run_started,
                run_ended,
                report_json,
                report_sha256,
                run_error,
            ) = row
            if str(run_schema) != POLYMARKET_RECORDER_SCHEMA_VERSION:
                errors.append(f"recorder_schema_mismatch:{selected}")
            if str(storage_schema) != POLYMARKET_STORAGE_SCHEMA_VERSION:
                errors.append(f"recorder_storage_schema_mismatch:{selected}")
            if str(run_status) not in {"complete", "degraded"}:
                errors.append(f"recorder_run_not_resumable:{selected}:{run_status}")
            if run_ended is None or str(run_error or ""):
                errors.append(f"recorder_terminal_state_invalid:{selected}")
            try:
                parsed = _strict_json_loads(str(report_json))
                if not isinstance(parsed, Mapping):
                    raise ValueError("recorder report is not an object")
                parsed_report = parsed
                if set(parsed_report) != set(RecorderReport.__dataclass_fields__):
                    errors.append(f"recorder_report_fields_mismatch:{selected}")
                if _canonical_json(parsed_report) != str(report_json):
                    errors.append(f"recorder_report_not_canonical:{selected}")
                unhashed_report = dict(parsed_report)
                embedded_sha256 = str(
                    unhashed_report.pop("report_sha256", "")
                )
                actual_sha256 = _canonical_sha256(unhashed_report)
                if not hmac.compare_digest(actual_sha256, str(report_sha256)):
                    errors.append(f"recorder_report_hash_mismatch:{selected}")
                if not hmac.compare_digest(embedded_sha256, str(report_sha256)):
                    errors.append(
                        f"recorder_report_embedded_hash_mismatch:{selected}"
                    )
                expected_report_fields = {
                    "schema_version": POLYMARKET_RECORDER_SCHEMA_VERSION,
                    "run_id": selected,
                    "status": str(run_status),
                    "started_at_ms": int(run_started),
                    "ended_at_ms": int(run_ended or 0),
                    "errors": [],
                    "integrity_errors": [],
                }
                for field, expected in expected_report_fields.items():
                    if parsed_report.get(field) != expected:
                        errors.append(
                            f"recorder_report_terminal_mismatch:{selected}:{field}"
                        )
            except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
                errors.append(
                    f"recorder_report_invalid:{selected}:{exc.__class__.__name__}:{exc}"
                )

            errors.extend(
                self._terminal_recovery_integrity_errors(selected, parsed_report)
            )
            try:
                raw_count, stream_counts = self._capture_frame_fast_counts(selected)
                manifest_sha256 = self._capture_manifest_sha256(selected)
            except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
                errors.append(
                    "resume_capture_metadata_invalid:"
                    f"{selected}:{exc.__class__.__name__}:{exc}"
                )
                raw_count, stream_counts, manifest_sha256 = 0, {}, ""

            snapshots = connection.execute(
                """
                SELECT snapshot_id, run_id, observed_wall_ms,
                       observed_monotonic_ns, asset, market_id, condition_id,
                       slug, question, event_start_ms, end_ms, up_token_id,
                       down_token_id, tick_size, minimum_order_size, fees_enabled,
                       fee_rate, fee_exponent, fee_taker_only, fee_rebate_rate,
                       liquidity_quote, volume_quote, resolution_source,
                       gamma_payload_json, gamma_payload_sha256, clob_info_json,
                       clob_info_sha256, up_fee_rate_json, up_fee_rate_sha256,
                       down_fee_rate_json, down_fee_rate_sha256, maker_base_fee,
                       taker_base_fee, taker_order_delay_enabled,
                       minimum_order_age_seconds, snapshot_payload_json,
                       snapshot_sha256
                FROM polymarket_market_snapshot
                WHERE run_id = ? ORDER BY snapshot_id
                """,
                [selected],
            ).fetchall()
            for snapshot in snapshots:
                errors.extend(_snapshot_integrity_errors(snapshot))
            gaps = connection.execute(
                """
                SELECT gap_id, run_id, stream, connection_id, opened_at_ms,
                       reason, last_sequence_number
                FROM polymarket_stream_gap WHERE run_id = ? ORDER BY gap_id
                """,
                [selected],
            ).fetchall()
            for gap in gaps:
                expected_gap_id = _canonical_sha256(
                    {
                        "run_id": str(gap[1]),
                        "stream": str(gap[2]),
                        "connection_id": str(gap[3]),
                        "opened_at_ms": int(gap[4]),
                        "reason": str(gap[5]),
                        "last_sequence_number": int(gap[6]),
                    }
                )
                if not hmac.compare_digest(str(gap[0]), expected_gap_id):
                    errors.append(f"stream_gap_id_mismatch:{gap[0]}")
            if parsed_report is not None:
                report_evidence = {
                    "market_snapshot_count": len(snapshots),
                    "raw_message_count": raw_count,
                    "stream_gap_count": len(gaps),
                    "stream_counts": stream_counts,
                    "assets": sorted({str(snapshot[4]) for snapshot in snapshots}),
                    "conditions": sorted(
                        {str(snapshot[6]) for snapshot in snapshots}
                    ),
                    "evidence_manifest_sha256": manifest_sha256,
                }
                for field, actual in report_evidence.items():
                    if parsed_report.get(field) != actual:
                        errors.append(
                            f"recorder_report_evidence_mismatch:{selected}:{field}"
                        )
                normalized_event_count = parsed_report.get("normalized_event_count")
                if (
                    type(normalized_event_count) is not int
                    or normalized_event_count < 0
                ):
                    errors.append(
                        f"recorder_report_evidence_mismatch:{selected}:"
                        "normalized_event_count"
                    )
                if run_ended is not None:
                    outside = sum(
                        not int(run_started) <= int(snapshot[2]) <= int(run_ended)
                        for snapshot in snapshots
                    )
                    if outside:
                        errors.append(
                            f"recorder_snapshot_outside_run:{selected}:{outside}"
                        )

        expected_chunk_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_raw_chunk WHERE run_id = ?",
                [selected],
            ).fetchone()[0]
        )
        payload_cursor = self._payload_connection().execute(
            """
            SELECT chunk_id, compressed_bytes, compressed_sha256,
                   compressed_payload
            FROM polymarket_raw_chunk WHERE run_id = ?
            """,
            [selected],
        )
        while batch := payload_cursor.fetchmany(64):
            for chunk_id, claimed_bytes, claimed_sha256, payload in batch:
                verified_chunk_count += 1
                compressed = bytes(payload)
                verified_compressed_bytes += len(compressed)
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", str(claimed_sha256))
                    or len(compressed) != int(claimed_bytes)
                    or not hmac.compare_digest(
                        hashlib.sha256(compressed).hexdigest(),
                        str(claimed_sha256),
                    )
                ):
                    errors.append(
                        f"raw_chunk_compressed_payload_mismatch:{chunk_id}"
                    )
                notify("resume-integrity-chunks")
        if verified_chunk_count != expected_chunk_count:
            errors.append(
                f"raw_chunk_count_mismatch:{selected}:"
                f"{verified_chunk_count}:{expected_chunk_count}"
            )
        fingerprint_after = self._terminal_evidence_fingerprint(selected)
        if not hmac.compare_digest(fingerprint_before, fingerprint_after):
            errors.append(f"resume_evidence_changed_during_audit:{selected}")
        evidence_errors = tuple(errors)
        self._terminal_evidence_integrity_cache[selected] = (
            fingerprint_after,
            evidence_errors,
        )
        if self.paper_journal is not None:
            errors.extend(self.paper_journal.integrity_errors())
        notify("resume-integrity-complete", force=True)
        return tuple(errors)

    def integrity_errors(
        self,
        run_id: str,
        *,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        progress_interval_seconds: int = 30,
        _skip_terminal_report_validation: bool = False,
    ) -> tuple[str, ...]:
        interval = max(1, int(progress_interval_seconds))
        audit_started = time.monotonic()
        last_progress_at = audit_started
        message_count = 0
        event_count = 0

        def notify(phase: str, *, force: bool = False) -> None:
            nonlocal last_progress_at
            if progress is None:
                return
            now = time.monotonic()
            if not force and now - last_progress_at < interval:
                return
            last_progress_at = now
            try:
                progress(
                    phase,
                    {
                        "audit_elapsed_seconds": max(0.0, now - audit_started),
                        "verified_raw_message_count": message_count,
                        "verified_event_count": event_count,
                    },
                )
            except Exception:
                return

        cached = self._terminal_evidence_integrity_cache.get(run_id)
        if (
            not _skip_terminal_report_validation
            and cached is not None
            and cached[0] == self._terminal_evidence_fingerprint(run_id)
        ):
            notify("integrity-cache-hit", force=True)
            errors = list(cached[1])
            if self.paper_journal is not None:
                errors.extend(self.paper_journal.integrity_errors())
            return tuple(errors)
        connection = self.connect()
        errors: list[str] = []
        notify("integrity-started", force=True)
        run_row = connection.execute(
            """
            SELECT schema_version, status, started_at_ms, ended_at_ms,
                   report_json, report_sha256
            FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if run_row is None:
            result = (f"missing_recorder_run:{run_id}",)
            if self.read_only:
                self._terminal_evidence_integrity_cache[run_id] = (
                    self._terminal_evidence_fingerprint(run_id),
                    result,
                )
            notify("integrity-complete", force=True)
            return result
        run_schema, run_status, run_started, run_ended, report_json, report_sha = (
            run_row
        )
        parsed_report: Mapping[str, object] | None = None
        if str(run_schema) != POLYMARKET_RECORDER_SCHEMA_VERSION:
            errors.append(f"recorder_schema_mismatch:{run_id}")
        if str(run_status) != "running" and not _skip_terminal_report_validation:
            try:
                parsed = _strict_json_loads(str(report_json))
                if not isinstance(parsed, Mapping):
                    raise ValueError("recorder report is not an object")
                parsed_report = parsed
                canonical_report = _canonical_json(parsed_report)
                if canonical_report != str(report_json):
                    errors.append(f"recorder_report_not_canonical:{run_id}")
                report_payload = dict(parsed_report)
                embedded_sha = str(report_payload.pop("report_sha256", ""))
                actual_sha = _canonical_sha256(report_payload)
                if not hmac.compare_digest(actual_sha, str(report_sha)):
                    errors.append(f"recorder_report_hash_mismatch:{run_id}")
                if not hmac.compare_digest(embedded_sha, str(report_sha)):
                    errors.append(f"recorder_report_embedded_hash_mismatch:{run_id}")
                if str(parsed_report.get("run_id") or "") != run_id:
                    errors.append(f"recorder_report_run_mismatch:{run_id}")
                if str(parsed_report.get("status") or "") != str(run_status):
                    errors.append(f"recorder_report_status_mismatch:{run_id}")
            except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
                errors.append(f"recorder_report_invalid:{run_id}:{exc}")
        storage_schema_version = self._storage_schema_version(run_id)
        compact = storage_schema_version in _COMPACT_STORAGE_SCHEMA_VERSIONS
        capture_framed = storage_schema_version == POLYMARKET_STORAGE_SCHEMA_VERSION
        stream_counts_from_audit: dict[str, int] = {}
        out_of_window_message_count = 0
        if capture_framed:
            last_order_key: tuple[int, int] | None = None
            lane_accumulators: dict[tuple[str, str], _RawMessageLaneAccumulator] = {}
            try:
                for stored in self._iter_capture_messages(run_id):
                    message = stored.message
                    message_count += 1
                    lane_key = (message.stream, message.connection_id)
                    lane_accumulator = lane_accumulators.get(lane_key)
                    if lane_accumulator is None:
                        lane_accumulators[lane_key] = (
                            _RawMessageLaneAccumulator.from_message(message)
                        )
                    else:
                        lane_accumulator.update(message)
                    stream_counts_from_audit[message.stream] = (
                        stream_counts_from_audit.get(message.stream, 0) + 1
                    )
                    order_key = _capture_receipt_key(message)
                    if last_order_key is not None and order_key < last_order_key:
                        errors.append(
                            f"capture_frame_receive_order_invalid:{stored.message_id}"
                        )
                    last_order_key = order_key
                    if (
                        (
                            str(run_status) != "running"
                            or _skip_terminal_report_validation
                        )
                        and run_ended is not None
                        and (
                            message.received_wall_ms < int(run_started)
                            or message.received_wall_ms > int(run_ended)
                        )
                    ):
                        out_of_window_message_count += 1
                    parse_status, parse_error, candidates = _stream_message_events(
                        message.raw_text
                    )
                    if parse_status == "invalid":
                        errors.append(f"invalid_stream_message:{stored.message_id}")
                        if parse_error:
                            errors.append(
                                f"invalid_stream_message_detail:{stored.message_id}:"
                                f"{parse_error}"
                            )
                    event_count += len(candidates)
                    for event in candidates:
                        try:
                            event_json = _canonical_json(dict(event))
                            hashlib.sha256(event_json.encode("ascii")).hexdigest()
                            _event_index(message.stream, event)
                        except (TypeError, ValueError, OverflowError) as exc:
                            errors.append(
                                f"capture_frame_event_invalid:{stored.message_id}:"
                                f"{exc.__class__.__name__}:{exc}"
                            )
                    notify("integrity-raw-messages")
            except (TypeError, ValueError, OverflowError) as exc:
                errors.append(
                    f"capture_frame_invalid:{run_id}:{exc.__class__.__name__}:{exc}"
                )
            self._evidence_audit_summary[run_id] = _EvidenceAuditSummary(
                raw_message_count=message_count,
                normalized_event_count=event_count,
                stream_counts=dict(sorted(stream_counts_from_audit.items())),
                out_of_window_message_count=out_of_window_message_count,
                lane_summaries=_freeze_lane_summaries(lane_accumulators),
            )
            notify("integrity-public-events", force=True)
        elif compact:
            self._frame_cache_id = ""
            self._frame_cache = b""
            chunk_rows = connection.execute(
                """
                SELECT chunk_id, chunk_index, message_count, first_message_id,
                       last_message_id, message_manifest_xor
                FROM polymarket_raw_chunk
                WHERE run_id = ? ORDER BY chunk_index
                """,
                [run_id],
            ).fetchall()
            chunks = {str(row[0]): row for row in chunk_rows}
            if [int(row[1]) for row in chunk_rows] != list(range(len(chunk_rows))):
                errors.append(f"raw_chunk_index_sequence_invalid:{run_id}")
            observed_chunks: dict[str, dict[str, object]] = {}
            expected_event_count = 0
            expected_event_manifest_xor = 0
            expected_event_manifest_sum = 0
            invalid_chunks: set[str] = set()
            for batch in _query_batches(
                connection,
                """
                SELECT message_id, run_id, schema_version, stream,
                       connection_id, sequence_number, received_wall_ms,
                       received_monotonic_ns, raw_payload_sha256, raw_text,
                       parse_status, parse_error, storage_chunk_id,
                       chunk_message_index, raw_offset, raw_size,
                       normalized_event_count
                FROM polymarket_raw_message WHERE run_id = ?
                """,
                [run_id],
            ):
                message_count += len(batch)
                for raw_row in batch:
                    (
                        message_id,
                        message_run,
                        _message_schema,
                        stream,
                        connection_id,
                        sequence,
                        _received_wall_ms,
                        _received_monotonic_ns,
                        claimed,
                        inline_raw_text,
                        parse_status,
                        parse_error,
                        chunk_id,
                        chunk_message_index,
                        raw_offset,
                        raw_size,
                        normalized_event_count,
                    ) = raw_row
                    normalized_chunk_id = str(chunk_id or "")
                    if str(inline_raw_text):
                        errors.append(f"compact_inline_raw_payload:{message_id}")
                    if normalized_chunk_id not in chunks:
                        errors.append(f"raw_message_missing_chunk:{message_id}")
                        continue
                    try:
                        normalized_count = int(normalized_event_count)
                        position = int(chunk_message_index)
                        manifest_hash = _compact_message_manifest_hash(raw_row)
                    except (TypeError, ValueError, OverflowError) as exc:
                        errors.append(
                            f"raw_message_metadata_invalid:{message_id}:"
                            f"{exc.__class__.__name__}:{exc}"
                        )
                        continue
                    if (
                        normalized_count < 0
                        or not 0 <= position < _RAW_CHUNK_MESSAGE_LIMIT
                    ):
                        errors.append(f"raw_message_metadata_invalid:{message_id}")
                        continue
                    expected_event_count += normalized_count
                    observed = observed_chunks.setdefault(
                        normalized_chunk_id,
                        {
                            "count": 0,
                            "manifest_xor": 0,
                            "minimum_index": _RAW_CHUNK_MESSAGE_LIMIT,
                            "maximum_index": -1,
                            "first_message_id": "",
                            "last_message_id": "",
                        },
                    )
                    observed["count"] = int(observed["count"]) + 1
                    observed["manifest_xor"] = int(observed["manifest_xor"]) ^ int(
                        manifest_hash, 16
                    )
                    observed["minimum_index"] = min(
                        int(observed["minimum_index"]), position
                    )
                    observed["maximum_index"] = max(
                        int(observed["maximum_index"]), position
                    )
                    chunk_message_count = int(chunks[normalized_chunk_id][2])
                    if position == 0:
                        observed["first_message_id"] = str(message_id)
                    if position == chunk_message_count - 1:
                        observed["last_message_id"] = str(message_id)
                    expected_id = _canonical_sha256(
                        {
                            "run_id": str(message_run),
                            "stream": str(stream),
                            "connection_id": str(connection_id),
                            "sequence_number": int(sequence),
                            "raw_payload_sha256": str(claimed),
                        }
                    )
                    if not hmac.compare_digest(str(message_id), expected_id):
                        errors.append(f"raw_message_id_mismatch:{message_id}")
                    if normalized_chunk_id in invalid_chunks:
                        continue
                    try:
                        raw_text = self._decode_compact_raw_text(
                            run_id=run_id,
                            chunk_id=normalized_chunk_id,
                            raw_offset=raw_offset,
                            raw_size=raw_size,
                            raw_payload_sha256=claimed,
                        )
                    except (TypeError, ValueError) as exc:
                        invalid_chunks.add(normalized_chunk_id)
                        errors.append(
                            f"raw_chunk_invalid:{normalized_chunk_id}:"
                            f"{exc.__class__.__name__}:{exc}"
                        )
                        continue
                    expected_status = "ok"
                    expected_error = ""
                    expected_events: list[Mapping[str, object]] = []
                    try:
                        parsed = _strict_json_loads(raw_text)
                        candidates = parsed if isinstance(parsed, list) else [parsed]
                        if not all(isinstance(item, Mapping) for item in candidates):
                            raise ValueError(
                                "JSON stream message contains a non-object event"
                            )
                        expected_events = [
                            item for item in candidates if isinstance(item, Mapping)
                        ]
                    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
                        expected_status = (
                            "control"
                            if raw_text.strip() in {"", "PING", "PONG"}
                            else "invalid"
                        )
                        expected_error = (
                            ""
                            if expected_status == "control"
                            else f"{exc.__class__.__name__}:{exc}"
                        )
                    if (
                        str(parse_status) != expected_status
                        or str(parse_error) != expected_error
                        or normalized_count != len(expected_events)
                    ):
                        errors.append(f"raw_message_parse_mismatch:{message_id}")
                    if expected_status == "invalid":
                        errors.append(f"invalid_stream_message:{message_id}")
                    for sub_index, event in enumerate(expected_events):
                        event_json = _canonical_json(dict(event))
                        event_sha256 = hashlib.sha256(
                            event_json.encode("ascii")
                        ).hexdigest()
                        event_id = _canonical_sha256(
                            {
                                "message_id": str(message_id),
                                "sub_index": sub_index,
                                "event_sha256": event_sha256,
                            }
                        )
                        normalized = _event_index(str(stream), event)
                        manifest_value = int(
                            _public_event_manifest_hash(
                                event_id=event_id,
                                run_id=message_run,
                                message_id=message_id,
                                sub_index=sub_index,
                                stream=stream,
                                event_type=normalized["event_type"],
                                symbol=normalized["symbol"],
                                condition_id=normalized["condition_id"],
                                asset_id=normalized["asset_id"],
                                source_time_ms=normalized["source_time_ms"],
                                publisher_time_ms=normalized["publisher_time_ms"],
                                event_sha256=event_sha256,
                            ),
                            16,
                        )
                        expected_event_manifest_xor ^= manifest_value
                        expected_event_manifest_sum = (
                            expected_event_manifest_sum + manifest_value
                        ) % (1 << 256)
                notify("integrity-raw-messages")
            for chunk_id, chunk_row in chunks.items():
                observed = observed_chunks.get(chunk_id)
                expected_count = int(chunk_row[2])
                if (
                    observed is None
                    or int(observed["count"]) != expected_count
                    or int(observed["minimum_index"]) != 0
                    or int(observed["maximum_index"]) != expected_count - 1
                    or str(observed["first_message_id"]) != str(chunk_row[3])
                    or str(observed["last_message_id"]) != str(chunk_row[4])
                    or f"{int(observed['manifest_xor']):064x}" != str(chunk_row[5])
                ):
                    errors.append(f"raw_chunk_manifest_mismatch:{chunk_id}")
            notify("integrity-public-events", force=True)
            observed_event_manifest_xor = 0
            observed_event_manifest_sum = 0
            try:
                for batch in _query_batches(
                    connection,
                    """
                    SELECT event_id, run_id, message_id, sub_index, stream,
                           event_type, symbol, condition_id, asset_id,
                           source_time_ms, publisher_time_ms, event_json,
                           event_sha256
                    FROM polymarket_public_event WHERE run_id = ?
                    """,
                    [run_id],
                ):
                    event_count += len(batch)
                    for row in batch:
                        if str(row[11]):
                            errors.append(f"compact_inline_event_payload:{row[0]}")
                        manifest_value = int(
                            _public_event_manifest_hash(
                                event_id=row[0],
                                run_id=row[1],
                                message_id=row[2],
                                sub_index=row[3],
                                stream=row[4],
                                event_type=row[5],
                                symbol=row[6],
                                condition_id=row[7],
                                asset_id=row[8],
                                source_time_ms=row[9],
                                publisher_time_ms=row[10],
                                event_sha256=row[12],
                            ),
                            16,
                        )
                        observed_event_manifest_xor ^= manifest_value
                        observed_event_manifest_sum = (
                            observed_event_manifest_sum + manifest_value
                        ) % (1 << 256)
                    notify("integrity-public-events")
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"event_manifest_validation_failed:{exc.__class__.__name__}:{exc}"
                )
            if event_count != expected_event_count:
                errors.append(
                    f"normalized_event_count_mismatch:{run_id}:"
                    f"{event_count}:{expected_event_count}"
                )
            if observed_event_manifest_xor != expected_event_manifest_xor:
                errors.append(f"normalized_event_manifest_xor_mismatch:{run_id}")
            if observed_event_manifest_sum != expected_event_manifest_sum:
                errors.append(f"normalized_event_manifest_sum_mismatch:{run_id}")
        else:
            expected_event_count = int(
                connection.execute(
                    "SELECT count(*) FROM polymarket_public_event WHERE run_id = ?",
                    [run_id],
                ).fetchone()[0]
            )
            for batch in _query_batches(
                connection,
                """
                SELECT message_id, run_id, stream, connection_id,
                       sequence_number, raw_payload_sha256, raw_text
                FROM polymarket_raw_message WHERE run_id = ?
                """,
                [run_id],
            ):
                message_count += len(batch)
                for (
                    message_id,
                    message_run,
                    stream,
                    connection_id,
                    sequence,
                    claimed,
                    raw_text,
                ) in batch:
                    actual = hashlib.sha256(str(raw_text).encode("utf-8")).hexdigest()
                    if actual != str(claimed):
                        errors.append(f"raw_message_hash_mismatch:{message_id}")
                    expected_id = _canonical_sha256(
                        {
                            "run_id": str(message_run),
                            "stream": str(stream),
                            "connection_id": str(connection_id),
                            "sequence_number": int(sequence),
                            "raw_payload_sha256": str(claimed),
                        }
                    )
                    if not hmac.compare_digest(str(message_id), expected_id):
                        errors.append(f"raw_message_id_mismatch:{message_id}")
                notify("integrity-raw-messages")
            notify("integrity-public-events", force=True)
            try:
                for _event in self.iter_public_events(run_id, ordered=False):
                    event_count += 1
                    notify("integrity-public-events")
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"event_source_reconstruction_failed:{exc.__class__.__name__}:{exc}"
                )
            if event_count != expected_event_count:
                errors.append(
                    f"normalized_event_count_mismatch:{run_id}:"
                    f"{event_count}:{expected_event_count}"
                )
        notify("integrity-relational-checks", force=True)
        if storage_schema_version != POLYMARKET_STORAGE_SCHEMA_VERSION:
            orphan_events = connection.execute(
                """
                SELECT e.event_id
                FROM polymarket_public_event AS e
                WHERE e.run_id = ? AND NOT EXISTS (
                    SELECT 1 FROM polymarket_raw_message AS r
                    WHERE r.run_id = e.run_id AND r.message_id = e.message_id
                )
                ORDER BY e.event_id
                """,
                [run_id],
            ).fetchall()
            errors.extend(f"event_without_message:{row[0]}" for row in orphan_events)
        if not compact and not capture_framed:
            invalid_messages = connection.execute(
                """
                SELECT message_id FROM polymarket_raw_message
                WHERE run_id = ? AND parse_status = 'invalid' ORDER BY message_id
                """,
                [run_id],
            ).fetchall()
            errors.extend(
                f"invalid_stream_message:{row[0]}" for row in invalid_messages
            )
        snapshots = connection.execute(
            """
            SELECT snapshot_id, run_id, observed_wall_ms, observed_monotonic_ns,
                   asset, market_id, condition_id, slug, question, event_start_ms,
                   end_ms, up_token_id, down_token_id, tick_size,
                   minimum_order_size, fees_enabled, fee_rate, fee_exponent,
                   fee_taker_only, fee_rebate_rate, liquidity_quote, volume_quote,
                   resolution_source, gamma_payload_json, gamma_payload_sha256,
                   clob_info_json, clob_info_sha256, up_fee_rate_json,
                   up_fee_rate_sha256, down_fee_rate_json, down_fee_rate_sha256,
                   maker_base_fee, taker_base_fee, taker_order_delay_enabled,
                   minimum_order_age_seconds, snapshot_payload_json, snapshot_sha256
            FROM polymarket_market_snapshot WHERE run_id = ? ORDER BY snapshot_id
            """,
            [run_id],
        ).fetchall()
        for row in snapshots:
            errors.extend(_snapshot_integrity_errors(row))
        gaps = connection.execute(
            """
            SELECT gap_id, run_id, stream, connection_id, opened_at_ms, reason,
                   last_sequence_number
            FROM polymarket_stream_gap WHERE run_id = ? ORDER BY gap_id
            """,
            [run_id],
        ).fetchall()
        for gap_id, gap_run, stream, connection_id, opened_at, reason, sequence in gaps:
            expected_id = _canonical_sha256(
                {
                    "run_id": str(gap_run),
                    "stream": str(stream),
                    "connection_id": str(connection_id),
                    "opened_at_ms": int(opened_at),
                    "reason": str(reason),
                    "last_sequence_number": int(sequence),
                }
            )
            if not hmac.compare_digest(str(gap_id), expected_id):
                errors.append(f"stream_gap_id_mismatch:{gap_id}")
        if parsed_report is not None:
            actual_stream_counts = (
                dict(sorted(stream_counts_from_audit.items()))
                if capture_framed
                else {
                    str(stream): int(count)
                    for stream, count in connection.execute(
                        """
                        SELECT stream, count(*) FROM polymarket_raw_message
                        WHERE run_id = ? GROUP BY stream ORDER BY stream
                        """,
                        [run_id],
                    ).fetchall()
                }
            )
            actual_assets = sorted({str(row[4]) for row in snapshots})
            actual_conditions = sorted({str(row[6]) for row in snapshots})
            expected_values: list[tuple[str, object, object]] = [
                ("started_at_ms", int(run_started), parsed_report.get("started_at_ms")),
                ("ended_at_ms", int(run_ended or 0), parsed_report.get("ended_at_ms")),
                (
                    "market_snapshot_count",
                    len(snapshots),
                    parsed_report.get("market_snapshot_count"),
                ),
                (
                    "raw_message_count",
                    message_count,
                    parsed_report.get("raw_message_count"),
                ),
                (
                    "normalized_event_count",
                    event_count,
                    parsed_report.get("normalized_event_count"),
                ),
                ("stream_gap_count", len(gaps), parsed_report.get("stream_gap_count")),
                (
                    "stream_counts",
                    actual_stream_counts,
                    parsed_report.get("stream_counts"),
                ),
                ("assets", actual_assets, parsed_report.get("assets")),
                ("conditions", actual_conditions, parsed_report.get("conditions")),
            ]
            if capture_framed:
                expected_values.append(
                    (
                        "evidence_manifest_sha256",
                        self._capture_manifest_sha256(run_id),
                        parsed_report.get("evidence_manifest_sha256"),
                    )
                )
            for field, actual, reported in expected_values:
                if actual != reported:
                    errors.append(f"recorder_report_evidence_mismatch:{run_id}:{field}")
        if str(run_status) != "running" or _skip_terminal_report_validation:
            if run_ended is None:
                errors.append(f"recorder_run_missing_end:{run_id}")
            else:
                out_of_window_messages = (
                    out_of_window_message_count
                    if capture_framed
                    else int(
                        connection.execute(
                            """
                            SELECT count(*) FROM polymarket_raw_message
                            WHERE run_id = ?
                              AND (received_wall_ms < ? OR received_wall_ms > ?)
                            """,
                            [run_id, int(run_started), int(run_ended)],
                        ).fetchone()[0]
                    )
                )
                out_of_window_snapshots = int(
                    connection.execute(
                        """
                        SELECT count(*) FROM polymarket_market_snapshot
                        WHERE run_id = ?
                          AND (observed_wall_ms < ? OR observed_wall_ms > ?)
                        """,
                        [run_id, int(run_started), int(run_ended)],
                    ).fetchone()[0]
                )
                if out_of_window_messages:
                    errors.append(
                        f"recorder_message_outside_run:{run_id}:{out_of_window_messages}"
                    )
                if out_of_window_snapshots:
                    errors.append(
                        f"recorder_snapshot_outside_run:{run_id}:{out_of_window_snapshots}"
                    )
        if not capture_framed:
            self._evidence_audit_summary[run_id] = _EvidenceAuditSummary(
                raw_message_count=message_count,
                normalized_event_count=event_count,
                stream_counts=(
                    {
                        str(stream): int(count)
                        for stream, count in connection.execute(
                            """
                            SELECT stream, count(*) FROM polymarket_raw_message
                            WHERE run_id = ? GROUP BY stream ORDER BY stream
                            """,
                            [run_id],
                        ).fetchall()
                    }
                ),
            )
        if str(run_status) != "running" and not _skip_terminal_report_validation:
            errors.extend(
                self._terminal_recovery_integrity_errors(run_id, parsed_report)
            )
        evidence_result = tuple(errors)
        if str(run_status) != "running" and not _skip_terminal_report_validation:
            self._terminal_evidence_integrity_cache[run_id] = (
                self._terminal_evidence_fingerprint(run_id),
                evidence_result,
            )
        if self.paper_journal is not None:
            errors.extend(self.paper_journal.integrity_errors())
        notify("integrity-complete", force=True)
        return tuple(errors)


def _snapshot_integrity_errors(row: Sequence[object]) -> tuple[str, ...]:
    snapshot_id = str(row[0])
    market = {
        "schema_version": POLYMARKET_MARKET_SCHEMA_VERSION,
        "asset": str(row[4]),
        "market_id": str(row[5]),
        "condition_id": str(row[6]),
        "slug": str(row[7]),
        "question": str(row[8]),
        "event_start_ms": int(row[9]),
        "end_ms": int(row[10]),
        "up_token_id": str(row[11]),
        "down_token_id": str(row[12]),
        "tick_size": str(row[13]),
        "minimum_order_size": str(row[14]),
        "fees_enabled": bool(row[15]),
        "fee_rate": str(row[16]),
        "fee_exponent": int(row[17]),
        "fee_taker_only": bool(row[18]),
        "fee_rebate_rate": str(row[19]),
        "liquidity_quote": str(row[20]),
        "volume_quote": str(row[21]),
        "resolution_source": str(row[22]),
        "gamma_payload_sha256": str(row[24]),
    }
    identity = {
        "run_id": str(row[1]),
        "observed_wall_ms": int(row[2]),
        "market": market,
        "clob_info_sha256": str(row[26]),
        "up_fee_rate_sha256": str(row[28]),
        "down_fee_rate_sha256": str(row[30]),
    }
    expected_payload = {
        **identity,
        "observed_monotonic_ns": int(row[3]),
        "maker_base_fee": int(row[31]),
        "taker_base_fee": int(row[32]),
        "taker_order_delay_enabled": bool(row[33]),
        "minimum_order_age_seconds": int(row[34]),
    }
    errors: list[str] = []
    for label, payload_json, claimed_sha in (
        ("gamma", str(row[23]), str(row[24])),
        ("clob", str(row[25]), str(row[26])),
        ("up_fee", str(row[27]), str(row[28])),
        ("down_fee", str(row[29]), str(row[30])),
    ):
        try:
            _validated_canonical_payload(label, payload_json, claimed_sha)
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            errors.append(f"snapshot_component_invalid:{snapshot_id}:{label}:{exc}")
    expected_id = _canonical_sha256(identity)
    if not hmac.compare_digest(snapshot_id, expected_id):
        errors.append(f"snapshot_id_mismatch:{snapshot_id}")
    expected_json = _canonical_json(expected_payload)
    if expected_json != str(row[35]):
        errors.append(f"snapshot_payload_mismatch:{snapshot_id}")
    expected_sha = hashlib.sha256(expected_json.encode("ascii")).hexdigest()
    if not hmac.compare_digest(expected_sha, str(row[36])):
        errors.append(f"snapshot_hash_mismatch:{snapshot_id}")
    return tuple(errors)


def _event_index(
    stream: str,
    event: Mapping[str, object],
    *,
    canonicalize_chainlink_topic: bool = True,
) -> dict[str, object]:
    event_type = ""
    symbol = ""
    condition_id = ""
    asset_id = ""
    source_time_ms: int | None = None
    publisher_time_ms: int | None = None
    if stream in {"clob_market", "clob_rest_book"}:
        event_type = str(event.get("event_type") or "book")
        condition_id = str(event.get("market") or "").lower()
        asset_id = str(event.get("asset_id") or "")
        source_time_ms = _safe_int(event.get("timestamp"))
        publisher_time_ms = source_time_ms
    elif stream == "polymarket_rtds":
        topic = str(event.get("topic") or "")
        message_type = str(event.get("type") or "")
        publisher_time_ms = _safe_int(event.get("timestamp"))
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            raw_symbol = str(payload.get("symbol") or "").lower()
            # RTDS currently labels Chainlink bootstrap snapshots as
            # ``crypto_prices``; its slash-delimited symbol remains unambiguous.
            if (
                canonicalize_chainlink_topic
                and topic == "crypto_prices"
                and "/" in raw_symbol
            ):
                topic = "crypto_prices_chainlink"
            symbol = (
                raw_symbol.split("/")[0].upper()
                if "/" in raw_symbol
                else raw_symbol.removesuffix("usdt").upper()
            )
            source_time_ms = _safe_int(payload.get("timestamp"))
        event_type = f"{topic}:{message_type}".strip(":") or "unknown"
    elif stream == "binance_spot":
        stream_name = str(event.get("stream") or "").lower()
        symbol = stream_name.split("@")[0].removesuffix("usdt").upper()
        payload = event.get("data")
        if isinstance(payload, Mapping):
            event_type = str(
                payload.get("e") or stream_name.split("@")[-1] or "unknown"
            )
            publisher_time_ms = _safe_int(payload.get("E"))
            source_time_ms = _safe_int(payload.get("T")) or publisher_time_ms
    return {
        "event_type": event_type or "unknown",
        "symbol": symbol,
        "condition_id": condition_id,
        "asset_id": asset_id,
        "source_time_ms": source_time_ms,
        "publisher_time_ms": publisher_time_ms,
    }


class _MarketRegistry:
    def __init__(self) -> None:
        self.markets: dict[str, PolymarketFiveMinuteMarket] = {}
        self.changed = asyncio.Event()

    def update(
        self,
        discovered: Sequence[PolymarketFiveMinuteMarket],
        *,
        now_ms: int,
    ) -> tuple[PolymarketFiveMinuteMarket, ...]:
        previous_ids = set(self.markets)
        retained = {
            condition: market
            for condition, market in self.markets.items()
            if market.end_ms + 900_000 > now_ms
        }
        new_markets = tuple(
            market for market in discovered if market.condition_id not in self.markets
        )
        retained.update({market.condition_id: market for market in discovered})
        self.markets = retained
        if set(self.markets) != previous_ids:
            self.changed.set()
        return new_markets

    def token_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                token for market in self.markets.values() for token in market.token_ids
            )
        )


class PolymarketPublicRecorder:
    """Run independent public streams into one fail-closed evidence store."""

    def __init__(
        self,
        database: str | Path,
        *,
        client: PolymarketPublicClient | None = None,
        queue_capacity: int = 500_000,
        discovery_interval_seconds: int = 60,
        memory_limit: str = "4GB",
        database_threads: int = 2,
    ) -> None:
        self.database = Path(database)
        self.client = client or PolymarketPublicClient()
        self.queue_capacity = int(queue_capacity)
        if self.queue_capacity < 1_000 or self.queue_capacity > 1_000_000:
            raise ValueError("queue_capacity must lie in [1000, 1000000]")
        self.discovery_interval_seconds = int(discovery_interval_seconds)
        if (
            self.discovery_interval_seconds < 30
            or self.discovery_interval_seconds > 300
        ):
            raise ValueError("discovery_interval_seconds must lie in [30, 300]")
        self.memory_limit = str(memory_limit).upper()
        if not _DUCKDB_MEMORY_LIMIT.fullmatch(self.memory_limit):
            raise ValueError("memory_limit must be a positive KB, MB, GB, or TB value")
        self.database_threads = int(database_threads)
        if self.database_threads < 1 or self.database_threads > 8:
            raise ValueError("database_threads must lie in [1, 8]")
        self.registry = _MarketRegistry()
        self.errors: list[str] = []
        self._written_message_count = 0
        self._written_market_snapshot_count = 0
        self._written_gap_count = 0
        self._written_gap_counts: dict[str, int] = {}
        self._last_written_gap: dict[str, object] | None = None
        self._received_message_count = 0
        self._received_stream_counts: dict[str, int] = {}
        self._queue_high_watermark = 0

    def _record_queue_saturation(self) -> None:
        if self._queue_high_watermark < self.queue_capacity:
            return
        detail = (
            "evidence_queue_saturated:"
            f"{self._queue_high_watermark}/{self.queue_capacity}"
        )
        if detail not in self.errors:
            self.errors.append(detail)

    def _record_written_gap(self, gap: StreamGap) -> None:
        lane, separator, suffix = gap.connection_id.rpartition(":")
        if not separator or not re.fullmatch(r"[0-9a-f]{32}", suffix):
            lane = gap.connection_id
        self._written_gap_count += 1
        self._written_gap_counts[gap.stream] = (
            self._written_gap_counts.get(gap.stream, 0) + 1
        )
        self._last_written_gap = {
            "stream": gap.stream,
            "lane": lane[:128],
            "opened_at_ms": gap.opened_at_ms,
            "reason": gap.reason,
            "last_sequence_number": gap.last_sequence_number,
        }

    def _notify_progress(
        self,
        progress: Callable[[str, Mapping[str, object]], None] | None,
        phase: str,
        *,
        run_id: str,
        started_at_ms: int,
        duration_seconds: int,
        queue_size: int,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if progress is None:
            return
        observed_at_ms = _wall_ms()
        payload: dict[str, object] = {
            "schema_version": POLYMARKET_RECORDER_PROGRESS_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": phase,
            "observed_at_ms": observed_at_ms,
            "elapsed_seconds": max(0.0, (observed_at_ms - started_at_ms) / 1_000.0),
            "duration_seconds": int(duration_seconds),
            "written_message_count": self._written_message_count,
            "written_market_snapshot_count": self._written_market_snapshot_count,
            "written_gap_count": self._written_gap_count,
            "written_gap_counts": dict(sorted(self._written_gap_counts.items())),
            "last_written_gap": (
                dict(self._last_written_gap)
                if self._last_written_gap is not None
                else None
            ),
            "received_message_count": self._received_message_count,
            "received_stream_counts": dict(
                sorted(self._received_stream_counts.items())
            ),
            "queue_capacity": self.queue_capacity,
            "queue_high_watermark": self._queue_high_watermark,
            "queue_size": max(0, int(queue_size)),
            "error_count": len(self.errors),
        }
        if details:
            payload.update(details)
        try:
            progress(phase, payload)
        except Exception:
            # Operator telemetry cannot interrupt or invalidate captured evidence.
            return

    async def run(
        self,
        *,
        duration_seconds: int,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        progress_interval_seconds: int = 30,
    ) -> RecorderReport:
        duration = int(duration_seconds)
        if duration < 5 or duration > 86_400:
            raise ValueError("duration_seconds must lie in [5, 86400]")
        progress_interval = int(progress_interval_seconds)
        if progress_interval < 5 or progress_interval > 300:
            raise ValueError("progress_interval_seconds must lie in [5, 300]")
        self.registry = _MarketRegistry()
        self.errors = []
        self._written_message_count = 0
        self._written_market_snapshot_count = 0
        self._written_gap_count = 0
        self._written_gap_counts = {}
        self._last_written_gap = None
        self._received_message_count = 0
        self._received_stream_counts = {}
        self._queue_high_watermark = 0
        run_id = uuid.uuid4().hex
        started = _wall_ms()
        stop = asyncio.Event()
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None] = (
            asyncio.Queue(maxsize=self.queue_capacity)
        )
        with PolymarketEvidenceStore(
            self.database,
            memory_limit=self.memory_limit,
            threads=self.database_threads,
        ) as store:
            store.start_run(run_id, started)
            self._notify_progress(
                progress,
                "capture-started",
                run_id=run_id,
                started_at_ms=started,
                duration_seconds=duration,
                queue_size=output.qsize(),
            )
            writer = asyncio.create_task(self._writer(run_id, store, output))
            progress_task = (
                asyncio.create_task(
                    self._progress_loop(
                        progress,
                        stop,
                        output,
                        run_id=run_id,
                        started_at_ms=started,
                        duration_seconds=duration,
                        interval_seconds=progress_interval,
                    )
                )
                if progress is not None
                else None
            )
            producers: list[asyncio.Task[None]] = []
            try:
                try:
                    await asyncio.wait_for(
                        self._discover(run_id, output, now_ms=started),
                        timeout=min(30.0, float(duration)),
                    )
                except Exception as exc:
                    self.errors.append(
                        f"initial_discovery:{exc.__class__.__name__}:{exc}"
                    )
                    stop.set()
                if not stop.is_set():
                    producers = [
                        asyncio.create_task(
                            self._supervise(
                                "discovery",
                                self._discovery_loop(run_id, output, stop),
                                stop,
                            )
                        ),
                        asyncio.create_task(
                            self._supervise(
                                "clob_market", self._clob_stream(output, stop), stop
                            )
                        ),
                        asyncio.create_task(
                            self._supervise(
                                "polymarket_rtds", self._rtds_stream(output, stop), stop
                            )
                        ),
                        asyncio.create_task(
                            self._supervise(
                                "binance_spot", self._binance_stream(output, stop), stop
                            )
                        ),
                    ]
                    stopped = asyncio.create_task(stop.wait())
                    done, _ = await asyncio.wait(
                        {stopped, writer},
                        timeout=max(
                            0.0,
                            duration - ((_wall_ms() - started) / 1_000.0),
                        ),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if writer in done:
                        writer.result()
                        self.errors.append("writer:RuntimeError:writer_stopped_early")
                    if stopped not in done:
                        stopped.cancel()
                    await asyncio.gather(stopped, return_exceptions=True)
            except Exception as exc:
                self.errors.append(f"recorder:{exc.__class__.__name__}:{exc}")
            finally:
                stop.set()
                if progress_task is not None:
                    progress_task.cancel()
                    await asyncio.gather(progress_task, return_exceptions=True)
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
                if not writer.done():
                    try:
                        self._notify_progress(
                            progress,
                            "writer-draining",
                            run_id=run_id,
                            started_at_ms=started,
                            duration_seconds=duration,
                            queue_size=output.qsize(),
                        )
                        await asyncio.wait_for(output.put(None), timeout=5.0)
                        await _wait_for_writer_drain(writer, output)
                    except Exception as exc:
                        self.errors.append(
                            f"writer_shutdown:{exc.__class__.__name__}:{exc}"
                        )
                if not writer.done():
                    writer.cancel()
                    await asyncio.gather(writer, return_exceptions=True)
                elif not writer.cancelled():
                    writer_exception = writer.exception()
                    if writer_exception is not None:
                        detail = (
                            f"writer:{writer_exception.__class__.__name__}:"
                            f"{writer_exception}"
                        )
                        if detail not in self.errors:
                            self.errors.append(detail)
            self._record_queue_saturation()
            ended = _wall_ms()

            def audit_progress(
                phase: str,
                details: Mapping[str, object],
            ) -> None:
                self._notify_progress(
                    progress,
                    phase,
                    run_id=run_id,
                    started_at_ms=started,
                    duration_seconds=duration,
                    queue_size=output.qsize(),
                    details=details,
                )

            try:
                report = store.finish_run(
                    run_id,
                    started_at_ms=started,
                    ended_at_ms=ended,
                    database=str(self.database.resolve()),
                    errors=self.errors,
                    progress=audit_progress,
                    progress_interval_seconds=progress_interval,
                )
            except Exception as exc:
                self.errors.append(f"finish_run:{exc.__class__.__name__}:{exc}")
                report = store.fail_run(
                    run_id,
                    started_at_ms=started,
                    ended_at_ms=ended,
                    database=str(self.database.resolve()),
                    errors=self.errors,
                )
            self._notify_progress(
                progress,
                "finalized",
                run_id=run_id,
                started_at_ms=started,
                duration_seconds=duration,
                queue_size=output.qsize(),
                details={
                    "status": report.status,
                    "report_sha256": report.report_sha256,
                    "raw_message_count": report.raw_message_count,
                    "normalized_event_count": report.normalized_event_count,
                    "verified_raw_message_count": report.raw_message_count,
                    "verified_event_count": report.normalized_event_count,
                    "stream_gap_count": report.stream_gap_count,
                },
            )
            return report

    async def _emit_output(
        self,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        item: RawStreamMessage | StreamGap | MarketEvidence,
    ) -> None:
        try:
            await asyncio.wait_for(
                output.put(item),
                timeout=_OUTPUT_PUT_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"evidence queue remained full for {_OUTPUT_PUT_TIMEOUT_SECONDS:.1f}s"
            ) from exc

    async def _emit_raw_message(
        self,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        message: RawStreamMessage,
    ) -> None:
        validated = message.validated()
        await self._emit_output(output, validated)
        self._received_message_count += 1
        self._received_stream_counts[validated.stream] = (
            self._received_stream_counts.get(validated.stream, 0) + 1
        )
        self._queue_high_watermark = max(
            self._queue_high_watermark,
            output.qsize(),
        )

    async def _progress_loop(
        self,
        progress: Callable[[str, Mapping[str, object]], None],
        stop: asyncio.Event,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        *,
        run_id: str,
        started_at_ms: int,
        duration_seconds: int,
        interval_seconds: int,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=int(interval_seconds))
                return
            except TimeoutError:
                self._notify_progress(
                    progress,
                    "capturing",
                    run_id=run_id,
                    started_at_ms=started_at_ms,
                    duration_seconds=duration_seconds,
                    queue_size=output.qsize(),
                )

    async def _supervise(
        self,
        name: str,
        operation: Awaitable[None],
        stop: asyncio.Event,
    ) -> None:
        try:
            await operation
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.errors.append(f"{name}:{exc.__class__.__name__}:{exc}")
            stop.set()

    async def _discover(
        self,
        run_id: str,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        *,
        now_ms: int,
    ) -> None:
        markets = await asyncio.to_thread(
            self.client.discover_five_minute_markets,
            now_ms=now_ms,
            include_next=True,
            require_all_assets=True,
        )
        new_markets = self.registry.update(markets, now_ms=now_ms)
        for market in new_markets:
            evidence, books = await asyncio.to_thread(
                self._fetch_market_evidence,
                market,
                now_ms,
            )
            await self._emit_output(output, evidence)
            for sequence, raw in enumerate(books, start=1):
                await self._emit_raw_message(
                    output,
                    RawStreamMessage(
                        stream="clob_rest_book",
                        connection_id=f"rest-{run_id}-{market.condition_id}",
                        sequence_number=sequence,
                        received_wall_ms=raw[0],
                        received_monotonic_ns=raw[1],
                        raw_text=raw[2],
                    ),
                )

    def _fetch_market_evidence(
        self,
        market: PolymarketFiveMinuteMarket,
        now_ms: int,
    ) -> tuple[MarketEvidence, list[tuple[int, int, str]]]:
        observed_wall = _wall_ms()
        observed_monotonic = _monotonic_ns()
        clob = self.client.clob_market_info(market.condition_id)
        clob_evidence = validate_clob_market_info(market, clob)
        up_fee = dict(self.client.fee_rate(market.up_token_id))
        down_fee = dict(self.client.fee_rate(market.down_token_id))
        up_fee_json = _canonical_json(up_fee)
        down_fee_json = _canonical_json(down_fee)
        books: list[tuple[int, int, str]] = []
        if market.event_start_ms <= now_ms < market.end_ms:
            for token in market.token_ids:
                book_payload = dict(self.client.order_book(token))
                wall = _wall_ms()
                monotonic = _monotonic_ns()
                validate_clob_order_book(
                    market,
                    token,
                    book_payload,
                    received_wall_ms=wall,
                    received_monotonic_ns=monotonic,
                )
                books.append((wall, monotonic, _canonical_json(book_payload)))
        return (
            MarketEvidence(
                market=market,
                observed_wall_ms=observed_wall,
                observed_monotonic_ns=observed_monotonic,
                clob_info_json=str(clob_evidence["payload_json"]),
                clob_info_sha256=str(clob_evidence["payload_sha256"]),
                up_fee_rate_json=up_fee_json,
                up_fee_rate_sha256=hashlib.sha256(
                    up_fee_json.encode("ascii")
                ).hexdigest(),
                down_fee_rate_json=down_fee_json,
                down_fee_rate_sha256=hashlib.sha256(
                    down_fee_json.encode("ascii")
                ).hexdigest(),
                maker_base_fee=int(clob_evidence["maker_base_fee"]),
                taker_base_fee=int(clob_evidence["taker_base_fee"]),
                taker_order_delay_enabled=bool(
                    clob_evidence["taker_order_delay_enabled"]
                ),
                minimum_order_age_seconds=int(
                    clob_evidence["minimum_order_age_seconds"]
                ),
            ),
            books,
        )

    async def _discovery_loop(
        self,
        run_id: str,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.discovery_interval_seconds
                )
                return
            except TimeoutError:
                await self._discover(run_id, output, now_ms=_wall_ms())

    async def _writer(
        self,
        run_id: str,
        store: PolymarketEvidenceStore,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
    ) -> None:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="polymarket-evidence-writer",
        )
        writer_store: PolymarketEvidenceStore | None = None

        def open_writer_store() -> PolymarketEvidenceStore:
            opened = PolymarketEvidenceStore(
                store.path,
                memory_limit=store.memory_limit,
                threads=store.threads,
            )
            opened.connect()
            return opened

        async def invoke(function: Callable[..., object], *args: object) -> object:
            return await loop.run_in_executor(
                executor,
                partial(function, *args),
            )

        pending_messages: list[RawStreamMessage] = []

        async def flush_pending_messages() -> None:
            nonlocal pending_messages
            if not pending_messages:
                return
            batch = tuple(pending_messages)
            await invoke(writer_store.append_messages, run_id, batch)
            self._written_message_count += len(batch)
            pending_messages = []

        try:
            writer_store = await loop.run_in_executor(executor, open_writer_store)
            while True:
                item = await output.get()
                if item is None:
                    await flush_pending_messages()
                    return
                if isinstance(item, RawStreamMessage):
                    pending_messages.append(item)
                    deadline = loop.time() + _WRITER_COALESCE_SECONDS
                    control_item: StreamGap | MarketEvidence | None = None
                    control_pending = False
                    while len(pending_messages) < _WRITER_BATCH_SIZE:
                        remaining = deadline - loop.time()
                        if remaining <= 0.0:
                            break
                        try:
                            candidate = output.get_nowait()
                        except asyncio.QueueEmpty:
                            try:
                                candidate = await asyncio.wait_for(
                                    output.get(),
                                    timeout=remaining,
                                )
                            except TimeoutError:
                                break
                        if isinstance(candidate, RawStreamMessage):
                            pending_messages.append(candidate)
                            continue
                        control_item = candidate
                        control_pending = True
                        break
                    await flush_pending_messages()
                    if not control_pending:
                        continue
                    item = control_item
                    if item is None:
                        return
                if isinstance(item, StreamGap):
                    await invoke(writer_store.record_gap, run_id, item)
                    self._record_written_gap(item)
                elif isinstance(item, MarketEvidence):
                    await invoke(writer_store.record_market_evidence, run_id, item)
                    self._written_market_snapshot_count += 1
        finally:
            if writer_store is not None:
                await loop.run_in_executor(executor, writer_store.close)
            executor.shutdown(wait=True, cancel_futures=True)

    async def _clob_stream(
        self,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
    ) -> None:
        backoff = 1.0
        while not stop.is_set():
            connection_id = f"clob:{uuid.uuid4().hex}"
            sequence = 0
            connected_at = 0.0
            try:
                tokens = self.registry.token_ids()
                if not tokens:
                    raise RuntimeError("no validated Polymarket token subscriptions")
                async with connect(
                    CLOB_MARKET_WEBSOCKET,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=_CLOB_MAX_MESSAGE_BYTES,
                    max_queue=_CLOB_MAX_QUEUE_FRAMES,
                    compression=None,
                ) as websocket:
                    connected_at = asyncio.get_running_loop().time()
                    current = set(tokens)
                    await websocket.send(
                        _canonical_json(
                            {
                                "assets_ids": sorted(current),
                                "type": "market",
                                "custom_feature_enabled": True,
                            }
                        )
                    )
                    heartbeat_task = asyncio.create_task(
                        _periodic_text_heartbeat(websocket, stop, "PING", 10.0)
                    )
                    receive = asyncio.create_task(websocket.recv())
                    changed = asyncio.create_task(self.registry.changed.wait())
                    stopping = asyncio.create_task(stop.wait())
                    try:
                        while not stop.is_set():
                            done, _ = await asyncio.wait(
                                {receive, changed, stopping, heartbeat_task},
                                timeout=_STREAM_INACTIVITY_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                raise RuntimeError(
                                    "CLOB market stream exceeded the inactivity bound"
                                )
                            if heartbeat_task in done:
                                heartbeat_task.result()
                                if stop.is_set():
                                    return
                                raise RuntimeError("CLOB heartbeat stopped early")
                            if stopping in done and stopping.result():
                                return
                            if receive in done:
                                raw = receive.result()
                                next_sequence = sequence + 1
                                await self._emit_raw_message(
                                    output,
                                    RawStreamMessage(
                                        "clob_market",
                                        connection_id,
                                        next_sequence,
                                        _wall_ms(),
                                        _monotonic_ns(),
                                        _text_frame(raw),
                                    ),
                                )
                                sequence = next_sequence
                                receive = asyncio.create_task(websocket.recv())
                            if changed in done and changed.result():
                                self.registry.changed.clear()
                                desired = set(self.registry.token_ids())
                                additions = sorted(desired - current)
                                removals = sorted(current - desired)
                                if additions:
                                    await websocket.send(
                                        _canonical_json(
                                            {
                                                "assets_ids": additions,
                                                "operation": "subscribe",
                                                "custom_feature_enabled": True,
                                            }
                                        )
                                    )
                                if removals:
                                    await websocket.send(
                                        _canonical_json(
                                            {
                                                "assets_ids": removals,
                                                "operation": "unsubscribe",
                                            }
                                        )
                                    )
                                current = desired
                                changed = asyncio.create_task(
                                    self.registry.changed.wait()
                                )
                    finally:
                        tasks = (receive, changed, stopping, heartbeat_task)
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit_output(
                    output,
                    StreamGap(
                        "clob_market",
                        connection_id,
                        _wall_ms(),
                        f"{exc.__class__.__name__}:{exc}",
                        sequence,
                    ),
                )
                if (
                    connected_at > 0.0
                    and asyncio.get_running_loop().time() - connected_at
                    >= _STABLE_CONNECTION_SECONDS
                ):
                    backoff = 1.0
                await _bounded_backoff(stop, backoff)
                backoff = min(30.0, backoff * 2.0)

    async def _rtds_stream(
        self,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
    ) -> None:
        # The live endpoint's filter behavior differs from the current web docs.
        # Keep this wire contract aligned with the bounded probe artifact.
        assets = ("btc", "eth", "sol")
        subscriptions = (
            *(
                (
                    f"rtds:binance:{asset}",
                    _canonical_json(
                        {
                            "action": "subscribe",
                            "subscriptions": [
                                {
                                    "topic": "crypto_prices",
                                    "type": "update",
                                    "filters": _canonical_json(
                                        {"symbol": f"{asset.upper()}USDT"}
                                    ),
                                }
                            ],
                        }
                    ),
                )
                for asset in assets
            ),
            *(
                (
                    f"rtds:chainlink:{asset}",
                    _canonical_json(
                        {
                            "action": "subscribe",
                            "subscriptions": [
                                {
                                    "topic": "crypto_prices_chainlink",
                                    "type": "update",
                                    "filters": _canonical_json(
                                        {"symbol": f"{asset}/usd"}
                                    ),
                                }
                            ],
                        }
                    ),
                )
                for asset in assets
            ),
        )
        # Keep every topic/symbol subscription isolated. RTDS keys subscriptions
        # by topic and can otherwise replace one asset while acknowledging all.
        async with asyncio.TaskGroup() as task_group:
            for lane, subscription in subscriptions:
                task_group.create_task(
                    self._simple_stream(
                        stream="polymarket_rtds",
                        lane=lane,
                        url=POLYMARKET_RTDS_WEBSOCKET,
                        subscription=subscription,
                        heartbeat="PING",
                        heartbeat_seconds=5.0,
                        output=output,
                        stop=stop,
                    )
                )

    async def _binance_stream(
        self,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
    ) -> None:
        await self._simple_stream(
            stream="binance_spot",
            lane="binance:combined:btc-eth-sol",
            url=BINANCE_SPOT_WEBSOCKET,
            subscription=None,
            heartbeat=None,
            heartbeat_seconds=20.0,
            output=output,
            stop=stop,
        )

    async def _simple_stream(
        self,
        *,
        stream: str,
        lane: str,
        url: str,
        subscription: str | None,
        heartbeat: str | None,
        heartbeat_seconds: float,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
    ) -> None:
        normalized_lane = str(lane or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9:_-]{0,95}", normalized_lane):
            raise ValueError("public stream lane is invalid")
        backoff = 1.0
        while not stop.is_set():
            connection_id = f"{normalized_lane}:{uuid.uuid4().hex}"
            sequence = 0
            connected_at = 0.0
            try:
                async with connect(
                    url,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=_SIMPLE_STREAM_MAX_MESSAGE_BYTES,
                    max_queue=_SIMPLE_STREAM_MAX_QUEUE_FRAMES,
                    compression=None,
                ) as websocket:
                    connected_at = asyncio.get_running_loop().time()
                    if subscription is not None:
                        await websocket.send(subscription)
                    heartbeat_task = (
                        asyncio.create_task(
                            _periodic_text_heartbeat(
                                websocket,
                                stop,
                                heartbeat,
                                heartbeat_seconds,
                            )
                        )
                        if heartbeat is not None
                        else None
                    )
                    receive = asyncio.create_task(websocket.recv())
                    stopping = asyncio.create_task(stop.wait())
                    try:
                        while not stop.is_set():
                            watched: set[asyncio.Task[object]] = {receive, stopping}
                            if heartbeat_task is not None:
                                watched.add(heartbeat_task)
                            done, _ = await asyncio.wait(
                                watched,
                                timeout=_STREAM_INACTIVITY_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                raise RuntimeError(
                                    f"{stream} exceeded the inactivity bound"
                                )
                            if heartbeat_task is not None and heartbeat_task in done:
                                heartbeat_task.result()
                                if stop.is_set():
                                    return
                                raise RuntimeError(f"{stream} heartbeat stopped early")
                            if stopping in done and stopping.result():
                                return
                            raw = receive.result()
                            next_sequence = sequence + 1
                            text = _text_frame(raw)
                            await self._emit_raw_message(
                                output,
                                RawStreamMessage(
                                    stream,
                                    connection_id,
                                    next_sequence,
                                    _wall_ms(),
                                    _monotonic_ns(),
                                    text,
                                ),
                            )
                            sequence = next_sequence
                            if text == "PING":
                                await websocket.send("PONG")
                            receive = asyncio.create_task(websocket.recv())
                    finally:
                        tasks = [receive, stopping]
                        if heartbeat_task is not None:
                            tasks.append(heartbeat_task)
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._emit_output(
                    output,
                    StreamGap(
                        stream,
                        connection_id,
                        _wall_ms(),
                        f"{exc.__class__.__name__}:{exc}",
                        sequence,
                    ),
                )
                if (
                    connected_at > 0.0
                    and asyncio.get_running_loop().time() - connected_at
                    >= _STABLE_CONNECTION_SECONDS
                ):
                    backoff = 1.0
                await _bounded_backoff(stop, backoff)
                backoff = min(30.0, backoff * 2.0)


def _text_frame(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="strict")
    return str(raw)


async def _periodic_text_heartbeat(
    websocket: _TextSender,
    stop: asyncio.Event,
    message: str,
    seconds: float,
) -> None:
    interval = float(seconds)
    if interval <= 0.0:
        raise ValueError("heartbeat interval must be positive")
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            await websocket.send(message)


async def _wait_for_writer_drain(
    writer: asyncio.Task[None],
    output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
) -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    last_progress = started
    remaining = output.qsize()
    budget = min(
        _WRITER_MAX_DRAIN_SECONDS,
        max(
            _WRITER_MIN_DRAIN_SECONDS,
            (remaining / _WRITER_ASSUMED_MINIMUM_MESSAGES_PER_SECOND) + 30.0,
        ),
    )
    while not writer.done():
        done, _ = await asyncio.wait({writer}, timeout=1.0)
        if done:
            break
        now = loop.time()
        current = output.qsize()
        if current < remaining:
            remaining = current
            last_progress = now
        if now - last_progress > _WRITER_STALL_SECONDS:
            raise TimeoutError(
                f"writer drain stalled with {current} queued evidence items"
            )
        if now - started > budget:
            raise TimeoutError(
                f"writer drain exceeded {budget:.1f}s with "
                f"{current} queued evidence items"
            )
    writer.result()


async def _bounded_backoff(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=max(0.1, float(seconds)))
    except TimeoutError:
        return


__all__ = [
    "BINANCE_SPOT_WEBSOCKET",
    "CLOB_MARKET_WEBSOCKET",
    "POLYMARKET_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_EVIDENCE_SCHEMA_VERSION",
    "POLYMARKET_RECORDER_SCHEMA_VERSION",
    "POLYMARKET_STORAGE_SCHEMA_VERSION",
    "POLYMARKET_RTDS_WEBSOCKET",
    "DecodedPublicEvent",
    "MarketEvidence",
    "PolymarketEvidenceStore",
    "PolymarketPublicRecorder",
    "RawMessageLaneSummary",
    "RawStreamMessage",
    "RecorderReport",
    "StreamGap",
]
