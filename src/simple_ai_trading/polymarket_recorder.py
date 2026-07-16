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


POLYMARKET_EVIDENCE_SCHEMA_VERSION = "polymarket-public-evidence-v1"
POLYMARKET_RECORDER_SCHEMA_VERSION = "polymarket-public-recorder-v1"
POLYMARKET_RECORDER_PROGRESS_SCHEMA_VERSION = "polymarket-recorder-progress-v1"
POLYMARKET_STORAGE_SCHEMA_VERSION = "polymarket-evidence-storage-v2"
_LEGACY_STORAGE_SCHEMA_VERSION = "polymarket-public-evidence-v1"
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
_WRITER_BATCH_SIZE = 8_192
_WRITER_COALESCE_SECONDS = 0.5
_WRITER_MIN_DRAIN_SECONDS = 60.0
_WRITER_MAX_DRAIN_SECONDS = 600.0
_WRITER_STALL_SECONDS = 30.0
_WRITER_ASSUMED_MINIMUM_MESSAGES_PER_SECOND = 100.0
_INTEGRITY_FETCH_SIZE = 4_096

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
    report_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


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
        self._next_chunk_index_by_run: dict[str, int] = {}
        self._frame_cache_id = ""
        self._frame_cache = b""
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
            self._next_chunk_index_by_run.clear()
            self._frame_cache_id = ""
            self._frame_cache = b""

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
                normalized_event_count UINTEGER,
                UNIQUE(run_id, stream, connection_id, sequence_number)
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
                event_sha256 VARCHAR NOT NULL,
                UNIQUE(message_id, sub_index)
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

            CREATE TABLE IF NOT EXISTS polymarket_ai_veto_cache (
                cache_key_sha256 VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                identity_json VARCHAR NOT NULL,
                response_json VARCHAR NOT NULL,
                response_sha256 VARCHAR NOT NULL,
                latency_seconds DOUBLE NOT NULL,
                created_at_ms BIGINT NOT NULL,
                CHECK(latency_seconds >= 0.0 AND latency_seconds <= 60.0)
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
                   response_sha256, latency_seconds
            FROM polymarket_ai_veto_cache
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
        response = _validated_canonical_payload(
            "Polymarket AI cache response",
            str(row[2]),
            str(row[3]),
        )
        latency = float(row[4])
        if not math.isfinite(latency) or not 0.0 <= latency <= 60.0:
            raise ValueError("Polymarket AI cache latency is corrupt")
        return {
            "identity": dict(identity),
            "response_payload": response,
            "response_sha256": str(row[3]),
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
            or not 0.0 <= latency <= 60.0
        ):
            raise ValueError("Polymarket AI cache entry is invalid")
        response_json = _canonical_json(response_payload)
        response_sha256 = hashlib.sha256(response_json.encode("ascii")).hexdigest()
        self.connect().execute(
            """
            INSERT INTO polymarket_ai_veto_cache (
                cache_key_sha256, schema_version, identity_json, response_json,
                response_sha256, latency_seconds, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cache_key_sha256) DO NOTHING
            """,
            [
                key,
                schema_version,
                identity_json,
                response_json,
                response_sha256,
                latency,
                _wall_ms(),
            ],
        )

    def start_run(self, run_id: str, started_at_ms: int) -> None:
        self.connect().execute(
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
            POLYMARKET_STORAGE_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported Polymarket storage schema: {version}")
        return version

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
                   (SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?)
            FROM polymarket_recorder_run AS r WHERE r.run_id = ?
            """,
                [run_id, run_id, run_id, run_id, run_id],
            )
            .fetchone()
        )
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
                "database_file_state": file_state,
                "terminal_metadata_and_counts": None
                if row is None
                else [None if value is None else str(value) for value in row],
            }
        )

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
        if self._storage_schema_version(run_id) == _LEGACY_STORAGE_SCHEMA_VERSION:
            for start in range(0, len(validated), _RAW_CHUNK_MESSAGE_LIMIT):
                self._append_legacy_message_batch(
                    run_id,
                    validated[start : start + _RAW_CHUNK_MESSAGE_LIMIT],
                )
            return
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
                self._append_compact_message_chunk(run_id, chunk, connection)
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            finally:
                if cached_chunk_index_present:
                    assert cached_chunk_index is not None
                    self._next_chunk_index_by_run[run_id] = cached_chunk_index
                else:
                    self._next_chunk_index_by_run.pop(run_id, None)
            raise

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
            connection.execute("ROLLBACK")
            raise

    def _append_compact_message_chunk(
        self,
        run_id: str,
        messages: Sequence[RawStreamMessage],
        connection: duckdb.DuckDBPyConnection,
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
                "schema_version": POLYMARKET_STORAGE_SCHEMA_VERSION,
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
            INSERT INTO polymarket_raw_chunk VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                chunk_id,
                run_id,
                POLYMARKET_STORAGE_SCHEMA_VERSION,
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
        parse_status = "ok"
        parse_error = ""
        events: list[Mapping[str, object]] = []
        try:
            parsed = _strict_json_loads(message.raw_text)
            candidates = parsed if isinstance(parsed, list) else [parsed]
            if not all(isinstance(item, Mapping) for item in candidates):
                raise ValueError("JSON stream message contains a non-object event")
            events = [item for item in candidates if isinstance(item, Mapping)]
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            parse_status = (
                "control"
                if message.raw_text.strip() in {"", "PING", "PONG"}
                else "invalid"
            )
            parse_error = (
                "" if parse_status == "control" else f"{exc.__class__.__name__}:{exc}"
            )
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

    def _load_compact_frame(self, run_id: str, chunk_id: str) -> bytes:
        if chunk_id == self._frame_cache_id:
            return self._frame_cache
        row = (
            self._payload_connection()
            .execute(
                """
            SELECT chunk_id, run_id, schema_version, chunk_index, frame_format,
                   codec, compression_level, message_count, first_message_id,
                   last_message_id, message_manifest_xor, uncompressed_bytes,
                   uncompressed_sha256, compressed_bytes, compressed_sha256,
                   compressed_payload
            FROM polymarket_raw_chunk
            WHERE run_id = ? AND chunk_id = ?
            """,
                [run_id, chunk_id],
            )
            .fetchone()
        )
        if row is None:
            raise ValueError("compact raw message references a missing chunk")
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
        if (
            str(chunk_run) != run_id
            or str(schema_version) != POLYMARKET_STORAGE_SCHEMA_VERSION
            or str(frame_format) != _RAW_CHUNK_FRAME_FORMAT
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
        self._frame_cache_id = chunk_id
        self._frame_cache = frame
        return frame

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
        compact = storage_version == POLYMARKET_STORAGE_SCHEMA_VERSION
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
            if selected_conditions is None:
                filters = ["r.run_id = ?"]
                parameters: list[object] = [run_id]
                if selected_streams is not None:
                    stream_placeholders = ", ".join("?" for _ in selected_streams)
                    filters.append(f"r.stream IN ({stream_placeholders})")
                    parameters.extend(selected_streams)
                order_clause = ""
                if ordered:
                    order_clause = """
                        ORDER BY r.received_monotonic_ns, r.received_wall_ms,
                                 r.connection_id, r.sequence_number
                    """
                cursor = self.connect().execute(
                    f"""
                    SELECT unhex(r.message_id) AS message_id_bytes
                    FROM polymarket_raw_message AS r
                    WHERE {" AND ".join(filters)}
                    {order_clause}
                    """,
                    parameters,
                )
                for key_batch in iter(
                    lambda: cursor.fetchmany(_INTEGRITY_FETCH_SIZE), []
                ):
                    message_ids = tuple(bytes(row[0]).hex() for row in key_batch)
                    placeholders = ", ".join("?" for _ in message_ids)
                    rows = (
                        self._payload_connection()
                        .execute(
                            wide_select
                            + f" WHERE e.run_id = ? AND e.message_id IN ({placeholders})",
                            [run_id, *message_ids],
                        )
                        .fetchall()
                    )
                    rows_by_message: dict[str, list[tuple[object, ...]]] = {}
                    seen_event_ids: set[str] = set()
                    for row in rows:
                        event_id = str(row[0])
                        if event_id in seen_event_ids:
                            raise ValueError("normalized event identity is duplicated")
                        seen_event_ids.add(event_id)
                        rows_by_message.setdefault(str(row[2]), []).append(row)
                    ordered_rows: list[tuple[object, ...]] = []
                    for message_id in message_ids:
                        ordered_rows.extend(
                            sorted(
                                rows_by_message.get(message_id, ()),
                                key=lambda row: (int(row[3]), str(row[0])),
                            )
                        )
                    yield ordered_rows
                return

            filters = ["e.run_id = ?"]
            parameters = [run_id]
            if selected_streams is not None:
                stream_placeholders = ", ".join("?" for _ in selected_streams)
                filters.append(f"e.stream IN ({stream_placeholders})")
                parameters.extend(selected_streams)
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
                f"""
                SELECT unhex(e.event_id) AS event_id_bytes
                FROM polymarket_public_event AS e
                JOIN polymarket_raw_message AS r
                  ON r.run_id = e.run_id AND r.message_id = e.message_id
                WHERE {" AND ".join(filters)}
                {order_clause}
                """,
                parameters,
            )
            for key_batch in iter(lambda: cursor.fetchmany(_INTEGRITY_FETCH_SIZE), []):
                event_ids = tuple(bytes(row[0]).hex() for row in key_batch)
                placeholders = ", ".join("?" for _ in event_ids)
                rows = (
                    self._payload_connection()
                    .execute(
                        wide_select
                        + f" WHERE e.run_id = ? AND e.event_id IN ({placeholders})",
                        [run_id, *event_ids],
                    )
                    .fetchall()
                )
                rows_by_id: dict[str, tuple[object, ...]] = {}
                for row in rows:
                    row_event_id = str(row[0])
                    if row_event_id in rows_by_id:
                        raise ValueError("normalized event identity is duplicated")
                    rows_by_id[row_event_id] = row
                if len(rows_by_id) != len(event_ids) or any(
                    event_id not in rows_by_id for event_id in event_ids
                ):
                    raise ValueError("normalized event batch lookup is incomplete")
                yield [rows_by_id[event_id] for event_id in event_ids]

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
    ) -> RecorderReport:
        connection = self.connect()
        market_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_market_snapshot WHERE run_id = ?",
                [run_id],
            ).fetchone()[0]
        )
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
        gap_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?",
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
        raw_count = query_count("messages", "polymarket_raw_message")
        event_count = query_count("events", "polymarket_public_event")
        gap_count = query_count("gaps", "polymarket_stream_gap")
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

    def integrity_errors(
        self,
        run_id: str,
        *,
        progress: Callable[[str, Mapping[str, object]], None] | None = None,
        progress_interval_seconds: int = 30,
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
        if cached is not None and cached[0] == self._terminal_evidence_fingerprint(
            run_id
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
        if str(run_status) != "running":
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
        compact = (
            self._storage_schema_version(run_id) == POLYMARKET_STORAGE_SCHEMA_VERSION
        )
        if compact:
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
        if not compact:
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
            actual_stream_counts = {
                str(stream): int(count)
                for stream, count in connection.execute(
                    """
                    SELECT stream, count(*) FROM polymarket_raw_message
                    WHERE run_id = ? GROUP BY stream ORDER BY stream
                    """,
                    [run_id],
                ).fetchall()
            }
            actual_assets = sorted({str(row[4]) for row in snapshots})
            actual_conditions = sorted({str(row[6]) for row in snapshots})
            expected_values: tuple[tuple[str, object, object], ...] = (
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
            )
            for field, actual, reported in expected_values:
                if actual != reported:
                    errors.append(f"recorder_report_evidence_mismatch:{run_id}:{field}")
            if run_ended is None:
                errors.append(f"recorder_run_missing_end:{run_id}")
            else:
                out_of_window_messages = int(
                    connection.execute(
                        """
                        SELECT count(*) FROM polymarket_raw_message
                        WHERE run_id = ?
                          AND (received_wall_ms < ? OR received_wall_ms > ?)
                        """,
                        [run_id, int(run_started), int(run_ended)],
                    ).fetchone()[0]
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
        evidence_result = tuple(errors)
        if str(run_status) != "running":
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
                    try:
                        while not stop.is_set():
                            receive = asyncio.create_task(websocket.recv())
                            changed = asyncio.create_task(self.registry.changed.wait())
                            stopping = asyncio.create_task(stop.wait())
                            transient = (receive, changed, stopping)
                            try:
                                done, _ = await asyncio.wait(
                                    {*transient, heartbeat_task},
                                    timeout=_STREAM_INACTIVITY_SECONDS,
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                            finally:
                                for task in transient:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.gather(
                                    *transient,
                                    return_exceptions=True,
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
                    finally:
                        heartbeat_task.cancel()
                        await asyncio.gather(heartbeat_task, return_exceptions=True)
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
                    try:
                        while not stop.is_set():
                            receive = asyncio.create_task(websocket.recv())
                            stopping = asyncio.create_task(stop.wait())
                            transient = (receive, stopping)
                            watched: set[asyncio.Task[object]] = set(transient)
                            if heartbeat_task is not None:
                                watched.add(heartbeat_task)
                            try:
                                done, _ = await asyncio.wait(
                                    watched,
                                    timeout=_STREAM_INACTIVITY_SECONDS,
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                            finally:
                                for task in transient:
                                    if not task.done():
                                        task.cancel()
                                await asyncio.gather(
                                    *transient,
                                    return_exceptions=True,
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
                    finally:
                        if heartbeat_task is not None:
                            heartbeat_task.cancel()
                            await asyncio.gather(heartbeat_task, return_exceptions=True)
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
    "POLYMARKET_EVIDENCE_SCHEMA_VERSION",
    "POLYMARKET_RECORDER_SCHEMA_VERSION",
    "POLYMARKET_STORAGE_SCHEMA_VERSION",
    "POLYMARKET_RTDS_WEBSOCKET",
    "DecodedPublicEvent",
    "MarketEvidence",
    "PolymarketEvidenceStore",
    "PolymarketPublicRecorder",
    "RawStreamMessage",
    "RecorderReport",
    "StreamGap",
]
