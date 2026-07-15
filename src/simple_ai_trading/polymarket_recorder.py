"""Prospective public-data recorder for Polymarket BTC/ETH/SOL five-minute paper trading."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import hmac
import json
from pathlib import Path
import re
import time
from typing import Mapping, Protocol, Sequence
import uuid

import duckdb
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
_DUCKDB_MEMORY_LIMIT = re.compile(r"[1-9][0-9]*(?:KB|MB|GB|TB)", re.IGNORECASE)
_WRITER_BATCH_SIZE = 256
_WRITER_MIN_DRAIN_SECONDS = 60.0
_WRITER_MAX_DRAIN_SECONDS = 600.0
_WRITER_STALL_SECONDS = 30.0
_WRITER_ASSUMED_MINIMUM_MESSAGES_PER_SECOND = 100.0


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


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
    ) -> None:
        self.path = Path(path)
        self.memory_limit = str(memory_limit).upper()
        if not _DUCKDB_MEMORY_LIMIT.fullmatch(self.memory_limit):
            raise ValueError("memory_limit must be a positive KB, MB, GB, or TB value")
        self.threads = int(threads)
        if self.threads < 1 or self.threads > 8:
            raise ValueError("threads must lie in [1, 8]")
        self.connection: duckdb.DuckDBPyConnection | None = None
        self.paper_journal: PaperOrderJournal | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self.connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(str(self.path))
            self.connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self.connection.execute(f"SET threads={self.threads}")
            self.connection.execute("SET TimeZone='UTC'")
            self.connection.execute("SET preserve_insertion_order=false")
            self._init_schema()
            self.paper_journal = PaperOrderJournal(self.connection)
        return self.connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.paper_journal = None

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
                message_id VARCHAR PRIMARY KEY,
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
                UNIQUE(run_id, stream, connection_id, sequence_number)
            );

            CREATE TABLE IF NOT EXISTS polymarket_public_event (
                event_id VARCHAR PRIMARY KEY,
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
            """
        )

    def start_run(self, run_id: str, started_at_ms: int) -> None:
        self.connect().execute(
            """
            INSERT INTO polymarket_recorder_run (
                run_id, schema_version, status, started_at_ms, ended_at_ms,
                report_json, report_sha256, error
            ) VALUES (?, ?, 'running', ?, NULL, '', '', '')
            """,
            [run_id, POLYMARKET_RECORDER_SCHEMA_VERSION, int(started_at_ms)],
        )

    def record_market_evidence(self, run_id: str, evidence: MarketEvidence) -> str:
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
        for start in range(0, len(messages), _WRITER_BATCH_SIZE):
            self._append_message_batch(
                run_id,
                messages[start : start + _WRITER_BATCH_SIZE],
            )

    def _append_message_batch(
        self,
        run_id: str,
        messages: Sequence[RawStreamMessage],
    ) -> None:
        raw_rows: list[tuple[object, ...]] = []
        event_rows: list[tuple[object, ...]] = []
        for message in messages:
            raw_row, normalized_rows = self._message_rows(
                run_id,
                message.validated(),
            )
            raw_rows.append(raw_row)
            event_rows.extend(normalized_rows)
        connection = self.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.executemany(
                """
                INSERT INTO polymarket_raw_message VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                raw_rows,
            )
            if event_rows:
                connection.executemany(
                    """
                    INSERT INTO polymarket_public_event VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    event_rows,
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _message_rows(
        run_id: str,
        message: RawStreamMessage,
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
            message.raw_text,
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
                    event_json,
                    event_sha,
                )
            )
        return raw_row, tuple(event_rows)

    def record_gap(self, run_id: str, gap: StreamGap) -> str:
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
        integrity = self.integrity_errors(run_id)
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

    def integrity_errors(self, run_id: str) -> tuple[str, ...]:
        connection = self.connect()
        errors: list[str] = []
        run_row = connection.execute(
            """
            SELECT schema_version, status, started_at_ms, ended_at_ms,
                   report_json, report_sha256
            FROM polymarket_recorder_run WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if run_row is None:
            return (f"missing_recorder_run:{run_id}",)
        run_schema, run_status, run_started, run_ended, report_json, report_sha = run_row
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
        messages = connection.execute(
            """
            SELECT message_id, run_id, stream, connection_id, sequence_number,
                   raw_payload_sha256, raw_text
            FROM polymarket_raw_message WHERE run_id = ? ORDER BY message_id
            """,
            [run_id],
        ).fetchall()
        for (
            message_id,
            message_run,
            stream,
            connection_id,
            sequence,
            claimed,
            raw_text,
        ) in messages:
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
        events = connection.execute(
            """
            SELECT event_id, message_id, sub_index, event_sha256, event_json
            FROM polymarket_public_event WHERE run_id = ? ORDER BY event_id
            """,
            [run_id],
        ).fetchall()
        message_ids = {str(row[0]) for row in messages}
        for event_id, message_id, sub_index, claimed, event_json in events:
            if str(message_id) not in message_ids:
                errors.append(f"event_without_message:{event_id}")
            actual = hashlib.sha256(str(event_json).encode("ascii")).hexdigest()
            if actual != str(claimed):
                errors.append(f"event_hash_mismatch:{event_id}")
            expected_id = _canonical_sha256(
                {
                    "message_id": str(message_id),
                    "sub_index": int(sub_index),
                    "event_sha256": str(claimed),
                }
            )
            if not hmac.compare_digest(str(event_id), expected_id):
                errors.append(f"event_id_mismatch:{event_id}")
            try:
                canonical = _canonical_json(_strict_json_loads(str(event_json)))
            except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
                errors.append(f"event_json_invalid:{event_id}:{exc}")
            else:
                if canonical != str(event_json):
                    errors.append(f"event_json_not_canonical:{event_id}")
        invalid_messages = connection.execute(
            """
            SELECT message_id FROM polymarket_raw_message
            WHERE run_id = ? AND parse_status = 'invalid' ORDER BY message_id
            """,
            [run_id],
        ).fetchall()
        errors.extend(f"invalid_stream_message:{row[0]}" for row in invalid_messages)
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
                ("market_snapshot_count", len(snapshots), parsed_report.get("market_snapshot_count")),
                ("raw_message_count", len(messages), parsed_report.get("raw_message_count")),
                ("normalized_event_count", len(events), parsed_report.get("normalized_event_count")),
                ("stream_gap_count", len(gaps), parsed_report.get("stream_gap_count")),
                ("stream_counts", actual_stream_counts, parsed_report.get("stream_counts")),
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
        if self.paper_journal is not None:
            errors.extend(self.paper_journal.integrity_errors())
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


def _event_index(stream: str, event: Mapping[str, object]) -> dict[str, object]:
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
        event_type = f"{topic}:{message_type}".strip(":") or "unknown"
        publisher_time_ms = _safe_int(event.get("timestamp"))
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            raw_symbol = str(payload.get("symbol") or "").lower()
            symbol = (
                raw_symbol.split("/")[0].upper()
                if "/" in raw_symbol
                else raw_symbol.removesuffix("usdt").upper()
            )
            source_time_ms = _safe_int(payload.get("timestamp"))
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
        queue_capacity: int = 20_000,
        discovery_interval_seconds: int = 60,
        memory_limit: str = "1GB",
        database_threads: int = 2,
    ) -> None:
        self.database = Path(database)
        self.client = client or PolymarketPublicClient()
        self.queue_capacity = int(queue_capacity)
        if self.queue_capacity < 1_000 or self.queue_capacity > 200_000:
            raise ValueError("queue_capacity must lie in [1000, 200000]")
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

    async def run(self, *, duration_seconds: int) -> RecorderReport:
        duration = int(duration_seconds)
        if duration < 5 or duration > 86_400:
            raise ValueError("duration_seconds must lie in [5, 86400]")
        self.registry = _MarketRegistry()
        self.errors = []
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
            writer = asyncio.create_task(self._writer(run_id, store, output))
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
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
                if not writer.done():
                    try:
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
            ended = _wall_ms()
            return store.finish_run(
                run_id,
                started_at_ms=started,
                ended_at_ms=ended,
                database=str(self.database.resolve()),
                errors=self.errors,
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
            await output.put(evidence)
            for sequence, raw in enumerate(books, start=1):
                await output.put(
                    RawStreamMessage(
                        stream="clob_rest_book",
                        connection_id=f"rest-{run_id}-{market.condition_id}",
                        sequence_number=sequence,
                        received_wall_ms=raw[0],
                        received_monotonic_ns=raw[1],
                        raw_text=raw[2],
                    )
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
        try:
            writer_store = await loop.run_in_executor(executor, open_writer_store)
            while True:
                item = await output.get()
                if item is None:
                    if pending_messages:
                        await invoke(
                            writer_store.append_messages,
                            run_id,
                            tuple(pending_messages),
                        )
                    return
                if isinstance(item, RawStreamMessage):
                    pending_messages.append(item)
                    if (
                        len(pending_messages) < _WRITER_BATCH_SIZE
                        and not output.empty()
                    ):
                        continue
                if pending_messages:
                    await invoke(
                        writer_store.append_messages,
                        run_id,
                        tuple(pending_messages),
                    )
                    pending_messages = []
                if isinstance(item, StreamGap):
                    await invoke(writer_store.record_gap, run_id, item)
                elif isinstance(item, MarketEvidence):
                    await invoke(writer_store.record_market_evidence, run_id, item)
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
            connection_id = uuid.uuid4().hex
            sequence = 0
            try:
                tokens = self.registry.token_ids()
                if not tokens:
                    raise RuntimeError("no validated Polymarket token subscriptions")
                async with connect(
                    CLOB_MARKET_WEBSOCKET,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=None,
                    max_size=_MAX_RAW_MESSAGE_BYTES,
                    max_queue=1024,
                ) as websocket:
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
                    backoff = 1.0
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
                            if heartbeat_task in done:
                                heartbeat_task.result()
                                if stop.is_set():
                                    return
                                raise RuntimeError("CLOB heartbeat stopped early")
                            if stopping in done and stopping.result():
                                return
                            if receive in done:
                                raw = receive.result()
                                sequence += 1
                                await output.put(
                                    RawStreamMessage(
                                        "clob_market",
                                        connection_id,
                                        sequence,
                                        _wall_ms(),
                                        _monotonic_ns(),
                                        _text_frame(raw),
                                    )
                                )
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
                await output.put(
                    StreamGap(
                        "clob_market",
                        connection_id,
                        _wall_ms(),
                        f"{exc.__class__.__name__}:{exc}",
                        sequence,
                    )
                )
                await _bounded_backoff(stop, backoff)
                backoff = min(30.0, backoff * 2.0)

    async def _rtds_stream(
        self,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
    ) -> None:
        subscriptions = (
            _canonical_json(
                {
                    "action": "subscribe",
                    "subscriptions": [
                        {
                            "topic": "crypto_prices",
                            "type": "update",
                            "filters": "btcusdt,ethusdt,solusdt",
                        }
                    ],
                }
            ),
            *(
                _canonical_json(
                    {
                        "action": "subscribe",
                        "subscriptions": [
                            {
                                "topic": "crypto_prices_chainlink",
                                "type": "*",
                                "filters": _canonical_json(
                                    {"symbol": f"{asset}/usd"}
                                ),
                            }
                        ],
                    }
                )
                for asset in ("btc", "eth", "sol")
            ),
        )
        # Isolate filtered Chainlink symbols so server-side replacement of one
        # topic subscription cannot silently remove another asset's feed.
        async with asyncio.TaskGroup() as task_group:
            for subscription in subscriptions:
                task_group.create_task(
                    self._simple_stream(
                        stream="polymarket_rtds",
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
            url=BINANCE_SPOT_WEBSOCKET,
            subscription=None,
            heartbeat=None,
            heartbeat_seconds=20.0,
            output=output,
            stop=stop,
            protocol_ping_interval=20.0,
        )

    async def _simple_stream(
        self,
        *,
        stream: str,
        url: str,
        subscription: str | None,
        heartbeat: str | None,
        heartbeat_seconds: float,
        output: asyncio.Queue[RawStreamMessage | StreamGap | MarketEvidence | None],
        stop: asyncio.Event,
        protocol_ping_interval: float | None = None,
    ) -> None:
        backoff = 1.0
        while not stop.is_set():
            connection_id = uuid.uuid4().hex
            sequence = 0
            try:
                async with connect(
                    url,
                    open_timeout=10,
                    close_timeout=3,
                    ping_interval=protocol_ping_interval,
                    ping_timeout=20,
                    max_size=_MAX_RAW_MESSAGE_BYTES,
                    max_queue=1024,
                ) as websocket:
                    if subscription is not None:
                        await websocket.send(subscription)
                    backoff = 1.0
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
                            if heartbeat_task is not None and heartbeat_task in done:
                                heartbeat_task.result()
                                if stop.is_set():
                                    return
                                raise RuntimeError(f"{stream} heartbeat stopped early")
                            if stopping in done and stopping.result():
                                return
                            raw = receive.result()
                            sequence += 1
                            text = _text_frame(raw)
                            await output.put(
                                RawStreamMessage(
                                    stream,
                                    connection_id,
                                    sequence,
                                    _wall_ms(),
                                    _monotonic_ns(),
                                    text,
                                )
                            )
                            if text == "PING":
                                await websocket.send("PONG")
                    finally:
                        if heartbeat_task is not None:
                            heartbeat_task.cancel()
                            await asyncio.gather(
                                heartbeat_task, return_exceptions=True
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await output.put(
                    StreamGap(
                        stream,
                        connection_id,
                        _wall_ms(),
                        f"{exc.__class__.__name__}:{exc}",
                        sequence,
                    )
                )
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
    "POLYMARKET_RTDS_WEBSOCKET",
    "MarketEvidence",
    "PolymarketEvidenceStore",
    "PolymarketPublicRecorder",
    "RawStreamMessage",
    "RecorderReport",
    "StreamGap",
]
