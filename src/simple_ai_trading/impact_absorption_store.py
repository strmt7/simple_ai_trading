"""Transactional Round 73 prospective evidence storage and audit."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence, TypeAlias

import duckdb
import zstandard

from .assets import normalize_symbol
from .duckdb_batch import insert_rows_columnar
from .impact_absorption import (
    AggregateTradeEvent,
    BookTickerEvent,
    DepthUpdate,
    L2BookState,
    LiquidationSnapshotEvent,
    MarkPriceEvent,
    ROUND73_DESIGN_SHA256,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
)
from .impact_capture_frame import (
    IMPACT_CAPTURE_FRAME_FORMAT,
    ImpactCaptureFrameRecord,
    decode_impact_capture_frame,
    encode_impact_capture_frame,
)


IMPACT_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v1"
IMPACT_CAPTURE_CONTRACT_SHA256 = (
    "f379b53b86d20f16b686132ef8fe4dc5eb47b6a0910e6ba85c38ddf0caa01c7b"
)
IMPACT_CAPTURE_COMPRESSION_LEVEL = 3
IMPACT_CAPTURE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES = 2_147_483_648

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEMORY_LIMIT = re.compile(r"[1-9][0-9]*(?:KB|MB|GB|TB)", re.IGNORECASE)
_SECRET_KEYS = (
    "api_key",
    "apikey",
    "password",
    "private_key",
    "secret",
    "signature",
    "token",
)
_REST_ENDPOINTS = {
    "serverTime": ("/fapi/v1/time", frozenset()),
    "exchangeInfo": ("/fapi/v1/exchangeInfo", frozenset()),
    "depthSnapshot": ("/fapi/v1/depth", frozenset({"limit", "symbol"})),
    "openInterest": ("/fapi/v1/openInterest", frozenset({"symbol"})),
}


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


def _strict_json_object(raw_text: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("exact-wire payload is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("exact-wire payload must be a JSON object")
    return parsed


def _validate_run_id(value: object, label: str = "run ID") -> str:
    candidate = str(value).strip().lower()
    if not _RUN_ID.fullmatch(candidate):
        raise ValueError(f"{label} must be 32 lowercase hexadecimal characters")
    return candidate


def _validate_sha256(value: object, label: str) -> str:
    candidate = str(value).strip().lower()
    if not _SHA256.fullmatch(candidate):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return candidate


def _positive_integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if parsed < 0 or (parsed == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return parsed


def _signed_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return parsed


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return parsed


def _reject_secret_fields(value: object, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if any(marker in normalized for marker in _SECRET_KEYS):
                raise ValueError(
                    f"secret-bearing field is forbidden in evidence: {path}.{key}"
                )
            _reject_secret_fields(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_secret_fields(nested, f"{path}[{index}]")


def _reject_nonfinite(value: object, path: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value is forbidden in evidence: {path}")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _reject_nonfinite(getattr(value, field.name), f"{path}.{field.name}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, f"{path}[{index}]")


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    return f"INSERT INTO {table} ({', '.join(columns)}) SELECT " + ", ".join(
        "unnest(?)" for _column in columns
    )


@dataclass(frozen=True)
class ImpactRestEvent:
    """Typed metadata for one credential-free public REST response."""

    event_type: Literal["serverTime", "exchangeInfo", "depthSnapshot", "openInterest"]
    request_path: str
    request_parameters: Mapping[str, object]
    response_status: int
    request_started_wall_ns: int
    request_started_monotonic_ns: int
    symbol: str = ""
    exchange_time_ms: int | None = None
    update_id: int | None = None
    open_interest: float | None = None
    used_weight_1m: int | None = None


ImpactParsedEvent: TypeAlias = (
    DepthUpdate
    | BookTickerEvent
    | AggregateTradeEvent
    | MarkPriceEvent
    | LiquidationSnapshotEvent
    | ImpactRestEvent
)


@dataclass(frozen=True)
class ImpactCaptureMessage:
    """Exact frame record plus its typed, raw-linked observation."""

    record: ImpactCaptureFrameRecord
    event: ImpactParsedEvent
    segment_id: str = ""
    l2_state: L2BookState | None = None


@dataclass(frozen=True)
class ImpactFrameWriteResult:
    run_id: str
    frame_index: int
    frame_sha256: str
    message_count: int
    uncompressed_bytes: int
    compressed_bytes: int
    compressed_payload_total_bytes: int
    payload_cap_reached: bool


@dataclass(frozen=True)
class ImpactCaptureAudit:
    run_id: str
    passed: bool
    errors: tuple[str, ...]
    frame_count: int
    message_count: int
    compressed_payload_bytes: int
    last_frame_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "round-073-capture-audit-v1",
            "run_id": self.run_id,
            "passed": self.passed,
            "errors": list(self.errors),
            "frame_count": self.frame_count,
            "message_count": self.message_count,
            "compressed_payload_bytes": self.compressed_payload_bytes,
            "last_frame_sha256": self.last_frame_sha256,
            "capture_contract_sha256": IMPACT_CAPTURE_CONTRACT_SHA256,
        }


_EVENT_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "message_id",
    "segment_id",
    "stream",
    "connection_id",
    "sequence_number",
    "received_wall_ns",
    "received_monotonic_ns",
    "raw_payload_sha256",
    "event_type",
    "symbol",
    "event_time_ms",
    "transaction_time_ms",
    "update_id",
    "typed_event_sha256",
)
_DEPTH_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    "first_update_id",
    "final_update_id",
    "previous_update_id",
    "stale",
    "best_bid",
    "best_ask",
    "bid_added_qty",
    "bid_removed_qty",
    "ask_added_qty",
    "ask_removed_qty",
    "bid_added_quote",
    "bid_removed_quote",
    "ask_added_quote",
    "ask_removed_quote",
)
_L2_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    "update_id",
    "best_bid",
    "best_ask",
    "spread_bps",
    "mid",
    "bid_prices",
    "bid_quantities",
    "ask_prices",
    "ask_quantities",
    "bid_depth_quote_5",
    "ask_depth_quote_5",
    "bid_depth_quote_10",
    "ask_depth_quote_10",
    "bid_depth_quote_20",
    "ask_depth_quote_20",
    "imbalance_5",
    "imbalance_10",
    "imbalance_20",
)
_BOOK_TICKER_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    "event_time_ms",
    "transaction_time_ms",
    "update_id",
    "bid",
    "bid_qty",
    "ask",
    "ask_qty",
)
_TRADE_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    "event_time_ms",
    "transaction_time_ms",
    "aggregate_trade_id",
    "first_trade_id",
    "last_trade_id",
    "price",
    "qty",
    "normalized_qty",
    "buyer_is_maker",
)
_MARK_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    "event_time_ms",
    "mark_price",
    "index_price",
    "estimated_settlement_price",
    "funding_rate",
    "next_funding_time_ms",
)
_LIQUIDATION_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    "event_time_ms",
    "order_time_ms",
    "side",
    "order_type",
    "time_in_force",
    "original_qty",
    "price",
    "average_price",
    "order_status",
    "last_filled_qty",
    "accumulated_filled_qty",
)
_REST_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "event_type",
    "symbol",
    "request_path",
    "request_parameters_json",
    "response_status",
    "request_started_wall_ns",
    "request_started_monotonic_ns",
    "used_weight_1m",
    "exchange_time_ms",
    "update_id",
    "open_interest",
)


class ImpactAbsorptionStore:
    """One-writer DuckDB store for exact and typed Round 73 observations."""

    def __init__(
        self,
        path: str | Path = "data/microstructure.duckdb",
        *,
        memory_limit: str = "2GB",
        threads: int = 2,
        read_only: bool = False,
    ) -> None:
        if not _MEMORY_LIMIT.fullmatch(str(memory_limit).strip()):
            raise ValueError(
                "memory_limit must be a positive integer followed by a byte unit"
            )
        self.path = str(path)
        self.memory_limit = str(memory_limit).strip().upper()
        self.threads = max(1, min(8, int(threads)))
        self.read_only = bool(read_only)
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            if self.path != ":memory:":
                target = Path(self.path)
                if self.read_only and not target.is_file():
                    raise FileNotFoundError(
                        f"read-only impact store does not exist: {target}"
                    )
                if not self.read_only:
                    target.parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(self.path, read_only=self.read_only)
            self._connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self._connection.execute(f"SET threads={self.threads}")
            self._connection.execute("SET TimeZone='UTC'")
            self._connection.execute("SET preserve_insertion_order=false")
            if not self.read_only:
                self._init_schema()
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "ImpactAbsorptionStore":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.connect().execute(
            """
            CREATE TABLE IF NOT EXISTS impact_capture_run (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                design_sha256 VARCHAR NOT NULL,
                capture_contract_sha256 VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                started_wall_ns UBIGINT NOT NULL,
                started_monotonic_ns UBIGINT NOT NULL,
                ended_wall_ns UBIGINT,
                symbols_json VARCHAR NOT NULL,
                config_json VARCHAR NOT NULL,
                config_sha256 VARCHAR NOT NULL,
                compressed_payload_cap_bytes UBIGINT NOT NULL,
                compressed_payload_bytes UBIGINT NOT NULL,
                payload_cap_reached BOOLEAN NOT NULL,
                frame_count UINTEGER NOT NULL,
                message_count UBIGINT NOT NULL,
                segment_count UINTEGER NOT NULL,
                last_frame_sha256 VARCHAR NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_capture_segment (
                run_id VARCHAR NOT NULL,
                segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                started_wall_ns UBIGINT NOT NULL,
                started_monotonic_ns UBIGINT NOT NULL,
                ended_wall_ns UBIGINT,
                snapshot_update_id UBIGINT NOT NULL,
                tick_size DOUBLE NOT NULL,
                clock_offset_ns BIGINT NOT NULL,
                clock_rtt_ns UBIGINT NOT NULL,
                cooldown_until_wall_ns UBIGINT NOT NULL,
                first_final_update_id UBIGINT,
                last_final_update_id UBIGINT,
                depth_message_count UBIGINT NOT NULL,
                invalid_event_count UBIGINT NOT NULL,
                sequence_gap_count UBIGINT NOT NULL,
                crossed_book_count UBIGINT NOT NULL,
                reason VARCHAR NOT NULL,
                PRIMARY KEY (run_id, segment_id)
            );

            CREATE TABLE IF NOT EXISTS impact_capture_frame (
                run_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                schema_version VARCHAR NOT NULL,
                frame_format VARCHAR NOT NULL,
                previous_frame_sha256 VARCHAR NOT NULL,
                message_count UINTEGER NOT NULL,
                first_message_id VARCHAR NOT NULL,
                last_message_id VARCHAR NOT NULL,
                message_manifest_sha256 VARCHAR NOT NULL,
                first_received_wall_ns UBIGINT NOT NULL,
                last_received_wall_ns UBIGINT NOT NULL,
                first_received_monotonic_ns UBIGINT NOT NULL,
                last_received_monotonic_ns UBIGINT NOT NULL,
                uncompressed_bytes UINTEGER NOT NULL,
                uncompressed_sha256 VARCHAR NOT NULL,
                compressed_bytes UINTEGER NOT NULL,
                compressed_sha256 VARCHAR NOT NULL,
                stream_counts_json VARCHAR NOT NULL,
                compressed_payload BLOB NOT NULL,
                frame_sha256 VARCHAR NOT NULL,
                PRIMARY KEY (run_id, frame_index)
            );

            CREATE TABLE IF NOT EXISTS impact_capture_lane_state (
                run_id VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                last_sequence_number UBIGINT NOT NULL,
                last_received_monotonic_ns UBIGINT NOT NULL,
                PRIMARY KEY (run_id, stream, connection_id)
            );

            CREATE TABLE IF NOT EXISTS impact_event_index (
                run_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL,
                message_id VARCHAR NOT NULL,
                segment_id VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                sequence_number UBIGINT NOT NULL,
                received_wall_ns UBIGINT NOT NULL,
                received_monotonic_ns UBIGINT NOT NULL,
                raw_payload_sha256 VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                event_time_ms BIGINT,
                transaction_time_ms BIGINT,
                update_id UBIGINT,
                typed_event_sha256 VARCHAR NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_depth_update (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, first_update_id UBIGINT NOT NULL,
                final_update_id UBIGINT NOT NULL, previous_update_id UBIGINT NOT NULL,
                stale BOOLEAN NOT NULL, best_bid DOUBLE NOT NULL, best_ask DOUBLE NOT NULL,
                bid_added_qty DOUBLE NOT NULL, bid_removed_qty DOUBLE NOT NULL,
                ask_added_qty DOUBLE NOT NULL, ask_removed_qty DOUBLE NOT NULL,
                bid_added_quote DOUBLE NOT NULL, bid_removed_quote DOUBLE NOT NULL,
                ask_added_quote DOUBLE NOT NULL, ask_removed_quote DOUBLE NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_l2_state (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, update_id UBIGINT NOT NULL,
                best_bid DOUBLE NOT NULL, best_ask DOUBLE NOT NULL,
                spread_bps DOUBLE NOT NULL, mid DOUBLE NOT NULL,
                bid_prices DOUBLE[] NOT NULL, bid_quantities DOUBLE[] NOT NULL,
                ask_prices DOUBLE[] NOT NULL, ask_quantities DOUBLE[] NOT NULL,
                bid_depth_quote_5 DOUBLE NOT NULL, ask_depth_quote_5 DOUBLE NOT NULL,
                bid_depth_quote_10 DOUBLE NOT NULL, ask_depth_quote_10 DOUBLE NOT NULL,
                bid_depth_quote_20 DOUBLE NOT NULL, ask_depth_quote_20 DOUBLE NOT NULL,
                imbalance_5 DOUBLE NOT NULL, imbalance_10 DOUBLE NOT NULL,
                imbalance_20 DOUBLE NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_book_ticker (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                transaction_time_ms BIGINT NOT NULL, update_id UBIGINT NOT NULL,
                bid DOUBLE NOT NULL, bid_qty DOUBLE NOT NULL,
                ask DOUBLE NOT NULL, ask_qty DOUBLE NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_aggregate_trade (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                transaction_time_ms BIGINT NOT NULL, aggregate_trade_id UBIGINT NOT NULL,
                first_trade_id UBIGINT NOT NULL, last_trade_id UBIGINT NOT NULL,
                price DOUBLE NOT NULL, qty DOUBLE NOT NULL, normalized_qty DOUBLE NOT NULL,
                buyer_is_maker BOOLEAN NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_mark_price (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                mark_price DOUBLE NOT NULL, index_price DOUBLE NOT NULL,
                estimated_settlement_price DOUBLE, funding_rate DOUBLE NOT NULL,
                next_funding_time_ms BIGINT NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_liquidation_snapshot (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                order_time_ms BIGINT NOT NULL, side VARCHAR NOT NULL,
                order_type VARCHAR NOT NULL, time_in_force VARCHAR NOT NULL,
                original_qty DOUBLE NOT NULL, price DOUBLE NOT NULL,
                average_price DOUBLE NOT NULL, order_status VARCHAR NOT NULL,
                last_filled_qty DOUBLE NOT NULL, accumulated_filled_qty DOUBLE NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_rest_event (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL, symbol VARCHAR NOT NULL,
                request_path VARCHAR NOT NULL, request_parameters_json VARCHAR NOT NULL,
                response_status USMALLINT NOT NULL,
                request_started_wall_ns UBIGINT NOT NULL,
                request_started_monotonic_ns UBIGINT NOT NULL,
                used_weight_1m UINTEGER,
                exchange_time_ms BIGINT,
                update_id UBIGINT, open_interest DOUBLE,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_capture_report (
                run_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                capture_contract_sha256 VARCHAR NOT NULL,
                report_json VARCHAR NOT NULL,
                report_sha256 VARCHAR NOT NULL,
                recorded_at_wall_ns UBIGINT NOT NULL
            );
            """
        )

    def start_run(
        self,
        *,
        run_id: str,
        started_wall_ns: int,
        started_monotonic_ns: int,
        config: Mapping[str, object],
        symbols: Sequence[str] = IMPACT_CAPTURE_SYMBOLS,
        compressed_payload_cap_bytes: int = IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES,
    ) -> None:
        selected = _validate_run_id(run_id)
        normalized = tuple(normalize_symbol(value, default="") for value in symbols)
        if normalized != IMPACT_CAPTURE_SYMBOLS:
            raise ValueError("Round 73 capture requires BTCUSDT, ETHUSDT, and SOLUSDT")
        _reject_secret_fields(config)
        config_json = _canonical_json(config)
        cap = _positive_integer(compressed_payload_cap_bytes, "payload cap")
        self.connect().execute(
            """
            INSERT INTO impact_capture_run VALUES (
                ?, ?, ?, ?, 'running', ?, ?, NULL, ?, ?, ?, ?, 0, false,
                0, 0, 0, '', ''
            )
            """,
            [
                selected,
                IMPACT_CAPTURE_SCHEMA_VERSION,
                ROUND73_DESIGN_SHA256,
                IMPACT_CAPTURE_CONTRACT_SHA256,
                _positive_integer(started_wall_ns, "run wall clock"),
                _positive_integer(started_monotonic_ns, "run monotonic clock"),
                _canonical_json(list(normalized)),
                config_json,
                hashlib.sha256(config_json.encode("ascii")).hexdigest(),
                cap,
            ],
        )

    def start_segment(
        self,
        *,
        run_id: str,
        segment_id: str,
        symbol: str,
        started_wall_ns: int,
        started_monotonic_ns: int,
        snapshot_update_id: int,
        tick_size: float,
        clock_offset_ns: int,
        clock_rtt_ns: int,
        cooldown_until_wall_ns: int,
    ) -> None:
        selected = _validate_run_id(run_id)
        segment = _validate_run_id(segment_id, "segment ID")
        normalized = normalize_symbol(symbol, default="")
        if normalized not in IMPACT_CAPTURE_SYMBOLS:
            raise ValueError("segment symbol is outside the Round 73 universe")
        connection = self.connect()
        row = connection.execute(
            "SELECT status FROM impact_capture_run WHERE run_id = ?", [selected]
        ).fetchone()
        if row is None or str(row[0]) != "running":
            raise ValueError("capture run is missing or not running")
        active = connection.execute(
            """
            SELECT count(*) FROM impact_capture_segment
            WHERE run_id = ? AND symbol = ? AND status = 'active'
            """,
            [selected, normalized],
        ).fetchone()[0]
        if int(active) != 0:
            raise ValueError(f"an active segment already exists for {normalized}")
        offset = _signed_integer(clock_offset_ns, "clock offset")
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                INSERT INTO impact_capture_segment VALUES (
                    ?, ?, ?, 'active', ?, ?, NULL, ?, ?, ?, ?, ?,
                    NULL, NULL, 0, 0, 0, 0, ''
                )
                """,
                [
                    selected,
                    segment,
                    normalized,
                    _positive_integer(started_wall_ns, "segment wall clock"),
                    _positive_integer(started_monotonic_ns, "segment monotonic clock"),
                    _positive_integer(snapshot_update_id, "snapshot update ID"),
                    _finite_float(tick_size, "tick size", positive=True),
                    offset,
                    _positive_integer(clock_rtt_ns, "clock RTT"),
                    _positive_integer(
                        cooldown_until_wall_ns,
                        "cooldown wall clock",
                        allow_zero=True,
                    ),
                ],
            )
            connection.execute(
                """
                UPDATE impact_capture_run SET segment_count = segment_count + 1
                WHERE run_id = ?
                """,
                [selected],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def finish_segment(
        self,
        *,
        run_id: str,
        segment_id: str,
        status: Literal["valid", "invalid", "stopped"],
        ended_wall_ns: int,
        reason: str = "",
        invalid_event_count: int = 0,
        sequence_gap_count: int = 0,
        crossed_book_count: int = 0,
    ) -> None:
        selected = _validate_run_id(run_id)
        segment = _validate_run_id(segment_id, "segment ID")
        final_status = str(status)
        if final_status not in {"valid", "invalid", "stopped"}:
            raise ValueError("segment terminal status is invalid")
        if final_status == "invalid" and not str(reason).strip():
            raise ValueError("invalid segment requires a reason")
        connection = self.connect()
        updated = connection.execute(
            """
            UPDATE impact_capture_segment
            SET status = ?, ended_wall_ns = ?, reason = ?,
                invalid_event_count = invalid_event_count + ?,
                sequence_gap_count = sequence_gap_count + ?,
                crossed_book_count = crossed_book_count + ?
            WHERE run_id = ? AND segment_id = ? AND status = 'active'
            RETURNING segment_id
            """,
            [
                final_status,
                _positive_integer(ended_wall_ns, "segment end wall clock"),
                str(reason)[:2_000],
                _positive_integer(
                    invalid_event_count, "invalid event count", allow_zero=True
                ),
                _positive_integer(
                    sequence_gap_count, "sequence gap count", allow_zero=True
                ),
                _positive_integer(
                    crossed_book_count, "crossed book count", allow_zero=True
                ),
                selected,
                segment,
            ],
        ).fetchone()
        if updated is None:
            raise ValueError("active capture segment was not found")

    def finish_run(
        self,
        *,
        run_id: str,
        status: Literal["completed", "failed", "stopped"],
        ended_wall_ns: int,
        error: str = "",
    ) -> None:
        selected = _validate_run_id(run_id)
        final_status = str(status)
        if final_status not in {"completed", "failed", "stopped"}:
            raise ValueError("run terminal status is invalid")
        if final_status == "failed" and not str(error).strip():
            raise ValueError("failed run requires an error")
        active = int(
            self.connect()
            .execute(
                """
                SELECT count(*) FROM impact_capture_segment
                WHERE run_id = ? AND status = 'active'
                """,
                [selected],
            )
            .fetchone()[0]
        )
        if active:
            raise ValueError("capture run cannot finish with active segments")
        updated = (
            self.connect()
            .execute(
                """
            UPDATE impact_capture_run
            SET status = ?, ended_wall_ns = ?, error = ?
            WHERE run_id = ? AND status = 'running'
            RETURNING run_id
            """,
                [
                    final_status,
                    _positive_integer(ended_wall_ns, "run end wall clock"),
                    str(error)[:2_000],
                    selected,
                ],
            )
            .fetchone()
        )
        if updated is None:
            raise ValueError("running capture run was not found")

    def record_report(
        self,
        *,
        run_id: str,
        report: Mapping[str, object],
        recorded_at_wall_ns: int,
    ) -> str:
        selected = _validate_run_id(run_id)
        _reject_secret_fields(report, "capture report")
        if str(report.get("run_id", "")) != selected:
            raise ValueError("capture report run ID does not match its storage key")
        if str(report.get("schema_version", "")) != "round-073-capture-report-v2":
            raise ValueError("capture report schema version is invalid")
        run = (
            self.connect()
            .execute(
                "SELECT status FROM impact_capture_run WHERE run_id = ?", [selected]
            )
            .fetchone()
        )
        if run is None or str(run[0]) == "running":
            raise ValueError("capture report requires a terminal run")
        report_json = _canonical_json(report)
        report_sha256 = hashlib.sha256(report_json.encode("ascii")).hexdigest()
        self.connect().execute(
            """
            INSERT INTO impact_capture_report VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                selected,
                "round-073-capture-report-v2",
                IMPACT_CAPTURE_CONTRACT_SHA256,
                report_json,
                report_sha256,
                _positive_integer(recorded_at_wall_ns, "report wall clock"),
            ],
        )
        return report_sha256

    def _active_segments(self, run_id: str) -> dict[str, str]:
        rows = (
            self.connect()
            .execute(
                """
            SELECT segment_id, symbol FROM impact_capture_segment
            WHERE run_id = ? AND status = 'active'
            """,
                [run_id],
            )
            .fetchall()
        )
        return {str(segment_id): str(symbol) for segment_id, symbol in rows}

    @staticmethod
    def _rest_event(
        event: ImpactRestEvent,
    ) -> tuple[
        str,
        str,
        str,
        int,
        int,
        int,
        int | None,
        int | None,
        int | None,
        float | None,
    ]:
        try:
            expected_path, allowed_parameters = _REST_ENDPOINTS[event.event_type]
        except KeyError as exc:
            raise ValueError("unsupported Round 73 REST event") from exc
        if str(event.request_path) != expected_path:
            raise ValueError("REST event path does not match its type")
        parameters = {
            str(key): value for key, value in event.request_parameters.items()
        }
        if set(parameters) - allowed_parameters:
            raise ValueError(
                "REST event contains unsupported or secret-bearing parameters"
            )
        _reject_secret_fields(parameters, "REST parameters")
        symbol = normalize_symbol(event.symbol, default="") if event.symbol else ""
        if event.event_type in {"depthSnapshot", "openInterest"}:
            if symbol not in IMPACT_CAPTURE_SYMBOLS:
                raise ValueError("symbol REST event is outside the Round 73 universe")
            if parameters.get("symbol") != symbol:
                raise ValueError("REST request symbol does not match typed symbol")
        elif symbol:
            raise ValueError("global REST event cannot carry a symbol")
        status = _positive_integer(event.response_status, "REST status")
        if not 100 <= status <= 599:
            raise ValueError("REST status is outside the HTTP range")
        request_started_wall_ns = _positive_integer(
            event.request_started_wall_ns, "REST request wall clock"
        )
        request_started_monotonic_ns = _positive_integer(
            event.request_started_monotonic_ns, "REST request monotonic clock"
        )
        used_weight_1m = (
            None
            if event.used_weight_1m is None
            else _positive_integer(
                event.used_weight_1m,
                "REST used weight",
                allow_zero=True,
            )
        )
        exchange_time = (
            None
            if event.exchange_time_ms is None
            else _positive_integer(event.exchange_time_ms, "REST exchange time")
        )
        update_id = (
            None
            if event.update_id is None
            else _positive_integer(event.update_id, "REST update ID")
        )
        if event.event_type == "serverTime" and exchange_time is None:
            raise ValueError("server-time evidence requires the exchange time")
        if event.event_type == "depthSnapshot" and update_id is None:
            raise ValueError("depth snapshot evidence requires an update ID")
        open_interest = (
            None
            if event.open_interest is None
            else _finite_float(event.open_interest, "open interest")
        )
        if open_interest is not None and open_interest < 0.0:
            raise ValueError("open interest must be non-negative")
        if event.event_type == "openInterest":
            if exchange_time is None or open_interest is None:
                raise ValueError(
                    "open-interest evidence requires value and exchange time"
                )
        elif open_interest is not None:
            raise ValueError(
                "only open-interest evidence may carry an open-interest value"
            )
        return (
            symbol,
            expected_path,
            _canonical_json(parameters),
            status,
            request_started_wall_ns,
            request_started_monotonic_ns,
            used_weight_1m,
            exchange_time,
            update_id,
            open_interest,
        )

    @staticmethod
    def _event_identity(
        event: ImpactParsedEvent,
    ) -> tuple[str, str, int | None, int | None, int | None]:
        if isinstance(event, DepthUpdate):
            return (
                "depthUpdate",
                event.symbol,
                event.event_time_ms,
                event.transaction_time_ms,
                event.final_update_id,
            )
        if isinstance(event, BookTickerEvent):
            return (
                "bookTicker",
                event.symbol,
                event.event_time_ms,
                event.transaction_time_ms,
                event.update_id,
            )
        if isinstance(event, AggregateTradeEvent):
            return (
                "aggTrade",
                event.symbol,
                event.event_time_ms,
                event.transaction_time_ms,
                event.aggregate_trade_id,
            )
        if isinstance(event, MarkPriceEvent):
            return "markPriceUpdate", event.symbol, event.event_time_ms, None, None
        if isinstance(event, LiquidationSnapshotEvent):
            return (
                "forceOrder",
                event.symbol,
                event.event_time_ms,
                event.order_time_ms,
                None,
            )
        if isinstance(event, ImpactRestEvent):
            (
                symbol,
                _path,
                _parameters,
                _status,
                _request_wall,
                _request_monotonic,
                _used_weight,
                exchange_time,
                update_id,
                _open_interest,
            ) = ImpactAbsorptionStore._rest_event(event)
            return event.event_type, symbol, exchange_time, None, update_id
        raise TypeError(f"unsupported impact event type: {type(event).__name__}")

    @staticmethod
    def _expected_stream(event_type: str) -> str:
        if event_type in {"depthUpdate", "bookTicker"}:
            return "binance_futures_public"
        if event_type in {"aggTrade", "markPriceUpdate", "forceOrder"}:
            return "binance_futures_market"
        if event_type in _REST_ENDPOINTS:
            return "binance_futures_rest"
        raise ValueError(f"unsupported impact event type: {event_type}")

    @staticmethod
    def _validate_raw_semantics(message: ImpactCaptureMessage) -> None:
        root = _strict_json_object(message.record.raw_text)
        event = message.event
        if isinstance(event, ImpactRestEvent):
            if message.record.stream != "binance_futures_rest":
                raise ValueError(
                    "REST evidence is stored on the wrong exact-wire stream"
                )
            ImpactAbsorptionStore._rest_event(event)
            if event.request_started_wall_ns > message.record.received_wall_ns:
                raise ValueError("REST request wall clock follows its response receipt")
            if (
                event.request_started_monotonic_ns
                > message.record.received_monotonic_ns
            ):
                raise ValueError(
                    "REST request monotonic clock follows its response receipt"
                )
            if event.event_type == "serverTime":
                if _positive_integer(root.get("serverTime"), "raw server time") != int(
                    event.exchange_time_ms or 0
                ):
                    raise ValueError("raw server time differs from typed REST evidence")
            elif event.event_type == "depthSnapshot":
                if _positive_integer(
                    root.get("lastUpdateId"), "raw snapshot update ID"
                ) != int(event.update_id or 0):
                    raise ValueError(
                        "raw snapshot update ID differs from typed REST evidence"
                    )
                if not isinstance(root.get("bids"), Sequence) or not isinstance(
                    root.get("asks"), Sequence
                ):
                    raise ValueError("raw depth snapshot is missing its ladders")
            elif event.event_type == "openInterest":
                if normalize_symbol(root.get("symbol"), default="") != event.symbol:
                    raise ValueError(
                        "raw open-interest symbol differs from typed REST evidence"
                    )
                if _positive_integer(root.get("time"), "raw open-interest time") != int(
                    event.exchange_time_ms or 0
                ):
                    raise ValueError(
                        "raw open-interest time differs from typed REST evidence"
                    )
                observed = _finite_float(root.get("openInterest"), "raw open interest")
                if observed != event.open_interest:
                    raise ValueError(
                        "raw open interest differs from typed REST evidence"
                    )
            elif event.event_type == "exchangeInfo":
                if not isinstance(root.get("symbols"), Sequence):
                    raise ValueError("raw exchange information has no symbol metadata")
            return

        stream_name = root.get("stream")
        payload = root.get("data")
        if not isinstance(stream_name, str) or not stream_name.strip():
            raise ValueError(
                "exact WebSocket evidence is missing its combined-stream name"
            )
        if not isinstance(payload, Mapping):
            raise ValueError("exact WebSocket evidence is missing its data object")
        received = message.record.received_monotonic_ns
        if isinstance(event, DepthUpdate):
            expected = (
                str(payload.get("e", "")),
                normalize_symbol(payload.get("s"), default=""),
                _positive_integer(payload.get("E"), "raw depth event time"),
                _positive_integer(payload.get("T"), "raw depth transaction time"),
                _positive_integer(payload.get("U"), "raw depth first update ID"),
                _positive_integer(payload.get("u"), "raw depth final update ID"),
                _positive_integer(payload.get("pu"), "raw depth previous update ID"),
                _positive_integer(payload.get("st"), "raw depth stream type"),
                normalize_symbol(payload.get("ps"), default=""),
            )
            actual = (
                "depthUpdate",
                event.symbol,
                event.event_time_ms,
                event.transaction_time_ms,
                event.first_update_id,
                event.final_update_id,
                event.previous_update_id,
                1,
                event.symbol,
            )
            if expected != actual:
                raise ValueError("raw depth identity differs from typed evidence")
            if not event.stale:
                raw_changes: list[tuple[str, float, float]] = []
                for side, key in (("bid", "b"), ("ask", "a")):
                    levels = payload.get(key)
                    if not isinstance(levels, Sequence) or isinstance(
                        levels, (str, bytes, bytearray)
                    ):
                        raise ValueError("raw depth levels are malformed")
                    for level in levels:
                        if (
                            not isinstance(level, Sequence)
                            or isinstance(level, (str, bytes, bytearray))
                            or len(level) < 2
                        ):
                            raise ValueError("raw depth level is malformed")
                        raw_changes.append(
                            (
                                side,
                                _finite_float(
                                    level[0], "raw depth price", positive=True
                                ),
                                _finite_float(level[1], "raw depth quantity"),
                            )
                        )
                typed_changes = [
                    (change.side, change.price, change.new_qty)
                    for change in event.changes
                ]
                if raw_changes != typed_changes:
                    raise ValueError("raw depth changes differ from typed evidence")
        elif isinstance(event, BookTickerEvent):
            if (
                parse_book_ticker(
                    payload,
                    symbol=event.symbol,
                    receive_time_ns=received,
                )
                != event
            ):
                raise ValueError("raw book ticker differs from typed evidence")
        elif isinstance(event, AggregateTradeEvent):
            if (
                parse_aggregate_trade(
                    payload,
                    symbol=event.symbol,
                    receive_time_ns=received,
                )
                != event
            ):
                raise ValueError("raw aggregate trade differs from typed evidence")
        elif isinstance(event, MarkPriceEvent):
            if (
                parse_mark_price(
                    payload,
                    symbol=event.symbol,
                    receive_time_ns=received,
                )
                != event
            ):
                raise ValueError("raw mark price differs from typed evidence")
        elif isinstance(event, LiquidationSnapshotEvent):
            if (
                parse_liquidation_snapshot(
                    payload,
                    symbol=event.symbol,
                    receive_time_ns=received,
                )
                != event
            ):
                raise ValueError("raw liquidation snapshot differs from typed evidence")
        else:
            raise TypeError(f"unsupported impact event type: {type(event).__name__}")

    def append_frame(
        self,
        *,
        run_id: str,
        messages: Sequence[ImpactCaptureMessage],
    ) -> ImpactFrameWriteResult:
        selected = _validate_run_id(run_id)
        if not messages:
            raise ValueError("impact frame cannot be empty")
        connection = self.connect()
        run = connection.execute(
            """
            SELECT status, compressed_payload_cap_bytes, compressed_payload_bytes,
                   payload_cap_reached, frame_count, last_frame_sha256
            FROM impact_capture_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if run is None or str(run[0]) != "running":
            raise ValueError("capture run is missing or not running")
        if bool(run[3]):
            raise ValueError("capture compressed-payload cap has already been reached")
        frame_index = int(run[4])
        previous_frame_sha256 = str(run[5])
        active_segments = self._active_segments(selected)
        lane_rows = connection.execute(
            """
            SELECT stream, connection_id, last_sequence_number,
                   last_received_monotonic_ns
            FROM impact_capture_lane_state WHERE run_id = ?
            """,
            [selected],
        ).fetchall()
        lanes = {
            (str(stream), str(connection_id)): (int(sequence), int(monotonic))
            for stream, connection_id, sequence, monotonic in lane_rows
        }

        records: list[ImpactCaptureFrameRecord] = []
        identities: list[tuple[str, str, int | None, int | None, int | None]] = []
        for message in messages:
            _reject_nonfinite(message.event, "event")
            if message.l2_state is not None:
                _reject_nonfinite(message.l2_state, "l2_state")
            self._validate_raw_semantics(message)
            event_type, symbol, event_time, transaction_time, update_id = (
                self._event_identity(message.event)
            )
            if message.record.stream != self._expected_stream(event_type):
                raise ValueError("typed event does not match the exact-wire stream")
            segment_id = str(message.segment_id).strip().lower()
            if event_type in {"serverTime", "exchangeInfo"}:
                if segment_id:
                    raise ValueError(
                        "global REST evidence cannot belong to a symbol segment"
                    )
            else:
                segment_id = _validate_run_id(segment_id, "segment ID")
                if active_segments.get(segment_id) != symbol:
                    raise ValueError(
                        "typed event does not belong to an active symbol segment"
                    )
            if isinstance(message.event, DepthUpdate):
                if message.event.stale and message.l2_state is not None:
                    raise ValueError("stale depth evidence cannot create an L2 state")
                if not message.event.stale:
                    if message.l2_state is None:
                        raise ValueError("accepted depth evidence requires an L2 state")
                    if (
                        message.l2_state.symbol != symbol
                        or message.l2_state.update_id != message.event.final_update_id
                    ):
                        raise ValueError("depth update and L2 state identity differ")
                    if (
                        len(message.l2_state.bid_levels) != 20
                        or len(message.l2_state.ask_levels) != 20
                    ):
                        raise ValueError(
                            "stored L2 state must contain exactly 20 levels per side"
                        )
            elif message.l2_state is not None:
                raise ValueError("only depth evidence may carry an L2 state")
            event_receive_time = getattr(message.event, "receive_time_ns", None)
            if (
                event_receive_time is not None
                and int(event_receive_time) != message.record.received_monotonic_ns
            ):
                raise ValueError(
                    "typed event receipt time differs from exact-wire metadata"
                )
            lane = (message.record.stream, message.record.connection_id)
            prior = lanes.get(lane)
            expected_sequence = 0 if prior is None else prior[0] + 1
            if message.record.sequence_number != expected_sequence:
                raise ValueError(
                    f"capture lane sequence mismatch: expected={expected_sequence} "
                    f"actual={message.record.sequence_number}"
                )
            if prior is not None and message.record.received_monotonic_ns <= prior[1]:
                raise ValueError("capture lane monotonic receipt time did not increase")
            lanes[lane] = (
                message.record.sequence_number,
                message.record.received_monotonic_ns,
            )
            records.append(message.record)
            identities.append(
                (event_type, symbol, event_time, transaction_time, update_id)
            )

        uncompressed, located = encode_impact_capture_frame(records)
        uncompressed_sha256 = hashlib.sha256(uncompressed).hexdigest()
        compressed = zstandard.ZstdCompressor(
            level=IMPACT_CAPTURE_COMPRESSION_LEVEL,
            write_checksum=True,
            write_content_size=True,
            threads=0,
        ).compress(uncompressed)
        compressed_sha256 = hashlib.sha256(compressed).hexdigest()
        message_ids: list[str] = []
        event_rows: list[tuple[object, ...]] = []
        depth_rows: list[tuple[object, ...]] = []
        l2_rows: list[tuple[object, ...]] = []
        ticker_rows: list[tuple[object, ...]] = []
        trade_rows: list[tuple[object, ...]] = []
        mark_rows: list[tuple[object, ...]] = []
        liquidation_rows: list[tuple[object, ...]] = []
        rest_rows: list[tuple[object, ...]] = []
        stream_counts: dict[str, int] = {}

        for index, (message, item, identity) in enumerate(
            zip(messages, located, identities, strict=True)
        ):
            event_type, symbol, event_time, transaction_time, update_id = identity
            raw_sha256 = hashlib.sha256(
                message.record.raw_text.encode("utf-8")
            ).hexdigest()
            message_id = _canonical_sha256(
                {
                    "run_id": selected,
                    "stream": message.record.stream,
                    "connection_id": message.record.connection_id,
                    "sequence_number": message.record.sequence_number,
                    "raw_payload_sha256": raw_sha256,
                }
            )
            message_ids.append(message_id)
            stream_counts[message.record.stream] = (
                stream_counts.get(message.record.stream, 0) + 1
            )
            segment_id = str(message.segment_id).strip().lower()
            event_row_prefix = (
                selected,
                frame_index,
                index,
                message_id,
                segment_id,
                message.record.stream,
                message.record.connection_id,
                message.record.sequence_number,
                message.record.received_wall_ns,
                message.record.received_monotonic_ns,
                raw_sha256,
                event_type,
                symbol,
                event_time,
                transaction_time,
                update_id,
            )
            event = message.event
            key = (selected, frame_index, index, segment_id)
            typed_row_for_hash: tuple[object, ...] | None = None
            l2_row_for_hash: tuple[object, ...] | None = None
            if isinstance(event, DepthUpdate):
                sums = {
                    ("bid", "added_qty"): 0.0,
                    ("bid", "removed_qty"): 0.0,
                    ("ask", "added_qty"): 0.0,
                    ("ask", "removed_qty"): 0.0,
                    ("bid", "added_quote"): 0.0,
                    ("bid", "removed_quote"): 0.0,
                    ("ask", "added_quote"): 0.0,
                    ("ask", "removed_quote"): 0.0,
                }
                for change in event.changes:
                    sums[(change.side, "added_qty")] += change.added_qty
                    sums[(change.side, "removed_qty")] += change.removed_qty
                    sums[(change.side, "added_quote")] += change.added_quote
                    sums[(change.side, "removed_quote")] += change.removed_quote
                depth_rows.append(
                    key
                    + (
                        event.symbol,
                        event.first_update_id,
                        event.final_update_id,
                        event.previous_update_id,
                        event.stale,
                        event.best_bid,
                        event.best_ask,
                        sums[("bid", "added_qty")],
                        sums[("bid", "removed_qty")],
                        sums[("ask", "added_qty")],
                        sums[("ask", "removed_qty")],
                        sums[("bid", "added_quote")],
                        sums[("bid", "removed_quote")],
                        sums[("ask", "added_quote")],
                        sums[("ask", "removed_quote")],
                    )
                )
                typed_row_for_hash = depth_rows[-1]
                if message.l2_state is not None:
                    state = message.l2_state
                    l2_rows.append(
                        key
                        + (
                            state.symbol,
                            state.update_id,
                            state.best_bid,
                            state.best_ask,
                            state.spread_bps,
                            state.mid,
                            [price for price, _qty in state.bid_levels],
                            [qty for _price, qty in state.bid_levels],
                            [price for price, _qty in state.ask_levels],
                            [qty for _price, qty in state.ask_levels],
                            state.bid_depth_quote_5,
                            state.ask_depth_quote_5,
                            state.bid_depth_quote_10,
                            state.ask_depth_quote_10,
                            state.bid_depth_quote_20,
                            state.ask_depth_quote_20,
                            state.imbalance_5,
                            state.imbalance_10,
                            state.imbalance_20,
                        )
                    )
                    l2_row_for_hash = l2_rows[-1]
            elif isinstance(event, BookTickerEvent):
                ticker_rows.append(
                    key
                    + (
                        event.symbol,
                        event.event_time_ms,
                        event.transaction_time_ms,
                        event.update_id,
                        event.bid,
                        event.bid_qty,
                        event.ask,
                        event.ask_qty,
                    )
                )
                typed_row_for_hash = ticker_rows[-1]
            elif isinstance(event, AggregateTradeEvent):
                trade_rows.append(
                    key
                    + (
                        event.symbol,
                        event.event_time_ms,
                        event.transaction_time_ms,
                        event.aggregate_trade_id,
                        event.first_trade_id,
                        event.last_trade_id,
                        event.price,
                        event.qty,
                        event.normalized_qty,
                        event.buyer_is_maker,
                    )
                )
                typed_row_for_hash = trade_rows[-1]
            elif isinstance(event, MarkPriceEvent):
                mark_rows.append(
                    key
                    + (
                        event.symbol,
                        event.event_time_ms,
                        event.mark_price,
                        event.index_price,
                        event.estimated_settlement_price,
                        event.funding_rate,
                        event.next_funding_time_ms,
                    )
                )
                typed_row_for_hash = mark_rows[-1]
            elif isinstance(event, LiquidationSnapshotEvent):
                liquidation_rows.append(
                    key
                    + (
                        event.symbol,
                        event.event_time_ms,
                        event.order_time_ms,
                        event.side,
                        event.order_type,
                        event.time_in_force,
                        event.original_qty,
                        event.price,
                        event.average_price,
                        event.order_status,
                        event.last_filled_qty,
                        event.accumulated_filled_qty,
                    )
                )
                typed_row_for_hash = liquidation_rows[-1]
            elif isinstance(event, ImpactRestEvent):
                (
                    rest_symbol,
                    path,
                    parameters_json,
                    status,
                    request_started_wall_ns,
                    request_started_monotonic_ns,
                    used_weight_1m,
                    exchange_time,
                    rest_update_id,
                    open_interest,
                ) = self._rest_event(event)
                rest_rows.append(
                    key
                    + (
                        event.event_type,
                        rest_symbol,
                        path,
                        parameters_json,
                        status,
                        request_started_wall_ns,
                        request_started_monotonic_ns,
                        used_weight_1m,
                        exchange_time,
                        rest_update_id,
                        open_interest,
                    )
                )
                typed_row_for_hash = rest_rows[-1]
            else:
                raise TypeError(
                    f"unsupported impact event type: {type(event).__name__}"
                )
            if typed_row_for_hash is None:
                raise RuntimeError("typed event row was not materialized")
            typed_event_sha256 = _canonical_sha256(
                {
                    "event_type": event_type,
                    "typed_row": typed_row_for_hash,
                    "l2_row": l2_row_for_hash,
                }
            )
            event_rows.append(event_row_prefix + (typed_event_sha256,))
            if item.message_index != index:
                raise RuntimeError("encoded frame message index changed unexpectedly")

        manifest_sha256 = _canonical_sha256(message_ids)
        stream_counts_json = _canonical_json(dict(sorted(stream_counts.items())))
        first_received_wall_ns = min(record.received_wall_ns for record in records)
        last_received_wall_ns = max(record.received_wall_ns for record in records)
        first_received_monotonic_ns = min(
            record.received_monotonic_ns for record in records
        )
        last_received_monotonic_ns = max(
            record.received_monotonic_ns for record in records
        )
        frame_identity = {
            "schema_version": IMPACT_CAPTURE_SCHEMA_VERSION,
            "capture_contract_sha256": IMPACT_CAPTURE_CONTRACT_SHA256,
            "run_id": selected,
            "frame_index": frame_index,
            "previous_frame_sha256": previous_frame_sha256,
            "message_count": len(messages),
            "first_message_id": message_ids[0],
            "last_message_id": message_ids[-1],
            "message_manifest_sha256": manifest_sha256,
            "first_received_wall_ns": first_received_wall_ns,
            "last_received_wall_ns": last_received_wall_ns,
            "first_received_monotonic_ns": first_received_monotonic_ns,
            "last_received_monotonic_ns": last_received_monotonic_ns,
            "uncompressed_bytes": len(uncompressed),
            "uncompressed_sha256": uncompressed_sha256,
            "compressed_bytes": len(compressed),
            "compressed_sha256": compressed_sha256,
            "stream_counts_json": stream_counts_json,
        }
        frame_sha256 = _canonical_sha256(frame_identity)
        total_compressed = int(run[2]) + len(compressed)
        cap_reached = total_compressed >= int(run[1])

        lane_updates = [
            (selected, stream, connection_id, sequence, monotonic)
            for (stream, connection_id), (sequence, monotonic) in sorted(lanes.items())
        ]
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                INSERT INTO impact_capture_frame VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    selected,
                    frame_index,
                    IMPACT_CAPTURE_SCHEMA_VERSION,
                    IMPACT_CAPTURE_FRAME_FORMAT,
                    previous_frame_sha256,
                    len(messages),
                    message_ids[0],
                    message_ids[-1],
                    manifest_sha256,
                    first_received_wall_ns,
                    last_received_wall_ns,
                    first_received_monotonic_ns,
                    last_received_monotonic_ns,
                    len(uncompressed),
                    uncompressed_sha256,
                    len(compressed),
                    compressed_sha256,
                    stream_counts_json,
                    compressed,
                    frame_sha256,
                ],
            )
            for table, columns, rows in (
                ("impact_event_index", _EVENT_COLUMNS, event_rows),
                ("impact_depth_update", _DEPTH_COLUMNS, depth_rows),
                ("impact_l2_state", _L2_COLUMNS, l2_rows),
                ("impact_book_ticker", _BOOK_TICKER_COLUMNS, ticker_rows),
                ("impact_aggregate_trade", _TRADE_COLUMNS, trade_rows),
                ("impact_mark_price", _MARK_COLUMNS, mark_rows),
                ("impact_liquidation_snapshot", _LIQUIDATION_COLUMNS, liquidation_rows),
                ("impact_rest_event", _REST_COLUMNS, rest_rows),
            ):
                if rows:
                    insert_rows_columnar(
                        connection,
                        sql=_insert_sql(table, columns),
                        rows=rows,
                        width=len(columns),
                        batch_size=1_024,
                    )
            connection.executemany(
                """
                INSERT INTO impact_capture_lane_state VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (run_id, stream, connection_id) DO UPDATE SET
                    last_sequence_number = excluded.last_sequence_number,
                    last_received_monotonic_ns = excluded.last_received_monotonic_ns
                """,
                lane_updates,
            )
            for segment_id, symbol in active_segments.items():
                accepted = [
                    event
                    for message in messages
                    if str(message.segment_id).strip().lower() == segment_id
                    and isinstance((event := message.event), DepthUpdate)
                    and not event.stale
                ]
                if not accepted:
                    continue
                connection.execute(
                    """
                    UPDATE impact_capture_segment
                    SET first_final_update_id = coalesce(first_final_update_id, ?),
                        last_final_update_id = ?,
                        depth_message_count = depth_message_count + ?
                    WHERE run_id = ? AND segment_id = ? AND symbol = ? AND status = 'active'
                    """,
                    [
                        accepted[0].final_update_id,
                        accepted[-1].final_update_id,
                        len(accepted),
                        selected,
                        segment_id,
                        symbol,
                    ],
                )
            updated = connection.execute(
                """
                UPDATE impact_capture_run
                SET compressed_payload_bytes = ?, payload_cap_reached = ?,
                    frame_count = frame_count + 1,
                    message_count = message_count + ?, last_frame_sha256 = ?
                WHERE run_id = ? AND status = 'running' AND frame_count = ?
                RETURNING run_id
                """,
                [
                    total_compressed,
                    cap_reached,
                    len(messages),
                    frame_sha256,
                    selected,
                    frame_index,
                ],
            ).fetchone()
            if updated is None:
                raise RuntimeError("capture run changed during atomic frame append")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        return ImpactFrameWriteResult(
            run_id=selected,
            frame_index=frame_index,
            frame_sha256=frame_sha256,
            message_count=len(messages),
            uncompressed_bytes=len(uncompressed),
            compressed_bytes=len(compressed),
            compressed_payload_total_bytes=total_compressed,
            payload_cap_reached=cap_reached,
        )

    @staticmethod
    def _stored_typed_rows(
        connection: duckdb.DuckDBPyConnection,
        *,
        run_id: str,
        frame_index: int,
        message_index: int,
        event_type: str,
    ) -> tuple[tuple[object, ...] | None, tuple[object, ...] | None]:
        table_contract = {
            "depthUpdate": ("impact_depth_update", _DEPTH_COLUMNS),
            "bookTicker": ("impact_book_ticker", _BOOK_TICKER_COLUMNS),
            "aggTrade": ("impact_aggregate_trade", _TRADE_COLUMNS),
            "markPriceUpdate": ("impact_mark_price", _MARK_COLUMNS),
            "forceOrder": ("impact_liquidation_snapshot", _LIQUIDATION_COLUMNS),
            "serverTime": ("impact_rest_event", _REST_COLUMNS),
            "exchangeInfo": ("impact_rest_event", _REST_COLUMNS),
            "depthSnapshot": ("impact_rest_event", _REST_COLUMNS),
            "openInterest": ("impact_rest_event", _REST_COLUMNS),
        }
        contract = table_contract.get(event_type)
        if contract is None:
            return None, None
        table, columns = contract
        typed = connection.execute(
            f"""
            SELECT {", ".join(columns)} FROM {table}
            WHERE run_id = ? AND frame_index = ? AND message_index = ?
            """,
            [run_id, frame_index, message_index],
        ).fetchone()
        l2_row = None
        if event_type == "depthUpdate":
            l2_row = connection.execute(
                f"""
                SELECT {", ".join(_L2_COLUMNS)} FROM impact_l2_state
                WHERE run_id = ? AND frame_index = ? AND message_index = ?
                """,
                [run_id, frame_index, message_index],
            ).fetchone()
        return (
            None if typed is None else tuple(typed),
            None if l2_row is None else tuple(l2_row),
        )

    def audit_run(self, run_id: str) -> ImpactCaptureAudit:
        selected = _validate_run_id(run_id)
        connection = self.connect()
        run = connection.execute(
            """
            SELECT schema_version, design_sha256, capture_contract_sha256,
                   frame_count, message_count, compressed_payload_bytes,
                   last_frame_sha256
            FROM impact_capture_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if run is None:
            raise ValueError("capture run was not found")
        errors: list[str] = []
        if str(run[0]) != IMPACT_CAPTURE_SCHEMA_VERSION:
            errors.append("run_schema_mismatch")
        if str(run[1]) != ROUND73_DESIGN_SHA256:
            errors.append("run_design_mismatch")
        if str(run[2]) != IMPACT_CAPTURE_CONTRACT_SHA256:
            errors.append("run_capture_contract_mismatch")
        frames = connection.execute(
            """
            SELECT frame_index, schema_version, frame_format,
                   previous_frame_sha256, message_count, first_message_id,
                   last_message_id, message_manifest_sha256,
                   first_received_wall_ns, last_received_wall_ns,
                   first_received_monotonic_ns, last_received_monotonic_ns,
                   uncompressed_bytes, uncompressed_sha256,
                   compressed_bytes, compressed_sha256, stream_counts_json,
                   compressed_payload, frame_sha256
            FROM impact_capture_frame WHERE run_id = ? ORDER BY frame_index
            """,
            [selected],
        ).fetchall()
        prior_frame_sha256 = ""
        total_messages = 0
        total_compressed = 0
        lane_state: dict[tuple[str, str], tuple[int, int]] = {}
        decompressor = zstandard.ZstdDecompressor()
        for expected_index, row in enumerate(frames):
            frame_index = int(row[0])
            if frame_index != expected_index:
                errors.append(f"frame_index_gap:{expected_index}:{frame_index}")
            if str(row[1]) != IMPACT_CAPTURE_SCHEMA_VERSION:
                errors.append(f"frame_schema_mismatch:{frame_index}")
            if str(row[2]) != IMPACT_CAPTURE_FRAME_FORMAT:
                errors.append(f"frame_format_mismatch:{frame_index}")
            if str(row[3]) != prior_frame_sha256:
                errors.append(f"frame_chain_mismatch:{frame_index}")
            message_count = int(row[4])
            compressed = bytes(row[17])
            if len(compressed) != int(row[14]):
                errors.append(f"compressed_size_mismatch:{frame_index}")
            if hashlib.sha256(compressed).hexdigest() != str(row[15]):
                errors.append(f"compressed_sha256_mismatch:{frame_index}")
            try:
                uncompressed = decompressor.decompress(
                    compressed,
                    max_output_size=int(row[12]),
                )
            except zstandard.ZstdError as exc:
                errors.append(
                    f"frame_decompression_failed:{frame_index}:{type(exc).__name__}"
                )
                continue
            if len(uncompressed) != int(row[12]):
                errors.append(f"uncompressed_size_mismatch:{frame_index}")
            if hashlib.sha256(uncompressed).hexdigest() != str(row[13]):
                errors.append(f"uncompressed_sha256_mismatch:{frame_index}")
            try:
                decoded = decode_impact_capture_frame(
                    uncompressed,
                    expected_message_count=message_count,
                )
            except ValueError as exc:
                errors.append(f"frame_decode_failed:{frame_index}:{type(exc).__name__}")
                continue
            index_rows = connection.execute(
                """
                SELECT message_index, message_id, stream, connection_id,
                       sequence_number, received_wall_ns, received_monotonic_ns,
                       raw_payload_sha256, event_type, typed_event_sha256
                FROM impact_event_index
                WHERE run_id = ? AND frame_index = ? ORDER BY message_index
                """,
                [selected, frame_index],
            ).fetchall()
            if len(index_rows) != message_count:
                errors.append(f"event_index_count_mismatch:{frame_index}")
            message_ids: list[str] = []
            stream_counts: dict[str, int] = {}
            for message_index, item in enumerate(decoded):
                if message_index >= len(index_rows):
                    break
                indexed = index_rows[message_index]
                record = item.record
                raw_sha256 = hashlib.sha256(record.raw_text.encode("utf-8")).hexdigest()
                message_id = _canonical_sha256(
                    {
                        "run_id": selected,
                        "stream": record.stream,
                        "connection_id": record.connection_id,
                        "sequence_number": record.sequence_number,
                        "raw_payload_sha256": raw_sha256,
                    }
                )
                expected_values = (
                    message_index,
                    message_id,
                    record.stream,
                    record.connection_id,
                    record.sequence_number,
                    record.received_wall_ns,
                    record.received_monotonic_ns,
                    raw_sha256,
                )
                if tuple(indexed[:8]) != expected_values:
                    errors.append(
                        f"raw_to_index_mismatch:{frame_index}:{message_index}"
                    )
                indexed_event_type = str(indexed[8])
                try:
                    root = _strict_json_object(record.raw_text)
                    if record.stream != "binance_futures_rest":
                        raw_payload = root.get("data")
                        if (
                            not isinstance(raw_payload, Mapping)
                            or str(raw_payload.get("e", "")) != indexed_event_type
                        ):
                            errors.append(
                                f"raw_event_type_mismatch:{frame_index}:{message_index}"
                            )
                except ValueError:
                    errors.append(f"raw_json_invalid:{frame_index}:{message_index}")
                typed_row, l2_row = self._stored_typed_rows(
                    connection,
                    run_id=selected,
                    frame_index=frame_index,
                    message_index=message_index,
                    event_type=indexed_event_type,
                )
                if typed_row is None:
                    errors.append(f"typed_row_missing:{frame_index}:{message_index}")
                else:
                    typed_sha256 = _canonical_sha256(
                        {
                            "event_type": indexed_event_type,
                            "typed_row": typed_row,
                            "l2_row": l2_row,
                        }
                    )
                    if typed_sha256 != str(indexed[9]):
                        errors.append(
                            f"typed_sha256_mismatch:{frame_index}:{message_index}"
                        )
                message_ids.append(message_id)
                stream_counts[record.stream] = stream_counts.get(record.stream, 0) + 1
                lane = (record.stream, record.connection_id)
                prior = lane_state.get(lane)
                expected_sequence = 0 if prior is None else prior[0] + 1
                if record.sequence_number != expected_sequence:
                    errors.append(
                        f"lane_sequence_mismatch:{frame_index}:{message_index}"
                    )
                if prior is not None and record.received_monotonic_ns <= prior[1]:
                    errors.append(
                        f"lane_monotonic_mismatch:{frame_index}:{message_index}"
                    )
                lane_state[lane] = (
                    record.sequence_number,
                    record.received_monotonic_ns,
                )
            manifest_sha256 = _canonical_sha256(message_ids)
            if message_ids:
                if str(row[5]) != message_ids[0] or str(row[6]) != message_ids[-1]:
                    errors.append(f"frame_message_bounds_mismatch:{frame_index}")
            if str(row[7]) != manifest_sha256:
                errors.append(f"frame_message_manifest_mismatch:{frame_index}")
            stream_counts_json = _canonical_json(dict(sorted(stream_counts.items())))
            if str(row[16]) != stream_counts_json:
                errors.append(f"frame_stream_counts_mismatch:{frame_index}")
            frame_identity = {
                "schema_version": IMPACT_CAPTURE_SCHEMA_VERSION,
                "capture_contract_sha256": IMPACT_CAPTURE_CONTRACT_SHA256,
                "run_id": selected,
                "frame_index": frame_index,
                "previous_frame_sha256": str(row[3]),
                "message_count": message_count,
                "first_message_id": str(row[5]),
                "last_message_id": str(row[6]),
                "message_manifest_sha256": str(row[7]),
                "first_received_wall_ns": int(row[8]),
                "last_received_wall_ns": int(row[9]),
                "first_received_monotonic_ns": int(row[10]),
                "last_received_monotonic_ns": int(row[11]),
                "uncompressed_bytes": int(row[12]),
                "uncompressed_sha256": str(row[13]),
                "compressed_bytes": int(row[14]),
                "compressed_sha256": str(row[15]),
                "stream_counts_json": str(row[16]),
            }
            calculated_frame_sha256 = _canonical_sha256(frame_identity)
            if calculated_frame_sha256 != str(row[18]):
                errors.append(f"frame_identity_mismatch:{frame_index}")
            prior_frame_sha256 = str(row[18])
            total_messages += message_count
            total_compressed += int(row[14])

        typed_tables = {
            "depthUpdate": "impact_depth_update",
            "bookTicker": "impact_book_ticker",
            "aggTrade": "impact_aggregate_trade",
            "markPriceUpdate": "impact_mark_price",
            "forceOrder": "impact_liquidation_snapshot",
            "serverTime": "impact_rest_event",
            "exchangeInfo": "impact_rest_event",
            "depthSnapshot": "impact_rest_event",
            "openInterest": "impact_rest_event",
        }
        expected_counts = {
            str(event_type): int(count)
            for event_type, count in connection.execute(
                """
                SELECT event_type, count(*) FROM impact_event_index
                WHERE run_id = ? GROUP BY event_type
                """,
                [selected],
            ).fetchall()
        }
        for event_type, table in typed_tables.items():
            if table == "impact_rest_event":
                actual = int(
                    connection.execute(
                        f"SELECT count(*) FROM {table} WHERE run_id = ? AND event_type = ?",
                        [selected, event_type],
                    ).fetchone()[0]
                )
            else:
                actual = int(
                    connection.execute(
                        f"SELECT count(*) FROM {table} WHERE run_id = ?", [selected]
                    ).fetchone()[0]
                )
            if actual != expected_counts.get(event_type, 0):
                errors.append(f"typed_count_mismatch:{event_type}")

        stored_lanes = {
            (str(stream), str(connection_id)): (int(sequence), int(monotonic))
            for stream, connection_id, sequence, monotonic in connection.execute(
                """
                SELECT stream, connection_id, last_sequence_number,
                       last_received_monotonic_ns
                FROM impact_capture_lane_state WHERE run_id = ?
                """,
                [selected],
            ).fetchall()
        }
        if stored_lanes != lane_state:
            errors.append("lane_terminal_state_mismatch")
        if int(run[3]) != len(frames):
            errors.append("run_frame_count_mismatch")
        if int(run[4]) != total_messages:
            errors.append("run_message_count_mismatch")
        if int(run[5]) != total_compressed:
            errors.append("run_compressed_payload_bytes_mismatch")
        if str(run[6]) != prior_frame_sha256:
            errors.append("run_last_frame_sha256_mismatch")
        return ImpactCaptureAudit(
            run_id=selected,
            passed=not errors,
            errors=tuple(errors),
            frame_count=len(frames),
            message_count=total_messages,
            compressed_payload_bytes=total_compressed,
            last_frame_sha256=prior_frame_sha256,
        )


__all__ = [
    "IMPACT_CAPTURE_COMPRESSION_LEVEL",
    "IMPACT_CAPTURE_CONTRACT_SHA256",
    "IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES",
    "IMPACT_CAPTURE_SCHEMA_VERSION",
    "IMPACT_CAPTURE_SYMBOLS",
    "ImpactAbsorptionStore",
    "ImpactCaptureAudit",
    "ImpactCaptureMessage",
    "ImpactFrameWriteResult",
    "ImpactRestEvent",
]
