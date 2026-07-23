"""Transactional Round 73 prospective evidence storage and audit."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Iterator, Literal, Mapping, Sequence, TypeAlias

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
    ROUND73_LEVEL_BANDS,
    ROUND73_DESIGN_SHA256,
    pre_event_level_band,
    validate_combined_stream_name,
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


IMPACT_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v8"
IMPACT_CAPTURE_CONTRACT_SHA256 = (
    "b64feb9c4b686b00d1a6a9c464e50e397e258d9f07abbd84702199b387a54462"
)
IMPACT_CAPTURE_REPORT_SCHEMA_VERSION = "round-073-capture-report-v8"
IMPACT_CAPTURE_V9_SCHEMA_VERSION = "round-073-prospective-evidence-v9"
IMPACT_CAPTURE_V9_CONTRACT_SHA256 = (
    "3c105ac411ca4b2cf5469f065f507a21a2442bbcbdf39257239203261586f254"
)
IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION = "round-073-capture-report-v9"
_LEGACY_CAPTURE_CONTRACTS = {
    "round-073-prospective-evidence-v1": (
        "f379b53b86d20f16b686132ef8fe4dc5eb47b6a0910e6ba85c38ddf0caa01c7b"
    ),
    "round-073-prospective-evidence-v2": (
        "1b46f178e335b3473b86ee71a113e2538a9068e287c50f0867aab13f3230557c"
    ),
    "round-073-prospective-evidence-v3": (
        "9228f8243531e44a264d5f88cf8498282986d1b9cb4a6b64e12ee0cede47dc5b"
    ),
    "round-073-prospective-evidence-v4": (
        "c34687c5dff9a4eda98b2e50d6444a12ee1a4f5594806c2410e15cb0242d7529"
    ),
    "round-073-prospective-evidence-v5": (
        "63a440f1fb875db8ee78bab1631033f24850a65cc7ed80d4fd37078dd6ee9a1b"
    ),
    "round-073-prospective-evidence-v6": (
        "a256f16f1904d6c23b4563e7cbb603353dd7e0fe8253e3c3f2df4a67305da021"
    ),
    "round-073-prospective-evidence-v7": (
        "18013fc14bad234b241bf05122a6363ad94e6722a598319ae1059cde1941a9f1"
    ),
}
_V7_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v7"
_V6_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v6"
_V5_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v5"
_V4_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v4"
_V3_CAPTURE_SCHEMA_VERSION = "round-073-prospective-evidence-v3"
_COMPACT_CAPTURE_SCHEMAS = frozenset(
    {
        IMPACT_CAPTURE_SCHEMA_VERSION,
        _V7_CAPTURE_SCHEMA_VERSION,
        _V6_CAPTURE_SCHEMA_VERSION,
        _V5_CAPTURE_SCHEMA_VERSION,
        _V4_CAPTURE_SCHEMA_VERSION,
        _V3_CAPTURE_SCHEMA_VERSION,
    }
)
_EVENT_TIME_LINK_SCHEMAS = frozenset(
    {
        IMPACT_CAPTURE_SCHEMA_VERSION,
        _V7_CAPTURE_SCHEMA_VERSION,
        _V6_CAPTURE_SCHEMA_VERSION,
        _V5_CAPTURE_SCHEMA_VERSION,
        _V4_CAPTURE_SCHEMA_VERSION,
    }
)
_DEPTH_BAND_SCHEMAS = frozenset(
    {
        IMPACT_CAPTURE_SCHEMA_VERSION,
        _V7_CAPTURE_SCHEMA_VERSION,
        _V6_CAPTURE_SCHEMA_VERSION,
        _V5_CAPTURE_SCHEMA_VERSION,
    }
)
_LEGACY_DEPTH_BAND_SCHEMAS = frozenset(
    {
        _V7_CAPTURE_SCHEMA_VERSION,
        _V6_CAPTURE_SCHEMA_VERSION,
        _V5_CAPTURE_SCHEMA_VERSION,
    }
)
IMPACT_CAPTURE_CHECKPOINT_THRESHOLD = "16MiB"
IMPACT_CAPTURE_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES = 100_000
IMPACT_CAPTURE_COMPRESSION_LEVEL = 3
IMPACT_CAPTURE_V9_CHECKPOINT_THRESHOLD = "512MiB"
IMPACT_CAPTURE_V9_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES = 512 * 1024 * 1024
IMPACT_CAPTURE_V9_COMPRESSION_LEVEL = 9
IMPACT_CAPTURE_V9_MAX_CROSS_FRAME_REORDER_LAG_NS = 10_000_000
IMPACT_CAPTURE_INITIAL_COOLDOWN_NS = 60_000_000_000
IMPACT_CAPTURE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES = 2_147_483_648
_AUDIT_FRAME_BATCH_SIZE = 64

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEMORY_LIMIT = re.compile(r"[1-9][0-9]*(?:KB|MB|GB|TB)", re.IGNORECASE)
_DUCKDB_BYTE_SETTING = re.compile(
    r"(?P<amount>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>B|KiB|MiB|GiB|TiB)",
    re.IGNORECASE,
)
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


def validate_impact_store_resources(
    memory_limit: object,
    threads: object,
) -> tuple[str, int]:
    normalized_memory = str(memory_limit).strip().upper()
    if not _MEMORY_LIMIT.fullmatch(normalized_memory):
        raise ValueError(
            "memory_limit must be a positive integer followed by a byte unit"
        )
    normalized_threads = int(threads)
    if not 1 <= normalized_threads <= 8:
        raise ValueError("impact store threads must be between 1 and 8")
    return normalized_memory, normalized_threads


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


def _duckdb_byte_setting_bytes(value: object) -> int:
    candidate = str(value).strip()
    match = _DUCKDB_BYTE_SETTING.fullmatch(candidate)
    if match is None:
        raise ValueError("DuckDB byte setting has an unsupported format")
    unit_multipliers = {
        "b": 1,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    byte_value = (
        float(match.group("amount")) * unit_multipliers[match.group("unit").lower()]
    )
    if not math.isfinite(byte_value) or not byte_value.is_integer():
        raise ValueError("DuckDB byte setting does not resolve to whole bytes")
    return int(byte_value)


def _capture_storage_policy_request(schema_version: object) -> tuple[str, int]:
    selected = str(schema_version)
    if selected == IMPACT_CAPTURE_V9_SCHEMA_VERSION:
        return (
            IMPACT_CAPTURE_V9_CHECKPOINT_THRESHOLD,
            IMPACT_CAPTURE_V9_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES,
        )
    if selected == IMPACT_CAPTURE_SCHEMA_VERSION:
        return (
            IMPACT_CAPTURE_CHECKPOINT_THRESHOLD,
            IMPACT_CAPTURE_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES,
        )
    raise ValueError("capture run schema version is unsupported")


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


@dataclass(frozen=True)
class ImpactRejectedWireEvent:
    """An exact WebSocket receipt rejected before trusted event typing."""

    observed_stream_name: str
    observed_event_type: str
    observed_symbol: str
    rejection_class: str
    rejection_reason: str
    receive_time_ns: int


ImpactParsedEvent: TypeAlias = (
    DepthUpdate
    | BookTickerEvent
    | AggregateTradeEvent
    | MarkPriceEvent
    | LiquidationSnapshotEvent
    | ImpactRestEvent
    | ImpactRejectedWireEvent
)


@dataclass(frozen=True)
class ImpactCaptureMessage:
    """Exact frame record plus its typed, raw-linked observation."""

    record: ImpactCaptureFrameRecord
    event: ImpactParsedEvent
    segment_id: str = ""
    pre_l2_state: L2BookState | None = None
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
    capture_contract_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "round-073-capture-audit-v2",
            "run_id": self.run_id,
            "passed": self.passed,
            "errors": list(self.errors),
            "frame_count": self.frame_count,
            "message_count": self.message_count,
            "compressed_payload_bytes": self.compressed_payload_bytes,
            "last_frame_sha256": self.last_frame_sha256,
            "capture_contract_sha256": self.capture_contract_sha256,
        }


@dataclass(frozen=True)
class ImpactCaptureV9Preflight:
    ready_wall_ns: int
    snapshot_records: tuple[tuple[str, ImpactCaptureFrameRecord], ...]


@dataclass(frozen=True)
class _StoredEventLink:
    message_index: int
    message_id: str | None
    segment_id: str
    stream: str
    connection_id: str
    sequence_number: int
    received_wall_ns: int
    received_monotonic_ns: int
    raw_payload_sha256: str
    event_type: str
    symbol: str
    event_time_ms: int | None
    event_time_stored: bool
    typed_event_sha256: str


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
_EVENT_LINK_V3_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "stream",
    "connection_id",
    "sequence_number",
    "received_wall_ns",
    "received_monotonic_ns",
    "raw_payload_sha256",
    "event_type",
    "symbol",
    "typed_event_sha256",
)
_EVENT_LINK_V4_COLUMNS = (
    *_EVENT_LINK_V3_COLUMNS[:-1],
    "event_time_ms",
    "typed_event_sha256",
)
_EVENT_LINK_V5_COLUMNS = _EVENT_LINK_V4_COLUMNS
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
_REJECTED_WIRE_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "observed_stream_name",
    "observed_event_type",
    "observed_symbol",
    "rejection_class",
    "rejection_reason",
    "receive_time_ns",
)

_DEPTH_BAND_FLOW_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "symbol",
    *(
        f"{side}_{metric}_{band}"
        for side in ("bid", "ask")
        for band in ROUND73_LEVEL_BANDS
        for metric in ("added_quote", "removed_quote", "change_count")
    ),
)

IMPACT_CAPTURE_FRAME_TABLE = "impact_capture_frame_v8"
IMPACT_EVENT_LINK_TABLE = "impact_event_link_v8"
IMPACT_DEPTH_UPDATE_TABLE = "impact_depth_update_v8"
IMPACT_L2_STATE_TABLE = "impact_l2_state_v8"
IMPACT_BOOK_TICKER_TABLE = "impact_book_ticker_v8"
IMPACT_AGGREGATE_TRADE_TABLE = "impact_aggregate_trade_v8"
IMPACT_MARK_PRICE_TABLE = "impact_mark_price_v8"
IMPACT_LIQUIDATION_SNAPSHOT_TABLE = "impact_liquidation_snapshot_v8"
IMPACT_REST_EVENT_TABLE = "impact_rest_event_v8"
IMPACT_REJECTED_WIRE_EVENT_TABLE = "impact_rejected_wire_event_v8"
IMPACT_DEPTH_BAND_FLOW_TABLE = "impact_depth_band_flow_v8"
IMPACT_CAPTURE_V9_FRAME_TABLE = "impact_capture_frame_v9"
IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE = "impact_rest_event_v9"

_LEGACY_TYPED_TABLES = {
    "depthUpdate": ("impact_depth_update", _DEPTH_COLUMNS),
    "bookTicker": ("impact_book_ticker", _BOOK_TICKER_COLUMNS),
    "aggTrade": ("impact_aggregate_trade", _TRADE_COLUMNS),
    "markPriceUpdate": ("impact_mark_price", _MARK_COLUMNS),
    "forceOrder": ("impact_liquidation_snapshot", _LIQUIDATION_COLUMNS),
    "serverTime": ("impact_rest_event", _REST_COLUMNS),
    "exchangeInfo": ("impact_rest_event", _REST_COLUMNS),
    "depthSnapshot": ("impact_rest_event", _REST_COLUMNS),
    "openInterest": ("impact_rest_event", _REST_COLUMNS),
    "rejectedWire": ("impact_rejected_wire_event", _REJECTED_WIRE_COLUMNS),
}
_V3_TYPED_TABLES = {
    "depthUpdate": ("impact_depth_update_v3", _DEPTH_COLUMNS),
    "bookTicker": ("impact_book_ticker_v3", _BOOK_TICKER_COLUMNS),
    "aggTrade": ("impact_aggregate_trade_v3", _TRADE_COLUMNS),
    "markPriceUpdate": ("impact_mark_price_v3", _MARK_COLUMNS),
    "forceOrder": ("impact_liquidation_snapshot_v3", _LIQUIDATION_COLUMNS),
    "serverTime": ("impact_rest_event_v3", _REST_COLUMNS),
    "exchangeInfo": ("impact_rest_event_v3", _REST_COLUMNS),
    "depthSnapshot": ("impact_rest_event_v3", _REST_COLUMNS),
    "openInterest": ("impact_rest_event_v3", _REST_COLUMNS),
    "rejectedWire": ("impact_rejected_wire_event_v3", _REJECTED_WIRE_COLUMNS),
}
_V8_TYPED_TABLES = {
    "depthUpdate": (IMPACT_DEPTH_UPDATE_TABLE, _DEPTH_COLUMNS),
    "bookTicker": (IMPACT_BOOK_TICKER_TABLE, _BOOK_TICKER_COLUMNS),
    "aggTrade": (IMPACT_AGGREGATE_TRADE_TABLE, _TRADE_COLUMNS),
    "markPriceUpdate": (IMPACT_MARK_PRICE_TABLE, _MARK_COLUMNS),
    "forceOrder": (IMPACT_LIQUIDATION_SNAPSHOT_TABLE, _LIQUIDATION_COLUMNS),
    "serverTime": (IMPACT_REST_EVENT_TABLE, _REST_COLUMNS),
    "exchangeInfo": (IMPACT_REST_EVENT_TABLE, _REST_COLUMNS),
    "depthSnapshot": (IMPACT_REST_EVENT_TABLE, _REST_COLUMNS),
    "openInterest": (IMPACT_REST_EVENT_TABLE, _REST_COLUMNS),
    "rejectedWire": (IMPACT_REJECTED_WIRE_EVENT_TABLE, _REJECTED_WIRE_COLUMNS),
}


def _typed_tables_for_schema(
    schema_version: str,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    if schema_version == IMPACT_CAPTURE_SCHEMA_VERSION:
        return dict(_V8_TYPED_TABLES)
    if schema_version in _COMPACT_CAPTURE_SCHEMAS:
        return dict(_V3_TYPED_TABLES)
    if schema_version in _LEGACY_CAPTURE_CONTRACTS:
        tables = dict(_LEGACY_TYPED_TABLES)
        if schema_version == "round-073-prospective-evidence-v1":
            tables.pop("rejectedWire")
        return tables
    raise ValueError("impact capture schema version is unsupported")


def _quoted_identifier(value: str) -> str:
    return f'"{str(value).replace(chr(34), chr(34) * 2)}"'


def _ensure_isolated_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_table: str,
    target_table: str,
    primary_key: tuple[str, ...] = (),
    digest_length_check_columns: tuple[str, ...] = (),
) -> None:
    source_info = connection.execute(f"PRAGMA table_info('{source_table}')").fetchall()
    if not source_info:
        raise RuntimeError(f"source table is missing: {source_table}")
    target_exists = bool(
        connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [target_table],
        ).fetchone()[0]
    )
    target_info = (
        connection.execute(f"PRAGMA table_info('{target_table}')").fetchall()
        if target_exists
        else []
    )
    if target_info:
        source_signature = tuple(
            (str(row[1]), str(row[2]), bool(row[3])) for row in source_info
        )
        target_signature = tuple(
            (str(row[1]), str(row[2]), bool(row[3])) for row in target_info
        )
        target_primary_key = tuple(str(row[1]) for row in target_info if bool(row[5]))
        target_digest_checks = {
            tuple(str(column) for column in row[0])
            for row in connection.execute(
                "SELECT constraint_column_names FROM duckdb_constraints() "
                "WHERE table_name = ? AND constraint_type = 'CHECK'",
                [target_table],
            ).fetchall()
        }
        expected_digest_checks = {(column,) for column in digest_length_check_columns}
        if (
            source_signature != target_signature
            or target_primary_key != primary_key
            or target_digest_checks != expected_digest_checks
        ):
            raise RuntimeError(f"isolated table schema mismatch: {target_table}")
        return
    definitions = []
    for _cid, name, data_type, not_null, default_value, _pk in source_info:
        definition = f"{_quoted_identifier(str(name))} {data_type}"
        if bool(not_null):
            definition += " NOT NULL"
        if default_value is not None:
            definition += f" DEFAULT {default_value}"
        definitions.append(definition)
    constraints = [
        f"CHECK (octet_length({_quoted_identifier(column)}) = 32)"
        for column in digest_length_check_columns
    ]
    if primary_key:
        constraints.append(
            "PRIMARY KEY ("
            + ", ".join(_quoted_identifier(column) for column in primary_key)
            + ")"
        )
    connection.execute(
        f"CREATE TABLE {_quoted_identifier(target_table)} ("
        + ", ".join((*definitions, *constraints))
        + ")"
    )


def iter_impact_capture_v9_records(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> Iterator[tuple[int, int, ImpactCaptureFrameRecord]]:
    """Yield v9 exact-wire records in bounded global receipt order."""

    selected = _validate_run_id(run_id)
    cursor = connection.cursor()
    cursor.execute(
        f"""
        SELECT frame_index, message_count, uncompressed_bytes, compressed_payload
        FROM {IMPACT_CAPTURE_V9_FRAME_TABLE}
        WHERE run_id = ? ORDER BY frame_index
        """,
        [selected],
    )
    decompressor = zstandard.ZstdDecompressor()
    heap: list[tuple[int, int, int, ImpactCaptureFrameRecord]] = []
    running_high_water = 0
    prior_frame_high_water = 0
    try:
        while frame_rows := cursor.fetchmany(_AUDIT_FRAME_BATCH_SIZE):
            for (
                frame_index_value,
                message_count,
                uncompressed_bytes,
                blob,
            ) in frame_rows:
                frame_index = int(frame_index_value)
                decoded = decode_impact_capture_frame(
                    decompressor.decompress(
                        bytes(blob),
                        max_output_size=int(uncompressed_bytes),
                    ),
                    expected_message_count=int(message_count),
                )
                receipts = [item.record.received_monotonic_ns for item in decoded]
                if receipts != sorted(receipts):
                    raise ValueError("Round 73 v9 frame receipt order differs")
                frame_low_water = receipts[0]
                frame_high_water = receipts[-1]
                if frame_low_water < (
                    prior_frame_high_water
                    - IMPACT_CAPTURE_V9_MAX_CROSS_FRAME_REORDER_LAG_NS
                ):
                    raise ValueError("Round 73 v9 cross-frame reorder lag exceeded")
                prior_frame_high_water = max(
                    prior_frame_high_water,
                    frame_high_water,
                )
                running_high_water = max(running_high_water, frame_high_water)
                for item in decoded:
                    heapq.heappush(
                        heap,
                        (
                            item.record.received_monotonic_ns,
                            frame_index,
                            item.message_index,
                            item.record,
                        ),
                    )
                watermark = (
                    running_high_water
                    - IMPACT_CAPTURE_V9_MAX_CROSS_FRAME_REORDER_LAG_NS
                )
                while heap and heap[0][0] <= watermark:
                    _receipt, source_frame, message_index, record = heapq.heappop(heap)
                    yield source_frame, message_index, record
        while heap:
            _receipt, source_frame, message_index, record = heapq.heappop(heap)
            yield source_frame, message_index, record
    finally:
        cursor.close()


def load_impact_capture_v9_preflight(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> ImpactCaptureV9Preflight:
    """Load immutable snapshots and the persisted all-symbol ready boundary."""

    selected = _validate_run_id(run_id)
    segments = connection.execute(
        """
        SELECT symbol, started_wall_ns, cooldown_until_wall_ns, snapshot_update_id
        FROM impact_capture_segment WHERE run_id = ? ORDER BY symbol
        """,
        [selected],
    ).fetchall()
    if tuple(str(row[0]) for row in segments) != IMPACT_CAPTURE_SYMBOLS:
        raise ValueError("Round 73 v9 preflight symbol segments are incomplete")
    ready_markers = tuple(
        int(row[2]) - IMPACT_CAPTURE_INITIAL_COOLDOWN_NS for row in segments
    )
    if len(set(ready_markers)) != 1 or any(
        ready <= int(row[1]) for ready, row in zip(ready_markers, segments, strict=True)
    ):
        raise ValueError("Round 73 v9 feature-ready marker is invalid")
    ready_wall_ns = ready_markers[0]
    expected_update_ids = {str(row[0]): int(row[3]) for row in segments}
    context_rows = connection.execute(
        f"""
        SELECT frame_index, message_index, symbol
        FROM {IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE}
        WHERE run_id = ? AND event_type = 'depthSnapshot'
        ORDER BY symbol
        """,
        [selected],
    ).fetchall()
    if (
        tuple(str(row[2]) for row in context_rows) != IMPACT_CAPTURE_SYMBOLS
        or len({(int(row[0]), int(row[1])) for row in context_rows})
        != len(IMPACT_CAPTURE_SYMBOLS)
    ):
        raise ValueError("Round 73 v9 snapshot context is incomplete")
    expected_keys = {
        (int(frame_index), int(message_index)): str(symbol)
        for frame_index, message_index, symbol in context_rows
    }
    snapshots: dict[str, ImpactCaptureFrameRecord] = {}
    for frame_index, message_index, record in iter_impact_capture_v9_records(
        connection,
        run_id=selected,
    ):
        symbol = expected_keys.get((frame_index, message_index))
        if symbol is None:
            continue
        if record.stream != "binance_futures_rest":
            raise ValueError("Round 73 v9 snapshot stream differs")
        snapshot = _strict_json_object(record.raw_text)
        if int(snapshot.get("lastUpdateId", -1)) != expected_update_ids[symbol]:
            raise ValueError("Round 73 v9 snapshot update ID differs")
        if record.received_wall_ns > ready_wall_ns:
            raise ValueError("Round 73 v9 snapshot follows feature-ready marker")
        snapshots[symbol] = record
        if len(snapshots) == len(IMPACT_CAPTURE_SYMBOLS):
            break
    if tuple(sorted(snapshots)) != IMPACT_CAPTURE_SYMBOLS:
        raise ValueError("Round 73 v9 exact snapshot records are incomplete")
    return ImpactCaptureV9Preflight(
        ready_wall_ns=ready_wall_ns,
        snapshot_records=tuple(
            (symbol, snapshots[symbol]) for symbol in IMPACT_CAPTURE_SYMBOLS
        ),
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
        self.path = str(path)
        self.memory_limit, self.threads = validate_impact_store_resources(
            memory_limit,
            threads,
        )
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
                self._connection.execute(
                    f"SET checkpoint_threshold='{IMPACT_CAPTURE_CHECKPOINT_THRESHOLD}'"
                )
                self._connection.execute(
                    "SET auto_checkpoint_skip_wal_threshold=?",
                    [IMPACT_CAPTURE_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES],
                )
                self._init_schema()
        return self._connection

    def _current_storage_policy(self) -> dict[str, object]:
        connection = self.connect()
        checkpoint_threshold, skip_wal_threshold = connection.execute(
            """
            SELECT current_setting('checkpoint_threshold'),
                   current_setting('auto_checkpoint_skip_wal_threshold')
            """
        ).fetchone()
        return {
            "checkpoint_threshold": str(checkpoint_threshold),
            "auto_checkpoint_skip_wal_threshold_bytes": int(skip_wal_threshold),
        }

    def _apply_storage_policy(self, schema_version: object) -> dict[str, object]:
        requested_checkpoint, requested_skip_wal = _capture_storage_policy_request(
            schema_version
        )
        requested_checkpoint_bytes = _duckdb_byte_setting_bytes(requested_checkpoint)
        observed = self._current_storage_policy()
        try:
            policy_matches = (
                _duckdb_byte_setting_bytes(observed["checkpoint_threshold"])
                == requested_checkpoint_bytes
                and observed["auto_checkpoint_skip_wal_threshold_bytes"]
                == requested_skip_wal
            )
        except ValueError:
            policy_matches = False
        if not policy_matches:
            connection = self.connect()
            connection.execute(f"SET checkpoint_threshold='{requested_checkpoint}'")
            connection.execute(
                "SET auto_checkpoint_skip_wal_threshold=?",
                [requested_skip_wal],
            )
            observed = self._current_storage_policy()
        if (
            _duckdb_byte_setting_bytes(observed["checkpoint_threshold"])
            != requested_checkpoint_bytes
            or observed["auto_checkpoint_skip_wal_threshold_bytes"]
            != requested_skip_wal
        ):
            raise RuntimeError("DuckDB capture storage policy did not apply")
        return observed

    def bind_run_storage_policy(
        self,
        run_id: str,
        *,
        expected_schema_version: str | None = None,
    ) -> dict[str, object]:
        """Bind this connection to the persisted policy of one active run."""

        selected = _validate_run_id(run_id)
        connection = self.connect()
        row = connection.execute(
            """
            SELECT schema_version, status, config_json, config_sha256
            FROM impact_capture_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if row is None or str(row[1]) != "running":
            raise ValueError("capture run is missing or not running")
        schema_version = str(row[0])
        _capture_storage_policy_request(schema_version)
        if expected_schema_version is not None and schema_version != str(
            expected_schema_version
        ):
            raise ValueError("writer schema differs from persisted capture run")
        config_json = str(row[2])
        config_sha256 = _validate_sha256(row[3], "capture config SHA-256")
        try:
            observed_config_sha256 = hashlib.sha256(
                config_json.encode("ascii")
            ).hexdigest()
        except UnicodeEncodeError as exc:
            raise ValueError("persisted capture run config must be ASCII") from exc
        if observed_config_sha256 != config_sha256:
            raise ValueError("persisted capture run config hash mismatch")
        try:
            config = _strict_json_object(config_json)
        except ValueError as exc:
            raise ValueError("persisted capture run config is invalid") from exc
        applied = self._apply_storage_policy(schema_version)
        if (
            not isinstance(config.get("checkpoint_threshold"), str)
            or config.get("checkpoint_threshold") != applied["checkpoint_threshold"]
        ):
            raise ValueError(
                "persisted checkpoint threshold differs from applied writer policy"
            )
        configured_skip_wal = config.get("auto_checkpoint_skip_wal_threshold_bytes")
        if (
            isinstance(configured_skip_wal, bool)
            or not isinstance(configured_skip_wal, int)
            or configured_skip_wal
            != applied["auto_checkpoint_skip_wal_threshold_bytes"]
        ):
            raise ValueError(
                "persisted WAL threshold differs from applied writer policy"
            )
        return {"schema_version": schema_version, **applied}

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

            CREATE TABLE IF NOT EXISTS impact_rejected_wire_event (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                observed_stream_name VARCHAR NOT NULL,
                observed_event_type VARCHAR NOT NULL,
                observed_symbol VARCHAR NOT NULL,
                rejection_class VARCHAR NOT NULL,
                rejection_reason VARCHAR NOT NULL,
                receive_time_ns UBIGINT NOT NULL,
                PRIMARY KEY (run_id, frame_index, message_index)
            );

            CREATE TABLE IF NOT EXISTS impact_event_link_v3 (
                run_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                message_index USMALLINT NOT NULL,
                segment_id VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                sequence_number UBIGINT NOT NULL,
                received_wall_ns UBIGINT NOT NULL,
                received_monotonic_ns UBIGINT NOT NULL,
                raw_payload_sha256 BLOB NOT NULL
                    CHECK (octet_length(raw_payload_sha256) = 32),
                event_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                typed_event_sha256 BLOB NOT NULL
                    CHECK (octet_length(typed_event_sha256) = 32)
            );

            CREATE TABLE IF NOT EXISTS impact_event_link_v4 (
                run_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                message_index USMALLINT NOT NULL,
                segment_id VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                sequence_number UBIGINT NOT NULL,
                received_wall_ns UBIGINT NOT NULL,
                received_monotonic_ns UBIGINT NOT NULL,
                raw_payload_sha256 BLOB NOT NULL
                    CHECK (octet_length(raw_payload_sha256) = 32),
                event_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                event_time_ms BIGINT,
                typed_event_sha256 BLOB NOT NULL
                    CHECK (octet_length(typed_event_sha256) = 32)
            );

            CREATE TABLE IF NOT EXISTS impact_event_link_v5 (
                run_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                message_index USMALLINT NOT NULL,
                segment_id VARCHAR NOT NULL,
                stream VARCHAR NOT NULL,
                connection_id VARCHAR NOT NULL,
                sequence_number UBIGINT NOT NULL,
                received_wall_ns UBIGINT NOT NULL,
                received_monotonic_ns UBIGINT NOT NULL,
                raw_payload_sha256 BLOB NOT NULL
                    CHECK (octet_length(raw_payload_sha256) = 32),
                event_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                event_time_ms BIGINT,
                typed_event_sha256 BLOB NOT NULL
                    CHECK (octet_length(typed_event_sha256) = 32)
            );

            CREATE TABLE IF NOT EXISTS impact_depth_update_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, first_update_id UBIGINT NOT NULL,
                final_update_id UBIGINT NOT NULL, previous_update_id UBIGINT NOT NULL,
                stale BOOLEAN NOT NULL, best_bid DOUBLE NOT NULL, best_ask DOUBLE NOT NULL,
                bid_added_qty DOUBLE NOT NULL, bid_removed_qty DOUBLE NOT NULL,
                ask_added_qty DOUBLE NOT NULL, ask_removed_qty DOUBLE NOT NULL,
                bid_added_quote DOUBLE NOT NULL, bid_removed_quote DOUBLE NOT NULL,
                ask_added_quote DOUBLE NOT NULL, ask_removed_quote DOUBLE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_l2_state_v3 (
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
                imbalance_20 DOUBLE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_book_ticker_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                transaction_time_ms BIGINT NOT NULL, update_id UBIGINT NOT NULL,
                bid DOUBLE NOT NULL, bid_qty DOUBLE NOT NULL,
                ask DOUBLE NOT NULL, ask_qty DOUBLE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_aggregate_trade_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                transaction_time_ms BIGINT NOT NULL,
                aggregate_trade_id UBIGINT NOT NULL,
                first_trade_id UBIGINT NOT NULL, last_trade_id UBIGINT NOT NULL,
                price DOUBLE NOT NULL, qty DOUBLE NOT NULL,
                normalized_qty DOUBLE NOT NULL, buyer_is_maker BOOLEAN NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_mark_price_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                mark_price DOUBLE NOT NULL, index_price DOUBLE NOT NULL,
                estimated_settlement_price DOUBLE, funding_rate DOUBLE NOT NULL,
                next_funding_time_ms BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_liquidation_snapshot_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL, event_time_ms BIGINT NOT NULL,
                order_time_ms BIGINT NOT NULL, side VARCHAR NOT NULL,
                order_type VARCHAR NOT NULL, time_in_force VARCHAR NOT NULL,
                original_qty DOUBLE NOT NULL, price DOUBLE NOT NULL,
                average_price DOUBLE NOT NULL, order_status VARCHAR NOT NULL,
                last_filled_qty DOUBLE NOT NULL,
                accumulated_filled_qty DOUBLE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_rest_event_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL, symbol VARCHAR NOT NULL,
                request_path VARCHAR NOT NULL,
                request_parameters_json VARCHAR NOT NULL,
                response_status USMALLINT NOT NULL,
                request_started_wall_ns UBIGINT NOT NULL,
                request_started_monotonic_ns UBIGINT NOT NULL,
                used_weight_1m UINTEGER, exchange_time_ms BIGINT,
                update_id UBIGINT, open_interest DOUBLE
            );

            CREATE TABLE IF NOT EXISTS impact_rejected_wire_event_v3 (
                run_id VARCHAR NOT NULL, frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL, segment_id VARCHAR NOT NULL,
                observed_stream_name VARCHAR NOT NULL,
                observed_event_type VARCHAR NOT NULL,
                observed_symbol VARCHAR NOT NULL,
                rejection_class VARCHAR NOT NULL,
                rejection_reason VARCHAR NOT NULL,
                receive_time_ns UBIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS impact_depth_band_flow_v5 (
                run_id VARCHAR NOT NULL,
                frame_index UINTEGER NOT NULL,
                message_index UINTEGER NOT NULL,
                segment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                bid_added_quote_levels_1_5 DOUBLE NOT NULL,
                bid_removed_quote_levels_1_5 DOUBLE NOT NULL,
                bid_change_count_levels_1_5 UINTEGER NOT NULL,
                bid_added_quote_levels_6_10 DOUBLE NOT NULL,
                bid_removed_quote_levels_6_10 DOUBLE NOT NULL,
                bid_change_count_levels_6_10 UINTEGER NOT NULL,
                bid_added_quote_levels_11_20 DOUBLE NOT NULL,
                bid_removed_quote_levels_11_20 DOUBLE NOT NULL,
                bid_change_count_levels_11_20 UINTEGER NOT NULL,
                bid_added_quote_outside_20 DOUBLE NOT NULL,
                bid_removed_quote_outside_20 DOUBLE NOT NULL,
                bid_change_count_outside_20 UINTEGER NOT NULL,
                ask_added_quote_levels_1_5 DOUBLE NOT NULL,
                ask_removed_quote_levels_1_5 DOUBLE NOT NULL,
                ask_change_count_levels_1_5 UINTEGER NOT NULL,
                ask_added_quote_levels_6_10 DOUBLE NOT NULL,
                ask_removed_quote_levels_6_10 DOUBLE NOT NULL,
                ask_change_count_levels_6_10 UINTEGER NOT NULL,
                ask_added_quote_levels_11_20 DOUBLE NOT NULL,
                ask_removed_quote_levels_11_20 DOUBLE NOT NULL,
                ask_change_count_levels_11_20 UINTEGER NOT NULL,
                ask_added_quote_outside_20 DOUBLE NOT NULL,
                ask_removed_quote_outside_20 DOUBLE NOT NULL,
                ask_change_count_outside_20 UINTEGER NOT NULL
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
        _ensure_isolated_table(
            self.connect(),
            source_table="impact_capture_frame",
            target_table=IMPACT_CAPTURE_FRAME_TABLE,
            primary_key=("run_id", "frame_index"),
        )
        _ensure_isolated_table(
            self.connect(),
            source_table="impact_event_link_v5",
            target_table=IMPACT_EVENT_LINK_TABLE,
            digest_length_check_columns=(
                "raw_payload_sha256",
                "typed_event_sha256",
            ),
        )
        for source_table, target_table in (
            ("impact_depth_update_v3", IMPACT_DEPTH_UPDATE_TABLE),
            ("impact_l2_state_v3", IMPACT_L2_STATE_TABLE),
            ("impact_book_ticker_v3", IMPACT_BOOK_TICKER_TABLE),
            ("impact_aggregate_trade_v3", IMPACT_AGGREGATE_TRADE_TABLE),
            ("impact_mark_price_v3", IMPACT_MARK_PRICE_TABLE),
            (
                "impact_liquidation_snapshot_v3",
                IMPACT_LIQUIDATION_SNAPSHOT_TABLE,
            ),
            ("impact_rest_event_v3", IMPACT_REST_EVENT_TABLE),
            ("impact_rejected_wire_event_v3", IMPACT_REJECTED_WIRE_EVENT_TABLE),
            ("impact_depth_band_flow_v5", IMPACT_DEPTH_BAND_FLOW_TABLE),
        ):
            _ensure_isolated_table(
                self.connect(),
                source_table=source_table,
                target_table=target_table,
            )
        _ensure_isolated_table(
            self.connect(),
            source_table="impact_capture_frame",
            target_table=IMPACT_CAPTURE_V9_FRAME_TABLE,
            primary_key=("run_id", "frame_index"),
        )
        _ensure_isolated_table(
            self.connect(),
            source_table="impact_rest_event_v3",
            target_table=IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE,
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
        schema_version: str = IMPACT_CAPTURE_SCHEMA_VERSION,
    ) -> None:
        selected = _validate_run_id(run_id)
        normalized = tuple(normalize_symbol(value, default="") for value in symbols)
        if normalized != IMPACT_CAPTURE_SYMBOLS:
            raise ValueError("Round 73 capture requires BTCUSDT, ETHUSDT, and SOLUSDT")
        _reject_secret_fields(config)
        cap = _positive_integer(compressed_payload_cap_bytes, "payload cap")
        selected_schema = str(schema_version)
        contracts = {
            IMPACT_CAPTURE_SCHEMA_VERSION: IMPACT_CAPTURE_CONTRACT_SHA256,
            IMPACT_CAPTURE_V9_SCHEMA_VERSION: IMPACT_CAPTURE_V9_CONTRACT_SHA256,
        }
        try:
            selected_contract = contracts[selected_schema]
        except KeyError as exc:
            raise ValueError("capture run schema version is unsupported") from exc
        connection = self.connect()
        applied_policy = self._apply_storage_policy(selected_schema)
        config_payload = dict(config)
        for key, observed in applied_policy.items():
            if key in config_payload and config_payload[key] != observed:
                raise ValueError(f"capture config {key} differs from applied policy")
            config_payload[key] = observed
        config_json = _canonical_json(config_payload)
        connection.execute(
            """
            INSERT INTO impact_capture_run VALUES (
                ?, ?, ?, ?, 'running', ?, ?, NULL, ?, ?, ?, ?, 0, false,
                0, 0, 0, '', ''
            )
            """,
            [
                selected,
                selected_schema,
                ROUND73_DESIGN_SHA256,
                selected_contract,
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
        run = (
            self.connect()
            .execute(
                "SELECT status, schema_version, capture_contract_sha256 "
                "FROM impact_capture_run WHERE run_id = ?",
                [selected],
            )
            .fetchone()
        )
        if run is None or str(run[0]) == "running":
            raise ValueError("capture report requires a terminal run")
        report_schemas = {
            IMPACT_CAPTURE_SCHEMA_VERSION: IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
            IMPACT_CAPTURE_V9_SCHEMA_VERSION: IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
        }
        try:
            expected_report_schema = report_schemas[str(run[1])]
        except KeyError as exc:
            raise ValueError("capture report run schema is unsupported") from exc
        if str(report.get("schema_version", "")) != expected_report_schema:
            raise ValueError("capture report schema version is invalid")
        report_json = _canonical_json(report)
        report_sha256 = hashlib.sha256(report_json.encode("ascii")).hexdigest()
        self.connect().execute(
            """
            INSERT INTO impact_capture_report VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                selected,
                expected_report_schema,
                str(run[2]),
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
        if isinstance(event, ImpactRejectedWireEvent):
            observed_symbol = normalize_symbol(event.observed_symbol, default="")
            indexed_symbol = (
                observed_symbol if observed_symbol in IMPACT_CAPTURE_SYMBOLS else ""
            )
            return "rejectedWire", indexed_symbol, None, None, None
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
    def _validate_rejected_wire_event(
        message: ImpactCaptureMessage,
        event: ImpactRejectedWireEvent,
    ) -> None:
        if message.record.stream not in {
            "binance_futures_public",
            "binance_futures_market",
        }:
            raise ValueError(
                "rejected wire evidence must originate from WebSocket data"
            )
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", event.rejection_class):
            raise ValueError("rejected wire class is invalid")
        if not event.rejection_reason.strip() or len(event.rejection_reason) > 2_000:
            raise ValueError(
                "rejected wire reason must contain at most 2000 characters"
            )
        for label, value, limit in (
            ("stream name", event.observed_stream_name, 256),
            ("event type", event.observed_event_type, 128),
            ("symbol", event.observed_symbol, 128),
        ):
            if len(value) > limit:
                raise ValueError(f"rejected wire {label} exceeds its bound")
        if int(event.receive_time_ns) != message.record.received_monotonic_ns:
            raise ValueError("rejected wire receipt time differs from frame metadata")

        try:
            root = _strict_json_object(message.record.raw_text)
        except ValueError:
            if any(
                (
                    event.observed_stream_name,
                    event.observed_event_type,
                    event.observed_symbol,
                )
            ):
                raise ValueError(
                    "unparseable rejected wire evidence cannot claim an observed identity"
                ) from None
            return
        stream_name = root.get("stream")
        payload = root.get("data")
        observed_stream = stream_name if isinstance(stream_name, str) else ""
        observed_event = ""
        observed_symbol = ""
        if isinstance(payload, Mapping):
            observed_event = str(payload.get("e", ""))
            symbol_value = payload.get("s")
            if observed_event == "forceOrder" and isinstance(payload.get("o"), Mapping):
                symbol_value = payload["o"].get("s")
            observed_symbol = normalize_symbol(symbol_value, default="")
        if (
            event.observed_stream_name,
            event.observed_event_type,
            event.observed_symbol,
        ) != (observed_stream, observed_event, observed_symbol):
            raise ValueError("rejected wire identity differs from exact evidence")

    @staticmethod
    def _validate_raw_semantics(message: ImpactCaptureMessage) -> None:
        event = message.event
        if isinstance(event, ImpactRejectedWireEvent):
            ImpactAbsorptionStore._validate_rejected_wire_event(message, event)
            return
        root = _strict_json_object(message.record.raw_text)
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
        event_type, symbol, _event_time, _transaction_time, _update_id = (
            ImpactAbsorptionStore._event_identity(event)
        )
        validate_combined_stream_name(
            stream_name,
            event_type=event_type,
            symbol=symbol,
        )
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
        self.bind_run_storage_policy(selected)
        connection = self.connect()
        run = connection.execute(
            """
            SELECT schema_version, status, compressed_payload_cap_bytes,
                   compressed_payload_bytes, payload_cap_reached, frame_count,
                   last_frame_sha256
            FROM impact_capture_run WHERE run_id = ?
            """,
            [selected],
        ).fetchone()
        if run is None or str(run[1]) != "running":
            raise ValueError("capture run is missing or not running")
        run_schema_version = str(run[0])
        if run_schema_version not in {
            IMPACT_CAPTURE_SCHEMA_VERSION,
            IMPACT_CAPTURE_V9_SCHEMA_VERSION,
        }:
            raise ValueError("legacy capture runs cannot accept current frame appends")
        compact_exact_frame = run_schema_version == IMPACT_CAPTURE_V9_SCHEMA_VERSION
        capture_contract_sha256 = (
            IMPACT_CAPTURE_V9_CONTRACT_SHA256
            if compact_exact_frame
            else IMPACT_CAPTURE_CONTRACT_SHA256
        )
        frame_table = (
            IMPACT_CAPTURE_V9_FRAME_TABLE
            if compact_exact_frame
            else IMPACT_CAPTURE_FRAME_TABLE
        )
        if bool(run[4]):
            raise ValueError("capture compressed-payload cap has already been reached")
        frame_index = int(run[5])
        previous_frame_sha256 = str(run[6])
        if compact_exact_frame:
            messages = tuple(
                sorted(
                    messages,
                    key=lambda message: (
                        message.record.received_monotonic_ns,
                        message.record.stream,
                        message.record.connection_id,
                        message.record.sequence_number,
                        message.record.received_wall_ns,
                    ),
                )
            )
            if frame_index:
                prior_high_water = connection.execute(
                    f"SELECT max(last_received_monotonic_ns) FROM "
                    f"{IMPACT_CAPTURE_V9_FRAME_TABLE} "
                    "WHERE run_id = ?",
                    [selected],
                ).fetchone()
                if prior_high_water is None or prior_high_water[0] is None:
                    raise RuntimeError("v9 prior frame is missing")
                current_low_water = min(
                    message.record.received_monotonic_ns for message in messages
                )
                if current_low_water < (
                    int(prior_high_water[0])
                    - IMPACT_CAPTURE_V9_MAX_CROSS_FRAME_REORDER_LAG_NS
                ):
                    raise ValueError("v9 cross-frame receipt reorder lag exceeded")
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
            if isinstance(message.event, ImpactRejectedWireEvent):
                if message.record.stream not in {
                    "binance_futures_public",
                    "binance_futures_market",
                }:
                    raise ValueError(
                        "rejected wire evidence does not belong to a WebSocket lane"
                    )
            elif message.record.stream != self._expected_stream(event_type):
                raise ValueError("typed event does not match the exact-wire stream")
            segment_id = str(message.segment_id).strip().lower()
            if isinstance(message.event, ImpactRejectedWireEvent):
                if segment_id:
                    segment_id = _validate_run_id(segment_id, "segment ID")
                    if active_segments.get(segment_id) != symbol:
                        raise ValueError(
                            "rejected wire event does not belong to its claimed segment"
                        )
            elif event_type in {"serverTime", "exchangeInfo"}:
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
                if message.pre_l2_state is None:
                    raise ValueError("depth evidence requires a pre-event L2 state")
                if (
                    message.pre_l2_state.symbol != symbol
                    or len(message.pre_l2_state.bid_levels) != 20
                    or len(message.pre_l2_state.ask_levels) != 20
                ):
                    raise ValueError("pre-event L2 state does not match depth evidence")
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
            elif message.pre_l2_state is not None or message.l2_state is not None:
                raise ValueError("only depth evidence may carry L2 states")
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
            level=(
                IMPACT_CAPTURE_V9_COMPRESSION_LEVEL
                if compact_exact_frame
                else IMPACT_CAPTURE_COMPRESSION_LEVEL
            ),
            write_checksum=True,
            write_content_size=True,
            threads=0,
        ).compress(uncompressed)
        compressed_sha256 = hashlib.sha256(compressed).hexdigest()
        message_ids: list[str] = []
        event_rows: list[tuple[object, ...]] = []
        depth_rows: list[tuple[object, ...]] = []
        depth_band_rows: list[tuple[object, ...]] = []
        l2_rows: list[tuple[object, ...]] = []
        ticker_rows: list[tuple[object, ...]] = []
        trade_rows: list[tuple[object, ...]] = []
        mark_rows: list[tuple[object, ...]] = []
        liquidation_rows: list[tuple[object, ...]] = []
        rest_rows: list[tuple[object, ...]] = []
        rejected_wire_rows: list[tuple[object, ...]] = []
        stream_counts: dict[str, int] = {}

        for index, (message, item, identity) in enumerate(
            zip(messages, located, identities, strict=True)
        ):
            event_type, symbol, event_time, _transaction_time, _update_id = identity
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
                segment_id,
                message.record.stream,
                message.record.connection_id,
                message.record.sequence_number,
                message.record.received_wall_ns,
                message.record.received_monotonic_ns,
                bytes.fromhex(raw_sha256),
                event_type,
                symbol,
                event_time,
            )
            event = message.event
            key = (selected, frame_index, index, segment_id)
            typed_row_for_hash: tuple[object, ...] | None = None
            l2_row_for_hash: tuple[object, ...] | None = None
            depth_band_row_for_hash: tuple[object, ...] | None = None
            if isinstance(event, ImpactRejectedWireEvent):
                rejected_wire_rows.append(
                    key
                    + (
                        event.observed_stream_name,
                        event.observed_event_type,
                        event.observed_symbol,
                        event.rejection_class,
                        event.rejection_reason,
                        event.receive_time_ns,
                    )
                )
                typed_row_for_hash = rejected_wire_rows[-1]
            elif isinstance(event, DepthUpdate):
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
                if message.pre_l2_state is None:
                    raise RuntimeError("pre-event L2 state was not retained")
                band_flow = {
                    side: {
                        band: {
                            "added_quote": 0.0,
                            "removed_quote": 0.0,
                            "change_count": 0,
                        }
                        for band in ROUND73_LEVEL_BANDS
                    }
                    for side in ("bid", "ask")
                }
                for change in event.changes:
                    band = pre_event_level_band(message.pre_l2_state, change)
                    bucket = band_flow[change.side][band]
                    bucket["added_quote"] += change.added_quote
                    bucket["removed_quote"] += change.removed_quote
                    bucket["change_count"] += 1
                depth_band_rows.append(
                    key
                    + (event.symbol,)
                    + tuple(
                        value
                        for side in ("bid", "ask")
                        for band in ROUND73_LEVEL_BANDS
                        for value in (
                            band_flow[side][band]["added_quote"],
                            band_flow[side][band]["removed_quote"],
                            band_flow[side][band]["change_count"],
                        )
                    )
                )
                depth_band_row_for_hash = depth_band_rows[-1]
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
            typed_payload = {
                "event_type": event_type,
                "typed_row": typed_row_for_hash,
                "l2_row": l2_row_for_hash,
            }
            if event_type == "depthUpdate":
                typed_payload["depth_band_row"] = depth_band_row_for_hash
            typed_event_sha256 = _canonical_sha256(typed_payload)
            event_rows.append(event_row_prefix + (bytes.fromhex(typed_event_sha256),))
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
            "schema_version": run_schema_version,
            "capture_contract_sha256": capture_contract_sha256,
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
        total_compressed = int(run[3]) + len(compressed)
        cap_reached = total_compressed >= int(run[2])

        lane_updates = [
            (selected, stream, connection_id, sequence, monotonic)
            for (stream, connection_id), (sequence, monotonic) in sorted(lanes.items())
        ]
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                f"""
                INSERT INTO {frame_table} VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    selected,
                    frame_index,
                    run_schema_version,
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
            persistence_batches = (
                ((IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE, _REST_COLUMNS, rest_rows),)
                if compact_exact_frame
                else (
                    (IMPACT_EVENT_LINK_TABLE, _EVENT_LINK_V5_COLUMNS, event_rows),
                    (IMPACT_DEPTH_UPDATE_TABLE, _DEPTH_COLUMNS, depth_rows),
                    (
                        IMPACT_DEPTH_BAND_FLOW_TABLE,
                        _DEPTH_BAND_FLOW_COLUMNS,
                        depth_band_rows,
                    ),
                    (IMPACT_L2_STATE_TABLE, _L2_COLUMNS, l2_rows),
                    (IMPACT_BOOK_TICKER_TABLE, _BOOK_TICKER_COLUMNS, ticker_rows),
                    (IMPACT_AGGREGATE_TRADE_TABLE, _TRADE_COLUMNS, trade_rows),
                    (IMPACT_MARK_PRICE_TABLE, _MARK_COLUMNS, mark_rows),
                    (
                        IMPACT_LIQUIDATION_SNAPSHOT_TABLE,
                        _LIQUIDATION_COLUMNS,
                        liquidation_rows,
                    ),
                    (IMPACT_REST_EVENT_TABLE, _REST_COLUMNS, rest_rows),
                    (
                        IMPACT_REJECTED_WIRE_EVENT_TABLE,
                        _REJECTED_WIRE_COLUMNS,
                        rejected_wire_rows,
                    ),
                )
            )
            for table, columns, rows in persistence_batches:
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
        schema_version: str = IMPACT_CAPTURE_SCHEMA_VERSION,
    ) -> tuple[tuple[object, ...] | None, tuple[object, ...] | None]:
        table_contract = _typed_tables_for_schema(schema_version)
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
            if schema_version == IMPACT_CAPTURE_SCHEMA_VERSION:
                l2_table = IMPACT_L2_STATE_TABLE
            elif schema_version in _COMPACT_CAPTURE_SCHEMAS:
                l2_table = "impact_l2_state_v3"
            else:
                l2_table = "impact_l2_state"
            l2_row = connection.execute(
                f"""
                SELECT {", ".join(_L2_COLUMNS)} FROM {l2_table}
                WHERE run_id = ? AND frame_index = ? AND message_index = ?
                """,
                [run_id, frame_index, message_index],
            ).fetchone()
        return (
            None if typed is None else tuple(typed),
            None if l2_row is None else tuple(l2_row),
        )

    @staticmethod
    def _v9_websocket_event_type(record: ImpactCaptureFrameRecord) -> str:
        """Reparse one v9 WebSocket record or classify it as rejected wire."""

        try:
            root = _strict_json_object(record.raw_text)
            stream_name = root.get("stream")
            payload = root.get("data")
            if not isinstance(stream_name, str) or not isinstance(payload, Mapping):
                raise ValueError("WebSocket wrapper is incomplete")
            event_type = str(payload.get("e", ""))
            symbol_value = payload.get("s")
            if event_type == "forceOrder" and isinstance(payload.get("o"), Mapping):
                symbol_value = payload["o"].get("s")
            symbol = normalize_symbol(symbol_value, default="")
            if symbol not in IMPACT_CAPTURE_SYMBOLS:
                raise ValueError("WebSocket symbol is outside the capture universe")
            if record.stream != ImpactAbsorptionStore._expected_stream(event_type):
                raise ValueError("WebSocket event is on the wrong capture lane")
            validate_combined_stream_name(
                stream_name,
                event_type=event_type,
                symbol=symbol,
            )
            if event_type == "bookTicker":
                parse_book_ticker(
                    payload,
                    symbol=symbol,
                    receive_time_ns=record.received_monotonic_ns,
                )
            elif event_type == "aggTrade":
                parse_aggregate_trade(
                    payload,
                    symbol=symbol,
                    receive_time_ns=record.received_monotonic_ns,
                )
            elif event_type == "markPriceUpdate":
                parse_mark_price(
                    payload,
                    symbol=symbol,
                    receive_time_ns=record.received_monotonic_ns,
                )
            elif event_type == "forceOrder":
                parse_liquidation_snapshot(
                    payload,
                    symbol=symbol,
                    receive_time_ns=record.received_monotonic_ns,
                )
            elif event_type == "depthUpdate":
                for key in ("E", "T", "U", "u", "pu"):
                    _positive_integer(payload.get(key), f"raw depth {key}")
                if _positive_integer(payload.get("st"), "raw depth stream type") != 1:
                    raise ValueError("raw depth stream type differs")
                if normalize_symbol(payload.get("ps"), default="") != symbol:
                    raise ValueError("raw depth parent symbol differs")
                for key in ("b", "a"):
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
                        _finite_float(level[0], "raw depth price", positive=True)
                        quantity = _finite_float(level[1], "raw depth quantity")
                        if quantity < 0.0:
                            raise ValueError("raw depth quantity is negative")
            else:
                raise ValueError("WebSocket event type is unsupported")
            return event_type
        except (TypeError, ValueError):
            return "rejectedWire"

    def _audit_v9_run(
        self,
        *,
        run_id: str,
        run: tuple[object, ...],
    ) -> ImpactCaptureAudit:
        connection = self.connect()
        errors: list[str] = []
        run_contract_sha256 = str(run[2])
        if str(run[1]) != ROUND73_DESIGN_SHA256:
            errors.append("run_design_mismatch")
        if run_contract_sha256 != IMPACT_CAPTURE_V9_CONTRACT_SHA256:
            errors.append("run_capture_contract_mismatch")
        frames = connection.execute(
            f"""
            SELECT frame_index, schema_version, frame_format,
                   previous_frame_sha256, message_count, first_message_id,
                   last_message_id, message_manifest_sha256,
                   first_received_wall_ns, last_received_wall_ns,
                   first_received_monotonic_ns, last_received_monotonic_ns,
                   uncompressed_bytes, uncompressed_sha256,
                   compressed_bytes, compressed_sha256, stream_counts_json,
                   compressed_payload, frame_sha256
            FROM {IMPACT_CAPTURE_V9_FRAME_TABLE}
            WHERE run_id = ? ORDER BY frame_index
            """,
            [run_id],
        ).fetchall()
        rest_source_rows = connection.execute(
            f"""
            SELECT frame_index, message_index, segment_id, event_type, symbol,
                   request_path, request_parameters_json, response_status,
                   request_started_wall_ns, request_started_monotonic_ns,
                   used_weight_1m, exchange_time_ms, update_id, open_interest
            FROM {IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE}
            WHERE run_id = ? ORDER BY frame_index, message_index
            """,
            [run_id],
        ).fetchall()
        rest_rows = {
            (int(row[0]), int(row[1])): tuple(row[2:]) for row in rest_source_rows
        }
        if len(rest_rows) != len(rest_source_rows):
            errors.append("rest_context_duplicate_key")
        segment_symbols = {
            str(segment_id): str(symbol)
            for segment_id, symbol in connection.execute(
                "SELECT segment_id, symbol FROM impact_capture_segment WHERE run_id = ?",
                [run_id],
            ).fetchall()
        }
        decompressor = zstandard.ZstdDecompressor()
        lane_state: dict[tuple[str, str], tuple[int, int]] = {}
        prior_frame_sha256 = ""
        total_messages = 0
        total_compressed = 0
        consumed_rest_keys: set[tuple[int, int]] = set()
        event_counts: dict[str, int] = {}
        prior_receipt_high_water = 0
        for expected_index, row in enumerate(frames):
            frame_index = int(row[0])
            if frame_index != expected_index:
                errors.append(f"frame_index_gap:{expected_index}:{frame_index}")
            if str(row[1]) != IMPACT_CAPTURE_V9_SCHEMA_VERSION:
                errors.append(f"frame_schema_mismatch:{frame_index}")
            if str(row[2]) != IMPACT_CAPTURE_FRAME_FORMAT:
                errors.append(f"frame_format_mismatch:{frame_index}")
            if str(row[3]) != prior_frame_sha256:
                errors.append(f"frame_chain_mismatch:{frame_index}")
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
                    expected_message_count=int(row[4]),
                )
            except ValueError as exc:
                errors.append(f"frame_decode_failed:{frame_index}:{type(exc).__name__}")
                continue
            receipt_order = [
                (
                    item.record.received_monotonic_ns,
                    item.record.stream,
                    item.record.connection_id,
                    item.record.sequence_number,
                    item.record.received_wall_ns,
                )
                for item in decoded
            ]
            if receipt_order != sorted(receipt_order):
                errors.append(f"frame_receipt_order_mismatch:{frame_index}")
            frame_low_water = min(item[0] for item in receipt_order)
            frame_high_water = max(item[0] for item in receipt_order)
            if frame_low_water < (
                prior_receipt_high_water
                - IMPACT_CAPTURE_V9_MAX_CROSS_FRAME_REORDER_LAG_NS
            ):
                errors.append(f"frame_cross_frame_reorder_lag:{frame_index}")
            prior_receipt_high_water = max(
                prior_receipt_high_water,
                frame_high_water,
            )
            message_ids: list[str] = []
            stream_counts: dict[str, int] = {}
            for message_index, item in enumerate(decoded):
                record = item.record
                raw_sha256 = hashlib.sha256(record.raw_text.encode("utf-8")).hexdigest()
                message_ids.append(
                    _canonical_sha256(
                        {
                            "run_id": run_id,
                            "stream": record.stream,
                            "connection_id": record.connection_id,
                            "sequence_number": record.sequence_number,
                            "raw_payload_sha256": raw_sha256,
                        }
                    )
                )
                stream_counts[record.stream] = stream_counts.get(record.stream, 0) + 1
                lane = (record.stream, record.connection_id)
                prior_lane = lane_state.get(lane)
                expected_sequence = 0 if prior_lane is None else prior_lane[0] + 1
                if record.sequence_number != expected_sequence:
                    errors.append(
                        f"lane_sequence_mismatch:{frame_index}:{message_index}"
                    )
                if (
                    prior_lane is not None
                    and record.received_monotonic_ns <= prior_lane[1]
                ):
                    errors.append(
                        f"lane_monotonic_mismatch:{frame_index}:{message_index}"
                    )
                lane_state[lane] = (
                    record.sequence_number,
                    record.received_monotonic_ns,
                )
                key = (frame_index, message_index)
                if record.stream == "binance_futures_rest":
                    context = rest_rows.get(key)
                    if context is None:
                        errors.append(
                            f"rest_context_missing:{frame_index}:{message_index}"
                        )
                        continue
                    consumed_rest_keys.add(key)
                    try:
                        parameters = _strict_json_object(str(context[4]))
                        event = ImpactRestEvent(
                            event_type=str(context[1]),  # type: ignore[arg-type]
                            request_path=str(context[3]),
                            request_parameters=parameters,
                            response_status=int(context[5]),
                            request_started_wall_ns=int(context[6]),
                            request_started_monotonic_ns=int(context[7]),
                            symbol=str(context[2]),
                            used_weight_1m=(
                                None if context[8] is None else int(context[8])
                            ),
                            exchange_time_ms=(
                                None if context[9] is None else int(context[9])
                            ),
                            update_id=(
                                None if context[10] is None else int(context[10])
                            ),
                            open_interest=(
                                None if context[11] is None else float(context[11])
                            ),
                        )
                        segment_id = str(context[0])
                        if event.symbol:
                            if segment_symbols.get(segment_id) != event.symbol:
                                raise ValueError("REST segment context differs")
                        elif segment_id:
                            raise ValueError("global REST context has a segment")
                        self._validate_raw_semantics(
                            ImpactCaptureMessage(
                                record=record,
                                event=event,
                                segment_id=segment_id,
                            )
                        )
                        event_type = event.event_type
                    except (TypeError, ValueError):
                        errors.append(
                            f"rest_context_mismatch:{frame_index}:{message_index}"
                        )
                        continue
                else:
                    event_type = self._v9_websocket_event_type(record)
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            manifest_sha256 = _canonical_sha256(message_ids)
            if message_ids and (
                str(row[5]) != message_ids[0] or str(row[6]) != message_ids[-1]
            ):
                errors.append(f"frame_message_bounds_mismatch:{frame_index}")
            if str(row[7]) != manifest_sha256:
                errors.append(f"frame_message_manifest_mismatch:{frame_index}")
            if int(row[8]) != min(item.record.received_wall_ns for item in decoded):
                errors.append(f"frame_first_wall_mismatch:{frame_index}")
            if int(row[9]) != max(item.record.received_wall_ns for item in decoded):
                errors.append(f"frame_last_wall_mismatch:{frame_index}")
            if int(row[10]) != min(
                item.record.received_monotonic_ns for item in decoded
            ):
                errors.append(f"frame_first_monotonic_mismatch:{frame_index}")
            if int(row[11]) != max(
                item.record.received_monotonic_ns for item in decoded
            ):
                errors.append(f"frame_last_monotonic_mismatch:{frame_index}")
            stream_counts_json = _canonical_json(dict(sorted(stream_counts.items())))
            if str(row[16]) != stream_counts_json:
                errors.append(f"frame_stream_counts_mismatch:{frame_index}")
            frame_identity = {
                "schema_version": IMPACT_CAPTURE_V9_SCHEMA_VERSION,
                "capture_contract_sha256": run_contract_sha256,
                "run_id": run_id,
                "frame_index": frame_index,
                "previous_frame_sha256": str(row[3]),
                "message_count": int(row[4]),
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
            if _canonical_sha256(frame_identity) != str(row[18]):
                errors.append(f"frame_identity_mismatch:{frame_index}")
            prior_frame_sha256 = str(row[18])
            total_messages += int(row[4])
            total_compressed += int(row[14])
        if consumed_rest_keys != set(rest_rows):
            errors.append("rest_context_orphan_rows")
        stored_lanes = {
            (str(stream), str(connection_id)): (int(sequence), int(monotonic))
            for stream, connection_id, sequence, monotonic in connection.execute(
                """
                SELECT stream, connection_id, last_sequence_number,
                       last_received_monotonic_ns
                FROM impact_capture_lane_state WHERE run_id = ?
                """,
                [run_id],
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
        report_row = connection.execute(
            "SELECT schema_version, capture_contract_sha256, report_json, "
            "report_sha256 FROM impact_capture_report WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if report_row is not None:
            report_text = str(report_row[2])
            if (
                str(report_row[0]) != IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION
                or str(report_row[1]) != IMPACT_CAPTURE_V9_CONTRACT_SHA256
                or hashlib.sha256(report_text.encode("ascii")).hexdigest()
                != str(report_row[3])
            ):
                errors.append("stored_report_identity_mismatch")
            else:
                report = _strict_json_object(report_text)
                reported_counts = report.get("event_counts")
                if (
                    isinstance(reported_counts, Mapping)
                    and {str(key): int(value) for key, value in reported_counts.items()}
                    != event_counts
                ):
                    errors.append("stored_report_event_counts_mismatch")
        return ImpactCaptureAudit(
            run_id=run_id,
            passed=not errors,
            errors=tuple(errors),
            frame_count=len(frames),
            message_count=total_messages,
            compressed_payload_bytes=total_compressed,
            last_frame_sha256=prior_frame_sha256,
            capture_contract_sha256=run_contract_sha256,
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
        if str(run[0]) == IMPACT_CAPTURE_V9_SCHEMA_VERSION:
            return self._audit_v9_run(run_id=selected, run=tuple(run))
        errors: list[str] = []
        run_schema_version = str(run[0])
        run_contract_sha256 = str(run[2])
        expected_contract = (
            IMPACT_CAPTURE_CONTRACT_SHA256
            if run_schema_version == IMPACT_CAPTURE_SCHEMA_VERSION
            else _LEGACY_CAPTURE_CONTRACTS.get(run_schema_version)
        )
        if expected_contract is None:
            errors.append("run_schema_mismatch")
        if str(run[1]) != ROUND73_DESIGN_SHA256:
            errors.append("run_design_mismatch")
        if expected_contract is not None and run_contract_sha256 != expected_contract:
            errors.append("run_capture_contract_mismatch")
        frame_table = (
            IMPACT_CAPTURE_FRAME_TABLE
            if run_schema_version == IMPACT_CAPTURE_SCHEMA_VERSION
            else "impact_capture_frame"
        )
        frames = connection.execute(
            f"""
            SELECT frame_index, schema_version, frame_format,
                   previous_frame_sha256, message_count, first_message_id,
                   last_message_id, message_manifest_sha256,
                   first_received_wall_ns, last_received_wall_ns,
                   first_received_monotonic_ns, last_received_monotonic_ns,
                   uncompressed_bytes, uncompressed_sha256,
                   compressed_bytes, compressed_sha256, stream_counts_json,
                   compressed_payload, frame_sha256
            FROM {frame_table} WHERE run_id = ? ORDER BY frame_index
            """,
            [selected],
        ).fetchall()
        compact_schema = run_schema_version in _COMPACT_CAPTURE_SCHEMAS
        event_time_link_schema = run_schema_version in _EVENT_TIME_LINK_SCHEMAS
        depth_band_schema = run_schema_version in _DEPTH_BAND_SCHEMAS
        if expected_contract is None:
            typed_contracts: dict[str, tuple[str, tuple[str, ...]]] = {}
        else:
            typed_contracts = _typed_tables_for_schema(run_schema_version)
        if run_schema_version == IMPACT_CAPTURE_SCHEMA_VERSION:
            event_link_table = IMPACT_EVENT_LINK_TABLE
        elif run_schema_version in _LEGACY_DEPTH_BAND_SCHEMAS:
            event_link_table = "impact_event_link_v5"
        elif run_schema_version == _V4_CAPTURE_SCHEMA_VERSION:
            event_link_table = "impact_event_link_v4"
        elif run_schema_version == _V3_CAPTURE_SCHEMA_VERSION:
            event_link_table = "impact_event_link_v3"
        else:
            event_link_table = "impact_event_index"
        if run_schema_version == IMPACT_CAPTURE_SCHEMA_VERSION:
            l2_table = IMPACT_L2_STATE_TABLE
            depth_band_table = IMPACT_DEPTH_BAND_FLOW_TABLE
        elif compact_schema:
            l2_table = "impact_l2_state_v3"
            depth_band_table = "impact_depth_band_flow_v5"
        else:
            l2_table = "impact_l2_state"
            depth_band_table = "impact_depth_band_flow_v5"
        event_index_by_frame: dict[int, list[_StoredEventLink]] = {}
        typed_rows_by_event: dict[str, dict[tuple[int, int], tuple[object, ...]]] = {}
        l2_rows: dict[tuple[int, int], tuple[object, ...]] = {}
        depth_band_rows: dict[tuple[int, int], tuple[object, ...]] = {}
        prior_frame_sha256 = ""
        total_messages = 0
        total_compressed = 0
        lane_state: dict[tuple[str, str], tuple[int, int]] = {}
        decompressor = zstandard.ZstdDecompressor()
        for expected_index, row in enumerate(frames):
            if expected_index % _AUDIT_FRAME_BATCH_SIZE == 0:
                batch = frames[
                    expected_index : expected_index + _AUDIT_FRAME_BATCH_SIZE
                ]
                first_frame_index = int(batch[0][0])
                last_frame_index = int(batch[-1][0])
                event_index_by_frame = {}
                if event_time_link_schema:
                    index_rows = connection.execute(
                        f"""
                        SELECT frame_index, message_index, segment_id, stream,
                               connection_id, sequence_number, received_wall_ns,
                               received_monotonic_ns, raw_payload_sha256, event_type,
                               symbol, event_time_ms, typed_event_sha256
                        FROM {event_link_table}
                        WHERE run_id = ? AND frame_index BETWEEN ? AND ?
                        ORDER BY frame_index, message_index
                        """,
                        [selected, first_frame_index, last_frame_index],
                    ).fetchall()
                elif compact_schema:
                    index_rows = connection.execute(
                        f"""
                        SELECT frame_index, message_index, segment_id, stream,
                               connection_id, sequence_number, received_wall_ns,
                               received_monotonic_ns, raw_payload_sha256, event_type,
                               symbol, typed_event_sha256
                        FROM {event_link_table}
                        WHERE run_id = ? AND frame_index BETWEEN ? AND ?
                        ORDER BY frame_index, message_index
                        """,
                        [selected, first_frame_index, last_frame_index],
                    ).fetchall()
                else:
                    index_rows = connection.execute(
                        f"""
                        SELECT frame_index, message_index, message_id, segment_id,
                               stream, connection_id, sequence_number,
                               received_wall_ns, received_monotonic_ns,
                               raw_payload_sha256, event_type, symbol,
                               event_time_ms, typed_event_sha256
                        FROM {event_link_table}
                        WHERE run_id = ? AND frame_index BETWEEN ? AND ?
                        ORDER BY frame_index, message_index
                        """,
                        [selected, first_frame_index, last_frame_index],
                    ).fetchall()
                for index_row in index_rows:
                    if compact_schema:
                        raw_digest = bytes(index_row[8])
                        typed_digest_index = 12 if event_time_link_schema else 11
                        typed_digest = bytes(index_row[typed_digest_index])
                        if len(raw_digest) != 32:
                            errors.append(
                                f"raw_digest_size_mismatch:{index_row[0]}:{index_row[1]}"
                            )
                        if len(typed_digest) != 32:
                            errors.append(
                                f"typed_digest_size_mismatch:{index_row[0]}:{index_row[1]}"
                            )
                        link = _StoredEventLink(
                            message_index=int(index_row[1]),
                            message_id=None,
                            segment_id=str(index_row[2]),
                            stream=str(index_row[3]),
                            connection_id=str(index_row[4]),
                            sequence_number=int(index_row[5]),
                            received_wall_ns=int(index_row[6]),
                            received_monotonic_ns=int(index_row[7]),
                            raw_payload_sha256=raw_digest.hex(),
                            event_type=str(index_row[9]),
                            symbol=str(index_row[10]),
                            event_time_ms=(
                                None
                                if not event_time_link_schema or index_row[11] is None
                                else int(index_row[11])
                            ),
                            event_time_stored=event_time_link_schema,
                            typed_event_sha256=typed_digest.hex(),
                        )
                    else:
                        link = _StoredEventLink(
                            message_index=int(index_row[1]),
                            message_id=str(index_row[2]),
                            segment_id=str(index_row[3]),
                            stream=str(index_row[4]),
                            connection_id=str(index_row[5]),
                            sequence_number=int(index_row[6]),
                            received_wall_ns=int(index_row[7]),
                            received_monotonic_ns=int(index_row[8]),
                            raw_payload_sha256=str(index_row[9]),
                            event_type=str(index_row[10]),
                            symbol=str(index_row[11]),
                            event_time_ms=(
                                None if index_row[12] is None else int(index_row[12])
                            ),
                            event_time_stored=True,
                            typed_event_sha256=str(index_row[13]),
                        )
                    event_index_by_frame.setdefault(int(index_row[0]), []).append(link)
                typed_rows_by_event = {}
                for event_type, (table, columns) in typed_contracts.items():
                    parameters: list[object] = [
                        selected,
                        first_frame_index,
                        last_frame_index,
                    ]
                    event_filter = ""
                    if event_type in _REST_ENDPOINTS:
                        event_filter = " AND event_type = ?"
                        parameters.append(event_type)
                    rows = connection.execute(
                        f"SELECT {', '.join(columns)} FROM {table} "
                        "WHERE run_id = ? AND frame_index BETWEEN ? AND ?"
                        f"{event_filter}",
                        parameters,
                    ).fetchall()
                    typed_rows_by_event[event_type] = {
                        (int(typed_row[1]), int(typed_row[2])): tuple(typed_row)
                        for typed_row in rows
                    }
                l2_rows = {
                    (int(l2_row[1]), int(l2_row[2])): tuple(l2_row)
                    for l2_row in connection.execute(
                        f"SELECT {', '.join(_L2_COLUMNS)} FROM {l2_table} "
                        "WHERE run_id = ? AND frame_index BETWEEN ? AND ?",
                        [selected, first_frame_index, last_frame_index],
                    ).fetchall()
                }
                if depth_band_schema:
                    depth_band_rows = {
                        (int(band_row[1]), int(band_row[2])): tuple(band_row)
                        for band_row in connection.execute(
                            f"SELECT {', '.join(_DEPTH_BAND_FLOW_COLUMNS)} "
                            f"FROM {depth_band_table} "
                            "WHERE run_id = ? AND frame_index BETWEEN ? AND ?",
                            [selected, first_frame_index, last_frame_index],
                        ).fetchall()
                    }
                else:
                    depth_band_rows = {}
            frame_index = int(row[0])
            if frame_index != expected_index:
                errors.append(f"frame_index_gap:{expected_index}:{frame_index}")
            if str(row[1]) != run_schema_version:
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
            index_rows = event_index_by_frame.get(frame_index, [])
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
                if (
                    indexed.message_index != message_index
                    or (
                        indexed.message_id is not None
                        and indexed.message_id != message_id
                    )
                    or indexed.stream != record.stream
                    or indexed.connection_id != record.connection_id
                    or indexed.sequence_number != record.sequence_number
                    or indexed.received_wall_ns != record.received_wall_ns
                    or indexed.received_monotonic_ns != record.received_monotonic_ns
                    or indexed.raw_payload_sha256 != raw_sha256
                ):
                    errors.append(
                        f"raw_to_index_mismatch:{frame_index}:{message_index}"
                    )
                indexed_event_type = indexed.event_type
                raw_event_time_ms: int | None = None
                if indexed_event_type != "rejectedWire":
                    try:
                        root = _strict_json_object(record.raw_text)
                        if record.stream == "binance_futures_rest":
                            rest_time_key = {
                                "serverTime": "serverTime",
                                "openInterest": "time",
                            }.get(indexed_event_type)
                            if rest_time_key is not None:
                                raw_event_time_ms = _positive_integer(
                                    root.get(rest_time_key),
                                    "raw REST event time",
                                )
                        else:
                            raw_payload = root.get("data")
                            if (
                                not isinstance(raw_payload, Mapping)
                                or str(raw_payload.get("e", "")) != indexed_event_type
                            ):
                                errors.append(
                                    f"raw_event_type_mismatch:{frame_index}:{message_index}"
                                )
                            elif raw_payload.get("E") is not None:
                                raw_event_time_ms = _positive_integer(
                                    raw_payload.get("E"),
                                    "raw WebSocket event time",
                                )
                    except ValueError:
                        errors.append(f"raw_json_invalid:{frame_index}:{message_index}")
                if (
                    indexed.event_time_stored
                    and indexed.event_time_ms != raw_event_time_ms
                ):
                    errors.append(
                        f"event_time_link_mismatch:{frame_index}:{message_index}"
                    )
                typed_row = typed_rows_by_event.get(indexed_event_type, {}).get(
                    (frame_index, message_index)
                )
                l2_row = (
                    l2_rows.get((frame_index, message_index))
                    if indexed_event_type == "depthUpdate"
                    else None
                )
                depth_band_row = (
                    depth_band_rows.get((frame_index, message_index))
                    if indexed_event_type == "depthUpdate" and depth_band_schema
                    else None
                )
                if typed_row is None:
                    errors.append(f"typed_row_missing:{frame_index}:{message_index}")
                else:
                    expected_segment_id = str(typed_row[3])
                    if indexed_event_type == "rejectedWire":
                        observed_symbol = normalize_symbol(typed_row[6], default="")
                        expected_symbol = (
                            observed_symbol
                            if observed_symbol in IMPACT_CAPTURE_SYMBOLS
                            else ""
                        )
                    elif indexed_event_type in _REST_ENDPOINTS:
                        expected_symbol = str(typed_row[5])
                    else:
                        expected_symbol = str(typed_row[4])
                    if indexed.segment_id != expected_segment_id:
                        errors.append(
                            f"segment_link_mismatch:{frame_index}:{message_index}"
                        )
                    if indexed.symbol != expected_symbol:
                        errors.append(
                            f"symbol_link_mismatch:{frame_index}:{message_index}"
                        )
                    typed_payload = {
                        "event_type": indexed_event_type,
                        "typed_row": typed_row,
                        "l2_row": l2_row,
                    }
                    if depth_band_schema and indexed_event_type == "depthUpdate":
                        typed_payload["depth_band_row"] = depth_band_row
                    typed_sha256 = _canonical_sha256(typed_payload)
                    if typed_sha256 != indexed.typed_event_sha256:
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
                "schema_version": run_schema_version,
                "capture_contract_sha256": run_contract_sha256,
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

        expected_counts = {
            str(event_type): int(count)
            for event_type, count in connection.execute(
                f"""
                SELECT event_type, count(*) FROM {event_link_table}
                WHERE run_id = ? GROUP BY event_type
                """,
                [selected],
            ).fetchall()
        }
        for event_type, (table, _columns) in typed_contracts.items():
            if event_type in _REST_ENDPOINTS:
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
        if depth_band_schema:
            band_count = int(
                connection.execute(
                    f"SELECT count(*) FROM {depth_band_table} WHERE run_id = ?",
                    [selected],
                ).fetchone()[0]
            )
            if band_count != expected_counts.get("depthUpdate", 0):
                errors.append("typed_count_mismatch:depthBandFlow")

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
            capture_contract_sha256=run_contract_sha256,
        )


__all__ = [
    "IMPACT_CAPTURE_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES",
    "IMPACT_CAPTURE_CHECKPOINT_THRESHOLD",
    "IMPACT_CAPTURE_COMPRESSION_LEVEL",
    "IMPACT_CAPTURE_CONTRACT_SHA256",
    "IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES",
    "IMPACT_CAPTURE_FRAME_TABLE",
    "IMPACT_CAPTURE_INITIAL_COOLDOWN_NS",
    "IMPACT_CAPTURE_REPORT_SCHEMA_VERSION",
    "IMPACT_CAPTURE_SCHEMA_VERSION",
    "IMPACT_CAPTURE_SYMBOLS",
    "IMPACT_CAPTURE_V9_AUTO_CHECKPOINT_SKIP_WAL_THRESHOLD_BYTES",
    "IMPACT_CAPTURE_V9_CHECKPOINT_THRESHOLD",
    "IMPACT_CAPTURE_V9_COMPRESSION_LEVEL",
    "IMPACT_CAPTURE_V9_CONTRACT_SHA256",
    "IMPACT_CAPTURE_V9_FRAME_TABLE",
    "IMPACT_CAPTURE_V9_MAX_CROSS_FRAME_REORDER_LAG_NS",
    "IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION",
    "IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE",
    "IMPACT_CAPTURE_V9_SCHEMA_VERSION",
    "IMPACT_AGGREGATE_TRADE_TABLE",
    "IMPACT_BOOK_TICKER_TABLE",
    "IMPACT_DEPTH_BAND_FLOW_TABLE",
    "IMPACT_DEPTH_UPDATE_TABLE",
    "IMPACT_EVENT_LINK_TABLE",
    "IMPACT_L2_STATE_TABLE",
    "IMPACT_LIQUIDATION_SNAPSHOT_TABLE",
    "IMPACT_MARK_PRICE_TABLE",
    "IMPACT_REJECTED_WIRE_EVENT_TABLE",
    "IMPACT_REST_EVENT_TABLE",
    "ImpactAbsorptionStore",
    "ImpactCaptureAudit",
    "ImpactCaptureMessage",
    "ImpactCaptureV9Preflight",
    "ImpactFrameWriteResult",
    "ImpactRejectedWireEvent",
    "ImpactRestEvent",
    "iter_impact_capture_v9_records",
    "load_impact_capture_v9_preflight",
    "validate_impact_store_resources",
]
