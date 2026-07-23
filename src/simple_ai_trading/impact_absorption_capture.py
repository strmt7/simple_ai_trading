"""Bounded prospective Binance USD-M capture for Round 73 research."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading
import time
from typing import Literal, Mapping, Sequence
import uuid

import requests
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .assets import normalize_symbol
from .impact_absorption import (
    ImpactFeedIntegrityError,
    ROUND73_DESIGN_SHA256,
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
    validate_combined_stream_name,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_CONTRACT_SHA256,
    IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES,
    IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    IMPACT_DEPTH_UPDATE_TABLE,
    IMPACT_EVENT_LINK_TABLE,
    IMPACT_REST_EVENT_TABLE,
    ImpactAbsorptionStore,
    ImpactCaptureAudit,
    ImpactCaptureMessage,
    ImpactFrameWriteResult,
    ImpactRejectedWireEvent,
    ImpactRestEvent,
    iter_impact_capture_v9_records,
    validate_impact_store_resources,
)
from .impact_capture_frame import (
    ImpactCaptureFrameRecord,
    impact_capture_frame_record_size,
)


BINANCE_FUTURES_REST = "https://fapi.binance.com"
BINANCE_FUTURES_PUBLIC_STREAM = "wss://fstream.binance.com/public/stream?streams="
BINANCE_FUTURES_MARKET_STREAM = "wss://fstream.binance.com/market/stream?streams="

_PUBLIC_SUFFIXES = ("depth@100ms", "bookTicker")
_MARKET_SUFFIXES = ("aggTrade", "markPrice@1s", "forceOrder")
_FRAME_MESSAGE_LIMIT = 16_384
_FRAME_UNCOMPRESSED_LIMIT = 32 * 1024 * 1024
_FRAME_FLUSH_SECONDS = 4.0
_QUEUE_CAPACITY = 65_536
_WRITER_STALL_SECONDS = 5.0
_SOURCE_STALL_SECONDS = 15.0
_CLOCK_PROBE_COUNT = 3
_RATE_LIMIT_FRACTION = 0.80
_RECONNECT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
IMPACT_CAPTURE_DEFAULT_DATABASE_SIZE_CAP_BYTES = 8 * 1024 * 1024 * 1024
_DATABASE_SIZE_CAP_RESERVE_BYTES = 512 * 1024 * 1024
_STORAGE_EFFICIENCY_MINIMUM_SECONDS = 180.0
_STORAGE_EFFICIENCY_MAXIMUM_FRAMES_PER_MINUTE = 25.0
_CAPTURE_PROTOCOLS = {
    IMPACT_CAPTURE_SCHEMA_VERSION: (
        IMPACT_CAPTURE_CONTRACT_SHA256,
        IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
        4_096.0,
        1_024.0,
    ),
    IMPACT_CAPTURE_V9_SCHEMA_VERSION: (
        IMPACT_CAPTURE_V9_CONTRACT_SHA256,
        IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
        1_024.0,
        512.0,
    ),
}

CaptureFailureClass = Literal[
    "none",
    "transport",
    "rest_transport",
    "feed_integrity",
    "processing",
    "rate_limit",
    "writer",
    "resource_limit",
    "audit",
    "post_capture",
]


def _capture_protocol(schema_version: str) -> tuple[str, str, float, float]:
    try:
        return _CAPTURE_PROTOCOLS[str(schema_version)]
    except KeyError as exc:
        raise ValueError("impact capture schema version is unsupported") from exc


class _ImpactRateLimitGuardError(RuntimeError):
    pass


class _ImpactWriterFault(RuntimeError):
    pass


class _ImpactWriterQueueOverflow(RuntimeError):
    pass


class _ImpactPayloadCapReached(RuntimeError):
    pass


class _ImpactDatabaseSizeCapReached(RuntimeError):
    pass


@dataclass(frozen=True)
class _ProcessIoSnapshot:
    provider: str
    semantics: str
    write_bytes: int | None


@dataclass(frozen=True)
class _ProcessIoInterval:
    provider: str
    semantics: str
    start_write_bytes: int | None
    end_write_bytes: int | None
    delta_write_bytes: int | None


def _process_io_interval(
    start: _ProcessIoSnapshot,
    end: _ProcessIoSnapshot,
) -> _ProcessIoInterval:
    same_provider = start.provider == end.provider
    delta = None
    if (
        same_provider
        and start.write_bytes is not None
        and end.write_bytes is not None
        and end.write_bytes >= start.write_bytes
    ):
        delta = end.write_bytes - start.write_bytes
    return _ProcessIoInterval(
        provider=end.provider if same_provider else "provider_changed",
        semantics=end.semantics if same_provider else "process I/O provider changed",
        start_write_bytes=start.write_bytes,
        end_write_bytes=end.write_bytes,
        delta_write_bytes=delta,
    )


@dataclass(frozen=True)
class _WriterResourceMetrics:
    process_io_provider: str
    process_io_semantics: str
    process_io_start_write_bytes: int | None
    process_io_end_write_bytes: int | None
    process_io_delta_write_bytes: int | None
    process_io_write_bytes_per_message: float | None
    database_physical_start_bytes: int | None
    database_physical_end_bytes: int
    database_physical_growth_bytes: int | None
    database_physical_growth_bytes_per_message: float | None
    frames_per_stream_minute: float
    storage_efficiency_passed: bool


def _process_io_snapshot() -> _ProcessIoSnapshot:
    if os.name == "nt":
        import ctypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("read_operation_count", ctypes.c_ulonglong),
                ("write_operation_count", ctypes.c_ulonglong),
                ("other_operation_count", ctypes.c_ulonglong),
                ("read_transfer_count", ctypes.c_ulonglong),
                ("write_transfer_count", ctypes.c_ulonglong),
                ("other_transfer_count", ctypes.c_ulonglong),
            ]

        counters = IoCounters()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessIoCounters.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(IoCounters),
        ]
        kernel32.GetProcessIoCounters.restype = ctypes.c_int
        if kernel32.GetProcessIoCounters(
            kernel32.GetCurrentProcess(), ctypes.byref(counters)
        ):
            return _ProcessIoSnapshot(
                provider="windows_GetProcessIoCounters_WriteTransferCount",
                semantics="process I/O transfer bytes; not physical SSD writes",
                write_bytes=int(counters.write_transfer_count),
            )
    if sys.platform.startswith("linux"):
        try:
            values = {
                key.strip(): int(value.strip())
                for key, value in (
                    line.split(":", 1)
                    for line in Path("/proc/self/io")
                    .read_text(encoding="ascii")
                    .splitlines()
                    if ":" in line
                )
            }
        except (OSError, ValueError):
            pass
        else:
            if "write_bytes" in values:
                return _ProcessIoSnapshot(
                    provider="linux_proc_self_io_write_bytes",
                    semantics="bytes caused to be sent to storage by this process",
                    write_bytes=values["write_bytes"],
                )
    return _ProcessIoSnapshot(
        provider="unavailable",
        semantics="host does not expose a supported process write counter",
        write_bytes=None,
    )


def _database_physical_bytes(database: str) -> int:
    if str(database).strip() == ":memory:":
        return 0
    path = Path(database)
    total = 0
    for candidate in (path, Path(f"{path}.wal")):
        try:
            total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


class _ImpactRestResponseError(RuntimeError):
    def __init__(self, path: str, status_code: int) -> None:
        self.status_code = int(status_code)
        super().__init__(f"public Binance REST {path} returned HTTP {status_code}")


def _strict_json_object(raw_text: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, Mapping):
        raise ValueError("provider payload must be a JSON object")
    return parsed


def _combined_payload(raw_text: str) -> tuple[str, Mapping[str, object]]:
    wrapper = _strict_json_object(raw_text)
    stream = wrapper.get("stream")
    payload = wrapper.get("data")
    if not isinstance(stream, str) or not stream:
        raise ValueError("combined stream name is missing")
    if not isinstance(payload, Mapping):
        raise ValueError("combined stream data is missing")
    return stream, payload


def _best_effort_wire_identity(raw_text: str) -> tuple[str, str, str]:
    try:
        wrapper = _strict_json_object(raw_text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return "", "", ""
    stream_value = wrapper.get("stream")
    stream_name = stream_value if isinstance(stream_value, str) else ""
    payload = wrapper.get("data")
    if not isinstance(payload, Mapping):
        return stream_name, "", ""
    event_type = str(payload.get("e", ""))
    symbol_value = payload.get("s")
    if event_type == "forceOrder" and isinstance(payload.get("o"), Mapping):
        symbol_value = payload["o"].get("s")
    return stream_name, event_type, normalize_symbol(symbol_value, default="")


def _rejected_wire_message(
    *,
    record: ImpactCaptureFrameRecord,
    error: BaseException,
    segment_ids: Mapping[str, str],
) -> ImpactCaptureMessage:
    stream_name, event_type, symbol = _best_effort_wire_identity(record.raw_text)
    return ImpactCaptureMessage(
        record=record,
        event=ImpactRejectedWireEvent(
            observed_stream_name=stream_name,
            observed_event_type=event_type,
            observed_symbol=symbol,
            rejection_class=_failure_class_for_exception(error),
            rejection_reason=f"{type(error).__name__}:{error}"[:2_000],
            receive_time_ns=record.received_monotonic_ns,
        ),
        segment_id=segment_ids.get(symbol, ""),
    )


@dataclass(frozen=True)
class ImpactCaptureConfig:
    """Resource and authority bounds for one prospective capture."""

    database: str = "data/microstructure.duckdb"
    schema_version: str = IMPACT_CAPTURE_SCHEMA_VERSION
    mode: Literal["probe", "qualification"] = "probe"
    duration_seconds: float = 180.0
    request_timeout_seconds: float = 10.0
    queue_capacity_messages: int = _QUEUE_CAPACITY
    frame_message_limit: int = _FRAME_MESSAGE_LIMIT
    frame_uncompressed_limit_bytes: int = _FRAME_UNCOMPRESSED_LIMIT
    frame_flush_seconds: float = _FRAME_FLUSH_SECONDS
    compressed_payload_cap_bytes: int = IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES
    database_size_cap_bytes: int = IMPACT_CAPTURE_DEFAULT_DATABASE_SIZE_CAP_BYTES
    writer_stall_seconds: float = _WRITER_STALL_SECONDS
    source_stall_seconds: float = _SOURCE_STALL_SECONDS
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 2
    maximum_reconnects: int = len(_RECONNECT_BACKOFF_SECONDS)

    def validate(self) -> None:
        _capture_protocol(self.schema_version)
        if self.mode not in {"probe", "qualification"}:
            raise ValueError("impact capture mode must be probe or qualification")
        duration = float(self.duration_seconds)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("impact capture duration must be finite and positive")
        if self.mode == "probe" and duration > 300.0:
            raise ValueError("schema probes are capped at 300 seconds")
        if self.mode == "qualification" and duration < 3_600.0:
            raise ValueError("qualification capture requires at least 3600 seconds")
        if not 1.0 <= float(self.request_timeout_seconds) <= 60.0:
            raise ValueError("REST timeout must be between 1 and 60 seconds")
        if not 1 <= int(self.queue_capacity_messages) <= _QUEUE_CAPACITY:
            raise ValueError("writer queue capacity is outside the frozen bound")
        if not 1 <= int(self.frame_message_limit) <= _FRAME_MESSAGE_LIMIT:
            raise ValueError("frame message limit is outside the frozen bound")
        if (
            not 1
            <= int(self.frame_uncompressed_limit_bytes)
            <= _FRAME_UNCOMPRESSED_LIMIT
        ):
            raise ValueError("frame byte limit is outside the frozen bound")
        if not 0.01 <= float(self.frame_flush_seconds) <= _FRAME_FLUSH_SECONDS:
            raise ValueError("frame flush interval is outside the frozen bound")
        if int(self.compressed_payload_cap_bytes) < 1:
            raise ValueError("compressed payload cap must be positive")
        if int(self.database_size_cap_bytes) <= _DATABASE_SIZE_CAP_RESERVE_BYTES:
            raise ValueError("database size cap must exceed the 512 MiB safety reserve")
        if not 1.0 <= float(self.writer_stall_seconds) <= _WRITER_STALL_SECONDS:
            raise ValueError("writer stall timeout is outside the frozen bound")
        if not 5.0 <= float(self.source_stall_seconds) <= _SOURCE_STALL_SECONDS:
            raise ValueError("source stall timeout is outside the frozen bound")
        if not 0 <= int(self.maximum_reconnects) <= len(_RECONNECT_BACKOFF_SECONDS):
            raise ValueError("maximum reconnects is outside the frozen bound")
        validate_impact_store_resources(self.duckdb_memory_limit, self.duckdb_threads)


@dataclass(frozen=True)
class _WireReceipt:
    raw_text: str
    sequence_number: int
    received_wall_ns: int
    received_monotonic_ns: int


@dataclass(frozen=True)
class _RestEvidence:
    message: ImpactCaptureMessage
    body: Mapping[str, object]
    used_weight_1m: int | None


@dataclass(frozen=True)
class ImpactCaptureReport:
    run_id: str
    capture_schema_version: str
    mode: str
    status: str
    capture_gate_passed: bool
    qualification_passed: bool
    started_wall_ns: int
    ended_wall_ns: int
    elapsed_seconds: float
    queue_high_water_messages: int
    queue_capacity_messages: int
    queue_maximum_utilization: float
    writer_frame_count: int
    writer_message_count: int
    writer_compressed_payload_bytes: int
    payload_cap_reached: bool
    database_physical_start_bytes: int | None
    database_physical_bytes: int
    database_physical_growth_bytes: int | None
    database_physical_growth_bytes_per_message: float | None
    database_size_cap_bytes: int
    database_size_cap_reached: bool
    process_io_scope: str
    process_io_provider: str
    process_io_semantics: str
    process_io_start_write_bytes: int | None
    process_io_end_write_bytes: int | None
    process_io_delta_write_bytes: int | None
    process_io_write_bytes_per_message: float | None
    frames_per_stream_minute: float
    storage_efficiency_passed: bool
    terminal_process_io_provider: str
    terminal_process_io_semantics: str
    terminal_process_io_start_write_bytes: int | None
    terminal_process_io_end_write_bytes: int | None
    terminal_process_io_delta_write_bytes: int | None
    event_counts: dict[str, int]
    symbol_event_counts: dict[str, dict[str, int]]
    negative_corrected_latency_fraction: float | None
    audit_passed: bool
    audit_errors: tuple[str, ...]
    error: str
    failure_class: CaptureFailureClass

    def as_dict(self) -> dict[str, object]:
        capture_contract, report_schema, _write_limit, _growth_limit = (
            _capture_protocol(self.capture_schema_version)
        )
        result = asdict(self)
        result.pop("capture_schema_version")
        result["schema_version"] = report_schema
        result["design_sha256"] = ROUND73_DESIGN_SHA256
        result["capture_contract_sha256"] = capture_contract
        result["audit_errors"] = list(self.audit_errors)
        return result


@dataclass(frozen=True)
class ImpactCaptureSupervisorReport:
    """Bounded recovery result without combining evidence across attempts."""

    status: Literal["completed", "failed"]
    capture_schema_version: str
    qualification_passed: bool
    selected_run_id: str
    attempt_count: int
    reconnect_count: int
    reconnect_delays_seconds: tuple[float, ...]
    attempts: tuple[ImpactCaptureReport, ...]
    startup_errors: tuple[str, ...]
    terminal_error: str

    def as_dict(self) -> dict[str, object]:
        capture_contract, _report_schema, _write_limit, _growth_limit = (
            _capture_protocol(self.capture_schema_version)
        )
        return {
            "schema_version": "round-073-capture-supervisor-report-v1",
            "design_sha256": ROUND73_DESIGN_SHA256,
            "capture_schema_version": self.capture_schema_version,
            "capture_contract_sha256": capture_contract,
            "status": self.status,
            "qualification_passed": self.qualification_passed,
            "selected_run_id": self.selected_run_id,
            "attempt_count": self.attempt_count,
            "reconnect_count": self.reconnect_count,
            "reconnect_delays_seconds": list(self.reconnect_delays_seconds),
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "startup_errors": list(self.startup_errors),
            "terminal_error": self.terminal_error,
            "attempt_evidence_combined": False,
        }


class _ImpactFrameWriter:
    """Daemonized one-writer boundary that cannot block the asyncio control loop."""

    _STOP = object()

    def __init__(self, config: ImpactCaptureConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.queue: Queue[ImpactCaptureMessage | object] = Queue(
            maxsize=int(config.queue_capacity_messages)
        )
        self.thread = threading.Thread(
            target=self._run,
            name=f"impact-writer-{run_id[:8]}",
            daemon=True,
        )
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.failed = threading.Event()
        self.cap_reached = threading.Event()
        self.database_cap_reached = threading.Event()
        self.error = ""
        self.last_progress_monotonic = time.monotonic()
        self.high_water_messages = 0
        self.frame_count = 0
        self.message_count = 0
        self.compressed_payload_bytes = 0
        self.database_physical_start_bytes = _database_physical_bytes(config.database)
        self.database_physical_bytes = self.database_physical_start_bytes
        self.process_io_start = _process_io_snapshot()
        self.process_io_end: _ProcessIoSnapshot | None = None

    def seal_resource_endpoint(self) -> None:
        if self.process_io_end is None:
            self.process_io_end = _process_io_snapshot()
            self.database_physical_bytes = _database_physical_bytes(
                self.config.database
            )

    def process_io_metrics(
        self,
    ) -> tuple[str, str, int | None, int | None, int | None]:
        end = self.process_io_end or _process_io_snapshot()
        delta = None
        if (
            end.provider == self.process_io_start.provider
            and end.write_bytes is not None
            and self.process_io_start.write_bytes is not None
            and end.write_bytes >= self.process_io_start.write_bytes
        ):
            delta = end.write_bytes - self.process_io_start.write_bytes
        return (
            end.provider,
            end.semantics,
            self.process_io_start.write_bytes,
            end.write_bytes,
            delta,
        )

    def start(self, timeout_seconds: float = 5.0) -> None:
        if self.database_physical_bytes + _DATABASE_SIZE_CAP_RESERVE_BYTES >= int(
            self.config.database_size_cap_bytes
        ):
            raise _ImpactDatabaseSizeCapReached(
                "Round 73 database is already inside its 512 MiB cap reserve"
            )
        self.thread.start()
        if not self.started.wait(timeout=max(0.1, float(timeout_seconds))):
            raise _ImpactWriterFault(
                "Round 73 writer did not start within its deadline"
            )
        if self.failed.is_set():
            raise _ImpactWriterFault(
                self.error or "Round 73 writer failed during startup"
            )

    def put(self, message: ImpactCaptureMessage) -> None:
        if self.failed.is_set():
            raise _ImpactWriterFault(self.error or "Round 73 writer has failed")
        if self.cap_reached.is_set():
            raise _ImpactPayloadCapReached(
                "Round 73 compressed payload cap was reached"
            )
        if self.database_cap_reached.is_set():
            raise _ImpactDatabaseSizeCapReached(
                "Round 73 database size cap reserve was reached"
            )
        try:
            self.queue.put_nowait(message)
        except Full as exc:
            raise _ImpactWriterQueueOverflow("Round 73 writer queue overflow") from exc
        self.high_water_messages = max(self.high_water_messages, self.queue.qsize())

    def stop(self, timeout_seconds: float = 10.0) -> bool:
        if not self.thread.is_alive():
            self.seal_resource_endpoint()
            return True
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            try:
                self.queue.put(self._STOP, timeout=0.1)
                break
            except Full:
                if self.failed.is_set():
                    break
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        joined = not self.thread.is_alive()
        if joined:
            self.seal_resource_endpoint()
        return joined

    def _run(self) -> None:
        pending: list[ImpactCaptureMessage] = []
        pending_bytes = 0
        flush_deadline = time.monotonic() + float(self.config.frame_flush_seconds)

        def flush(store: ImpactAbsorptionStore) -> ImpactFrameWriteResult | None:
            nonlocal pending_bytes, flush_deadline
            if not pending:
                flush_deadline = time.monotonic() + float(
                    self.config.frame_flush_seconds
                )
                return None
            result = store.append_frame(run_id=self.run_id, messages=tuple(pending))
            self.frame_count += 1
            self.message_count += result.message_count
            self.compressed_payload_bytes = result.compressed_payload_total_bytes
            pending.clear()
            pending_bytes = 0
            flush_deadline = time.monotonic() + float(self.config.frame_flush_seconds)
            self.last_progress_monotonic = time.monotonic()
            if result.payload_cap_reached:
                self.cap_reached.set()
            self.database_physical_bytes = _database_physical_bytes(
                self.config.database
            )
            if self.database_physical_bytes + _DATABASE_SIZE_CAP_RESERVE_BYTES >= int(
                self.config.database_size_cap_bytes
            ):
                self.database_cap_reached.set()
            return result

        try:
            with ImpactAbsorptionStore(
                self.config.database,
                memory_limit=self.config.duckdb_memory_limit,
                threads=self.config.duckdb_threads,
            ) as store:
                self.started.set()
                while True:
                    timeout = max(0.0, flush_deadline - time.monotonic())
                    try:
                        item = self.queue.get(timeout=timeout)
                    except Empty:
                        flush(store)
                        continue
                    if item is self._STOP:
                        self.queue.task_done()
                        flush(store)
                        break
                    if not isinstance(item, ImpactCaptureMessage):
                        raise TypeError(
                            "Round 73 writer received an invalid queue item"
                        )
                    item_bytes = impact_capture_frame_record_size(item.record)
                    if item_bytes > int(self.config.frame_uncompressed_limit_bytes):
                        raise ValueError(
                            "one exact-wire record exceeds the frame byte bound"
                        )
                    if pending and (
                        len(pending) >= int(self.config.frame_message_limit)
                        or pending_bytes + item_bytes
                        > int(self.config.frame_uncompressed_limit_bytes)
                    ):
                        flush(store)
                    pending.append(item)
                    pending_bytes += item_bytes
                    self.queue.task_done()
                    self.last_progress_monotonic = time.monotonic()
                    if (
                        len(pending) >= int(self.config.frame_message_limit)
                        or pending_bytes
                        >= int(self.config.frame_uncompressed_limit_bytes)
                        or time.monotonic() >= flush_deadline
                    ):
                        flush(store)
                    if self.cap_reached.is_set() or self.database_cap_reached.is_set():
                        break
        except BaseException as exc:
            self.error = f"{type(exc).__name__}:{exc}"[:2_000]
            self.failed.set()
            self.started.set()
        finally:
            self.stopped.set()


def _writer_resource_metrics(
    writer: _ImpactFrameWriter | None,
    *,
    elapsed_seconds: float,
    queue_capacity_messages: int,
) -> _WriterResourceMetrics:
    if writer is None:
        snapshot = _process_io_snapshot()
        return _WriterResourceMetrics(
            process_io_provider=snapshot.provider,
            process_io_semantics=snapshot.semantics,
            process_io_start_write_bytes=None,
            process_io_end_write_bytes=snapshot.write_bytes,
            process_io_delta_write_bytes=None,
            process_io_write_bytes_per_message=None,
            database_physical_start_bytes=None,
            database_physical_end_bytes=0,
            database_physical_growth_bytes=None,
            database_physical_growth_bytes_per_message=None,
            frames_per_stream_minute=0.0,
            storage_efficiency_passed=False,
        )
    if not writer.thread.is_alive():
        writer.seal_resource_endpoint()
    provider, semantics, start_bytes, end_bytes, delta_bytes = (
        writer.process_io_metrics()
    )
    write_bytes_per_message = (
        None
        if delta_bytes is None or writer.message_count == 0
        else delta_bytes / writer.message_count
    )
    frames_per_minute = (
        0.0 if elapsed_seconds <= 0.0 else writer.frame_count * 60.0 / elapsed_seconds
    )
    queue_utilization = writer.high_water_messages / queue_capacity_messages
    physical_growth_bytes = (
        writer.database_physical_bytes - writer.database_physical_start_bytes
    )
    physical_growth_bytes_per_message = (
        None
        if writer.message_count == 0
        else physical_growth_bytes / writer.message_count
    )
    _contract, _report_schema, write_limit, physical_growth_limit = _capture_protocol(
        writer.config.schema_version
    )
    passed = (
        elapsed_seconds >= _STORAGE_EFFICIENCY_MINIMUM_SECONDS
        and frames_per_minute <= _STORAGE_EFFICIENCY_MAXIMUM_FRAMES_PER_MINUTE
        and write_bytes_per_message is not None
        and write_bytes_per_message <= write_limit
        and physical_growth_bytes_per_message is not None
        and physical_growth_bytes_per_message <= physical_growth_limit
        and queue_utilization <= 0.8
        and not writer.cap_reached.is_set()
        and not writer.database_cap_reached.is_set()
        and not writer.failed.is_set()
    )
    return _WriterResourceMetrics(
        process_io_provider=provider,
        process_io_semantics=semantics,
        process_io_start_write_bytes=start_bytes,
        process_io_end_write_bytes=end_bytes,
        process_io_delta_write_bytes=delta_bytes,
        process_io_write_bytes_per_message=write_bytes_per_message,
        database_physical_start_bytes=writer.database_physical_start_bytes,
        database_physical_end_bytes=writer.database_physical_bytes,
        database_physical_growth_bytes=physical_growth_bytes,
        database_physical_growth_bytes_per_message=(physical_growth_bytes_per_message),
        frames_per_stream_minute=frames_per_minute,
        storage_efficiency_passed=passed,
    )


def _v9_terminal_event_analysis(
    connection,
    *,
    run_id: str,
) -> tuple[dict[str, int], dict[str, dict[str, int]], float | None]:
    rest_context = {
        (int(row[0]), int(row[1])): tuple(row[2:])
        for row in connection.execute(
            f"""
            SELECT frame_index, message_index, event_type, symbol,
                   request_started_wall_ns, request_started_monotonic_ns,
                   exchange_time_ms
            FROM {IMPACT_CAPTURE_V9_REST_CONTEXT_TABLE}
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchall()
    }
    event_counts: dict[str, int] = {}
    symbol_event_counts: dict[str, dict[str, int]] = {
        symbol: {} for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    best_rtt: int | None = None
    active_offset: int | None = None
    pending_offset: int | None = None
    pending_receipt_ns = 0
    negative_count = 0
    latency_with_clock_count = 0
    consumed_rest_keys: set[tuple[int, int]] = set()
    for frame_index, message_index, record in iter_impact_capture_v9_records(
        connection,
        run_id=run_id,
    ):
        receipt_ns = record.received_monotonic_ns
        if pending_offset is not None and receipt_ns > pending_receipt_ns:
            active_offset = pending_offset
        key = (frame_index, message_index)
        if record.stream == "binance_futures_rest":
            try:
                (
                    event_type_value,
                    symbol_value,
                    request_started_wall_ns,
                    request_started_monotonic_ns,
                    exchange_time_ms,
                ) = rest_context[key]
            except KeyError as exc:
                raise ValueError(
                    "Round 73 v9 terminal REST context is missing"
                ) from exc
            consumed_rest_keys.add(key)
            event_type = str(event_type_value)
            symbol = str(symbol_value)
            if event_type == "serverTime":
                if exchange_time_ms is None:
                    raise ValueError("Round 73 v9 server-time context is incomplete")
                rtt_ns = receipt_ns - int(request_started_monotonic_ns)
                if rtt_ns < 0:
                    raise ValueError("Round 73 v9 server-time RTT is negative")
                if best_rtt is None or rtt_ns < best_rtt:
                    best_rtt = rtt_ns
                    midpoint_wall_ns = (
                        int(request_started_wall_ns) + record.received_wall_ns
                    ) // 2
                    pending_offset = (
                        int(exchange_time_ms) * 1_000_000 - midpoint_wall_ns
                    )
                    pending_receipt_ns = receipt_ns
        else:
            event_type = ImpactAbsorptionStore._v9_websocket_event_type(record)
            symbol = ""
            event_time_ms: int | None = None
            if event_type != "rejectedWire":
                root = _strict_json_object(record.raw_text)
                payload = root.get("data")
                if not isinstance(payload, Mapping):
                    raise ValueError("Round 73 v9 terminal payload is missing")
                symbol_value = payload.get("s")
                if event_type == "forceOrder" and isinstance(payload.get("o"), Mapping):
                    symbol_value = payload["o"].get("s")
                symbol = normalize_symbol(symbol_value, default="")
                event_time_value = payload.get("E")
                if (
                    isinstance(event_time_value, bool)
                    or not isinstance(event_time_value, int)
                    or event_time_value < 1
                ):
                    raise ValueError("Round 73 v9 event time is invalid")
                event_time_ms = event_time_value
            if active_offset is not None and event_time_ms is not None:
                latency_with_clock_count += 1
                if record.received_wall_ns + active_offset < event_time_ms * 1_000_000:
                    negative_count += 1
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        if symbol:
            if symbol not in symbol_event_counts:
                raise ValueError("Round 73 v9 terminal symbol differs")
            symbol_counts = symbol_event_counts[symbol]
            symbol_counts[event_type] = symbol_counts.get(event_type, 0) + 1
    if consumed_rest_keys != set(rest_context):
        raise ValueError("Round 73 v9 terminal REST context has orphan rows")
    for symbol, depth_message_count in connection.execute(
        "SELECT symbol, depth_message_count FROM impact_capture_segment "
        "WHERE run_id = ? ORDER BY symbol",
        [run_id],
    ).fetchall():
        selected_symbol = str(symbol)
        if selected_symbol not in symbol_event_counts:
            raise ValueError("Round 73 v9 terminal segment symbol differs")
        symbol_event_counts[selected_symbol]["synchronizedDepthUpdate"] = int(
            depth_message_count
        )
    negative_fraction = (
        None
        if latency_with_clock_count == 0
        else negative_count / latency_with_clock_count
    )
    return event_counts, symbol_event_counts, negative_fraction


def _terminal_read_only_analysis(
    config: ImpactCaptureConfig,
    *,
    run_id: str,
) -> tuple[
    ImpactCaptureAudit,
    dict[str, int],
    dict[str, dict[str, int]],
    float | None,
]:
    with ImpactAbsorptionStore(
        config.database,
        read_only=True,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    ) as store:
        audit = store.audit_run(run_id)
        connection = store.connect()
        run_schema_row = connection.execute(
            "SELECT schema_version FROM impact_capture_run WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if run_schema_row is None:
            raise ValueError("Round 73 terminal capture run is missing")
        if str(run_schema_row[0]) == IMPACT_CAPTURE_V9_SCHEMA_VERSION:
            event_counts, symbol_event_counts, negative_latency_fraction = (
                _v9_terminal_event_analysis(connection, run_id=run_id)
            )
            return (
                audit,
                event_counts,
                symbol_event_counts,
                negative_latency_fraction,
            )
        event_counts = {
            str(event_type): int(count)
            for event_type, count in connection.execute(
                f"SELECT event_type, count(*) FROM {IMPACT_EVENT_LINK_TABLE} "
                "WHERE run_id = ? GROUP BY event_type ORDER BY event_type",
                [run_id],
            ).fetchall()
        }
        symbol_event_counts: dict[str, dict[str, int]] = {
            symbol: {} for symbol in IMPACT_CAPTURE_SYMBOLS
        }
        for symbol, event_type, count in connection.execute(
            f"SELECT symbol, event_type, count(*) FROM {IMPACT_EVENT_LINK_TABLE} "
            "WHERE run_id = ? AND symbol <> '' "
            "GROUP BY symbol, event_type ORDER BY symbol, event_type",
            [run_id],
        ).fetchall():
            symbol_event_counts[str(symbol)][str(event_type)] = int(count)
        for symbol, count in connection.execute(
            f"SELECT symbol, count(*) FROM {IMPACT_DEPTH_UPDATE_TABLE} "
            "WHERE run_id = ? AND stale = false "
            "GROUP BY symbol ORDER BY symbol",
            [run_id],
        ).fetchall():
            symbol_event_counts[str(symbol)]["synchronizedDepthUpdate"] = int(count)

        latency_rows = connection.execute(
            f"SELECT received_wall_ns, event_time_ms "
            f"FROM {IMPACT_EVENT_LINK_TABLE} "
            "WHERE run_id = ? AND event_time_ms IS NOT NULL "
            "AND stream <> 'binance_futures_rest' ORDER BY received_wall_ns",
            [run_id],
        ).fetchall()
        clock_rows = connection.execute(
            f"SELECT e.received_wall_ns, r.request_started_wall_ns, "
            "e.received_monotonic_ns, r.request_started_monotonic_ns, "
            f"r.exchange_time_ms FROM {IMPACT_EVENT_LINK_TABLE} e "
            f"JOIN {IMPACT_REST_EVENT_TABLE} r "
            "USING (run_id, frame_index, message_index) "
            "WHERE e.run_id = ? AND e.event_type = 'serverTime' "
            "ORDER BY e.received_wall_ns",
            [run_id],
        ).fetchall()
        causal_clock: list[tuple[int, int]] = []
        best_rtt: int | None = None
        best_offset = 0
        for (
            received_wall_ns,
            request_started_wall_ns,
            received_monotonic_ns,
            request_started_monotonic_ns,
            exchange_time_ms,
        ) in clock_rows:
            rtt = int(received_monotonic_ns) - int(request_started_monotonic_ns)
            if best_rtt is None or rtt < best_rtt:
                best_rtt = rtt
                midpoint = (int(request_started_wall_ns) + int(received_wall_ns)) // 2
                best_offset = int(exchange_time_ms) * 1_000_000 - midpoint
            causal_clock.append((int(received_wall_ns), best_offset))
        negative_count = 0
        latency_with_clock_count = 0
        clock_index = 0
        active_offset: int | None = None
        for received_wall_ns, event_time_ms in latency_rows:
            while clock_index < len(causal_clock) and causal_clock[clock_index][
                0
            ] <= int(received_wall_ns):
                active_offset = causal_clock[clock_index][1]
                clock_index += 1
            if active_offset is not None:
                latency_with_clock_count += 1
                if (
                    int(received_wall_ns) + active_offset
                    < int(event_time_ms) * 1_000_000
                ):
                    negative_count += 1
        negative_latency_fraction = (
            None
            if latency_with_clock_count == 0
            else negative_count / latency_with_clock_count
        )
    return (
        audit,
        event_counts,
        symbol_event_counts,
        negative_latency_fraction,
    )


def _rest_request(
    *,
    run_id: str,
    sequence_number: int,
    event_type: Literal["serverTime", "exchangeInfo", "depthSnapshot", "openInterest"],
    path: str,
    parameters: Mapping[str, object],
    timeout_seconds: float,
) -> _RestEvidence:
    request_started_wall_ns = time.time_ns()
    request_started_monotonic_ns = time.perf_counter_ns()
    response = requests.get(
        f"{BINANCE_FUTURES_REST}{path}",
        params=dict(parameters),
        timeout=float(timeout_seconds),
        headers={"User-Agent": "simple-ai-trading-round73/0.1.0-beta.1"},
    )
    received_monotonic_ns = time.perf_counter_ns()
    received_wall_ns = time.time_ns()
    raw_text = response.content.decode("utf-8", errors="strict")
    body = _strict_json_object(raw_text)
    used_raw = response.headers.get("X-MBX-USED-WEIGHT-1M")
    used_weight = None if used_raw is None else int(used_raw)
    if response.status_code != 200:
        raise _ImpactRestResponseError(path, response.status_code)
    symbol = str(parameters.get("symbol", ""))
    exchange_time_ms = None
    update_id = None
    open_interest = None
    if event_type == "serverTime":
        exchange_time_ms = int(body["serverTime"])
    elif event_type == "depthSnapshot":
        update_id = int(body["lastUpdateId"])
    elif event_type == "openInterest":
        exchange_time_ms = int(body["time"])
        open_interest = float(body["openInterest"])
    event = ImpactRestEvent(
        event_type=event_type,
        request_path=path,
        request_parameters=dict(parameters),
        response_status=response.status_code,
        request_started_wall_ns=request_started_wall_ns,
        request_started_monotonic_ns=request_started_monotonic_ns,
        symbol=symbol,
        exchange_time_ms=exchange_time_ms,
        update_id=update_id,
        open_interest=open_interest,
        used_weight_1m=used_weight,
    )
    record = ImpactCaptureFrameRecord(
        stream="binance_futures_rest",
        connection_id=f"binance-rest:{run_id}",
        sequence_number=sequence_number,
        received_wall_ns=received_wall_ns,
        received_monotonic_ns=received_monotonic_ns,
        raw_text=raw_text,
    )
    return _RestEvidence(
        message=ImpactCaptureMessage(record=record, event=event),
        body=body,
        used_weight_1m=used_weight,
    )


def _request_weight_limit(exchange_info: Mapping[str, object]) -> int:
    limits = exchange_info.get("rateLimits")
    if not isinstance(limits, Sequence) or isinstance(limits, (str, bytes, bytearray)):
        raise ValueError("exchange information has no rate-limit array")
    matches = [
        int(item["limit"])
        for item in limits
        if isinstance(item, Mapping)
        and item.get("rateLimitType") == "REQUEST_WEIGHT"
        and item.get("interval") == "MINUTE"
        and int(item.get("intervalNum", 0)) == 1
    ]
    if len(matches) != 1 or matches[0] <= 0:
        raise ValueError("one-minute request-weight limit is missing or ambiguous")
    return matches[0]


def _tick_sizes(exchange_info: Mapping[str, object]) -> dict[str, str]:
    rows = exchange_info.get("symbols")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("exchange information has no symbol array")
    result: dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("symbol") not in IMPACT_CAPTURE_SYMBOLS
        ):
            continue
        filters = row.get("filters")
        if not isinstance(filters, Sequence):
            raise ValueError("exchange symbol has no filters")
        price_filters = [
            item
            for item in filters
            if isinstance(item, Mapping) and item.get("filterType") == "PRICE_FILTER"
        ]
        if len(price_filters) != 1:
            raise ValueError("exchange symbol has ambiguous price filters")
        result[str(row["symbol"])] = str(price_filters[0]["tickSize"])
    if set(result) != set(IMPACT_CAPTURE_SYMBOLS):
        raise ValueError("exchange information is missing a Round 73 symbol")
    return {symbol: result[symbol] for symbol in IMPACT_CAPTURE_SYMBOLS}


def _public_stream_url() -> str:
    streams = [
        f"{symbol.lower()}@{suffix}"
        for symbol in IMPACT_CAPTURE_SYMBOLS
        for suffix in _PUBLIC_SUFFIXES
    ]
    return BINANCE_FUTURES_PUBLIC_STREAM + "/".join(streams)


def _market_stream_url() -> str:
    streams = [
        f"{symbol.lower()}@{suffix}"
        for symbol in IMPACT_CAPTURE_SYMBOLS
        for suffix in _MARKET_SUFFIXES
    ]
    return BINANCE_FUTURES_MARKET_STREAM + "/".join(streams)


def _failure_class_for_exception(exc: BaseException) -> CaptureFailureClass:
    if isinstance(exc, _ImpactRateLimitGuardError):
        return "rate_limit"
    if isinstance(exc, _ImpactRestResponseError):
        return "rate_limit" if exc.status_code in {418, 429} else "rest_transport"
    if isinstance(
        exc,
        _ImpactWriterQueueOverflow
        | _ImpactPayloadCapReached
        | _ImpactDatabaseSizeCapReached,
    ):
        return "resource_limit"
    if isinstance(exc, _ImpactWriterFault):
        return "writer"
    if isinstance(exc, requests.exceptions.RequestException):
        return "rest_transport"
    if isinstance(exc, WebSocketException | TimeoutError | ConnectionError):
        return "transport"
    if isinstance(exc, ImpactFeedIntegrityError | ValueError | KeyError):
        return "feed_integrity"
    return "processing"


def _terminal_post_capture_failure_report(
    config: ImpactCaptureConfig,
    *,
    run_id: str,
    error: BaseException,
    writer: _ImpactFrameWriter | None,
) -> ImpactCaptureReport | None:
    """Bind a non-retriable failed report to an already terminal current run."""

    capture_contract, _report_schema, _write_limit, _growth_limit = _capture_protocol(
        config.schema_version
    )
    with ImpactAbsorptionStore(
        config.database,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    ) as store:
        connection = store.connect()
        run = connection.execute(
            """
            SELECT status, started_wall_ns, ended_wall_ns, frame_count,
                   message_count, compressed_payload_bytes, payload_cap_reached
            FROM impact_capture_run WHERE run_id = ?
              AND schema_version = ? AND capture_contract_sha256 = ?
            """,
            [
                run_id,
                config.schema_version,
                capture_contract,
            ],
        ).fetchone()
        if run is None or str(run[0]) == "running" or run[2] is None:
            return None
        if (
            connection.execute(
                "SELECT count(*) FROM impact_capture_report WHERE run_id = ?",
                [run_id],
            ).fetchone()[0]
            != 0
        ):
            return None
        audit = store.audit_run(run_id)
        if config.schema_version == IMPACT_CAPTURE_V9_SCHEMA_VERSION:
            event_counts, symbol_event_counts, _negative_fraction = (
                _v9_terminal_event_analysis(connection, run_id=run_id)
            )
        else:
            event_counts = {
                str(event_type): int(count)
                for event_type, count in connection.execute(
                    f"SELECT event_type, count(*) FROM {IMPACT_EVENT_LINK_TABLE} "
                    "WHERE run_id = ? GROUP BY event_type ORDER BY event_type",
                    [run_id],
                ).fetchall()
            }
            symbol_event_counts = {symbol: {} for symbol in IMPACT_CAPTURE_SYMBOLS}
            for symbol, event_type, count in connection.execute(
                f"SELECT symbol, event_type, count(*) FROM {IMPACT_EVENT_LINK_TABLE} "
                "WHERE run_id = ? AND symbol <> '' "
                "GROUP BY symbol, event_type ORDER BY symbol, event_type",
                [run_id],
            ).fetchall():
                symbol_event_counts[str(symbol)][str(event_type)] = int(count)
            for symbol, count in connection.execute(
                f"SELECT symbol, count(*) FROM {IMPACT_DEPTH_UPDATE_TABLE} "
                "WHERE run_id = ? AND stale = false GROUP BY symbol ORDER BY symbol",
                [run_id],
            ).fetchall():
                symbol_event_counts[str(symbol)]["synchronizedDepthUpdate"] = int(count)
        elapsed_seconds = max(
            0.0,
            (int(run[2]) - int(run[1])) / 1_000_000_000,
        )
        metrics = _writer_resource_metrics(
            writer,
            elapsed_seconds=elapsed_seconds,
            queue_capacity_messages=config.queue_capacity_messages,
        )
        terminal_snapshot = _process_io_snapshot()
        database_end = _database_physical_bytes(config.database)
        report = ImpactCaptureReport(
            run_id=run_id,
            capture_schema_version=config.schema_version,
            mode=config.mode,
            status="failed",
            capture_gate_passed=False,
            qualification_passed=False,
            started_wall_ns=int(run[1]),
            ended_wall_ns=int(run[2]),
            elapsed_seconds=elapsed_seconds,
            queue_high_water_messages=(
                0 if writer is None else writer.high_water_messages
            ),
            queue_capacity_messages=config.queue_capacity_messages,
            queue_maximum_utilization=(
                0.0
                if writer is None
                else writer.high_water_messages / config.queue_capacity_messages
            ),
            writer_frame_count=int(run[3]),
            writer_message_count=int(run[4]),
            writer_compressed_payload_bytes=int(run[5]),
            payload_cap_reached=bool(run[6]),
            database_physical_start_bytes=metrics.database_physical_start_bytes,
            database_physical_bytes=database_end,
            database_physical_growth_bytes=metrics.database_physical_growth_bytes,
            database_physical_growth_bytes_per_message=(
                metrics.database_physical_growth_bytes_per_message
            ),
            database_size_cap_bytes=config.database_size_cap_bytes,
            database_size_cap_reached=(
                False if writer is None else writer.database_cap_reached.is_set()
            ),
            process_io_scope="capture phase through writer connection close",
            process_io_provider=metrics.process_io_provider,
            process_io_semantics=metrics.process_io_semantics,
            process_io_start_write_bytes=metrics.process_io_start_write_bytes,
            process_io_end_write_bytes=metrics.process_io_end_write_bytes,
            process_io_delta_write_bytes=metrics.process_io_delta_write_bytes,
            process_io_write_bytes_per_message=(
                metrics.process_io_write_bytes_per_message
            ),
            frames_per_stream_minute=metrics.frames_per_stream_minute,
            storage_efficiency_passed=metrics.storage_efficiency_passed,
            terminal_process_io_provider=terminal_snapshot.provider,
            terminal_process_io_semantics=terminal_snapshot.semantics,
            terminal_process_io_start_write_bytes=None,
            terminal_process_io_end_write_bytes=terminal_snapshot.write_bytes,
            terminal_process_io_delta_write_bytes=None,
            event_counts=event_counts,
            symbol_event_counts=symbol_event_counts,
            negative_corrected_latency_fraction=None,
            audit_passed=audit.passed,
            audit_errors=audit.errors,
            error=f"post_capture:{type(error).__name__}:{error}"[:2_000],
            failure_class="post_capture",
        )
        store.record_report(
            run_id=run_id,
            report=report.as_dict(),
            recorded_at_wall_ns=time.time_ns(),
        )
        return report


async def capture_round73(config: ImpactCaptureConfig) -> ImpactCaptureReport:
    """Capture one bounded prospective run; never place or authenticate an order."""

    config.validate()
    if _database_physical_bytes(
        config.database
    ) + _DATABASE_SIZE_CAP_RESERVE_BYTES >= int(config.database_size_cap_bytes):
        raise _ImpactDatabaseSizeCapReached(
            "Round 73 database is already inside its 512 MiB cap reserve"
        )
    run_id = uuid.uuid4().hex
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.perf_counter_ns()
    public_queue: asyncio.Queue[_WireReceipt] = asyncio.Queue(
        maxsize=int(config.queue_capacity_messages)
    )
    fatal = asyncio.Event()
    stop = asyncio.Event()
    errors: list[str] = []
    failure_class: CaptureFailureClass = "none"
    public_last_receipt = time.monotonic()
    market_last_receipt = time.monotonic()
    public_sequence = 0
    market_sequence = 0

    def fail(reason: str, category: CaptureFailureClass) -> None:
        nonlocal failure_class
        if not errors:
            errors.append(str(reason)[:2_000])
            failure_class = category
        fatal.set()

    async def public_receiver(websocket) -> None:
        nonlocal public_last_receipt, public_sequence
        try:
            while not stop.is_set():
                raw = await asyncio.wait_for(
                    websocket.recv(), timeout=float(config.source_stall_seconds)
                )
                received_monotonic_ns = time.perf_counter_ns()
                received_wall_ns = time.time_ns()
                public_last_receipt = time.monotonic()
                if not isinstance(raw, str):
                    raw = bytes(raw).decode("utf-8", errors="strict")
                receipt = _WireReceipt(
                    raw_text=raw,
                    sequence_number=public_sequence,
                    received_wall_ns=received_wall_ns,
                    received_monotonic_ns=received_monotonic_ns,
                )
                public_sequence += 1
                try:
                    public_queue.put_nowait(receipt)
                except asyncio.QueueFull as exc:
                    raise RuntimeError(
                        "Round 73 public processing queue overflow"
                    ) from exc
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            fail(f"public_source:{type(exc).__name__}:{exc}", "transport")

    async with connect(
        _public_stream_url(),
        open_timeout=float(config.request_timeout_seconds),
        close_timeout=2,
        ping_interval=20,
        ping_timeout=20,
        max_size=1024 * 1024,
        max_queue=4_096,
    ) as public_websocket:
        public_task = asyncio.create_task(public_receiver(public_websocket))
        writer: _ImpactFrameWriter | None = None
        rest_sequence = 0
        retained_rest: list[_RestEvidence] = []
        clock_probes: list[_RestEvidence] = []
        try:
            for _index in range(_CLOCK_PROBE_COUNT):
                evidence = await asyncio.to_thread(
                    _rest_request,
                    run_id=run_id,
                    sequence_number=rest_sequence,
                    event_type="serverTime",
                    path="/fapi/v1/time",
                    parameters={},
                    timeout_seconds=config.request_timeout_seconds,
                )
                rest_sequence += 1
                retained_rest.append(evidence)
                clock_probes.append(evidence)
            exchange_evidence = await asyncio.to_thread(
                _rest_request,
                run_id=run_id,
                sequence_number=rest_sequence,
                event_type="exchangeInfo",
                path="/fapi/v1/exchangeInfo",
                parameters={},
                timeout_seconds=config.request_timeout_seconds,
            )
            rest_sequence += 1
            retained_rest.append(exchange_evidence)
            tick_sizes = _tick_sizes(exchange_evidence.body)
            request_weight_limit = _request_weight_limit(exchange_evidence.body)
            snapshots: dict[str, _RestEvidence] = {}
            for symbol in IMPACT_CAPTURE_SYMBOLS:
                snapshot = await asyncio.to_thread(
                    _rest_request,
                    run_id=run_id,
                    sequence_number=rest_sequence,
                    event_type="depthSnapshot",
                    path="/fapi/v1/depth",
                    parameters={"symbol": symbol, "limit": 1000},
                    timeout_seconds=config.request_timeout_seconds,
                )
                rest_sequence += 1
                retained_rest.append(snapshot)
                snapshots[symbol] = snapshot
                open_interest = await asyncio.to_thread(
                    _rest_request,
                    run_id=run_id,
                    sequence_number=rest_sequence,
                    event_type="openInterest",
                    path="/fapi/v1/openInterest",
                    parameters={"symbol": symbol},
                    timeout_seconds=config.request_timeout_seconds,
                )
                rest_sequence += 1
                retained_rest.append(open_interest)
            if fatal.is_set():
                raise RuntimeError(
                    errors[0] if errors else "public source failed during preflight"
                )
            observed_weights = [
                item.used_weight_1m
                for item in retained_rest
                if item.used_weight_1m is not None
            ]
            latest_weight = max(observed_weights, default=0)
            if latest_weight / request_weight_limit >= _RATE_LIMIT_FRACTION:
                raise _ImpactRateLimitGuardError(
                    "Round 73 capture blocked because one-minute request weight is at or above 80%"
                )

            selected_clock = min(
                clock_probes,
                key=lambda item: (
                    item.message.record.received_monotonic_ns
                    - item.message.event.request_started_monotonic_ns
                ),
            )
            selected_clock_event = selected_clock.message.event
            if not isinstance(selected_clock_event, ImpactRestEvent):
                raise RuntimeError("clock probe lost its typed REST contract")
            clock_rtt_ns = (
                selected_clock.message.record.received_monotonic_ns
                - selected_clock_event.request_started_monotonic_ns
            )
            clock_midpoint_wall_ns = (
                selected_clock_event.request_started_wall_ns
                + selected_clock.message.record.received_wall_ns
            ) // 2
            clock_offset_ns = (
                int(selected_clock_event.exchange_time_ms or 0) * 1_000_000
                - clock_midpoint_wall_ns
            )
            books: dict[str, SynchronizedDepthBook] = {}
            segment_ids = {
                symbol: uuid.uuid4().hex for symbol in IMPACT_CAPTURE_SYMBOLS
            }
            for symbol in IMPACT_CAPTURE_SYMBOLS:
                book = SynchronizedDepthBook(symbol, tick_sizes[symbol])
                book.initialize(snapshots[symbol].body)
                books[symbol] = book

            with ImpactAbsorptionStore(
                config.database,
                memory_limit=config.duckdb_memory_limit,
                threads=config.duckdb_threads,
            ) as store:
                store.start_run(
                    run_id=run_id,
                    started_wall_ns=started_wall_ns,
                    started_monotonic_ns=started_monotonic_ns,
                    config={
                        "mode": config.mode,
                        "duration_seconds": config.duration_seconds,
                        "queue_capacity_messages": config.queue_capacity_messages,
                        "frame_message_limit": config.frame_message_limit,
                        "frame_uncompressed_limit_bytes": config.frame_uncompressed_limit_bytes,
                        "frame_flush_seconds": config.frame_flush_seconds,
                        "maximum_reconnects": config.maximum_reconnects,
                        "database_size_cap_bytes": config.database_size_cap_bytes,
                        "duckdb_memory_limit": config.duckdb_memory_limit,
                        "duckdb_threads": config.duckdb_threads,
                    },
                    compressed_payload_cap_bytes=config.compressed_payload_cap_bytes,
                    schema_version=config.schema_version,
                )
                cooldown_until = time.time_ns() + 60_000_000_000
                for symbol in IMPACT_CAPTURE_SYMBOLS:
                    store.start_segment(
                        run_id=run_id,
                        segment_id=segment_ids[symbol],
                        symbol=symbol,
                        started_wall_ns=started_wall_ns,
                        started_monotonic_ns=started_monotonic_ns,
                        snapshot_update_id=int(snapshots[symbol].body["lastUpdateId"]),
                        tick_size=float(tick_sizes[symbol]),
                        clock_offset_ns=clock_offset_ns,
                        clock_rtt_ns=clock_rtt_ns,
                        cooldown_until_wall_ns=cooldown_until,
                    )

            retained_messages: list[ImpactCaptureMessage] = []
            for evidence in retained_rest:
                event = evidence.message.event
                segment_id = ""
                if isinstance(event, ImpactRestEvent) and event.symbol:
                    segment_id = segment_ids[event.symbol]
                retained_messages.append(
                    ImpactCaptureMessage(
                        record=evidence.message.record,
                        event=event,
                        segment_id=segment_id,
                    )
                )

            writer = _ImpactFrameWriter(config, run_id)
            writer.start()
            for message in retained_messages:
                writer.put(message)
            event_clock: dict[tuple[str, str, str], int] = {}

            def check_event_clock(
                source: str, event_type: str, symbol: str, value: int
            ) -> None:
                key = (source, event_type, symbol)
                prior = event_clock.get(key)
                if prior is not None and value < prior:
                    raise ImpactFeedIntegrityError(
                        f"exchange event time regressed for {source}:{event_type}:{symbol}"
                    )
                event_clock[key] = value

            async def public_processor() -> None:
                try:
                    while not stop.is_set() or not public_queue.empty():
                        try:
                            receipt = await asyncio.wait_for(
                                public_queue.get(), timeout=0.1
                            )
                        except TimeoutError:
                            continue
                        record = ImpactCaptureFrameRecord(
                            stream="binance_futures_public",
                            connection_id=f"binance-public:{run_id}",
                            sequence_number=receipt.sequence_number,
                            received_wall_ns=receipt.received_wall_ns,
                            received_monotonic_ns=receipt.received_monotonic_ns,
                            raw_text=receipt.raw_text,
                        )
                        try:
                            stream_name, payload = _combined_payload(receipt.raw_text)
                            event_type = str(payload.get("e", ""))
                            symbol = normalize_symbol(payload.get("s"), default="")
                            if symbol not in segment_ids:
                                raise ImpactFeedIntegrityError(
                                    "public stream symbol is outside Round 73"
                                )
                            validate_combined_stream_name(
                                stream_name,
                                event_type=event_type,
                                symbol=symbol,
                            )
                            check_event_clock(
                                "public", event_type, symbol, int(payload.get("E", 0))
                            )
                            if event_type == "depthUpdate":
                                pre_state = books[symbol].state()
                                event = books[symbol].apply(
                                    payload,
                                    receive_time_ns=receipt.received_monotonic_ns,
                                )
                                state = None if event.stale else books[symbol].state()
                                message = ImpactCaptureMessage(
                                    record=record,
                                    event=event,
                                    segment_id=segment_ids[symbol],
                                    pre_l2_state=pre_state,
                                    l2_state=state,
                                )
                            elif event_type == "bookTicker":
                                message = ImpactCaptureMessage(
                                    record=record,
                                    event=parse_book_ticker(
                                        payload,
                                        symbol=symbol,
                                        receive_time_ns=receipt.received_monotonic_ns,
                                    ),
                                    segment_id=segment_ids[symbol],
                                )
                            else:
                                raise ImpactFeedIntegrityError(
                                    f"unsupported public event: {event_type or 'missing'}"
                                )
                        except asyncio.CancelledError:
                            raise
                        except BaseException as exc:
                            try:
                                writer.put(
                                    _rejected_wire_message(
                                        record=record,
                                        error=exc,
                                        segment_ids=segment_ids,
                                    )
                                )
                            except BaseException as persistence_error:
                                fail(
                                    "public_rejection_persistence:"
                                    f"{type(persistence_error).__name__}:"
                                    f"{persistence_error}",
                                    "writer",
                                )
                                return
                            fail(
                                f"public_processor:{type(exc).__name__}:{exc}",
                                _failure_class_for_exception(exc),
                            )
                        else:
                            writer.put(message)
                        finally:
                            public_queue.task_done()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    fail(
                        f"public_processor:{type(exc).__name__}:{exc}",
                        _failure_class_for_exception(exc),
                    )

            async def market_receiver() -> None:
                nonlocal market_last_receipt, market_sequence
                try:
                    async with connect(
                        _market_stream_url(),
                        open_timeout=float(config.request_timeout_seconds),
                        close_timeout=2,
                        ping_interval=20,
                        ping_timeout=20,
                        max_size=1024 * 1024,
                        max_queue=4_096,
                    ) as websocket:
                        while not stop.is_set():
                            raw = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=float(config.source_stall_seconds),
                            )
                            received_monotonic_ns = time.perf_counter_ns()
                            received_wall_ns = time.time_ns()
                            market_last_receipt = time.monotonic()
                            if not isinstance(raw, str):
                                raw = bytes(raw).decode("utf-8", errors="strict")
                            sequence = market_sequence
                            market_sequence += 1
                            record = ImpactCaptureFrameRecord(
                                stream="binance_futures_market",
                                connection_id=f"binance-market:{run_id}",
                                sequence_number=sequence,
                                received_wall_ns=received_wall_ns,
                                received_monotonic_ns=received_monotonic_ns,
                                raw_text=raw,
                            )
                            try:
                                stream_name, payload = _combined_payload(raw)
                                event_type = str(payload.get("e", ""))
                                symbol_value = payload.get("s")
                                if event_type == "forceOrder" and isinstance(
                                    payload.get("o"), Mapping
                                ):
                                    symbol_value = payload["o"].get("s")
                                symbol = normalize_symbol(symbol_value, default="")
                                if symbol not in segment_ids:
                                    raise ImpactFeedIntegrityError(
                                        "market stream symbol is outside Round 73"
                                    )
                                validate_combined_stream_name(
                                    stream_name,
                                    event_type=event_type,
                                    symbol=symbol,
                                )
                                check_event_clock(
                                    "market",
                                    event_type,
                                    symbol,
                                    int(payload.get("E", 0)),
                                )
                                if event_type == "aggTrade":
                                    event = parse_aggregate_trade(
                                        payload,
                                        symbol=symbol,
                                        receive_time_ns=received_monotonic_ns,
                                    )
                                elif event_type == "markPriceUpdate":
                                    event = parse_mark_price(
                                        payload,
                                        symbol=symbol,
                                        receive_time_ns=received_monotonic_ns,
                                    )
                                elif event_type == "forceOrder":
                                    event = parse_liquidation_snapshot(
                                        payload,
                                        symbol=symbol,
                                        receive_time_ns=received_monotonic_ns,
                                    )
                                else:
                                    raise ImpactFeedIntegrityError(
                                        "unsupported market event: "
                                        f"{event_type or 'missing'}"
                                    )
                            except asyncio.CancelledError:
                                raise
                            except BaseException as exc:
                                try:
                                    writer.put(
                                        _rejected_wire_message(
                                            record=record,
                                            error=exc,
                                            segment_ids=segment_ids,
                                        )
                                    )
                                except BaseException as persistence_error:
                                    raise _ImpactWriterFault(
                                        "could not persist rejected market wire evidence: "
                                        f"{type(persistence_error).__name__}:"
                                        f"{persistence_error}"
                                    ) from persistence_error
                                raise
                            else:
                                writer.put(
                                    ImpactCaptureMessage(
                                        record=record,
                                        event=event,
                                        segment_id=segment_ids[symbol],
                                    )
                                )
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    fail(
                        f"market_source:{type(exc).__name__}:{exc}",
                        _failure_class_for_exception(exc),
                    )

            async def rest_poller() -> None:
                nonlocal rest_sequence
                try:
                    while not stop.is_set():
                        try:
                            await asyncio.wait_for(stop.wait(), timeout=60.0)
                            break
                        except TimeoutError:
                            pass
                        evidence = await asyncio.to_thread(
                            _rest_request,
                            run_id=run_id,
                            sequence_number=rest_sequence,
                            event_type="serverTime",
                            path="/fapi/v1/time",
                            parameters={},
                            timeout_seconds=config.request_timeout_seconds,
                        )
                        rest_sequence += 1
                        writer.put(evidence.message)
                        if (
                            evidence.used_weight_1m is not None
                            and evidence.used_weight_1m / request_weight_limit
                            >= _RATE_LIMIT_FRACTION
                        ):
                            raise _ImpactRateLimitGuardError(
                                "one-minute request weight reached the 80% capture stop gate"
                            )
                        for symbol in IMPACT_CAPTURE_SYMBOLS:
                            open_interest = await asyncio.to_thread(
                                _rest_request,
                                run_id=run_id,
                                sequence_number=rest_sequence,
                                event_type="openInterest",
                                path="/fapi/v1/openInterest",
                                parameters={"symbol": symbol},
                                timeout_seconds=config.request_timeout_seconds,
                            )
                            rest_sequence += 1
                            writer.put(
                                ImpactCaptureMessage(
                                    record=open_interest.message.record,
                                    event=open_interest.message.event,
                                    segment_id=segment_ids[symbol],
                                )
                            )
                            if (
                                open_interest.used_weight_1m is not None
                                and open_interest.used_weight_1m / request_weight_limit
                                >= _RATE_LIMIT_FRACTION
                            ):
                                raise _ImpactRateLimitGuardError(
                                    "one-minute request weight reached the 80% capture stop gate"
                                )
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    fail(
                        f"rest_poller:{type(exc).__name__}:{exc}",
                        _failure_class_for_exception(exc),
                    )

            processor_task = asyncio.create_task(public_processor())
            market_last_receipt = time.monotonic()
            market_task = asyncio.create_task(market_receiver())
            poll_task = asyncio.create_task(rest_poller())
            capture_started = time.monotonic()
            try:
                deadline = capture_started + float(config.duration_seconds)
                while time.monotonic() < deadline and not fatal.is_set():
                    await asyncio.sleep(0.1)
                    now = time.monotonic()
                    if writer.failed.is_set():
                        fail(writer.error or "writer_failed", "writer")
                    elif writer.cap_reached.is_set():
                        fail("compressed_payload_cap_reached", "resource_limit")
                    elif writer.database_cap_reached.is_set():
                        fail("database_size_cap_reserve_reached", "resource_limit")
                    elif (
                        writer.queue.qsize() > 0
                        and now - writer.last_progress_monotonic
                        > float(config.writer_stall_seconds)
                    ):
                        fail("writer_heartbeat_timeout", "writer")
                    elif now - public_last_receipt > float(config.source_stall_seconds):
                        fail("public_source_stall", "transport")
                    elif now - market_last_receipt > float(config.source_stall_seconds):
                        fail("market_source_stall", "transport")
            finally:
                stop.set()
                public_task.cancel()
                market_task.cancel()
                poll_task.cancel()
                await asyncio.gather(
                    public_task,
                    market_task,
                    poll_task,
                    return_exceptions=True,
                )
                try:
                    await asyncio.wait_for(public_queue.join(), timeout=5.0)
                except TimeoutError:
                    fail("public_processing_drain_timeout", "processing")
                    processor_task.cancel()
                await asyncio.gather(processor_task, return_exceptions=True)
                writer_joined = writer.stop(timeout_seconds=10.0)
                if not writer_joined:
                    fail("writer_shutdown_timeout", "writer")
                if writer.failed.is_set():
                    fail(writer.error or "writer_failed", "writer")
                if writer.cap_reached.is_set():
                    fail("compressed_payload_cap_reached", "resource_limit")
                if writer.database_cap_reached.is_set():
                    fail("database_size_cap_reserve_reached", "resource_limit")

            ended_wall_ns = time.time_ns()
            elapsed_seconds = max(0.0, time.monotonic() - capture_started)
            if writer.thread.is_alive():
                metrics = _writer_resource_metrics(
                    writer,
                    elapsed_seconds=elapsed_seconds,
                    queue_capacity_messages=config.queue_capacity_messages,
                )
                terminal_snapshot = _process_io_snapshot()
                return ImpactCaptureReport(
                    run_id=run_id,
                    capture_schema_version=config.schema_version,
                    mode=config.mode,
                    status="failed",
                    capture_gate_passed=False,
                    qualification_passed=False,
                    started_wall_ns=started_wall_ns,
                    ended_wall_ns=ended_wall_ns,
                    elapsed_seconds=elapsed_seconds,
                    queue_high_water_messages=writer.high_water_messages,
                    queue_capacity_messages=config.queue_capacity_messages,
                    queue_maximum_utilization=(
                        writer.high_water_messages / config.queue_capacity_messages
                    ),
                    writer_frame_count=writer.frame_count,
                    writer_message_count=writer.message_count,
                    writer_compressed_payload_bytes=writer.compressed_payload_bytes,
                    payload_cap_reached=writer.cap_reached.is_set(),
                    database_physical_start_bytes=(
                        metrics.database_physical_start_bytes
                    ),
                    database_physical_bytes=_database_physical_bytes(config.database),
                    database_physical_growth_bytes=(
                        metrics.database_physical_growth_bytes
                    ),
                    database_physical_growth_bytes_per_message=(
                        metrics.database_physical_growth_bytes_per_message
                    ),
                    database_size_cap_bytes=config.database_size_cap_bytes,
                    database_size_cap_reached=writer.database_cap_reached.is_set(),
                    process_io_scope="capture phase through writer connection close",
                    process_io_provider=metrics.process_io_provider,
                    process_io_semantics=metrics.process_io_semantics,
                    process_io_start_write_bytes=(metrics.process_io_start_write_bytes),
                    process_io_end_write_bytes=metrics.process_io_end_write_bytes,
                    process_io_delta_write_bytes=(metrics.process_io_delta_write_bytes),
                    process_io_write_bytes_per_message=(
                        metrics.process_io_write_bytes_per_message
                    ),
                    frames_per_stream_minute=metrics.frames_per_stream_minute,
                    storage_efficiency_passed=metrics.storage_efficiency_passed,
                    terminal_process_io_provider=terminal_snapshot.provider,
                    terminal_process_io_semantics=terminal_snapshot.semantics,
                    terminal_process_io_start_write_bytes=None,
                    terminal_process_io_end_write_bytes=(terminal_snapshot.write_bytes),
                    terminal_process_io_delta_write_bytes=None,
                    event_counts={},
                    symbol_event_counts={},
                    negative_corrected_latency_fraction=None,
                    audit_passed=False,
                    audit_errors=("writer_shutdown_timeout",),
                    error=errors[0] if errors else "writer_shutdown_timeout",
                    failure_class="writer",
                )

            capture_metrics = _writer_resource_metrics(
                writer,
                elapsed_seconds=elapsed_seconds,
                queue_capacity_messages=config.queue_capacity_messages,
            )
            terminal_io_start = _process_io_snapshot()
            with ImpactAbsorptionStore(
                config.database,
                memory_limit=config.duckdb_memory_limit,
                threads=config.duckdb_threads,
            ) as store:
                terminal_segment_status: Literal["valid", "invalid"] = (
                    "invalid" if errors else "valid"
                )
                for segment_id in segment_ids.values():
                    store.finish_segment(
                        run_id=run_id,
                        segment_id=segment_id,
                        status=terminal_segment_status,
                        ended_wall_ns=ended_wall_ns,
                        reason=errors[0] if errors else "",
                        invalid_event_count=1 if errors else 0,
                    )
                store.finish_run(
                    run_id=run_id,
                    status="failed" if errors else "completed",
                    ended_wall_ns=ended_wall_ns,
                    error=errors[0] if errors else "",
                )

            (
                audit,
                event_counts,
                symbol_event_counts,
                negative_latency_fraction,
            ) = _terminal_read_only_analysis(config, run_id=run_id)
            terminal_io = _process_io_interval(
                terminal_io_start,
                _process_io_snapshot(),
            )

            queue_utilization = (
                writer.high_water_messages / config.queue_capacity_messages
            )
            complete_minutes = max(1, math.floor(elapsed_seconds / 60.0))
            depth_minimum = complete_minutes * 300
            one_per_minute_minimum = complete_minutes
            feed_gates_passed = (
                not errors
                and audit.passed
                and queue_utilization <= 0.8
                and negative_latency_fraction is not None
                and negative_latency_fraction <= 0.001
                and all(
                    symbol_event_counts[symbol].get("synchronizedDepthUpdate", 0)
                    >= depth_minimum
                    and symbol_event_counts[symbol].get("bookTicker", 0)
                    >= one_per_minute_minimum
                    and symbol_event_counts[symbol].get("aggTrade", 0)
                    >= one_per_minute_minimum
                    for symbol in IMPACT_CAPTURE_SYMBOLS
                )
            )
            capture_gate_passed = (
                feed_gates_passed
                and elapsed_seconds >= _STORAGE_EFFICIENCY_MINIMUM_SECONDS
                and capture_metrics.storage_efficiency_passed
            )
            qualification_passed = (
                config.mode == "qualification"
                and elapsed_seconds >= 3_600.0
                and capture_gate_passed
            )
            status = "failed" if errors or not audit.passed else "completed"
            terminal_failure_class: CaptureFailureClass = failure_class
            if status == "failed" and terminal_failure_class == "none":
                terminal_failure_class = "audit"
            report = ImpactCaptureReport(
                run_id=run_id,
                capture_schema_version=config.schema_version,
                mode=config.mode,
                status=status,
                capture_gate_passed=capture_gate_passed,
                qualification_passed=qualification_passed,
                started_wall_ns=started_wall_ns,
                ended_wall_ns=ended_wall_ns,
                elapsed_seconds=elapsed_seconds,
                queue_high_water_messages=writer.high_water_messages,
                queue_capacity_messages=config.queue_capacity_messages,
                queue_maximum_utilization=queue_utilization,
                writer_frame_count=writer.frame_count,
                writer_message_count=writer.message_count,
                writer_compressed_payload_bytes=writer.compressed_payload_bytes,
                payload_cap_reached=writer.cap_reached.is_set(),
                database_physical_start_bytes=(
                    capture_metrics.database_physical_start_bytes
                ),
                database_physical_bytes=_database_physical_bytes(config.database),
                database_physical_growth_bytes=(
                    capture_metrics.database_physical_growth_bytes
                ),
                database_physical_growth_bytes_per_message=(
                    capture_metrics.database_physical_growth_bytes_per_message
                ),
                database_size_cap_bytes=config.database_size_cap_bytes,
                database_size_cap_reached=writer.database_cap_reached.is_set(),
                process_io_scope="capture phase through writer connection close",
                process_io_provider=capture_metrics.process_io_provider,
                process_io_semantics=capture_metrics.process_io_semantics,
                process_io_start_write_bytes=(
                    capture_metrics.process_io_start_write_bytes
                ),
                process_io_end_write_bytes=(capture_metrics.process_io_end_write_bytes),
                process_io_delta_write_bytes=(
                    capture_metrics.process_io_delta_write_bytes
                ),
                process_io_write_bytes_per_message=(
                    capture_metrics.process_io_write_bytes_per_message
                ),
                frames_per_stream_minute=(capture_metrics.frames_per_stream_minute),
                storage_efficiency_passed=(capture_metrics.storage_efficiency_passed),
                terminal_process_io_provider=terminal_io.provider,
                terminal_process_io_semantics=terminal_io.semantics,
                terminal_process_io_start_write_bytes=terminal_io.start_write_bytes,
                terminal_process_io_end_write_bytes=terminal_io.end_write_bytes,
                terminal_process_io_delta_write_bytes=terminal_io.delta_write_bytes,
                event_counts=event_counts,
                symbol_event_counts=symbol_event_counts,
                negative_corrected_latency_fraction=negative_latency_fraction,
                audit_passed=audit.passed,
                audit_errors=audit.errors,
                error=errors[0] if errors else "",
                failure_class=terminal_failure_class,
            )
            with ImpactAbsorptionStore(
                config.database,
                memory_limit=config.duckdb_memory_limit,
                threads=config.duckdb_threads,
            ) as store:
                store.record_report(
                    run_id=run_id,
                    report=report.as_dict(),
                    recorded_at_wall_ns=ended_wall_ns,
                )
            return report
        except BaseException as exc:
            stop.set()
            public_task.cancel()
            await asyncio.gather(public_task, return_exceptions=True)
            if not isinstance(
                exc, asyncio.CancelledError | KeyboardInterrupt | SystemExit
            ):
                try:
                    recovered = _terminal_post_capture_failure_report(
                        config,
                        run_id=run_id,
                        error=exc,
                        writer=writer,
                    )
                except Exception as recovery_error:
                    exc.add_note(
                        "Round 73 terminal-report recovery also failed: "
                        f"{type(recovery_error).__name__}:{recovery_error}"
                    )
                    recovered = None
                if recovered is not None:
                    return recovered
            raise


def _is_retriable_capture_failure(failure_class: CaptureFailureClass) -> bool:
    return failure_class in {
        "transport",
        "rest_transport",
        "feed_integrity",
        "processing",
    }


def _is_retriable_startup_exception(exc: Exception) -> bool:
    if isinstance(exc, _ImpactRestResponseError):
        return 500 <= exc.status_code <= 599
    return isinstance(
        exc,
        requests.exceptions.RequestException
        | WebSocketException
        | TimeoutError
        | ConnectionError,
    )


async def _sleep_before_reconnect(seconds: float) -> None:
    await asyncio.sleep(float(seconds))


async def capture_round73_supervised(
    config: ImpactCaptureConfig,
) -> ImpactCaptureSupervisorReport:
    """Retry bounded source failures without pooling disconnected evidence."""

    config.validate()
    attempts: list[ImpactCaptureReport] = []
    startup_errors: list[str] = []
    reconnect_delays: list[float] = []
    terminal_error = ""
    maximum_attempts = int(config.maximum_reconnects) + 1

    for attempt_index in range(maximum_attempts):
        try:
            report = await capture_round73(config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = f"startup:{type(exc).__name__}:{exc}"[:2_000]
            startup_errors.append(reason)
            retriable = _is_retriable_startup_exception(exc)
        else:
            attempts.append(report)
            if report.status == "completed":
                return ImpactCaptureSupervisorReport(
                    status="completed",
                    capture_schema_version=config.schema_version,
                    qualification_passed=report.qualification_passed,
                    selected_run_id=report.run_id,
                    attempt_count=attempt_index + 1,
                    reconnect_count=len(reconnect_delays),
                    reconnect_delays_seconds=tuple(reconnect_delays),
                    attempts=tuple(attempts),
                    startup_errors=tuple(startup_errors),
                    terminal_error="",
                )
            reason = report.error or "capture attempt failed without a reason"
            retriable = _is_retriable_capture_failure(report.failure_class)

        terminal_error = reason
        if attempt_index + 1 >= maximum_attempts or not retriable:
            break
        delay = _RECONNECT_BACKOFF_SECONDS[attempt_index]
        reconnect_delays.append(delay)
        await _sleep_before_reconnect(delay)

    return ImpactCaptureSupervisorReport(
        status="failed",
        capture_schema_version=config.schema_version,
        qualification_passed=False,
        selected_run_id="",
        attempt_count=len(attempts) + len(startup_errors),
        reconnect_count=len(reconnect_delays),
        reconnect_delays_seconds=tuple(reconnect_delays),
        attempts=tuple(attempts),
        startup_errors=tuple(startup_errors),
        terminal_error=terminal_error,
    )


__all__ = [
    "BINANCE_FUTURES_MARKET_STREAM",
    "BINANCE_FUTURES_PUBLIC_STREAM",
    "BINANCE_FUTURES_REST",
    "CaptureFailureClass",
    "ImpactCaptureConfig",
    "ImpactCaptureReport",
    "ImpactCaptureSupervisorReport",
    "capture_round73",
    "capture_round73_supervised",
]
