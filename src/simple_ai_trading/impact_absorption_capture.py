"""Bounded prospective Binance USD-M capture for Round 73 research."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import math
from queue import Empty, Full, Queue
import threading
import time
from typing import Literal, Mapping, Sequence
import uuid

import requests
from websockets.asyncio.client import connect

from .impact_absorption import (
    ImpactFeedIntegrityError,
    ROUND73_DESIGN_SHA256,
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_CONTRACT_SHA256,
    IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES,
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
    ImpactFrameWriteResult,
    ImpactRestEvent,
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
_FRAME_MESSAGE_LIMIT = 1_024
_FRAME_UNCOMPRESSED_LIMIT = 8 * 1024 * 1024
_FRAME_FLUSH_SECONDS = 0.250
_QUEUE_CAPACITY = 65_536
_WRITER_STALL_SECONDS = 5.0
_SOURCE_STALL_SECONDS = 15.0
_CLOCK_PROBE_COUNT = 3
_RATE_LIMIT_FRACTION = 0.80


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


@dataclass(frozen=True)
class ImpactCaptureConfig:
    """Resource and authority bounds for one prospective capture."""

    database: str = "data/microstructure.duckdb"
    mode: Literal["probe", "qualification"] = "probe"
    duration_seconds: float = 30.0
    request_timeout_seconds: float = 10.0
    queue_capacity_messages: int = _QUEUE_CAPACITY
    frame_message_limit: int = _FRAME_MESSAGE_LIMIT
    frame_uncompressed_limit_bytes: int = _FRAME_UNCOMPRESSED_LIMIT
    frame_flush_seconds: float = _FRAME_FLUSH_SECONDS
    compressed_payload_cap_bytes: int = IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES
    writer_stall_seconds: float = _WRITER_STALL_SECONDS
    source_stall_seconds: float = _SOURCE_STALL_SECONDS
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = 2

    def validate(self) -> None:
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
        if not 1.0 <= float(self.writer_stall_seconds) <= _WRITER_STALL_SECONDS:
            raise ValueError("writer stall timeout is outside the frozen bound")
        if not 5.0 <= float(self.source_stall_seconds) <= _SOURCE_STALL_SECONDS:
            raise ValueError("source stall timeout is outside the frozen bound")


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
    mode: str
    status: str
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
    event_counts: dict[str, int]
    symbol_event_counts: dict[str, dict[str, int]]
    negative_corrected_latency_fraction: float | None
    audit_passed: bool
    audit_errors: tuple[str, ...]
    error: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["schema_version"] = "round-073-capture-report-v1"
        result["design_sha256"] = ROUND73_DESIGN_SHA256
        result["capture_contract_sha256"] = IMPACT_CAPTURE_CONTRACT_SHA256
        result["audit_errors"] = list(self.audit_errors)
        return result


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
        self.error = ""
        self.last_progress_monotonic = time.monotonic()
        self.high_water_messages = 0
        self.frame_count = 0
        self.message_count = 0
        self.compressed_payload_bytes = 0

    def start(self, timeout_seconds: float = 5.0) -> None:
        self.thread.start()
        if not self.started.wait(timeout=max(0.1, float(timeout_seconds))):
            raise RuntimeError("Round 73 writer did not start within its deadline")
        if self.failed.is_set():
            raise RuntimeError(self.error or "Round 73 writer failed during startup")

    def put(self, message: ImpactCaptureMessage) -> None:
        if self.failed.is_set():
            raise RuntimeError(self.error or "Round 73 writer has failed")
        if self.cap_reached.is_set():
            raise RuntimeError("Round 73 compressed payload cap was reached")
        try:
            self.queue.put_nowait(message)
        except Full as exc:
            raise RuntimeError("Round 73 writer queue overflow") from exc
        self.high_water_messages = max(self.high_water_messages, self.queue.qsize())

    def stop(self, timeout_seconds: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            try:
                self.queue.put(self._STOP, timeout=0.1)
                break
            except Full:
                if self.failed.is_set():
                    break
        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self.thread.is_alive()

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
                    if self.cap_reached.is_set():
                        break
        except BaseException as exc:
            self.error = f"{type(exc).__name__}:{exc}"[:2_000]
            self.failed.set()
            self.started.set()
        finally:
            self.stopped.set()


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
        raise RuntimeError(
            f"public Binance REST {path} returned HTTP {response.status_code}"
        )
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


async def capture_round73(config: ImpactCaptureConfig) -> ImpactCaptureReport:
    """Capture one bounded prospective run; never place or authenticate an order."""

    config.validate()
    run_id = uuid.uuid4().hex
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.perf_counter_ns()
    public_queue: asyncio.Queue[_WireReceipt] = asyncio.Queue(
        maxsize=int(config.queue_capacity_messages)
    )
    fatal = asyncio.Event()
    stop = asyncio.Event()
    errors: list[str] = []
    public_last_receipt = time.monotonic()
    market_last_receipt = time.monotonic()
    public_sequence = 0
    market_sequence = 0

    def fail(reason: str) -> None:
        if not errors:
            errors.append(str(reason)[:2_000])
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
            fail(f"public_source:{type(exc).__name__}:{exc}")

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
                raise RuntimeError(
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
                    },
                    compressed_payload_cap_bytes=config.compressed_payload_cap_bytes,
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
                        try:
                            _stream, payload = _combined_payload(receipt.raw_text)
                            event_type = str(payload.get("e", ""))
                            symbol = str(payload.get("s", ""))
                            if symbol not in segment_ids:
                                raise ImpactFeedIntegrityError(
                                    "public stream symbol is outside Round 73"
                                )
                            check_event_clock(
                                "public", event_type, symbol, int(payload.get("E", 0))
                            )
                            record = ImpactCaptureFrameRecord(
                                stream="binance_futures_public",
                                connection_id=f"binance-public:{run_id}",
                                sequence_number=receipt.sequence_number,
                                received_wall_ns=receipt.received_wall_ns,
                                received_monotonic_ns=receipt.received_monotonic_ns,
                                raw_text=receipt.raw_text,
                            )
                            if event_type == "depthUpdate":
                                event = books[symbol].apply(
                                    payload,
                                    receive_time_ns=receipt.received_monotonic_ns,
                                )
                                state = None if event.stale else books[symbol].state()
                                message = ImpactCaptureMessage(
                                    record=record,
                                    event=event,
                                    segment_id=segment_ids[symbol],
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
                            writer.put(message)
                        finally:
                            public_queue.task_done()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    fail(f"public_processor:{type(exc).__name__}:{exc}")

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
                            _stream, payload = _combined_payload(raw)
                            event_type = str(payload.get("e", ""))
                            symbol = (
                                str(payload.get("s", ""))
                                if event_type != "forceOrder"
                                else str(
                                    payload.get("o", {}).get("s", "")
                                    if isinstance(payload.get("o"), Mapping)
                                    else ""
                                )
                            )
                            sequence = market_sequence
                            market_sequence += 1
                            if symbol not in segment_ids:
                                raise ImpactFeedIntegrityError(
                                    "market stream symbol is outside Round 73"
                                )
                            check_event_clock(
                                "market", event_type, symbol, int(payload.get("E", 0))
                            )
                            record = ImpactCaptureFrameRecord(
                                stream="binance_futures_market",
                                connection_id=f"binance-market:{run_id}",
                                sequence_number=sequence,
                                received_wall_ns=received_wall_ns,
                                received_monotonic_ns=received_monotonic_ns,
                                raw_text=raw,
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
                                    f"unsupported market event: {event_type or 'missing'}"
                                )
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
                    fail(f"market_source:{type(exc).__name__}:{exc}")

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
                            raise RuntimeError(
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
                                raise RuntimeError(
                                    "one-minute request weight reached the 80% capture stop gate"
                                )
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    fail(f"rest_poller:{type(exc).__name__}:{exc}")

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
                        fail(writer.error or "writer_failed")
                    elif writer.cap_reached.is_set():
                        fail("compressed_payload_cap_reached")
                    elif (
                        writer.queue.qsize() > 0
                        and now - writer.last_progress_monotonic
                        > float(config.writer_stall_seconds)
                    ):
                        fail("writer_heartbeat_timeout")
                    elif now - public_last_receipt > float(config.source_stall_seconds):
                        fail("public_source_stall")
                    elif now - market_last_receipt > float(config.source_stall_seconds):
                        fail("market_source_stall")
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
                    fail("public_processing_drain_timeout")
                    processor_task.cancel()
                await asyncio.gather(processor_task, return_exceptions=True)
                writer_joined = writer.stop(timeout_seconds=10.0)
                if not writer_joined:
                    fail("writer_shutdown_timeout")
                if writer.failed.is_set():
                    fail(writer.error or "writer_failed")

            ended_wall_ns = time.time_ns()
            elapsed_seconds = max(0.0, time.monotonic() - capture_started)
            if writer.thread.is_alive():
                return ImpactCaptureReport(
                    run_id=run_id,
                    mode=config.mode,
                    status="failed",
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
                    event_counts={},
                    symbol_event_counts={},
                    negative_corrected_latency_fraction=None,
                    audit_passed=False,
                    audit_errors=("writer_shutdown_timeout",),
                    error=errors[0] if errors else "writer_shutdown_timeout",
                )

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
                audit = store.audit_run(run_id)
                event_counts = {
                    str(event_type): int(count)
                    for event_type, count in store.connect()
                    .execute(
                        """
                        SELECT event_type, count(*) FROM impact_event_index
                        WHERE run_id = ? GROUP BY event_type ORDER BY event_type
                        """,
                        [run_id],
                    )
                    .fetchall()
                }
                symbol_event_counts: dict[str, dict[str, int]] = {
                    symbol: {} for symbol in IMPACT_CAPTURE_SYMBOLS
                }
                for symbol, event_type, count in (
                    store.connect()
                    .execute(
                        """
                    SELECT symbol, event_type, count(*) FROM impact_event_index
                    WHERE run_id = ? AND symbol <> ''
                    GROUP BY symbol, event_type ORDER BY symbol, event_type
                    """,
                        [run_id],
                    )
                    .fetchall()
                ):
                    symbol_event_counts[str(symbol)][str(event_type)] = int(count)
                for symbol, count in (
                    store.connect()
                    .execute(
                        """
                    SELECT symbol, count(*) FROM impact_depth_update
                    WHERE run_id = ? AND stale = false
                    GROUP BY symbol ORDER BY symbol
                    """,
                        [run_id],
                    )
                    .fetchall()
                ):
                    symbol_event_counts[str(symbol)]["synchronizedDepthUpdate"] = int(
                        count
                    )
                latency_rows = (
                    store.connect()
                    .execute(
                        """
                    SELECT received_wall_ns, event_time_ms
                    FROM impact_event_index
                    WHERE run_id = ? AND event_time_ms IS NOT NULL
                      AND stream <> 'binance_futures_rest'
                    ORDER BY received_wall_ns
                    """,
                        [run_id],
                    )
                    .fetchall()
                )
                clock_rows = (
                    store.connect()
                    .execute(
                        """
                    SELECT e.received_wall_ns, r.request_started_wall_ns,
                           e.received_monotonic_ns,
                           r.request_started_monotonic_ns, r.exchange_time_ms
                    FROM impact_event_index e
                    JOIN impact_rest_event r USING (run_id, frame_index, message_index)
                    WHERE e.run_id = ? AND e.event_type = 'serverTime'
                    ORDER BY e.received_wall_ns
                    """,
                        [run_id],
                    )
                    .fetchall()
                )
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
                        midpoint = (
                            int(request_started_wall_ns) + int(received_wall_ns)
                        ) // 2
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

            queue_utilization = (
                writer.high_water_messages / config.queue_capacity_messages
            )
            complete_minutes = max(1, math.floor(elapsed_seconds / 60.0))
            depth_minimum = complete_minutes * 300
            one_per_minute_minimum = complete_minutes
            qualification_passed = (
                config.mode == "qualification"
                and not errors
                and audit.passed
                and elapsed_seconds >= 3_600.0
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
            status = "failed" if errors or not audit.passed else "completed"
            report = ImpactCaptureReport(
                run_id=run_id,
                mode=config.mode,
                status=status,
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
                event_counts=event_counts,
                symbol_event_counts=symbol_event_counts,
                negative_corrected_latency_fraction=negative_latency_fraction,
                audit_passed=audit.passed,
                audit_errors=audit.errors,
                error=errors[0] if errors else "",
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
        except BaseException:
            stop.set()
            public_task.cancel()
            await asyncio.gather(public_task, return_exceptions=True)
            raise


__all__ = [
    "BINANCE_FUTURES_MARKET_STREAM",
    "BINANCE_FUTURES_PUBLIC_STREAM",
    "BINANCE_FUTURES_REST",
    "ImpactCaptureConfig",
    "ImpactCaptureReport",
    "capture_round73",
]
