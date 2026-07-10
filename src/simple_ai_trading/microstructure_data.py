"""Real Binance futures order-book capture and HftBacktest normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
from statistics import median
import time
from typing import Mapping, Sequence

import requests

from .assets import is_supported_major_symbol, normalize_symbol


BINANCE_FUTURES_REST_URL = "https://fapi.binance.com"
BINANCE_FUTURES_PUBLIC_STREAM_URL = "wss://fstream.binance.com/public/stream"
MICROSTRUCTURE_SCHEMA_VERSION = "binance-usdm-l2-v1"
MAX_LATENCY_SAMPLES = 100_000


@dataclass(frozen=True)
class ClockSyncEvidence:
    offset_ms: float
    median_rtt_ms: float
    minimum_rtt_ms: float
    samples: int
    measured_at_ms: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolMicrostructureEvidence:
    symbol: str
    raw_path: str
    synchronized_raw_path: str
    snapshot_json_path: str
    initial_snapshot_path: str
    normalized_path: str
    raw_sha256: str
    normalized_sha256: str
    raw_bytes: int
    normalized_bytes: int
    snapshot_last_update_id: int
    tick_size: float
    lot_size: float
    raw_messages: int
    synchronized_messages: int
    normalized_rows: int
    depth_messages: int
    depth_rows: int
    trade_messages: int
    ignored_non_market_trade_messages: int
    book_ticker_messages: int
    sequence_gap_count: int
    crossed_book_count: int
    invalid_event_count: int
    first_exchange_time_ms: int | None
    last_exchange_time_ms: int | None
    feed_latency_p50_ms: float | None
    feed_latency_p95_ms: float | None
    feed_latency_p99_ms: float | None
    feed_latency_max_ms: float | None
    replay_smoke_passed: bool
    replay_first_bid: float | None
    replay_first_ask: float | None
    error: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MicrostructureCaptureResult:
    status: str
    capture_id: str
    schema_version: str
    provider: str
    market_type: str
    stream_url: str
    output_dir: str
    manifest_path: str
    started_at_ms: int
    completed_at_ms: int
    requested_duration_seconds: float
    clock_sync: ClockSyncEvidence
    symbols: tuple[str, ...]
    evidence: tuple[SymbolMicrostructureEvidence, ...]
    errors: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["clock_sync"] = self.clock_sync.asdict()
        payload["evidence"] = [item.asdict() for item in self.evidence]
        payload["symbols"] = list(self.symbols)
        payload["errors"] = list(self.errors)
        return payload


@dataclass
class _MutableSymbolStats:
    symbol: str
    raw_messages: int = 0
    depth_messages: int = 0
    depth_rows: int = 0
    trade_messages: int = 0
    ignored_non_market_trade_messages: int = 0
    book_ticker_messages: int = 0
    sequence_gap_count: int = 0
    crossed_book_count: int = 0
    invalid_event_count: int = 0
    previous_depth_update_id: int | None = None
    first_exchange_time_ms: int | None = None
    last_exchange_time_ms: int | None = None
    latency_seen: int = 0
    latency_samples_ms: list[float] = field(default_factory=list)

    def observe_latency(self, value: float, rng: random.Random) -> None:
        if not math.isfinite(value):
            return
        self.latency_seen += 1
        if len(self.latency_samples_ms) < MAX_LATENCY_SAMPLES:
            self.latency_samples_ms.append(float(value))
            return
        replacement = rng.randrange(self.latency_seen)
        if replacement < MAX_LATENCY_SAMPLES:
            self.latency_samples_ms[replacement] = float(value)


def _utc_capture_id(now_ms: int) -> str:
    stamp = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"binance-usdm-{stamp}-{secrets.token_hex(4)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def probe_binance_futures_clock(
    *,
    session: requests.Session | None = None,
    rest_url: str = BINANCE_FUTURES_REST_URL,
    samples: int = 5,
    timeout_seconds: float = 5.0,
) -> ClockSyncEvidence:
    """Estimate server-minus-host clock offset using minimum-RTT probes."""

    own_session = session is None
    active_session = session or requests.Session()
    measurements: list[tuple[float, float]] = []
    try:
        for _ in range(max(3, int(samples))):
            wall_start_ns = time.time_ns()
            monotonic_start_ns = time.perf_counter_ns()
            response = active_session.get(
                f"{rest_url.rstrip('/')}/fapi/v1/time",
                timeout=max(0.5, float(timeout_seconds)),
            )
            monotonic_end_ns = time.perf_counter_ns()
            response.raise_for_status()
            payload = response.json()
            server_time_ms = int(payload["serverTime"])
            rtt_ns = max(0, monotonic_end_ns - monotonic_start_ns)
            midpoint_wall_ns = wall_start_ns + rtt_ns // 2
            offset_ms = server_time_ms - midpoint_wall_ns / 1_000_000.0
            measurements.append((rtt_ns / 1_000_000.0, float(offset_ms)))
    finally:
        if own_session:
            active_session.close()
    best = sorted(measurements, key=lambda item: item[0])[: min(3, len(measurements))]
    return ClockSyncEvidence(
        offset_ms=float(median(item[1] for item in best)),
        median_rtt_ms=float(median(item[0] for item in measurements)),
        minimum_rtt_ms=float(min(item[0] for item in measurements)),
        samples=len(measurements),
        measured_at_ms=int(time.time() * 1000),
    )


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in symbols:
        symbol = normalize_symbol(value)
        if not is_supported_major_symbol(symbol):
            raise ValueError(f"unsupported futures microstructure symbol: {symbol}")
        if symbol not in output:
            output.append(symbol)
    if not output:
        raise ValueError("at least one BTC, ETH, or SOL USDC/USDT symbol is required")
    return tuple(output)


def _exchange_filters(payload: Mapping[str, object], symbols: Sequence[str]) -> dict[str, tuple[float, float]]:
    requested = set(symbols)
    output: dict[str, tuple[float, float]] = {}
    for item in payload.get("symbols", []):
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol", "")).upper()
        if symbol not in requested:
            continue
        tick_size = 0.0
        lot_size = 0.0
        for filter_item in item.get("filters", []):
            if not isinstance(filter_item, Mapping):
                continue
            filter_type = str(filter_item.get("filterType", ""))
            if filter_type == "PRICE_FILTER":
                tick_size = float(filter_item.get("tickSize", 0.0))
            elif filter_type == "LOT_SIZE":
                lot_size = float(filter_item.get("stepSize", 0.0))
        if tick_size > 0.0 and lot_size > 0.0:
            output[symbol] = (tick_size, lot_size)
    missing = sorted(requested - set(output))
    if missing:
        raise ValueError(f"missing exchange filters for: {','.join(missing)}")
    return output


def _fetch_exchange_filters(
    session: requests.Session,
    symbols: Sequence[str],
    *,
    rest_url: str,
    timeout_seconds: float,
) -> dict[str, tuple[float, float]]:
    response = session.get(
        f"{rest_url.rstrip('/')}/fapi/v1/exchangeInfo",
        timeout=max(1.0, float(timeout_seconds)),
    )
    response.raise_for_status()
    return _exchange_filters(response.json(), symbols)


def _fetch_depth_snapshot(
    session: requests.Session,
    symbol: str,
    *,
    rest_url: str,
    timeout_seconds: float,
) -> dict[str, object]:
    response = session.get(
        f"{rest_url.rstrip('/')}/fapi/v1/depth",
        params={"symbol": symbol, "limit": 1000},
        timeout=max(1.0, float(timeout_seconds)),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"invalid depth snapshot payload for {symbol}")
    last_update_id = int(payload.get("lastUpdateId", 0))
    bids = payload.get("bids")
    asks = payload.get("asks")
    if last_update_id <= 0 or not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        raise ValueError(f"incomplete depth snapshot for {symbol}")
    return payload


def _validate_price_levels(levels: object) -> int:
    if not isinstance(levels, list):
        raise ValueError("depth levels must be a list")
    for level in levels:
        if not isinstance(level, list) or len(level) < 2:
            raise ValueError("invalid depth level")
        price = float(level[0])
        quantity = float(level[1])
        if not math.isfinite(price) or price <= 0.0 or not math.isfinite(quantity) or quantity < 0.0:
            raise ValueError("non-finite or negative depth level")
    return len(levels)


def _observe_event(
    stats: _MutableSymbolStats,
    data: Mapping[str, object],
    *,
    receive_time_ns: int,
    clock_offset_ms: float,
    rng: random.Random,
) -> None:
    event_type = str(data.get("e", ""))
    exchange_time_ms = int(data.get("E", data.get("T", 0)) or 0)
    if exchange_time_ms > 0:
        if stats.first_exchange_time_ms is None:
            stats.first_exchange_time_ms = exchange_time_ms
        stats.last_exchange_time_ms = max(exchange_time_ms, stats.last_exchange_time_ms or exchange_time_ms)
        corrected_receive_ms = receive_time_ns / 1_000_000.0 + float(clock_offset_ms)
        stats.observe_latency(corrected_receive_ms - exchange_time_ms, rng)
    try:
        if event_type == "depthUpdate":
            first_update = int(data["U"])
            final_update = int(data["u"])
            previous_update = int(data["pu"])
            if first_update <= 0 or final_update < first_update or previous_update <= 0:
                raise ValueError("invalid depth update identifiers")
            if stats.previous_depth_update_id is not None and previous_update != stats.previous_depth_update_id:
                stats.sequence_gap_count += 1
            stats.previous_depth_update_id = final_update
            stats.depth_messages += 1
            stats.depth_rows += _validate_price_levels(data.get("b", []))
            stats.depth_rows += _validate_price_levels(data.get("a", []))
        elif event_type == "trade":
            price = float(data.get("p", 0.0))
            quantity = float(data.get("q", 0.0))
            if price <= 0.0 or quantity <= 0.0:
                if str(data.get("X", "")).upper() == "NA" and price == 0.0 and quantity == 0.0:
                    stats.ignored_non_market_trade_messages += 1
                    return
                raise ValueError("invalid trade price or quantity")
            stats.trade_messages += 1
        elif event_type == "bookTicker":
            bid = float(data.get("b", 0.0))
            ask = float(data.get("a", 0.0))
            bid_quantity = float(data.get("B", 0.0))
            ask_quantity = float(data.get("A", 0.0))
            if min(bid, ask, bid_quantity, ask_quantity) <= 0.0:
                raise ValueError("invalid book ticker")
            if bid >= ask:
                stats.crossed_book_count += 1
            stats.book_ticker_messages += 1
        else:
            raise ValueError(f"unsupported stream event: {event_type or 'missing'}")
    except (KeyError, TypeError, ValueError, OverflowError):
        stats.invalid_event_count += 1


def _stream_url(symbols: Sequence[str], base_url: str) -> str:
    streams: list[str] = []
    for symbol in symbols:
        lower = symbol.lower()
        streams.extend((f"{lower}@depth@100ms", f"{lower}@bookTicker", f"{lower}@trade"))
    return f"{base_url.rstrip('/')}?streams={'/'.join(streams)}"


def _initial_snapshot_array(payload: Mapping[str, object]):
    try:
        import numpy as np
        from hftbacktest import (
            BUY_EVENT,
            DEPTH_SNAPSHOT_EVENT,
            EXCH_EVENT,
            LOCAL_EVENT,
            SELL_EVENT,
            event_dtype,
        )
    except ImportError as exc:  # pragma: no cover - exercised by dependency readiness checks
        raise RuntimeError("hftbacktest is required for normalized L2 replay data") from exc
    bids = payload.get("bids", [])
    asks = payload.get("asks", [])
    _validate_price_levels(bids)
    _validate_price_levels(asks)
    values = np.empty(len(bids) + len(asks), dtype=event_dtype)
    index = 0
    for side_event, levels in ((BUY_EVENT, bids), (SELL_EVENT, asks)):
        for level in levels:
            values[index] = (
                DEPTH_SNAPSHOT_EVENT | side_event | EXCH_EVENT | LOCAL_EVENT,
                0,
                0,
                float(level[0]),
                float(level[1]),
                0,
                0,
                0.0,
            )
            index += 1
    return values


def _synchronize_raw_capture(
    raw_path: Path,
    output_path: Path,
    *,
    snapshot_last_update_id: int,
) -> tuple[int, int]:
    """Drop pre-snapshot events and require the first depth update to bridge the snapshot."""

    snapshot_update = int(snapshot_last_update_id)
    synchronized = False
    previous_update: int | None = None
    message_count = 0
    normalized_row_estimate = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as source, gzip.open(
        output_path,
        "wt",
        encoding="utf-8",
        compresslevel=6,
        newline="\n",
    ) as target:
        for line in source:
            stripped = line.rstrip("\n")
            try:
                _received, raw_json = stripped.split(" ", 1)
                payload = json.loads(raw_json)
                data = payload.get("data", payload)
                event_type = str(data.get("e", ""))
            except (AttributeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid raw capture line: {exc}") from exc
            if event_type == "depthUpdate":
                first_update = int(data["U"])
                final_update = int(data["u"])
                previous = int(data["pu"])
                if not synchronized:
                    if final_update <= snapshot_update:
                        continue
                    overlaps_snapshot = first_update <= snapshot_update <= final_update
                    immediately_follows_snapshot = previous == snapshot_update
                    if not (overlaps_snapshot or immediately_follows_snapshot):
                        raise ValueError(
                            "depth snapshot bridge missing: "
                            f"snapshot={snapshot_update} first={first_update} final={final_update} previous={previous}"
                        )
                    synchronized = True
                elif previous_update is not None and previous != previous_update:
                    raise ValueError(
                        f"depth sequence gap after snapshot: expected_previous={previous_update} actual_previous={previous}"
                    )
                previous_update = final_update
                normalized_row_estimate += len(data.get("b", [])) + len(data.get("a", []))
            elif not synchronized:
                continue
            elif event_type == "trade":
                price = float(data.get("p", 0.0))
                quantity = float(data.get("q", 0.0))
                if (
                    price == 0.0
                    and quantity == 0.0
                    and str(data.get("X", "")).upper() == "NA"
                ):
                    continue
                normalized_row_estimate += 1
            elif event_type == "bookTicker":
                normalized_row_estimate += 2
            target.write(stripped + "\n")
            message_count += 1
    if not synchronized:
        raise ValueError("capture ended before a depth event bridged the REST snapshot")
    return message_count, normalized_row_estimate


def _convert_to_hftbacktest(
    synchronized_path: Path,
    normalized_path: Path,
    snapshot_path: Path,
    snapshot_payload: Mapping[str, object],
    *,
    row_estimate: int,
) -> int:
    try:
        import numpy as np
        from hftbacktest.data.utils import binancefutures
    except ImportError as exc:  # pragma: no cover - exercised by dependency readiness checks
        raise RuntimeError("hftbacktest is required for normalized L2 replay data") from exc
    snapshot = _initial_snapshot_array(snapshot_payload)
    np.savez_compressed(snapshot_path, data=snapshot)
    buffer_size = max(10_000, int(max(1, row_estimate) * 1.15) + 10_000)
    converted = binancefutures.convert(
        str(synchronized_path),
        output_filename=str(normalized_path),
        opt="t",
        combined_stream=True,
        buffer_size=buffer_size,
    )
    return int(len(converted))


def _replay_smoke(
    normalized_path: Path,
    snapshot_path: Path,
    *,
    tick_size: float,
    lot_size: float,
) -> tuple[bool, float | None, float | None]:
    try:
        from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("hftbacktest is required for L2 replay validation") from exc
    asset = (
        BacktestAsset()
        .data([str(normalized_path)])
        .initial_snapshot(str(snapshot_path))
        .linear_asset(1.0)
        .constant_latency(10_000_000, 10_000_000)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0002, 0.0005)
        .tick_size(float(tick_size))
        .lot_size(float(lot_size))
        .last_trades_capacity(0)
    )
    replay = HashMapMarketDepthBacktest([asset])
    try:
        replay.elapse(1_000_000_000)
        depth = replay.depth(0)
        bid = float(depth.best_bid)
        ask = float(depth.best_ask)
        passed = math.isfinite(bid) and math.isfinite(ask) and bid > 0.0 and ask > bid
        return passed, (bid if math.isfinite(bid) else None), (ask if math.isfinite(ask) else None)
    finally:
        replay.close()


def capture_binance_futures_microstructure(
    symbols: Sequence[str],
    *,
    duration_seconds: float = 60.0,
    output_root: str | Path = "data/microstructure",
    rest_url: str = BINANCE_FUTURES_REST_URL,
    stream_base_url: str = BINANCE_FUTURES_PUBLIC_STREAM_URL,
    timeout_seconds: float = 10.0,
    convert: bool = True,
    capture_id: str | None = None,
) -> MicrostructureCaptureResult:
    """Capture real L2/trade/BBO events and produce promotion-grade replay evidence."""

    from websockets.sync.client import connect

    normalized_symbols = _normalize_symbols(symbols)
    requested_duration = float(duration_seconds)
    if not math.isfinite(requested_duration) or requested_duration < 1.0 or requested_duration > 86_400.0:
        raise ValueError("duration_seconds must be between 1 and 86400")
    started_at_ms = int(time.time() * 1000)
    resolved_capture_id = capture_id or _utc_capture_id(started_at_ms)
    capture_dir = Path(output_root) / resolved_capture_id
    capture_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = capture_dir / "manifest.json"
    errors: list[str] = []
    stats = {symbol: _MutableSymbolStats(symbol) for symbol in normalized_symbols}
    raw_paths = {symbol: capture_dir / f"{symbol.lower()}.raw.jsonl.gz" for symbol in normalized_symbols}
    synchronized_paths = {
        symbol: capture_dir / f"{symbol.lower()}.synchronized.jsonl.gz" for symbol in normalized_symbols
    }
    snapshot_paths = {symbol: capture_dir / f"{symbol.lower()}.initial-depth.json" for symbol in normalized_symbols}
    initial_snapshot_paths = {
        symbol: capture_dir / f"{symbol.lower()}.initial-depth.npz" for symbol in normalized_symbols
    }
    normalized_paths = {symbol: capture_dir / f"{symbol.lower()}.hft.npz" for symbol in normalized_symbols}
    stream_url = _stream_url(normalized_symbols, stream_base_url)
    rng = random.Random(0)
    snapshots: dict[str, dict[str, object]] = {}
    filters: dict[str, tuple[float, float]] = {}
    clock_sync: ClockSyncEvidence | None = None
    writers: dict[str, object] = {}
    session = requests.Session()
    try:
        clock_sync = probe_binance_futures_clock(
            session=session,
            rest_url=rest_url,
            timeout_seconds=timeout_seconds,
        )
        filters = _fetch_exchange_filters(
            session,
            normalized_symbols,
            rest_url=rest_url,
            timeout_seconds=timeout_seconds,
        )
        for symbol, path in raw_paths.items():
            writers[symbol] = gzip.open(path, "wt", encoding="utf-8", compresslevel=6, newline="\n")
        with connect(
            stream_url,
            open_timeout=max(1.0, float(timeout_seconds)),
            close_timeout=3.0,
            ping_interval=15.0,
            ping_timeout=max(5.0, float(timeout_seconds)),
            max_size=16_000_000,
            max_queue=4096,
        ) as websocket:
            for symbol in normalized_symbols:
                snapshot = _fetch_depth_snapshot(
                    session,
                    symbol,
                    rest_url=rest_url,
                    timeout_seconds=timeout_seconds,
                )
                snapshots[symbol] = snapshot
                _write_json(snapshot_paths[symbol], snapshot)
            deadline = time.perf_counter() + requested_duration
            while time.perf_counter() < deadline:
                remaining = max(0.05, min(1.0, deadline - time.perf_counter()))
                try:
                    raw_message = websocket.recv(timeout=remaining)
                except TimeoutError:
                    continue
                receive_time_ns = time.time_ns()
                if isinstance(raw_message, bytes):
                    raw_text = raw_message.decode("utf-8")
                else:
                    raw_text = str(raw_message)
                try:
                    payload = json.loads(raw_text)
                    data = payload.get("data", payload)
                    symbol = str(data.get("s", "")).upper()
                except (AttributeError, json.JSONDecodeError):
                    errors.append("invalid_websocket_json")
                    continue
                if symbol not in stats:
                    errors.append(f"unexpected_stream_symbol:{symbol or 'missing'}")
                    continue
                stats[symbol].raw_messages += 1
                _observe_event(
                    stats[symbol],
                    data,
                    receive_time_ns=receive_time_ns,
                    clock_offset_ms=clock_sync.offset_ms,
                    rng=rng,
                )
                writer = writers[symbol]
                writer.write(f"{receive_time_ns} {raw_text}\n")  # type: ignore[attr-defined]
    except Exception as exc:  # capture failures are persisted and fail closed
        errors.append(f"capture:{type(exc).__name__}:{str(exc)[:500]}")
    finally:
        for writer in writers.values():
            writer.close()  # type: ignore[attr-defined]
        session.close()

    evidence: list[SymbolMicrostructureEvidence] = []
    for symbol in normalized_symbols:
        item_errors: list[str] = []
        synchronized_messages = 0
        normalized_rows = 0
        replay_passed = False
        replay_bid: float | None = None
        replay_ask: float | None = None
        snapshot = snapshots.get(symbol)
        tick_size, lot_size = filters.get(symbol, (0.0, 0.0))
        if stats[symbol].raw_messages <= 0:
            item_errors.append("no_raw_messages")
        if snapshot is None:
            item_errors.append("missing_initial_snapshot")
        if stats[symbol].sequence_gap_count > 0:
            item_errors.append(f"depth_sequence_gaps:{stats[symbol].sequence_gap_count}")
        if stats[symbol].crossed_book_count > 0:
            item_errors.append(f"crossed_books:{stats[symbol].crossed_book_count}")
        if stats[symbol].invalid_event_count > 0:
            item_errors.append(f"invalid_events:{stats[symbol].invalid_event_count}")
        if snapshot is not None and stats[symbol].raw_messages > 0:
            try:
                synchronized_messages, row_estimate = _synchronize_raw_capture(
                    raw_paths[symbol],
                    synchronized_paths[symbol],
                    snapshot_last_update_id=int(snapshot["lastUpdateId"]),
                )
                if convert:
                    normalized_rows = _convert_to_hftbacktest(
                        synchronized_paths[symbol],
                        normalized_paths[symbol],
                        initial_snapshot_paths[symbol],
                        snapshot,
                        row_estimate=row_estimate,
                    )
                    replay_passed, replay_bid, replay_ask = _replay_smoke(
                        normalized_paths[symbol],
                        initial_snapshot_paths[symbol],
                        tick_size=tick_size,
                        lot_size=lot_size,
                    )
                    if not replay_passed:
                        item_errors.append("hftbacktest_replay_smoke_failed")
            except Exception as exc:
                item_errors.append(f"normalization:{type(exc).__name__}:{str(exc)[:500]}")
        latencies = stats[symbol].latency_samples_ms
        raw_path = raw_paths[symbol]
        normalized_path = normalized_paths[symbol]
        evidence.append(
            SymbolMicrostructureEvidence(
                symbol=symbol,
                raw_path=str(raw_path),
                synchronized_raw_path=str(synchronized_paths[symbol]),
                snapshot_json_path=str(snapshot_paths[symbol]),
                initial_snapshot_path=(str(initial_snapshot_paths[symbol]) if convert else ""),
                normalized_path=(str(normalized_path) if convert else ""),
                raw_sha256=(_sha256(raw_path) if raw_path.exists() else ""),
                normalized_sha256=(_sha256(normalized_path) if normalized_path.exists() else ""),
                raw_bytes=(raw_path.stat().st_size if raw_path.exists() else 0),
                normalized_bytes=(normalized_path.stat().st_size if normalized_path.exists() else 0),
                snapshot_last_update_id=int(snapshot.get("lastUpdateId", 0)) if snapshot else 0,
                tick_size=float(tick_size),
                lot_size=float(lot_size),
                raw_messages=int(stats[symbol].raw_messages),
                synchronized_messages=int(synchronized_messages),
                normalized_rows=int(normalized_rows),
                depth_messages=int(stats[symbol].depth_messages),
                depth_rows=int(stats[symbol].depth_rows),
                trade_messages=int(stats[symbol].trade_messages),
                ignored_non_market_trade_messages=int(stats[symbol].ignored_non_market_trade_messages),
                book_ticker_messages=int(stats[symbol].book_ticker_messages),
                sequence_gap_count=int(stats[symbol].sequence_gap_count),
                crossed_book_count=int(stats[symbol].crossed_book_count),
                invalid_event_count=int(stats[symbol].invalid_event_count),
                first_exchange_time_ms=stats[symbol].first_exchange_time_ms,
                last_exchange_time_ms=stats[symbol].last_exchange_time_ms,
                feed_latency_p50_ms=_percentile(latencies, 0.50),
                feed_latency_p95_ms=_percentile(latencies, 0.95),
                feed_latency_p99_ms=_percentile(latencies, 0.99),
                feed_latency_max_ms=(max(latencies) if latencies else None),
                replay_smoke_passed=bool(replay_passed),
                replay_first_bid=replay_bid,
                replay_first_ask=replay_ask,
                error="; ".join(item_errors),
            )
        )
    if clock_sync is None:
        clock_sync = ClockSyncEvidence(0.0, 0.0, 0.0, 0, started_at_ms)
    completed_at_ms = int(time.time() * 1000)
    passed = not errors and all(not item.error for item in evidence)
    if not convert:
        passed = passed and all(item.raw_messages > 0 and item.sequence_gap_count == 0 for item in evidence)
    result = MicrostructureCaptureResult(
        status="pass" if passed else "fail",
        capture_id=resolved_capture_id,
        schema_version=MICROSTRUCTURE_SCHEMA_VERSION,
        provider="binance",
        market_type="futures",
        stream_url=stream_url,
        output_dir=str(capture_dir),
        manifest_path=str(manifest_path),
        started_at_ms=started_at_ms,
        completed_at_ms=completed_at_ms,
        requested_duration_seconds=requested_duration,
        clock_sync=clock_sync,
        symbols=normalized_symbols,
        evidence=tuple(evidence),
        errors=tuple(errors),
    )
    _write_json(manifest_path, result.asdict())
    return result
