"""Strict public Polymarket historical order-book ingestion primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import time
from typing import Protocol
from urllib.parse import urlparse

import requests
import zstandard


POLYMARKET_ORDERBOOK_HISTORY_URL = "https://clob.polymarket.com/orderbook-history"
POLYMARKET_HISTORICAL_L2_SCHEMA_VERSION = "polymarket-historical-l2-window-v1"
POLYMARKET_HISTORICAL_L2_CODEC = "canonical-json-zstd-3"

_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_BOOK_HASH = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "poly_address",
        "poly_signature",
        "poly_timestamp",
        "poly_nonce",
        "poly_api_key",
        "poly_passphrase",
    }
)
_PAGE_FIELDS = frozenset(("count", "data"))
_SNAPSHOT_FIELDS = frozenset(
    (
        "market",
        "asset_id",
        "timestamp",
        "hash",
        "bids",
        "asks",
        "min_order_size",
        "tick_size",
        "neg_risk",
        "last_trade_price",
    )
)
_LEVEL_FIELDS = frozenset(("price", "size"))
_ARCHIVED_SNAPSHOT_FIELDS = frozenset(
    (
        "asks",
        "asset_id",
        "bids",
        "book_hash",
        "condition_id",
        "last_trade_price",
        "minimum_order_size",
        "negative_risk",
        "source_payload_sha256",
        "tick_size",
        "timestamp_ms",
    )
)
_ARCHIVED_WINDOW_FIELDS = frozenset(
    (
        "asset_id",
        "authority",
        "condition_id",
        "event_end_ms",
        "event_start_ms",
        "schema_version",
        "snapshots",
        "source",
    )
)
_MAXIMUM_RESPONSE_BYTES = 32 * 1024 * 1024
_MAXIMUM_PAGE_RECORDS = 1_000
_MAXIMUM_LEVELS_PER_SIDE = 1_000
_MAXIMUM_WINDOW_RECORDS = 20_000
_MINIMUM_CLOSED_AGE_MS = 10 * 60 * 1_000
_RETRYABLE_STATUS_CODES = frozenset((425, 429, 500, 502, 503, 504))


class _Response(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]
    url: str


class _Session(Protocol):
    headers: Mapping[str, str]
    cookies: object

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> _Response: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("historical order-book JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"historical order-book JSON contains {value}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decimal_string(value: object, *, name: str, allow_zero: bool) -> str:
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise ValueError(f"{name} is not a canonical nonnegative decimal")
    try:
        selected = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} is not a decimal") from exc
    if not selected.is_finite() or selected < 0 or (not allow_zero and selected == 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


@dataclass(frozen=True, slots=True)
class HistoricalBookLevel:
    price: str
    size: str

    def as_dict(self) -> dict[str, str]:
        return {"price": self.price, "size": self.size}


@dataclass(frozen=True, slots=True)
class HistoricalBookSnapshot:
    condition_id: str
    asset_id: str
    timestamp_ms: int
    book_hash: str
    bids: tuple[HistoricalBookLevel, ...]
    asks: tuple[HistoricalBookLevel, ...]
    minimum_order_size: str
    tick_size: str
    negative_risk: bool
    last_trade_price: str
    source_payload_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "asks": [level.as_dict() for level in self.asks],
            "asset_id": self.asset_id,
            "bids": [level.as_dict() for level in self.bids],
            "book_hash": self.book_hash,
            "condition_id": self.condition_id,
            "last_trade_price": self.last_trade_price,
            "minimum_order_size": self.minimum_order_size,
            "negative_risk": self.negative_risk,
            "source_payload_sha256": self.source_payload_sha256,
            "tick_size": self.tick_size,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(frozen=True, slots=True)
class HistoricalOrderbookPage:
    remaining_record_count: int
    snapshots: tuple[HistoricalBookSnapshot, ...]
    source_payload_sha256: str


@dataclass(frozen=True, slots=True)
class HistoricalL2Window:
    condition_id: str
    asset_id: str
    event_start_ms: int
    event_end_ms: int
    snapshots: tuple[HistoricalBookSnapshot, ...]
    source_chain_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "authority": {
                "execution_calibration": False,
                "fill_claim": False,
                "live_trading": False,
                "model_data_eligible": False,
                "paper_trading": False,
                "profitability_claim": False,
                "queue_position_calibration": False,
                "transport_latency_calibration": False,
            },
            "condition_id": self.condition_id,
            "event_end_ms": self.event_end_ms,
            "event_start_ms": self.event_start_ms,
            "schema_version": POLYMARKET_HISTORICAL_L2_SCHEMA_VERSION,
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "source": {
                "authentication_used": False,
                "binance_used": False,
                "endpoint": POLYMARKET_ORDERBOOK_HISTORY_URL,
                "source_chain_sha256": self.source_chain_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class HistoricalL2Chunk:
    codec: str
    record_count: int
    raw_size_bytes: int
    compressed_size_bytes: int
    raw_sha256: str
    compressed_sha256: str
    payload: bytes


def _parse_level(value: object, *, name: str) -> HistoricalBookLevel:
    if not isinstance(value, Mapping) or set(value) != _LEVEL_FIELDS:
        raise ValueError(f"{name} has an unexpected schema")
    price = _decimal_string(value.get("price"), name=f"{name}.price", allow_zero=False)
    size = _decimal_string(value.get("size"), name=f"{name}.size", allow_zero=False)
    if not Decimal(0) < Decimal(price) < Decimal(1):
        raise ValueError(f"{name}.price is outside the binary-contract range")
    return HistoricalBookLevel(price=price, size=size)


def _parse_side(
    value: object,
    *,
    name: str,
    reverse: bool,
) -> tuple[HistoricalBookLevel, ...]:
    if not isinstance(value, list) or len(value) > _MAXIMUM_LEVELS_PER_SIDE:
        raise ValueError(f"{name} is not a bounded level list")
    levels = tuple(
        _parse_level(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )
    prices = [Decimal(level.price) for level in levels]
    if len(set(prices)) != len(prices):
        raise ValueError(f"{name} contains duplicate prices")
    return tuple(sorted(levels, key=lambda item: Decimal(item.price), reverse=reverse))


def _parse_snapshot(
    value: object,
    *,
    expected_condition_id: str,
    expected_asset_id: str,
    start_ms: int,
    end_ms: int,
) -> HistoricalBookSnapshot:
    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_FIELDS:
        raise ValueError("historical order-book snapshot has an unexpected schema")
    condition_id = str(value.get("market") or "").strip().lower()
    asset_id = str(value.get("asset_id") or "").strip()
    timestamp_text = str(value.get("timestamp") or "").strip()
    book_hash = str(value.get("hash") or "").strip().lower()
    if (
        condition_id != expected_condition_id
        or _CONDITION_ID.fullmatch(condition_id) is None
        or asset_id != expected_asset_id
        or _TOKEN_ID.fullmatch(asset_id) is None
        or not timestamp_text.isdigit()
        or _BOOK_HASH.fullmatch(book_hash) is None
    ):
        raise ValueError("historical order-book snapshot identity differs")
    timestamp_ms = int(timestamp_text)
    if not start_ms <= timestamp_ms < end_ms:
        raise ValueError(
            "historical order-book snapshot is outside the requested window"
        )
    bids = _parse_side(value.get("bids"), name="bids", reverse=True)
    asks = _parse_side(value.get("asks"), name="asks", reverse=False)
    if bids and asks and Decimal(bids[0].price) >= Decimal(asks[0].price):
        raise ValueError("historical order-book snapshot is crossed")
    minimum_order_size = _decimal_string(
        value.get("min_order_size"),
        name="minimum_order_size",
        allow_zero=False,
    )
    tick_size = _decimal_string(
        value.get("tick_size"),
        name="tick_size",
        allow_zero=False,
    )
    if not Decimal(0) < Decimal(tick_size) < Decimal(1):
        raise ValueError("tick_size is outside the binary-contract range")
    negative_risk = value.get("neg_risk")
    if type(negative_risk) is not bool:
        raise ValueError("negative_risk is not boolean")
    last_trade = value.get("last_trade_price")
    if last_trade != "":
        last_trade = _decimal_string(
            last_trade,
            name="last_trade_price",
            allow_zero=True,
        )
        if not Decimal(0) <= Decimal(last_trade) <= Decimal(1):
            raise ValueError("last_trade_price is outside the binary-contract range")
    raw_sha = _sha256(_canonical_json(value))
    return HistoricalBookSnapshot(
        condition_id=condition_id,
        asset_id=asset_id,
        timestamp_ms=timestamp_ms,
        book_hash=book_hash,
        bids=bids,
        asks=asks,
        minimum_order_size=minimum_order_size,
        tick_size=tick_size,
        negative_risk=negative_risk,
        last_trade_price=str(last_trade),
        source_payload_sha256=raw_sha,
    )


def parse_historical_orderbook_page(
    raw: bytes,
    *,
    expected_condition_id: str,
    expected_asset_id: str,
    start_ms: int,
    end_ms: int,
) -> HistoricalOrderbookPage:
    if not isinstance(raw, bytes) or not 2 <= len(raw) <= _MAXIMUM_RESPONSE_BYTES:
        raise ValueError("historical order-book response size is outside the bound")
    condition_id = str(expected_condition_id or "").strip().lower()
    asset_id = str(expected_asset_id or "").strip()
    if (
        _CONDITION_ID.fullmatch(condition_id) is None
        or _TOKEN_ID.fullmatch(asset_id) is None
        or type(start_ms) is not int
        or type(end_ms) is not int
        or start_ms <= 0
        or end_ms <= start_ms
    ):
        raise ValueError("historical order-book request identity is invalid")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("historical order-book response is not strict JSON") from exc
    if not isinstance(decoded, Mapping) or set(decoded) != _PAGE_FIELDS:
        raise ValueError("historical order-book page has an unexpected schema")
    count = decoded.get("count")
    data = decoded.get("data")
    if (
        type(count) is not int
        or count < 0
        or not isinstance(data, list)
        or len(data) > _MAXIMUM_PAGE_RECORDS
        or count < len(data)
    ):
        raise ValueError("historical order-book page bounds differ")
    snapshots = tuple(
        _parse_snapshot(
            item,
            expected_condition_id=condition_id,
            expected_asset_id=asset_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for item in data
    )
    timestamps = [snapshot.timestamp_ms for snapshot in snapshots]
    if any(
        right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("historical order-book page is not strictly chronological")
    return HistoricalOrderbookPage(
        remaining_record_count=count,
        snapshots=snapshots,
        source_payload_sha256=_sha256(raw),
    )


class PolymarketHistoricalL2Client:
    """Credential-free, bounded client for closed Polymarket book windows."""

    def __init__(
        self,
        *,
        session: _Session | None = None,
        timeout_seconds: float = 30.0,
        minimum_request_interval_seconds: float = 0.2,
        maximum_attempts: int = 4,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = max(2.0, min(60.0, float(timeout_seconds)))
        self.minimum_request_interval_seconds = max(
            0.0,
            min(5.0, float(minimum_request_interval_seconds)),
        )
        if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 8:
            raise ValueError("maximum_attempts is outside the bound")
        self.maximum_attempts = maximum_attempts
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic = monotonic or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._last_request_at: float | None = None

    def _assert_public_session(self) -> None:
        header_names = {str(name).strip().lower() for name in self.session.headers}
        if header_names & _SENSITIVE_HEADERS:
            raise ValueError("historical order-book session contains authority headers")
        cookies = getattr(self.session, "cookies", None)
        if cookies is not None and len(cookies):
            raise ValueError("historical order-book session contains cookies")

    def _discard_response_cookies(self) -> None:
        cookies = getattr(self.session, "cookies", None)
        if cookies is None or not len(cookies):
            return
        clear = getattr(cookies, "clear", None)
        if not callable(clear):
            raise ValueError("historical order-book session retained cookies")
        clear()
        if len(cookies):
            raise ValueError("historical order-book session retained cookies")

    def _wait_for_rate_limit(self) -> None:
        now = float(self.monotonic())
        if self._last_request_at is not None:
            remaining = self.minimum_request_interval_seconds - (
                now - self._last_request_at
            )
            if remaining > 0:
                self.sleeper(remaining)
                now = float(self.monotonic())
        self._last_request_at = now

    @staticmethod
    def _retry_after_seconds(response: _Response, attempt: int) -> float:
        fallback = min(8.0, 0.5 * (2**attempt))
        value = str(response.headers.get("Retry-After") or "").strip()
        if not value.isdigit():
            return fallback
        return max(fallback, min(30.0, float(value)))

    def _request(self, params: Mapping[str, object]) -> bytes:
        self._assert_public_session()
        for attempt in range(self.maximum_attempts):
            self._wait_for_rate_limit()
            response = self.session.get(
                POLYMARKET_ORDERBOOK_HISTORY_URL,
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "simple-ai-trading-historical-l2/0.1",
                },
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
            self._discard_response_cookies()
            status = int(response.status_code)
            if status == 200:
                parsed = urlparse(str(response.url))
                if (
                    parsed.scheme != "https"
                    or parsed.netloc.lower() != "clob.polymarket.com"
                    or parsed.path != "/orderbook-history"
                ):
                    raise ValueError("historical order-book response changed origin")
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if not content_type.startswith("application/json"):
                    raise ValueError("historical order-book response is not JSON")
                content = bytes(response.content)
                if len(content) > _MAXIMUM_RESPONSE_BYTES:
                    raise ValueError(
                        "historical order-book response exceeded size bound"
                    )
                return content
            if (
                status not in _RETRYABLE_STATUS_CODES
                or attempt + 1 == self.maximum_attempts
            ):
                raise ValueError(
                    f"historical order-book request failed with HTTP {status}"
                )
            self.sleeper(self._retry_after_seconds(response, attempt))
        raise AssertionError("unreachable historical order-book retry state")

    def fetch_page(
        self,
        *,
        condition_id: str,
        asset_id: str,
        start_ms: int,
        end_ms: int,
        limit: int = _MAXIMUM_PAGE_RECORDS,
    ) -> HistoricalOrderbookPage:
        if type(limit) is not int or not 1 <= limit <= _MAXIMUM_PAGE_RECORDS:
            raise ValueError("historical order-book page limit is outside the bound")
        raw = self._request(
            {
                "asset_id": asset_id,
                "endTs": end_ms,
                "limit": limit,
                "startTs": start_ms,
            }
        )
        return parse_historical_orderbook_page(
            raw,
            expected_condition_id=condition_id,
            expected_asset_id=asset_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def fetch_closed_window(
        self,
        *,
        condition_id: str,
        asset_id: str,
        event_start_ms: int,
        event_end_ms: int,
        limit: int = _MAXIMUM_PAGE_RECORDS,
    ) -> HistoricalL2Window:
        now_ms = int(self.clock_ms())
        if event_end_ms > now_ms - _MINIMUM_CLOSED_AGE_MS:
            raise ValueError("historical order-book window is not durably closed")
        cursor = event_start_ms
        snapshots: list[HistoricalBookSnapshot] = []
        source_chain = hashlib.sha256(
            _canonical_json(
                {
                    "asset_id": asset_id,
                    "condition_id": condition_id,
                    "end_ms": event_end_ms,
                    "endpoint": POLYMARKET_ORDERBOOK_HISTORY_URL,
                    "start_ms": event_start_ms,
                }
            )
        )
        while cursor < event_end_ms:
            page = self.fetch_page(
                condition_id=condition_id,
                asset_id=asset_id,
                start_ms=cursor,
                end_ms=event_end_ms,
                limit=limit,
            )
            source_chain.update(bytes.fromhex(page.source_payload_sha256))
            if not page.snapshots:
                break
            if (
                snapshots
                and page.snapshots[0].timestamp_ms <= snapshots[-1].timestamp_ms
            ):
                raise ValueError("historical order-book pages overlap")
            snapshots.extend(page.snapshots)
            if len(snapshots) > _MAXIMUM_WINDOW_RECORDS:
                raise ValueError("historical order-book window exceeded record bound")
            cursor = page.snapshots[-1].timestamp_ms + 1
            if len(page.snapshots) < limit:
                break
        if not snapshots:
            raise ValueError("historical order-book window contains no snapshots")
        return HistoricalL2Window(
            condition_id=str(condition_id).lower(),
            asset_id=str(asset_id),
            event_start_ms=event_start_ms,
            event_end_ms=event_end_ms,
            snapshots=tuple(snapshots),
            source_chain_sha256=source_chain.hexdigest(),
        )


def encode_historical_l2_window(window: HistoricalL2Window) -> HistoricalL2Chunk:
    raw = _canonical_json(window.as_dict())
    compressed = zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)
    return HistoricalL2Chunk(
        codec=POLYMARKET_HISTORICAL_L2_CODEC,
        record_count=len(window.snapshots),
        raw_size_bytes=len(raw),
        compressed_size_bytes=len(compressed),
        raw_sha256=_sha256(raw),
        compressed_sha256=_sha256(compressed),
        payload=compressed,
    )


def decode_historical_l2_chunk(chunk: HistoricalL2Chunk) -> dict[str, object]:
    if (
        chunk.codec != POLYMARKET_HISTORICAL_L2_CODEC
        or chunk.record_count <= 0
        or chunk.raw_size_bytes <= 0
        or chunk.compressed_size_bytes != len(chunk.payload)
        or chunk.compressed_sha256 != _sha256(chunk.payload)
    ):
        raise ValueError("historical L2 chunk envelope differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            chunk.payload,
            max_output_size=chunk.raw_size_bytes,
        )
    except zstandard.ZstdError as exc:
        raise ValueError("historical L2 chunk is not valid zstd") from exc
    if len(raw) != chunk.raw_size_bytes or _sha256(raw) != chunk.raw_sha256:
        raise ValueError("historical L2 chunk content hash differs")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("historical L2 chunk is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("historical L2 chunk body is not an object")
    snapshots = value.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != chunk.record_count:
        raise ValueError("historical L2 chunk record count differs")
    return value


def decode_historical_l2_window(chunk: HistoricalL2Chunk) -> HistoricalL2Window:
    """Reconstruct and semantically validate one archived full-book window."""

    value = decode_historical_l2_chunk(chunk)
    if set(value) != _ARCHIVED_WINDOW_FIELDS:
        raise ValueError("historical L2 window archive schema differs")
    condition_id = str(value.get("condition_id") or "").strip().lower()
    asset_id = str(value.get("asset_id") or "").strip()
    event_start_ms = value.get("event_start_ms")
    event_end_ms = value.get("event_end_ms")
    source = value.get("source")
    authority = value.get("authority")
    snapshots_value = value.get("snapshots")
    expected_authority = {
        "execution_calibration": False,
        "fill_claim": False,
        "live_trading": False,
        "model_data_eligible": False,
        "paper_trading": False,
        "profitability_claim": False,
        "queue_position_calibration": False,
        "transport_latency_calibration": False,
    }
    if (
        value.get("schema_version") != POLYMARKET_HISTORICAL_L2_SCHEMA_VERSION
        or _CONDITION_ID.fullmatch(condition_id) is None
        or _TOKEN_ID.fullmatch(asset_id) is None
        or type(event_start_ms) is not int
        or type(event_end_ms) is not int
        or event_start_ms <= 0
        or event_end_ms <= event_start_ms
        or authority != expected_authority
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "authentication_used",
            "binance_used",
            "endpoint",
            "source_chain_sha256",
        }
        or source.get("authentication_used") is not False
        or source.get("binance_used") is not False
        or source.get("endpoint") != POLYMARKET_ORDERBOOK_HISTORY_URL
        or not isinstance(source.get("source_chain_sha256"), str)
        or _SHA256.fullmatch(str(source["source_chain_sha256"])) is None
        or not isinstance(snapshots_value, list)
        or len(snapshots_value) != chunk.record_count
    ):
        raise ValueError("historical L2 window archive identity differs")
    snapshots: list[HistoricalBookSnapshot] = []
    previous_timestamp = event_start_ms - 1
    for index, snapshot_value in enumerate(snapshots_value):
        if (
            not isinstance(snapshot_value, Mapping)
            or set(snapshot_value) != _ARCHIVED_SNAPSHOT_FIELDS
        ):
            raise ValueError("historical L2 archived snapshot schema differs")
        snapshot_condition = (
            str(snapshot_value.get("condition_id") or "").strip().lower()
        )
        snapshot_asset = str(snapshot_value.get("asset_id") or "").strip()
        timestamp_ms = snapshot_value.get("timestamp_ms")
        book_hash = str(snapshot_value.get("book_hash") or "").strip().lower()
        source_sha = (
            str(snapshot_value.get("source_payload_sha256") or "").strip().lower()
        )
        if (
            snapshot_condition != condition_id
            or snapshot_asset != asset_id
            or type(timestamp_ms) is not int
            or not event_start_ms <= timestamp_ms < event_end_ms
            or timestamp_ms <= previous_timestamp
            or _BOOK_HASH.fullmatch(book_hash) is None
            or _SHA256.fullmatch(source_sha) is None
        ):
            raise ValueError("historical L2 archived snapshot identity differs")
        bids = _parse_side(
            snapshot_value.get("bids"),
            name=f"archived_snapshots[{index}].bids",
            reverse=True,
        )
        asks = _parse_side(
            snapshot_value.get("asks"),
            name=f"archived_snapshots[{index}].asks",
            reverse=False,
        )
        if bids and asks and Decimal(bids[0].price) >= Decimal(asks[0].price):
            raise ValueError("historical L2 archived snapshot is crossed")
        minimum_order_size = _decimal_string(
            snapshot_value.get("minimum_order_size"),
            name=f"archived_snapshots[{index}].minimum_order_size",
            allow_zero=False,
        )
        tick_size = _decimal_string(
            snapshot_value.get("tick_size"),
            name=f"archived_snapshots[{index}].tick_size",
            allow_zero=False,
        )
        if not Decimal(0) < Decimal(tick_size) < Decimal(1):
            raise ValueError("historical L2 archived tick size differs")
        negative_risk = snapshot_value.get("negative_risk")
        if type(negative_risk) is not bool:
            raise ValueError("historical L2 archived negative-risk flag differs")
        last_trade = snapshot_value.get("last_trade_price")
        if last_trade != "":
            last_trade = _decimal_string(
                last_trade,
                name=f"archived_snapshots[{index}].last_trade_price",
                allow_zero=True,
            )
            if not Decimal(0) <= Decimal(last_trade) <= Decimal(1):
                raise ValueError("historical L2 archived last-trade price differs")
        snapshot = HistoricalBookSnapshot(
            condition_id=condition_id,
            asset_id=asset_id,
            timestamp_ms=timestamp_ms,
            book_hash=book_hash,
            bids=bids,
            asks=asks,
            minimum_order_size=minimum_order_size,
            tick_size=tick_size,
            negative_risk=negative_risk,
            last_trade_price=str(last_trade),
            source_payload_sha256=source_sha,
        )
        if snapshot.as_dict() != dict(snapshot_value):
            raise ValueError("historical L2 archived snapshot is not canonical")
        snapshots.append(snapshot)
        previous_timestamp = timestamp_ms
    window = HistoricalL2Window(
        condition_id=condition_id,
        asset_id=asset_id,
        event_start_ms=event_start_ms,
        event_end_ms=event_end_ms,
        snapshots=tuple(snapshots),
        source_chain_sha256=str(source["source_chain_sha256"]),
    )
    if window.as_dict() != value:
        raise ValueError("historical L2 window archive is not canonical")
    return window


__all__ = [
    "HistoricalBookLevel",
    "HistoricalBookSnapshot",
    "HistoricalL2Chunk",
    "HistoricalL2Window",
    "HistoricalOrderbookPage",
    "POLYMARKET_HISTORICAL_L2_CODEC",
    "POLYMARKET_HISTORICAL_L2_SCHEMA_VERSION",
    "POLYMARKET_ORDERBOOK_HISTORY_URL",
    "PolymarketHistoricalL2Client",
    "decode_historical_l2_chunk",
    "decode_historical_l2_window",
    "encode_historical_l2_window",
    "parse_historical_orderbook_page",
]
