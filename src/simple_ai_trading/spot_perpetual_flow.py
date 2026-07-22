"""Strict streaming aggregation for Binance spot/perpetual aggregate trades."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
from typing import Protocol
import zipfile

import numpy as np

from .binance_archive import normalize_archive_timestamp_ms
from .assets import is_supported_major_symbol, normalize_symbol


FLOW_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
FLOW_MARKET_TYPES = ("spot", "futures")
SECONDS_PER_DAY = 86_400
FLOW_SCHEMA_VERSION = "spot-perpetual-flow-day-v1"


class _Digest(Protocol):
    def update(self, data: bytes | bytearray | memoryview, /) -> None: ...


@dataclass(frozen=True)
class FlowSourceAudit:
    source_rows: int
    aggregate_trade_count: int
    constituent_trade_count: int
    first_aggregate_trade_id: int
    last_aggregate_trade_id: int
    aggregate_trade_id_gaps: int
    constituent_trade_id_gaps: int
    first_trade_time_ms: int
    last_trade_time_ms: int
    best_match_false_count: int
    header_present: bool
    member_name: str
    member_uncompressed_bytes: int


@dataclass(frozen=True)
class FlowDay:
    symbol: str
    market_type: str
    period: str
    second_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    base_volume: np.ndarray
    quote_volume: np.ndarray
    aggressive_buy_quote: np.ndarray
    aggressive_sell_quote: np.ndarray
    aggregate_count: np.ndarray
    constituent_trade_count: np.ndarray
    maximum_aggregate_quote: np.ndarray
    squared_aggregate_quote_sum: np.ndarray
    last_trade_age_seconds: np.ndarray
    audit: FlowSourceAudit
    flow_sha256: str

    @property
    def rows(self) -> int:
        return int(self.second_ms.size)

    @property
    def observed_trade_seconds(self) -> int:
        return int(np.count_nonzero(self.aggregate_count))

    def validate(self, *, require_full_day: bool = True) -> None:
        expected_rows = SECONDS_PER_DAY if require_full_day else self.rows
        arrays = (
            self.second_ms,
            self.open,
            self.high,
            self.low,
            self.close,
            self.base_volume,
            self.quote_volume,
            self.aggressive_buy_quote,
            self.aggressive_sell_quote,
            self.aggregate_count,
            self.constituent_trade_count,
            self.maximum_aggregate_quote,
            self.squared_aggregate_quote_sum,
            self.last_trade_age_seconds,
        )
        if self.rows != expected_rows or any(value.ndim != 1 for value in arrays):
            raise ValueError("spot/perpetual flow arrays have invalid dimensions")
        if any(len(value) != self.rows for value in arrays):
            raise ValueError("spot/perpetual flow arrays have different lengths")
        if self.rows > 1 and np.any(np.diff(self.second_ms) != 1_000):
            raise ValueError("spot/perpetual flow seconds are not contiguous")
        numeric = (
            self.base_volume,
            self.quote_volume,
            self.aggressive_buy_quote,
            self.aggressive_sell_quote,
            self.maximum_aggregate_quote,
            self.squared_aggregate_quote_sum,
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("spot/perpetual flow values are nonfinite")
        if any(np.any(value < 0.0) for value in numeric):
            raise ValueError("spot/perpetual flow values are negative")
        traded = self.aggregate_count > 0
        if not np.any(traded):
            raise ValueError("spot/perpetual flow day has no trades")
        if np.any(self.constituent_trade_count[traded] < self.aggregate_count[traded]):
            raise ValueError("constituent trade count is below aggregate count")
        if np.any(self.aggressive_buy_quote > self.quote_volume + 1e-8) or np.any(
            self.aggressive_sell_quote > self.quote_volume + 1e-8
        ):
            raise ValueError("aggressive flow exceeds quote volume")
        if np.any(
            np.abs(
                self.aggressive_buy_quote
                + self.aggressive_sell_quote
                - self.quote_volume
            )
            > np.maximum(1e-8, self.quote_volume * 1e-10)
        ):
            raise ValueError("aggressive flow does not reconcile to quote volume")
        prices = (self.open, self.high, self.low, self.close)
        if any(np.any(~np.isfinite(value[traded])) for value in prices):
            raise ValueError("traded seconds have missing prices")
        if np.any(self.high[traded] < self.low[traded]) or np.any(
            self.open[traded] <= 0.0
        ):
            raise ValueError("traded seconds have invalid prices")
        if (
            np.any(self.high[traded] < self.open[traded])
            or np.any(self.high[traded] < self.close[traded])
            or np.any(self.low[traded] > self.open[traded])
            or np.any(self.low[traded] > self.close[traded])
            or np.any(self.low[traded] <= 0.0)
        ):
            raise ValueError("traded seconds violate OHLC bounds")
        if np.any(self.last_trade_age_seconds[traded] != 0):
            raise ValueError("traded seconds have a nonzero last-trade age")
        expected_hash = _flow_sha256(self)
        if self.flow_sha256 != expected_hash:
            raise ValueError("spot/perpetual flow fingerprint differs")


def _period_bounds_ms(period: str) -> tuple[int, int]:
    try:
        parsed = date.fromisoformat(str(period))
    except ValueError as exc:
        raise ValueError("flow period must be a valid YYYY-MM-DD value") from exc
    if parsed.isoformat() != period:
        raise ValueError("flow period must be canonical YYYY-MM-DD")
    start = int(datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC).timestamp() * 1_000)
    return start, start + SECONDS_PER_DAY * 1_000


def _normalize_market(symbol: str, market_type: str) -> tuple[str, str]:
    normalized_symbol = normalize_symbol(symbol, default="")
    normalized_market = str(market_type or "").strip().lower()
    if (
        normalized_symbol not in FLOW_SYMBOLS
        or not is_supported_major_symbol(normalized_symbol)
    ):
        raise ValueError(f"unsupported flow symbol: {normalized_symbol}")
    if normalized_market not in FLOW_MARKET_TYPES:
        raise ValueError(f"unsupported flow market type: {market_type}")
    return normalized_symbol, normalized_market


def _strict_bool(value: object, *, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"aggregate-trade {field} is not boolean")


def _is_header(row: list[str]) -> bool:
    if len(row) < 3:
        return False
    first = row[0].strip().lower().replace(" ", "_")
    second = row[1].strip().lower().replace(" ", "_")
    return first in {"agg_trade_id", "aggtradeid", "aggregate_tradeid"} and second == "price"


def _safe_member(
    archive: zipfile.ZipFile,
    *,
    maximum_uncompressed_bytes: int,
    expected_member_name: str,
) -> zipfile.ZipInfo:
    members = [value for value in archive.infolist() if not value.is_dir()]
    csv_members = [value for value in members if value.filename.lower().endswith(".csv")]
    if len(members) != 1 or len(csv_members) != 1:
        raise ValueError("aggregate-trade ZIP must contain exactly one CSV member")
    member = csv_members[0]
    path = PurePosixPath(member.filename.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError("aggregate-trade ZIP member path is unsafe")
    if member.filename != expected_member_name:
        raise ValueError("aggregate-trade ZIP member identity differs from the request")
    if (
        member.file_size <= 0
        or member.file_size > int(maximum_uncompressed_bytes)
        or member.compress_size <= 0
        or member.file_size / member.compress_size > 250.0
    ):
        raise ValueError("aggregate-trade ZIP expansion bounds are invalid")
    return member


def _parse_row(
    row: list[str],
    *,
    market_type: str,
) -> tuple[int, float, float, int, int, int, bool, bool]:
    normalized = [value.strip() for value in row]
    while normalized and normalized[-1] == "":
        normalized.pop()
    expected_columns = 8 if market_type == "spot" else 7
    if len(normalized) != expected_columns:
        raise ValueError("aggregate-trade row has an unexpected column count")
    try:
        aggregate_id = int(normalized[0])
        price = float(normalized[1])
        quantity = float(normalized[2])
        first_trade_id = int(normalized[3])
        last_trade_id = int(normalized[4])
        trade_time_ms = normalize_archive_timestamp_ms(normalized[5])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("aggregate-trade row has an invalid numeric field") from exc
    buyer_is_maker = _strict_bool(normalized[6], field="buyer-is-maker")
    best_match = (
        _strict_bool(normalized[7], field="best-match")
        if market_type == "spot"
        else True
    )
    if (
        aggregate_id < 0
        or first_trade_id < 0
        or last_trade_id < first_trade_id
        or not math.isfinite(price)
        or not math.isfinite(quantity)
        or price <= 0.0
        or quantity <= 0.0
        or trade_time_ms <= 0
    ):
        raise ValueError("aggregate-trade row violates value bounds")
    return (
        aggregate_id,
        price,
        quantity,
        first_trade_id,
        last_trade_id,
        trade_time_ms,
        buyer_is_maker,
        best_match,
    )


def _array_digest(digest: _Digest, value: np.ndarray) -> None:
    canonical_dtype = value.dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
    if np.issubdtype(canonical_dtype, np.floating) and np.any(np.isnan(contiguous)):
        contiguous = contiguous.copy()
        contiguous[np.isnan(contiguous)] = np.nan
    digest.update(canonical_dtype.str.encode("ascii"))
    digest.update(int(contiguous.size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(contiguous).cast("B"))


def _flow_sha256(value: FlowDay) -> str:
    digest = hashlib.sha256()
    for text in (
        FLOW_SCHEMA_VERSION,
        value.symbol,
        value.market_type,
        value.period,
    ):
        digest.update(text.encode("ascii"))
        digest.update(b"\x00")
    for array in (
        value.second_ms,
        value.open,
        value.high,
        value.low,
        value.close,
        value.base_volume,
        value.quote_volume,
        value.aggressive_buy_quote,
        value.aggressive_sell_quote,
        value.aggregate_count,
        value.constituent_trade_count,
        value.maximum_aggregate_quote,
        value.squared_aggregate_quote_sum,
        value.last_trade_age_seconds,
    ):
        _array_digest(digest, array)
    for field in value.audit.__dataclass_fields__:
        digest.update(str(getattr(value.audit, field)).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def aggregate_trade_zip(
    path: str | Path,
    *,
    symbol: str,
    market_type: str,
    period: str,
    maximum_uncompressed_bytes: int,
    expected_seconds: int = SECONDS_PER_DAY,
) -> FlowDay:
    """Strictly aggregate one official daily archive without extracting it."""

    normalized_symbol, normalized_market = _normalize_market(symbol, market_type)
    day_start_ms, day_end_ms = _period_bounds_ms(period)
    row_count = int(expected_seconds)
    if not 1 <= row_count <= SECONDS_PER_DAY:
        raise ValueError("expected_seconds is outside one UTC day")
    if int(maximum_uncompressed_bytes) <= 0:
        raise ValueError("maximum_uncompressed_bytes must be positive")
    effective_end_ms = day_start_ms + row_count * 1_000
    second_ms = day_start_ms + np.arange(row_count, dtype=np.int64) * 1_000
    open_price = np.full(row_count, np.nan, dtype=np.float64)
    high_price = np.full(row_count, np.nan, dtype=np.float64)
    low_price = np.full(row_count, np.nan, dtype=np.float64)
    close_price = np.full(row_count, np.nan, dtype=np.float64)
    base_volume = np.zeros(row_count, dtype=np.float64)
    quote_volume = np.zeros(row_count, dtype=np.float64)
    aggressive_buy_quote = np.zeros(row_count, dtype=np.float64)
    aggressive_sell_quote = np.zeros(row_count, dtype=np.float64)
    aggregate_count = np.zeros(row_count, dtype=np.uint32)
    constituent_count = np.zeros(row_count, dtype=np.uint32)
    maximum_quote = np.zeros(row_count, dtype=np.float64)
    squared_quote_sum = np.zeros(row_count, dtype=np.float64)
    last_trade_age = np.full(row_count, np.iinfo(np.uint32).max, dtype=np.uint32)

    source_rows = 0
    aggregate_total = 0
    constituent_total = 0
    first_aggregate_id = -1
    previous_aggregate_id = -1
    previous_last_trade_id = -1
    aggregate_id_gaps = 0
    constituent_id_gaps = 0
    first_time = -1
    previous_time = -1
    best_match_false = 0
    header_present = False
    member_name = ""
    member_size = 0

    with zipfile.ZipFile(Path(path)) as archive:
        member = _safe_member(
            archive,
            maximum_uncompressed_bytes=int(maximum_uncompressed_bytes),
            expected_member_name=f"{normalized_symbol}-aggTrades-{period}.csv",
        )
        member_name = member.filename
        member_size = int(member.file_size)
        with archive.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.reader(text)
            for line_number, row in enumerate(reader, start=1):
                if line_number == 1 and _is_header(row):
                    header_present = True
                    continue
                if not row or all(not value.strip() for value in row):
                    raise ValueError("aggregate-trade archive contains a blank row")
                source_rows += 1
                (
                    aggregate_id,
                    price,
                    quantity,
                    first_trade_id,
                    last_trade_id,
                    trade_time_ms,
                    buyer_is_maker,
                    best_match,
                ) = _parse_row(row, market_type=normalized_market)
                if not day_start_ms <= trade_time_ms < day_end_ms:
                    raise ValueError("aggregate-trade row lies outside its UTC day")
                if trade_time_ms >= effective_end_ms:
                    raise ValueError("aggregate-trade row lies beyond the requested test window")
                if previous_aggregate_id >= 0:
                    if aggregate_id <= previous_aggregate_id:
                        raise ValueError("aggregate-trade ID is duplicate or regressing")
                    aggregate_id_gaps += max(0, aggregate_id - previous_aggregate_id - 1)
                    if first_trade_id <= previous_last_trade_id:
                        raise ValueError("constituent trade IDs overlap or regress")
                    constituent_id_gaps += max(
                        0, first_trade_id - previous_last_trade_id - 1
                    )
                    if trade_time_ms < previous_time:
                        raise ValueError("aggregate-trade transaction time regressed")
                else:
                    first_aggregate_id = aggregate_id
                    first_time = trade_time_ms
                previous_aggregate_id = aggregate_id
                previous_last_trade_id = last_trade_id
                previous_time = trade_time_ms
                aggregate_total += 1
                trade_count = last_trade_id - first_trade_id + 1
                constituent_total += trade_count
                best_match_false += int(not best_match)
                index = (trade_time_ms - day_start_ms) // 1_000
                quote = price * quantity
                if not math.isfinite(quote) or not math.isfinite(quote * quote):
                    raise ValueError("aggregate-trade notional is nonfinite")
                uint32_max = np.iinfo(np.uint32).max
                if int(aggregate_count[index]) >= uint32_max or trade_count > (
                    uint32_max - int(constituent_count[index])
                ):
                    raise ValueError("per-second aggregate-trade counts overflow uint32")
                next_base = float(base_volume[index]) + quantity
                next_quote = float(quote_volume[index]) + quote
                next_squared_quote = float(squared_quote_sum[index]) + quote * quote
                if not all(
                    math.isfinite(value)
                    for value in (next_base, next_quote, next_squared_quote)
                ):
                    raise ValueError("per-second aggregate-trade sums are nonfinite")
                if aggregate_count[index] == 0:
                    open_price[index] = price
                    high_price[index] = price
                    low_price[index] = price
                else:
                    high_price[index] = max(high_price[index], price)
                    low_price[index] = min(low_price[index], price)
                close_price[index] = price
                base_volume[index] = next_base
                quote_volume[index] = next_quote
                if buyer_is_maker:
                    aggressive_sell_quote[index] += quote
                else:
                    aggressive_buy_quote[index] += quote
                aggregate_count[index] += 1
                constituent_count[index] += trade_count
                maximum_quote[index] = max(maximum_quote[index], quote)
                squared_quote_sum[index] = next_squared_quote
    if source_rows <= 0 or aggregate_total != source_rows:
        raise ValueError("aggregate-trade archive has no data rows")

    last_close = math.nan
    age = np.iinfo(np.uint32).max
    for index in range(row_count):
        if aggregate_count[index] > 0:
            last_close = float(close_price[index])
            age = 0
        elif math.isfinite(last_close):
            age = min(int(age) + 1, np.iinfo(np.uint32).max - 1)
            open_price[index] = last_close
            high_price[index] = last_close
            low_price[index] = last_close
            close_price[index] = last_close
        last_trade_age[index] = age

    audit = FlowSourceAudit(
        source_rows=source_rows,
        aggregate_trade_count=aggregate_total,
        constituent_trade_count=constituent_total,
        first_aggregate_trade_id=first_aggregate_id,
        last_aggregate_trade_id=previous_aggregate_id,
        aggregate_trade_id_gaps=aggregate_id_gaps,
        constituent_trade_id_gaps=constituent_id_gaps,
        first_trade_time_ms=first_time,
        last_trade_time_ms=previous_time,
        best_match_false_count=best_match_false,
        header_present=header_present,
        member_name=member_name,
        member_uncompressed_bytes=member_size,
    )
    provisional = FlowDay(
        symbol=normalized_symbol,
        market_type=normalized_market,
        period=period,
        second_ms=second_ms,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        base_volume=base_volume,
        quote_volume=quote_volume,
        aggressive_buy_quote=aggressive_buy_quote,
        aggressive_sell_quote=aggressive_sell_quote,
        aggregate_count=aggregate_count,
        constituent_trade_count=constituent_count,
        maximum_aggregate_quote=maximum_quote,
        squared_aggregate_quote_sum=squared_quote_sum,
        last_trade_age_seconds=last_trade_age,
        audit=audit,
        flow_sha256="",
    )
    result = replace(provisional, flow_sha256=_flow_sha256(provisional))
    result.validate(require_full_day=row_count == SECONDS_PER_DAY)
    return result


__all__ = [
    "FLOW_MARKET_TYPES",
    "FLOW_SCHEMA_VERSION",
    "FLOW_SYMBOLS",
    "SECONDS_PER_DAY",
    "FlowDay",
    "FlowSourceAudit",
    "aggregate_trade_zip",
]
