"""Binance public data archive ingestion into the market-data store."""

from __future__ import annotations

import calendar
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import re
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from defusedxml import ElementTree

from .api import Candle
from .market_data import clean_candles
from .market_store import MarketDataStore


BINANCE_ARCHIVE_BASE_URL = "https://data.binance.vision/data"
_ZIP_LINK_PATTERN = re.compile(r'href=["\'](?P<href>[^"\']+\.zip)["\']', re.IGNORECASE)
_CHECKSUM_PATTERN = re.compile(r"\b(?P<sha256>[a-fA-F0-9]{64})\b")
_ARCHIVE_PERIOD_PATTERN = re.compile(r"-(?P<period>\d{4}-\d{2}(?:-\d{2})?)$")


@dataclass(frozen=True)
class ArchiveIngestResult:
    url: str
    symbol: str
    market_type: str
    interval: str
    data_type: str
    period: str
    status: str
    rows_inserted: int
    rows_read: int
    bytes_downloaded: int
    sha256: str
    checksum_sha256: str = ""
    checksum_status: str = "unverified"
    error: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveListingItem:
    url: str
    key: str
    period: str
    size_bytes: int = 0
    last_modified: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ListingEntry:
    key: str
    size_bytes: int = 0
    last_modified: str = ""


_ARCHIVE_LISTING_ITEM_CACHE: dict[str, ArchiveListingItem] = {}


def _archive_market_segment(market_type: str) -> str:
    market = str(market_type or "spot").lower()
    if market == "spot":
        return "spot"
    if market == "futures":
        return "futures/um"
    raise ValueError("market_type must be 'spot' or 'futures'")


def _normalize_archive_data_type(data_type: str = "klines") -> str:
    value = str(data_type or "klines").strip()
    if value == "klines":
        return "klines"
    if value == "aggTrades":
        return "aggTrades"
    raise ValueError("data_type must be 'klines' or 'aggTrades'")


def archive_directory_url(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    data_type: str = "klines",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
) -> str:
    cadence = str(cadence or "monthly").lower()
    if cadence not in {"daily", "monthly"}:
        raise ValueError("cadence must be 'daily' or 'monthly'")
    segment = _archive_market_segment(market_type)
    kind = _normalize_archive_data_type(data_type)
    if kind == "aggTrades":
        return f"{base_url.rstrip('/')}/{segment}/{cadence}/aggTrades/{symbol.upper()}/"
    return f"{base_url.rstrip('/')}/{segment}/{cadence}/klines/{symbol.upper()}/{interval}/"


def archive_listing_url(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    data_type: str = "klines",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
    marker: str | None = None,
) -> str:
    cadence = str(cadence or "monthly").lower()
    if cadence not in {"daily", "monthly"}:
        raise ValueError("cadence must be 'daily' or 'monthly'")
    segment = _archive_market_segment(market_type)
    kind = _normalize_archive_data_type(data_type)
    if kind == "aggTrades":
        prefix = f"data/{segment}/{cadence}/aggTrades/{symbol.upper()}/"
    else:
        prefix = f"data/{segment}/{cadence}/klines/{symbol.upper()}/{interval}/"
    params = {"delimiter": "/", "prefix": prefix}
    if marker:
        params["marker"] = marker
    return "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?" + urlencode(params)


def archive_file_url(
    *,
    symbol: str,
    interval: str,
    period: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    data_type: str = "klines",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
) -> str:
    directory = archive_directory_url(
        symbol=symbol,
        interval=interval,
        market_type=market_type,
        cadence=cadence,
        data_type=data_type,
        base_url=base_url,
    )
    if _normalize_archive_data_type(data_type) == "aggTrades":
        return f"{directory}{symbol.upper()}-aggTrades-{period}.zip"
    return f"{directory}{symbol.upper()}-{interval}-{period}.zip"


def list_archive_urls(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    data_type: str = "klines",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
    timeout: int = 20,
    html_loader: Callable[[str], str] | None = None,
) -> list[str]:
    return [
        item.url
        for item in list_archive_items(
            symbol=symbol,
            interval=interval,
            market_type=market_type,
            cadence=cadence,
            data_type=data_type,
            base_url=base_url,
            timeout=timeout,
            html_loader=html_loader,
        )
    ]


def _archive_url_from_key(origin: str, key: str) -> str:
    if key.startswith("http://") or key.startswith("https://"):
        return key
    return f"{origin}/{key.lstrip('/')}"


def _remember_archive_listing_items(items: Sequence[ArchiveListingItem]) -> None:
    for item in items:
        _ARCHIVE_LISTING_ITEM_CACHE[item.url] = item


def archive_listing_items_by_url(urls: Sequence[str]) -> dict[str, ArchiveListingItem]:
    """Return cached official listing metadata for URLs from recent listings."""

    return {url: _ARCHIVE_LISTING_ITEM_CACHE[url] for url in urls if url in _ARCHIVE_LISTING_ITEM_CACHE}


def list_archive_items(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    data_type: str = "klines",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
    timeout: int = 20,
    html_loader: Callable[[str], str] | None = None,
) -> list[ArchiveListingItem]:
    origin = base_url.rstrip("/").removesuffix("/data")
    items: list[ArchiveListingItem] = []
    marker: str | None = None
    while True:
        listing = archive_listing_url(
            symbol=symbol,
            interval=interval,
            market_type=market_type,
            cadence=cadence,
            data_type=data_type,
            base_url=base_url,
            marker=marker,
        )
        if html_loader is None:
            with urlopen(listing, timeout=timeout) as response:  # nosec B310 - official public archive URL
                listing_text = response.read().decode("utf-8", errors="ignore")
        else:
            listing_text = html_loader(listing)
        entries, next_marker, truncated = _parse_listing_entries(listing_text)
        keys = [entry.key for entry in entries]
        for entry in entries:
            key = entry.key
            if key.endswith(".zip") and not key.endswith(".zip.CHECKSUM"):
                url = _archive_url_from_key(origin, key)
                items.append(
                    ArchiveListingItem(
                        url=url,
                        key=key,
                        period=archive_url_period(url),
                        size_bytes=max(0, int(entry.size_bytes)),
                        last_modified=entry.last_modified,
                    )
                )
        if not truncated:
            break
        marker = next_marker or (keys[-1] if keys else None)
        if marker is None:
            break
    unique = {item.url: item for item in items}
    selected = [unique[url] for url in sorted(unique)]
    _remember_archive_listing_items(selected)
    return selected


def archive_url_period(url: str) -> str:
    """Extract the official archive period from a Binance archive URL."""

    filename = Path(urlparse(str(url)).path).stem
    match = _ARCHIVE_PERIOD_PATTERN.search(filename)
    return match.group("period") if match else ""


def _period_date_bounds(period: str) -> tuple[date, date] | None:
    value = str(period or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        year, month, day = (int(part) for part in value.split("-"))
        parsed = date(year, month, day)
        return parsed, parsed
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (int(part) for part in value.split("-"))
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, 1), date(year, month, last_day)
    return None


def _period_ms_bounds(period: str) -> tuple[int, int] | None:
    bounds = _period_date_bounds(period)
    if bounds is None:
        return None
    start_date, end_date = bounds
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000) + 86_400_000 - 1000
    return start_ms, end_ms


def archive_period_in_range(
    period: str,
    *,
    start_period: str | None = None,
    end_period: str | None = None,
) -> bool:
    """Return whether an archive period overlaps the inclusive period window."""

    bounds = _period_date_bounds(period)
    if bounds is None:
        return False
    period_start, period_end = bounds
    if start_period:
        start_bounds = _period_date_bounds(start_period)
        if start_bounds is None:
            raise ValueError("start_period must be YYYY-MM or YYYY-MM-DD")
        if period_end < start_bounds[0]:
            return False
    if end_period:
        end_bounds = _period_date_bounds(end_period)
        if end_bounds is None:
            raise ValueError("end_period must be YYYY-MM or YYYY-MM-DD")
        if period_start > end_bounds[1]:
            return False
    return True


def validate_archive_period_window(
    *,
    start_period: str | None = None,
    end_period: str | None = None,
) -> None:
    """Validate an inclusive archive period window."""

    start_bounds = None
    end_bounds = None
    if start_period:
        start_bounds = _period_date_bounds(start_period)
        if start_bounds is None:
            raise ValueError("start_period must be YYYY-MM or YYYY-MM-DD")
    if end_period:
        end_bounds = _period_date_bounds(end_period)
        if end_bounds is None:
            raise ValueError("end_period must be YYYY-MM or YYYY-MM-DD")
    if start_bounds is not None and end_bounds is not None and start_bounds[0] > end_bounds[1]:
        raise ValueError("start_period must be earlier than or equal to end_period")


def filter_archive_urls_by_period(
    urls: Sequence[str],
    *,
    start_period: str | None = None,
    end_period: str | None = None,
) -> list[str]:
    """Filter official Binance archive URLs by inclusive daily/monthly period."""

    if not start_period and not end_period:
        return list(urls)
    selected: list[str] = []
    for url in urls:
        period = archive_url_period(url)
        if period and archive_period_in_range(period, start_period=start_period, end_period=end_period):
            selected.append(url)
    return selected


def _xml_text(element: object, tag: str) -> str:
    if element is None:
        return ""
    found = element.find(f"{{*}}{tag}")  # type: ignore[attr-defined]
    return str(found.text or "") if found is not None else ""


def _parse_int(value: object, default: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed >= 0 else default


def _parse_listing_entries(listing_text: str) -> tuple[list[_ListingEntry], str | None, bool]:
    try:
        root = ElementTree.fromstring(listing_text)
    except ElementTree.ParseError:
        links = [match.group("href") for match in _ZIP_LINK_PATTERN.finditer(listing_text)]
        return [_ListingEntry(key=link) for link in links], None, False
    entries: list[_ListingEntry] = []
    for contents in root.findall(".//{*}Contents"):
        key = _xml_text(contents, "Key")
        if key:
            entries.append(
                _ListingEntry(
                    key=key,
                    size_bytes=_parse_int(_xml_text(contents, "Size")),
                    last_modified=_xml_text(contents, "LastModified"),
                )
            )
    next_marker = _xml_text(root, "NextMarker") or None
    truncated = _xml_text(root, "IsTruncated").strip().lower() == "true"
    return entries, next_marker, truncated


def _parse_listing_keys(listing_text: str) -> tuple[list[str], str | None, bool]:
    entries, next_marker, truncated = _parse_listing_entries(listing_text)
    keys = [entry.key for entry in entries]
    return keys, next_marker, truncated


def _normalize_archive_timestamp(value: str | int | float) -> int:
    parsed = int(float(value))
    # Binance public archive spot timestamps are documented as microseconds from
    # 2025-01-01 onward. Store all candles in millisecond time to match REST.
    if abs(parsed) >= 10_000_000_000_000:
        parsed = parsed // 1000
    return parsed


def _parse_archive_row(row: Sequence[str]) -> Candle | None:
    if len(row) < 7:
        return None
    try:
        return Candle(
            open_time=_normalize_archive_timestamp(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=_normalize_archive_timestamp(row[6]),
            quote_volume=float(row[7]) if len(row) > 7 and row[7] != "" else 0.0,
            trade_count=int(float(row[8])) if len(row) > 8 and row[8] != "" else 0,
            taker_buy_base_volume=float(row[9]) if len(row) > 9 and row[9] != "" else 0.0,
            taker_buy_quote_volume=float(row[10]) if len(row) > 10 and row[10] != "" else 0.0,
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _iter_zip_candles(path: Path) -> Iterable[Candle]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        for name in sorted(names):
            with archive.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.reader(text)
                for row in reader:
                    candle = _parse_archive_row(row)
                    if candle is not None:
                        yield candle


@dataclass
class _AggTradeSecond:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_agg_trade_row(row: Sequence[str]) -> tuple[int, float, float, int, bool] | None:
    if len(row) < 6:
        return None
    try:
        price = float(row[1])
        quantity = float(row[2])
        first_id = int(float(row[3])) if len(row) > 3 and row[3] != "" else 0
        last_id = int(float(row[4])) if len(row) > 4 and row[4] != "" else first_id
        timestamp = _normalize_archive_timestamp(row[5])
        buyer_is_maker = _parse_bool(row[6]) if len(row) > 6 else False
    except (TypeError, ValueError, OverflowError):
        return None
    if price <= 0.0 or quantity <= 0.0 or timestamp <= 0:
        return None
    trade_count = max(1, last_id - first_id + 1)
    return timestamp, price, quantity, trade_count, buyer_is_maker


def _agg_second_to_candle(second: _AggTradeSecond) -> Candle:
    return Candle(
        open_time=second.open_time,
        open=second.open,
        high=second.high,
        low=second.low,
        close=second.close,
        volume=second.volume,
        close_time=second.open_time + 999,
        quote_volume=second.quote_volume,
        trade_count=second.trade_count,
        taker_buy_base_volume=second.taker_buy_base_volume,
        taker_buy_quote_volume=second.taker_buy_quote_volume,
    )


def _no_trade_candle(open_time: int, price: float) -> Candle:
    return Candle(
        open_time=open_time,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=0.0,
        close_time=open_time + 999,
        quote_volume=0.0,
        trade_count=0,
        taker_buy_base_volume=0.0,
        taker_buy_quote_volume=0.0,
    )


def _iter_zip_agg_trade_candles(path: Path) -> Iterable[Candle]:
    """Stream aggregate trades and emit deterministic 1-second OHLCV candles."""

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        for name in sorted(names):
            current: _AggTradeSecond | None = None
            with archive.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.reader(text)
                for row in reader:
                    parsed = _parse_agg_trade_row(row)
                    if parsed is None:
                        continue
                    timestamp, price, quantity, trade_count, buyer_is_maker = parsed
                    open_time = (timestamp // 1000) * 1000
                    quote_volume = price * quantity
                    taker_buy_base = 0.0 if buyer_is_maker else quantity
                    taker_buy_quote = 0.0 if buyer_is_maker else quote_volume
                    if current is not None and open_time != current.open_time:
                        previous_close = current.close
                        previous_open_time = current.open_time
                        yield _agg_second_to_candle(current)
                        gap_time = previous_open_time + 1000
                        while gap_time < open_time:
                            yield _no_trade_candle(gap_time, previous_close)
                            gap_time += 1000
                        current = None
                    if current is None:
                        current = _AggTradeSecond(
                            open_time=open_time,
                            open=price,
                            high=price,
                            low=price,
                            close=price,
                            volume=quantity,
                            quote_volume=quote_volume,
                            trade_count=trade_count,
                            taker_buy_base_volume=taker_buy_base,
                            taker_buy_quote_volume=taker_buy_quote,
                        )
                    else:
                        current.high = max(current.high, price)
                        current.low = min(current.low, price)
                        current.close = price
                        current.volume += quantity
                        current.quote_volume += quote_volume
                        current.trade_count += trade_count
                        current.taker_buy_base_volume += taker_buy_base
                        current.taker_buy_quote_volume += taker_buy_quote
            if current is not None:
                yield _agg_second_to_candle(current)


def _iter_period_bounded_agg_trade_candles(
    candles: Iterable[Candle],
    *,
    start_ms: int,
    end_ms: int,
    prior_close: float | None = None,
) -> Iterable[Candle]:
    """Fill verified no-trade seconds at archive period edges without inventing unknown prices."""

    expected_open_time = int(start_ms)
    previous_close = prior_close
    seen_trade_candle = False
    for candle in candles:
        open_time = int(candle.open_time)
        if open_time < start_ms or open_time > end_ms:
            continue
        if open_time > expected_open_time and previous_close is not None:
            gap_time = expected_open_time
            while gap_time < open_time:
                yield _no_trade_candle(gap_time, previous_close)
                gap_time += 1000
        yield candle
        previous_close = float(candle.close)
        expected_open_time = open_time + 1000
        seen_trade_candle = True
    if seen_trade_candle and previous_close is not None:
        gap_time = expected_open_time
        while gap_time <= end_ms:
            yield _no_trade_candle(gap_time, previous_close)
            gap_time += 1000


def _download_to_temp(url: str, *, timeout: int, chunk_size: int = 1024 * 1024) -> tuple[Path, int, str]:
    digest = hashlib.sha256()
    handle = tempfile.NamedTemporaryFile(prefix="simple-ai-trading-binance-", suffix=".zip", delete=False)
    path = Path(handle.name)
    bytes_downloaded = 0
    try:
        with handle:
            with urlopen(url, timeout=timeout) as response:  # nosec B310 - official public archive URL
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    bytes_downloaded += len(chunk)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path, bytes_downloaded, digest.hexdigest()


def _checksum_url(url: str) -> str:
    return f"{url}.CHECKSUM"


def _parse_checksum_text(text: str) -> str | None:
    match = _CHECKSUM_PATTERN.search(text)
    return match.group("sha256").lower() if match else None


def _fetch_archive_checksum(url: str, *, timeout: int) -> str | None:
    try:
        with urlopen(_checksum_url(url), timeout=timeout) as response:  # nosec B310 - official public archive URL
            return _parse_checksum_text(response.read(4096).decode("utf-8", errors="ignore"))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        return None
    except (OSError, URLError, ValueError):
        return None


def ingest_archive_url(
    store: MarketDataStore,
    *,
    url: str,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    data_type: str = "klines",
    period: str = "",
    timeout: int = 120,
    chunk_size: int = 10_000,
    force: bool = False,
    verify_checksum: bool = True,
    require_checksum: bool = False,
    fill_period_edges: bool = True,
) -> ArchiveIngestResult:
    symbol = symbol.upper()
    kind = _normalize_archive_data_type(data_type)
    if kind == "aggTrades" and str(interval) != "1s":
        raise ValueError("aggTrades archive ingestion currently emits 1s candles; interval must be '1s'")
    if not force and store.archive_file_status(url) == "complete":
        return ArchiveIngestResult(
            url=url,
            symbol=symbol,
            market_type=market_type,
            interval=interval,
            data_type=kind,
            period=period,
            status="skipped",
            rows_inserted=0,
            rows_read=0,
            bytes_downloaded=0,
            sha256="",
            checksum_sha256="",
            checksum_status="skipped",
        )

    store.begin_archive_file(url=url, symbol=symbol, market_type=market_type, interval=interval, period=period)
    zip_path: Path | None = None
    rows_inserted = 0
    rows_read = 0
    bytes_downloaded = 0
    sha256 = ""
    checksum_sha256 = ""
    checksum_status = "unverified"
    try:
        zip_path, bytes_downloaded, sha256 = _download_to_temp(url, timeout=timeout)
        if verify_checksum:
            checksum_sha256 = _fetch_archive_checksum(url, timeout=max(1, min(timeout, 30))) or ""
            if checksum_sha256:
                if checksum_sha256.lower() != sha256.lower():
                    checksum_status = "mismatch"
                    raise ValueError(
                        f"archive checksum mismatch expected={checksum_sha256.lower()} actual={sha256.lower()}"
                    )
                checksum_status = "verified"
            elif require_checksum:
                checksum_status = "missing"
                raise ValueError(f"archive checksum sidecar missing for {url}")
            else:
                checksum_status = "unavailable"
        batch: list[Candle] = []
        ingested_at_ms = int(time.time() * 1000)
        if kind == "aggTrades":
            candle_iterable = _iter_zip_agg_trade_candles(zip_path)
            period_bounds = _period_ms_bounds(period or archive_url_period(url)) if fill_period_edges else None
            if period_bounds is not None:
                period_start_ms, period_end_ms = period_bounds
                prior_candles = store.fetch_candles(
                    symbol,
                    market_type,
                    interval,
                    end_ms=period_start_ms - 1000,
                    limit=1,
                )
                prior_close = (
                    prior_candles[-1].close
                    if prior_candles and int(prior_candles[-1].open_time) == period_start_ms - 1000
                    else None
                )
                candle_iterable = _iter_period_bounded_agg_trade_candles(
                    candle_iterable,
                    start_ms=period_start_ms,
                    end_ms=period_end_ms,
                    prior_close=prior_close,
                )
        else:
            candle_iterable = _iter_zip_candles(zip_path)
        source = "binance_public_archive_aggTrades" if kind == "aggTrades" else "binance_public_archive"
        for candle in candle_iterable:
            rows_read += 1
            batch.append(candle)
            if len(batch) >= max(1, int(chunk_size)):
                cleaned = clean_candles(batch, drop_unclosed=False)
                rows_inserted += store.upsert_candles(
                    symbol,
                    market_type,
                    interval,
                    cleaned,
                    source=source,
                    ingested_at_ms=ingested_at_ms,
                )
                batch.clear()
        if batch:
            cleaned = clean_candles(batch, drop_unclosed=False)
            rows_inserted += store.upsert_candles(
                symbol,
                market_type,
                interval,
                cleaned,
                source=source,
                ingested_at_ms=ingested_at_ms,
            )
        store.complete_archive_file(
            url=url,
            status="complete",
            rows_inserted=rows_inserted,
            bytes_downloaded=bytes_downloaded,
            sha256=sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
        )
        return ArchiveIngestResult(
            url=url,
            symbol=symbol,
            market_type=market_type,
            interval=interval,
            data_type=kind,
            period=period,
            status="complete",
            rows_inserted=rows_inserted,
            rows_read=rows_read,
            bytes_downloaded=bytes_downloaded,
            sha256=sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
        )
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        store.complete_archive_file(
            url=url,
            status="error",
            rows_inserted=rows_inserted,
            bytes_downloaded=bytes_downloaded,
            sha256=sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
            error=str(exc)[:500],
        )
        return ArchiveIngestResult(
            url=url,
            symbol=symbol,
            market_type=market_type,
            interval=interval,
            data_type=kind,
            period=period,
            status="error",
            rows_inserted=rows_inserted,
            rows_read=rows_read,
            bytes_downloaded=bytes_downloaded,
            sha256=sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
            error=str(exc)[:500],
        )
    finally:
        if zip_path is not None:
            try:
                zip_path.unlink()
            except OSError:
                pass


def ingest_archive_urls(
    *,
    db_path: str | Path,
    symbol: str,
    interval: str,
    urls: Sequence[str],
    market_type: str = "spot",
    data_type: str = "klines",
    timeout: int = 120,
    force: bool = False,
    verify_checksum: bool = True,
    require_checksum: bool = False,
) -> list[ArchiveIngestResult]:
    results: list[ArchiveIngestResult] = []
    kind = _normalize_archive_data_type(data_type)
    with MarketDataStore(db_path) as store:
        for url in urls:
            stem = Path(url).stem
            prefix = f"{symbol.upper()}-aggTrades-" if kind == "aggTrades" else f"{symbol.upper()}-{interval}-"
            period = stem[len(prefix):] if stem.startswith(prefix) else stem.rsplit("-", 1)[-1]
            results.append(
                ingest_archive_url(
                    store,
                    url=url,
                    symbol=symbol,
                    interval=interval,
                    market_type=market_type,
                    data_type=kind,
                    period=period,
                    timeout=timeout,
                    force=force,
                    verify_checksum=verify_checksum,
                    require_checksum=require_checksum,
                )
            )
    return results


__all__ = [
    "ArchiveIngestResult",
    "ArchiveListingItem",
    "BINANCE_ARCHIVE_BASE_URL",
    "archive_listing_items_by_url",
    "archive_directory_url",
    "archive_file_url",
    "archive_listing_url",
    "archive_period_in_range",
    "archive_url_period",
    "_checksum_url",
    "filter_archive_urls_by_period",
    "ingest_archive_url",
    "ingest_archive_urls",
    "list_archive_items",
    "list_archive_urls",
    "validate_archive_period_window",
]
