"""Binance public data archive ingestion into the market-data store."""

from __future__ import annotations

import csv
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
from urllib.parse import urlencode
from urllib.request import urlopen

from defusedxml import ElementTree

from .api import Candle
from .market_data import clean_candles
from .market_store import MarketDataStore


BINANCE_ARCHIVE_BASE_URL = "https://data.binance.vision/data"
_ZIP_LINK_PATTERN = re.compile(r'href=["\'](?P<href>[^"\']+\.zip)["\']', re.IGNORECASE)
_CHECKSUM_PATTERN = re.compile(r"\b(?P<sha256>[a-fA-F0-9]{64})\b")


@dataclass(frozen=True)
class ArchiveIngestResult:
    url: str
    symbol: str
    market_type: str
    interval: str
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


def _archive_market_segment(market_type: str) -> str:
    market = str(market_type or "spot").lower()
    if market == "spot":
        return "spot"
    if market == "futures":
        return "futures/um"
    raise ValueError("market_type must be 'spot' or 'futures'")


def archive_directory_url(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
) -> str:
    cadence = str(cadence or "monthly").lower()
    if cadence not in {"daily", "monthly"}:
        raise ValueError("cadence must be 'daily' or 'monthly'")
    segment = _archive_market_segment(market_type)
    return f"{base_url.rstrip('/')}/{segment}/{cadence}/klines/{symbol.upper()}/{interval}/"


def archive_listing_url(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
    marker: str | None = None,
) -> str:
    cadence = str(cadence or "monthly").lower()
    if cadence not in {"daily", "monthly"}:
        raise ValueError("cadence must be 'daily' or 'monthly'")
    segment = _archive_market_segment(market_type)
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
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
) -> str:
    directory = archive_directory_url(
        symbol=symbol,
        interval=interval,
        market_type=market_type,
        cadence=cadence,
        base_url=base_url,
    )
    return f"{directory}{symbol.upper()}-{interval}-{period}.zip"


def list_archive_urls(
    *,
    symbol: str,
    interval: str,
    market_type: str = "spot",
    cadence: str = "monthly",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
    timeout: int = 20,
    html_loader: Callable[[str], str] | None = None,
) -> list[str]:
    origin = base_url.rstrip("/").removesuffix("/data")
    urls: list[str] = []
    marker: str | None = None
    while True:
        listing = archive_listing_url(
            symbol=symbol,
            interval=interval,
            market_type=market_type,
            cadence=cadence,
            base_url=base_url,
            marker=marker,
        )
        if html_loader is None:
            with urlopen(listing, timeout=timeout) as response:  # nosec B310 - official public archive URL
                listing_text = response.read().decode("utf-8", errors="ignore")
        else:
            listing_text = html_loader(listing)
        keys, next_marker, truncated = _parse_listing_keys(listing_text)
        for key in keys:
            if key.endswith(".zip") and not key.endswith(".zip.CHECKSUM"):
                urls.append(f"{origin}/{key}")
        if not truncated:
            break
        marker = next_marker or (keys[-1] if keys else None)
        if marker is None:
            break
    return sorted(dict.fromkeys(urls))


def _xml_text(element: object, tag: str) -> str:
    if element is None:
        return ""
    found = element.find(f"{{*}}{tag}")  # type: ignore[attr-defined]
    return str(found.text or "") if found is not None else ""


def _parse_listing_keys(listing_text: str) -> tuple[list[str], str | None, bool]:
    try:
        root = ElementTree.fromstring(listing_text)
    except ElementTree.ParseError:
        links = [match.group("href") for match in _ZIP_LINK_PATTERN.finditer(listing_text)]
        return links, None, False
    keys: list[str] = []
    for contents in root.findall(".//{*}Contents"):
        key = _xml_text(contents, "Key")
        if key:
            keys.append(key)
    next_marker = _xml_text(root, "NextMarker") or None
    truncated = _xml_text(root, "IsTruncated").strip().lower() == "true"
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
    period: str = "",
    timeout: int = 120,
    chunk_size: int = 10_000,
    force: bool = False,
    verify_checksum: bool = True,
    require_checksum: bool = False,
) -> ArchiveIngestResult:
    symbol = symbol.upper()
    if not force and store.archive_file_status(url) == "complete":
        return ArchiveIngestResult(
            url=url,
            symbol=symbol,
            market_type=market_type,
            interval=interval,
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
        for candle in _iter_zip_candles(zip_path):
            rows_read += 1
            batch.append(candle)
            if len(batch) >= max(1, int(chunk_size)):
                cleaned = clean_candles(batch, drop_unclosed=False)
                rows_inserted += store.upsert_candles(
                    symbol,
                    market_type,
                    interval,
                    cleaned,
                    source="binance_public_archive",
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
                source="binance_public_archive",
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
    timeout: int = 120,
    force: bool = False,
    verify_checksum: bool = True,
    require_checksum: bool = False,
) -> list[ArchiveIngestResult]:
    results: list[ArchiveIngestResult] = []
    with MarketDataStore(db_path) as store:
        for url in urls:
            stem = Path(url).stem
            prefix = f"{symbol.upper()}-{interval}-"
            period = stem[len(prefix):] if stem.startswith(prefix) else stem.rsplit("-", 1)[-1]
            results.append(
                ingest_archive_url(
                    store,
                    url=url,
                    symbol=symbol,
                    interval=interval,
                    market_type=market_type,
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
    "BINANCE_ARCHIVE_BASE_URL",
    "archive_directory_url",
    "archive_file_url",
    "archive_listing_url",
    "_checksum_url",
    "ingest_archive_url",
    "ingest_archive_urls",
    "list_archive_urls",
]
