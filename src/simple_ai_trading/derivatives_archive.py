"""Verified Binance USD-M premium-index and funding archive ingestion."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Callable, Sequence
import zipfile

from .binance_archive import (
    BINANCE_ARCHIVE_BASE_URL,
    _download_to_temp,
    _fetch_archive_checksum,
    _normalize_archive_timestamp,
)
from .market_store import (
    FundingRateRecord,
    FuturesReferenceBar,
    MarketDataStore,
)


DERIVATIVES_DATA_TYPES = ("premiumIndexKlines", "fundingRate")
SUPPORTED_DERIVATIVES_DATA_TYPES = (
    "premiumIndexKlines",
    "markPriceKlines",
    "fundingRate",
)
_REFERENCE_DATA_TYPES = ("premiumIndexKlines", "markPriceKlines")
_SOURCE = "binance_public_archive"
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 250.0


@dataclass(frozen=True)
class DerivativesArchiveIngestResult:
    url: str
    symbol: str
    data_type: str
    interval: str
    period: str
    status: str
    rows_inserted: int
    rows_read: int
    bytes_downloaded: int
    sha256: str
    checksum_sha256: str
    checksum_status: str
    row_stream_sha256: str
    error: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def derivatives_archive_file_url(
    *,
    symbol: str,
    data_type: str,
    period: str,
    interval: str = "1m",
    base_url: str = BINANCE_ARCHIVE_BASE_URL,
) -> str:
    """Build an official monthly USD-M derivatives archive URL."""

    symbol = str(symbol).upper()
    if data_type not in SUPPORTED_DERIVATIVES_DATA_TYPES:
        raise ValueError(f"unsupported derivatives data type: {data_type}")
    root = f"{base_url.rstrip('/')}/futures/um/monthly/{data_type}/{symbol}"
    if data_type in _REFERENCE_DATA_TYPES:
        if interval != "1m":
            raise ValueError(f"{data_type} interval must be 1m")
        return f"{root}/{interval}/{symbol}-{interval}-{period}.zip"
    if interval:
        raise ValueError("fundingRate archives do not use an interval")
    return f"{root}/{symbol}-fundingRate-{period}.zip"


def monthly_periods(start: str, end: str) -> list[str]:
    """Return an inclusive, strictly ordered YYYY-MM period sequence."""

    try:
        start_date = date.fromisoformat(f"{start}-01")
        end_date = date.fromisoformat(f"{end}-01")
    except ValueError as exc:
        raise ValueError("monthly bounds must use YYYY-MM") from exc
    if end_date < start_date:
        raise ValueError("monthly end precedes start")
    values: list[str] = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        values.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return values


def _validated_csv_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = [item for item in archive.infolist() if not item.is_dir()]
    if len(members) != 1:
        raise ValueError("derivatives archive must contain exactly one CSV member")
    member = members[0]
    if Path(member.filename).name != member.filename or not member.filename.endswith(
        ".csv"
    ):
        raise ValueError("derivatives archive member path or extension is invalid")
    if member.file_size <= 0 or member.file_size > _MAX_MEMBER_BYTES:
        raise ValueError("derivatives archive member size is invalid")
    ratio = member.file_size / max(1, member.compress_size)
    if ratio > _MAX_COMPRESSION_RATIO:
        raise ValueError("derivatives archive compression ratio is unsafe")
    return member


def _canonical_row_digest_update(
    digest: hashlib._Hash, values: Sequence[object]
) -> None:
    encoded = json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    digest.update(encoded)
    digest.update(b"\n")


def _is_header(row: Sequence[str]) -> bool:
    if not row:
        return True
    try:
        float(str(row[0]).strip())
    except ValueError:
        return True
    return False


def _parse_reference_archive(
    zip_path: Path,
    *,
    symbol: str,
    data_type: str,
) -> tuple[list[FuturesReferenceBar], str]:
    if data_type not in _REFERENCE_DATA_TYPES:
        raise ValueError(f"unsupported reference data type: {data_type}")
    kind = "premium_index" if data_type == "premiumIndexKlines" else "mark_price"
    records: list[FuturesReferenceBar] = []
    digest = hashlib.sha256()
    previous_time: int | None = None
    with zipfile.ZipFile(zip_path) as archive:
        member = _validated_csv_member(archive)
        with archive.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                if _is_header(row):
                    continue
                if len(row) < 7:
                    raise ValueError("premium-index row has fewer than seven columns")
                open_time = _normalize_archive_timestamp(row[0])
                close_time = _normalize_archive_timestamp(row[6])
                values = tuple(float(row[index]) for index in range(1, 5))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("reference-price row contains a non-finite value")
                open_value, high, low, close = values
                if kind == "mark_price" and any(value <= 0.0 for value in values):
                    raise ValueError("mark-price row contains a nonpositive value")
                if high < max(open_value, low, close) or low > min(
                    open_value, high, close
                ):
                    raise ValueError("reference-price OHLC bounds are invalid")
                if previous_time is not None and open_time <= previous_time:
                    raise ValueError(
                        "reference-price timestamps are not strictly increasing"
                    )
                record = FuturesReferenceBar(
                    symbol=symbol,
                    market_type="futures",
                    kind=kind,
                    interval="1m",
                    open_time=open_time,
                    open=open_value,
                    high=high,
                    low=low,
                    close=close,
                    close_time=close_time,
                )
                records.append(record)
                _canonical_row_digest_update(
                    digest,
                    (open_time, open_value, high, low, close, close_time),
                )
                previous_time = open_time
    if not records:
        raise ValueError("reference-price archive contains no data rows")
    return records, digest.hexdigest()


def _parse_funding_archive(
    zip_path: Path,
    *,
    symbol: str,
) -> tuple[list[FundingRateRecord], str]:
    records: list[FundingRateRecord] = []
    digest = hashlib.sha256()
    previous_time: int | None = None
    with zipfile.ZipFile(zip_path) as archive:
        member = _validated_csv_member(archive)
        with archive.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                if _is_header(row):
                    continue
                if len(row) < 3:
                    raise ValueError("funding row has fewer than three columns")
                calc_time = _normalize_archive_timestamp(row[0])
                interval_hours = int(row[1])
                rate = float(row[2])
                if not math.isfinite(rate) or abs(rate) > 0.1:
                    raise ValueError(
                        "funding rate is non-finite or outside sanity bounds"
                    )
                if not 1 <= interval_hours <= 8:
                    raise ValueError(
                        "funding interval is outside the frozen 1..8h range"
                    )
                if previous_time is not None and calc_time <= previous_time:
                    raise ValueError("funding timestamps are not strictly increasing")
                record = FundingRateRecord(
                    symbol=symbol,
                    market_type="futures",
                    calc_time=calc_time,
                    funding_interval_hours=interval_hours,
                    funding_rate=rate,
                )
                records.append(record)
                _canonical_row_digest_update(digest, (calc_time, interval_hours, rate))
                previous_time = calc_time
    if not records:
        raise ValueError("funding archive contains no data rows")
    return records, digest.hexdigest()


def ingest_derivatives_archive_url(
    store: MarketDataStore,
    *,
    url: str,
    symbol: str,
    data_type: str,
    period: str,
    interval: str = "1m",
    timeout: int = 120,
    force: bool = False,
) -> DerivativesArchiveIngestResult:
    """Verify, parse, and atomically store one official derivatives archive."""

    symbol = symbol.upper()
    if data_type not in SUPPORTED_DERIVATIVES_DATA_TYPES:
        raise ValueError(f"unsupported derivatives data type: {data_type}")
    stored_interval = interval if data_type in _REFERENCE_DATA_TYPES else ""
    if not force and store.derivatives_archive_file_status(url) == "complete":
        return DerivativesArchiveIngestResult(
            url=url,
            symbol=symbol,
            data_type=data_type,
            interval=stored_interval,
            period=period,
            status="skipped",
            rows_inserted=0,
            rows_read=0,
            bytes_downloaded=0,
            sha256="",
            checksum_sha256="",
            checksum_status="skipped",
            row_stream_sha256="",
        )
    store.begin_derivatives_archive_file(
        url=url,
        symbol=symbol,
        market_type="futures",
        data_type=data_type,
        interval=stored_interval,
        period=period,
    )
    zip_path: Path | None = None
    rows_inserted = 0
    rows_read = 0
    bytes_downloaded = 0
    zip_sha256 = ""
    checksum_sha256 = ""
    checksum_status = "unverified"
    row_stream_sha256 = ""
    try:
        zip_path, bytes_downloaded, zip_sha256 = _download_to_temp(url, timeout=timeout)
        checksum_sha256 = (
            _fetch_archive_checksum(url, timeout=max(1, min(timeout, 30))) or ""
        )
        if not checksum_sha256:
            checksum_status = "missing"
            raise ValueError("required archive checksum sidecar is unavailable")
        if checksum_sha256.lower() != zip_sha256.lower():
            checksum_status = "mismatch"
            raise ValueError(
                "archive checksum mismatch "
                f"expected={checksum_sha256.lower()} actual={zip_sha256.lower()}"
            )
        checksum_status = "verified"
        ingested_at_ms = int(time.time() * 1000)
        if data_type in _REFERENCE_DATA_TYPES:
            reference, row_stream_sha256 = _parse_reference_archive(
                zip_path, symbol=symbol, data_type=data_type
            )
            rows_read = len(reference)
            rows_inserted = store.upsert_futures_reference_bars(
                reference,
                source=f"{_SOURCE}_{data_type}",
                ingested_at_ms=ingested_at_ms,
            )
        else:
            funding, row_stream_sha256 = _parse_funding_archive(zip_path, symbol=symbol)
            rows_read = len(funding)
            rows_inserted = store.upsert_funding_rates(
                funding,
                source=f"{_SOURCE}_fundingRate",
                ingested_at_ms=ingested_at_ms,
            )
        store.complete_derivatives_archive_file(
            url=url,
            status="complete",
            rows_inserted=rows_inserted,
            rows_read=rows_read,
            bytes_downloaded=bytes_downloaded,
            sha256=zip_sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
            row_stream_sha256=row_stream_sha256,
        )
        return DerivativesArchiveIngestResult(
            url=url,
            symbol=symbol,
            data_type=data_type,
            interval=stored_interval,
            period=period,
            status="complete",
            rows_inserted=rows_inserted,
            rows_read=rows_read,
            bytes_downloaded=bytes_downloaded,
            sha256=zip_sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
            row_stream_sha256=row_stream_sha256,
        )
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        store.complete_derivatives_archive_file(
            url=url,
            status="error",
            rows_inserted=rows_inserted,
            rows_read=rows_read,
            bytes_downloaded=bytes_downloaded,
            sha256=zip_sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
            row_stream_sha256=row_stream_sha256,
            error=str(exc),
        )
        return DerivativesArchiveIngestResult(
            url=url,
            symbol=symbol,
            data_type=data_type,
            interval=stored_interval,
            period=period,
            status="error",
            rows_inserted=rows_inserted,
            rows_read=rows_read,
            bytes_downloaded=bytes_downloaded,
            sha256=zip_sha256,
            checksum_sha256=checksum_sha256,
            checksum_status=checksum_status,
            row_stream_sha256=row_stream_sha256,
            error=str(exc)[:500],
        )
    finally:
        if zip_path is not None:
            try:
                zip_path.unlink()
            except OSError:
                pass


def ingest_derivatives_archive_range(
    *,
    db_path: str | Path,
    symbols: Sequence[str],
    start_period: str,
    end_period: str,
    timeout: int = 120,
    force: bool = False,
    progress: Callable[[DerivativesArchiveIngestResult], None] | None = None,
) -> list[DerivativesArchiveIngestResult]:
    """Ingest the fixed premium/funding monthly range without retaining ZIPs."""

    periods = monthly_periods(start_period, end_period)
    results: list[DerivativesArchiveIngestResult] = []
    with MarketDataStore(db_path) as store:
        for symbol in (str(item).upper() for item in symbols):
            for data_type in DERIVATIVES_DATA_TYPES:
                interval = "1m" if data_type == "premiumIndexKlines" else ""
                for period in periods:
                    url = derivatives_archive_file_url(
                        symbol=symbol,
                        data_type=data_type,
                        period=period,
                        interval=interval,
                    )
                    result = ingest_derivatives_archive_url(
                        store,
                        url=url,
                        symbol=symbol,
                        data_type=data_type,
                        period=period,
                        interval=interval,
                        timeout=timeout,
                        force=force,
                    )
                    results.append(result)
                    if progress is not None:
                        progress(result)
                    if result.status == "error":
                        raise RuntimeError(
                            f"derivatives archive ingestion failed: {url}: {result.error}"
                        )
    return results


__all__ = [
    "DERIVATIVES_DATA_TYPES",
    "SUPPORTED_DERIVATIVES_DATA_TYPES",
    "DerivativesArchiveIngestResult",
    "derivatives_archive_file_url",
    "ingest_derivatives_archive_range",
    "ingest_derivatives_archive_url",
    "monthly_periods",
]
