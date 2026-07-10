"""Checksummed Binance tick archives in a bounded embedded DuckDB warehouse."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
from typing import Callable, Iterator, Mapping
from urllib.parse import urlparse
import zipfile

import duckdb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .assets import is_supported_major_symbol, normalize_symbol
from .binance_archive import archive_file_url


TICK_WAREHOUSE_SCHEMA_VERSION = "binance-usdm-tick-v5"
BOOK_TICKER_FEATURE_BUILD_VERSION = "book-ticker-event-time-v1"
SUPPORTED_TICK_ARCHIVES = frozenset({"bookTicker", "trades", "bookDepth"})
_CHECKSUM_PATTERN = re.compile(r"\b([0-9a-fA-F]{64})\b")
_MEMORY_LIMIT_PATTERN = re.compile(r"^[1-9][0-9]*(?:MB|GB)$", re.IGNORECASE)
_BOOK_TICKER_HEADER = (
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
)
_TRADES_HEADER = ("id", "price", "qty", "quote_qty", "time", "is_buyer_maker")
_BOOK_DEPTH_HEADER = ("timestamp", "percentage", "depth", "notional")


ProgressCallback = Callable[[str, int, int | None], None]


def _resilient_http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    session.mount("https://", adapter)
    return session


@dataclass(frozen=True)
class TickArchiveIngestResult:
    archive_id: str
    status: str
    symbol: str
    data_type: str
    period: str
    url: str
    archive_path: str
    source_sha256: str
    expected_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int
    rows_read: int
    derived_rows: int
    first_exchange_time_ms: int | None
    last_exchange_time_ms: int | None
    invalid_rows: int
    duplicate_ids: int
    update_id_regressions: int
    event_time_regressions: int
    out_of_order_rows: int
    crossed_books: int
    error: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_id(url: str, source_sha256: str) -> str:
    value = f"{url}\n{source_sha256.lower()}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _period_bounds_ms(period: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("tick archives require a daily YYYY-MM-DD period") from exc
    start = int(parsed.timestamp() * 1000)
    return start, start + 86_400_000 - 1


def _normalize_tick_request(symbol: str, data_type: str, period: str) -> tuple[str, str, str]:
    normalized_symbol = normalize_symbol(symbol)
    if not is_supported_major_symbol(normalized_symbol):
        raise ValueError(f"unsupported tick archive symbol: {normalized_symbol}")
    normalized_type = str(data_type or "").strip()
    if normalized_type not in SUPPORTED_TICK_ARCHIVES:
        raise ValueError(f"unsupported tick archive data_type: {normalized_type}")
    _period_bounds_ms(period)
    return normalized_symbol, normalized_type, period


def official_tick_archive_url(*, symbol: str, data_type: str, period: str) -> str:
    symbol, data_type, period = _normalize_tick_request(symbol, data_type, period)
    return archive_file_url(
        symbol=symbol,
        interval="tick",
        period=period,
        market_type="futures",
        cadence="daily",
        data_type=data_type,
    )


def _validate_official_url(url: str, *, symbol: str, data_type: str, period: str) -> None:
    expected = official_tick_archive_url(symbol=symbol, data_type=data_type, period=period)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "data.binance.vision" or url != expected:
        raise ValueError(f"unexpected Binance archive URL: expected {expected}")


def _parse_checksum(text: str) -> str:
    match = _CHECKSUM_PATTERN.search(text)
    if match is None:
        raise ValueError("Binance checksum sidecar does not contain SHA-256")
    return match.group(1).lower()


def _fetch_checksum(session: requests.Session, url: str, *, timeout_seconds: float) -> str:
    response = session.get(f"{url}.CHECKSUM", timeout=max(1.0, timeout_seconds))
    response.raise_for_status()
    return _parse_checksum(response.text[:4096])


def _safe_archive_path(cache_root: Path, *, symbol: str, data_type: str, period: str) -> Path:
    filename = f"{symbol}-{data_type}-{period}.zip"
    return cache_root / "binance" / "usdm" / data_type / symbol / filename


def _download_verified_archive(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    timeout_seconds: float,
    expected_bytes: int = 0,
    max_download_bytes: int = 8 * 1024**3,
    progress: ProgressCallback | None = None,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_hash = _sha256_file(destination)
        if existing_hash == expected_sha256:
            if progress:
                progress("archive-cache-hit", destination.stat().st_size, destination.stat().st_size)
            return destination.stat().st_size, existing_hash
        quarantine = destination.with_name(
            f"{destination.name}.checksum-mismatch-{int(time.time())}"
        )
        destination.replace(quarantine)

    byte_limit = max(1, int(max_download_bytes))
    if expected_bytes > 0:
        byte_limit = min(byte_limit, max(expected_bytes + 1024 * 1024, int(expected_bytes * 1.05)))
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with session.get(url, stream=True, timeout=max(1.0, timeout_seconds)) as response:
            response.raise_for_status()
            response_length = int(response.headers.get("Content-Length", "0") or 0)
            if response_length > byte_limit:
                raise ValueError(
                    f"archive exceeds bounded download size: {response_length}>{byte_limit}"
                )
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > byte_limit:
                        raise ValueError(
                            f"archive exceeded bounded download size: {downloaded}>{byte_limit}"
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                    if progress:
                        progress("archive-download", downloaded, response_length or expected_bytes or None)
                handle.flush()
                os.fsync(handle.fileno())
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"archive checksum mismatch expected={expected_sha256} actual={actual_sha256}"
            )
        temporary.replace(destination)
        return downloaded, actual_sha256
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _expected_header(data_type: str) -> tuple[str, ...]:
    if data_type == "bookTicker":
        return _BOOK_TICKER_HEADER
    if data_type == "trades":
        return _TRADES_HEADER
    if data_type == "bookDepth":
        return _BOOK_DEPTH_HEADER
    raise ValueError(f"unsupported tick archive data_type: {data_type}")


def _inspect_zip(path: Path, *, data_type: str, max_uncompressed_bytes: int) -> tuple[zipfile.ZipInfo, bool]:
    expected_header = _expected_header(data_type)
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"archive must contain exactly one data file, found {len(members)}")
        member = members[0]
        member_path = Path(member.filename)
        if member_path.name != member.filename or member_path.suffix.lower() != ".csv":
            raise ValueError(f"unsafe or unexpected ZIP member: {member.filename}")
        if member.flag_bits & 0x1:
            raise ValueError("encrypted ZIP members are not supported")
        if member.file_size <= 0 or member.file_size > max(1, int(max_uncompressed_bytes)):
            raise ValueError(
                f"archive uncompressed size outside bounds: {member.file_size}"
            )
        if member.compress_size <= 0 or member.file_size / member.compress_size > 200.0:
            raise ValueError("archive compression ratio exceeds safety bound")
        with archive.open(member) as raw:
            first_line = raw.readline(16 * 1024)
        try:
            columns = tuple(
                item.strip().lower() for item in first_line.decode("utf-8-sig").strip().split(",")
            )
        except UnicodeDecodeError as exc:
            raise ValueError("archive CSV header is not UTF-8") from exc
        has_header = columns == expected_header
        if not has_header:
            if data_type != "trades" or len(columns) != len(expected_header):
                raise ValueError(
                    f"unexpected {data_type} CSV header: {','.join(columns[:10])}"
                )
            try:
                int(columns[0])
                float(columns[1])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"unexpected {data_type} CSV header: {','.join(columns[:10])}"
                ) from exc
        return member, has_header


def _extract_member(path: Path, member: zipfile.ZipInfo, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    try:
        with zipfile.ZipFile(path) as archive, archive.open(member) as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


@contextmanager
def _exclusive_operation_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at_ms": int(time.time() * 1000),
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        detail = lock_path.read_text(encoding="utf-8", errors="replace")[:500]
        try:
            existing = json.loads(detail)
            existing_pid = int(existing.get("pid", 0))
            existing_host = str(existing.get("host", ""))
            started_at_ms = int(existing.get("started_at_ms", 0))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            existing_pid = 0
            existing_host = ""
            started_at_ms = 0
        age_ms = int(time.time() * 1000) - started_at_ms
        if (
            existing_host == socket.gethostname()
            and existing_pid > 0
            and age_ms >= 30_000
            and not _pid_is_running(existing_pid)
        ):
            stale = lock_path.with_name(f"{lock_path.name}.stale-{int(time.time())}")
            try:
                lock_path.replace(stale)
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except OSError as recovery_exc:
                raise RuntimeError(
                    f"could not recover stale tick warehouse writer lock: {detail}"
                ) from recovery_exc
        else:
            raise RuntimeError(f"tick warehouse writer is already active: {detail}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class MicrostructureWarehouse:
    """One-writer, compressed tick warehouse for public Binance futures archives."""

    def __init__(
        self,
        path: str | Path = "data/microstructure.duckdb",
        *,
        cache_root: str | Path = "data/archive-cache",
        memory_limit: str = "8GB",
        threads: int = 8,
    ) -> None:
        if not _MEMORY_LIMIT_PATTERN.fullmatch(str(memory_limit).strip()):
            raise ValueError("memory_limit must be a positive integer followed by MB or GB")
        self.path = Path(path)
        self.cache_root = Path(cache_root)
        self.memory_limit = str(memory_limit).upper()
        self.threads = max(1, min(32, int(threads)))
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._reconciled = False

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.path))
            self._conn.execute(f"SET memory_limit='{self.memory_limit}'")
            self._conn.execute(f"SET threads={self.threads}")
            self._conn.execute("SET TimeZone='UTC'")
            self._conn.execute("SET preserve_insertion_order=false")
            self._init_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MicrostructureWarehouse":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        conn = self._conn
        if conn is None:
            raise RuntimeError("warehouse connection is not available")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS archive_manifest (
                archive_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                market_type VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                data_type VARCHAR NOT NULL,
                period VARCHAR NOT NULL,
                url VARCHAR NOT NULL,
                archive_path VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                is_current BOOLEAN NOT NULL,
                expected_bytes UBIGINT NOT NULL,
                compressed_bytes UBIGINT NOT NULL,
                uncompressed_bytes UBIGINT NOT NULL,
                source_sha256 VARCHAR NOT NULL,
                expected_sha256 VARCHAR NOT NULL,
                checksum_status VARCHAR NOT NULL,
                rows_read UBIGINT NOT NULL,
                derived_rows UBIGINT NOT NULL,
                first_exchange_time_ms BIGINT,
                last_exchange_time_ms BIGINT,
                invalid_rows UBIGINT NOT NULL,
                duplicate_ids UBIGINT NOT NULL,
                update_id_regressions UBIGINT NOT NULL,
                event_time_regressions UBIGINT NOT NULL,
                out_of_order_rows UBIGINT NOT NULL,
                crossed_books UBIGINT NOT NULL,
                ingested_at_ms BIGINT NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS book_ticker_raw (
                archive_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                update_id UBIGINT NOT NULL,
                bid_price DOUBLE NOT NULL,
                bid_qty DOUBLE NOT NULL,
                ask_price DOUBLE NOT NULL,
                ask_qty DOUBLE NOT NULL,
                transaction_time_ms BIGINT NOT NULL,
                event_time_ms BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_raw (
                archive_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                trade_id UBIGINT NOT NULL,
                price DOUBLE NOT NULL,
                qty DOUBLE NOT NULL,
                quote_qty DOUBLE NOT NULL,
                trade_time_ms BIGINT NOT NULL,
                buyer_is_maker BOOLEAN NOT NULL
            );

            CREATE TABLE IF NOT EXISTS book_depth_aggregate_raw (
                archive_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                timestamp_ms BIGINT NOT NULL,
                percentage DECIMAL(4,2) NOT NULL,
                depth DOUBLE NOT NULL,
                notional DOUBLE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS book_ticker_feature_1s (
                build_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                second_ms BIGINT NOT NULL,
                open_mid DOUBLE NOT NULL,
                high_mid DOUBLE NOT NULL,
                low_mid DOUBLE NOT NULL,
                close_mid DOUBLE NOT NULL,
                close_bid DOUBLE NOT NULL,
                close_ask DOUBLE NOT NULL,
                close_bid_qty DOUBLE NOT NULL,
                close_ask_qty DOUBLE NOT NULL,
                event_weighted_spread_bps DOUBLE NOT NULL,
                max_spread_bps DOUBLE NOT NULL,
                event_weighted_l1_imbalance DOUBLE NOT NULL,
                close_l1_imbalance DOUBLE NOT NULL,
                event_weighted_microprice_offset_bps DOUBLE NOT NULL,
                quote_updates UINTEGER NOT NULL,
                event_delay_p50_ms DOUBLE NOT NULL,
                event_delay_p99_ms DOUBLE NOT NULL,
                event_delay_max_ms BIGINT NOT NULL,
                first_event_time_ms BIGINT NOT NULL,
                last_event_time_ms BIGINT NOT NULL,
                last_transaction_time_ms BIGINT NOT NULL,
                source_archive_count UINTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS book_ticker_feature_build_audit (
                build_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                feature_build_version VARCHAR NOT NULL,
                availability_clock VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                is_current BOOLEAN NOT NULL,
                manifest_fingerprint VARCHAR NOT NULL,
                source_archive_count UINTEGER NOT NULL,
                source_manifest_rows UBIGINT NOT NULL,
                source_raw_rows UBIGINT NOT NULL,
                first_transaction_time_ms BIGINT,
                last_transaction_time_ms BIGINT,
                first_event_time_ms BIGINT,
                last_event_time_ms BIGINT,
                feature_rows UBIGINT NOT NULL,
                first_feature_second_ms BIGINT,
                last_feature_second_ms BIGINT,
                duplicate_seconds UBIGINT NOT NULL,
                invalid_feature_rows UBIGINT NOT NULL,
                quote_update_sum UBIGINT NOT NULL,
                built_at_ms BIGINT NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS terminal_holdout_audit (
                reservation_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                first_utc_day BIGINT NOT NULL,
                last_utc_day BIGINT NOT NULL,
                candidate_sha256 VARCHAR NOT NULL,
                source_manifest_fingerprint VARCHAR NOT NULL,
                source_feature_build_id VARCHAR NOT NULL,
                feature_version VARCHAR NOT NULL,
                model_schema_version VARCHAR NOT NULL,
                prequential_report_sha256 VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                reserved_at_ms BIGINT NOT NULL,
                completed_at_ms BIGINT,
                result_status VARCHAR NOT NULL,
                error VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS book_ticker_path_1s (
                archive_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                second_ms BIGINT NOT NULL,
                min_bid DOUBLE NOT NULL,
                max_bid DOUBLE NOT NULL,
                close_bid DOUBLE NOT NULL,
                min_ask DOUBLE NOT NULL,
                max_ask DOUBLE NOT NULL,
                close_ask DOUBLE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS book_ticker_100ms (
                archive_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                bucket_ms BIGINT NOT NULL,
                min_bid DOUBLE NOT NULL,
                max_bid DOUBLE NOT NULL,
                close_bid DOUBLE NOT NULL,
                close_bid_qty DOUBLE NOT NULL,
                min_ask DOUBLE NOT NULL,
                max_ask DOUBLE NOT NULL,
                close_ask DOUBLE NOT NULL,
                close_ask_qty DOUBLE NOT NULL,
                last_transaction_time_ms BIGINT NOT NULL,
                last_event_time_ms BIGINT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_1s (
                archive_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                second_ms BIGINT NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                base_volume DOUBLE NOT NULL,
                quote_volume DOUBLE NOT NULL,
                aggressive_buy_volume DOUBLE NOT NULL,
                aggressive_sell_volume DOUBLE NOT NULL,
                trade_imbalance DOUBLE NOT NULL,
                trade_count UINTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orphan_reconciliation_audit (
                reconciled_at_ms BIGINT NOT NULL,
                table_name VARCHAR NOT NULL,
                archive_id VARCHAR NOT NULL,
                row_count UBIGINT NOT NULL,
                reason VARCHAR NOT NULL
            );

            CREATE OR REPLACE VIEW current_book_ticker_raw AS
                SELECT q.* FROM book_ticker_raw q
                JOIN archive_manifest m USING (archive_id)
                WHERE m.status = 'complete' AND m.is_current;
            CREATE OR REPLACE VIEW current_trade_raw AS
                SELECT t.* FROM trade_raw t
                JOIN archive_manifest m USING (archive_id)
                WHERE m.status = 'complete' AND m.is_current;
            CREATE OR REPLACE VIEW current_book_depth_aggregate_raw AS
                SELECT d.* FROM book_depth_aggregate_raw d
                JOIN archive_manifest m USING (archive_id)
                WHERE m.status = 'complete' AND m.is_current;
            CREATE OR REPLACE VIEW current_book_depth_snapshots AS
                WITH pivoted AS (
                    SELECT
                        symbol,
                        timestamp_ms,
                        max(depth) FILTER (WHERE percentage = -0.20) AS bid_depth_0_2,
                        max(depth) FILTER (WHERE percentage = 0.20) AS ask_depth_0_2,
                        max(notional) FILTER (WHERE percentage = -0.20) AS bid_notional_0_2,
                        max(notional) FILTER (WHERE percentage = 0.20) AS ask_notional_0_2,
                        max(depth) FILTER (WHERE percentage = -1.00) AS bid_depth_1,
                        max(depth) FILTER (WHERE percentage = 1.00) AS ask_depth_1,
                        max(notional) FILTER (WHERE percentage = -1.00) AS bid_notional_1,
                        max(notional) FILTER (WHERE percentage = 1.00) AS ask_notional_1,
                        max(depth) FILTER (WHERE percentage = -5.00) AS bid_depth_5,
                        max(depth) FILTER (WHERE percentage = 5.00) AS ask_depth_5,
                        max(notional) FILTER (WHERE percentage = -5.00) AS bid_notional_5,
                        max(notional) FILTER (WHERE percentage = 5.00) AS ask_notional_5,
                        count(*) AS band_count
                    FROM current_book_depth_aggregate_raw
                    GROUP BY symbol, timestamp_ms
                )
                SELECT
                    *,
                    (bid_depth_0_2 - ask_depth_0_2)
                        / nullif(bid_depth_0_2 + ask_depth_0_2, 0) AS depth_imbalance_0_2,
                    (bid_depth_1 - ask_depth_1)
                        / nullif(bid_depth_1 + ask_depth_1, 0) AS depth_imbalance_1,
                    (bid_depth_5 - ask_depth_5)
                        / nullif(bid_depth_5 + ask_depth_5, 0) AS depth_imbalance_5
                FROM pivoted
                WHERE band_count = 12;
            CREATE OR REPLACE VIEW current_book_ticker_1s AS
                SELECT
                    q.symbol, q.second_ms, q.open_mid, q.high_mid, q.low_mid,
                    q.close_mid, q.close_bid, q.close_ask, q.close_bid_qty,
                    q.close_ask_qty, q.event_weighted_spread_bps,
                    q.max_spread_bps, q.event_weighted_l1_imbalance,
                    q.close_l1_imbalance, q.event_weighted_microprice_offset_bps,
                    q.quote_updates, q.event_delay_p50_ms, q.event_delay_p99_ms,
                    q.event_delay_max_ms, q.first_event_time_ms,
                    q.last_event_time_ms, q.last_transaction_time_ms,
                    q.source_archive_count
                FROM book_ticker_feature_1s q
                JOIN book_ticker_feature_build_audit a USING (build_id, symbol)
                WHERE a.status = 'complete' AND a.is_current;
            CREATE OR REPLACE VIEW current_book_ticker_path_1s AS
                SELECT q.* FROM book_ticker_path_1s q
                JOIN archive_manifest m USING (archive_id)
                WHERE m.status = 'complete' AND m.is_current;
            CREATE OR REPLACE VIEW current_book_ticker_100ms AS
                SELECT q.*, q.bucket_ms + 100 AS available_time_ms
                FROM book_ticker_100ms q
                JOIN archive_manifest m USING (archive_id)
                WHERE m.status = 'complete' AND m.is_current;
            CREATE OR REPLACE VIEW current_trade_1s AS
                SELECT t.* FROM trade_1s t
                JOIN archive_manifest m USING (archive_id)
                WHERE m.status = 'complete' AND m.is_current;
            CREATE OR REPLACE VIEW current_trade_depth_1s AS
                SELECT
                    t.*,
                    d.timestamp_ms AS depth_time_ms,
                    t.second_ms - d.timestamp_ms AS depth_age_ms,
                    d.bid_depth_0_2, d.ask_depth_0_2,
                    d.bid_notional_0_2, d.ask_notional_0_2,
                    d.bid_depth_1, d.ask_depth_1,
                    d.bid_notional_1, d.ask_notional_1,
                    d.bid_depth_5, d.ask_depth_5,
                    d.bid_notional_5, d.ask_notional_5,
                    d.depth_imbalance_0_2,
                    d.depth_imbalance_1,
                    d.depth_imbalance_5
                FROM current_trade_1s t
                ASOF LEFT JOIN current_book_depth_snapshots d
                  ON t.symbol = d.symbol AND t.second_ms >= d.timestamp_ms;
            CREATE OR REPLACE VIEW microstructure_1s AS
                SELECT
                    q.symbol,
                    q.second_ms,
                    q.open_mid,
                    q.high_mid,
                    q.low_mid,
                    q.close_mid,
                    q.close_bid,
                    q.close_ask,
                    q.close_bid_qty,
                    q.close_ask_qty,
                    q.event_weighted_spread_bps,
                    q.max_spread_bps,
                    q.event_weighted_l1_imbalance,
                    q.close_l1_imbalance,
                    q.event_weighted_microprice_offset_bps,
                    q.quote_updates,
                    q.event_delay_p50_ms,
                    q.event_delay_p99_ms,
                    q.event_delay_max_ms,
                    t.open AS trade_open,
                    t.high AS trade_high,
                    t.low AS trade_low,
                    t.close AS trade_close,
                    t.base_volume,
                    t.quote_volume,
                    t.aggressive_buy_volume,
                    t.aggressive_sell_volume,
                    t.trade_imbalance,
                    t.trade_count
                FROM current_book_ticker_1s q
                LEFT JOIN current_trade_1s t USING (symbol, second_ms);
            """
        )
        manifest_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info('archive_manifest')").fetchall()
        }
        if "update_id_regressions" not in manifest_columns:
            conn.execute(
                "ALTER TABLE archive_manifest ADD COLUMN update_id_regressions UBIGINT DEFAULT 0"
            )
        if "event_time_regressions" not in manifest_columns:
            conn.execute(
                "ALTER TABLE archive_manifest ADD COLUMN event_time_regressions UBIGINT DEFAULT 0"
            )
        book_depth_columns = {
            str(row[1]): str(row[2]).upper()
            for row in conn.execute(
                "PRAGMA table_info('book_depth_aggregate_raw')"
            ).fetchall()
        }
        if book_depth_columns.get("percentage") != "DECIMAL(4,2)":
            conn.execute(
                "ALTER TABLE book_depth_aggregate_raw "
                "ALTER COLUMN percentage TYPE DECIMAL(4,2)"
            )
        terminal_columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info('terminal_holdout_audit')"
            ).fetchall()
        }
        if "prequential_report_sha256" not in terminal_columns:
            conn.execute(
                "ALTER TABLE terminal_holdout_audit "
                "ADD COLUMN prequential_report_sha256 VARCHAR DEFAULT ''"
            )

    def reconcile_orphan_rows(self) -> dict[str, int]:
        """Audit and remove rows that cannot be traced to an archive manifest."""

        conn = self.connect()
        tables = (
            "book_ticker_raw",
            "trade_raw",
            "book_depth_aggregate_raw",
            "book_ticker_path_1s",
            "book_ticker_100ms",
            "trade_1s",
        )
        removed: dict[str, int] = {}
        now_ms = int(time.time() * 1000)
        conn.execute("BEGIN TRANSACTION")
        try:
            for table in tables:
                rows = conn.execute(
                    f"""
                    SELECT r.archive_id, count(*)::UBIGINT
                    FROM {table} r
                    LEFT JOIN archive_manifest m USING (archive_id)
                    WHERE m.archive_id IS NULL
                    GROUP BY r.archive_id
                    """
                ).fetchall()
                count = sum(int(row[1]) for row in rows)
                removed[table] = count
                if rows:
                    conn.executemany(
                        "INSERT INTO orphan_reconciliation_audit VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                now_ms,
                                table,
                                str(archive_id),
                                int(row_count),
                                "manifest_missing_after_interrupted_ingestion",
                            )
                            for archive_id, row_count in rows
                        ],
                    )
                    conn.execute(
                        f"""
                        DELETE FROM {table}
                        WHERE archive_id NOT IN (SELECT archive_id FROM archive_manifest)
                        """
                    )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        self._reconciled = True
        return removed

    def _completed_manifest(self, archive_id: str) -> Mapping[str, object] | None:
        row = self.connect().execute(
            "SELECT * FROM archive_manifest WHERE archive_id = ? AND status = 'complete'",
            [archive_id],
        ).fetchone()
        if row is None:
            return None
        columns = [item[0] for item in self.connect().description]
        return dict(zip(columns, row, strict=True))

    def ingest_public_archive(
        self,
        *,
        symbol: str,
        data_type: str,
        period: str,
        url: str | None = None,
        expected_bytes: int = 0,
        timeout_seconds: float = 180.0,
        max_download_bytes: int = 8 * 1024**3,
        max_uncompressed_bytes: int = 64 * 1024**3,
        retain_archive: bool = True,
        progress: ProgressCallback | None = None,
        session: requests.Session | None = None,
    ) -> TickArchiveIngestResult:
        symbol, data_type, period = _normalize_tick_request(symbol, data_type, period)
        source_url = url or official_tick_archive_url(
            symbol=symbol,
            data_type=data_type,
            period=period,
        )
        _validate_official_url(source_url, symbol=symbol, data_type=data_type, period=period)
        active_session = session or _resilient_http_session()
        own_session = session is None
        csv_path: Path | None = None
        archive_path = _safe_archive_path(
            self.cache_root,
            symbol=symbol,
            data_type=data_type,
            period=period,
        )
        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        try:
            with _exclusive_operation_lock(lock_path):
                if not self._reconciled:
                    reconciled = self.reconcile_orphan_rows()
                    if progress and any(reconciled.values()):
                        progress("warehouse-reconcile", sum(reconciled.values()), sum(reconciled.values()))
                expected_sha256 = _fetch_checksum(
                    active_session,
                    source_url,
                    timeout_seconds=min(30.0, timeout_seconds),
                )
                archive_id = _archive_id(source_url, expected_sha256)
                completed = self._completed_manifest(archive_id)
                if completed is not None:
                    return self._result_from_manifest(completed, status="skipped")
                compressed_bytes, source_sha256 = _download_verified_archive(
                    active_session,
                    source_url,
                    archive_path,
                    expected_sha256=expected_sha256,
                    timeout_seconds=timeout_seconds,
                    expected_bytes=max(0, int(expected_bytes)),
                    max_download_bytes=max_download_bytes,
                    progress=progress,
                )
                member, has_header = _inspect_zip(
                    archive_path,
                    data_type=data_type,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                )
                csv_path = archive_path.with_suffix(".csv.ingest")
                if csv_path.exists():
                    csv_path.unlink()
                if progress:
                    progress("archive-extract", 0, member.file_size)
                _extract_member(archive_path, member, csv_path)
                if csv_path.stat().st_size != member.file_size:
                    raise ValueError("extracted CSV size does not match ZIP metadata")
                if progress:
                    progress("archive-extract", member.file_size, member.file_size)
                result = self._ingest_csv(
                    archive_id=archive_id,
                    symbol=symbol,
                    data_type=data_type,
                    period=period,
                    url=source_url,
                    archive_path=archive_path,
                    csv_path=csv_path,
                    has_header=has_header,
                    source_sha256=source_sha256,
                    expected_sha256=expected_sha256,
                    expected_bytes=max(0, int(expected_bytes)),
                    compressed_bytes=compressed_bytes,
                    uncompressed_bytes=member.file_size,
                    progress=progress,
                )
                if not retain_archive:
                    archive_path.unlink()
                return result
        finally:
            if csv_path is not None:
                try:
                    csv_path.unlink()
                except OSError:
                    pass
            if own_session:
                active_session.close()

    def _ingest_csv(
        self,
        *,
        archive_id: str,
        symbol: str,
        data_type: str,
        period: str,
        url: str,
        archive_path: Path,
        csv_path: Path,
        has_header: bool,
        source_sha256: str,
        expected_sha256: str,
        expected_bytes: int,
        compressed_bytes: int,
        uncompressed_bytes: int,
        progress: ProgressCallback | None,
    ) -> TickArchiveIngestResult:
        if progress:
            progress("warehouse-import", 0, None)
        now_ms = int(time.time() * 1000)
        conn = self.connect()
        conn.execute("BEGIN TRANSACTION")
        try:
            if data_type == "bookTicker":
                metrics = self._ingest_book_ticker_csv(
                    csv_path,
                    has_header=has_header,
                    symbol=symbol,
                    archive_id=archive_id,
                    period=period,
                )
            elif data_type == "trades":
                metrics = self._ingest_trades_csv(
                    csv_path,
                    has_header=has_header,
                    symbol=symbol,
                    archive_id=archive_id,
                    period=period,
                )
            else:
                metrics = self._ingest_book_depth_csv(
                    csv_path,
                    symbol=symbol,
                    archive_id=archive_id,
                    period=period,
                )
            conn.execute(
                "DELETE FROM archive_manifest WHERE archive_id = ? AND status <> 'complete'",
                [archive_id],
            )
            conn.execute(
                "UPDATE archive_manifest SET is_current = false WHERE url = ? AND is_current",
                [url],
            )
            conn.execute(
                """
                INSERT INTO archive_manifest (
                    archive_id, schema_version, provider, market_type, symbol, data_type,
                    period, url, archive_path, status, is_current, expected_bytes,
                    compressed_bytes, uncompressed_bytes, source_sha256, expected_sha256,
                    checksum_status, rows_read, derived_rows, first_exchange_time_ms,
                    last_exchange_time_ms, invalid_rows, duplicate_ids, update_id_regressions,
                    event_time_regressions, out_of_order_rows, crossed_books, ingested_at_ms, error
                ) VALUES (
                    ?, ?, 'binance', 'futures', ?, ?, ?, ?, ?, 'complete', true,
                    ?, ?, ?, ?, ?, 'verified', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ''
                )
                """,
                [
                    archive_id,
                    TICK_WAREHOUSE_SCHEMA_VERSION,
                    symbol,
                    data_type,
                    period,
                    url,
                    str(archive_path),
                    expected_bytes,
                    compressed_bytes,
                    uncompressed_bytes,
                    source_sha256,
                    expected_sha256,
                    metrics["rows_read"],
                    metrics["derived_rows"],
                    metrics["first_exchange_time_ms"],
                    metrics["last_exchange_time_ms"],
                    metrics["invalid_rows"],
                    metrics["duplicate_ids"],
                    metrics["update_id_regressions"],
                    metrics["event_time_regressions"],
                    metrics["out_of_order_rows"],
                    metrics["crossed_books"],
                    now_ms,
                ],
            )
            if data_type == "bookTicker":
                conn.execute(
                    """
                    UPDATE book_ticker_feature_build_audit
                    SET is_current = false
                    WHERE symbol = ? AND is_current
                    """,
                    [symbol],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            stage = {
                "bookTicker": "stage_book_ticker",
                "trades": "stage_trades",
                "bookDepth": "stage_book_depth",
            }[data_type]
            conn.execute(f"DROP TABLE IF EXISTS {stage}")
            conn.execute("SET preserve_insertion_order=false")
        if progress:
            progress("warehouse-import", int(metrics["rows_read"]), int(metrics["rows_read"]))
        return TickArchiveIngestResult(
            archive_id=archive_id,
            status="complete",
            symbol=symbol,
            data_type=data_type,
            period=period,
            url=url,
            archive_path=str(archive_path),
            source_sha256=source_sha256,
            expected_sha256=expected_sha256,
            compressed_bytes=compressed_bytes,
            uncompressed_bytes=uncompressed_bytes,
            rows_read=int(metrics["rows_read"]),
            derived_rows=int(metrics["derived_rows"]),
            first_exchange_time_ms=int(metrics["first_exchange_time_ms"]),
            last_exchange_time_ms=int(metrics["last_exchange_time_ms"]),
            invalid_rows=int(metrics["invalid_rows"]),
            duplicate_ids=int(metrics["duplicate_ids"]),
            update_id_regressions=int(metrics["update_id_regressions"]),
            event_time_regressions=int(metrics["event_time_regressions"]),
            out_of_order_rows=int(metrics["out_of_order_rows"]),
            crossed_books=int(metrics["crossed_books"]),
        )

    def _ingest_book_ticker_csv(
        self,
        csv_path: Path,
        *,
        has_header: bool,
        symbol: str,
        archive_id: str,
        period: str,
    ) -> dict[str, int]:
        if not has_header:
            raise ValueError("bookTicker archive must contain its documented header")
        start_ms, end_ms = _period_bounds_ms(period)
        conn = self.connect()
        conn.execute("SET preserve_insertion_order=true")
        conn.execute("DROP TABLE IF EXISTS stage_book_ticker")
        conn.execute(
            """
            CREATE TEMP TABLE stage_book_ticker AS
            SELECT
                row_number() OVER ()::UBIGINT AS source_row,
                update_id::UBIGINT AS update_id,
                best_bid_price::DOUBLE AS bid_price,
                best_bid_qty::DOUBLE AS bid_qty,
                best_ask_price::DOUBLE AS ask_price,
                best_ask_qty::DOUBLE AS ask_qty,
                transaction_time::BIGINT AS transaction_time_ms,
                event_time::BIGINT AS event_time_ms
            FROM read_csv(
                ?, header=true, strict_mode=true,
                columns={
                    'update_id':'UBIGINT', 'best_bid_price':'DOUBLE',
                    'best_bid_qty':'DOUBLE', 'best_ask_price':'DOUBLE',
                    'best_ask_qty':'DOUBLE', 'transaction_time':'BIGINT',
                    'event_time':'BIGINT'
                }
            )
            """,
            [str(csv_path)],
        )
        conn.execute("SET preserve_insertion_order=false")
        row = conn.execute(
            """
            WITH sequenced AS (
                SELECT *,
                    lag(update_id) OVER (ORDER BY source_row) AS previous_id,
                    lag(transaction_time_ms) OVER (ORDER BY source_row) AS previous_time,
                    lag(event_time_ms) OVER (ORDER BY source_row) AS previous_event_time
                FROM stage_book_ticker
            )
            SELECT
                count(*)::UBIGINT,
                min(transaction_time_ms), max(transaction_time_ms),
                count(*) FILTER (
                    WHERE NOT isfinite(bid_price) OR NOT isfinite(ask_price)
                       OR NOT isfinite(bid_qty) OR NOT isfinite(ask_qty)
                       OR bid_price <= 0 OR ask_price <= 0 OR bid_qty <= 0 OR ask_qty <= 0
                       OR event_time_ms < transaction_time_ms
                       OR transaction_time_ms NOT BETWEEN ? AND ?
                )::UBIGINT,
                count(*) - count(DISTINCT update_id),
                count(*) FILTER (
                    WHERE previous_time IS NOT NULL
                      AND update_id < previous_id
                )::UBIGINT,
                count(*) FILTER (
                    WHERE previous_time IS NOT NULL
                      AND transaction_time_ms = previous_time
                      AND event_time_ms < previous_event_time
                )::UBIGINT,
                count(*) FILTER (
                    WHERE previous_time IS NOT NULL
                      AND transaction_time_ms < previous_time
                )::UBIGINT,
                count(*) FILTER (WHERE bid_price >= ask_price)::UBIGINT
            FROM sequenced
            """,
            [start_ms, end_ms],
        ).fetchone()
        if row is None:
            raise ValueError("bookTicker validation returned no result")
        rows_read, first_ms, last_ms, invalid, duplicates, id_regressions, event_regressions, out_of_order, crossed = (
            int(value or 0) for value in row
        )
        # Binance may physically interleave independently produced chunks in an
        # otherwise valid daily archive. Preserve source-order regressions as an
        # audit diagnostic, then canonicalize every persisted row by exchange
        # timestamps. Integrity still requires valid, unique, uncrossed quotes.
        if rows_read <= 0 or invalid or duplicates or crossed:
            raise ValueError(
                "bookTicker validation failed: "
                f"rows={rows_read} invalid={invalid} duplicates={duplicates} "
                f"id_regressions={id_regressions} event_regressions={event_regressions} "
                f"out_of_order={out_of_order} crossed={crossed}"
            )
        conn.execute("DELETE FROM book_ticker_raw WHERE archive_id = ?", [archive_id])
        conn.execute("DELETE FROM book_ticker_path_1s WHERE archive_id = ?", [archive_id])
        conn.execute("DELETE FROM book_ticker_100ms WHERE archive_id = ?", [archive_id])
        conn.execute(
                """
                INSERT INTO book_ticker_raw
                SELECT ?, ?, update_id, bid_price, bid_qty, ask_price, ask_qty,
                       transaction_time_ms, event_time_ms
                FROM stage_book_ticker ORDER BY transaction_time_ms, event_time_ms, update_id
                """,
                [archive_id, symbol],
            )
        conn.execute(
            """
            INSERT INTO book_ticker_path_1s
            SELECT
                ?, ?, (transaction_time_ms // 1000) * 1000 AS second_ms,
                min(bid_price), max(bid_price),
                last(bid_price ORDER BY transaction_time_ms, event_time_ms, update_id),
                min(ask_price), max(ask_price),
                last(ask_price ORDER BY transaction_time_ms, event_time_ms, update_id)
            FROM stage_book_ticker GROUP BY second_ms ORDER BY second_ms
            """,
            [archive_id, symbol],
        )
        conn.execute(
            """
            INSERT INTO book_ticker_100ms
            SELECT
                ?, ?, (transaction_time_ms // 100) * 100 AS bucket_ms,
                min(bid_price), max(bid_price),
                last(bid_price ORDER BY transaction_time_ms, event_time_ms, update_id),
                last(bid_qty ORDER BY transaction_time_ms, event_time_ms, update_id),
                min(ask_price), max(ask_price),
                last(ask_price ORDER BY transaction_time_ms, event_time_ms, update_id),
                last(ask_qty ORDER BY transaction_time_ms, event_time_ms, update_id),
                max(transaction_time_ms),
                last(event_time_ms ORDER BY transaction_time_ms, event_time_ms, update_id)
            FROM stage_book_ticker GROUP BY bucket_ms ORDER BY bucket_ms
            """,
            [archive_id, symbol],
        )
        derived_rows = int(
            conn.execute(
                "SELECT count(*) FROM book_ticker_path_1s WHERE archive_id = ?",
                [archive_id],
            ).fetchone()[0]
        )
        return {
            "rows_read": rows_read,
            "derived_rows": derived_rows,
            "first_exchange_time_ms": first_ms,
            "last_exchange_time_ms": last_ms,
            "invalid_rows": invalid,
            "duplicate_ids": duplicates,
            "update_id_regressions": id_regressions,
            "event_time_regressions": event_regressions,
            "out_of_order_rows": out_of_order,
            "crossed_books": crossed,
        }

    def backfill_book_ticker_paths(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, dict[str, int]]:
        """Materialize missing execution states under the warehouse writer lock."""

        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        with _exclusive_operation_lock(lock_path):
            if not self._reconciled:
                reconciled = self.reconcile_orphan_rows()
                if progress and any(reconciled.values()):
                    removed = sum(reconciled.values())
                    progress("warehouse-reconcile", removed, removed)
            return self._backfill_book_ticker_paths_unlocked(progress=progress)

    def _backfill_book_ticker_paths_unlocked(
        self,
        *,
        progress: ProgressCallback | None,
    ) -> dict[str, dict[str, int]]:
        """Materialize missing rows; caller must hold the writer lock."""

        conn = self.connect()
        archive_ids = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT m.archive_id
                FROM archive_manifest m
                LEFT JOIN (
                    SELECT archive_id, count(*) AS path_rows
                    FROM book_ticker_path_1s GROUP BY archive_id
                ) p USING (archive_id)
                WHERE m.data_type = 'bookTicker' AND m.status = 'complete'
                  AND coalesce(p.path_rows, 0) = 0
                ORDER BY m.period, m.archive_id
                """
            ).fetchall()
        ]
        output: dict[str, dict[str, int]] = {}
        if progress:
            progress("path-backfill", 0, len(archive_ids))
        for index, archive_id in enumerate(archive_ids, start=1):
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    """
                    INSERT INTO book_ticker_path_1s
                    SELECT
                        archive_id, symbol, (transaction_time_ms // 1000) * 1000 AS second_ms,
                        min(bid_price), max(bid_price),
                        last(bid_price ORDER BY transaction_time_ms, event_time_ms, update_id),
                        min(ask_price), max(ask_price),
                        last(ask_price ORDER BY transaction_time_ms, event_time_ms, update_id)
                    FROM book_ticker_raw
                    WHERE archive_id = ?
                    GROUP BY archive_id, symbol, second_ms ORDER BY second_ms
                    """,
                    [archive_id],
                )
                count = int(
                    conn.execute(
                        "SELECT count(*) FROM book_ticker_path_1s WHERE archive_id = ?",
                        [archive_id],
                    ).fetchone()[0]
                )
                conn.execute("COMMIT")
                output[archive_id] = {"path_1s_rows": count, "quote_100ms_rows": 0}
                if progress:
                    progress("path-backfill", index, len(archive_ids))
            except Exception:
                conn.execute("ROLLBACK")
                raise
        quote_archive_ids = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT m.archive_id
                FROM archive_manifest m
                LEFT JOIN (
                    SELECT archive_id, count(*) AS quote_rows
                    FROM book_ticker_100ms GROUP BY archive_id
                ) q USING (archive_id)
                WHERE m.data_type = 'bookTicker' AND m.status = 'complete'
                  AND coalesce(q.quote_rows, 0) = 0
                ORDER BY m.period, m.archive_id
                """
            ).fetchall()
        ]
        if progress:
            progress("quote-100ms-backfill", 0, len(quote_archive_ids))
        for index, archive_id in enumerate(quote_archive_ids, start=1):
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    """
                    INSERT INTO book_ticker_100ms
                    SELECT
                        archive_id, symbol, (transaction_time_ms // 100) * 100 AS bucket_ms,
                        min(bid_price), max(bid_price),
                        last(bid_price ORDER BY transaction_time_ms, event_time_ms, update_id),
                        last(bid_qty ORDER BY transaction_time_ms, event_time_ms, update_id),
                        min(ask_price), max(ask_price),
                        last(ask_price ORDER BY transaction_time_ms, event_time_ms, update_id),
                        last(ask_qty ORDER BY transaction_time_ms, event_time_ms, update_id),
                        max(transaction_time_ms),
                        last(event_time_ms ORDER BY transaction_time_ms, event_time_ms, update_id)
                    FROM book_ticker_raw
                    WHERE archive_id = ?
                    GROUP BY archive_id, symbol, bucket_ms ORDER BY bucket_ms
                    """,
                    [archive_id],
                )
                count = int(
                    conn.execute(
                        "SELECT count(*) FROM book_ticker_100ms WHERE archive_id = ?",
                        [archive_id],
                    ).fetchone()[0]
                )
                conn.execute("COMMIT")
                output.setdefault(archive_id, {"path_1s_rows": 0, "quote_100ms_rows": 0})[
                    "quote_100ms_rows"
                ] = count
                if progress:
                    progress("quote-100ms-backfill", index, len(quote_archive_ids))
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return output

    def _book_ticker_source_snapshot(self, symbol: str) -> dict[str, object]:
        symbol = normalize_symbol(symbol)
        rows = self.connect().execute(
            """
            SELECT archive_id, period, source_sha256, expected_sha256,
                   checksum_status, rows_read, first_exchange_time_ms,
                   last_exchange_time_ms, invalid_rows, duplicate_ids,
                   crossed_books
            FROM archive_manifest
            WHERE symbol = ? AND data_type = 'bookTicker'
              AND status = 'complete' AND is_current
            ORDER BY period, archive_id
            """,
            [symbol],
        ).fetchall()
        if not rows:
            raise ValueError(f"no current complete bookTicker archives exist for {symbol}")
        periods = [str(row[1]) for row in rows]
        if len(periods) != len(set(periods)):
            raise ValueError(f"multiple current bookTicker archives exist for a {symbol} UTC day")
        failures = []
        manifest_rows = 0
        fingerprint_rows: list[dict[str, object]] = []
        for row in rows:
            (
                archive_id,
                period,
                source_sha256,
                expected_sha256,
                checksum_status,
                rows_read,
                first_ms,
                last_ms,
                invalid_rows,
                duplicate_ids,
                crossed_books,
            ) = row
            source_hash = str(source_sha256).lower()
            expected_hash = str(expected_sha256).lower()
            row_count = int(rows_read or 0)
            if (
                str(checksum_status) != "verified"
                or len(source_hash) != 64
                or source_hash != expected_hash
                or row_count <= 0
                or int(invalid_rows or 0) != 0
                or int(duplicate_ids or 0) != 0
                or int(crossed_books or 0) != 0
            ):
                failures.append(str(archive_id))
            manifest_rows += row_count
            fingerprint_rows.append(
                {
                    "archive_id": str(archive_id),
                    "period": str(period),
                    "source_sha256": source_hash,
                    "rows_read": row_count,
                    "first_exchange_time_ms": int(first_ms) if first_ms is not None else None,
                    "last_exchange_time_ms": int(last_ms) if last_ms is not None else None,
                }
            )
        if failures:
            raise ValueError(
                "bookTicker source manifests failed checksum or integrity gates: "
                + ", ".join(failures[:5])
            )
        canonical = json.dumps(
            fingerprint_rows,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return {
            "symbol": symbol,
            "archive_count": len(rows),
            "manifest_rows": manifest_rows,
            "manifest_fingerprint": hashlib.sha256(canonical).hexdigest(),
            "first_transaction_time_ms": min(int(row[6]) for row in rows if row[6] is not None),
            "last_transaction_time_ms": max(int(row[7]) for row in rows if row[7] is not None),
        }

    def rebuild_causal_feature_bars(
        self,
        symbol: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Atomically rebuild availability-time feature bars across archive boundaries."""

        normalized = normalize_symbol(symbol)
        if not is_supported_major_symbol(normalized):
            raise ValueError(f"unsupported feature-bar symbol: {normalized}")
        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        with _exclusive_operation_lock(lock_path):
            if not self._reconciled:
                reconciled = self.reconcile_orphan_rows()
                if progress and any(reconciled.values()):
                    removed = sum(reconciled.values())
                    progress("warehouse-reconcile", removed, removed)
            return self._rebuild_causal_feature_bars_unlocked(normalized, progress=progress)

    def _rebuild_causal_feature_bars_unlocked(
        self,
        symbol: str,
        *,
        progress: ProgressCallback | None,
    ) -> dict[str, object]:
        conn = self.connect()
        snapshot = self._book_ticker_source_snapshot(symbol)
        raw = conn.execute(
            """
            SELECT count(*)::UBIGINT, count(DISTINCT archive_id)::UINTEGER,
                   min(transaction_time_ms), max(transaction_time_ms),
                   min(event_time_ms), max(event_time_ms)
            FROM current_book_ticker_raw
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        if raw is None:
            raise ValueError(f"bookTicker raw-row audit returned no result for {symbol}")
        source_rows, source_archives, first_tx, last_tx, first_event, last_event = raw
        source_rows = int(source_rows or 0)
        source_archives = int(source_archives or 0)
        if source_rows != int(snapshot["manifest_rows"]):
            raise ValueError(
                f"bookTicker raw rows do not match manifests for {symbol}: "
                f"raw={source_rows} manifest={snapshot['manifest_rows']}"
            )
        if source_archives != int(snapshot["archive_count"]):
            raise ValueError(
                f"bookTicker raw archive coverage does not match manifests for {symbol}: "
                f"raw={source_archives} manifest={snapshot['archive_count']}"
            )
        built_at_ms = int(time.time() * 1000)
        build_seed = (
            f"{BOOK_TICKER_FEATURE_BUILD_VERSION}\n{symbol}\n"
            f"{snapshot['manifest_fingerprint']}\n{built_at_ms}\n{time.time_ns()}"
        ).encode("ascii")
        build_id = hashlib.sha256(build_seed).hexdigest()
        if progress:
            progress("causal-feature-aggregate", 0, source_rows)
        conn.execute("DROP TABLE IF EXISTS stage_book_ticker_feature_1s")
        try:
            conn.execute(
                """
                CREATE TEMP TABLE stage_book_ticker_feature_1s AS
                WITH enriched AS (
                    SELECT *,
                        (event_time_ms // 1000) * 1000 AS second_ms,
                        (bid_price + ask_price) / 2.0 AS mid,
                        (ask_price - bid_price) * 10000.0
                            / ((ask_price + bid_price) / 2.0) AS spread_bps,
                        (bid_qty - ask_qty) / (bid_qty + ask_qty) AS imbalance,
                        (((ask_price * bid_qty + bid_price * ask_qty) / (bid_qty + ask_qty))
                            / ((ask_price + bid_price) / 2.0) - 1.0)
                            * 10000.0 AS microprice_offset_bps
                    FROM current_book_ticker_raw
                    WHERE symbol = ?
                )
                SELECT
                    ?::VARCHAR AS build_id,
                    symbol,
                    second_ms,
                    first(mid ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS open_mid,
                    max(mid) AS high_mid,
                    min(mid) AS low_mid,
                    last(mid ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS close_mid,
                    last(bid_price ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS close_bid,
                    last(ask_price ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS close_ask,
                    last(bid_qty ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS close_bid_qty,
                    last(ask_qty ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS close_ask_qty,
                    avg(spread_bps) AS event_weighted_spread_bps,
                    max(spread_bps) AS max_spread_bps,
                    avg(imbalance) AS event_weighted_l1_imbalance,
                    last(imbalance ORDER BY event_time_ms, transaction_time_ms, update_id, archive_id)
                        AS close_l1_imbalance,
                    avg(microprice_offset_bps) AS event_weighted_microprice_offset_bps,
                    count(*)::UINTEGER AS quote_updates,
                    quantile_cont((event_time_ms - transaction_time_ms)::DOUBLE, 0.5)
                        AS event_delay_p50_ms,
                    quantile_cont((event_time_ms - transaction_time_ms)::DOUBLE, 0.99)
                        AS event_delay_p99_ms,
                    max(event_time_ms - transaction_time_ms) AS event_delay_max_ms,
                    min(event_time_ms) AS first_event_time_ms,
                    max(event_time_ms) AS last_event_time_ms,
                    last(transaction_time_ms ORDER BY event_time_ms, transaction_time_ms,
                         update_id, archive_id) AS last_transaction_time_ms,
                    count(DISTINCT archive_id)::UINTEGER AS source_archive_count
                FROM enriched
                GROUP BY symbol, second_ms
                ORDER BY symbol, second_ms
                """,
                [symbol, build_id],
            )
            if progress:
                progress("causal-feature-aggregate", source_rows, source_rows)
            audit = conn.execute(
                """
                SELECT count(*)::UBIGINT,
                       sum(quote_updates)::UBIGINT,
                       min(second_ms), max(second_ms),
                       count(*) - count(DISTINCT second_ms) AS duplicate_seconds,
                       count(*) FILTER (
                           WHERE NOT isfinite(open_mid) OR NOT isfinite(high_mid)
                              OR NOT isfinite(low_mid) OR NOT isfinite(close_mid)
                              OR NOT isfinite(close_bid) OR NOT isfinite(close_ask)
                              OR NOT isfinite(close_bid_qty) OR NOT isfinite(close_ask_qty)
                              OR close_bid <= 0 OR close_ask <= 0 OR close_bid >= close_ask
                              OR close_bid_qty <= 0 OR close_ask_qty <= 0
                              OR low_mid > high_mid OR open_mid < low_mid OR open_mid > high_mid
                              OR close_mid < low_mid OR close_mid > high_mid
                              OR quote_updates <= 0 OR source_archive_count <= 0
                              OR first_event_time_ms < second_ms
                              OR last_event_time_ms >= second_ms + 1000
                              OR first_event_time_ms > last_event_time_ms
                              OR last_transaction_time_ms > last_event_time_ms
                              OR event_delay_p50_ms < 0 OR event_delay_p99_ms < 0
                              OR event_delay_max_ms < 0
                       )::UBIGINT
                FROM stage_book_ticker_feature_1s
                """
            ).fetchone()
            if audit is None:
                raise ValueError("causal feature-bar validation returned no result")
            feature_rows, quote_update_sum, first_second, last_second, duplicates, invalid = (
                int(value or 0) for value in audit
            )
            if feature_rows <= 0 or quote_update_sum != source_rows or duplicates or invalid:
                raise ValueError(
                    "causal feature-bar validation failed: "
                    f"rows={feature_rows} quote_updates={quote_update_sum}/{source_rows} "
                    f"duplicates={duplicates} invalid={invalid}"
                )
            if first_second != (int(first_event) // 1000) * 1000:
                raise ValueError("causal feature bars do not start at the first available event second")
            if last_second != (int(last_event) // 1000) * 1000:
                raise ValueError("causal feature bars do not end at the last available event second")
            final_snapshot = self._book_ticker_source_snapshot(symbol)
            if final_snapshot["manifest_fingerprint"] != snapshot["manifest_fingerprint"]:
                raise RuntimeError("bookTicker source manifests changed during feature-bar rebuild")

            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    "UPDATE book_ticker_feature_build_audit SET is_current = false "
                    "WHERE symbol = ? AND is_current",
                    [symbol],
                )
                conn.execute("DROP TABLE IF EXISTS book_ticker_1s")
                conn.execute("DELETE FROM book_ticker_feature_1s WHERE symbol = ?", [symbol])
                conn.execute(
                    "INSERT INTO book_ticker_feature_1s "
                    "SELECT * FROM stage_book_ticker_feature_1s"
                )
                conn.execute(
                    """
                    INSERT INTO book_ticker_feature_build_audit VALUES (
                        ?, ?, ?, 'event_time_ms', ?, 'complete', true, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ''
                    )
                    """,
                    [
                        build_id,
                        TICK_WAREHOUSE_SCHEMA_VERSION,
                        BOOK_TICKER_FEATURE_BUILD_VERSION,
                        symbol,
                        snapshot["manifest_fingerprint"],
                        source_archives,
                        snapshot["manifest_rows"],
                        source_rows,
                        int(first_tx),
                        int(last_tx),
                        int(first_event),
                        int(last_event),
                        feature_rows,
                        first_second,
                        last_second,
                        duplicates,
                        invalid,
                        quote_update_sum,
                        built_at_ms,
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.execute("DROP TABLE IF EXISTS stage_book_ticker_feature_1s")
        if progress:
            progress("causal-feature-finalize", feature_rows, feature_rows)
        return self.require_causal_feature_bars(symbol)

    def require_causal_feature_bars(self, symbol: str) -> dict[str, object]:
        """Return verified provenance or reject absent, stale, or partial feature bars."""

        normalized = normalize_symbol(symbol)
        snapshot = self._book_ticker_source_snapshot(normalized)
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT * FROM book_ticker_feature_build_audit
            WHERE symbol = ? AND status = 'complete' AND is_current
            ORDER BY built_at_ms DESC, build_id DESC
            """,
            [normalized],
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                f"{normalized} requires exactly one current causal feature build; found {len(rows)}"
            )
        columns = [item[0] for item in conn.description]
        evidence = dict(zip(columns, rows[0], strict=True))
        if evidence["schema_version"] != TICK_WAREHOUSE_SCHEMA_VERSION:
            raise ValueError("causal feature build uses an unsupported warehouse schema")
        if evidence["feature_build_version"] != BOOK_TICKER_FEATURE_BUILD_VERSION:
            raise ValueError("causal feature build uses an unsupported aggregation contract")
        if evidence["availability_clock"] != "event_time_ms":
            raise ValueError("causal feature build does not use the availability clock")
        if evidence["manifest_fingerprint"] != snapshot["manifest_fingerprint"]:
            raise ValueError("causal feature build is stale relative to current source manifests")
        if int(evidence["source_manifest_rows"]) != int(snapshot["manifest_rows"]):
            raise ValueError("causal feature build source-row provenance is inconsistent")
        materialized = conn.execute(
            """
            SELECT count(*)::UBIGINT, sum(quote_updates)::UBIGINT,
                   min(second_ms), max(second_ms),
                   count(*) - count(DISTINCT second_ms) AS duplicate_seconds
            FROM book_ticker_feature_1s
            WHERE symbol = ? AND build_id = ?
            """,
            [normalized, evidence["build_id"]],
        ).fetchone()
        if materialized is None:
            raise ValueError("causal feature build materialization audit returned no result")
        feature_rows, quote_updates, first_second, last_second, duplicates = (
            int(value or 0) for value in materialized
        )
        if (
            feature_rows != int(evidence["feature_rows"])
            or quote_updates != int(evidence["source_raw_rows"])
            or first_second != int(evidence["first_feature_second_ms"])
            or last_second != int(evidence["last_feature_second_ms"])
            or duplicates != 0
            or int(evidence["duplicate_seconds"]) != 0
            or int(evidence["invalid_feature_rows"]) != 0
        ):
            raise ValueError("causal feature build is partial or failed its recorded integrity audit")
        output = {key: value for key, value in evidence.items() if key != "error"}
        output["verified"] = True
        output["manifest_current"] = True
        return output

    def causal_feature_evidence(self, symbol: str) -> dict[str, object]:
        normalized = normalize_symbol(symbol)
        try:
            return self.require_causal_feature_bars(normalized)
        except ValueError as exc:
            return {
                "symbol": normalized,
                "status": "missing_or_stale",
                "verified": False,
                "error": str(exc),
            }

    def reserve_terminal_holdout(
        self,
        *,
        symbol: str,
        first_utc_day: int,
        last_utc_day: int,
        candidate_sha256: str,
        source_manifest_fingerprint: str,
        source_feature_build_id: str,
        feature_version: str,
        model_schema_version: str,
        prequential_report_sha256: str,
    ) -> dict[str, object]:
        """Irreversibly reserve a non-overlapping terminal market period once."""

        normalized = normalize_symbol(symbol)
        if not is_supported_major_symbol(normalized):
            raise ValueError(f"unsupported terminal holdout symbol: {normalized}")
        first_day = int(first_utc_day)
        last_day = int(last_utc_day)
        if first_day < 0 or first_day > last_day:
            raise ValueError("terminal holdout UTC-day range is invalid")
        digests = {
            "candidate_sha256": str(candidate_sha256).lower(),
            "source_manifest_fingerprint": str(source_manifest_fingerprint).lower(),
            "source_feature_build_id": str(source_feature_build_id).lower(),
            "prequential_report_sha256": str(prequential_report_sha256).lower(),
        }
        for label, digest in digests.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"terminal holdout {label} is not a SHA-256 digest")
        feature_contract = str(feature_version).strip()
        model_contract = str(model_schema_version).strip()
        if not feature_contract or not model_contract:
            raise ValueError("terminal holdout model contracts cannot be empty")
        now_ms = int(time.time() * 1000)
        reservation_seed = (
            f"{normalized}\n{first_day}\n{last_day}\n{digests['candidate_sha256']}\n"
            f"{digests['source_manifest_fingerprint']}\n{now_ms}\n{time.time_ns()}"
        ).encode("ascii")
        reservation_id = hashlib.sha256(reservation_seed).hexdigest()
        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        with _exclusive_operation_lock(lock_path):
            conn = self.connect()
            conn.execute("BEGIN TRANSACTION")
            try:
                overlaps = conn.execute(
                    """
                    SELECT reservation_id, first_utc_day, last_utc_day, status
                    FROM terminal_holdout_audit
                    WHERE symbol = ?
                      AND first_utc_day <= ? AND last_utc_day >= ?
                    ORDER BY reserved_at_ms, reservation_id
                    """,
                    [normalized, last_day, first_day],
                ).fetchall()
                if overlaps:
                    prior = overlaps[0]
                    raise ValueError(
                        "terminal holdout overlaps a previously consumed or reserved period: "
                        f"reservation={prior[0]} days={prior[1]}..{prior[2]} status={prior[3]}"
                    )
                conn.execute(
                    """
                    INSERT INTO terminal_holdout_audit (
                        reservation_id, symbol, first_utc_day, last_utc_day,
                        candidate_sha256, source_manifest_fingerprint,
                        source_feature_build_id, feature_version,
                        model_schema_version, prequential_report_sha256,
                        status, reserved_at_ms, completed_at_ms,
                        result_status, error
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, NULL, '', ''
                    )
                    """,
                    [
                        reservation_id,
                        normalized,
                        first_day,
                        last_day,
                        digests["candidate_sha256"],
                        digests["source_manifest_fingerprint"],
                        digests["source_feature_build_id"],
                        feature_contract,
                        model_contract,
                        digests["prequential_report_sha256"],
                        now_ms,
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {
            "reservation_id": reservation_id,
            "symbol": normalized,
            "first_utc_day": first_day,
            "last_utc_day": last_day,
            "candidate_sha256": digests["candidate_sha256"],
            "prequential_report_sha256": digests["prequential_report_sha256"],
            "status": "reserved",
            "reserved_at_ms": now_ms,
        }

    def finalize_terminal_holdout(
        self,
        reservation_id: str,
        *,
        result_status: str,
        error: str = "",
    ) -> dict[str, object]:
        reservation = str(reservation_id).lower()
        if len(reservation) != 64 or any(char not in "0123456789abcdef" for char in reservation):
            raise ValueError("terminal holdout reservation_id is invalid")
        result = str(result_status).strip().lower()
        if result not in {"accepted", "validated", "rejected", "evaluation_error"}:
            raise ValueError("terminal holdout result_status is invalid")
        detail = str(error).strip()[:2_000]
        status = "failed" if result == "evaluation_error" else "complete"
        completed_at_ms = int(time.time() * 1000)
        lock_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        with _exclusive_operation_lock(lock_path):
            conn = self.connect()
            current = conn.execute(
                "SELECT status FROM terminal_holdout_audit WHERE reservation_id = ?",
                [reservation],
            ).fetchone()
            if current is None:
                raise ValueError("terminal holdout reservation does not exist")
            if str(current[0]) != "reserved":
                raise ValueError("terminal holdout reservation has already been finalized")
            conn.execute(
                """
                UPDATE terminal_holdout_audit
                SET status = ?, completed_at_ms = ?, result_status = ?, error = ?
                WHERE reservation_id = ? AND status = 'reserved'
                """,
                [status, completed_at_ms, result, detail, reservation],
            )
        return {
            "reservation_id": reservation,
            "status": status,
            "result_status": result,
            "completed_at_ms": completed_at_ms,
            "error": detail,
        }

    def _ingest_trades_csv(
        self,
        csv_path: Path,
        *,
        has_header: bool,
        symbol: str,
        archive_id: str,
        period: str,
    ) -> dict[str, int]:
        start_ms, end_ms = _period_bounds_ms(period)
        conn = self.connect()
        conn.execute("SET preserve_insertion_order=true")
        conn.execute("DROP TABLE IF EXISTS stage_trades")
        conn.execute(
            """
            CREATE TEMP TABLE stage_trades AS
            SELECT
                row_number() OVER ()::UBIGINT AS source_row,
                id::UBIGINT AS trade_id,
                price::DOUBLE AS price,
                qty::DOUBLE AS qty,
                quote_qty::DOUBLE AS quote_qty,
                time::BIGINT AS trade_time_ms,
                CASE lower(trim(is_buyer_maker::VARCHAR))
                    WHEN 'true' THEN true WHEN '1' THEN true
                    WHEN 'false' THEN false WHEN '0' THEN false
                    ELSE NULL
                END AS buyer_is_maker
            FROM read_csv(
                ?, header=?, strict_mode=true,
                columns={
                    'id':'UBIGINT', 'price':'DOUBLE', 'qty':'DOUBLE',
                    'quote_qty':'DOUBLE', 'time':'BIGINT', 'is_buyer_maker':'VARCHAR'
                }
            )
            """,
            [str(csv_path), has_header],
        )
        conn.execute("SET preserve_insertion_order=false")
        row = conn.execute(
            """
            WITH sequenced AS (
                SELECT *, lag(trade_id) OVER (ORDER BY source_row) AS previous_id,
                          lag(trade_time_ms) OVER (ORDER BY source_row) AS previous_time
                FROM stage_trades
            )
            SELECT count(*)::UBIGINT, min(trade_time_ms), max(trade_time_ms),
                count(*) FILTER (
                    WHERE NOT isfinite(price) OR NOT isfinite(qty) OR NOT isfinite(quote_qty)
                       OR price <= 0 OR qty <= 0 OR quote_qty <= 0 OR buyer_is_maker IS NULL
                       OR trade_time_ms NOT BETWEEN ? AND ?
                )::UBIGINT,
                count(*) - count(DISTINCT trade_id),
                count(*) FILTER (
                    WHERE previous_time IS NOT NULL
                      AND (trade_time_ms < previous_time OR trade_id <= previous_id)
                )::UBIGINT
            FROM sequenced
            """,
            [start_ms, end_ms],
        ).fetchone()
        if row is None:
            raise ValueError("trades validation returned no result")
        rows_read, first_ms, last_ms, invalid, duplicates, out_of_order = (int(value or 0) for value in row)
        if rows_read <= 0 or invalid or duplicates or out_of_order:
            raise ValueError(
                "trades validation failed: "
                f"rows={rows_read} invalid={invalid} duplicates={duplicates} out_of_order={out_of_order}"
            )
        conn.execute("DELETE FROM trade_raw WHERE archive_id = ?", [archive_id])
        conn.execute("DELETE FROM trade_1s WHERE archive_id = ?", [archive_id])
        conn.execute(
                """
                INSERT INTO trade_raw
                SELECT ?, ?, trade_id, price, qty, quote_qty, trade_time_ms, buyer_is_maker
                FROM stage_trades ORDER BY trade_time_ms, trade_id
                """,
                [archive_id, symbol],
            )
        conn.execute(
                """
                INSERT INTO trade_1s
                SELECT
                    ?, ?, (trade_time_ms // 1000) * 1000 AS second_ms,
                    first(price ORDER BY trade_time_ms, trade_id), max(price), min(price),
                    last(price ORDER BY trade_time_ms, trade_id),
                    sum(qty), sum(quote_qty),
                    sum(CASE WHEN NOT buyer_is_maker THEN qty ELSE 0 END),
                    sum(CASE WHEN buyer_is_maker THEN qty ELSE 0 END),
                    (sum(CASE WHEN NOT buyer_is_maker THEN qty ELSE 0 END)
                        - sum(CASE WHEN buyer_is_maker THEN qty ELSE 0 END)) / sum(qty),
                    count(*)::UINTEGER
                FROM stage_trades GROUP BY second_ms ORDER BY second_ms
                """,
                [archive_id, symbol],
            )
        derived_rows = int(
            conn.execute(
                "SELECT count(*) FROM trade_1s WHERE archive_id = ?",
                [archive_id],
            ).fetchone()[0]
        )
        return {
            "rows_read": rows_read,
            "derived_rows": derived_rows,
            "first_exchange_time_ms": first_ms,
            "last_exchange_time_ms": last_ms,
            "invalid_rows": invalid,
            "duplicate_ids": duplicates,
            "update_id_regressions": 0,
            "event_time_regressions": 0,
            "out_of_order_rows": out_of_order,
            "crossed_books": 0,
        }

    def _ingest_book_depth_csv(
        self,
        csv_path: Path,
        *,
        symbol: str,
        archive_id: str,
        period: str,
    ) -> dict[str, int]:
        start_ms, end_ms = _period_bounds_ms(period)
        conn = self.connect()
        conn.execute("DROP TABLE IF EXISTS stage_book_depth")
        conn.execute(
            """
            CREATE TEMP TABLE stage_book_depth AS
            SELECT
                epoch_ms(strptime(timestamp, '%Y-%m-%d %H:%M:%S'))::BIGINT AS timestamp_ms,
                percentage::DECIMAL(4,2) AS percentage,
                depth::DOUBLE AS depth,
                notional::DOUBLE AS notional
            FROM read_csv(
                ?, header=true, strict_mode=true,
                columns={
                    'timestamp':'VARCHAR','percentage':'DECIMAL(4,2)',
                    'depth':'DOUBLE','notional':'DOUBLE'
                }
            )
            """,
            [str(csv_path)],
        )
        row = conn.execute(
            """
            SELECT count(*)::UBIGINT, min(timestamp_ms), max(timestamp_ms),
                count(*) FILTER (
                    WHERE percentage NOT IN (-5,-4,-3,-2,-1,-0.20,0.20,1,2,3,4,5)
                       OR NOT isfinite(depth) OR NOT isfinite(notional)
                       OR depth < 0 OR notional < 0 OR timestamp_ms NOT BETWEEN ? AND ?
                )::UBIGINT,
                count(*) - count(DISTINCT (timestamp_ms, percentage))
            FROM stage_book_depth
            """,
            [start_ms, end_ms],
        ).fetchone()
        if row is None:
            raise ValueError("bookDepth validation returned no result")
        rows_read, first_ms, last_ms, invalid, duplicates = (int(value or 0) for value in row)
        expected_groups = int(
            conn.execute(
                "SELECT count(*) FROM ("
                "SELECT timestamp_ms FROM stage_book_depth "
                "GROUP BY timestamp_ms HAVING count(*) <> 12)"
            ).fetchone()[0]
        )
        if rows_read <= 0 or invalid or duplicates or expected_groups:
            raise ValueError(
                "bookDepth validation failed: "
                f"rows={rows_read} invalid={invalid} duplicates={duplicates} incomplete_groups={expected_groups}"
            )
        conn.execute("DELETE FROM book_depth_aggregate_raw WHERE archive_id = ?", [archive_id])
        conn.execute(
                """
                INSERT INTO book_depth_aggregate_raw
                SELECT ?, ?, timestamp_ms, percentage, depth, notional
                FROM stage_book_depth ORDER BY timestamp_ms, percentage
                """,
                [archive_id, symbol],
            )
        return {
            "rows_read": rows_read,
            "derived_rows": rows_read // 12,
            "first_exchange_time_ms": first_ms,
            "last_exchange_time_ms": last_ms,
            "invalid_rows": invalid,
            "duplicate_ids": duplicates,
            "update_id_regressions": 0,
            "event_time_regressions": 0,
            "out_of_order_rows": 0,
            "crossed_books": 0,
        }

    def _result_from_manifest(
        self,
        payload: Mapping[str, object],
        *,
        status: str | None = None,
    ) -> TickArchiveIngestResult:
        def integer(name: str) -> int:
            value = payload.get(name, 0)
            return int(value or 0)

        def optional_integer(name: str) -> int | None:
            value = payload.get(name)
            return None if value is None else int(value)

        return TickArchiveIngestResult(
            archive_id=str(payload.get("archive_id", "")),
            status=status or str(payload.get("status", "")),
            symbol=str(payload.get("symbol", "")),
            data_type=str(payload.get("data_type", "")),
            period=str(payload.get("period", "")),
            url=str(payload.get("url", "")),
            archive_path=str(payload.get("archive_path", "")),
            source_sha256=str(payload.get("source_sha256", "")),
            expected_sha256=str(payload.get("expected_sha256", "")),
            compressed_bytes=integer("compressed_bytes"),
            uncompressed_bytes=integer("uncompressed_bytes"),
            rows_read=integer("rows_read"),
            derived_rows=integer("derived_rows"),
            first_exchange_time_ms=optional_integer("first_exchange_time_ms"),
            last_exchange_time_ms=optional_integer("last_exchange_time_ms"),
            invalid_rows=integer("invalid_rows"),
            duplicate_ids=integer("duplicate_ids"),
            update_id_regressions=integer("update_id_regressions"),
            event_time_regressions=integer("event_time_regressions"),
            out_of_order_rows=integer("out_of_order_rows"),
            crossed_books=integer("crossed_books"),
            error=str(payload.get("error", "")),
        )

    def evidence(self, symbol: str) -> dict[str, object]:
        symbol = normalize_symbol(symbol)
        rows = self.connect().execute(
            """
            SELECT data_type, count(*) AS archive_count, count(DISTINCT period) AS unique_days,
                   sum(rows_read) AS raw_rows, sum(derived_rows) AS derived_rows,
                   min(first_exchange_time_ms) AS first_ms,
                   max(last_exchange_time_ms) AS last_ms,
                   sum(invalid_rows) AS invalid_rows,
                   sum(duplicate_ids) AS duplicate_ids,
                   sum(update_id_regressions) AS update_id_regressions,
                   sum(event_time_regressions) AS event_time_regressions,
                   sum(out_of_order_rows) AS out_of_order_rows,
                   sum(crossed_books) AS crossed_books,
                   bool_and(checksum_status = 'verified' AND source_sha256 = expected_sha256) AS hashes_verified
            FROM archive_manifest
            WHERE symbol = ? AND status = 'complete' AND is_current
            GROUP BY data_type ORDER BY data_type
            """,
            [symbol],
        ).fetchall()
        columns = [item[0] for item in self.connect().description]
        by_type = [dict(zip(columns, row, strict=True)) for row in rows]
        first_values = [int(item["first_ms"]) for item in by_type if item.get("first_ms") is not None]
        last_values = [int(item["last_ms"]) for item in by_type if item.get("last_ms") is not None]
        first_ms = min(first_values) if first_values else None
        last_ms = max(last_values) if last_values else None
        span_days = (
            max(0.0, (last_ms - first_ms) / 86_400_000.0)
            if first_ms is not None and last_ms is not None
            else 0.0
        )
        return {
            "schema_version": TICK_WAREHOUSE_SCHEMA_VERSION,
            "engine": "duckdb",
            "engine_version": duckdb.__version__,
            "database_path": str(self.path),
            "symbol": symbol,
            "first_exchange_time_ms": first_ms,
            "last_exchange_time_ms": last_ms,
            "span_days": span_days,
            "data_types": by_type,
            "hashes_verified": bool(by_type) and all(bool(item["hashes_verified"]) for item in by_type),
            "causal_feature_bars": self.causal_feature_evidence(symbol),
        }


__all__ = [
    "BOOK_TICKER_FEATURE_BUILD_VERSION",
    "MicrostructureWarehouse",
    "SUPPORTED_TICK_ARCHIVES",
    "TICK_WAREHOUSE_SCHEMA_VERSION",
    "TickArchiveIngestResult",
    "official_tick_archive_url",
]
